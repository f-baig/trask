"""Deterministic measurements of a compiled scene.

A probe is a measuring instrument, not an agent. Every rollout here is driven by
fixed code with the scene's own seed, so a probe report is a pure function of the
scene. That is what makes an outcome-level generation metric attributable: if two
generator arms produce different lap times, the difference is caused by the
geometry and parameters they authored, never by a driver that happened to have a
better day. No probe may call a model.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from .models import Action, ActionName, SceneSpec
from .racing import (
    RacingLineController, RacingWorld,
    _angle_delta, _bearing, _nearest_point_index,
)

ORDER_SAMPLE_TICKS = 10
"""Rank the field once per simulated second, so lane jitter is not an overtake."""


class ProbeReport(BaseModel):
    """Everything measurable about a scene without a model in the loop."""

    probe: str = "generation-probe-v1"
    checked_seed: int

    # Reference-driver outcome: the oracle is the same controller that certifies
    # the scene, so these numbers describe the circuit at competent-driver skill.
    oracle_finished: bool = False
    oracle_seconds: float = 0
    """Active race duration at the scene's own control rate, excluding the countdown."""
    oracle_failure: str | None = None
    oracle_mean_speed: float = 0
    oracle_top_speed: float = 0
    off_track_ticks: int = 0
    brake_fraction: float = 0
    """Share of active ticks the reference driver spent braking."""
    steer_fraction: float = 0
    throttle_fraction: float = 0

    # Difficulty floor: an identically-steered driver that never lifts off. A
    # circuit an unbraked driver can also complete is not asking anything of
    # speed control.
    naive_finished: bool = False
    naive_seconds: float = 0
    naive_failure: str | None = None
    naive_off_track_ticks: int = 0
    """Graded difficulty: how far an unbraked driver runs wide, not just whether
    it survived. A competent circuit can be finished flat out on a wide asphalt
    oval, so the binary alone does not separate easy from demanding geometry."""
    naive_top_speed: float = 0

    # Race shape.
    order_changes: int = 0
    """Kendall-tau distance summed over one-second field rankings."""
    player_finish_position: int | None = None
    field_size: int = 1
    opponents_finished: int = 0
    field_spread_seconds: float | None = None
    """Last-to-first finish gap across opponents that completed the race."""

    # Static geometry, lifted from the compiler's own report for convenience.
    corner_count: int = 0
    length_pixels: float = 0
    longest_straight_pixels: float = 0
    minimum_radius_pixels: float = 0
    sector_count: int = 0
    simulated_ticks: int = 0
    """Total ticks stepped across every rollout; the cost of this measurement."""


def measure(scene: SceneSpec) -> ProbeReport:
    """Run every deterministic probe over one compiled scene."""
    oracle = _rollout(scene, driver="oracle")
    naive = _rollout(scene, driver="full-throttle")
    report = scene.track_report
    return ProbeReport(
        checked_seed=scene.seed,
        oracle_finished=oracle["finished"],
        oracle_seconds=oracle["seconds"],
        oracle_failure=oracle["failure"],
        oracle_mean_speed=oracle["mean_speed"],
        oracle_top_speed=oracle["top_speed"],
        off_track_ticks=oracle["off_track_ticks"],
        brake_fraction=oracle["brake_fraction"],
        steer_fraction=oracle["steer_fraction"],
        throttle_fraction=oracle["throttle_fraction"],
        naive_finished=naive["finished"],
        naive_seconds=naive["seconds"],
        naive_failure=naive["failure"],
        naive_off_track_ticks=naive["off_track_ticks"],
        naive_top_speed=naive["top_speed"],
        order_changes=oracle["order_changes"],
        player_finish_position=oracle["player_position"],
        field_size=oracle["field_size"],
        opponents_finished=oracle["opponents_finished"],
        field_spread_seconds=oracle["field_spread_seconds"],
        corner_count=len(report.corners) if report else 0,
        length_pixels=report.length_pixels if report else 0,
        longest_straight_pixels=report.longest_straight_pixels if report else 0,
        minimum_radius_pixels=report.minimum_radius_pixels if report else 0,
        sector_count=scene.sector_count,
        simulated_ticks=oracle["ticks"] + naive["ticks"],
    )


def _rollout(scene: SceneSpec, driver: str) -> dict:
    """Drive one fixed policy through the authoritative runtime."""
    world = RacingWorld.from_scene(scene)
    # A probe measures each car's completion time and the field spread. Ending
    # at the winner would erase every later measurement, so instrumentation runs
    # the full field even though user-facing races stop at the first finisher.
    world.terminate_on_opponent_win = False
    control_hz = max(1, scene.dynamics.control_hz)
    budget = 1_400 * scene.laps
    controller = RacingLineController()
    controller.reset(scene, scene.seed)

    active_ticks = brake_ticks = steer_ticks = throttle_ticks = off_track_ticks = 0
    speeds: list[float] = []
    order_changes = 0
    previous_order: tuple[str, ...] | None = None

    for _ in range(budget):
        if world.terminated:
            break
        if world.countdown_ticks_remaining > 0:
            world.step(Action())
            continue
        observation = world.observe()
        if driver == "oracle":
            action, decision = controller.act(observation)
        else:
            action, decision = _full_throttle_action(scene, observation), None
        world.step(action, decision)
        active_ticks += 1
        keys = set(action.keys) or {_key_for(action.name)}
        brake_ticks += "s" in keys
        throttle_ticks += "w" in keys
        steer_ticks += bool(keys & {"a", "d"})
        off_track_ticks += world.off_track
        speeds.append(world.speed)
        if active_ticks % ORDER_SAMPLE_TICKS == 0:
            order = _field_order(world)
            if previous_order is not None:
                order_changes += _kendall_tau_distance(previous_order, order)
            previous_order = order

    finished_steps = [
        opponent.finished_step for opponent in world.opponents
        if opponent.finished_step is not None
    ]
    return {
        "finished": world.succeeded,
        "seconds": round(active_ticks / control_hz, 3),
        "failure": None if world.succeeded else (world.reason or "step budget exhausted"),
        "mean_speed": round(sum(speeds) / len(speeds), 3) if speeds else 0,
        "top_speed": round(max(speeds), 3) if speeds else 0,
        "off_track_ticks": off_track_ticks,
        "brake_fraction": round(brake_ticks / active_ticks, 4) if active_ticks else 0,
        "steer_fraction": round(steer_ticks / active_ticks, 4) if active_ticks else 0,
        "throttle_fraction": round(throttle_ticks / active_ticks, 4) if active_ticks else 0,
        "order_changes": order_changes,
        "player_position": world.player_position,
        "field_size": world.field_size,
        "opponents_finished": len(finished_steps),
        "field_spread_seconds": (
            round((max(finished_steps) - min(finished_steps)) / control_hz, 3)
            if len(finished_steps) >= 2 else None
        ),
        "ticks": active_ticks,
    }


def _full_throttle_action(scene: SceneSpec, observation) -> Action:
    """Steer like the oracle, but never lift off.

    Sharing the oracle's aiming logic is deliberate: the only difference between
    the two drivers is speed control, so `naive_finished` isolates whether the
    circuit actually demands braking rather than whether it demands navigation.
    """
    points = scene.track_centerline
    nearest = _nearest_point_index(points, observation.proprioception)
    lookahead = max(4, min(8, 4 + round(observation.speed / 3)))
    target = points[(nearest + lookahead) % len(points)]
    delta = _angle_delta(observation.heading, _bearing(observation.proprioception, target))
    if delta > 6:
        return Action(name=ActionName.RIGHT, keys=["w", "d"])
    if delta < -6:
        return Action(name=ActionName.LEFT, keys=["w", "a"])
    return Action(name=ActionName.FORWARD, keys=["w"])


def _key_for(name: ActionName) -> str:
    return {
        ActionName.FORWARD: "w", ActionName.BACKWARD: "s",
        ActionName.LEFT: "a", ActionName.RIGHT: "d",
    }.get(name, "")


def _field_order(world: RacingWorld) -> tuple[str, ...]:
    """Rank every car by race distance on one common scale.

    The player's progress is authoritative in ordered gate crossings while an
    opponent's is centerline samples, so opponents are converted into the gate
    scale exactly as `live_position` does rather than comparing two units.
    """
    samples = max(1, len(world.scene.track_centerline))
    sectors = max(1, world.scene.sector_count)
    ranked: list[tuple[float, str]] = [(float(world.objective_index), "player")]
    for opponent in world.opponents:
        ranked.append((opponent.progress_samples / samples * sectors, opponent.entity_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(identifier for _, identifier in ranked)


def _kendall_tau_distance(before: tuple[str, ...], after: tuple[str, ...]) -> int:
    """Count pairs of cars whose relative order changed between two samples."""
    rank = {identifier: index for index, identifier in enumerate(after)}
    changed = 0
    for left in range(len(before)):
        for right in range(left + 1, len(before)):
            if rank.get(before[right], right) < rank.get(before[left], left):
                changed += 1
    return changed

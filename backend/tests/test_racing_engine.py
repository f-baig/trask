import math

import pytest

from harness.models import Action, ActionName, DynamicsSpec, EntityKind, Vec2, VehicleDynamicsSpec
from harness.play import _interpolated_world, _terminal_message
from harness.racing import (
    COUNTDOWN_TICKS,
    NITRO_CAPACITY,
    NITRO_DRAIN_PER_TICK,
    NITRO_MAX_SPEED_MULTIPLIER,
    NITRO_RECHARGE_PER_TICK,
    RacingDesignDraft,
    RacingWorld,
    _nearest_point_index,
    compile_racing_scene,
    validate_racing_scene,
    verify_racing_playability,
)
from harness.vehicle_physics import apply_dynamics_preset


def green_flag(world: RacingWorld) -> RacingWorld:
    world.countdown_ticks_remaining = 0
    return world


def design(
    circuit: str = "technical", surface: str = "asphalt", obstacles: int = 2, npcs: int = 1,
) -> RacingDesignDraft:
    return RacingDesignDraft(
        title=f"{circuit.title()} engine test",
        rationale="A deterministic scene for adversarial engine contract tests.",
        circuit=circuit,
        surface=surface,
        obstacle_count=obstacles,
        npc_count=npcs,
    )


@pytest.mark.parametrize("surface", ["asphalt", "clay", "ice"])
def test_speed_is_bounded_for_every_keyboard_combination(surface: str) -> None:
    scene = compile_racing_scene("speed invariant", design(surface=surface, obstacles=0, npcs=0), seed=5)
    for keys in ([], ["w"], ["s"], ["a"], ["d"], ["w", "a"], ["w", "d"], ["s", "a"], ["s", "d"]):
        world = green_flag(RacingWorld.from_scene(scene))
        world.speed = 1_000
        world.step(Action(keys=keys))
        max_display_speed = (
            scene.dynamics.vehicle.max_speed_mps * NITRO_MAX_SPEED_MULTIPLIER
            * scene.dynamics.pixels_per_meter / scene.dynamics.control_hz
        )
        assert 0 <= world.speed <= max_display_speed
        assert 0 <= world.heading < 360
        assert math.isfinite(world.player.x) and math.isfinite(world.player.y)


def test_same_scene_snapshot_and_actions_produce_identical_trajectory() -> None:
    scene = compile_racing_scene("snapshot determinism", design(obstacles=0, npcs=2), seed=41)
    original = green_flag(RacingWorld.from_scene(scene))
    warmup = (["w"], ["w", "a"], ["w"], ["d"], [], ["w"])
    for keys in warmup:
        original.step(Action(keys=keys))

    restored = RacingWorld.from_scene(scene)
    restored.restore(original.snapshot())
    assert restored.snapshot() == original.snapshot()

    continuation = (["w"], ["a"], ["w", "d"], [], ["s"], ["w"]) * 3
    for keys in continuation:
        original_frame = original.step(Action(keys=keys))
        restored_frame = restored.step(Action(keys=keys))
        assert restored.snapshot() == original.snapshot()
        assert restored_frame.model_dump() == original_frame.model_dump()


def test_straights_are_faster_than_turns_and_nitro_only_boosts_straight() -> None:
    scene = compile_racing_scene("speed modes", design(obstacles=0, npcs=0), seed=5)
    straight = green_flag(RacingWorld.from_scene(scene))
    for _ in range(20):
        straight.step(Action(keys=["w"]))
    initial_straight_speed = straight.speed
    assert 0 < initial_straight_speed

    turning = green_flag(RacingWorld.from_scene(scene))
    turning.speed = initial_straight_speed
    turning.longitudinal_velocity_mps = straight.longitudinal_velocity_mps
    for _ in range(8):
        turning.step(Action(keys=["w", "a"]))
        straight.step(Action(keys=["w"]))
    assert turning.turning
    assert turning.speed < straight.speed
    assert abs(turning.lateral_acceleration_mps2) > 0
    assert abs(turning.slip_angle_radians) > 0

    boosted = green_flag(RacingWorld.from_scene(scene))
    boosted.speed = initial_straight_speed
    boosted.longitudinal_velocity_mps = straight.longitudinal_velocity_mps
    boosted.nitro = NITRO_CAPACITY
    comparison = green_flag(RacingWorld.from_scene(scene))
    comparison.speed = boosted.speed
    comparison.longitudinal_velocity_mps = boosted.longitudinal_velocity_mps
    comparison.step(Action(keys=["w"]))
    boosted.step(Action(keys=["w", "space"]))
    assert boosted.nitro_active
    assert boosted.speed > comparison.speed
    assert boosted.nitro == NITRO_CAPACITY - NITRO_DRAIN_PER_TICK

    boosted.step(Action(keys=["w", "a", "space"]))
    assert not boosted.nitro_active
    assert boosted.turning


def test_nitro_recharges_deterministically_while_inactive() -> None:
    scene = compile_racing_scene("nitro recharge", design(obstacles=0, npcs=0), seed=5)
    world = green_flag(RacingWorld.from_scene(scene))
    world.nitro = 0
    for _ in range(12):
        world.step(Action())
    assert world.nitro == 12 * NITRO_RECHARGE_PER_TICK
    assert not world.nitro_active


def test_public_nitro_prediction_matches_runtime_tick() -> None:
    from harness.racing import racing_physics_context

    scene = compile_racing_scene("nitro prediction", design(obstacles=0, npcs=0), seed=5)
    world = green_flag(RacingWorld.from_scene(scene))
    world.speed = (
        scene.dynamics.vehicle.max_speed_mps * .75
        * scene.dynamics.pixels_per_meter / scene.dynamics.control_hz
    )
    world.longitudinal_velocity_mps = world.speed * world.dynamics.control_hz / world.dynamics.pixels_per_meter
    world.nitro = NITRO_CAPACITY
    observation = world.observe()
    predicted = racing_physics_context(scene, observation)["next_tick_outcomes"]["nitro"]
    start = world.player.model_copy()
    world.step(Action(keys=["w", "space"]))
    assert world.speed == pytest.approx(predicted["next_speed"], abs=.01)
    assert math.hypot(world.player.x - start.x, world.player.y - start.y) == pytest.approx(
        predicted["travel_distance_this_tick"], abs=.01,
    )


@pytest.mark.parametrize("laps", [1, 2, 4])
def test_configurable_laps_compile_and_complete(laps: int) -> None:
    scene = compile_racing_scene(
        "multi lap", design(obstacles=0, npcs=0).model_copy(update={"laps": laps}), seed=7,
    )
    assert scene.laps == laps
    assert len(scene.objectives) == laps * scene.sector_count
    certificate = verify_racing_playability(scene)
    assert certificate.playable
    assert certificate.route_steps <= 1_400 * laps
    world = green_flag(RacingWorld.from_scene(scene))
    from harness.racing import RacingLineController

    controller = RacingLineController()
    controller.reset(scene, scene.seed)
    frames = []
    for _ in range(1_400 * laps):
        if world.terminated:
            break
        action, decision = controller.act(world.observe())
        frames.append(world.step(action, decision))
    assert world.succeeded
    assert world.objective_index == laps * scene.sector_count
    assert world.privileged_state().lap == laps
    assert frames[-1].privileged_state.objective_index == len(scene.objectives)
    assert frames[-1].privileged_state.lap == laps
    assert frames[-1].events[-1].startswith("race completed P")
    terminal_step = world.step_number
    with pytest.raises(RuntimeError, match="Cannot step a terminated race"):
        world.step(Action(name=ActionName.IDLE))
    assert world.step_number == terminal_step


def test_npc_start_mode_defaults_to_starting_grid_and_can_be_distributed() -> None:
    grid_scene = compile_racing_scene("grid", design(obstacles=0, npcs=3), seed=3)
    grid_world = RacingWorld.from_scene(grid_scene)
    assert grid_scene.npc_start_mode == "grid"
    # The staggered grid gives every car its own longitudinal slot so launch
    # paths never cross before the field has opened up.
    assert all(
        1 <= (grid_scene.start_line_index - opponent.target_index) % len(grid_scene.track_centerline) <= 9
        for opponent in grid_world.opponents
    )
    # The player holds pole, so every opponent lines up behind it in staggered
    # rows: compact, ordered, and never sharing the player's grid slot.
    gaps = [
        math.hypot(
            opponent.position.x - grid_scene.player_spawn.x,
            opponent.position.y - grid_scene.player_spawn.y,
        )
        for opponent in grid_world.opponents
    ]
    assert all(2 * 11.0 < gap < 130 for gap in gaps), gaps
    assert gaps == pytest.approx(sorted(gaps)), gaps

    distributed_scene = compile_racing_scene(
        "distributed",
        design(obstacles=0, npcs=3).model_copy(update={"npc_start_mode": "distributed"}),
        seed=3,
    )
    distributed_world = RacingWorld.from_scene(distributed_scene)
    assert distributed_scene.npc_start_mode == "distributed"
    assert any(
        math.hypot(opponent.position.x - distributed_scene.player_spawn.x, opponent.position.y - distributed_scene.player_spawn.y) > 150
        for opponent in distributed_world.opponents
    )


def test_start_line_region_and_player_grid_slot_drive_one_shared_race_start() -> None:
    scene = compile_racing_scene(
        "start finish in the top right; player starts P3",
        design(obstacles=0, npcs=3).model_copy(update={
            "start_region": "top-right", "player_grid_position": 3,
        }),
        seed=31,
    )
    world = RacingWorld.from_scene(scene)
    finish = next(entity for entity in scene.entities if entity.id == "finish-line")
    finish_center = Vec2(
        x=finish.rect.x + finish.rect.width / 2,
        y=finish.rect.y + finish.rect.height / 2,
    )
    assert scene.player_grid_position == 3
    assert _nearest_point_index(scene.track_centerline, finish_center) == scene.start_line_index
    assert validate_racing_scene(scene) == ["Racing domain contract passed."]

    start = scene.track_centerline[scene.start_line_index]
    next_point = scene.track_centerline[(scene.start_line_index + 1) % len(scene.track_centerline)]
    forward = (next_point.x - start.x, next_point.y - start.y)
    competitors = [world.player, *(opponent.position for opponent in world.opponents)]
    # No grid car is placed on or beyond the finish gate before the race begins.
    assert all(
        (position.x - start.x) * forward[0] + (position.y - start.y) * forward[1] < 0
        for position in competitors
    )
    for left, first in enumerate(competitors):
        for second in competitors[left + 1:]:
            assert math.hypot(first.x - second.x, first.y - second.y) >= 2 * 11.0

    certificate = verify_racing_playability(scene)
    assert certificate.playable, certificate.failure


def test_countdown_freezes_every_car_and_keeps_nitro_empty() -> None:
    scene = compile_racing_scene("countdown", design(obstacles=0, npcs=2), seed=3)
    world = RacingWorld.from_scene(scene)
    player_start = world.player.model_copy()
    opponent_starts = [opponent.position.model_copy() for opponent in world.opponents]
    assert world.nitro == 0
    assert all(opponent.nitro == 0 for opponent in world.opponents)

    frames = [world.step(Action(keys=["w", "space"])) for _ in range(COUNTDOWN_TICKS)]
    assert world.player == player_start
    assert [opponent.position for opponent in world.opponents] == opponent_starts
    assert world.nitro == 0 and all(opponent.nitro == 0 for opponent in world.opponents)
    assert all(frame.action.value == "idle" and frame.keys == [] for frame in frames)
    assert frames[-1].events == ["go"]

    world.step(Action(keys=["w"]))
    assert world.speed > 0 and world.player != player_start
    assert world.nitro == NITRO_RECHARGE_PER_TICK


def test_partial_nitro_cannot_activate_and_interruption_requires_full_recharge() -> None:
    scene = compile_racing_scene("nitro lockout", design(obstacles=0, npcs=0), seed=5)
    world = green_flag(RacingWorld.from_scene(scene))
    world.nitro = NITRO_CAPACITY - NITRO_RECHARGE_PER_TICK
    world.step(Action(keys=["w", "space"]))
    assert not world.nitro_active and world.nitro == NITRO_CAPACITY

    world.step(Action(keys=["w", "space"]))
    assert world.nitro_active and world.nitro == NITRO_CAPACITY - NITRO_DRAIN_PER_TICK
    world.step(Action(keys=["w"]))
    partial_charge = world.nitro
    assert not world.nitro_active and partial_charge < NITRO_CAPACITY
    world.step(Action(keys=["w", "space"]))
    assert not world.nitro_active and world.nitro > partial_charge


def test_npc_recharges_and_uses_nitro_on_a_clear_straight() -> None:
    scene = compile_racing_scene("npc nitro", design(circuit="oval", obstacles=0, npcs=1), seed=3)
    world = green_flag(RacingWorld.from_scene(scene))
    # This is an NPC capability probe, not a competitive race outcome.
    world.terminate_on_opponent_win = False
    saw_boost = False
    peak_speed = 0.0
    for _ in range(240):
        if world.terminated:
            break
        world.step(Action())
        saw_boost = saw_boost or any(opponent.nitro_active for opponent in world.opponents)
        peak_speed = max(peak_speed, *(opponent.speed for opponent in world.opponents))
    assert saw_boost
    npc_cruise_speed = (
        scene.dynamics.vehicle.max_speed_mps * .9
        * scene.dynamics.pixels_per_meter / scene.dynamics.control_hz
    )
    assert peak_speed > npc_cruise_speed


def test_mass_tire_grip_and_aero_can_be_isolated_as_conditions() -> None:
    scene = compile_racing_scene("factor isolation", design(circuit="oval", obstacles=0, npcs=0), seed=3)

    def conditioned(preset: str) -> RacingWorld:
        variant = scene.model_copy(update={
            "dynamics": apply_dynamics_preset(scene.dynamics, preset),
        })
        return green_flag(RacingWorld.from_scene(variant))

    baseline = conditioned("balanced")
    heavy = conditioned("heavy_car")
    for _ in range(10):
        baseline.step(Action(keys=["w"]))
        heavy.step(Action(keys=["w"]))
    assert heavy.speed < baseline.speed

    fresh = conditioned("balanced")
    worn = conditioned("worn_tires")
    for world in (fresh, worn):
        world.speed = 7
        world.longitudinal_velocity_mps = 7 * world.dynamics.control_hz / world.dynamics.pixels_per_meter
        world.step(Action(keys=["s"]))
    assert worn.speed > fresh.speed

    normal_drag = conditioned("balanced")
    high_drag = conditioned("high_drag")
    for world in (normal_drag, high_drag):
        world.speed = 9
        world.longitudinal_velocity_mps = 9 * world.dynamics.control_hz / world.dynamics.pixels_per_meter
        world.step(Action())
    assert high_drag.aerodynamic_drag_n > normal_drag.aerodynamic_drag_n
    assert high_drag.speed < normal_drag.speed


def test_geometry_weight_distribution_and_cg_height_change_transient_turning() -> None:
    scene = compile_racing_scene("weight transfer", design(circuit="oval", obstacles=0, npcs=0), seed=3)
    tall_vehicle = scene.dynamics.vehicle.model_copy(update={"center_of_mass_height_m": .9})
    tall_scene = scene.model_copy(update={
        "dynamics": scene.dynamics.model_copy(update={"vehicle": tall_vehicle}),
    })
    rear_scene = scene.model_copy(update={
        "dynamics": apply_dynamics_preset(scene.dynamics, "rear_bias"),
    })
    worlds = [
        green_flag(RacingWorld.from_scene(candidate))
        for candidate in (scene, tall_scene, rear_scene)
    ]
    for world in worlds:
        world.speed = 6
        world.longitudinal_velocity_mps = 6 * world.dynamics.control_hz / world.dynamics.pixels_per_meter
        for _ in range(4):
            world.step(Action(keys=["w", "a"]))
    baseline, tall, rear = worlds
    assert tall.lateral_load_transfer_n > baseline.lateral_load_transfer_n
    assert rear.slip_angle_radians != pytest.approx(baseline.slip_angle_radians)
    assert rear.yaw_rate_radians_per_second != pytest.approx(baseline.yaw_rate_radians_per_second)


def test_steering_slews_across_six_fixed_physics_substeps() -> None:
    scene = compile_racing_scene("steering slew", design(obstacles=0, npcs=0), seed=3)
    world = green_flag(RacingWorld.from_scene(scene))
    world.step(Action(keys=["d"]))
    assert scene.dynamics.physics_hz // scene.dynamics.control_hz == 6
    assert math.degrees(world.steering_angle_radians) == pytest.approx(10.0)
    world.step(Action(keys=["d"]))
    assert math.degrees(world.steering_angle_radians) == pytest.approx(20.0)


def test_invalid_vehicle_geometry_and_step_ratio_are_rejected() -> None:
    with pytest.raises(ValueError, match="wheelbase"):
        VehicleDynamicsSpec(length_m=4, wheelbase_m=4.1)
    with pytest.raises(ValueError, match="integer multiple"):
        DynamicsSpec(physics_hz=60, control_hz=16)


def test_restore_exactly_applies_empty_opponent_state() -> None:
    scene = compile_racing_scene("empty traffic restore", design(obstacles=0, npcs=2), seed=4)
    world = green_flag(RacingWorld.from_scene(scene))
    snapshot = world.snapshot()
    snapshot["opponents"] = []
    world.restore(snapshot)
    assert world.opponents == []


def test_restore_rejects_non_finite_or_out_of_range_state() -> None:
    scene = compile_racing_scene("invalid restore", design(obstacles=0, npcs=0), seed=4)
    world = RacingWorld.from_scene(scene)
    invalid_speed = world.snapshot() | {"speed": float("inf")}
    with pytest.raises(ValueError, match="non-finite"):
        world.restore(invalid_speed)
    invalid_objective = world.snapshot() | {"objective_index": 99}
    with pytest.raises(ValueError, match="objective index"):
        world.restore(invalid_objective)


def test_terminated_world_rejects_additional_steps() -> None:
    scene = compile_racing_scene("terminal contract", design(obstacles=0, npcs=0), seed=4)
    world = RacingWorld.from_scene(scene)
    world.terminated = True
    world.reason = "test terminal"
    with pytest.raises(RuntimeError, match="Cannot step a terminated race"):
        world.step(Action(keys=["w"]))
    assert world.step_number == 0


def test_terminal_render_uses_exact_frozen_pose_and_result_copy() -> None:
    scene = compile_racing_scene("terminal render", design(obstacles=0, npcs=1), seed=4)
    world = RacingWorld.from_scene(scene)
    previous_player = world.player.model_copy(update={"x": world.player.x - 30})
    previous_heading = (world.heading - 25) % 360
    previous_opponents = {
        opponent.entity_id: (
            opponent.position.model_copy(update={"x": opponent.position.x - 30}),
            (opponent.heading - 25) % 360,
        )
        for opponent in world.opponents
    }
    world.player.x += 12
    world.heading = (world.heading + 15) % 360
    world.terminated = True
    world.reason = "collision with opponent-1"

    rendered = _interpolated_world(
        world, previous_player, previous_heading, previous_opponents, alpha=.15,
    )
    assert rendered.player == world.player
    assert rendered.heading == world.heading
    assert rendered.opponents[0].position == world.opponents[0].position
    assert rendered.opponents[0].heading == world.opponents[0].heading
    assert _terminal_message(world) == (
        "CRASH", "CAR DISABLED", "collision with opponent-1",
    )

    # Reaching the flag is not winning: the outcome is the finishing position.
    field = world.field_size
    assert field > 1, "this scene must have opponents for position to mean anything"
    world.terminated = world.succeeded = True
    world.reason = f"1-lap race completed in P1 of {field}"
    world.finish_order = ["player"]
    assert _terminal_message(world) == (
        "RACE ENDED", "YOU WON", f"Finished P1 of {field}",
    )
    world.finish_order = [*(f"opponent-{n}" for n in range(1, field)), "player"]
    assert _terminal_message(world) == (
        "RACE ENDED", f"P{field} OF {field}",
        f"Finished P{field} of {field} — {field - 1} car"
        + ("s" if field - 1 > 1 else "") + " ahead",
    )
    world.succeeded = False
    world.reason = "opponent finished first"
    assert _terminal_message(world) == (
        "RACE ENDED", "YOU LOST", "opponent finished first",
    )


def test_first_npc_finisher_ends_the_race_as_a_loss() -> None:
    scene = compile_racing_scene("npc winner", design(obstacles=0, npcs=1), seed=4)
    world = RacingWorld.from_scene(scene)
    world.countdown_ticks_remaining = 0
    opponent = world.opponents[0]
    # Complete the sector sequence first, then put the NPC one sample before the
    # finish gate so the next engine tick is a genuine winning crossing.
    before_finish = (scene.start_line_index - 1) % len(scene.track_centerline)
    opponent.position = scene.track_centerline[before_finish].model_copy()
    opponent.track_index = before_finish
    opponent.target_index = scene.start_line_index
    opponent.speed = 30.0
    opponent.checkpoint_index = scene.sector_count - 1

    frame = world.step(Action())

    assert world.terminated and not world.succeeded
    assert world.finish_order == [opponent.entity_id]
    assert opponent.finished_step == 0
    assert world.reason == f"{opponent.entity_id} finished first"
    assert frame.events == [world.reason]
    assert frame.reward < 0


def test_npc_cannot_finish_from_an_initial_start_line_crossing() -> None:
    scene = compile_racing_scene("npc start crossing", design(obstacles=0, npcs=1), seed=11)
    world = green_flag(RacingWorld.from_scene(scene))
    opponent = world.opponents[0]
    before_finish = (scene.start_line_index - 1) % len(scene.track_centerline)
    opponent.position = scene.track_centerline[before_finish].model_copy()
    opponent.track_index = before_finish
    opponent.target_index = scene.start_line_index
    opponent.speed = 30.0

    world.step(Action())

    assert opponent.checkpoint_index == 0
    assert opponent.completed_laps == 0
    assert opponent.finished_step is None
    assert not world.terminated


def test_npc_requires_each_physical_finish_gate_crossing_in_a_multilap_race() -> None:
    scene = compile_racing_scene(
        "two lap npc winner", design(obstacles=0, npcs=1).model_copy(update={"laps": 2}), seed=14,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    opponent = world.opponents[0]
    before_finish = (scene.start_line_index - 1) % len(scene.track_centerline)

    def cross_finish() -> None:
        opponent.position = scene.track_centerline[before_finish].model_copy()
        opponent.track_index = before_finish
        opponent.target_index = scene.start_line_index
        opponent.speed = 30.0
        opponent.checkpoint_index = scene.sector_count - 1
        world.step(Action())

    cross_finish()
    assert opponent.completed_laps == 1
    assert not world.terminated
    cross_finish()
    assert opponent.completed_laps == 2
    assert world.terminated and not world.succeeded
    assert world.finish_order == [opponent.entity_id]


def test_recorded_npc_winner_cannot_leave_a_restored_race_live() -> None:
    scene = compile_racing_scene("restored npc winner", design(obstacles=0, npcs=1), seed=18)
    world = green_flag(RacingWorld.from_scene(scene))
    opponent = world.opponents[0]
    snapshot = world.snapshot()
    snapshot["finish_order"] = [opponent.entity_id]
    snapshot["opponents"][0]["completed_laps"] = scene.laps
    snapshot["opponents"][0]["finished_step"] = 4
    world.restore(snapshot)

    frame = world.step(Action())

    assert world.terminated and not world.succeeded
    assert world.reason == f"{opponent.entity_id} finished first"
    assert frame.events == [world.reason]


def test_legacy_progress_counter_cannot_mark_an_npc_finished_away_from_the_gate() -> None:
    scene = compile_racing_scene("legacy npc progress", design(obstacles=0, npcs=1), seed=23)
    world = green_flag(RacingWorld.from_scene(scene))
    opponent = world.opponents[0]
    middle = (scene.start_line_index + len(scene.track_centerline) // 2) % len(scene.track_centerline)
    snapshot = world.snapshot()
    snapshot_opponent = snapshot["opponents"][0]
    snapshot_opponent.update({
        "position": scene.track_centerline[middle].model_dump(),
        "track_index": middle,
        "target_index": (middle + 1) % len(scene.track_centerline),
        "progress_samples": len(scene.track_centerline) * scene.laps * 4,
    })
    snapshot_opponent.pop("completed_laps")
    world.restore(snapshot)

    world.step(Action())

    assert world.opponents[0].completed_laps == 0
    assert not world.terminated


def test_checkpoint_requires_forward_entry_through_visible_gate() -> None:
    scene = compile_racing_scene("checkpoint gate", design(obstacles=0, npcs=0), seed=8)
    target = next(entity for entity in scene.entities if entity.kind == EntityKind.CHECKPOINT)
    center = Vec2(x=target.rect.x + target.rect.width / 2, y=target.rect.y + target.rect.height / 2)
    index = min(
        range(len(scene.track_centerline)),
        key=lambda item: math.hypot(scene.track_centerline[item].x - center.x, scene.track_centerline[item].y - center.y),
    )
    before = scene.track_centerline[(index - 1) % len(scene.track_centerline)]
    after = scene.track_centerline[(index + 1) % len(scene.track_centerline)]
    heading = math.degrees(math.atan2(after.y - before.y, after.x - before.x)) % 360
    radians = math.radians(heading)

    # This point was inside the old 62x62 trigger square but is beyond the
    # rendered gate's longitudinal collision depth.
    world = green_flag(RacingWorld.from_scene(scene))
    world.player = Vec2(x=center.x + math.cos(radians) * 30, y=center.y + math.sin(radians) * 30)
    world.heading, world.speed = heading, 1
    world.step(Action())
    assert world.objective_index == 0

    world.player = Vec2(x=center.x - math.cos(radians) * 22, y=center.y - math.sin(radians) * 22)
    world.heading, world.speed = (heading + 180) % 360, 5
    world.step(Action())
    assert world.objective_index == 0

    world.player = Vec2(x=center.x - math.cos(radians) * 22, y=center.y - math.sin(radians) * 22)
    world.heading, world.speed = heading, 5
    frame = world.step(Action())
    assert world.objective_index == 1
    assert frame.events == [f"crossed {target.id}"]


def test_runtime_rejects_duplicate_entities_before_simulation() -> None:
    scene = compile_racing_scene("invalid scene", design(obstacles=1, npcs=0), seed=2)
    duplicate = scene.entities[0].model_copy()
    invalid = scene.model_copy(update={"entities": [*scene.entities, duplicate]})
    findings = validate_racing_scene(invalid)
    assert "Entity ids must be unique" in findings
    with pytest.raises(ValueError, match="invalid scene"):
        RacingWorld.from_scene(invalid)

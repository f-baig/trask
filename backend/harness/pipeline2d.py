"""Shared 2D visual-controller machinery for latency pipeline evaluations.

All player-facing state in this module is derived from the forward-cone RGB frame,
except physical speed, which is the repository's permanently exposed scalar.  World
position, heading, centerline, checkpoints, and route progress are evaluator-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import Action
from .reflex.output import CommandOut, OutputState
from .reflex.runtime import ReflexRuntime
from .reflex.visual_2d import ConeVisionSense
from .vision import render_racing_forward_cone


CONE_PUBLIC_FIELDS = (
    "vision_center_near", "vision_center_far", "vision_turn_ahead",
    "vision_turn_severity", "vision_lookahead_depth", "vision_left_gap",
    "vision_right_gap", "vision_confidence", "vision_ego_road_contact",
    "vision_recovery_direction", "speed",
)


def normalized_speed(world) -> float:
    observation = world.observe()
    car_length_px = world.scene.dynamics.vehicle.length_m * world.scene.dynamics.pixels_per_meter
    return observation.speed * world.scene.dynamics.control_hz / car_length_px


def public_cone_state(values: dict) -> dict:
    """Compact state used for prediction and activation validation."""
    return {
        "speed": round(float(values.get("speed", 0.0)), 4),
        "center_near": round(float(values.get("vision_center_near", 0.0)), 4),
        "center_far": round(float(values.get("vision_center_far", 0.0)), 4),
        "turn_ahead": round(float(values.get("vision_turn_ahead", 0.0)), 4),
        "turn_severity": round(float(values.get("vision_turn_severity", 0.0)), 4),
        "visible_depth": round(float(values.get("vision_lookahead_depth", 0.0)), 4),
        "road_contact": bool(values.get("vision_ego_road_contact", False)),
        "confidence": round(float(values.get("vision_confidence", 0.0)), 4),
    }


def prediction_matches(plan, actual: dict) -> tuple[bool, dict]:
    predicted = plan.predicted.model_dump()
    speed_error = abs(actual["speed"] - predicted["speed"])
    lateral_error = abs(actual["center_near"] - predicted["center_near"])
    opposite_turn = (
        predicted["turn_ahead"] > 0.35 and actual["turn_ahead"] < -0.35
    ) or (
        predicted["turn_ahead"] < -0.35 and actual["turn_ahead"] > 0.35
    )
    contact_match = (
        actual["road_contact"] == predicted["road_contact"]
        or actual["confidence"] < 0.5
    )
    accepted = (
        speed_error <= max(0.35, float(plan.speed_tolerance))
        and lateral_error <= max(0.55, float(plan.lateral_tolerance))
        and not opposite_turn and contact_match
    )
    return accepted, {
        "predicted": predicted, "actual": actual,
        "speed_error": round(speed_error, 4),
        "lateral_error": round(lateral_error, 4),
        "opposite_turn": opposite_turn, "contact_match": contact_match,
        "accepted": accepted,
    }


class GeneratedConeDriver:
    """Install and execute model-authored sandbox controllers."""

    def __init__(self, scene):
        self.runtime = ReflexRuntime(scene, vision_only=True, visual_mode="2d")
        self.recent_controls: list[dict] = []
        self.writes: list[dict] = []
        self.last_install_error: str | None = None

    @property
    def current_source(self) -> str | None:
        record = self.runtime.controllers.get(self.runtime.active) if self.runtime.active else None
        return record.controller.source if record else None

    def observe(self, world) -> tuple[dict, object]:
        values = self.runtime.observe_visual(world)
        self.runtime.last_sense = values
        return public_cone_state(values), self.runtime.latest_frame

    def install(self, plan, tick: int) -> tuple[bool, str]:
        result = self.runtime.install(
            name="generated", source=plan.source, reads=plan.reads,
            safe_action={"steer": "hold", "throttle": -0.4}, activate=True,
        )
        accepted = bool(result.get("installed"))
        reason = "installed" if accepted else "; ".join(result.get("gate", {}).get("errors", ["gate rejected controller"]))
        self.last_install_error = None if accepted else reason
        self.writes.append({
            "tick": tick, "accepted": accepted, "reason": reason,
            "summary": plan.summary, "source": plan.source, "reads": plan.reads,
        })
        return accepted, reason

    def tick(self, world) -> Action:
        action = self.runtime.tick(world, None)
        self.recent_controls.append({"action": action.name.value, "keys": list(action.keys)})
        self.recent_controls = self.recent_controls[-16:]
        return action


@dataclass(frozen=True)
class ConeSkill:
    name: str = "stabilize"
    target_speed: float = 1.15
    target_offset: float = 0.0
    turn_direction: int = 0
    aggression: float = .78

    def clamped(self) -> "ConeSkill":
        ranges = {
            "follow_lane": (1.1, 2.5), "prepare_turn": (0.75, 1.7),
            "take_turn": (0.65, 1.55), "take_hairpin": (0.45, 1.15),
            "recover_track": (0.35, 0.8), "stabilize": (0.6, 1.35),
        }
        low, high = ranges[self.name]
        return ConeSkill(
            self.name, max(low, min(high, float(self.target_speed))),
            max(-0.75, min(0.75, float(self.target_offset))),
            max(-1, min(1, int(self.turn_direction))),
            max(0.0, min(1.0, float(self.aggression))),
        )


class ConeSkillDriver:
    """Execute reusable closed-loop primitives from fresh cone pixels."""

    def __init__(self, scene, aggression: float = .78):
        self.scene = scene
        self.sensor = ConeVisionSense()
        self.output_state = OutputState()
        self.aggression = max(0.0, min(1.0, float(aggression)))
        self.active = ConeSkill(aggression=self.aggression)
        self.last_values: dict = {}
        self.recent_controls: list[dict] = []
        self.activations: list[dict] = []
        self.last_control_terms: dict = {}

    def observe(self, world) -> tuple[dict, object]:
        frame = render_racing_forward_cone(world)
        return self.observe_frame(frame, normalized_speed(world))

    def observe_frame(self, frame, speed: float) -> tuple[dict, object]:
        """Measure one supplied cone frame without reading simulator state."""
        values = self.sensor.update(frame)
        values["speed"] = float(speed)
        self.last_values = values
        return public_cone_state(values), frame

    def install(self, plan, tick: int, aggression: float | None = None) -> tuple[bool, str]:
        applied_aggression = self.aggression if aggression is None else max(0.0, min(1.0, float(aggression)))
        self.active = ConeSkill(
            name=plan.skill, target_speed=plan.target_speed,
            target_offset=plan.target_offset, turn_direction=plan.turn_direction,
            aggression=applied_aggression,
        ).clamped()
        self.activations.append({"tick": tick, **asdict(self.active), "summary": plan.summary})
        return True, "activated"

    def tick(self, world) -> Action:
        state, _ = self.observe(world)
        return self.tick_state(state)

    def tick_state(self, state: dict) -> Action:
        """Run the active library skill from an already camera-derived state."""
        skill = self.active.clamped()
        aggression = skill.aggression
        contact = state["road_contact"]
        if not contact:
            # The compact public state omits recovery direction; take one fresh
            # sensor row for the primitive's camera-only recovery branch.
            recovery = float(self.last_values.get("vision_recovery_direction", 0.0))
            steer = recovery
            target_speed = min(skill.target_speed, 0.5)
        else:
            gains = {
                "follow_lane": 0.28, "prepare_turn": 0.48,
                "take_turn": 0.68, "take_hairpin": 0.90,
                "recover_track": 0.22, "stabilize": 0.16,
            }
            direction = state["turn_ahead"] if abs(state["turn_ahead"]) > 0.08 else skill.turn_direction
            steer = (
                0.82 * (state["center_near"] - skill.target_offset)
                + gains[skill.name] * (.85 + .30 * aggression) * direction
            )
            target_speed = skill.target_speed * (.86 + .26 * aggression)
            if state["turn_severity"] > 0.8:
                target_speed = min(
                    target_speed,
                    (.72 + .50 * aggression) if skill.name != "take_hairpin"
                    else (.55 + .34 * aggression),
                )
            if state["confidence"] < 0.5:
                target_speed = min(target_speed, .55 + .35 * aggression)
        throttle = 0.8 if state["speed"] < target_speed - 0.08 else -0.65 if state["speed"] > target_speed + 0.12 else 0.0
        out = CommandOut(state=self.output_state, on_track=contact)
        out.discretizer("hysteresis")
        out.steer(steer)
        out.throttle(throttle)
        action = out.resolve()
        self.last_control_terms = {
            "skill": skill.name, "aggression": round(aggression, 3),
            "target_speed": round(target_speed, 3), "steer_signal": round(steer, 3),
            "speed": round(state["speed"], 3), "road_contact": contact,
        }
        self.recent_controls.append({
            "action": action.name.value, "keys": list(action.keys), "skill": skill.name,
        })
        self.recent_controls = self.recent_controls[-16:]
        return action

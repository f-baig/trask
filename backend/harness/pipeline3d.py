"""Camera-grounded drivers for matched 3D controller-pipeline evaluations.

The player contract is one first-person RGB frame plus physical speed. Track geometry,
pose, elevation, progress, and checkpoints remain evaluator-only.
"""

from __future__ import annotations

from dataclasses import asdict

from .models import Action
from .predictive import DrivingSkill, SKILL_DEFAULTS, VisualSkillController, public_visual_state
from .reflex.runtime import ReflexRuntime
from .reflex.visual_3d import PerspectiveVisionSense
from .view3d import ViewMode, render_policy_view


def prediction_matches(plan, actual: dict) -> tuple[bool, dict]:
    predicted = plan.predicted.model_dump()
    errors = {
        "speed": abs(actual["speed"] - predicted["speed"]),
        "road_offset": abs(actual["road_offset"] - predicted["road_offset"]),
        "bend_ahead": abs(actual["bend_ahead"] - predicted["bend_ahead"]),
    }
    opposite_bend = (
        predicted["bend_ahead"] > 0.35 and actual["bend_ahead"] < -0.35
    ) or (
        predicted["bend_ahead"] < -0.35 and actual["bend_ahead"] > 0.35
    )
    contact_match = (
        actual["road_contact"] == predicted["road_contact"]
        or actual["confidence"] < 0.5
    )
    tolerances = {
        "speed": max(1.25, float(plan.speed_tolerance)),
        "road_offset": max(0.65, float(plan.offset_tolerance)),
        "bend_ahead": max(0.55, float(plan.bend_tolerance)),
    }
    accepted = (
        contact_match
        and errors["speed"] <= tolerances["speed"]
        and errors["road_offset"] <= tolerances["road_offset"]
        and not opposite_bend
    )
    return accepted, {
        "predicted": predicted, "actual": actual,
        "errors": {key: round(value, 4) for key, value in errors.items()},
        "tolerances": tolerances, "opposite_bend": opposite_bend,
        "contact_match": contact_match, "accepted": accepted,
    }


class GeneratedPerspectiveDriver:
    """Install exact model-authored controllers in the existing reflex sandbox."""

    def __init__(self, scene):
        self.runtime = ReflexRuntime(scene, vision_only=True, visual_mode="3d")
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
        return public_visual_state(values["speed"], values), self.runtime.latest_frame

    def install(self, plan, tick: int) -> tuple[bool, str]:
        result = self.runtime.install(
            name="generated-3d", source=plan.source, reads=plan.reads,
            safe_action={"steer": "hold", "throttle": -0.4}, activate=True,
        )
        accepted = bool(result.get("installed"))
        reason = (
            "installed" if accepted else
            "; ".join(result.get("gate", {}).get("errors", ["gate rejected controller"]))
        )
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


class PerspectiveSkillDriver:
    """Execute the existing reusable 3D skills from fresh camera measurements."""

    def __init__(self, scene):
        self.sensor = PerspectiveVisionSense()
        self.controller = VisualSkillController()
        self.active = SKILL_DEFAULTS["stabilize"]
        self.recent_controls: list[dict] = []
        self.activations: list[dict] = []
        self.last_values: dict = {}

    def observe(self, world) -> tuple[dict, object]:
        frame = render_policy_view(world, mode=ViewMode.FIRST_PERSON)
        values = self.sensor.update(frame)
        speed = world.observe().speed
        self.last_values = values
        return public_visual_state(speed, values), frame

    def install(self, plan, tick: int) -> tuple[bool, str]:
        self.active = DrivingSkill(
            name=plan.skill, target_speed=plan.target_speed,
            target_offset=plan.target_offset, turn_direction=plan.turn_direction,
        ).clamped()
        self.activations.append({"tick": tick, **asdict(self.active), "summary": plan.summary})
        return True, "activated"

    def tick(self, world) -> Action:
        state, _ = self.observe(world)
        action, _terms = self.controller.act(self.active, state["speed"], self.last_values)
        self.recent_controls.append({
            "action": action.name.value, "keys": list(action.keys), "skill": self.active.name,
        })
        self.recent_controls = self.recent_controls[-16:]
        return action

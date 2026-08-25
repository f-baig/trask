"""Latency-compensated, camera-grounded driving primitives.

The slow model chooses a primitive for the state it expects to exist when its
answer arrives.  The primitive then closes the loop from fresh screenshots on
every simulator tick.  Nothing in this module reads a track centerline, world
position, heading, elevation surface, checkpoint, or privileged physics state.
The only engine value admitted by the contract is scalar speed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Action, ActionName


SKILL_NAMES = (
    "follow_lane",
    "prepare_turn",
    "take_turn",
    "take_hairpin",
    "recover_track",
    "stabilize",
)


@dataclass(frozen=True)
class DrivingSkill:
    """One model-selected closed-loop primitive and its bounded parameters."""

    name: str
    target_speed: float
    target_offset: float = 0.0
    turn_direction: int = 0
    aggression: float = .78

    def clamped(self) -> "DrivingSkill":
        if self.name not in SKILL_NAMES:
            raise ValueError(f"unknown driving skill {self.name!r}")
        low_speed, high_speed = SKILL_SPEED_RANGES[self.name]
        return DrivingSkill(
            name=self.name,
            target_speed=max(low_speed, min(high_speed, float(self.target_speed))),
            target_offset=max(-0.75, min(0.75, float(self.target_offset))),
            turn_direction=max(-1, min(1, int(self.turn_direction))),
            aggression=max(0.0, min(1.0, float(self.aggression))),
        )


SKILL_DEFAULTS: dict[str, DrivingSkill] = {
    "follow_lane": DrivingSkill("follow_lane", 7.0),
    "prepare_turn": DrivingSkill("prepare_turn", 5.0),
    "take_turn": DrivingSkill("take_turn", 4.5),
    "take_hairpin": DrivingSkill("take_hairpin", 3.0),
    "recover_track": DrivingSkill("recover_track", 2.0),
    "stabilize": DrivingSkill("stabilize", 3.8),
}

SKILL_SPEED_RANGES: dict[str, tuple[float, float]] = {
    "follow_lane": (3.5, 10.0),
    "prepare_turn": (2.8, 7.5),
    "take_turn": (2.5, 7.0),
    "take_hairpin": (1.8, 5.0),
    "recover_track": (1.2, 3.0),
    "stabilize": (2.5, 5.0),
}


@dataclass
class VisualSkillController:
    """Execute a skill from fresh pixel measurements plus physical speed.

    Image signs follow the public 3D visual contract: positive offset, heading,
    bend, and recovery direction all point to image-right.  The controller is a
    proportional feedback law, not a route follower; it cannot steer toward
    road that is not visible in the current frame.
    """

    steer_deadband: float = 0.10
    speed_deadband: float = 0.35

    def act(self, skill: DrivingSkill, speed: float, sense: dict) -> tuple[Action, dict]:
        skill = skill.clamped()
        aggression = skill.aggression
        contact = bool(sense.get("vision_road_contact", False))
        confidence = float(sense.get("vision_confidence", 0.0))
        recovery = float(sense.get("vision_recovery_direction", 0.0))
        offset = float(sense.get("vision_track_offset", 0.0))
        heading = float(sense.get("vision_track_heading", 0.0))
        bend = float(sense.get("vision_bend_ahead", 0.0))
        severity = float(sense.get("vision_bend_severity", 0.0))
        crest = float(sense.get("vision_crest_risk", 0.0))
        visible_depth = float(sense.get("vision_visible_depth", 0.0))
        left_gap = float(sense.get("vision_left_gap", 0.0))
        right_gap = float(sense.get("vision_right_gap", 0.0))

        if not contact:
            steer = recovery
            target_speed = min(skill.target_speed, 1.2)
            mode = "visual recovery"
        else:
            anticipation = {
                "follow_lane": 0.16,
                "prepare_turn": 0.30,
                "take_turn": 0.42,
                "take_hairpin": 0.62,
                "recover_track": 0.18,
                "stabilize": 0.12,
            }[skill.name]
            direction = bend if abs(bend) >= 0.08 else float(skill.turn_direction)
            edge_balance = right_gap - left_gap
            steer = (
                1.12 * (offset - skill.target_offset)
                + 0.36 * heading
                + anticipation * (.85 + .30 * aggression) * direction
                + 0.72 * edge_balance
            )
            # These caps respond only to visible uncertainty.  The model still
            # chooses the nominal pace and primitive; the library prevents that
            # command from blindly accelerating into an occluded camera view.
            target_speed = skill.target_speed * (.86 + .26 * aggression)
            if skill.name in {"prepare_turn", "take_turn", "take_hairpin"}:
                visual_corner_cap = (6.0 - min(3.4, severity * 2.2)) * (.88 + .24 * aggression)
                target_speed = min(target_speed, visual_corner_cap)
            if crest > 0.55 or confidence < 0.55 or visible_depth < 0.50:
                target_speed = min(target_speed, 2.6 + 1.2 * aggression)
            mode = skill.name

        braking = speed > target_speed + 0.55
        throttling = speed < target_speed - self.speed_deadband
        longitudinal = "s" if braking else "w" if throttling else None
        if steer > self.steer_deadband:
            name, lateral = ActionName.RIGHT, "d"
        elif steer < -self.steer_deadband:
            name, lateral = ActionName.LEFT, "a"
        elif braking:
            name, lateral = ActionName.BACKWARD, None
        elif throttling:
            name, lateral = ActionName.FORWARD, None
        else:
            name, lateral = ActionName.IDLE, None
        keys = [key for key in (longitudinal, lateral) if key]
        return Action(name=name, keys=keys), {
            "skill": mode,
            "aggression": round(aggression, 3),
            "steer_signal": round(steer, 3),
            "target_speed": round(target_speed, 2),
            "speed": round(float(speed), 2),
            "road_contact": contact,
            "confidence": round(confidence, 3),
        }


def public_visual_state(speed: float, sense: dict) -> dict:
    """The complete state exposed to prediction and validation."""
    return {
        "speed": round(float(speed), 3),
        "road_offset": round(float(sense.get("vision_track_offset", 0.0)), 4),
        "road_heading": round(float(sense.get("vision_track_heading", 0.0)), 4),
        "bend_ahead": round(float(sense.get("vision_bend_ahead", 0.0)), 4),
        "bend_severity": round(float(sense.get("vision_bend_severity", 0.0)), 4),
        "visible_depth": round(float(sense.get("vision_visible_depth", 0.0)), 4),
        "road_contact": bool(sense.get("vision_road_contact", False)),
        "crest_risk": round(float(sense.get("vision_crest_risk", 0.0)), 4),
        "confidence": round(float(sense.get("vision_confidence", 0.0)), 4),
    }

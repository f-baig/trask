"""Camera-only within-race memory for repeated visual driving situations.

This module intentionally knows nothing about the racing world.  Its complete input
contract is a forward-cone ``VisualFrame``, the compact public cone measurements
derived from that frame, exposed physical speed, and a model-authored skill plan.
Lap number, pose, heading, checkpoints, centerline, surface grip, and evaluator
results are neither accepted nor stored.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from itertools import count

from PIL import Image

from .providers import ConeSkillPlan


_IDS = count(1)
_FEATURE_SCALES = {
    "speed": .55,
    "center_near": .32,
    "center_far": .40,
    "turn_ahead": .34,
    "turn_severity": .32,
    "visible_depth": .35,
}


def visual_fingerprint(frame) -> int:
    """Return a small perceptual fingerprint of the image the player received."""
    raw = base64.b64decode(frame.data_base64)
    with Image.open(io.BytesIO(raw)) as image:
        values = list(image.convert("L").resize((16, 12)).getdata())
    mean = sum(values) / max(1, len(values))
    fingerprint = 0
    for index, value in enumerate(values):
        if value >= mean:
            fingerprint |= 1 << index
    return fingerprint


def _hamming(left: int, right: int) -> float:
    return (left ^ right).bit_count() / 192


@dataclass(frozen=True)
class VisualSituation:
    public_state: dict
    fingerprint: int

    @classmethod
    def capture(cls, public_state: dict, frame) -> "VisualSituation":
        return cls(dict(public_state), visual_fingerprint(frame))


@dataclass
class SkillOutcome:
    situation: VisualSituation
    plan: ConeSkillPlan
    ticks: int
    road_contact_fraction: float
    mean_speed: float
    mean_abs_center_error: float
    source: str
    id: int = field(default_factory=lambda: next(_IDS))
    reuse_generation: int = 0

    @property
    def safe(self) -> bool:
        return (
            self.ticks >= 8
            and self.road_contact_fraction >= .985
            and self.mean_abs_center_error <= .58
        )

    @property
    def score(self) -> float:
        return round(
            self.mean_speed
            - 2.2 * (1 - self.road_contact_fraction)
            - .28 * self.mean_abs_center_error,
            4,
        )


@dataclass(frozen=True)
class MemoryMatch:
    outcome: SkillOutcome
    distance: float


class VisualSkillMemory:
    """Bounded nearest-neighbour recall over camera-derived situations."""

    def __init__(self, *, max_records: int = 160) -> None:
        self.max_records = max_records
        self.outcomes: list[SkillOutcome] = []

    @staticmethod
    def distance(left: VisualSituation, right: VisualSituation) -> float:
        a, b = left.public_state, right.public_state
        feature_distance = sum(
            abs(float(a.get(name, 0.0)) - float(b.get(name, 0.0))) / scale
            for name, scale in _FEATURE_SCALES.items()
        ) / len(_FEATURE_SCALES)
        contact_penalty = 2.0 if bool(a.get("road_contact")) != bool(b.get("road_contact")) else 0.0
        # The public measurements do most of the matching.  The low-weight image
        # term separates visually different stretches that share a similar bend.
        return round(feature_distance + .55 * _hamming(left.fingerprint, right.fingerprint) + contact_penalty, 4)

    def add(self, outcome: SkillOutcome) -> None:
        self.outcomes.append(outcome)
        self.outcomes = self.outcomes[-self.max_records :]

    def matches(self, situation: VisualSituation, *, limit: int = 4) -> list[MemoryMatch]:
        ranked = sorted(
            (MemoryMatch(item, self.distance(situation, item.situation)) for item in self.outcomes),
            key=lambda item: (item.distance, -item.outcome.score),
        )
        return ranked[:limit]

    def reusable(self, situation: VisualSituation, *, threshold: float = .38) -> MemoryMatch | None:
        candidates = [
            match for match in self.matches(situation, limit=12)
            if match.distance <= threshold and match.outcome.safe
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.outcome.score - .18 * item.distance)

    def retrieved_context(self, situation: VisualSituation, *, limit: int = 3) -> list[dict]:
        records = []
        for match in self.matches(situation, limit=limit):
            outcome = match.outcome
            records.append({
                "visual_match_distance": match.distance,
                "skill": outcome.plan.skill,
                "target_speed": outcome.plan.target_speed,
                "target_offset": outcome.plan.target_offset,
                "turn_direction": outcome.plan.turn_direction,
                "observed_ticks": outcome.ticks,
                "road_contact_fraction": outcome.road_contact_fraction,
                "mean_exposed_speed": outcome.mean_speed,
                "mean_abs_visual_center_error": outcome.mean_abs_center_error,
                "safe": outcome.safe,
                "lesson": (
                    "This camera-matched skill held the visible road; it is a candidate for reuse."
                    if outcome.safe else
                    "This camera-matched skill was unstable; choose a safer skill or lower target speed."
                ),
            })
        return records

    def tuned_replay(self, match: MemoryMatch, current_state: dict) -> ConeSkillPlan:
        """Replay a prior model choice and cautiously probe a faster safe target."""
        outcome = match.outcome
        safe_faster = outcome.plan.target_speed + min(.12, .025 * (outcome.reuse_generation + 1))
        nearby_failures = [
            other.outcome for other in self.matches(outcome.situation, limit=16)
            if not other.outcome.safe and other.distance <= .30
            and other.outcome.plan.skill == outcome.plan.skill
            and other.outcome.plan.target_speed > outcome.plan.target_speed
        ]
        if nearby_failures:
            safe_faster = min(safe_faster, min(item.plan.target_speed for item in nearby_failures) - .05)
        outcome.reuse_generation += 1
        prediction = {
            "speed": max(0.0, min(3.0, float(current_state["speed"]))),
            "center_near": max(-2.0, min(2.0, float(current_state["center_near"]))),
            "turn_ahead": max(-2.0, min(2.0, float(current_state["turn_ahead"]))),
            "road_contact": bool(current_state["road_contact"]),
        }
        return outcome.plan.model_copy(update={
            "target_speed": max(.2, min(2.5, safe_faster)),
            "predicted": prediction,
            "speed_tolerance": 2.0,
            "lateral_tolerance": 2.0,
            "summary": (
                f"Camera-memory replay of {outcome.plan.skill}; safe-pass speed probe "
                f"from {outcome.plan.target_speed:.2f}."
            ),
        })

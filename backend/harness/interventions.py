"""Deterministic natural-language resolution for replay-fork interventions.

The engine has a deliberately small perturbation vocabulary.  Resolving a sentence into
that vocabulary locally keeps a fork reproducible and prevents an unrecorded model call
from becoming part of the experimental treatment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ForkIntervention:
    perturbation: str
    guidance: str | None
    summary: str


_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("action_delay", (
        r"\baction delay\b", r"\binput delay\b", r"\bcontrol delay\b",
        r"\bsteering delay\b", r"\binput lag\b", r"\bcontrol lag\b",
    )),
    ("obstacle_shift", (
        r"\b(?:shift|move|relocate|reposition)(?: the)? (?:barriers?|obstacles?)\b",
        r"\b(?:barrier|obstacle) shift\b",
    )),
    ("worn_tires", (
        r"\bworn (?:tires|tyres)\b", r"\bbald (?:tires|tyres)\b",
        r"\bdegraded (?:tires|tyres)\b",
    )),
    ("heavy_car", (
        r"\bheav(?:y|ier) car\b", r"\bincrease(?:d)? (?:the )?(?:car )?mass\b",
        r"\bextra (?:car )?weight\b", r"\badd weight\b",
    )),
    ("rear_bias", (
        r"\brear[- ](?:weight )?bias\b", r"\brear[- ]heavy\b",
        r"\bmove (?:the )?weight rearward\b",
    )),
    ("high_downforce", (
        r"\bhigh downforce\b", r"\bmore downforce\b", r"\bincrease(?:d)? downforce\b",
    )),
    ("high_drag", (
        r"\bhigh drag\b", r"\bmore drag\b", r"\bincrease(?:d)? (?:aerodynamic )?drag\b",
        r"\bmore air resistance\b",
    )),
    ("low_grip", (
        r"\blow grip\b", r"\bless grip\b", r"\breduc(?:e|ed) grip\b",
        r"\blower (?:the )?(?:much )?grip\b", r"\b(?:the )?grip (?:much )?lower\b",
        r"\bslippery\b", r"\bicy\b",
        r"\blow friction\b", r"\bless friction\b", r"\breduc(?:e|ed) friction\b",
        r"\blower (?:the )?(?:much )?friction\b", r"\b(?:the )?friction (?:much )?lower\b",
    )),
)

_UNCHANGED = (
    r"\breplay (?:it )?exactly\b", r"\bno (?:physics )?(?:change|perturbation)\b",
    r"\bkeep (?:the )?physics (?:the )?same\b", r"\bunchanged physics\b",
    r"\bnormal conditions?\b",
)

_DRIVER_INSTRUCTION = re.compile(
    r"\b(?:driver|agent)\b|"
    r"\b(?:brake|steer|accelerate|throttle|slow down|speed up|hold|follow|avoid|recover|"
    r"stay|aim|turn)\b",
    re.IGNORECASE,
)


def resolve_fork_condition(condition: str | None) -> ForkIntervention:
    """Resolve one sentence into one engine perturbation and optional driver guidance."""
    text = " ".join((condition or "").strip().split())
    if not text:
        return ForkIntervention("none", None, "Replay from the selected tick with unchanged conditions.")
    lowered = text.lower()
    if re.search(r"\bfog(?:gy)?\b|\blow visibility\b|\breduced visibility\b", lowered):
        raise ValueError(
            "Fog is not an available condition: it did not alter the player camera, so it was retired."
        )
    matched = [
        name for name, patterns in _PATTERNS
        if any(re.search(pattern, lowered) for pattern in patterns)
    ]
    if len(matched) > 1:
        raise ValueError(
            "A fork can apply one engine perturbation at a time; this condition describes "
            f"multiple ({', '.join(matched)}). Choose one physical change."
        )
    perturbation = matched[0] if matched else "none"
    unchanged = any(re.search(pattern, lowered) for pattern in _UNCHANGED)

    # A sentence with no physical preset is a direct instruction to a model-backed player.
    # When a physical preset is present, forward the sentence only if it also explicitly
    # talks about driving; environmental facts alone remain blind perturbations.
    guidance = None
    if not matched and (not unchanged or _DRIVER_INSTRUCTION.search(text)):
        guidance = text
    elif matched and _DRIVER_INSTRUCTION.search(text):
        guidance = text

    summary = (
        f"Apply {perturbation.replace('_', ' ')} from the selected tick."
        if perturbation != "none" else
        "Replay from the selected tick with unchanged engine conditions."
    )
    if guidance:
        summary += " Pass the stated correction to each subsequent player decision."
    return ForkIntervention(perturbation, guidance, summary)

"""Remembering which engine settings actually satisfied a requirement, once a person said so.

The harness can already measure whether a circuit matches the contract read from a brief.
What it cannot measure is whether that reading was *right* — whether `grip_max=0.45` is
what "slippery" meant to the person who typed it. Only they know, so the coordinator asks,
and the answer is worth keeping.

A `Precedent` is one confirmed pairing: a requirement, the check that expressed it, and the
engine settings that delivered it, admitted only when both gates pass — the simulator
measured it as satisfied, and a human said it was what they wanted. Neither alone is
enough. A measurement can confirm a check that meant the wrong thing, and a compliment can
land on a circuit that missed.

Retrieval is a lookup, not a context dump. Precedents are indexed by check kind, which is
already the harness's shared vocabulary, so answering "has anything like this been asked
before" is a `WHERE` against one column. A generation only ever sees precedents for the
kinds its own contract contains, one or two lines each, and sees nothing at all when the
table has nothing to say.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .models import SceneSpec
from .prompt_spec import PromptSpec, Requirement, RequirementVerdict

MAX_PER_KIND = 2
"""Precedents shown per check kind. This is a hint, not a corpus."""


class Precedent(BaseModel):
    """One requirement whose realisation a person confirmed was what they meant."""

    id: str
    created_at: str
    check_kind: str
    """The index. Also the harness's shared vocabulary, which is why it works as one."""
    check_target: Any = None
    check_tolerance: float = 0.0
    statement: str = ""
    quote: str = ""
    """The user's own words, kept so a later hint can show how this was actually phrased."""
    category: str = "constraint"
    settings: dict[str, Any] = Field(default_factory=dict)
    """The compiled scene values that delivered it — the reusable part."""
    measured: str = ""
    confirmed: bool = True
    scene_id: str = ""
    note: str = ""
    """Anything the person said about it beyond yes or no."""

    def hint(self) -> str:
        """One line, as a creator is shown it."""
        settings = ", ".join(f"{key}={value!r}" for key, value in self.settings.items())
        phrasing = f' (they phrased it "{self.quote}")' if self.quote else ""
        return (
            f"{self.check_kind}={self.check_target!r}{phrasing} was satisfied with "
            f"{settings or 'the defaults'}, and the user confirmed that was what they wanted"
            + (f" — {self.note}" if self.note else "")
        )


# Which compiled-scene facts are worth remembering for each kind of check. Recording the
# whole scene would make every precedent a haystack; recording nothing would make it a
# label with no reusable content. These are the dials that actually move each check.
_SETTINGS_FOR_KIND: dict[str, tuple[str, ...]] = {
    "surface": ("surface", "grip"),
    "grip_max": ("grip", "surface"), "grip_min": ("grip", "surface"),
    "grip_target": ("grip", "surface"),
    "track_width_max": ("corridor_width_px",), "track_width_min": ("corridor_width_px",),
    "laps": ("laps",), "direction": ("direction",), "npc_start_mode": ("npc_start_mode",),
    "npc_count": ("opponents",), "npc_profiles": ("opponents",),
    "barrier_count": ("barrier_count",),
    "corner_count_min": ("corner_count",), "corner_count_max": ("corner_count",),
    "corner_in_region": ("corner_count", "corners"),
    "min_radius_max": ("tightest_corner_radius_px", "corner_count"),
    "min_radius_min": ("tightest_corner_radius_px", "corner_count"),
    "longest_straight_min": ("longest_straight_px", "corner_count"),
    "longest_straight_max": ("longest_straight_px", "corner_count"),
    "angle_fidelity_max": ("corners",), "closure_error_max": ("corners",),
}
_DEFAULT_SETTINGS: tuple[str, ...] = ("surface", "grip", "corridor_width_px", "corner_count")
"""Outcome checks have no single dial, so the whole handling envelope is what mattered."""


def distil(
    spec: PromptSpec, scene: SceneSpec, verdicts: list[RequirementVerdict],
    confirmations: dict[str, bool], notes: dict[str, str] | None = None,
    scene_facts_fn=None, now: str = "",
) -> list[Precedent]:
    """Turn one confirmed build into precedents, one per satisfied, confirmed check.

    Both gates are applied here rather than at the call site, so there is no path that
    stores a precedent on a measurement alone or on a compliment alone.
    """
    from .fidelity import scene_facts

    facts = (scene_facts_fn or scene_facts)(scene)
    by_id = {item.id: item for item in verdicts}
    notes = notes or {}
    found: list[Precedent] = []
    for requirement in spec.requirements:
        verdict = by_id.get(requirement.id)
        if verdict is None or not verdict.satisfied:
            # A requirement the circuit missed teaches nothing about how to hit it, however
            # generous the person was about the rest.
            continue
        if not confirmations.get(requirement.id, False):
            continue
        for index, check in enumerate(requirement.checks):
            found.append(Precedent(
                id=f"prec-{scene.id}-{requirement.id}-{index}",
                created_at=now,
                check_kind=check.kind,
                check_target=check.target,
                check_tolerance=check.tolerance,
                statement=requirement.statement,
                quote=requirement.quote,
                category=requirement.category,
                settings=_settings_for(check.kind, facts),
                measured=verdict.evidence[:200],
                confirmed=True,
                scene_id=scene.id,
                note=notes.get(requirement.id, "")[:200],
            ))
    return found


def _settings_for(kind: str, facts: dict[str, Any]) -> dict[str, Any]:
    wanted = _SETTINGS_FOR_KIND.get(kind, _DEFAULT_SETTINGS)
    return {key: facts[key] for key in wanted if key in facts}


def guidance(spec: PromptSpec, lookup, limit: int = MAX_PER_KIND) -> str:
    """Precedents for the kinds this contract actually contains, and nothing else.

    `lookup(kind) -> list[Precedent]` is the store query, injected so this module stays
    free of storage. The result is empty far more often than not, which is the point: a
    generation pays no context for a table with nothing relevant in it.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for requirement in spec.requirements:
        for check in requirement.checks:
            if check.kind in seen:
                continue
            seen.add(check.kind)
            for precedent in _rank(lookup(check.kind), check.target)[:limit]:
                lines.append(f"  {requirement.id}: {precedent.hint()}")
    if not lines:
        return ""
    return (
        "WHAT WORKED BEFORE — earlier circuits whose requirements this user confirmed were "
        "read correctly. Treat these as evidence about what their words usually mean, not "
        "as instructions; the contract above still governs.\n" + "\n".join(lines)
    )


def _rank(found: list[Precedent], target: Any) -> list[Precedent]:
    """Closest target first, so a hint is about this ask rather than a distant cousin."""
    def distance(precedent: Precedent) -> float:
        left, right = precedent.check_target, target
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not isinstance(left, bool) and not isinstance(right, bool):
                return abs(float(left) - float(right))
        return 0.0 if left == right else 1e6

    return sorted(found, key=distance)


def outstanding(spec: PromptSpec, verdicts: list[RequirementVerdict]) -> list[Requirement]:
    """Requirements worth asking the user about, hardest-to-judge first.

    A measured miss is already reported, so asking about it adds nothing. The valuable
    question is about the ones the harness believes it got right, because that belief rests
    on a reading of their words that only they can confirm.
    """
    satisfied = {item.id for item in verdicts if item.satisfied}
    return [item for item in spec.requirements if item.id in satisfied]

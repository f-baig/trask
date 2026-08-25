"""Closing fidelity gaps by local search instead of by asking the model again.

Some requirements cannot be authored, only tuned. A creator can place a corner in
the top right because that is a decision; it cannot author "an unbraked driver
runs wide for forty ticks", because that is a property of the simulation, not of
the plan. Asking a model to hit one by writing a different plan is asking it to
guess at the output of a physics engine it cannot run.

So the harness runs it instead. These moves are plain coordinate descent over the
plan's continuous dials, and every candidate is compiled and measured through the
same runtime that grades the final scene. It costs no model calls at all, which
is why it runs before any repair round: there is no reason to spend a model call
on something a search can settle for free.

Two rules keep the search honest. Moves are proposed only for dials that could
plausibly move a *failing* requirement, so a scene is never perturbed for no
reason. And a move is accepted only if it strictly improves fidelity without
regressing anything that already held — trading a satisfied requirement for an
unsatisfied one is not progress, it is just a different set of misses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import PlayabilityCertificate, SceneSpec
from .probes import ProbeReport, measure
from .prompt_spec import FidelityReport, PromptSpec, RequirementImplementation
from .racing import compile_certified_scene
from .track_grammar import CornerRadius, NpcSpec, TrackPlan

MAX_VARIANTS = 14
"""Compiles per solve. Each is a compile plus, when needed, a rollout."""

_RADIUS_LADDER: tuple[CornerRadius, ...] = (
    CornerRadius.HAIRPIN, CornerRadius.TIGHT, CornerRadius.MEDIUM,
    CornerRadius.OPEN, CornerRadius.SWEEPING,
)

# Which dial can move which check. A requirement that fails on a kind listed here
# makes its dials eligible; a failing kind in none of these lists is not something
# tuning can fix, and the solver correctly proposes nothing for it.
_DIALS_FOR_KIND: dict[str, tuple[str, ...]] = {
    "grip_max": ("grip",), "grip_min": ("grip",), "grip_target": ("grip",),
    "track_width_max": ("width",), "track_width_min": ("width",),
    "min_radius_min": ("radius",), "min_radius_max": ("radius",),
    "longest_straight_min": ("radius",), "longest_straight_max": ("radius",),
    "brake_fraction_min": ("grip", "radius"), "brake_fraction_max": ("grip", "radius"),
    "mean_speed_min": ("grip", "width", "radius"),
    "mean_speed_max": ("grip", "width", "radius"),
    "oracle_seconds": ("grip", "radius"), "oracle_seconds_max": ("grip", "radius"),
    "off_track_ticks_max": ("grip", "width"),
    "naive_finishes": ("grip", "width", "radius"),
    "naive_off_track_min": ("grip", "width", "radius"),
    "naive_off_track_max": ("grip", "width", "radius"),
    "oracle_finishes": ("grip", "width"),
    "field_spread_max": ("pace",), "order_changes_min": ("pace",),
}


@dataclass
class SolveOutcome:
    plan: TrackPlan
    scene: SceneSpec
    certificate: PlayabilityCertificate
    probes: ProbeReport | None
    report: FidelityReport
    moves: list[str]
    variants_tried: int = 0


Verify = Callable[[SceneSpec, ProbeReport | None], FidelityReport]
Compile = Callable[..., tuple[SceneSpec, PlayabilityCertificate, list[str]]]


def solve(
    spec: PromptSpec, plan: TrackPlan, scene: SceneSpec,
    certificate: PlayabilityCertificate, probes: ProbeReport | None,
    report: FidelityReport, verify: Verify, seed: int,
    compile_scene: Compile | None = None,
    mapping: list[RequirementImplementation] | None = None,
    max_variants: int = MAX_VARIANTS,
) -> SolveOutcome:
    """Descend the continuous dials until fidelity stops improving."""
    compile_scene = compile_scene or compile_certified_scene
    best = SolveOutcome(plan, scene, certificate, probes, report, moves=[])
    if report.faithful:
        return best

    wanted = _eligible_dials(spec, report)
    if not wanted:
        return best
    probing = probes is not None

    for label, variant in _variants(plan, wanted, max_variants):
        best.variants_tried += 1
        try:
            candidate, candidate_certificate, _notes = compile_scene(
                spec.prompt, variant, seed,
            )
        except ValueError:
            continue
        # A rollout is the expensive part of a variant, and most rejected variants
        # are rejected on geometry the compiler already settled for free. Verify
        # without probes first and discard anything that has broken a requirement
        # holding on the incumbent, so only plausible variants are ever simulated.
        if probing and _regresses(verify(candidate, None), best.report):
            continue
        candidate_probes = measure(candidate) if probing else None
        candidate_report = verify(candidate, candidate_probes)
        if not _better(candidate_report, best.report):
            continue
        best = SolveOutcome(
            plan=variant, scene=candidate, certificate=candidate_certificate,
            probes=candidate_probes, report=candidate_report,
            moves=[*best.moves, f"{label} -> {candidate_report.summary()}"],
            variants_tried=best.variants_tried,
        )
        if candidate_report.faithful:
            break
    return best


def _better(candidate: FidelityReport, incumbent: FidelityReport) -> bool:
    """Strictly better, and nothing that already held has broken.

    The regression guard matters more than the improvement test. Without it a
    move that satisfies two new requirements while quietly breaking one reads as
    progress, and the search walks away from a circuit that was closer to the
    brief than the one it ends on.
    """
    held = {item.id for item in incumbent.verdicts if item.satisfied}
    broken = {
        item.id for item in candidate.verdicts if not item.satisfied and item.id in held
    }
    if broken:
        return False
    if candidate.satisfied != incumbent.satisfied:
        return candidate.satisfied > incumbent.satisfied
    return candidate.residual < incumbent.residual - 1e-9


def _regresses(candidate: FidelityReport, incumbent: FidelityReport) -> bool:
    """Has this variant broken something the incumbent already satisfied?

    Used as the free pre-filter, so it is evaluated on a probe-less report where
    every probe-backed requirement reads as unsatisfied. Only requirements the
    incumbent settled by measurement can count as regressions here, or every
    variant would look like it had broken every outcome requirement at once.
    """
    held = {
        item.id for item in incumbent.verdicts
        if item.satisfied and item.method == "check"
    }
    return any(
        item.id in held and not item.satisfied and item.method == "check"
        for item in candidate.verdicts
    )


def _eligible_dials(spec: PromptSpec, report: FidelityReport) -> set[str]:
    """Only dials that could move something currently failing."""
    failing = {item.id for item in report.failures()}
    dials: set[str] = set()
    for requirement in spec.requirements:
        if requirement.id not in failing:
            continue
        for check in requirement.checks:
            dials.update(_DIALS_FOR_KIND.get(check.kind, ()))
    return dials


def _variants(
    plan: TrackPlan, wanted: set[str], limit: int,
) -> list[tuple[str, TrackPlan]]:
    """Candidate moves, ordered so the cheapest and most likely come first."""
    out: list[tuple[str, TrackPlan]] = []
    if "grip" in wanted:
        for grip in (0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.2):
            if abs(grip - plan.grip) > 1e-6:
                out.append((f"grip={grip:g}", plan.model_copy(update={"grip": grip})))
    if "width" in wanted:
        for width in (110.0, 120.0, 132.0, 150.0, 170.0):
            if abs(width - plan.track_width) > 1e-6:
                out.append((f"width={width:g}", plan.model_copy(update={"track_width": width})))
    if "radius" in wanted:
        for step, name in ((1, "open"), (-1, "tighten"), (2, "open2"), (-2, "tighten2")):
            shifted = _shift_radii(plan, step)
            if shifted is not None:
                out.append((f"radius={name}", shifted))
    if "pace" in wanted and plan.npcs:
        out.append(("field=compressed", _repaced(plan, 0.05)))
        out.append(("field=spread", _repaced(plan, 0.35)))
    return out[:limit]


def _shift_radii(plan: TrackPlan, step: int) -> TrackPlan | None:
    """Move every corner the same number of steps along the radius ladder.

    Uniform rather than per-corner: the compiler scales the whole path into a
    fixed box, so a single corner's category barely moves the achieved geometry
    while the whole ladder does. Returns None when nothing can shift, which is
    how the search learns it has hit the end of the ladder.
    """
    corners = []
    moved = False
    for corner in plan.corners:
        index = _RADIUS_LADDER.index(corner.radius)
        target = max(0, min(len(_RADIUS_LADDER) - 1, index + step))
        moved = moved or target != index
        corners.append(corner.model_copy(update={"radius": _RADIUS_LADDER[target]}))
    return plan.model_copy(update={"corners": corners}) if moved else None


def _repaced(plan: TrackPlan, spread: float) -> TrackPlan:
    """Rewrite the field's pace ladder without touching temperament.

    Temperament is what a brief names; pace within a temperament is a separate
    axis, which is what makes a finish-spread target solvable without abandoning
    the opponents the user actually asked for.
    """
    count = len(plan.npcs)
    npcs: list[NpcSpec] = []
    for index, npc in enumerate(plan.npcs):
        offset = 0.0 if count < 2 else spread * index / (count - 1)
        npcs.append(npc.model_copy(
            update={"pace": round(max(0.35, min(1.05, 0.95 - offset)), 3)},
        ))
    return plan.model_copy(update={"npcs": npcs})

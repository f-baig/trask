"""Does the world that got built actually satisfy the brief that asked for it?

Deliberately separate from playability. The engine verifier answers "is this a
valid, completable circuit"; this answers "is this the circuit that was asked
for". Both can pass while the other fails, and the two were conflated for long
enough that an unfaithful scene which happened to compile was reported as a
success. Keeping them apart is the whole point: a world can be perfectly valid
and completely wrong.

Verification runs requirement by requirement against the `PromptSpec`, so a
result is never a single number. It is a list of ids, each either settled by a
measurement or, when nothing in the engine can measure it, by a judge that is
shown only what was measured — never the creator's own description of what it
thinks it built. A generator that writes "a thrilling rollercoaster of a circuit"
in its rationale must not be able to pass a rollercoaster requirement with it.
"""

from __future__ import annotations

import json
from typing import Any

from .generation_spec import Assertion, _Context, _EVALUATORS, _PROBE_KINDS, _residual
from .models import SceneSpec
from .probes import ProbeReport
from .prompt_spec import (
    FidelityReport, PromptSpec, Requirement, RequirementImplementation, RequirementVerdict,
)
from .providers import ProviderError, active_provider, anthropic_json, configured_model


def verify(
    spec: PromptSpec,
    scene: SceneSpec | None,
    probes: ProbeReport | None = None,
    mapping: list[RequirementImplementation] | None = None,
    provider: str = "auto",
    judge: bool = True,
) -> FidelityReport:
    """Settle every requirement against the compiled world.

    `probes` may be omitted when the caller cannot afford rollouts; probe-backed
    checks then report as unverifiable rather than quietly passing. An unmeasured
    requirement is not a satisfied one, and recording the difference is what stops
    a cheap inline check from being mistaken for a full grade.
    """
    claimed = {item.id: item for item in (mapping or [])}
    verdicts: list[RequirementVerdict] = []
    judgeable: list[Requirement] = []

    for requirement in spec.requirements:
        location = claimed.get(requirement.id)
        if scene is None:
            verdicts.append(_verdict(
                requirement, False, "check", "no scene was produced", 1.0, location,
            ))
            continue
        if not requirement.mechanical:
            judgeable.append(requirement)
            continue
        verdicts.append(_settle(requirement, scene, probes, location))

    report = FidelityReport(verdicts=verdicts)
    if judgeable and scene is not None and judge:
        rulings, calls = _judge(spec, judgeable, scene, probes, provider)
        report.judge_calls = calls
        for requirement in judgeable:
            ruling = rulings.get(requirement.id)
            report.verdicts.append(_verdict(
                requirement,
                satisfied=bool(ruling and ruling.get("satisfied")),
                method="judge" if ruling else "unverifiable",
                evidence=str((ruling or {}).get("reason") or "the judge returned no ruling"),
                residual=0.0 if ruling and ruling.get("satisfied") else 1.0,
                location=claimed.get(requirement.id),
            ))
    elif judgeable:
        for requirement in judgeable:
            report.verdicts.append(_verdict(
                requirement, False, "unverifiable",
                "needs a judge, which was not run", 1.0, claimed.get(requirement.id),
            ))

    # Requirement order, so a report reads in the order the user said things.
    order = {requirement.id: index for index, requirement in enumerate(spec.requirements)}
    report.verdicts.sort(key=lambda item: order.get(item.id, 999))
    return report


def _settle(
    requirement: Requirement, scene: SceneSpec, probes: ProbeReport | None,
    location: RequirementImplementation | None,
) -> RequirementVerdict:
    """Run every check on one requirement. All must hold for it to be satisfied."""
    context = _Context(scene, probes)
    failures: list[str] = []
    evidence: list[str] = []
    residual = 0.0
    unverifiable = False

    for check in requirement.checks:
        if check.kind in _PROBE_KINDS and probes is None:
            unverifiable = True
            failures.append(f"{check.kind} needs a simulator rollout, which was not run")
            continue
        evaluator = _EVALUATORS.get(check.kind)
        if evaluator is None:
            unverifiable = True
            failures.append(f"no evaluator for {check.kind}")
            continue
        assertion = Assertion(
            id=f"{requirement.id}-{check.kind}", kind=check.kind,
            target=check.target, tolerance=check.tolerance,
            label=requirement.statement,
        )
        try:
            satisfied, achieved, message = evaluator(assertion, context)
        except (ValueError, KeyError, AttributeError, TypeError) as error:
            unverifiable = True
            failures.append(f"{check.kind} could not be measured: {error}")
            continue
        if satisfied:
            evidence.append(f"{check.kind}={achieved!r}")
        else:
            failures.append(message or f"{check.kind} did not hold")
            residual += _residual(assertion, satisfied, achieved)

    if failures:
        return _verdict(
            requirement, False, "unverifiable" if unverifiable and not evidence else "check",
            "; ".join(failures)[:400], residual or 1.0, location,
        )
    return _verdict(requirement, True, "check", "; ".join(evidence)[:400], 0.0, location)


def _verdict(
    requirement: Requirement, satisfied: bool, method: str, evidence: str,
    residual: float, location: RequirementImplementation | None,
) -> RequirementVerdict:
    return RequirementVerdict(
        id=requirement.id, category=requirement.category, statement=requirement.statement,
        quote=requirement.quote, priority=requirement.priority,
        satisfied=satisfied, method=method, evidence=evidence,
        residual=round(residual, 5),
        claimed_at=(location.location if location else ""),
    )


# --- the measured factsheet ----------------------------------------------------


def scene_facts(scene: SceneSpec, probes: ProbeReport | None = None) -> dict[str, Any]:
    """Everything true about a compiled scene, measured rather than described.

    The creator's title and rationale are deliberately excluded. A judge shown the
    generator's own prose about its circuit is grading the prose, and a confident
    sentence would then be able to satisfy a requirement the geometry does not.
    """
    report = scene.track_report
    facts: dict[str, Any] = {
        "surface": scene.surface,
        "grip": round(scene.grip, 3),
        "corridor_width_px": round(scene.track_width, 1),
        "laps": scene.laps,
        "npc_start_mode": scene.npc_start_mode,
        "start_line_region": scene.start_line_region.value,
        "player_grid_position": scene.player_grid_position,
        "opponents": [
            {"profile": behavior.profile.value, "pace": round(behavior.pace, 3),
             "skill": round(behavior.skill, 3), "intelligence": round(behavior.intelligence, 3),
             "aggression": round(behavior.aggression, 3)}
            for behavior in scene.npc_behaviors
        ],
        "barrier_count": sum(1 for entity in scene.entities if entity.kind == "obstacle"),
    }
    if report:
        facts.update({
            "loop_shape": report.loop_shape,
            "direction": report.direction,
            "corner_count": len(report.corners),
            "corners": [
                {"index": corner.index, "turns": corner.direction,
                 "angle_deg": round(corner.achieved_angle_degrees, 1),
                 "radius_px": round(corner.achieved_radius_pixels, 1),
                 "region": corner.achieved_region.value,
                 "entry_speed": round(corner.recommended_entry_speed, 2)}
                for corner in report.corners
            ],
            "lap_length_px": round(report.length_pixels, 1),
            "longest_straight_px": round(report.longest_straight_pixels, 1),
            "tightest_corner_radius_px": round(report.minimum_radius_pixels, 1),
            "compiler_relaxations": report.relaxations,
        })
    if probes:
        facts["measured"] = {
            "reference_driver_finished": probes.oracle_finished,
            "reference_race_seconds": probes.oracle_seconds,
            "reference_mean_speed": round(probes.oracle_mean_speed, 2),
            "reference_off_track_ticks": probes.off_track_ticks,
            "share_of_race_braking": round(probes.brake_fraction, 3),
            "unbraked_driver_finished": probes.naive_finished,
            "unbraked_driver_off_track_ticks": probes.naive_off_track_ticks,
            "position_changes": probes.order_changes,
            "opponent_finish_spread_seconds": probes.field_spread_seconds,
        }
    return facts


# --- the judge -----------------------------------------------------------------


_JUDGE_SYSTEM = (
    "You verify whether a generated 2D racing circuit satisfies specific requirements taken "
    "from a user's brief. You are shown only MEASURED facts about the compiled circuit — its "
    "real geometry, physics, and simulated outcomes. You are never shown the generator's own "
    "description of what it thinks it built, because that is the claim under test.\n\n"
    "Rule each requirement satisfied only if the measured facts actually support it. "
    "Uncertain means not satisfied: a requirement the facts do not evidence has not been "
    "delivered, whatever the numbers might allow. Give a one-sentence reason citing the "
    "specific measured values you used, so a person can check your reasoning against the "
    "same table.\n\n"
    "Be concrete about racing feel. 'Sweeping' means large corner radii. 'Technical' means "
    "many corners and short straights. A theme or colour requirement is satisfied by the "
    "visual plan's palette, not by the track geometry."
)


def _judge(
    spec: PromptSpec, requirements: list[Requirement], scene: SceneSpec,
    probes: ProbeReport | None, provider: str,
) -> tuple[dict[str, dict], int]:
    """Rule on the requirements no evaluator can settle, in one batched call."""
    resolved = active_provider() if provider == "auto" else provider
    if resolved not in {"anthropic", "openai"}:
        return {}, 0

    listing = "\n".join(
        f"  {item.id}: {item.statement}" + (f'   (user said: "{item.quote}")' if item.quote else "")
        for item in requirements
    )
    visual = getattr(scene, "visual", None)
    facts = scene_facts(scene, probes)
    if visual is not None:
        facts["visual_plan"] = visual.model_dump(exclude_none=True)
    try:
        payload, _ = anthropic_json(
            model=configured_model("ANTHROPIC_FIDELITY_MODEL"),
            max_tokens=1_600,
            system=_JUDGE_SYSTEM,
            prompt=(
                f"The user's brief was:\n{spec.prompt}\n\n"
                f"Requirements to rule on:\n{listing}\n\n"
                f"Measured facts about the circuit that was built:\n"
                f"{json.dumps(facts, indent=1, default=str)}\n\n"
                "Rule on every requirement listed, by id."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "rulings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "satisfied": {"type": "boolean"},
                                "reason": {"type": "string", "maxLength": 300},
                            },
                            "required": ["id", "satisfied", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["rulings"],
                "additionalProperties": False,
            },
            cache_system=True,
        )
    except ProviderError:
        # A judge outage must not be reported as fidelity. The caller sees these
        # requirements as unverifiable, which is the truth.
        return {}, 0
    rulings = {
        str(item.get("id")): item
        for item in (payload.get("rulings") or []) if isinstance(item, dict)
    }
    return rulings, 1


# --- targeted repair -----------------------------------------------------------


def repair_brief(spec: PromptSpec, report: FidelityReport, limit: int = 6) -> str:
    """Instructions to fix exactly what missed, and change nothing else.

    Regenerating from scratch on a partial miss is how a fix for one requirement
    silently breaks two others. This names the failing ids, the measurement that
    settled them, and — critically — the ids that already hold and must survive.
    """
    failures = report.failures()[:limit]
    if not failures:
        return ""
    held = [item.id for item in report.verdicts if item.satisfied]
    lines = [
        "Your circuit compiled, but it does not satisfy the whole brief. These requirements "
        "were measured against the world you actually built and FAILED:",
    ]
    for verdict in failures:
        lines.append(f"  {verdict.id} {verdict.statement}")
        if verdict.quote:
            lines.append(f"      the user asked: \"{verdict.quote}\"")
        if verdict.evidence:
            lines.append(f"      measured: {verdict.evidence}")
        if verdict.claimed_at:
            lines.append(f"      you said you implemented this at: {verdict.claimed_at}")
    if held:
        lines.append("")
        lines.append(
            "These already hold and must still hold in your revision: " + ", ".join(held) + "."
        )
    lines.append("")
    lines.append(
        "Revise the plan to fix the failing requirements specifically. Change as little else "
        "as possible, and do not drop a requirement that currently passes in order to satisfy "
        "one that does not."
    )
    return "\n".join(lines)

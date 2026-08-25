"""Environment-generation arms, from a bare baseline to the verified harness.

Every arm here consumes the same brief, emits plans in the same grammar, and is
compiled by the same deterministic compiler. The only thing that varies is what
the arm is allowed to know about the scene its plan became:

- `oneshot`   the production single-proposal path. Retries only when the compiler
              rejects a plan outright, which is the honest baseline: it isolates
              the search loop as the single independent variable.
- `selfjudge` N independent proposals, ranked by the model reading its own plans.
              A compute control: it spends more model calls than the harness and
              still has no access to the simulator.
- `harness`   N proposals scored against a locally extracted specification using
              probe measurements, with measured residuals fed back between
              attempts, followed by a local dial solve over continuous parameters.

`selfjudge` exists so a win cannot be attributed to sampling more candidates, and
`harness` never sees held-out assertions, so a win cannot be attributed to
optimizing the grader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .generation_spec import GenerationSpec, SpecScore, extract_spec, score
from .models import SceneSpec, PlayabilityCertificate
from .probes import ProbeReport, measure
from .providers import ProviderError, anthropic_json, configured_model
from .racing import compile_certified_scene, design_racing_environment
from .track_grammar import NpcSpec, TrackPlan

ARMS = ("oneshot", "spec-oneshot", "selfjudge", "harness")
DEFAULT_CANDIDATES = 4
_RECOVERY_ATTEMPTS = 2
"""Extra proposals the search arm may spend only if nothing certified at all.

Stating a specification makes a creator author more ambitious geometry, which is
rejected by the compiler more often. Without this the search arm could spend its
whole budget on plans that never compiled and return nothing, which would make it
look worse than the baseline for a reason that has nothing to do with searching.
"""

Compile = Callable[
    [str, TrackPlan, int],
    tuple[SceneSpec, PlayabilityCertificate, list[str]],
]


@dataclass
class Candidate:
    """One plan, the scene it compiled into, and how well that scene scored."""

    origin: str
    plan: TrackPlan
    scene: SceneSpec | None = None
    certificate: PlayabilityCertificate | None = None
    probes: ProbeReport | None = None
    spec_score: SpecScore | None = None
    relaxations: list[str] = field(default_factory=list)
    failure: str | None = None

    @property
    def residual(self) -> float:
        """Search cost of this candidate; unusable candidates sort last."""
        if self.scene is None or self.spec_score is None:
            return float("inf")
        return self.spec_score.weighted_residual

    @property
    def satisfied(self) -> int:
        return self.spec_score.satisfied if self.spec_score else -1


@dataclass
class GenerationOutcome:
    arm: str
    prompt: str
    seed: int
    scene: SceneSpec | None = None
    certificate: PlayabilityCertificate | None = None
    probes: ProbeReport | None = None
    relaxations: list[str] = field(default_factory=list)
    search_score: SpecScore | None = None
    """Score against the generator-visible spec only; never the held-out grade."""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    candidates_evaluated: int = 0
    simulated_ticks: int = 0
    failure: str | None = None
    trace: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "seed": self.seed,
            "produced_scene": self.scene is not None,
            "scene_id": self.scene.id if self.scene else None,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "candidates_evaluated": self.candidates_evaluated,
            "simulated_ticks": self.simulated_ticks,
            "relaxations": self.relaxations, "failure": self.failure,
        }


def generate(
    prompt: str, seed: int, arm: str = "harness", provider: str = "auto",
    candidates: int = DEFAULT_CANDIDATES, spec: GenerationSpec | None = None,
    compile_scene: Compile | None = None,
) -> GenerationOutcome:
    """Run one generation arm on one brief and seed."""
    if arm not in ARMS:
        raise ValueError(f"Unknown generation arm: {arm}. Available: {', '.join(ARMS)}")
    compile_scene = compile_scene or compile_certified_scene
    outcome = GenerationOutcome(arm=arm, prompt=prompt, seed=seed)
    if arm == "oneshot":
        _oneshot(prompt, seed, provider, outcome, compile_scene)
    elif arm == "spec-oneshot":
        # Isolates the checklist from the measurement. The harness's prompt states
        # the specification, so without this arm a gain could be attributed to
        # searching when restating the brief as a list was doing the work.
        _oneshot(prompt, seed, provider, outcome, compile_scene,
                 guidance=_specification_guidance((spec or extract_spec(prompt)).visible()))
    elif arm == "selfjudge":
        _selfjudge(prompt, seed, provider, candidates, outcome, compile_scene)
    else:
        _harness(prompt, seed, provider, candidates, spec, outcome, compile_scene)
    return outcome


# --- arms ---------------------------------------------------------------------


def _oneshot(
    prompt: str, seed: int, provider: str, outcome: GenerationOutcome,
    compile_scene: Compile,
    guidance: str | None = None,
) -> None:
    """The production path: propose, compile, accept whatever certified."""
    feedback: str | None = None
    for attempt in range(3):
        candidate = _propose(prompt, seed, provider, outcome, feedback=feedback,
                             guidance=guidance, origin=f"proposal-{attempt + 1}",
                             compile_scene=compile_scene)
        outcome.candidates_evaluated += 1
        if candidate.scene is not None:
            _adopt(outcome, candidate)
            return
        feedback = candidate.failure
        outcome.trace.append(f"{candidate.origin}: {candidate.failure}")
        if provider == "offline":
            break
    outcome.failure = "no proposal could be certified: " + " | ".join(outcome.trace[:3])


def _selfjudge(
    prompt: str, seed: int, provider: str, candidates: int, outcome: GenerationOutcome,
    compile_scene: Compile,
) -> None:
    """Sample N proposals and let the model pick, with no simulator access."""
    pool: list[Candidate] = []
    for index in range(candidates):
        candidate = _propose(
            prompt, seed, provider, outcome, origin=f"proposal-{index + 1}",
            guidance=_variation_guidance(index, candidates),
            compile_scene=compile_scene,
        )
        outcome.candidates_evaluated += 1
        if candidate.scene is not None:
            pool.append(candidate)
        else:
            outcome.trace.append(f"{candidate.origin}: {candidate.failure}")
    if not pool:
        outcome.failure = "no proposal could be certified: " + " | ".join(outcome.trace[:3])
        return
    chosen = pool[0] if len(pool) == 1 else _judge(prompt, pool, outcome)
    _adopt(outcome, chosen)


def _harness(
    prompt: str, seed: int, provider: str, candidates: int,
    spec: GenerationSpec | None, outcome: GenerationOutcome, compile_scene: Compile,
) -> None:
    """Propose, measure, feed residuals back, then solve continuous dials.

    The specification is extracted locally from the brief, so the objective is
    reproducible and the generator cannot reinterpret what it was asked for.
    """
    visible = (spec or extract_spec(prompt)).visible()
    best: Candidate | None = None
    guidance = _specification_guidance(visible)
    # A rejected plan goes back through the same dedicated channel the baseline
    # uses, so stating the specification can never leave this arm worse at
    # recovering from a hard compiler rejection than a bare proposal would be.
    feedback: str | None = None
    attempts = candidates + _RECOVERY_ATTEMPTS
    for index in range(attempts):
        if index >= candidates and best is not None:
            break
        candidate = _propose(
            prompt, seed, provider, outcome, origin=f"proposal-{index + 1}",
            feedback=feedback, guidance=guidance, compile_scene=compile_scene,
        )
        outcome.candidates_evaluated += 1
        if candidate.scene is None:
            outcome.trace.append(f"{candidate.origin}: {candidate.failure}")
            feedback = candidate.failure
            continue
        feedback = None
        _evaluate(candidate, visible, outcome)
        outcome.trace.append(
            f"{candidate.origin}: {candidate.spec_score.summary() if candidate.spec_score else 'unscored'}"
        )
        if best is None or candidate.residual < best.residual:
            best = candidate
        if best.spec_score and best.spec_score.conjunction:
            break
        guidance = _specification_guidance(visible, extra=(
            [f"Your last plan compiled, but the simulator measured: {item}"
             for item in (candidate.spec_score.feedback(4) if candidate.spec_score else [])]
        ))
    if best is None:
        outcome.failure = "no proposal could be certified: " + " | ".join(outcome.trace[:3])
        return
    best = _solve_dials(best, visible, prompt, seed, outcome, compile_scene)
    _adopt(outcome, best)


# --- shared machinery ---------------------------------------------------------


def _propose(
    prompt: str, seed: int, provider: str, outcome: GenerationOutcome,
    origin: str, compile_scene: Compile, feedback: str | None = None,
    guidance: str | None = None,
) -> Candidate:
    """One creator call plus deterministic compilation of its plan.

    A provider-level rejection (an unparseable or out-of-grammar plan) is a failed
    candidate, not a failed arm. Letting it propagate would have penalized
    whichever arm happened to draw a malformed sample, which is noise in the
    provider rather than a difference between the arms under test.
    """
    try:
        design = design_racing_environment(prompt, provider, feedback=feedback, guidance=guidance)
    except ProviderError as error:
        outcome.model_calls += 1
        return Candidate(origin=origin, plan=TrackPlan(
            title="rejected proposal", rationale="The creator returned no usable plan.",
            corners=[{}, {}, {}],
        ), failure=f"creator rejected: {error}")
    if design.provider != "offline":
        outcome.model_calls += 1
        outcome.input_tokens += design.input_tokens
        outcome.output_tokens += design.output_tokens
        outcome.latency_ms += design.latency_ms
    return _compile_candidate(origin, design.plan, prompt, seed, compile_scene)


def _compile_candidate(
    origin: str, plan: TrackPlan, prompt: str, seed: int, compile_scene: Compile,
) -> Candidate:
    try:
        scene, certificate, notes = compile_scene(prompt, plan, seed)
    except ValueError as error:
        return Candidate(origin=origin, plan=plan, failure=str(error))
    return Candidate(
        origin=origin, plan=plan, scene=scene, certificate=certificate,
        relaxations=[*(scene.track_report.relaxations if scene.track_report else []), *notes],
    )


def _evaluate(candidate: Candidate, visible: GenerationSpec, outcome: GenerationOutcome) -> None:
    """Measure a compiled candidate and score it against the visible spec."""
    assert candidate.scene is not None
    candidate.probes = measure(candidate.scene)
    outcome.simulated_ticks += candidate.probes.simulated_ticks
    candidate.spec_score = score(visible, candidate.scene, candidate.probes)


def _adopt(outcome: GenerationOutcome, candidate: Candidate) -> None:
    outcome.scene = candidate.scene
    outcome.certificate = candidate.certificate
    outcome.probes = candidate.probes
    outcome.relaxations = candidate.relaxations
    outcome.search_score = candidate.spec_score
    outcome.trace.append(f"adopted {candidate.origin}")


def _variation_guidance(index: int, total: int) -> str | None:
    """Force genuinely different proposals rather than the same answer N times."""
    if index == 0:
        return None
    return (
        f"This is independent proposal {index + 1} of {total} for the same brief. Author a "
        "materially different circuit from the most conventional answer: vary the corner "
        "sequence, radii, straight lengths, and where the located features sit, while still "
        "satisfying everything the brief asks for."
    )


def _specification_guidance(visible: GenerationSpec, extra: list[str] | None = None) -> str | None:
    """State the checkable contract, and any measured residuals, to the creator."""
    lines = [assertion.describe() for assertion in visible.assertions]
    if not lines and not extra:
        return None
    block = "This brief will be checked against a specification:\n" + "\n".join(
        f"- {line}" for line in lines
    )
    if extra:
        block += "\n\n" + "\n".join(f"- {item}" for item in extra) + (
            "\n\nAuthor a plan that fixes these specific measured gaps without losing the "
            "constraints that already hold."
        )
    return block


def _judge(prompt: str, pool: list[Candidate], outcome: GenerationOutcome) -> Candidate:
    """Let the model rank its own proposals from the plans alone."""
    listing = "\n\n".join(
        f"Proposal {index + 1}:\n{candidate.plan.model_dump_json(exclude_none=True)}"
        for index, candidate in enumerate(pool)
    )
    try:
        payload, usage = anthropic_json(
            model=configured_model("ANTHROPIC_ENVIRONMENT_MODEL"),
            max_tokens=600,
            system=(
                "You review track plans for a deterministic top-down racing engine and choose "
                "the one that best satisfies a race brief. Judge only from the plans given."
            ),
            prompt=(
                f"Race brief: {prompt}\n\n{listing}\n\n"
                "Return the 1-based index of the proposal that best satisfies the brief."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "choice": {"type": "integer", "minimum": 1, "maximum": len(pool)},
                    "reason": {"type": "string", "maxLength": 400},
                },
                "required": ["choice", "reason"],
                "additionalProperties": False,
            },
        )
    except ProviderError as error:
        outcome.trace.append(f"self-judgement unavailable ({error}); kept the first proposal")
        return pool[0]
    outcome.model_calls += 1
    outcome.input_tokens += usage.input_tokens
    outcome.output_tokens += usage.output_tokens
    outcome.latency_ms += usage.latency_ms
    index = max(1, min(len(pool), int(payload.get("choice", 1)))) - 1
    outcome.trace.append(f"self-judged proposal {index + 1}: {str(payload.get('reason'))[:120]}")
    return pool[index]


# --- local dial solving -------------------------------------------------------


def _pinned_kinds(visible: GenerationSpec) -> set[str]:
    return {assertion.kind for assertion in visible.assertions}


def _dial_variants(plan: TrackPlan, visible: GenerationSpec) -> list[tuple[str, TrackPlan]]:
    """Candidate moves over continuous parameters the brief did not pin.

    A dial the brief names is never touched: satisfying an outcome target by
    quietly abandoning a stated constraint is not a solution, and the acceptance
    guard below would reject it anyway.
    """
    pinned = _pinned_kinds(visible)
    variants: list[tuple[str, TrackPlan]] = []
    if not pinned & {"grip_max", "grip_min", "grip_target"}:
        for grip in (.45, .6, .8, 1.0, 1.15):
            if abs(grip - plan.grip) > 1e-6:
                variants.append((f"grip={grip}", plan.model_copy(update={"grip": grip})))
    if not pinned & {"track_width_max", "track_width_min"}:
        for width in (112.0, 132.0, 152.0, 168.0):
            if abs(width - plan.track_width) > 1e-6:
                variants.append((f"width={width:g}", plan.model_copy(update={"track_width": width})))
    if plan.npcs and "npc_profiles" not in pinned:
        variants.append(("field=compressed", _repaced(plan, spread=.05)))
        variants.append(("field=spread", _repaced(plan, spread=.35)))
    elif plan.npcs:
        # Temperament is pinned, but pace within a temperament is not: it is a
        # separate axis, which is what makes finish-gap targets solvable at all.
        variants.append(("pace=compressed", _repaced(plan, spread=.05)))
        variants.append(("pace=spread", _repaced(plan, spread=.35)))
    return variants


def _repaced(plan: TrackPlan, spread: float) -> TrackPlan:
    """Rewrite the field's pace ladder without touching temperament."""
    count = len(plan.npcs)
    top = .95
    npcs: list[NpcSpec] = []
    for index, npc in enumerate(plan.npcs):
        step = 0.0 if count < 2 else spread * index / (count - 1)
        npcs.append(npc.model_copy(update={"pace": round(max(.35, min(1.05, top - step)), 3)}))
    return plan.model_copy(update={"npcs": npcs})


def _solve_dials(
    best: Candidate, visible: GenerationSpec, prompt: str, seed: int,
    outcome: GenerationOutcome, compile_scene: Compile,
) -> Candidate:
    """Coordinate-descend continuous parameters against measured residuals.

    This is the part a base model structurally cannot do: the target is a
    property of the simulation, so hitting it means measuring, not guessing. A
    move is accepted only if the total residual falls and no already-satisfied
    assertion regresses, which keeps the search from trading a stated constraint
    for a numeric one.
    """
    if best.spec_score is None or best.spec_score.conjunction:
        return best
    satisfied_before = {
        result.id for result in best.spec_score.results if result.satisfied
    }
    for label, variant_plan in _dial_variants(best.plan, visible):
        candidate = _compile_candidate(
            f"solve[{label}]", variant_plan, prompt, seed, compile_scene,
        )
        outcome.candidates_evaluated += 1
        if candidate.scene is None:
            continue
        _evaluate(candidate, visible, outcome)
        assert candidate.spec_score is not None
        regressed = {
            result.id for result in candidate.spec_score.results
            if not result.satisfied and result.id in satisfied_before
        }
        if regressed or candidate.residual >= best.residual:
            continue
        outcome.trace.append(f"solve {label}: {candidate.spec_score.summary()}")
        best = candidate
        satisfied_before = {
            result.id for result in candidate.spec_score.results if result.satisfied
        }
        if candidate.spec_score.conjunction:
            break
    return best

"""The prompt-faithful generation pipeline.

    prompt -> comprehension -> PromptSpec -> authoring -> plan -> compile
           -> engine validation -> fidelity verification -> targeted repair

Two verifications, deliberately not merged. The compiler and certificate answer
"is this a valid, completable circuit". The fidelity verifier answers "is this
the circuit that was asked for". A scene must pass both, and the pipeline can
tell you which one it failed, because the fix is different: a compiler rejection
needs different geometry, and a fidelity miss needs the geometry that was
actually requested.

Cost is kept close to the floor. Comprehension is one call and authoring is one
call. Verification is local measurement, except for requirements no evaluator can
settle, which are batched into a single judge call. The dial solve costs nothing
at all — it is local coordinate descent over recompiles. Only a real, measured
miss buys another authoring call, and it is a repair aimed at named ids rather
than a regeneration that would put the passing requirements back at risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .authoring import AuthoredPlan, author
from .comprehension import comprehend, needs_probes
from .dials import solve
from .fidelity import repair_brief, verify
from .models import PlayabilityCertificate, SceneSpec
from .probes import ProbeReport, measure
from .prompt_spec import FidelityReport, PromptSpec
from .providers import ProviderError
from .racing import compile_certified_scene
from .track_grammar import TrackPlan

MAX_COMPILE_ATTEMPTS = 3
"""Hard compiler rejections tolerated per authoring round."""

MAX_REPAIRS = 1
"""Fidelity repair rounds. One measured retry, not an open-ended search."""


@dataclass
class FaithfulResult:
    prompt: str
    spec: PromptSpec | None = None
    scene: SceneSpec | None = None
    certificate: PlayabilityCertificate | None = None
    probes: ProbeReport | None = None
    plan: AuthoredPlan | None = None
    report: FidelityReport | None = None
    relaxations: list[str] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    repairs: int = 0
    trace: list[str] = field(default_factory=list)
    failure: str | None = None

    @property
    def faithful(self) -> bool:
        return bool(self.report and self.report.faithful)

    def fidelity_lines(self) -> list[str]:
        return self.report.lines() if self.report else []


Step = Callable[[str, str], None]
Compile = Callable[..., tuple[SceneSpec, PlayabilityCertificate, list[str]]]
"""How a plan becomes a scene. Injected so the 3D path, which fits an elevation
surface and certifies twice, reuses this whole pipeline instead of forking it."""

Lookup = Callable[[str], list]
"""`check_kind -> precedents`. Injected rather than imported so this module never
depends on storage, and so a caller with no store simply passes nothing."""


def generate_faithful(
    prompt: str, seed: int, provider: str = "auto", on_step: Step | None = None,
    spec: PromptSpec | None = None, judge: bool = True, repairs: int = MAX_REPAIRS,
    compile_scene: Compile | None = None, precedent_lookup: Lookup | None = None,
    conversation: list[dict[str, Any]] | None = None,
    dimensions: str = "2d",
) -> FaithfulResult:
    """Build the world the brief asked for, and prove which parts of it landed."""
    compile_scene = compile_scene or compile_certified_scene
    result = FaithfulResult(prompt=prompt)

    def step(stage: str, message: str) -> None:
        result.trace.append(f"{stage}: {message}")
        if on_step:
            on_step(stage, message)

    # --- comprehension --------------------------------------------------------
    if spec is None:
        try:
            spec, usage = comprehend(prompt, provider, conversation, dimensions)
        except ProviderError as error:
            result.failure = f"could not read the brief: {error}"
            return result
        if usage is not None:
            result.model_calls += 1
            result.input_tokens += usage.input_tokens
            result.output_tokens += usage.output_tokens
            result.latency_ms += usage.latency_ms
    result.spec = spec
    step("comprehension", (
        f"Read {len(spec.requirements)} requirement(s) from the brief: "
        + ", ".join(f"{item.id} {item.statement}" for item in spec.requirements[:6])
    ) if spec.requirements else "No specific requirements were found in the brief.")
    if spec.unsupported:
        # Said before the circuit exists, so it cannot read as an excuse for
        # whatever came out of the compiler.
        step("unsupported", "The engine has no dial for: " + "; ".join(spec.unsupported))

    # --- authoring, with compiler retries -------------------------------------
    # Only ever the kinds this contract contains, and empty when nothing relevant has been
    # confirmed, which is the usual case. A precedent is a hint about what this person's
    # words have meant before, never a constraint.
    hints = ""
    if precedent_lookup is not None:
        from .precedents import guidance

        hints = guidance(spec, precedent_lookup)
        if hints:
            step("precedents", f"Recalled {hints.count(chr(10))} confirmed reading(s) of similar asks")
    authored, scene, certificate = _build(
        spec, seed, provider, result, step, compile_scene, precedents=hints,
        conversation=conversation,
    )
    if scene is None or authored is None:
        return result

    # --- fidelity -------------------------------------------------------------
    probes = _probe(spec, scene, result, step)
    report = verify(spec, scene, probes, authored.mapping, provider, judge=judge)
    result.model_calls += report.judge_calls
    step("fidelity", report.summary())

    # --- free local search ----------------------------------------------------
    # Runs before any repair, because a dial the harness can solve by measuring is
    # never worth a model call spent guessing at it.
    plan, scene, certificate, probes, report = _solve(
        spec, authored, scene, certificate, probes, report, seed, provider, judge, result, step,
        compile_scene,
    )
    authored.plan = plan

    # --- targeted repair ------------------------------------------------------
    for _ in range(max(0, repairs)):
        if report.faithful:
            break
        brief = repair_brief(spec, report)
        if not brief:
            break
        step("repair", "Re-authoring to fix " + ", ".join(
            item.id for item in report.failures()
        ))
        result.repairs += 1
        retry, retry_scene, retry_certificate = _build(
            spec, seed, provider, result, step, compile_scene, repair=brief, precedents=hints,
            conversation=conversation,
        )
        if retry_scene is None or retry is None:
            step("repair", "The repair attempt could not be compiled; keeping the first circuit.")
            break
        retry_probes = _probe(spec, retry_scene, result, step)
        retry_report = verify(spec, retry_scene, retry_probes, retry.mapping, provider, judge=judge)
        result.model_calls += retry_report.judge_calls
        step("fidelity", f"after repair: {retry_report.summary()}")
        retry_plan, retry_scene, retry_certificate, retry_probes, retry_report = _solve(
            spec, retry, retry_scene, retry_certificate, retry_probes, retry_report,
            seed, provider, judge, result, step, compile_scene,
        )
        retry.plan = retry_plan
        # A repair is adopted only if it is actually better. An unlucky retry that
        # fixes one requirement and breaks two must not replace a closer circuit.
        if (retry_report.satisfied, -retry_report.residual) <= (report.satisfied, -report.residual):
            step("repair", "The repair did not improve fidelity; keeping the first circuit.")
            break
        authored, scene, certificate = retry, retry_scene, retry_certificate
        probes, report = retry_probes, retry_report

    unmapped = authored.unmapped(spec)
    if unmapped:
        step("fidelity", "The creator never placed: " + ", ".join(unmapped))

    result.plan = authored
    result.scene = scene
    result.certificate = certificate
    result.probes = probes
    result.report = report
    return result


def _build(
    spec: PromptSpec, seed: int, provider: str, result: FaithfulResult, step: Step,
    compile_scene: Compile, repair: str | None = None, precedents: str = "",
    conversation: list[dict[str, Any]] | None = None,
) -> tuple[AuthoredPlan | None, SceneSpec | None, PlayabilityCertificate | None]:
    """Author and compile, retrying only on hard compiler rejections."""
    feedback: str | None = None
    for attempt in range(MAX_COMPILE_ATTEMPTS):
        step("creator", f"Authoring a circuit against the contract (attempt {attempt + 1})")
        try:
            authored = author(spec, provider, feedback=feedback, repair=repair,
                              precedents=precedents or None, conversation=conversation)
        except ProviderError as error:
            feedback = str(error)
            step("rejected", feedback[:220])
            continue
        if authored.provider != "offline":
            result.model_calls += 1
            result.input_tokens += authored.input_tokens
            result.output_tokens += authored.output_tokens
            result.latency_ms += authored.latency_ms
        try:
            scene, certificate, notes = compile_scene(spec.prompt, authored.plan, seed)
        except ValueError as error:
            feedback = str(error)
            step("rejected", feedback[:220])
            if authored.provider == "offline":
                break
            continue
        result.relaxations = [
            *(scene.track_report.relaxations if scene.track_report else []), *notes,
        ]
        step("compiler", f"Compiled and certified the circuit in {certificate.route_steps} ticks")
        return authored, scene, certificate
    result.failure = f"no plan could be compiled into a circuit: {feedback}"
    return None, None, None


def _solve(
    spec: PromptSpec, authored: AuthoredPlan, scene: SceneSpec,
    certificate: PlayabilityCertificate, probes: ProbeReport | None,
    report: FidelityReport, seed: int, provider: str, judge: bool,
    result: FaithfulResult, step: Step, compile_scene: Compile,
) -> tuple[TrackPlan, SceneSpec, PlayabilityCertificate, ProbeReport | None, FidelityReport]:
    """Try to close the remaining gaps by measurement rather than by asking again.

    The judge is deliberately off inside the search. A judged requirement cannot
    be moved by a dial anyway, and re-running it per variant would spend a model
    call on every compile — which is the exact cost this stage exists to avoid.
    Judged verdicts are carried over from the report that already settled them.
    """
    if report.faithful:
        return authored.plan, scene, certificate, probes, report

    carried = {item.id: item for item in report.verdicts if item.method == "judge"}

    def revalidate(candidate: SceneSpec, candidate_probes: ProbeReport | None) -> FidelityReport:
        fresh = verify(spec, candidate, candidate_probes, authored.mapping, provider, judge=False)
        fresh.verdicts = [
            carried.get(item.id, item) if item.method == "unverifiable" else item
            for item in fresh.verdicts
        ]
        return fresh

    outcome = solve(
        spec, authored.plan, scene, certificate, probes, report,
        revalidate, seed, compile_scene, authored.mapping,
    )
    if outcome.moves:
        step("dials", f"Solved locally in {outcome.variants_tried} compiles: "
                      + "; ".join(outcome.moves[-2:]))
        step("fidelity", f"after dial solve: {outcome.report.summary()}")
    return outcome.plan, outcome.scene, outcome.certificate, outcome.probes, outcome.report


def _probe(
    spec: PromptSpec, scene: SceneSpec, result: FaithfulResult, step: Step,
) -> ProbeReport | None:
    """Run rollouts only when a requirement actually needs one."""
    if not needs_probes(spec):
        return None
    step("probes", "Measuring the circuit with reference and unbraked drivers")
    return measure(scene)

"""Compile one brief with two generator arms and drive both circuits.

A specification score says whether a circuit satisfied what was asked for. It does
not say whether the circuit is any good to drive, and that judgement is the one a
metric cannot make for you. This module produces a matched pair — same brief, same
seed, same compiler — so the only difference is how the plan was chosen.

Blind mode exists because inspection is worth much less when you already know
which circuit came from which arm. The order is derived from the seed, so a blind
comparison is still reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .generation import GenerationOutcome, generate
from .generation_spec import GenerationSpec, SpecScore, extract_spec, score
from .models import SceneSpec
from .probes import ProbeReport, measure


@dataclass
class ArmScene:
    """One arm's circuit for a brief, with everything measured about it."""

    arm: str
    outcome: GenerationOutcome
    scene: SceneSpec | None
    probes: ProbeReport | None
    spec_score: SpecScore | None

    @property
    def label(self) -> str:
        return {
            "oneshot": "base model (single proposal)",
            "selfjudge": "base model (best of N, self-judged)",
            "harness": "harness (measured search)",
        }.get(self.arm, self.arm)


def build_pair(
    prompt: str, seed: int, arms: tuple[str, ...] = ("oneshot", "harness"),
    candidates: int = 4, provider: str = "auto", spec: GenerationSpec | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[ArmScene]:
    """Generate the same brief with each arm and grade all of them identically.

    Grading measures its own probes rather than reusing whatever an arm happened to
    measure, so an arm that never ran a probe is described in the same terms as one
    that ran twenty.
    """
    contract = spec or extract_spec(prompt)
    built: list[ArmScene] = []
    for arm in arms:
        # A search arm can spend a minute on model calls and rollouts. Without a
        # progress line the command looks hung rather than busy.
        if progress:
            progress(f"generating with {arm}...")
        outcome = generate(prompt, seed, arm=arm, provider=provider, candidates=candidates)
        if progress:
            progress(
                f"  {arm}: {outcome.model_calls} model call(s), "
                + ("certified a circuit" if outcome.scene is not None else f"failed ({outcome.failure})")
            )
        probes = measure(outcome.scene) if outcome.scene is not None else None
        built.append(ArmScene(
            arm=arm, outcome=outcome, scene=outcome.scene, probes=probes,
            spec_score=score(contract, outcome.scene, probes),
        ))
    return built


def presentation_order(pair: list[ArmScene], seed: int, blind: bool) -> list[tuple[str, ArmScene]]:
    """Name each circuit for the driver, hiding provenance when blind.

    The permutation comes from the seed rather than a random source so a blind
    session can be reproduced exactly, including which circuit was shown first.
    """
    if not blind:
        return [(entry.label, entry) for entry in pair]
    ordered = list(pair)
    if seed % 2:
        ordered.reverse()
    return [(f"Circuit {chr(ord('A') + index)}", entry) for index, entry in enumerate(ordered)]


def write_scenes(pair: list[ArmScene], directory: Path, prompt: str, seed: int) -> dict[str, str]:
    """Persist every compiled scene so the same pair can be replayed later."""
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for entry in pair:
        if entry.scene is None:
            continue
        path = directory / f"{entry.arm}.json"
        path.write_text(entry.scene.model_dump_json(indent=2), encoding="utf-8")
        written[entry.arm] = str(path)
    (directory / "pair.json").write_text(json.dumps({
        "prompt": prompt,
        "seed": seed,
        "arms": {
            entry.arm: {
                "scene_file": written.get(entry.arm),
                "label": entry.label,
                "produced_scene": entry.scene is not None,
                "failure": entry.outcome.failure,
                "model_calls": entry.outcome.model_calls,
                "input_tokens": entry.outcome.input_tokens,
                "output_tokens": entry.outcome.output_tokens,
                "simulated_ticks": entry.outcome.simulated_ticks,
                "relaxations": entry.outcome.relaxations,
                "assertions_satisfied": None if entry.spec_score is None else entry.spec_score.satisfied,
                "assertions_total": None if entry.spec_score is None else entry.spec_score.total,
                "misses": [] if entry.spec_score is None else [
                    result.message for result in entry.spec_score.failures()
                ],
            }
            for entry in pair
        },
    }, indent=2), encoding="utf-8")
    return written


def load_scene(path: str | Path) -> SceneSpec:
    return SceneSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def scorecard_lines(entry: ArmScene) -> list[str]:
    """What was asked for, what was measured, and what it cost."""
    lines: list[str] = []
    if entry.scene is None:
        return [f"  produced no certified circuit: {entry.outcome.failure}"]
    if entry.spec_score is not None:
        lines.append(
            f"  specification: {entry.spec_score.satisfied}/{entry.spec_score.total} assertions"
            + (" (all satisfied)" if entry.spec_score.conjunction else "")
        )
        for result in entry.spec_score.failures():
            lines.append(f"    missed: {result.message}")
    if entry.probes is not None:
        probes = entry.probes
        lines.append(
            f"  measured: reference race {probes.oracle_seconds:.1f}s"
            f" · braking {probes.brake_fraction * 100:.0f}% of the lap"
            f" · off-track {probes.off_track_ticks} ticks"
            f" · unbraked driver off-track {probes.naive_off_track_ticks} ticks"
        )
        lines.append(
            f"            {probes.order_changes} position change(s) in the field"
            + (f" · finish spread {probes.field_spread_seconds:.1f}s"
               if probes.field_spread_seconds is not None else " · field spread not measurable")
        )
    lines.append(
        f"  cost: {entry.outcome.model_calls} model call(s), "
        f"{entry.outcome.input_tokens + entry.outcome.output_tokens} tokens, "
        f"{entry.outcome.simulated_ticks} simulated tick(s)"
    )
    return lines

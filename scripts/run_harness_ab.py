"""Same prompt, two circuits: one built by the harness, one by a bare model call.

Run with `make ab`. Both arms land in the store as ordinary environments, so they can
be browsed and judged in the WebUI like anything else.

The comparison is set up so a win cannot come from the scoring:

- The brief is comprehended ONCE, and the resulting contract grades both arms. The
  naive arm never sees it — it is measured by it, not guided by it.
- Both arms get the same engine, the same seed, and the same authoring grammar,
  including the visual plan. Withholding colour from the baseline would have rigged
  every appearance requirement in the harness's favour.
- The naive arm is one model call on the raw prompt, then compile. No contract, no
  requirement ids, no measurement, no repair, no dial solving, no precedents. It
  keeps only the compiler's own playability check, because an uncompilable scene
  cannot be looked at and the question here is fidelity, not whether the engine works.

Circuits are labelled `P1-A` / `P1-B` with the arm assignment shuffled per pair, so
the WebUI list does not tell you which is which before you have looked. The key is
printed at the end.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from harness.authoring import _VISUAL_SCHEMA  # noqa: E402
from harness.cli import _load_local_env  # noqa: E402
from harness.comprehension import comprehend  # noqa: E402
from harness.faithful import generate_faithful  # noqa: E402
from harness.fidelity import verify  # noqa: E402
from harness.models import EnvironmentAddress, EnvironmentRecord  # noqa: E402
from harness.probes import measure  # noqa: E402
from harness.providers import ProviderError, anthropic_json, configured_model  # noqa: E402
from harness.racing import (  # noqa: E402
    RACING_CREATOR_SYSTEM, _coerce_plan_payload, compile_certified_scene,
    validate_racing_scene,
)
from harness.service import HarnessService, timestamp  # noqa: E402
from harness.track_grammar import TrackPlan, track_plan_schema  # noqa: E402

PROMPTS = [
    "a clockwise circuit with a 90 degree right-hander in the top right and a 150 degree "
    "hairpin in the bottom left, black opponent cars, a slate grey road, and no red and "
    "white kerbs",

    "three aggressive npcs spread around the lap rather than lined up on a grid, a 45 degree "
    "kink in the top left, a wide corridor, a purple track on sand coloured ground, two laps",

    "an icy circuit, four laps, one blocker and one backmarker, a 120 degree corner in the "
    "bottom right, white barriers and a pale blue road",

    "a narrow twisty street circuit with seven corners and no barriers, a dark charcoal road "
    "with neon green edge lines, one racer for company",

    "a fast flowing counterclockwise oval with a long back straight for slipstreaming, a river "
    "running across the top of the map, a red player car and blue opponents",

    "a gravel rally stage with a 60 degree corner in the top centre and a 160 degree switchback "
    "on the right side, two backmarkers, a tan road on olive ground, no kerbs",
]


def naive_plan(prompt: str) -> tuple[TrackPlan, dict]:
    """One model call on the raw prompt. This is the whole baseline.

    Deliberately the same creator system prompt and the same grammar the harness arm
    authors into, visual plan included. The only thing withheld is the harness: no
    requirement ids, nothing measured, nothing fed back.
    """
    schema = track_plan_schema()
    schema = {**schema, "properties": {**schema["properties"], "visual": _VISUAL_SCHEMA}}
    schema["required"] = [*schema["required"], "visual"]
    payload, usage = anthropic_json(
        model=configured_model("ANTHROPIC_ENVIRONMENT_MODEL"),
        max_tokens=3_000,
        system=RACING_CREATOR_SYSTEM,
        prompt=(
            f"Race brief: {prompt}\n\n"
            "Return the track plan. Author 3 to 10 corners, 0 to 6 barriers, and 0 to 5 "
            "opponents, and set the visual block for any appearance the brief describes."
        ),
        json_schema=schema,
    )
    return TrackPlan.model_validate(_coerce_plan_payload(payload)), {
        "input": usage.input_tokens, "output": usage.output_tokens, "calls": 1,
    }


def build_naive(prompt: str, seed: int):
    """Author once, then compile — retrying only when the geometry cannot be built."""
    feedback: TrackPlan | None = None
    last_error = ""
    for _attempt in range(3):
        try:
            plan, usage = naive_plan(prompt)
        except ProviderError as error:
            last_error = str(error)
            continue
        try:
            scene, certificate, _notes = compile_certified_scene(prompt, plan, seed)
        except ValueError as error:
            last_error = str(error)
            continue
        return scene, certificate, plan, usage
    raise RuntimeError(f"the naive arm produced nothing: {last_error[:200]}")


def store_environment(service: HarnessService, scene, certificate, spec, report,
                      label: str, arm: str, experiment: int, variant: int) -> EnvironmentRecord:
    scene = scene.model_copy(update={
        "id": f"{scene.id}-{uuid.uuid4().hex[:6]}", "name": label,
    })
    record = EnvironmentRecord(
        id=scene.id, scene=scene, created_at=timestamp(),
        validation=validate_racing_scene(scene), baseline_solved=True,
        origin=f"ab:{arm}", generator_provider="anthropic",
        generator_model=configured_model("ANTHROPIC_ENVIRONMENT_MODEL"),
        playability_certificate=certificate, prompt_spec=spec, fidelity=report,
        study_name="Harness vs naive generation",
        address=EnvironmentAddress(experiment=experiment, environment=1, variant=variant),
    )
    service.store.save_environment(record)
    return record


def run_pair(index: int, prompt: str, seed: int) -> dict:
    """One prompt through both arms, graded by one shared contract."""
    service = HarnessService()
    spec, _usage = comprehend(prompt)

    outcome = generate_faithful(prompt, seed, spec=spec, precedent_lookup=service.store.precedents_for)
    if outcome.scene is None:
        raise RuntimeError(f"the harness arm produced nothing: {outcome.failure}")

    naive_scene, naive_certificate, _plan, naive_usage = build_naive(prompt, seed)
    # The same grader on both, with the same evidence available to it.
    naive_report = verify(spec, naive_scene, measure(naive_scene), None)

    # Shuffled so the list order does not give the answer away before you have looked.
    # Seeded per pair, not per run: one seed for every pair puts the same arm in slot A
    # every time, which is not a blind at all.
    arms = [("harness", outcome.scene, outcome.certificate, outcome.report),
            ("naive", naive_scene, naive_certificate, naive_report)]
    random.Random(seed * 1_000 + index).shuffle(arms)
    key = {}
    for variant, (arm, scene, certificate, report) in enumerate(arms, start=1):
        suffix = "AB"[variant - 1]
        store_environment(
            service, scene, certificate, spec, report,
            label=f"P{index}-{suffix} · {prompt[:52]}", arm=arm,
            experiment=index, variant=variant,
        )
        key[suffix] = arm
    return {
        "index": index, "prompt": prompt, "key": key,
        "requirements": len(spec.requirements),
        "harness": outcome.report.satisfied if outcome.report else 0,
        "naive": naive_report.satisfied,
        "total": len(spec.requirements),
        "harness_calls": outcome.model_calls,
        "naive_calls": naive_usage["calls"],
        "harness_misses": [item.id for item in (outcome.report.failures() if outcome.report else [])],
        "naive_misses": [item.id for item in naive_report.failures()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=len(PROMPTS))
    args = parser.parse_args()
    _load_local_env()

    prompts = PROMPTS[:args.limit]
    print(f"running {len(prompts)} paired generations, seed {args.seed}\n")
    results: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(run_pair, index, prompt, args.seed): index
            for index, prompt in enumerate(prompts, start=1)
        }
        for future in futures.as_completed(pending):
            index = pending[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - one bad pair must not lose the rest
                print(f"  P{index}: FAILED — {str(error)[:150]}")
                continue
            results.append(result)
            print(f"  P{result['index']}: harness {result['harness']}/{result['total']}  "
                  f"naive {result['naive']}/{result['total']}")

    if not results:
        print("\nno pairs completed")
        return 1

    results.sort(key=lambda item: item["index"])
    harness_total = sum(item["harness"] for item in results)
    naive_total = sum(item["naive"] for item in results)
    requirement_total = sum(item["total"] for item in results)
    harness_clean = sum(1 for item in results if not item["harness_misses"])
    naive_clean = sum(1 for item in results if not item["naive_misses"])

    print("\n" + "=" * 78)
    print(f"{'pair':<6}{'harness':<10}{'naive':<10}{'A is':<10}{'B is':<10}")
    for item in results:
        print(f"P{item['index']:<5}{item['harness']}/{item['total']:<8}"
              f"{item['naive']}/{item['total']:<8}{item['key']['A']:<10}{item['key']['B']:<10}")
    print("=" * 78)
    print(f"requirements met : harness {harness_total}/{requirement_total} "
          f"({harness_total / max(1, requirement_total):.0%})   "
          f"naive {naive_total}/{requirement_total} "
          f"({naive_total / max(1, requirement_total):.0%})")
    print(f"fully faithful   : harness {harness_clean}/{len(results)}   "
          f"naive {naive_clean}/{len(results)}")
    print(f"model calls      : harness {sum(i['harness_calls'] for i in results)}   "
          f"naive {sum(i['naive_calls'] for i in results)}")
    print("\nBoth arms are in the WebUI under Environments, labelled P<n>-A and P<n>-B.")
    print("The fidelity panel under each one lists what it was asked for and what landed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

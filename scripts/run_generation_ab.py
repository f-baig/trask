#!/usr/bin/env python3
"""Compare environment-generation arms against a pre-registered specification suite.

Every arm authors plans in the same grammar and is compiled by the same compiler,
so the only independent variable is what the arm knows about the scene its plan
became. Grading is done once, here, by code neither arm can influence: the final
scene is probed and scored against the hand-authored spec in `generation_suite`.

    python scripts/run_generation_ab.py --arms oneshot selfjudge harness --seeds 17 43 91

`--arms harness-given-spec` adds a diagnostic arm that receives the visible half
of the hand-authored spec instead of the locally extracted one, which separates
"the extractor misread the brief" from "the search could not satisfy it".
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from harness.generation import DEFAULT_CANDIDATES, generate
from harness.generation_spec import Assertion, GenerationSpec, score
from harness.providers import configured_model
from harness.probes import measure

from generation_suite import CLASSES, SUITE
from run_live_racing_benchmark import load_dotenv

ARM_CHOICES = ("oneshot", "spec-oneshot", "selfjudge", "harness", "harness-given-spec")


def grade(spec: GenerationSpec, outcome) -> dict:
    """Score a finished scene against the full pre-registered specification.

    The grader measures its own probes rather than reusing whatever the arm
    happened to measure, so an arm that never ran a probe is graded identically
    to one that ran twenty.
    """
    probes = measure(outcome.scene) if outcome.scene is not None else None
    full = score(spec, outcome.scene, probes)
    visible = score(spec.visible(), outcome.scene, probes)
    held_out = score(spec.held_out(), outcome.scene, probes)
    return {
        "conjunction": full.conjunction,
        "visible_conjunction": visible.conjunction,
        "held_out_conjunction": held_out.conjunction,
        "visible_satisfied": visible.satisfied,
        "visible_total": visible.total,
        "held_out_satisfied": held_out.satisfied,
        "held_out_total": held_out.total,
        "satisfaction_rate": visible.satisfaction_rate,
        "residual": visible.weighted_residual,
        "misses": [result.message for result in full.failures()],
        "grading_ticks": probes.simulated_ticks if probes else 0,
        "probe_summary": None if probes is None else {
            "oracle_seconds": probes.oracle_seconds,
            "oracle_finished": probes.oracle_finished,
            "brake_fraction": probes.brake_fraction,
            "off_track_ticks": probes.off_track_ticks,
            "naive_off_track_ticks": probes.naive_off_track_ticks,
            "order_changes": probes.order_changes,
            "field_spread_seconds": probes.field_spread_seconds,
        },
        "per_assertion": [
            {"id": result.id, "kind": result.kind, "held_out": result.held_out,
             "satisfied": result.satisfied, "achieved": result.achieved}
            for result in full.results
        ],
    }


def aggregate(rows: list[dict]) -> dict:
    """Collapse rows into the headline numbers, by arm and by case class."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    def summarize(items: list[dict]) -> dict:
        if not items:
            return {}
        return {
            "cases": len(items),
            "produced_scene_rate": round(
                sum(1 for item in items if item["produced_scene"]) / len(items), 3),
            "conjunction_rate": round(
                sum(1 for item in items if item["grade"]["conjunction"]) / len(items), 3),
            "visible_conjunction_rate": round(
                sum(1 for item in items if item["grade"]["visible_conjunction"]) / len(items), 3),
            "held_out_conjunction_rate": round(
                sum(1 for item in items if item["grade"]["held_out_conjunction"]) / len(items), 3),
            "mean_satisfaction_rate": round(
                statistics.mean(item["grade"]["satisfaction_rate"] for item in items), 4),
            "mean_residual": round(
                statistics.mean(item["grade"]["residual"] for item in items), 4),
            "mean_model_calls": round(
                statistics.mean(item["model_calls"] for item in items), 2),
            "total_input_tokens": sum(item["input_tokens"] for item in items),
            "total_output_tokens": sum(item["output_tokens"] for item in items),
            "mean_wall_seconds": round(
                statistics.mean(item["wall_seconds"] for item in items), 2),
            "total_search_ticks": sum(item["simulated_ticks"] for item in items),
        }

    return {
        "by_arm": {arm: summarize(items) for arm, items in by_arm.items()},
        "by_arm_and_class": {
            arm: {
                case_class: summarize([item for item in items if item["case_class"] == case_class])
                for case_class in CLASSES
            }
            for arm, items in by_arm.items()
        },
        "paired_vs_oneshot": paired_comparison(rows),
    }


def paired_comparison(rows: list[dict]) -> dict:
    """Sign test against the baseline on matched (case, seed) trials."""
    indexed: dict[tuple[str, str, int], dict] = {
        (row["arm"], row["case_id"], row["seed"]): row for row in rows
    }
    baseline_keys = [key for key in indexed if key[0] == "oneshot"]
    results: dict[str, dict] = {}
    for arm in {key[0] for key in indexed} - {"oneshot"}:
        wins = losses = ties = 0
        for _, case_id, seed in baseline_keys:
            treatment = indexed.get((arm, case_id, seed))
            baseline = indexed[("oneshot", case_id, seed)]
            if treatment is None:
                continue
            left = treatment["grade"]["visible_satisfied"]
            right = baseline["grade"]["visible_satisfied"]
            if left > right:
                wins += 1
            elif left < right:
                losses += 1
            else:
                ties += 1
        results[arm] = {
            "trials": wins + losses + ties, "wins": wins, "losses": losses, "ties": ties,
            "decisive": wins + losses,
            "win_rate_of_decisive": round(wins / (wins + losses), 3) if wins + losses else None,
        }
    return results


def markdown_report(summary: dict) -> str:
    lines = [
        "# Environment generation A/B", "",
        f"Created: {summary['created_at']}",
        f"Environment model: `{summary['environment_model']}`",
        f"Seeds: {summary['seeds']} · candidates per search arm: {summary['candidates']}", "",
        "Grading is a pre-registered specification suite. `visible` assertions may be",
        "optimized by a search arm; `held-out` assertions are shown to no arm and are the",
        "generalization check.", "",
        "## Headline", "",
        "| arm | all-constraint conjunction | visible | held-out | mean satisfied | model calls | wall s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, stats in summary["aggregate"]["by_arm"].items():
        lines.append(
            f"| {arm} | {stats['conjunction_rate']:.3f} | {stats['visible_conjunction_rate']:.3f} "
            f"| {stats['held_out_conjunction_rate']:.3f} | {stats['mean_satisfaction_rate']:.3f} "
            f"| {stats['mean_model_calls']:.2f} | {stats['mean_wall_seconds']:.1f} |"
        )
    lines += ["", "## By case class", "", "| arm | " + " | ".join(CLASSES) + " |",
              "| --- | " + " | ".join("---" for _ in CLASSES) + " |"]
    for arm, classes in summary["aggregate"]["by_arm_and_class"].items():
        cells = [
            f"{classes[case_class]['mean_satisfaction_rate']:.3f}" if classes.get(case_class) else "—"
            for case_class in CLASSES
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "Cells are the mean fraction of visible assertions satisfied.", "",
              "## Paired sign test against `oneshot`", "",
              "| arm | trials | wins | losses | ties | win rate of decisive |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm, stats in summary["aggregate"]["paired_vs_oneshot"].items():
        rate = stats["win_rate_of_decisive"]
        lines.append(
            f"| {arm} | {stats['trials']} | {stats['wins']} | {stats['losses']} "
            f"| {stats['ties']} | {'—' if rate is None else f'{rate:.3f}'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["oneshot", "selfjudge", "harness"],
                        choices=ARM_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 43, 91])
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Restrict to these case ids.")
    parser.add_argument("--provider", default="auto", choices=["auto", "offline", "anthropic", "openai"])
    parser.add_argument("--dimensions", default="2d", choices=["2d", "3d"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    import os
    model = configured_model("ANTHROPIC_ENVIRONMENT_MODEL")

    cases = [case for case in SUITE if not args.cases or case.case_id in args.cases]
    if args.dimensions == "3d":
        cases = [_three_dimensional(case) for case in cases]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or f".harness-data/generation_ab/{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # A long live run is checkpointed after every trial and resumable, so an
    # interruption costs the trial in flight rather than the whole matrix.
    summary_path = output_dir / "summary.json"
    rows: list[dict] = []
    completed: set[tuple[str, int, str]] = set()
    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8")).get("rows", [])
        completed = {(row["case_id"], row["seed"], row["arm"]) for row in rows}
        print(f"Resuming: {len(completed)} trials already recorded in {summary_path}", flush=True)

    def checkpoint() -> None:
        summary_path.write_text(json.dumps(build_summary(), indent=2), encoding="utf-8")

    def build_summary() -> dict:
        return {
            "created_at": datetime.now(UTC).isoformat(),
            "environment_model": model,
            "provider": args.provider,
            "dimensions": args.dimensions,
            "seeds": args.seeds,
            "candidates": args.candidates,
            "arms": args.arms,
            "cases": [
                {"case_id": case.case_id, "case_class": case.case_class, "prompt": case.prompt,
                 "assertions": [assertion.model_dump() for assertion in case.assertions]}
                for case in cases
            ],
            "rows": rows,
            "aggregate": aggregate(rows),
        }

    total = len(cases) * len(args.seeds) * len(args.arms)
    done = 0
    for case in cases:
        for seed in args.seeds:
            for arm in args.arms:
                done += 1
                if (case.case_id, seed, arm) in completed:
                    print(f"[{done}/{total}] {case.case_id} seed={seed} arm={arm} · already recorded",
                          flush=True)
                    continue
                print(f"[{done}/{total}] {case.case_id} seed={seed} arm={arm}", flush=True)
                started = time.monotonic()
                try:
                    compile_scene = None
                    if args.dimensions == "3d":
                        from harness.racing3d import compile_racing_3d_scene
                        from harness.track3d import parse_elevation_prompt

                        elevation = parse_elevation_prompt(case.prompt)

                        def compile_scene(scene_prompt, plan, scene_seed):
                            return compile_racing_3d_scene(
                                scene_prompt, plan, elevation, scene_seed,
                            )
                    outcome = generate(
                        case.prompt, seed,
                        arm="harness" if arm == "harness-given-spec" else arm,
                        provider=args.provider, candidates=args.candidates,
                        spec=case if arm == "harness-given-spec" else None,
                        compile_scene=compile_scene,
                    )
                except Exception as error:  # a crashed arm is a generation failure, not a gap
                    rows.append({
                        "arm": arm, "case_id": case.case_id, "case_class": case.case_class,
                        "seed": seed, "produced_scene": False, "error": str(error),
                        "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                        "simulated_ticks": 0, "wall_seconds": round(time.monotonic() - started, 2),
                        "grade": grade(case, _Empty()),
                    })
                    print(f"    error: {error}", flush=True)
                    checkpoint()
                    continue
                graded = grade(case, outcome)
                # Persist the circuit itself, not just its score. A number cannot
                # say whether a circuit is good to drive, so every graded scene
                # stays openable with `harness play --scene`.
                scene_file = None
                if outcome.scene is not None:
                    scene_dir = output_dir / "scenes"
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    scene_path = scene_dir / f"{case.case_id}-seed{seed}-{arm}.json"
                    scene_path.write_text(outcome.scene.model_dump_json(indent=2), encoding="utf-8")
                    scene_file = str(scene_path)
                rows.append({
                    "scene_file": scene_file,
                    "arm": arm, "case_id": case.case_id, "case_class": case.case_class,
                    "seed": seed, "produced_scene": outcome.scene is not None,
                    "scene_id": outcome.scene.id if outcome.scene else None,
                    "model_calls": outcome.model_calls,
                    "input_tokens": outcome.input_tokens,
                    "output_tokens": outcome.output_tokens,
                    "candidates_evaluated": outcome.candidates_evaluated,
                    "simulated_ticks": outcome.simulated_ticks,
                    "wall_seconds": round(time.monotonic() - started, 2),
                    "relaxations": outcome.relaxations,
                    "failure": outcome.failure,
                    "trace": outcome.trace,
                    "grade": graded,
                })
                checkpoint()
                print(
                    f"    visible {graded['visible_satisfied']}/{graded['visible_total']}"
                    f" · held-out {graded['held_out_satisfied']}/{graded['held_out_total']}"
                    f" · {outcome.model_calls} model calls"
                    + ("" if graded["conjunction"] else f" · missed: {graded['misses'][:2]}"),
                    flush=True,
                )

    summary = build_summary()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    print("\n" + markdown_report(summary), flush=True)
    print(f"Wrote {output_dir / 'REPORT.md'}", flush=True)


class _Empty:
    """Stands in for a crashed arm so it is graded as producing nothing."""

    scene = None


def _three_dimensional(case: GenerationSpec) -> GenerationSpec:
    """Lift a preregistered plan-view brief onto one matched rolling 3D contract."""
    prompt = case.prompt.rstrip(". ") + ". Build it as a 3D circuit over gentle rolling elevation."
    vertical = [
        Assertion(
            id=f"{case.case_id}-3d-profile", kind="elevation_profile", target="rolling",
            label="a rolling elevation profile",
        ),
        Assertion(
            id=f"{case.case_id}-3d-relief", kind="elevation_amplitude_min", target=2.0,
            label="at least 2 m of fitted vertical relief",
        ),
        Assertion(
            id=f"{case.case_id}-3d-crests", kind="elevation_hill_count", target=2,
            label="two elevation crests",
        ),
    ]
    return case.model_copy(update={"prompt": prompt, "assertions": [*case.assertions, *vertical]})


if __name__ == "__main__":
    main()

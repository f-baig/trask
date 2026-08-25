"""Render the corrected environment-generation A/B study for the write-up."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from run_generation_ab import aggregate  # noqa: E402


DEFAULT_STUDY = Path(".harness-data/generation_ab/full-run-3/summary.json")
DEFAULT_CORRECTION = Path(".harness-data/generation_ab/conj-clay-postfix/summary.json")
ARMS = ("oneshot", "selfjudge", "harness")
LABELS = {
    "oneshot": "Non-harnessed one-shot",
    "selfjudge": "Non-harnessed self-judge",
    "harness": "Harnessed generation",
}
COLORS = {
    "oneshot": "#8B9298",
    "selfjudge": "#D28B36",
    "harness": "#2F76BE",
}


def corrected_summary(study_path: Path, correction_path: Path) -> dict:
    study = json.loads(study_path.read_text(encoding="utf-8"))
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    replacements = {
        (row["case_id"], row["seed"], row["arm"]): row
        for row in correction["rows"]
    }
    rows = [
        replacements.get((row["case_id"], row["seed"], row["arm"]), row)
        for row in study["rows"]
    ]
    merged = {
        **study,
        "created_at": datetime.now(UTC).isoformat(),
        "study_created_at": study.get("created_at"),
        "source_summary": str(study_path.resolve()),
        "correction_summary": str(correction_path.resolve()),
        "correction": (
            "Replaced the three harness/conj-clay trials with the post-fix rerun "
            "recorded by the original study. All baseline and self-judge rows are unchanged."
        ),
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    merged["statistics"] = paired_conjunction(rows, "harness", "oneshot")
    return merged


def paired_conjunction(rows: list[dict], treatment: str, baseline: str) -> dict:
    indexed = {
        (row["arm"], row["case_id"], row["seed"]): bool(row["grade"]["conjunction"])
        for row in rows
    }
    wins = losses = ties = 0
    for arm, case_id, seed in indexed:
        if arm != baseline or (treatment, case_id, seed) not in indexed:
            continue
        left = indexed[(treatment, case_id, seed)]
        right = indexed[(baseline, case_id, seed)]
        if left and not right:
            wins += 1
        elif right and not left:
            losses += 1
        else:
            ties += 1
    decisive = wins + losses
    tail = sum(math.comb(decisive, value) for value in range(0, min(wins, losses) + 1))
    exact_p = min(1.0, 2 * tail / (2 ** decisive)) if decisive else 1.0
    return {
        "test": "two-sided exact McNemar/binomial sign test on matched conjunction outcomes",
        "treatment": treatment, "baseline": baseline,
        "trials": decisive + ties, "wins": wins, "losses": losses,
        "ties": ties, "decisive": decisive, "p_value": round(exact_p, 6),
    }


def style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#c9c9c9")
    axis.tick_params(colors="#4a4a4a", labelsize=9)
    axis.grid(axis=grid_axis, color="#e7e7e7", linewidth=.8, zorder=0)


def quality_chart(summary: dict, path: Path, dimension_label: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    aggregate_data = summary["aggregate"]
    figure, axis = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    width = .24
    classes = ("lexical", "conjunction", "outcome", "numeric")
    x = np.arange(len(classes))
    for index, arm in enumerate(ARMS):
        values = [
            aggregate_data["by_arm_and_class"][arm][case_class]["mean_satisfaction_rate"] * 100
            for case_class in classes
        ]
        bars = axis.bar(
            x + (index - 1) * width, values, width,
            label=LABELS[arm], color=COLORS[arm], zorder=2,
        )
        axis.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=2, fontsize=9)
    axis.set_xticks(x, [item.title() for item in classes])
    axis.set_ylim(0, 112)
    axis.set_ylabel("mean requirements satisfied (%)")
    stats = summary["statistics"]
    axis.set_title(
        f"Matched all-requirement outcome: {stats['wins']} wins / {stats['losses']} losses, p={stats['p_value']:.4f}",
        loc="left", fontsize=11.5, fontweight="bold",
    )
    style_axis(axis)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    figure.suptitle(
        f"{dimension_label + ' ' if dimension_label else ''}environment-generation fidelity by brief type",
        fontsize=18, fontweight="bold", color="#171717",
    )
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def cost_chart(summary: dict, path: Path, dimension_label: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    by_arm = summary["aggregate"]["by_arm"]
    panels = (
        ("mean_model_calls", "Model calls per environment", "calls"),
        ("tokens", "Tokens per environment", "tokens"),
        ("mean_wall_seconds", "Wall time per environment", "seconds"),
        ("ticks", "Deterministic search simulation", "ticks per environment"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.7), layout="constrained")
    labels = [LABELS[arm] for arm in ARMS]
    colors = [COLORS[arm] for arm in ARMS]
    for axis, (metric, title, unit) in zip(axes.flat, panels):
        if metric == "tokens":
            values = [
                (by_arm[arm]["total_input_tokens"] + by_arm[arm]["total_output_tokens"])
                / by_arm[arm]["cases"] for arm in ARMS
            ]
        elif metric == "ticks":
            values = [by_arm[arm]["total_search_ticks"] / by_arm[arm]["cases"] for arm in ARMS]
        else:
            values = [by_arm[arm][metric] for arm in ARMS]
        bars = axis.barh(labels, values, color=colors, height=.62, zorder=2)
        maximum = max(values) or 1
        axis.set_xlim(0, maximum * 1.22)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold")
        axis.set_xlabel(unit)
        style_axis(axis, grid_axis="x")
        if metric in {"tokens", "ticks"}:
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
        labels_out = [
            f"{value:,.0f}" if metric in {"tokens", "ticks"} else f"{value:.2f}"
            for value in values
        ]
        axis.bar_label(bars, labels=labels_out, padding=4, fontsize=9)
    figure.suptitle(
        f"{dimension_label + ' ' if dimension_label else ''}environment-generation cost",
        fontsize=18, fontweight="bold", color="#171717",
    )
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def concise(summary: dict) -> dict:
    return {
        "created_at": summary["created_at"],
        "study_created_at": summary.get("study_created_at"),
        "environment_model": summary["environment_model"],
        "provider": summary["provider"],
        "seeds": summary["seeds"],
        "candidates": summary["candidates"],
        "trial_count": len(summary["rows"]),
        "brief_count": len(summary["cases"]),
        "arms": summary["arms"],
        "source_summary": summary["source_summary"],
        "correction_summary": summary["correction_summary"],
        "correction": summary["correction"],
        "aggregate": summary["aggregate"],
        "statistics": summary["statistics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--correction", type=Path, default=DEFAULT_CORRECTION)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/environment-generation-comparison/writeup"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = corrected_summary(args.study, args.correction)
    quality = args.output_dir / "environment_generation_quality.png"
    cost = args.output_dir / "environment_generation_cost.png"
    quality_chart(summary, quality)
    cost_chart(summary, cost)
    payload = concise(summary)
    payload["quality_chart"] = str(quality.resolve())
    payload["cost_chart"] = str(cost.resolve())
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path.resolve()),
        "quality_chart": str(quality.resolve()),
        "cost_chart": str(cost.resolve()),
        "headline": payload["aggregate"]["by_arm"],
        "statistics": payload["statistics"],
    }, indent=2))


if __name__ == "__main__":
    main()

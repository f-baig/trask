"""Render write-up figures from the completed matched 3D player evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from harness.collision import track_edge_points  # noqa: E402
from harness.rendering import ReplayBundle  # noqa: E402


DEFAULT_STUDY = Path(".harness-data/3d_pipeline_ab/live-7-fixed-full-matrix/summary.json")


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#c9c9c9")
    axis.tick_params(colors="#4a4a4a", labelsize=9)
    axis.grid(axis="x", color="#e6e6e6", linewidth=.8, zorder=0)


def metrics_chart(results: list[dict], path: Path, control_hz: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    labels = [result["label"] for result in results]
    colors = ["#276FBF" if result["completed"] else "#C84C4C" for result in results]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    panels = (
        ("total_tokens", "Total token usage", "tokens"),
        ("model_calls", "Model calls", "calls"),
        ("steps", "Simulated time to outcome", "simulated seconds"),
        ("evaluation_wall_time_ms", "Evaluation wall time", "minutes"),
    )
    for axis, (field, title, unit) in zip(axes.flat, panels):
        raw = [result[field] for result in results]
        values = (
            [value / control_hz for value in raw] if field == "steps" else
            [value / 60_000 for value in raw] if field == "evaluation_wall_time_ms" else raw
        )
        bars = axis.barh(labels, values, color=colors, height=.64, zorder=2)
        maximum = max(values) if values else 1
        axis.set_xlim(0, maximum * 1.25 if maximum else 1)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#202020")
        axis.set_xlabel(unit, color="#4a4a4a")
        style_axis(axis)
        if field == "total_tokens":
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
        for bar, value, result in zip(bars, values, results):
            shown = f"{result[field]:,}" if field in {"total_tokens", "model_calls"} else f"{value:.1f}"
            status = "completed" if result["completed"] else "not completed"
            axis.text(
                value + maximum * .015, bar.get_y() + bar.get_height() / 2,
                f"{shown} · {status}", va="center", fontsize=9, color="#333333",
            )
    figure.suptitle(
        "Player-control methods on one matched 3D circuit",
        fontsize=18, fontweight="bold", color="#171717",
    )
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def trajectory_chart(scene, results: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left, right = track_edge_points(scene)
    center = [(point.x, point.y) for point in scene.track_centerline]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9), layout="constrained")
    for axis, result in zip(axes.flat, results):
        for edge in (left, right):
            closed = [*edge, edge[0]]
            axis.plot([p[0] for p in closed], [p[1] for p in closed], color="#4b4b4b", linewidth=1.3)
        closed_center = [*center, center[0]]
        axis.plot(
            [p[0] for p in closed_center], [p[1] for p in closed_center],
            color="#b5b5b5", linewidth=.8, linestyle=(0, (5, 6)),
        )
        trace = [
            (frame.privileged_state.player.x, frame.privileged_state.player.y)
            for frame in result["frames"]
            if frame.privileged_state.countdown_ticks_remaining == 0
        ]
        if trace:
            color = "#276FBF" if result["completed"] else "#C84C4C"
            axis.plot([p[0] for p in trace], [p[1] for p in trace], color=color, linewidth=2.2)
            axis.scatter(*trace[0], color="#2E8B57", s=34, zorder=4)
            axis.scatter(*trace[-1], color="#202020", marker="x", s=38, zorder=4)
        status = "completed" if result["completed"] else "not completed"
        axis.set_title(f"{result['label']}\n{result['steps']} ticks · {status}", fontsize=10.5)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d0d0d0")
    figure.suptitle("Matched 3D player trajectories · top-down", fontsize=18, fontweight="bold")
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/player-method-comparison-3d/writeup"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.study.read_text(encoding="utf-8"))
    results = []
    scene = None
    for row in summary["results"]:
        replay = ReplayBundle.model_validate_json(Path(row["replay"]).read_text(encoding="utf-8"))
        scene = replay.scene
        results.append({**row, "frames": replay.frames})
    assert scene is not None
    metrics = args.output_dir / "player_method_metrics_3d.png"
    trajectories = args.output_dir / "player_method_trajectories_3d.png"
    metrics_chart(results, metrics, scene.dynamics.control_hz)
    trajectory_chart(scene, results, trajectories)
    payload = {
        **summary,
        "source_summary": str(args.study.resolve()),
        "metrics_chart": str(metrics.resolve()),
        "trajectory_chart": str(trajectories.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": str(metrics.resolve()), "trajectories": str(trajectories.resolve())}, indent=2))


if __name__ == "__main__":
    main()

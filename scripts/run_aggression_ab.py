"""Run and plot a matched conservative-versus-attacking predictive-skill demo."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from harness.collision import track_edge_points
from harness.models import RunRequest
from harness.racing import CAR_RADIUS, _distance_to_polyline
from harness.service import HarnessService
from harness.store import HarnessStore


DEFAULT_FIXTURE_STORE = Path(".harness-data/player_reflex_ab/20260818T163834Z")


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def summarize(run, scene, label: str) -> dict:
    active = [
        frame for frame in run.frames
        if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    on_track = [
        _distance_to_polyline(
            frame.privileged_state.player, scene.track_centerline, closed=True,
        ) <= scene.track_width / 2 - CAR_RADIUS
        for frame in active
    ]
    realtime = run.realtime_metrics or {}
    return {
        "label": label,
        "run_id": run.id,
        "aggression": run.player_aggression,
        "completed": run.status.value == "succeeded",
        "status": run.status.value,
        "reason": run.result_reason,
        "ticks": len(active),
        "model_calls": run.player_turns,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "total_tokens": run.token_usage,
        "provider_latency_ms": run.latency_ms,
        "mean_decision_ticks": realtime.get("mean_decision_ticks"),
        "on_track_fraction": round(sum(on_track) / max(1, len(on_track)), 3),
        "off_track_ticks": sum(not value for value in on_track),
    }


def trajectory_chart(scene, runs: list[tuple[str, object]], summaries: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left, right = track_edge_points(scene)
    center = [(point.x, point.y) for point in scene.track_centerline]
    colors = ("#82cfff", "#ff9f43")
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.4), layout="constrained")
    figure.patch.set_facecolor("#101619")
    for axis, (label, run), summary, color in zip(axes, runs, summaries, colors):
        axis.set_facecolor("#182124")
        # Match the established RaceLab trajectory aesthetic: bright road edges,
        # dashed centreline, coloured trace, green start, red finish.
        road_polygon = [*left, *reversed(right)]
        axis.fill(
            [point[0] for point in road_polygon],
            [point[1] for point in road_polygon],
            color="#252f32", alpha=.95, zorder=0,
        )
        for edge in (left, right):
            closed = [*edge, edge[0]]
            axis.plot(
                [point[0] for point in closed], [point[1] for point in closed],
                color="#d9ded8", linewidth=1.5, zorder=2,
            )
        closed_center = [*center, center[0]]
        axis.plot(
            [point[0] for point in closed_center], [point[1] for point in closed_center],
            color="#748184", linewidth=.8, linestyle=(0, (5, 7)), zorder=1,
        )
        trace = [
            (frame.privileged_state.player.x, frame.privileged_state.player.y)
            for frame in run.frames
            if frame.privileged_state.countdown_ticks_remaining == 0
        ]
        if trace:
            axis.plot(
                [point[0] for point in trace], [point[1] for point in trace],
                color=color, linewidth=2.8, solid_capstyle="round", zorder=3,
            )
            axis.scatter(*trace[0], color="#77dd9b", s=52, edgecolor="#101619", zorder=5)
            axis.scatter(*trace[-1], color="#ff6b6b", s=52, edgecolor="#101619", zorder=5)
        status = "completed" if summary["completed"] else summary["status"]
        axis.set_title(
            f"{label} · aggression {summary['aggression']:.2f}\n"
            f"{summary['ticks']} ticks · {status} · {summary['on_track_fraction']:.1%} on track",
            color="#f4f0e5", fontsize=11, pad=12,
        )
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.tick_params(colors="#aebabc", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#526164")
    figure.suptitle(
        "Predictive visual skills · player aggression A/B",
        color="#f4f0e5", fontsize=18,
    )
    figure.savefig(path, dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, default=DEFAULT_FIXTURE_STORE)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--decision-budget", type=int, default=8)
    parser.add_argument("--conservative", type=float, default=.15)
    parser.add_argument("--aggressive", type=float, default=.95)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    output = (args.output_dir or Path(".harness-data/aggression_ab") /
              datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    store = HarnessStore(args.fixture_store)
    service = HarnessService(store=store)
    environment = next(
        (item for item in store.list_environments() if item.scene.seed == args.seed), None,
    )
    if environment is None:
        raise SystemExit(f"No seed-{args.seed} environment in {args.fixture_store}")

    runs: list[tuple[str, object]] = []
    for label, aggression in (
        ("Conservative", args.conservative), ("Aggressive", args.aggressive),
    ):
        print(f"\n[{label} · aggression {aggression:.2f}]", flush=True)
        run = service.run(RunRequest(
            environment_id=environment.id,
            policy_name="vision-2d-predictive-skills",
            max_steps=args.max_steps,
            policy_decision_budget=args.decision_budget,
            player_aggression=aggression,
        ))
        runs.append((label, run))
        print(json.dumps(summarize(run, environment.scene, label), indent=2), flush=True)

    summaries = [summarize(run, environment.scene, label) for label, run in runs]
    chart = output / "top_down_aggression_ab.png"
    trajectory_chart(environment.scene, runs, summaries, chart)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.player_model,
        "environment_id": environment.id,
        "track": environment.scene.name,
        "seed": environment.scene.seed,
        "agent_contract": (
            "forward-cone RGB + physical speed only; no map, pose, heading, progress, "
            "checkpoint, centerline, or privileged simulator rollout"
        ),
        "controlled_variable": "player_aggression",
        "max_steps": args.max_steps,
        "decision_budget": args.decision_budget,
        "results": summaries,
        "trajectory_chart": str(chart),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}\nTrajectories: {chart}", flush=True)


if __name__ == "__main__":
    main()

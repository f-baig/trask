"""Run a matched five-method 2D player-control pilot and make write-up figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from harness.collision import track_edge_points
from harness.models import Action
from harness.policies import ConeVisualRefreshPolicy, PolicyBudgetExhausted
from harness.providers import ProviderError
from harness.racing import CAR_RADIUS, RacingBackend, _distance_to_polyline
from harness.reflex.episode import replay_bundle, run_reflex_episode
from harness.rendering import ReplayBundle, ReplayMetadata
from harness.store import HarnessStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_2d_pipeline_ab import ARMS, DEFAULT_STORE, load_dotenv, run_arm  # noqa: E402


LABELS = {
    "predictive-skills": "Predictive + skill library",
    "predictive-generated": "Predictive + custom scripts",
    "event-generated": "Event-triggered custom scripts",
    "interval-generated": "Interval custom scripts",
    "every-tick": "Model call every tick",
}
ORDER = tuple(LABELS)


def evaluator_metrics(scene, frames) -> dict:
    active = [
        frame for frame in frames
        if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    on_track = [
        _distance_to_polyline(
            frame.privileged_state.player, scene.track_centerline, closed=True,
        ) <= scene.track_width / 2 - CAR_RADIUS
        for frame in active
    ]
    return {
        "evaluator_on_track_fraction": round(sum(on_track) / max(1, len(on_track)), 3),
        "evaluator_off_track_ticks": sum(not value for value in on_track),
    }


def run_event_arm(scene, *, model: str, max_steps: int, max_wakes: int) -> dict:
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    started = time.perf_counter()
    report = run_reflex_episode(
        world, model=model, max_steps=max_steps, max_wakes=max_wakes,
        rehearsal_budget=0, latency="measured", vision_only=True,
        visual_mode="2d", verbose=True,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1_000)
    result = {
        "arm": "event-generated", "label": LABELS["event-generated"],
        "completed": report.succeeded, "reason": report.reason,
        "steps": report.ticks, "checkpoints_reached": report.checkpoints_reached,
        "checkpoint_total": len(scene.objectives), "model_calls": report.usage.calls,
        "successful_model_calls": report.usage.calls,
        "input_tokens": report.usage.input_tokens,
        "output_tokens": report.usage.output_tokens,
        "total_tokens": report.usage.input_tokens + report.usage.output_tokens,
        "provider_latency_ms": report.usage.latency_ms,
        "mean_call_latency_ms": round(report.usage.latency_ms / max(1, report.usage.calls)),
        "evaluation_wall_time_ms": elapsed_ms,
        "game_latency_ticks": report.stale_ticks,
        "event_wakes": report.wakes,
        "rehearsal_budget": 0,
        "controller_writes": report.diagnostics_payload.get("final_runtime", {}).get("controllers", []),
        "event_diagnostics": report.diagnostics_payload,
        "frames": report.frames,
    }
    result.update(evaluator_metrics(scene, report.frames))
    return result


def run_every_tick_arm(scene, *, max_steps: int) -> dict:
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    policy = ConeVisualRefreshPolicy(refresh_ticks=1)
    policy.configure_episode(max_steps, max_steps)
    policy.reset(scene, scene.seed)
    frames = []
    active_ticks = 0
    failure = None
    started = time.perf_counter()
    while not world.terminated and active_ticks < max_steps:
        if world.countdown_ticks_remaining > 0:
            frames.append(world.step(Action()))
            continue
        try:
            observation = world.observe()
            action, decision = policy.act_visual(observation, policy.render_frame(world))
        except (PolicyBudgetExhausted, ProviderError) as error:
            failure = str(error)
            break
        frames.append(world.step(action, decision))
        active_ticks += 1
        if active_ticks % 25 == 0:
            print(
                f"  every-tick progress: {active_ticks}/{max_steps} ticks, "
                f"{policy.planning_turns} calls, "
                f"{policy.input_tokens + policy.output_tokens:,} tokens",
                flush=True,
            )
    elapsed_ms = round((time.perf_counter() - started) * 1_000)
    result = {
        "arm": "every-tick", "label": LABELS["every-tick"],
        "completed": world.succeeded,
        "reason": failure or world.reason or "step budget exhausted",
        "steps": active_ticks, "checkpoints_reached": world.objective_index,
        "checkpoint_total": len(scene.objectives),
        "model_calls": policy.planning_turns,
        "successful_model_calls": policy.planning_turns,
        "input_tokens": policy.input_tokens, "output_tokens": policy.output_tokens,
        "total_tokens": policy.input_tokens + policy.output_tokens,
        "provider_latency_ms": policy.latency_ms,
        "mean_call_latency_ms": round(policy.latency_ms / max(1, policy.planning_turns)),
        "evaluation_wall_time_ms": elapsed_ms,
        "game_latency_ticks": 0,
        "frames": frames,
    }
    result.update(evaluator_metrics(scene, frames))
    return result


def save_replay(scene, result: dict, output: Path) -> None:
    replay_path = output / f"{result['arm']}-replay.json"
    replay = ReplayBundle.from_frames(
        scene, result["frames"],
        metadata=ReplayMetadata(
            run_id=f"player-method-{result['arm']}",
            policy_name=result["arm"],
            status="succeeded" if result["completed"] else "failed",
            seed=scene.seed,
            total_reward=round(sum(frame.reward for frame in result["frames"]), 4),
        ),
    )
    replay_path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
    result["replay"] = str(replay_path)


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
        axis.set_xlim(0, maximum * 1.22 if maximum else 1)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#202020")
        axis.set_xlabel(unit, color="#4a4a4a")
        style_axis(axis)
        if field == "total_tokens":
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
        for bar, value, result in zip(bars, values, results):
            shown = (
                f"{result[field]:,}" if field in {"total_tokens", "model_calls"} else
                f"{value:.1f}"
            )
            status = "completed" if result["completed"] else "not completed"
            axis.text(
                value + maximum * .015, bar.get_y() + bar.get_height() / 2,
                f"{shown} · {status}", va="center", fontsize=9, color="#333333",
            )
    figure.suptitle(
        "Player-control methods on one matched 2D circuit",
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
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
    axes = list(axes.flat)
    for axis, result in zip(axes, results):
        for edge in (left, right):
            closed = [*edge, edge[0]]
            axis.plot([point[0] for point in closed], [point[1] for point in closed], color="#4b4b4b", linewidth=1.3)
        closed_center = [*center, center[0]]
        axis.plot(
            [point[0] for point in closed_center], [point[1] for point in closed_center],
            color="#b5b5b5", linewidth=.8, linestyle=(0, (5, 6)),
        )
        trace = [
            (frame.privileged_state.player.x, frame.privileged_state.player.y)
            for frame in result["frames"]
            if frame.privileged_state.countdown_ticks_remaining == 0
        ]
        if trace:
            color = "#276FBF" if result["completed"] else "#C84C4C"
            axis.plot([point[0] for point in trace], [point[1] for point in trace], color=color, linewidth=2.2)
            axis.scatter(*trace[0], color="#2E8B57", s=34, zorder=4)
            axis.scatter(*trace[-1], color="#202020", marker="x", s=38, zorder=4)
        status = "completed" if result["completed"] else "not completed"
        axis.set_title(f"{result['label']}\n{result['steps']} ticks · {status}", fontsize=10.5, color="#202020")
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d0d0d0")
    axes[-1].axis("off")
    figure.suptitle("Matched player trajectories", fontsize=18, fontweight="bold", color="#171717")
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--max-wakes", type=int, default=6)
    parser.add_argument("--wake-interval", type=int, default=70)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    os.environ.setdefault("OPENAI_REQUEST_TIMEOUT_SECONDS", "45")
    output = (args.output_dir or Path("artifacts/player-method-comparison") /
              datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = next(
        (item for item in HarnessStore(args.fixture_store).list_environments()
         if item.scene.seed == args.seed), None,
    )
    if environment is None:
        raise SystemExit(f"No seed-{args.seed} environment in {args.fixture_store}")
    scene = environment.scene
    source_arms = {arm.id: arm for arm in ARMS}
    results = []
    for arm_id in ("predictive-skills", "predictive-generated"):
        print(f"\n[{LABELS[arm_id]}]", flush=True)
        result = run_arm(
            scene, source_arms[arm_id], max_steps=args.max_steps,
            max_calls=args.max_calls, wake_interval=args.wake_interval,
        )
        result["label"] = LABELS[arm_id]
        results.append(result)
    print(f"\n[{LABELS['event-generated']}]", flush=True)
    results.append(run_event_arm(
        scene, model=args.player_model, max_steps=args.max_steps, max_wakes=args.max_wakes,
    ))
    print(f"\n[{LABELS['interval-generated']}]", flush=True)
    interval = run_arm(
        scene, source_arms["blocking-generated"], max_steps=args.max_steps,
        max_calls=args.max_calls, wake_interval=args.wake_interval,
    )
    interval["arm"] = "interval-generated"
    interval["label"] = LABELS["interval-generated"]
    results.append(interval)
    print(f"\n[{LABELS['every-tick']}]", flush=True)
    results.append(run_every_tick_arm(scene, max_steps=args.max_steps))

    results_by_id = {result["arm"]: result for result in results}
    ordered = [results_by_id[arm_id] for arm_id in ORDER]
    for result in ordered:
        save_replay(scene, result, output)
    metrics_path = output / "player_method_metrics.png"
    trajectories_path = output / "player_method_trajectories.png"
    metrics_chart(ordered, metrics_path, scene.dynamics.control_hz)
    trajectory_chart(scene, ordered, trajectories_path)
    summary_results = []
    for result in ordered:
        compact = dict(result)
        compact.pop("frames", None)
        summary_results.append(compact)
    summary = {
        "created_at": datetime.now(UTC).isoformat(), "study": "single-run matched pilot",
        "model": args.player_model, "environment_id": environment.id,
        "track": scene.name, "seed": scene.seed,
        "agent_contract": (
            "forward-cone RGB + scalar physical speed; no map, pose, heading, progress, "
            "checkpoint, centerline, or privileged simulator rollout"
        ),
        "max_steps": args.max_steps, "max_calls_for_long_horizon_arms": args.max_calls,
        "event_max_wakes": args.max_wakes, "event_rehearsal_budget": 0,
        "interval_ticks": args.wake_interval, "results": summary_results,
        "metrics_chart": str(metrics_path), "trajectory_chart": str(trajectories_path),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"Trajectories: {trajectories_path}", flush=True)


if __name__ == "__main__":
    main()

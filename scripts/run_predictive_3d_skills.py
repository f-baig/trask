"""Run the latency-compensated 3D visual-skill player and plot its path."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from harness.collision import track_edge_points
from harness.policies import PredictiveVisualSkillPolicy
from harness.racing import CAR_RADIUS, RacingBackend, _distance_to_polyline
from harness.realtime import run_realtime_episode
from harness.rendering import ReplayBundle, ReplayMetadata
from harness.store import HarnessStore


DEFAULT_FIXTURE = Path(".harness-data/direct_3d_visual/compact-fixture-20260819T020117Z")


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def plot_trajectory(scene, frames, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left, right = track_edge_points(scene)
    trace = [
        (frame.privileged_state.player.x, frame.privileged_state.player.y)
        for frame in frames if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    figure, axis = plt.subplots(figsize=(9.6, 6.4), layout="constrained")
    figure.patch.set_facecolor("#101619")
    axis.set_facecolor("#182124")
    for edge in (left, right):
        closed = [*edge, edge[0]]
        axis.plot(
            [point[0] for point in closed], [point[1] for point in closed],
            color="#d9ded8", linewidth=2.0, alpha=0.85,
        )
    center = [(point.x, point.y) for point in scene.track_centerline]
    center.append(center[0])
    axis.plot(
        [point[0] for point in center], [point[1] for point in center],
        color="#7e8b8d", linewidth=1.0, linestyle=(0, (5, 7)), alpha=0.8,
        label="track center",
    )
    if trace:
        axis.plot(
            [point[0] for point in trace], [point[1] for point in trace],
            color="#ffbf59", linewidth=3.2, solid_capstyle="round",
            solid_joinstyle="round", label="predictive skill path",
        )
        axis.scatter(*trace[0], color="#77dd9b", s=70, edgecolor="#101619", zorder=5, label="start")
        axis.scatter(*trace[-1], color="#ff6b6b", s=70, edgecolor="#101619", zorder=5, label="end")
    axis.set_title("3D predictive-overlap trajectory", color="#f4f0e5", fontsize=16, pad=14)
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.tick_params(colors="#aebabc")
    for spine in axis.spines.values():
        spine.set_color("#526164")
    axis.legend(facecolor="#101619", edgecolor="#526164", labelcolor="#f4f0e5")
    axis.set_xlabel("world x", color="#aebabc")
    axis.set_ylabel("world y", color="#aebabc")
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--decision-budget", type=int, default=30)
    parser.add_argument("--clock", choices=("wall", "measured", "fixed"), default="wall")
    parser.add_argument("--initial-latency-ticks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    os.environ.setdefault("OPENAI_REQUEST_TIMEOUT_SECONDS", "30")

    output = (args.output_dir or Path(".harness-data/predictive_3d") /
              datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environments = HarnessStore(args.fixture_store).list_environments()
    if not environments:
        raise SystemExit(f"No environment found in {args.fixture_store}")
    scene = environments[0].scene
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    policy = PredictiveVisualSkillPolicy()

    print(
        f"Running {policy.name} with {args.player_model} on {args.clock} clock "
        f"({args.max_steps} tick cap)", flush=True,
    )
    result = run_realtime_episode(
        world, policy, max_steps=args.max_steps, clock=args.clock,
        latency_ticks=args.initial_latency_ticks,
        decision_budget=args.decision_budget,
        progress=lambda tick, action, report: (
            print(
                f"tick={tick} speed={world.speed:.2f} action={action.keys} "
                f"skill={policy.active_skill.name} decisions={report.decisions} "
                f"rejected={report.rejected_decisions}", flush=True,
            ) if tick % 25 == 0 else None
        ),
    )
    active_frames = [
        frame for frame in result["frames"]
        if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    on_track = [
        _distance_to_polyline(
            frame.privileged_state.player, scene.track_centerline, closed=True,
        ) <= scene.track_width / 2 - CAR_RADIUS
        for frame in active_frames
    ]
    off_track_ticks = sum(not value for value in on_track)
    trajectory_path = output / "top_down_trajectory.png"
    plot_trajectory(scene, result["frames"], trajectory_path)
    replay = ReplayBundle.from_frames(
        scene, result["frames"],
        metadata=ReplayMetadata(
            run_id=f"predictive-{datetime.now(UTC).strftime('%H%M%S')}",
            policy_name=policy.name,
            status="succeeded" if result["succeeded"] else "failed",
            seed=scene.seed,
            total_reward=round(sum(frame.reward for frame in result["frames"]), 4),
        ),
    )
    replay_path = output / "replay.json"
    replay_path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.player_model,
        "policy": policy.name,
        "agent_contract": (
            "first-person screenshots + physical speed; road geometry is RGB-derived; "
            "no pose, heading, centerline, map, progress, elevation, or simulator rollout"
        ),
        "clock": args.clock,
        "completed": result["succeeded"],
        "reason": result["reason"],
        "steps": result["realtime"]["ticks"],
        "model_calls": len(policy.turn_usages or []),
        "input_tokens": policy.input_tokens,
        "output_tokens": policy.output_tokens,
        "total_tokens": policy.input_tokens + policy.output_tokens,
        "provider_latency_ms": policy.latency_ms,
        "evaluator_off_track_ticks": off_track_ticks,
        "evaluator_on_track_fraction": round(sum(on_track) / max(1, len(on_track)), 3),
        "mean_speed_mps": round(sum(frame.privileged_state.speed for frame in active_frames) / max(1, len(active_frames)), 3),
        "max_speed_mps": round(max((frame.privileged_state.speed for frame in active_frames), default=0), 3),
        "realtime": result["realtime"],
        "predictive": result["policy_realtime"],
        "trajectory": str(trajectory_path),
        "replay": str(replay_path),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in summary.items()
        if key not in {"predictive"}
    }, indent=2), flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

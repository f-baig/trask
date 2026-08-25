"""Evaluate four controller-latency architectures on one matched 3D circuit."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from harness.collision import track_edge_points
from harness.models import Action
from harness.pipeline3d import (
    GeneratedPerspectiveDriver, PerspectiveSkillDriver, prediction_matches,
)
from harness.providers import (
    ProviderError, plan_generated_perspective_controller,
    plan_predictive_driving_skill,
)
from harness.racing import CAR_RADIUS, RacingBackend, _distance_to_polyline
from harness.rendering import ReplayBundle, ReplayMetadata
from harness.store import HarnessStore


DEFAULT_STORE = Path(".harness-data/direct_3d_visual/compact-fixture-20260819T020117Z")


@dataclass(frozen=True)
class Arm:
    id: str
    label: str
    controller: str
    overlap: bool
    predictive: bool


ARMS = (
    Arm("blocking-generated", "Blocking generated controllers", "generated", False, False),
    Arm("overlap-stale", "Overlapped, stale controllers", "generated", True, False),
    Arm("predictive-generated", "Predictive overlap + generated controllers", "generated", True, True),
    Arm("predictive-skills", "Predictive overlap + skill library", "skills", True, True),
)


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def model_call(arm: Arm, driver, frame, state, horizon: int, control_hz: int):
    if arm.controller == "skills":
        return plan_predictive_driving_skill(
            frame, public_state=state, active_skill=asdict_safe(driver.active),
            previous_controls=driver.recent_controls,
            activation_horizon_ticks=horizon, control_hz=control_hz,
        )
    return plan_generated_perspective_controller(
        frame, public_state=state, current_source=driver.current_source,
        recent_controls=driver.recent_controls, predictive=arm.predictive,
        activation_horizon_ticks=horizon, control_hz=control_hz,
        install_feedback=driver.last_install_error,
    )


def asdict_safe(value) -> dict:
    return {
        "name": value.name, "target_speed": value.target_speed,
        "target_offset": value.target_offset, "turn_direction": value.turn_direction,
    }


def run_arm(scene, arm: Arm, *, max_steps: int, max_calls: int, wake_interval: int) -> dict:
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    driver = (
        PerspectiveSkillDriver(scene) if arm.controller == "skills"
        else GeneratedPerspectiveDriver(scene)
    )
    frames, calls, prediction_log, provider_failures = [], [], [], []
    attempted_calls = active_ticks = stale_ticks = 0
    expected_latency_ticks = 18
    next_wake = wake_interval
    failure = None
    started = time.perf_counter()

    def record_usage(usage, *, age_ticks: int, initial: bool) -> None:
        calls.append({
            "call": attempted_calls, "provider": usage.provider, "model": usage.model,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "latency_ms": usage.latency_ms, "application_age_ticks": age_ticks,
            "initial_pre_race_call": initial,
        })

    def record_provider_failure(error: ProviderError, *, initial: bool) -> None:
        usage = getattr(error, "usage", None)
        if usage is not None:
            record_usage(usage, age_ticks=0, initial=initial)
        provider_failures.append({
            "attempt": attempted_calls, "tick": active_ticks, "error": str(error),
            "usage_recorded": usage is not None,
        })

    def drive(ticks: int) -> None:
        nonlocal active_ticks
        for _ in range(ticks):
            if world.terminated or active_ticks >= max_steps:
                return
            frames.append(world.step(driver.tick(world)))
            active_ticks += 1

    # All modes receive an initial controller while the starting grid is frozen.
    for _attempt in range(2):
        state, frame = driver.observe(world)
        attempted_calls += 1
        try:
            plan, usage = model_call(arm, driver, frame, state, 0, scene.dynamics.control_hz)
        except ProviderError as error:
            record_provider_failure(error, initial=True)
            failure = str(error)
            continue
        record_usage(usage, age_ticks=0, initial=True)
        installed, reason = driver.install(plan, 0)
        if installed:
            failure = None
            break
        failure = f"initial controller rejected: {reason}"
    if failure is None:
        while world.countdown_ticks_remaining > 0:
            frames.append(world.step(Action()))

    while failure is None and not world.terminated and active_ticks < max_steps:
        if active_ticks >= next_wake and attempted_calls < max_calls:
            submitted_tick = active_ticks
            state, frame = driver.observe(world)
            attempted_calls += 1
            try:
                plan, usage = model_call(
                    arm, driver, frame, state,
                    expected_latency_ticks if arm.predictive else 0,
                    scene.dynamics.control_hz,
                )
            except ProviderError as error:
                record_provider_failure(error, initial=False)
                next_wake = active_ticks + wake_interval
                continue
            latency_ticks = math.ceil(max(0, usage.latency_ms) / (1_000 / scene.dynamics.control_hz))
            charged = latency_ticks if arm.overlap else 0
            if charged:
                drive(charged)
                stale_ticks += min(charged, active_ticks - submitted_tick)
            age = active_ticks - submitted_tick
            record_usage(usage, age_ticks=age, initial=False)
            ages = [item["application_age_ticks"] for item in calls if not item["initial_pre_race_call"]]
            expected_latency_ticks = max(1, round(sum(ages) / max(1, len(ages))))
            accepted, diagnostic = True, None
            if arm.predictive and not world.terminated and active_ticks < max_steps:
                actual, _ = driver.observe(world)
                accepted, diagnostic = prediction_matches(plan, actual)
                diagnostic.update({
                    "call": attempted_calls, "submitted_tick": submitted_tick,
                    "landed_tick": active_ticks,
                })
                prediction_log.append(diagnostic)
            if accepted and not world.terminated and active_ticks < max_steps:
                installed, reason = driver.install(plan, active_ticks)
                if not installed and arm.predictive:
                    prediction_log.append({
                        "call": attempted_calls, "submitted_tick": submitted_tick,
                        "landed_tick": active_ticks, "accepted": False,
                        "install_rejected": reason,
                    })
            next_wake = active_ticks + wake_interval
            continue
        drive(1)

    elapsed_ms = round((time.perf_counter() - started) * 1_000)
    active_frames = [
        frame for frame in frames if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    on_track = [
        _distance_to_polyline(frame.privileged_state.player, scene.track_centerline, closed=True)
        <= scene.track_width / 2 - CAR_RADIUS
        for frame in active_frames
    ]
    input_tokens = sum(item["input_tokens"] for item in calls)
    output_tokens = sum(item["output_tokens"] for item in calls)
    provider_latency_ms = sum(item["latency_ms"] for item in calls)
    latency_ages = [item["application_age_ticks"] for item in calls if not item["initial_pre_race_call"]]
    return {
        "arm": arm.id, "label": arm.label, "completed": world.succeeded,
        "reason": failure or world.reason or "step budget exhausted",
        "steps": active_ticks, "checkpoints_reached": world.objective_index,
        "checkpoint_total": len(scene.objectives), "model_calls": attempted_calls,
        "successful_model_calls": attempted_calls - len(provider_failures),
        "failed_model_calls": len(provider_failures),
        "billed_responses_with_usage": len(calls),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "provider_latency_ms": provider_latency_ms,
        "mean_call_latency_ms": round(provider_latency_ms / max(1, len(calls))),
        "evaluation_wall_time_ms": elapsed_ms, "game_latency_ticks": stale_ticks,
        "mean_application_age_ticks": round(sum(latency_ages) / max(1, len(latency_ages)), 2),
        "max_application_age_ticks": max(latency_ages, default=0),
        "prediction_attempts": len(prediction_log),
        "prediction_accepts": sum(bool(item.get("accepted")) for item in prediction_log),
        "prediction_rejections": sum(not bool(item.get("accepted")) for item in prediction_log),
        "evaluator_on_track_fraction": round(sum(on_track) / max(1, len(on_track)), 3),
        "evaluator_off_track_ticks": sum(not item for item in on_track),
        "calls": calls, "provider_failures": provider_failures,
        "prediction_log": prediction_log,
        "controller_writes": getattr(driver, "writes", []),
        "skill_activations": getattr(driver, "activations", []),
        "frames": frames,
    }


def trajectory_chart(scene, results: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left, right = track_edge_points(scene)
    center = [(point.x, point.y) for point in scene.track_centerline]
    colors = ("#ffbf59", "#f28e72", "#82cfff", "#bd93f9")
    if len(results) == 1:
        figure, only_axis = plt.subplots(figsize=(9.6, 6.4), layout="constrained")
        axes = [only_axis]
    elif len(results) == 2:
        figure, grid = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
        axes = list(grid.flat)
    else:
        figure, grid = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        axes = list(grid.flat)
    figure.patch.set_facecolor("#101619")
    for axis, result, color in zip(axes, results, colors):
        axis.set_facecolor("#182124")
        for edge in (left, right):
            closed = [*edge, edge[0]]
            axis.plot([p[0] for p in closed], [p[1] for p in closed], color="#d9ded8", linewidth=1.5)
        closed_center = [*center, center[0]]
        axis.plot([p[0] for p in closed_center], [p[1] for p in closed_center], color="#748184", linewidth=.8, linestyle=(0, (5, 7)))
        trace = [
            (frame.privileged_state.player.x, frame.privileged_state.player.y)
            for frame in result["frames"] if frame.privileged_state.countdown_ticks_remaining == 0
        ]
        if trace:
            axis.plot([p[0] for p in trace], [p[1] for p in trace], color=color, linewidth=2.8, solid_capstyle="round")
            axis.scatter(*trace[0], color="#77dd9b", s=45, edgecolor="#101619", zorder=5)
            axis.scatter(*trace[-1], color="#ff6b6b", s=45, edgecolor="#101619", zorder=5)
        status = "completed" if result["completed"] else f"{result['checkpoints_reached']}/{result['checkpoint_total']} gates"
        axis.set_title(f"{result['label']}\n{result['steps']} ticks · {status}", color="#f4f0e5", fontsize=11)
        axis.set_aspect("equal"); axis.invert_yaxis(); axis.tick_params(colors="#aebabc", labelsize=8)
        for spine in axis.spines.values(): spine.set_color("#526164")
    for axis in axes[len(results):]:
        axis.set_visible(False)
    figure.suptitle("Matched 3D controller-latency evaluation · top-down", color="#f4f0e5", fontsize=18)
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--wake-interval", type=int, default=70)
    parser.add_argument("--arms", nargs="+", choices=[arm.id for arm in ARMS])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    os.environ.setdefault("OPENAI_REQUEST_TIMEOUT_SECONDS", "45")
    output = (args.output_dir or Path(".harness-data/3d_pipeline_ab") /
              datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environments = HarnessStore(args.fixture_store).list_environments()
    if not environments:
        raise SystemExit(f"No 3D environment in {args.fixture_store}")
    environment, results = environments[0], []
    scene = environment.scene
    if scene.elevation is None or scene.elevation.is_flat:
        raise SystemExit("The matched environment must have non-flat elevation")
    selected_arms = [arm for arm in ARMS if not args.arms or arm.id in args.arms]
    for arm in selected_arms:
        print(f"\n[{arm.label}]", flush=True)
        result = run_arm(
            scene, arm, max_steps=args.max_steps,
            max_calls=args.max_calls, wake_interval=args.wake_interval,
        )
        replay_path = output / f"{arm.id}-replay.json"
        replay = ReplayBundle.from_frames(
            scene, result["frames"],
            metadata=ReplayMetadata(
                run_id=f"3d-{arm.id}", policy_name=arm.id,
                status="succeeded" if result["completed"] else "failed",
                seed=scene.seed,
                total_reward=round(sum(frame.reward for frame in result["frames"]), 4),
            ),
        )
        replay_path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
        result["replay"] = str(replay_path)
        result.pop("frames")
        results.append(result)
        print(json.dumps({
            key: result[key] for key in (
                "completed", "reason", "steps", "checkpoints_reached", "model_calls",
                "total_tokens", "provider_latency_ms", "game_latency_ticks",
                "mean_application_age_ticks", "evaluator_on_track_fraction",
            )
        }, indent=2), flush=True)
    plot_results = []
    for result in results:
        bundle = ReplayBundle.model_validate_json(Path(result["replay"]).read_text(encoding="utf-8"))
        plot_results.append({**result, "frames": bundle.frames})
    chart = output / "top_down_trajectories.png"
    trajectory_chart(scene, plot_results, chart)
    summary = {
        "created_at": datetime.now(UTC).isoformat(), "model": args.player_model,
        "environment_id": environment.id, "track": scene.name, "seed": scene.seed,
        "elevation": scene.elevation.model_dump(mode="json"),
        "agent_contract": (
            "first-person RGB + physical speed only; road/elevation cues are RGB-derived; "
            "no map, pose, world heading, progress, checkpoint, centerline, elevation telemetry, "
            "or privileged simulator rollout"
        ),
        "max_steps": args.max_steps, "max_calls": args.max_calls,
        "wake_interval": args.wake_interval, "results": results,
        "trajectory_chart": str(chart),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}\nTrajectories: {chart}", flush=True)


if __name__ == "__main__":
    main()

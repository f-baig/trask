"""Compare a direct screenshot agent with the predictive visual-skill harness."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from harness.models import Action, ActionName
from harness.providers import ProviderError, plan_perspective_visual_actions
from harness.racing import CAR_RADIUS, RacingBackend, _distance_to_polyline
from harness.rendering import ReplayBundle, ReplayMetadata
from harness.store import HarnessStore
from harness.view3d import ViewMode, render_policy_view
from run_3d_pipeline_ab import ARMS, load_dotenv, run_arm, trajectory_chart


DEFAULT_STORE = Path(".harness-data/direct_3d_visual/compact-fixture-20260819T020117Z")


def _actions(plan) -> list[Action]:
    return [
        Action(name=ActionName(segment.action), keys=segment.keys)
        for segment in plan.actions
    ]


def run_direct(scene, *, max_steps: int, max_calls: int) -> dict:
    """Run direct screenshot-to-actions control with response latency in game time.

    The model authors at most four individual tick actions per call. While a call is
    outstanding, the simulator continues using the last model-requested key state.
    No camera measurement, controller, prediction, or reusable skill is added.
    """
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    frames: list = []
    calls: list[dict] = []
    failures: list[dict] = []
    previous_controls: list[dict] = []
    queue: list[Action] = []
    last_action = Action()
    attempted_calls = active_ticks = stale_ticks = 0
    queue_exhausted_at: int | None = None
    failure: str | None = None
    started = time.perf_counter()

    def record_usage(usage, *, age_ticks: int, initial: bool, plan=None) -> None:
        calls.append({
            "call": attempted_calls,
            "provider": usage.provider,
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": usage.latency_ms,
            "application_age_ticks": age_ticks,
            "initial_pre_race_call": initial,
            "model_actions": [] if plan is None else [
                {"action": segment.action, "keys": list(segment.keys)}
                for segment in plan.actions
            ],
            "summary": None if plan is None else plan.summary,
        })

    def record_failure(error: ProviderError, *, initial: bool) -> None:
        usage = getattr(error, "usage", None)
        if usage is not None:
            record_usage(usage, age_ticks=0, initial=initial)
        failures.append({
            "attempt": attempted_calls,
            "tick": active_ticks,
            "error": str(error),
            "usage_recorded": usage is not None,
        })

    def drive(action: Action, ticks: int) -> None:
        nonlocal active_ticks, last_action
        for _ in range(ticks):
            if world.terminated or active_ticks >= max_steps:
                return
            last_action = action
            frames.append(world.step(action))
            active_ticks += 1
            previous_controls.append({
                "action": action.name.value,
                "keys": list(action.keys),
            })
            del previous_controls[:-12]

    def request(initial: bool):
        nonlocal attempted_calls
        attempted_calls += 1
        frame = render_policy_view(world, mode=ViewMode.FIRST_PERSON)
        speed = world.observe().speed
        return plan_perspective_visual_actions(
            frame,
            previous_controls=previous_controls,
            max_tokens=120,
            max_actions=4,
            speed_mps=speed,
            road_geometry=None,
        )

    # Match the harnessed arm's free cold start: the first response arrives while
    # the starting grid is frozen, so neither architecture pays countdown latency.
    for _attempt in range(2):
        try:
            plan, usage = request(initial=True)
        except ProviderError as error:
            record_failure(error, initial=True)
            failure = str(error)
            continue
        record_usage(usage, age_ticks=0, initial=True, plan=plan)
        queue = _actions(plan)
        failure = None
        break

    if failure is None:
        while world.countdown_ticks_remaining > 0:
            frames.append(world.step(Action()))

    while failure is None and not world.terminated and active_ticks < max_steps:
        if queue:
            drive(queue.pop(0), 1)
            continue

        if attempted_calls >= max_calls:
            if queue_exhausted_at is None:
                queue_exhausted_at = active_ticks
            drive(last_action, 1)
            continue

        submitted_tick = active_ticks
        try:
            plan, usage = request(initial=False)
        except ProviderError as error:
            record_failure(error, initial=False)
            drive(last_action, 1)
            continue

        latency_ticks = math.ceil(max(0, usage.latency_ms) / (1_000 / scene.dynamics.control_hz))
        drive(last_action, latency_ticks)
        charged_ticks = active_ticks - submitted_tick
        stale_ticks += charged_ticks
        record_usage(usage, age_ticks=charged_ticks, initial=False, plan=plan)
        if not world.terminated and active_ticks < max_steps:
            queue = _actions(plan)

    elapsed_ms = round((time.perf_counter() - started) * 1_000)
    active_frames = [
        frame for frame in frames
        if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    on_track = [
        _distance_to_polyline(
            frame.privileged_state.player, scene.track_centerline, closed=True,
        ) <= scene.track_width / 2 - CAR_RADIUS
        for frame in active_frames
    ]
    provider_latency_ms = sum(item["latency_ms"] for item in calls)
    latency_ages = [
        item["application_age_ticks"] for item in calls
        if not item["initial_pre_race_call"]
    ]
    input_tokens = sum(item["input_tokens"] for item in calls)
    output_tokens = sum(item["output_tokens"] for item in calls)
    return {
        "arm": "direct-actions",
        "label": "Non-harnessed direct actions",
        "completed": world.succeeded,
        "reason": failure or world.reason or "step budget exhausted",
        "steps": active_ticks,
        "checkpoints_reached": world.objective_index,
        "checkpoint_total": len(scene.objectives),
        "model_calls": attempted_calls,
        "successful_model_calls": attempted_calls - len(failures),
        "failed_model_calls": len(failures),
        "billed_responses_with_usage": len(calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "provider_latency_ms": provider_latency_ms,
        "mean_call_latency_ms": round(provider_latency_ms / max(1, len(calls))),
        "evaluation_wall_time_ms": elapsed_ms,
        "game_latency_ticks": stale_ticks,
        "mean_application_age_ticks": round(sum(latency_ages) / max(1, len(latency_ages)), 2),
        "max_application_age_ticks": max(latency_ages, default=0),
        "prediction_attempts": 0,
        "prediction_accepts": 0,
        "prediction_rejections": 0,
        "evaluator_on_track_fraction": round(sum(on_track) / max(1, len(on_track)), 3),
        "evaluator_off_track_ticks": sum(not item for item in on_track),
        "direct_action_queue_exhausted_at_tick": queue_exhausted_at,
        "calls": calls,
        "provider_failures": failures,
        "prediction_log": [],
        "controller_writes": [],
        "skill_activations": [],
        "frames": frames,
    }


def save_replay(scene, result: dict, output: Path) -> None:
    replay_path = output / f"{result['arm']}-replay.json"
    replay = ReplayBundle.from_frames(
        scene,
        result["frames"],
        metadata=ReplayMetadata(
            run_id=f"3d-{result['arm']}",
            policy_name=result["arm"],
            status="succeeded" if result["completed"] else "failed",
            seed=scene.seed,
            total_reward=round(sum(frame.reward for frame in result["frames"]), 4),
        ),
    )
    replay_path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
    result["replay"] = str(replay_path)


def compact(result: dict) -> dict:
    return {
        key: result[key]
        for key in (
            "completed", "reason", "steps", "checkpoints_reached", "model_calls",
            "total_tokens", "provider_latency_ms", "game_latency_ticks",
            "mean_application_age_ticks", "evaluator_on_track_fraction",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--wake-interval", type=int, default=70)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    os.environ.setdefault("OPENAI_REQUEST_TIMEOUT_SECONDS", "45")
    output = (
        args.output_dir
        or Path(".harness-data/3d_direct_vs_predictive")
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)

    environments = HarnessStore(args.fixture_store).list_environments()
    if not environments:
        raise SystemExit(f"No 3D environment in {args.fixture_store}")
    environment = environments[0]
    scene = environment.scene
    predictive_arm = next(arm for arm in ARMS if arm.id == "predictive-skills")

    print("\n[Non-harnessed direct actions]", flush=True)
    direct = run_direct(scene, max_steps=args.max_steps, max_calls=args.max_calls)
    print(json.dumps(compact(direct), indent=2), flush=True)

    print("\n[Predictive overlap + skill library]", flush=True)
    predictive = run_arm(
        scene,
        predictive_arm,
        max_steps=args.max_steps,
        max_calls=args.max_calls,
        wake_interval=args.wake_interval,
    )
    print(json.dumps(compact(predictive), indent=2), flush=True)

    results = [direct, predictive]
    for result in results:
        save_replay(scene, result, output)
    chart = output / "top_down_trajectories.png"
    trajectory_chart(scene, results, chart)
    for result in results:
        result.pop("frames")

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.player_model,
        "environment_id": environment.id,
        "track": scene.name,
        "seed": scene.seed,
        "max_steps": args.max_steps,
        "max_calls_per_arm": args.max_calls,
        "predictive_wake_interval": args.wake_interval,
        "shared_contract": "first-person RGB plus physical scalar speed; no simulator geometry or pose",
        "direct_contract": (
            "model emits up to four one-tick key states; no visual sensor, generated controller, "
            "prediction, or skill library; last requested keys are held during model latency"
        ),
        "predictive_contract": (
            "RGB-derived road measurements, predicted response-time state, and reusable "
            "camera-feedback driving skills; no privileged telemetry beyond speed"
        ),
        "results": results,
        "trajectory_chart": str(chart),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}\nTrajectories: {chart}", flush=True)


if __name__ == "__main__":
    main()

"""Wall-clock comparison of direct actions and predictive visual skills."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from harness.models import Action, ActionName
from harness.policies import PredictiveVisualSkillPolicy
from harness.providers import ProviderError, plan_perspective_visual_actions
from harness.racing import CAR_RADIUS, RacingBackend, _distance_to_polyline
from harness.realtime import run_realtime_episode
from harness.rendering import ReplayBundle, ReplayMetadata
from harness.store import HarnessStore
from harness.view3d import ViewMode, render_policy_view
from run_3d_direct_vs_predictive import compact
from run_3d_pipeline_ab import load_dotenv, trajectory_chart


DEFAULT_STORE = Path(".harness-data/direct_3d_visual/compact-fixture-20260819T020117Z")


def _evaluate(scene, frames: list) -> tuple[float, int]:
    active = [
        frame for frame in frames
        if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    flags = [
        _distance_to_polyline(
            frame.privileged_state.player, scene.track_centerline, closed=True,
        ) <= scene.track_width / 2 - CAR_RADIUS
        for frame in active
    ]
    return round(sum(flags) / max(1, len(flags)), 3), sum(not flag for flag in flags)


def run_direct_wall(scene, *, max_steps: int, max_calls: int) -> dict:
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    frames: list = []
    queue: list[Action] = []
    previous_controls: list[dict] = []
    calls: list[dict] = []
    failures: list[dict] = []
    pending: Future | None = None
    submitted_tick = 0
    attempted_calls = active_ticks = stale_ticks = 0
    last_action = Action()
    next_tick_at = time.monotonic()
    started = next_tick_at

    while world.countdown_ticks_remaining > 0:
        frames.append(world.step(Action()))

    def call(frame, speed: float, controls: list[dict]):
        return plan_perspective_visual_actions(
            frame, previous_controls=controls, max_tokens=120,
            max_actions=4, speed_mps=speed, road_geometry=None,
        )

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="direct-player")
    try:
        while not world.terminated and active_ticks < max_steps:
            if pending is not None and pending.done():
                age = active_ticks - submitted_tick
                try:
                    plan, usage = pending.result()
                except ProviderError as error:
                    usage = getattr(error, "usage", None)
                    failures.append({
                        "attempt": attempted_calls, "tick": active_ticks,
                        "error": str(error), "usage_recorded": usage is not None,
                    })
                    if usage is not None:
                        calls.append({
                            "call": attempted_calls, "provider": usage.provider,
                            "model": usage.model, "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "latency_ms": usage.latency_ms,
                            "application_age_ticks": age, "model_actions": [],
                        })
                else:
                    queue = [
                        Action(name=ActionName(segment.action), keys=segment.keys)
                        for segment in plan.actions
                    ]
                    calls.append({
                        "call": attempted_calls, "provider": usage.provider,
                        "model": usage.model, "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "latency_ms": usage.latency_ms,
                        "application_age_ticks": age,
                        "model_actions": [
                            {"action": item.action, "keys": list(item.keys)}
                            for item in plan.actions
                        ],
                        "summary": plan.summary,
                    })
                stale_ticks += age
                pending = None

            if not queue and pending is None and attempted_calls < max_calls:
                attempted_calls += 1
                submitted_tick = active_ticks
                frame = render_policy_view(world, mode=ViewMode.FIRST_PERSON)
                pending = pool.submit(
                    call, frame, world.observe().speed, list(previous_controls),
                )

            action = queue.pop(0) if queue else last_action
            last_action = action
            frames.append(world.step(action))
            active_ticks += 1
            previous_controls.append({
                "action": action.name.value, "keys": list(action.keys),
            })
            previous_controls = previous_controls[-12:]
            next_tick_at += 1 / scene.dynamics.control_hz
            remaining = next_tick_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick_at = time.monotonic()
    finally:
        if pending is not None:
            pending.cancel()
        pool.shutdown(wait=True, cancel_futures=True)

    on_track, off_ticks = _evaluate(scene, frames)
    provider_ms = sum(item["latency_ms"] for item in calls)
    ages = [item["application_age_ticks"] for item in calls]
    input_tokens = sum(item["input_tokens"] for item in calls)
    output_tokens = sum(item["output_tokens"] for item in calls)
    return {
        "arm": "direct-actions-wall",
        "label": "Non-harnessed direct actions · wall clock",
        "completed": world.succeeded,
        "reason": world.reason or "step budget exhausted",
        "steps": active_ticks,
        "checkpoints_reached": world.objective_index,
        "checkpoint_total": len(scene.objectives),
        "model_calls": attempted_calls,
        "successful_model_calls": attempted_calls - len(failures),
        "failed_model_calls": len(failures),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "provider_latency_ms": provider_ms,
        "mean_call_latency_ms": round(provider_ms / max(1, len(calls))),
        "evaluation_wall_time_ms": round((time.monotonic() - started) * 1000),
        "game_latency_ticks": stale_ticks,
        "mean_application_age_ticks": round(sum(ages) / max(1, len(ages)), 2),
        "max_application_age_ticks": max(ages, default=0),
        "evaluator_on_track_fraction": on_track,
        "evaluator_off_track_ticks": off_ticks,
        "prediction_attempts": 0, "prediction_accepts": 0,
        "prediction_rejections": 0,
        "calls": calls, "provider_failures": failures,
        "controller_writes": [], "skill_activations": [],
        "frames": frames,
    }


def run_predictive_wall(scene, *, max_steps: int, max_calls: int) -> dict:
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    policy = PredictiveVisualSkillPolicy()
    started = time.monotonic()
    outcome = run_realtime_episode(
        world, policy, max_steps=max_steps, clock="wall",
        latency_ticks=policy.initial_latency_ticks,
        decision_budget=max_calls,
    )
    frames = outcome["frames"]
    on_track, off_ticks = _evaluate(scene, frames)
    calls = [item.model_dump(mode="json") for item in policy.turn_usages or []]
    provider_ms = sum(item["latency_ms"] for item in calls)
    realtime = outcome["realtime"]
    return {
        "arm": "predictive-skills-wall",
        "label": "Predictive overlap + skills · wall clock",
        "completed": outcome["succeeded"],
        "reason": outcome["reason"],
        "steps": realtime["ticks"],
        "checkpoints_reached": outcome["checkpoints_reached"],
        "checkpoint_total": len(scene.objectives),
        "model_calls": len(calls),
        "successful_model_calls": len(calls), "failed_model_calls": 0,
        "input_tokens": policy.input_tokens, "output_tokens": policy.output_tokens,
        "total_tokens": policy.input_tokens + policy.output_tokens,
        "provider_latency_ms": provider_ms,
        "mean_call_latency_ms": round(provider_ms / max(1, len(calls))),
        "evaluation_wall_time_ms": round((time.monotonic() - started) * 1000),
        "game_latency_ticks": sum(realtime.get("decision_ticks", [])),
        "mean_application_age_ticks": realtime["mean_decision_ticks"],
        "max_application_age_ticks": realtime["max_decision_ticks"],
        "evaluator_on_track_fraction": on_track,
        "evaluator_off_track_ticks": off_ticks,
        "prediction_attempts": outcome["policy_realtime"].get("prediction_attempts", 0),
        "prediction_accepts": outcome["policy_realtime"].get("prediction_accepts", 0),
        "prediction_rejections": outcome["policy_realtime"].get("prediction_rejections", 0),
        "calls": calls, "provider_failures": [],
        "controller_writes": [],
        "skill_activations": outcome["policy_realtime"].get("skill_activations", []),
        "realtime": realtime,
        "frames": frames,
    }


def save_replay(scene, result: dict, output: Path) -> None:
    path = output / f"{result['arm']}-replay.json"
    bundle = ReplayBundle.from_frames(
        scene, result["frames"], metadata=ReplayMetadata(
            run_id=result["arm"], policy_name=result["arm"],
            status="succeeded" if result["completed"] else "failed",
            seed=scene.seed,
            total_reward=round(sum(frame.reward for frame in result["frames"]), 4),
        ),
    )
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    result["replay"] = str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    os.environ.setdefault("OPENAI_REQUEST_TIMEOUT_SECONDS", "45")
    output = (args.output_dir or Path(".harness-data/3d_wallclock_eval") /
              datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environments = HarnessStore(args.fixture_store).list_environments()
    if not environments:
        raise SystemExit(f"No 3D environment in {args.fixture_store}")
    environment = environments[0]
    scene = environment.scene

    print("\n[Non-harnessed direct actions · real wall clock]", flush=True)
    direct = run_direct_wall(scene, max_steps=args.max_steps, max_calls=args.max_calls)
    print(json.dumps(compact(direct), indent=2), flush=True)
    print("\n[Predictive skills · real wall clock]", flush=True)
    predictive = run_predictive_wall(scene, max_steps=args.max_steps, max_calls=args.max_calls)
    print(json.dumps(compact(predictive), indent=2), flush=True)

    results = [direct, predictive]
    for result in results:
        save_replay(scene, result, output)
    chart = output / "top_down_trajectories.png"
    trajectory_chart(scene, results, chart)
    for result in results:
        result.pop("frames")
    summary = {
        "created_at": datetime.now(UTC).isoformat(), "clock": "wall",
        "control_hz": scene.dynamics.control_hz, "model": args.player_model,
        "environment_id": environment.id, "track": scene.name, "seed": scene.seed,
        "max_steps": args.max_steps, "max_calls_per_arm": args.max_calls,
        "results": results, "trajectory_chart": str(chart),
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {path}\nTrajectories: {chart}", flush=True)


if __name__ == "__main__":
    main()

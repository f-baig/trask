#!/usr/bin/env python3
"""Run a ten-lap camera-only, self-improving visual-skill experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from harness.collision import track_edge_points  # noqa: E402
from harness.meta_memory import (  # noqa: E402
    SkillOutcome, VisualSituation, VisualSkillMemory,
)
from harness.models import Action, EntityKind  # noqa: E402
from harness.pipeline2d import (  # noqa: E402
    ConeSkillDriver, prediction_matches,
)
from harness.providers import ProviderError, plan_cone_driving_skill  # noqa: E402
from harness.racing import (  # noqa: E402
    CAR_RADIUS, RacingBackend, _distance_to_polyline,
)
from harness.rendering import ReplayBundle, ReplayMetadata  # noqa: E402
from harness.store import HarnessStore  # noqa: E402


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def multilap_scene(scene, laps: int):
    """Repeat the certified one-lap objective sequence without changing geometry."""
    one_lap = scene.objectives[: scene.sector_count]
    objectives = [
        objective.model_copy(update={
            "description": f"Lap {lap}/{laps}: {objective.description.split(': ', 1)[-1]}"
        })
        for lap in range(1, laps + 1)
        for objective in one_lap
    ]
    return scene.model_copy(update={
        "id": f"{scene.id}-meta-{laps}lap",
        "name": f"{scene.name} · {laps}-lap meta trial",
        "prompt": f"{scene.prompt} [evaluator repeats the same certified circuit for {laps} laps]",
        "laps": laps,
        "objectives": objectives,
        # Opponent collisions are an unrelated source of terminal variance.  This
        # study measures within-player lap learning on identical geometry.
        "entities": [item for item in scene.entities if item.kind != EntityKind.NPC],
        "npc_behaviors": [],
    })


@dataclass
class ActiveSegment:
    situation: VisualSituation
    plan: object
    source: str
    ticks: int = 0
    contact_ticks: int = 0
    speed_sum: float = 0.0
    center_error_sum: float = 0.0

    def observe(self, state: dict) -> None:
        self.ticks += 1
        self.contact_ticks += int(bool(state["road_contact"]))
        self.speed_sum += float(state["speed"])
        self.center_error_sum += abs(float(state["center_near"]))

    def outcome(self) -> SkillOutcome | None:
        if self.ticks < 1:
            return None
        return SkillOutcome(
            situation=self.situation,
            plan=self.plan,
            ticks=self.ticks,
            road_contact_fraction=round(self.contact_ticks / self.ticks, 4),
            mean_speed=round(self.speed_sum / self.ticks, 4),
            mean_abs_center_error=round(self.center_error_sum / self.ticks, 4),
            source=self.source,
        )


def _usage_dict(usage, *, tick: int, lap: int, age_ticks: int) -> dict:
    return {
        "tick": tick,
        "evaluator_lap": lap,
        "provider": usage.provider,
        "model": usage.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "latency_ms": usage.latency_ms,
        "application_age_ticks": age_ticks,
    }


def run(
    scene, *, max_steps: int, max_model_calls: int, wake_interval: int,
    memory_warmup_ticks: int, aggression: float,
) -> dict:
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    driver = ConeSkillDriver(scene, aggression=aggression)
    memory = VisualSkillMemory()
    frames = []
    calls: list[dict] = []
    replays: list[dict] = []
    failures: list[dict] = []
    prediction_log: list[dict] = []
    segment_log: list[dict] = []
    lap_end_ticks: list[int] = []
    active_ticks = 0
    expected_latency_ticks = 18
    next_wake = wake_interval
    current_segment: ActiveSegment | None = None
    current_plan = None
    completed_laps = 0
    public_off_track_ticks = 0
    started = time.perf_counter()

    def remember_segment() -> None:
        nonlocal current_segment
        if current_segment is None:
            return
        outcome = current_segment.outcome()
        if outcome is not None:
            memory.add(outcome)
            segment_log.append({
                "id": outcome.id,
                "source": outcome.source,
                "skill": outcome.plan.skill,
                "target_speed": outcome.plan.target_speed,
                "ticks": outcome.ticks,
                "road_contact_fraction": outcome.road_contact_fraction,
                "mean_speed": outcome.mean_speed,
                "mean_abs_center_error": outcome.mean_abs_center_error,
                "safe": outcome.safe,
                "score": outcome.score,
            })
        current_segment = None

    def start_segment(plan, source: str, state: dict, frame) -> None:
        nonlocal current_segment, current_plan
        current_plan = plan
        current_segment = ActiveSegment(
            situation=VisualSituation.capture(state, frame), plan=plan, source=source,
        )

    def note_laps() -> None:
        nonlocal completed_laps
        now = min(scene.laps, world.objective_index // scene.sector_count)
        while completed_laps < now:
            completed_laps += 1
            lap_end_ticks.append(active_ticks)

    def drive(ticks: int) -> None:
        nonlocal active_ticks, public_off_track_ticks
        for _ in range(ticks):
            if world.terminated or active_ticks >= max_steps:
                return
            state, _ = driver.observe(world)
            if current_segment is not None:
                current_segment.observe(state)
            if not state["road_contact"]:
                public_off_track_ticks += 1
            action = driver.tick_state(state)
            frames.append(world.step(action))
            active_ticks += 1
            note_laps()

    # Author the first skill while the start grid is frozen, exactly as in the
    # existing latency evaluations.  No simulated race time is hidden here.
    state, frame = driver.observe(world)
    try:
        plan, usage = plan_cone_driving_skill(
            frame,
            public_state=state,
            active_skill=driver.active.__dict__,
            recent_controls=driver.recent_controls,
            activation_horizon_ticks=0,
            control_hz=scene.dynamics.control_hz,
            driving_aggression=aggression,
        )
        calls.append(_usage_dict(usage, tick=0, lap=1, age_ticks=0))
        driver.install(plan, 0, aggression=aggression)
        start_segment(plan, "model", state, frame)
    except ProviderError as error:
        return {"failure": str(error), "frames": frames}

    while world.countdown_ticks_remaining > 0:
        frames.append(world.step(Action()))

    while not world.terminated and active_ticks < max_steps:
        if active_ticks < next_wake:
            drive(1)
            continue

        state, frame = driver.observe(world)
        situation = VisualSituation.capture(state, frame)
        # A local visual resemblance during the first exploratory circuit is not
        # evidence that a situation has recurred.  A fixed camera-observation
        # warmup prevents the meta layer from collapsing a whole track into one
        # generic bend; it receives no lap/progress signal.
        reusable = memory.reusable(situation) if active_ticks >= memory_warmup_ticks else None
        if reusable is not None:
            remember_segment()
            replay = memory.tuned_replay(reusable, state)
            driver.install(replay, active_ticks, aggression=aggression)
            replays.append({
                "tick": active_ticks,
                "evaluator_lap": completed_laps + 1,
                "matched_outcome_id": reusable.outcome.id,
                "match_distance": reusable.distance,
                "skill": replay.skill,
                "prior_target_speed": reusable.outcome.plan.target_speed,
                "replay_target_speed": replay.target_speed,
            })
            start_segment(replay, "memory", state, frame)
            next_wake = active_ticks + wake_interval
            continue

        if len(calls) >= max_model_calls:
            next_wake = active_ticks + wake_interval
            continue

        submitted_tick = active_ticks
        retrieved = memory.retrieved_context(situation)
        try:
            plan, usage = plan_cone_driving_skill(
                frame,
                public_state=state,
                active_skill=driver.active.__dict__,
                recent_controls=driver.recent_controls,
                activation_horizon_ticks=expected_latency_ticks,
                control_hz=scene.dynamics.control_hz,
                driving_aggression=aggression,
                retrieved_experience=retrieved,
            )
        except ProviderError as error:
            failures.append({"tick": active_ticks, "error": str(error)})
            next_wake = active_ticks + wake_interval
            continue

        latency_ticks = math.ceil(max(0, usage.latency_ms) / (1_000 / scene.dynamics.control_hz))
        drive(latency_ticks)
        age = active_ticks - submitted_tick
        calls.append(_usage_dict(
            usage, tick=submitted_tick, lap=completed_laps + 1, age_ticks=age,
        ))
        expected_latency_ticks = max(1, round(
            sum(item["application_age_ticks"] for item in calls[1:]) / max(1, len(calls) - 1)
        ))
        if world.terminated or active_ticks >= max_steps:
            break
        actual, activation_frame = driver.observe(world)
        accepted, diagnostic = prediction_matches(plan, actual)
        diagnostic.update({
            "submitted_tick": submitted_tick,
            "landed_tick": active_ticks,
            "retrieved_records": len(retrieved),
        })
        prediction_log.append(diagnostic)
        remember_segment()
        if accepted:
            driver.install(plan, active_ticks, aggression=aggression)
            start_segment(plan, "model", actual, activation_frame)
        elif current_plan is not None:
            # Keep executing the old model-authored skill, but begin a new outcome
            # window from the now-current camera state.
            start_segment(current_plan, "continued-after-rejection", actual, activation_frame)
        next_wake = active_ticks + wake_interval

    remember_segment()
    note_laps()
    elapsed_ms = round((time.perf_counter() - started) * 1_000)
    active_frames = [
        frame for frame in frames if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    evaluator_on_track = [
        _distance_to_polyline(frame.privileged_state.player, scene.track_centerline, closed=True)
        <= scene.track_width / 2 - CAR_RADIUS
        for frame in active_frames
    ]
    lap_starts = [0, *lap_end_ticks[:-1]]
    lap_metrics = []
    for index, (start, end) in enumerate(zip(lap_starts, lap_end_ticks), start=1):
        model_calls = sum(start <= item["tick"] < end for item in calls)
        memory_replays = sum(start <= item["tick"] < end for item in replays)
        interval = evaluator_on_track[start:end]
        lap_metrics.append({
            "lap": index,
            "start_tick": start,
            "end_tick": end,
            "lap_ticks": end - start,
            "lap_seconds": round((end - start) / scene.dynamics.control_hz, 2),
            "model_calls": model_calls,
            "memory_replays": memory_replays,
            "evaluator_on_track_fraction": round(sum(interval) / max(1, len(interval)), 4),
        })
    return {
        "completed": world.succeeded,
        "reason": world.reason or "step budget exhausted",
        "laps_completed": completed_laps,
        "steps": active_ticks,
        "model_calls": len(calls),
        "memory_replays": len(replays),
        "input_tokens": sum(item["input_tokens"] for item in calls),
        "output_tokens": sum(item["output_tokens"] for item in calls),
        "total_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in calls),
        "provider_latency_ms": sum(item["latency_ms"] for item in calls),
        "evaluation_wall_time_ms": elapsed_ms,
        "public_camera_off_track_ticks": public_off_track_ticks,
        "evaluator_on_track_fraction": round(sum(evaluator_on_track) / max(1, len(evaluator_on_track)), 4),
        "calls": calls,
        "memory_replay_log": replays,
        "model_failures": failures,
        "prediction_log": prediction_log,
        "memory_outcomes": segment_log,
        "lap_metrics": lap_metrics,
        "frames": frames,
    }


def charts(scene, result: dict, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 7.7), layout="constrained")
    figure.patch.set_facecolor("#101619")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.18, 1), height_ratios=(1, 1))
    route = figure.add_subplot(grid[:, 0])
    pace = figure.add_subplot(grid[0, 1])
    decisions = figure.add_subplot(grid[1, 1])
    for axis in (route, pace, decisions):
        axis.set_facecolor("#182124")
        axis.tick_params(colors="#aebabc", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#526164")

    left, right = track_edge_points(scene)
    for edge in (left, right):
        closed = [*edge, edge[0]]
        route.plot([p[0] for p in closed], [p[1] for p in closed], color="#d9ded8", linewidth=1.5)
    center = [(point.x, point.y) for point in scene.track_centerline]
    closed_center = [*center, center[0]]
    route.plot(
        [p[0] for p in closed_center], [p[1] for p in closed_center],
        color="#748184", linewidth=.8, linestyle=(0, (5, 7)),
    )
    trace = [
        (frame.privileged_state.player.x, frame.privileged_state.player.y)
        for frame in result["frames"] if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    if trace:
        route.plot([p[0] for p in trace], [p[1] for p in trace], color="#82cfff", linewidth=2.0, alpha=.74)
        route.scatter(*trace[0], color="#77dd9b", s=52, edgecolor="#101619", zorder=5)
        route.scatter(*trace[-1], color="#ff6b6b", s=52, edgecolor="#101619", zorder=5)
    route.set_title(
        f"{scene.laps}-lap trajectory · {result['laps_completed']}/{scene.laps} complete",
        color="#f4f0e5", fontsize=13,
    )
    route.set_aspect("equal")
    route.invert_yaxis()

    laps = [item["lap"] for item in result["lap_metrics"]]
    ticks = [item["lap_ticks"] for item in result["lap_metrics"]]
    pace.plot(laps, ticks, color="#ffbf59", marker="o", linewidth=2.4)
    if ticks:
        pace.axhline(ticks[0], color="#748184", linewidth=1, linestyle="--")
    pace.set_title("Per-lap pace (lower is faster)", color="#f4f0e5", fontsize=12)
    pace.set_xlabel("lap", color="#aebabc")
    pace.set_ylabel("simulator ticks", color="#aebabc")
    pace.set_xticks(laps)

    model = [item["model_calls"] for item in result["lap_metrics"]]
    replay = [item["memory_replays"] for item in result["lap_metrics"]]
    decisions.bar(laps, model, color="#f28e72", label="model calls")
    decisions.bar(laps, replay, bottom=model, color="#bd93f9", label="memory skill replays")
    decisions.set_title("Progressive transfer from model to learned sequence", color="#f4f0e5", fontsize=12)
    decisions.set_xlabel("lap", color="#aebabc")
    decisions.set_ylabel("decisions", color="#aebabc")
    decisions.set_xticks(laps)
    decisions.legend(facecolor="#182124", edgecolor="#526164", labelcolor="#f4f0e5")

    figure.suptitle("Camera-only multi-lap meta harness", color="#f4f0e5", fontsize=18)
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-store", type=Path,
        default=Path(".harness-data/player_reflex_ab/20260818T163834Z"),
    )
    parser.add_argument("--environment-id", default="race-a0a597c744-1232ec")
    parser.add_argument("--laps", type=int, default=10)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--max-model-calls", type=int, default=24)
    parser.add_argument("--wake-interval", type=int, default=70)
    parser.add_argument("--memory-warmup-ticks", type=int, default=500)
    parser.add_argument("--aggression", type=float, default=.78)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    os.environ.setdefault("OPENAI_REQUEST_TIMEOUT_SECONDS", "45")

    environment = HarnessStore(args.fixture_store).get_environment(args.environment_id)
    if environment is None:
        raise SystemExit(f"Unknown environment {args.environment_id}")
    scene = multilap_scene(environment.scene, args.laps)
    output = (args.output_dir or Path("artifacts/multilap-meta-harness") /
              datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {args.laps} laps on {environment.scene.name} with {args.player_model}; "
        "player contract is forward-cone RGB + exposed speed only.",
        flush=True,
    )
    result = run(
        scene, max_steps=args.max_steps,
        max_model_calls=args.max_model_calls, wake_interval=args.wake_interval,
        memory_warmup_ticks=args.memory_warmup_ticks,
        aggression=args.aggression,
    )
    if "failure" in result:
        raise SystemExit(result["failure"])

    replay_path = output / f"meta-{args.laps}lap-replay.json"
    replay = ReplayBundle.from_frames(
        scene, result["frames"],
        metadata=ReplayMetadata(
            run_id=f"meta-{args.laps}lap", policy_name="camera-meta-predictive-skills",
            status="succeeded" if result["completed"] else "failed",
            seed=scene.seed,
            total_reward=round(sum(frame.reward for frame in result["frames"]), 4),
        ),
    )
    replay_path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
    chart_path = output / f"meta_harness_{args.laps}lap.png"
    charts(scene, result, chart_path)
    result["replay"] = str(replay_path)
    result.pop("frames")
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "environment_id": environment.id,
        "track": environment.scene.name,
        "model": args.player_model,
        "agent_contract": (
            "forward-cone RGB plus exposed physical speed; memory keys and outcomes use only "
            "camera-derived public state, image fingerprint, speed, and prior model-authored skills"
        ),
        "evaluator_only": "lap count/times, checkpoints, top-down trajectory, and geometric road-contact audit",
        "result": result,
        "chart": str(chart_path),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "completed", "reason", "laps_completed", "steps", "model_calls",
            "memory_replays", "total_tokens", "provider_latency_ms",
            "public_camera_off_track_ticks", "evaluator_on_track_fraction",
        )
    }, indent=2), flush=True)
    print("Per-lap:", json.dumps(result["lap_metrics"], indent=2), flush=True)
    print(f"Summary: {summary_path}\nChart: {chart_path}\nReplay: {replay_path}", flush=True)


if __name__ == "__main__":
    main()

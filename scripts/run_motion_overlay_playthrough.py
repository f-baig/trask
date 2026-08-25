#!/usr/bin/env python3
"""Drive one compiled circuit with Haiku, with and without the motion overlay.

Both arms race the same environment record, from the same seed, through the same
engine, with the same model. The only difference is whether the frame the policy
sees carries a measured optical-flow field. That is the whole point: a paired
playthrough is the cheapest thing that can tell the difference between a vision
tool that helps and one that merely renders.

The circuit comes from the offline grammar rather than a creator model, so the
scene is free, identical between arms, and identical between invocations. Progress
prints as it goes because a live episode is hundreds of model calls and several
minutes of wall clock, and a silent command looks broken rather than busy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from harness.models import Action, RunRequest
from harness.policies import AnthropicRacingPolicy
from harness.policy_protocol import VisualFrame
from harness.realtime import run_realtime_episode
from harness.service import HarnessService
from harness.store import HarnessStore


ARMS = ("overlay", "plain")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class NarratedRacingPolicy(AnthropicRacingPolicy):
    """The stock Haiku driver, with a progress hook on every executed tick."""

    on_tick: Callable[[int, Action, object, "NarratedRacingPolicy"], None] | None = None
    ticks: int = 0

    def act_visual(self, observation, frame: VisualFrame):
        action, decision = super().act_visual(observation, frame)
        self.ticks += 1
        if self.on_tick is not None:
            self.on_tick(self.ticks, action, decision, self)
        return action, decision


def _mean_racing_speed(frames) -> float:
    """Mean speed over green-flag ticks only; the countdown would dilute it."""
    racing = [
        frame.privileged_state.speed for frame in frames
        if frame.privileged_state.countdown_ticks_remaining == 0
    ]
    return sum(racing) / len(racing) if racing else 0.0


def collect_milestones(frames) -> list[str]:
    """Sector crossings, laps, excursions and the terminal reason, in order."""
    interesting = ("crossed", "lap ", "race completed", "left track", "returned to track")
    return [
        f"tick {frame.step}: {event}"
        for frame in frames for event in frame.events
        if event.startswith(interesting)
    ]


def oracle_reference(service: HarnessService, environment_id: str, max_steps: int) -> dict:
    """How long the deterministic oracle needs, as the scale for everything else."""
    run = service.run(RunRequest(
        environment_id=environment_id, policy_name="oracle-racing-line", max_steps=max_steps,
    ))
    return {
        "status": run.status.value, "reason": run.result_reason,
        "steps": len(run.frames), "reward": round(run.total_reward, 2),
    }


def drive_realtime(
    service: HarnessService, environment_id: str, arm: str, view: str,
    max_steps: int, report_every: int, clock: str, latency_ticks: int, starve: str,
    pipeline_depth: int, terse: bool,
) -> dict:
    """Drive with the planner off the critical path, so latency costs ticks.

    This bypasses `service.run` and therefore produces no replay bundle: the
    synchronous path owns the durable record, and this one owns the latency report.
    """
    _configure_arm(arm, view)
    started = time.monotonic()

    def progress(tick: int, action: Action, report) -> None:
        if tick % report_every:
            return
        print(
            f"    tick {tick:4d}  {action.name.value:8s}  decisions={report.decisions:3d}"
            f"  fresh={report.fresh_input_fraction:5.3f}  age={report.mean_decision_ticks:4.1f}t"
            f"  {time.monotonic() - started:5.0f}s",
            flush=True,
        )

    policy = NarratedRacingPolicy(terse=terse)
    environment = service.store.get_environment(environment_id)
    assert environment is not None
    world = service.runtime.create(environment.scene)
    print(
        f"  [{arm}] driving ({view}, overlay={'on' if arm == 'overlay' else 'off'}, "
        f"clock={clock}, depth={pipeline_depth}, terse={terse})", flush=True,
    )
    outcome = run_realtime_episode(
        world, policy, max_steps=max_steps, clock=clock,
        latency_ticks=latency_ticks, starve=starve,
        pipeline_depth=pipeline_depth, progress=progress,
    )
    frames = outcome.pop("frames")
    result = {
        "arm": arm, "view": view, "overlay": arm == "overlay",
        "scheduler": "realtime",
        "status": "succeeded" if outcome["succeeded"] else "failed" if outcome["terminated"] else "timeout",
        "planner_calls": policy.planning_turns,
        "input_tokens": policy.input_tokens,
        "output_tokens": policy.output_tokens,
        "wall_seconds": round(time.monotonic() - started, 1),
        "flow_pairs_measured": (
            None if policy.motion_overlay is None else policy.motion_overlay.pairs_measured
        ),
        "off_track_excursions": sum(
            1 for frame in frames for event in frame.events if event.startswith("left track")
        ),
        "top_speed": round(max((f.privileged_state.speed for f in frames), default=0.0), 2),
        "mean_speed": round(_mean_racing_speed(frames), 2),
        "milestones": collect_milestones(frames),
        **outcome,
    }
    print(
        f"  [{arm}] {result['status']}: {outcome['reason']} — {outcome['steps']} ticks, "
        f"{policy.planning_turns} calls, fresh input {outcome['realtime']['fresh_input_fraction']}, "
        f"{result['wall_seconds']:.0f}s",
        flush=True,
    )
    return result


def _configure_arm(arm: str, view: str) -> None:
    if arm == "overlay":
        os.environ["RACING_MOTION_OVERLAY"] = "1"
    else:
        os.environ.pop("RACING_MOTION_OVERLAY", None)
    os.environ["RACING_POLICY_VIEW"] = view


def drive(
    service: HarnessService, environment_id: str, arm: str, view: str,
    max_steps: int, decision_budget: int | None, report_every: int,
) -> dict:
    _configure_arm(arm, view)
    started = time.monotonic()
    print(f"  [{arm}] driving ({view}, overlay={'on' if arm == 'overlay' else 'off'})", flush=True)

    def progress(tick: int, action: Action, decision, policy: NarratedRacingPolicy) -> None:
        if tick % report_every:
            return
        overlay = policy.motion_overlay
        elapsed = time.monotonic() - started
        print(
            f"    tick {tick:4d}  {action.name.value:8s}  calls={policy.planning_turns:3d}"
            f"  out={policy.output_tokens:6d}  flow_pairs={0 if overlay is None else overlay.pairs_measured:4d}"
            f"  {elapsed:5.0f}s  | {decision.summary[22:110]}",
            flush=True,
        )

    policy = NarratedRacingPolicy(on_tick=progress)
    service.policies["telemetry-direct"] = policy
    run = service.run(RunRequest(
        environment_id=environment_id, policy_name="telemetry-direct",
        max_steps=max_steps, policy_decision_budget=decision_budget,
    ))
    wall_seconds = time.monotonic() - started
    overlay = policy.motion_overlay
    result = {
        "arm": arm,
        "view": view,
        "overlay": arm == "overlay",
        "run_id": run.id,
        "status": run.status.value,
        "reason": run.result_reason,
        "steps": len(run.frames),
        "reward": round(run.total_reward, 3),
        "planner_calls": run.player_turns,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cache_read_input_tokens": run.cache_read_input_tokens,
        "cache_creation_input_tokens": run.cache_creation_input_tokens,
        "uncached_input_tokens": run.uncached_input_tokens,
        "model_latency_ms": run.latency_ms,
        "wall_seconds": round(wall_seconds, 1),
        "flow_pairs_measured": None if overlay is None else overlay.pairs_measured,
        # The engine reports leaving and rejoining the road as events rather than as
        # a per-tick flag, so excursions are the countable quantity.
        "off_track_excursions": sum(
            1 for frame in run.frames for event in frame.events
            if event.startswith("left track")
        ),
        "checkpoints_reached": max(
            (frame.privileged_state.objective_index for frame in run.frames), default=0,
        ),
        "top_speed": round(max(
            (frame.privileged_state.speed for frame in run.frames), default=0.0,
        ), 2),
        "mean_speed": round(_mean_racing_speed(run.frames), 2),
        "replay_uri": run.artifacts[0].uri if run.artifacts else None,
        "milestones": collect_milestones(run.frames),
    }
    print(
        f"  [{arm}] {run.status.value}: {run.result_reason} — {len(run.frames)} ticks, "
        f"{run.player_turns} calls, {run.output_tokens} out, {wall_seconds:.0f}s",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="A technical asphalt circuit with two barriers and one opponent.",
        help="Compiled offline, so the same brief always yields the same circuit",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--view", default="forward-cone", choices=("forward-cone", "overhead"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument(
        "--decision-budget", type=int,
        help="Cap planner calls; omit for the default of one call per tick",
    )
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument(
        "--clock", default="sync", choices=("sync", "measured", "wall", "fixed"),
        help="sync stops the world while the model thinks; the others charge latency in ticks",
    )
    parser.add_argument(
        "--latency-ticks", type=int, default=3,
        help="Ticks every decision costs under the fixed clock",
    )
    parser.add_argument("--starve", default="hold", choices=("hold", "coast"))
    parser.add_argument(
        "--pipeline-depth", type=int, default=1,
        help="Concurrent decisions in flight; above 1 requires --clock wall",
    )
    parser.add_argument(
        "--terse", action="store_true",
        help="Drop subgoal/summary prose, which is most of a decision's output tokens",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required for a live playthrough")
    os.environ["ANTHROPIC_PLAYER_MODEL"] = args.model
    os.environ["ANTHROPIC_INTERRUPT_MODEL"] = args.model

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path(".harness-data") / "playthroughs" / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    service = HarnessService(store=HarnessStore(output_dir))

    # The circuit is compiled offline so both arms inherit one identical scene and no
    # creator-model variance leaks into a driver comparison.
    environment = service.create_environment(
        args.prompt, seed=args.seed, provider="offline",
        origin="motion-overlay playthrough",
    )
    scene = environment.scene
    print(
        f"circuit: {scene.name} · {scene.surface} · {scene.laps} lap(s) · "
        f"{scene.sector_count} sectors · seed {scene.seed}",
        flush=True,
    )
    reference = oracle_reference(service, environment.id, args.max_steps)
    print(f"oracle reference: {reference['status']} in {reference['steps']} ticks", flush=True)

    results = [
        drive(
            service, environment.id, arm, args.view, args.max_steps,
            args.decision_budget, args.report_every,
        )
        if args.clock == "sync" else
        drive_realtime(
            service, environment.id, arm, args.view, args.max_steps,
            args.report_every, args.clock, args.latency_ticks, args.starve,
            args.pipeline_depth, args.terse,
        )
        for arm in args.arms
    ]

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "prompt": args.prompt,
        "seed": args.seed,
        "view": args.view,
        "player_model": args.model,
        "max_steps": args.max_steps,
        "decision_budget": args.decision_budget,
        "clock": args.clock,
        "latency_ticks": args.latency_ticks if args.clock == "fixed" else None,
        "starve": args.starve if args.clock != "sync" else None,
        "pipeline_depth": args.pipeline_depth,
        "terse": args.terse,
        "circuit": {
            "name": scene.name, "surface": scene.surface, "laps": scene.laps,
            "sectors": scene.sector_count, "track_width": scene.track_width,
            "environment_id": environment.id,
        },
        "oracle": reference,
        "arms": results,
    }
    (output_dir / "playthrough.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {output_dir / 'playthrough.json'}", flush=True)
    for row in results:
        realtime = row.get("realtime")
        print(
            f"  {row['arm']:8s} {row['status']:9s} {row['steps']:4d} ticks  "
            f"cp {row['checkpoints_reached']}/{summary['circuit']['sectors']}  "
            f"excursions {row['off_track_excursions']:2d}  "
            f"{row['planner_calls']:3d} calls  {row['output_tokens']:6d} out  "
            + (f"fresh {realtime['fresh_input_fraction']}  " if realtime else "")
            + str(row["reason"]),
            flush=True,
        )


if __name__ == "__main__":
    main()

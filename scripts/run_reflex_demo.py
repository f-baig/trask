#!/usr/bin/env python3
"""Drive a generated circuit with an agent-authored reflex controller.

Claude does not act every tick. It writes a controller, rehearses it against a fork of
the deterministic world, sets a target and the conditions that should wake it, and hands
control back. The harness runs the controller and decides when reasoning is worth a call.

    scripts/run_reflex_demo.py --model claude-sonnet-5
    scripts/run_reflex_demo.py --prompt "narrow slippery ice circuit with three hairpins"
    scripts/run_reflex_demo.py --baseline    # the fixture controller, no model calls

The baseline arm matters and is not optional: a reflex controller that finishes a lap
proves nothing on its own, because a hand-written lane keeper finishes a lap too. The only
question a reflex result can answer is whether what Claude wrote, and how it revised it,
beats a fixed controller nobody tuned for this circuit.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from harness.models import Action
from harness.racing import RacingWorld, compile_racing_scene
from harness.racing3d import Racing3DWorld
from harness.reflex.episode import EpisodeReport, replay_bundle, run_reflex_episode
from harness.reflex.runtime import ReflexRuntime
from harness.track_grammar import parse_track_prompt


BASELINE_SOURCE = """
def control(sense, ctrl, out):
    lane_error = sense.lane - ctrl.p.target_lane
    steer = ctrl.pid("lane", -lane_error, kp=ctrl.p.kp, kd=ctrl.p.kd)
    heading = ctrl.pid("heading", sense.heading_error / 45.0, kp=0.7)
    out.discretizer("hysteresis")
    out.steer(steer + heading)
    out.throttle(ctrl.pid("speed", ctrl.p.target_speed - sense.speed, kp=0.6))
"""
BASELINE_READS = ["lane", "heading_error", "speed"]
BASELINE_PARAMS = {"target_lane": 0.0, "kp": 0.9, "kd": 0.08, "target_speed": 3.0}


def load_dotenv(path: Path = Path(".env")) -> None:
    if os.environ.get("ANTHROPIC_API_KEY") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_world(prompt: str, seed: int, dimensions: str):
    scene = compile_racing_scene(prompt, parse_track_prompt(prompt), seed=seed)
    if dimensions == "3d":
        return Racing3DWorld.from_scene(scene)
    return RacingWorld.from_scene(scene)


def run_baseline(world, max_steps: int) -> tuple[dict, EpisodeReport]:
    """The fixture controller, installed once, never revised, no model in the loop."""
    runtime = ReflexRuntime(world.scene)
    installed = runtime.install(
        name="baseline", source=BASELINE_SOURCE, reads=BASELINE_READS,
        params=dict(BASELINE_PARAMS), safe_action={"steer": "hold", "throttle": -0.6},
    )
    assert installed["installed"], installed
    runtime.set_conditions([], deadline_ticks=10**6)
    report = EpisodeReport(field_size=world.field_size)
    while not world.terminated and report.ticks < max_steps:
        if world.countdown_ticks_remaining > 0:
            report.frames.append(world.step(Action()))
            continue
        report.frames.append(world.step(runtime.tick(world, world.observe())))
        runtime.clear_wake()
        report.ticks += 1
    ticks = report.ticks
    report.succeeded = world.succeeded
    return {
        "arm": "baseline",
        "succeeded": world.succeeded,
        "reason": world.reason or "step budget exhausted",
        "ticks": ticks,
        "checkpoints_reached": world.objective_index,
        "position": world.player_position,
        "field_size": world.field_size,
        "model_calls": 0,
        "runtime": runtime.summary(),
    }, report


def write_bundle(scene, episode: EpisodeReport, directory: Path, run_id: str, policy: str) -> Path:
    """Export the run as the bundle `python -m harness.native_viewer` already reads."""
    bundle = replay_bundle(scene, episode, run_id=run_id, policy_name=policy)
    return bundle.write_json(directory / f"{run_id}.json")


def watch(path: Path) -> None:
    """Open one bundle in the desktop viewer, blocking until the window closes.

    Launched as a separate process the way `service.launch_native_viewer` does it, because
    the viewer owns an SDL window and initializing pygame inside a script that has already
    rendered frames is how you get two conflicting SDL contexts.
    """
    import subprocess
    import sys

    environment = os.environ.copy()
    environment.pop("SDL_VIDEODRIVER", None)
    subprocess.run(
        [sys.executable, "-m", "harness.native_viewer", "--bundle", str(path)],
        check=False, env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="A technical asphalt circuit with two barriers and one opponent.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_PLAYER_MODEL", "claude-sonnet-5"))
    parser.add_argument("--dimensions", choices=("2d", "3d"), default="2d")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--max-wakes", type=int, default=12)
    parser.add_argument(
        "--latency", choices=("measured", "zero", "fixed"), default="measured",
        help="How many ticks a decision costs. 'zero' is the clean architectural baseline.",
    )
    parser.add_argument("--latency-ticks", type=int, default=12, help="Ticks per decision under --latency fixed.")
    parser.add_argument(
        "--rehearsal-budget", type=int, default=5,
        help="try_controller calls allowed per wake. Small on purpose: they should be spent lowering the lap time.",
    )
    parser.add_argument("--baseline", action="store_true", help="Run the fixture controller with no model calls.")
    parser.add_argument("--both", action="store_true", help="Run the baseline arm and then the agent arm.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(".harness-data/reflex-demo.json"))
    parser.add_argument(
        "--watch", action="store_true",
        help="Open each finished run in the native replay viewer (one window per arm, in order).",
    )
    parser.add_argument(
        "--replay-dir", type=Path, default=Path(".harness-data/replays"),
        help="Where the viewer bundles are written. Always written, watched or not.",
    )
    args = parser.parse_args()
    load_dotenv()
    bundles: list[tuple[str, Path]] = []

    scene = compile_racing_scene(args.prompt, parse_track_prompt(args.prompt), seed=args.seed)
    print(f"circuit: {scene.name} — {scene.surface} at {scene.grip:.2f}x grip, "
          f"{scene.laps} lap(s), corridor {scene.track_width:.0f}px, "
          f"{len(scene.npc_behaviors)} opponent(s), {scene.sector_count} gates", flush=True)

    results = []
    if args.baseline or args.both:
        print("\n--- baseline: fixture controller, zero model calls ---", flush=True)
        report, episode = run_baseline(
            build_world(args.prompt, args.seed, args.dimensions), args.max_steps,
        )
        results.append(report)
        print(f"  {report['reason']} in {report['ticks']} ticks "
              f"(P{report['position']} of {report['field_size']})", flush=True)
        bundles.append(("baseline", write_bundle(
            scene, episode, args.replay_dir, f"reflex-baseline-{args.seed}", "reflex-baseline",
        )))

    if not args.baseline or args.both:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is required for the agent arm; use --baseline without it.")
        print(f"\n--- agent: {args.model} writes its own controller ---", flush=True)
        world = build_world(args.prompt, args.seed, args.dimensions)
        report = run_reflex_episode(
            world, model=args.model, max_steps=args.max_steps, max_wakes=args.max_wakes,
            latency=args.latency, latency_ticks=args.latency_ticks, verbose=not args.quiet,
            rehearsal_budget=args.rehearsal_budget,
        )
        payload = {"arm": "agent", "model": args.model, **report.as_dict()}
        results.append(payload)
        bundles.append(("agent", write_bundle(
            scene, report, args.replay_dir, f"reflex-agent-{args.seed}", f"reflex-{args.model}",
        )))
        print(f"\n  {payload['reason']} in {payload['ticks']} ticks "
              f"(P{payload['position']} of {payload['field_size']})", flush=True)
        print(f"  {payload['wakes']} wakes, {payload['model_calls']} model calls, "
              f"{payload['ticks_per_wake']} ticks per wake, "
              f"{payload['stale_ticks']} ticks on a stale controller", flush=True)
        print(f"  wake causes: {payload['runtime']['wake_causes']}", flush=True)
        print(f"  tokens: {payload['input_tokens']} in "
              f"({payload['cache_read_input_tokens']} cache reads) / "
              f"{payload['output_tokens']} out, {payload['latency_ms'] / 1000:.1f}s of model time",
              flush=True)
        for turn in payload["turns"]:
            tools = ", ".join(call["name"] for call in turn["tools"]) or "(no tools)"
            print(f"    wake @{turn['tick']:>4} {','.join(turn['woke_because']) or 'first':<28} {tools}", flush=True)
        rehearsals = payload["runtime"].get("rehearsals") or []
        if rehearsals:
            print("\n  rehearsal ladder (what the agent was optimizing against):", flush=True)
            for attempt in rehearsals:
                lap = attempt["lap_time_ticks"]
                verdict = "no finish" if lap is None else (
                    f"{lap} ticks"
                    + ("  <- new best" if attempt["previous_best"] is None or lap < attempt["previous_best"] else "")
                )
                print(f"    @{attempt['at_tick']:>4} {attempt['controller']:<16} {verdict}", flush=True)
            print(f"  best rehearsed lap: {payload['runtime'].get('best_rehearsed_lap_ticks')} ticks "
                  f"-> actual race: {payload['ticks']} ticks", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "created_at": datetime.now(UTC).isoformat(),
        "prompt": args.prompt, "seed": args.seed, "dimensions": args.dimensions,
        "latency": args.latency, "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)
    for arm, path in bundles:
        print(f"replay ({arm}): {path}", flush=True)
    if bundles and not args.watch:
        print("\nwatch one with:\n  PYTHONPATH=backend .venv/bin/python -m harness.native_viewer "
              f"--bundle {bundles[-1][1]}", flush=True)
    for arm, path in bundles:
        if args.watch:
            print(f"\nopening the {arm} replay — space pauses, arrows step, Q closes", flush=True)
            watch(path)


if __name__ == "__main__":
    main()

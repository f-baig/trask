from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import ElevationProfile, ElevationSpec, ExperimentRequest, NpcProfile, RunRequest
from .policies import built_in_policies, canonical_policy_name
from .racing import RacingDesignDraft, compile_certified_scene
from .racing3d import compile_racing_3d_scene
from .track3d import compile_track_surface
from .track_grammar import parse_track_prompt
from .vehicle_physics import apply_dynamics_preset
from .view3d import DEFAULT_ROAD_DETAIL, ViewMode
from .service import HarnessService


def _plan_from_arguments(args) -> tuple[object, str]:
    """Resolve a track plan from either a brief or the archetype flags."""
    if args.prompt:
        plan = parse_track_prompt(args.prompt)
        if getattr(args, "edge_barriers", False):
            plan = plan.model_copy(update={"edge_barriers": True})
        return plan, args.prompt
    plan = RacingDesignDraft(
        title=f"Manual {args.circuit.title()} Circuit",
        rationale="A local human-playable circuit compiled without a model call.",
        circuit=args.circuit,
        surface=args.surface,
        obstacle_count=args.obstacles,
        edge_barriers=getattr(args, "edge_barriers", False),
        npc_count=args.npcs,
        laps=args.laps,
        npc_start_mode=args.npc_start,
        grip=args.grip,
        npc_profile=NpcProfile(args.npc_profile),
    ).to_plan()
    overrides = {
        key: value for key, value in (
            ("intelligence", getattr(args, "npc_intelligence", None)),
            ("aggression", getattr(args, "npc_aggression", None)),
        ) if value is not None
    }
    if overrides:
        plan = plan.model_copy(update={
            "npcs": [npc.model_copy(update=overrides) for npc in plan.npcs],
        })
    return plan, "manual local play"


def _describe_circuit(scene, certificate, notes: list[str]) -> None:
    """Print what the brief compiled into before the window opens."""
    report = scene.track_report
    print(f"{scene.name} — {scene.surface} at {scene.grip:.2f}x grip, {scene.laps} lap(s), "
          f"{scene.sector_count} sector gates, {scene.track_width:.0f}px wide")
    if report is not None:
        print(f"  {report.length_pixels:.0f}px {report.direction} circuit · longest straight "
              f"{report.longest_straight_pixels:.0f}px · tightest radius {report.minimum_radius_pixels:.0f}px "
              f"· angle error {report.angle_fidelity_degrees:.1f}° · region fidelity "
              f"{report.region_fidelity * 100:.0f}%")
        for corner in report.corners:
            requested = (
                f"asked {corner.requested_angle_degrees:.0f}°"
                if corner.requested_angle_degrees is not None else "solved for closure"
            )
            print(f"  corner {corner.index + 1}: {corner.achieved_angle_degrees:.0f}° "
                  f"{corner.direction} in {corner.achieved_region.value} at "
                  f"{corner.entry_progress_percent:.0f}% ({requested}, R={corner.achieved_radius_pixels:.0f}px, "
                  f"entry {corner.recommended_entry_speed:.1f}px/tick)")
        for note in [*report.relaxations, *notes]:
            print(f"  relaxed: {note}")
    if scene.elevation is not None:
        surface = compile_track_surface(scene)
        print(f"  elevation: {scene.elevation.profile.value}, "
              f"{scene.elevation.amplitude_m:.1f} m over {scene.elevation.hill_count} crest(s) "
              f"at {scene.elevation.crest_sharpness:.2f} sharpness "
              f"-> {surface.relief_pixels:.0f}px relief, steepest climb "
              f"{surface.steepest_grade_degrees:.1f}deg, banking up to "
              f"{surface.steepest_bank_degrees:.1f}deg")
    for behavior in scene.npc_behaviors:
        print(f"  {behavior.entity_id}: {behavior.profile.value} "
              f"(pace {behavior.pace:.2f}, skill {behavior.skill:.2f}, "
              f"intelligence {behavior.intelligence:.2f}, "
              f"aggression {behavior.aggression:.2f}"
              f"{', defends' if behavior.defends else ''})")
    print(f"  {certificate.verifier} certified in {certificate.route_steps} control ticks")


def _load_local_env(path: Path = Path(".env")) -> None:
    """Adopt the project's local credentials the way `make api` does.

    Without this a base-model comparison run from a shell would silently fall back
    to the offline generator and compare the harness against itself.
    """
    import os

    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _play_ab(args) -> None:
    """Compile one brief with each arm, then hand both circuits to the driver."""
    import os
    from datetime import UTC, datetime

    from .play_ab import build_pair, presentation_order, scorecard_lines, write_scenes
    from .vehicle_physics import apply_dynamics_preset

    _load_local_env()
    provider = args.provider
    if provider == "auto" and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        provider = "offline"
    if provider == "offline":
        print("No API key in scope: every arm falls back to the deterministic offline\n"
              "generator, so the arms differ only by the harness's local search. Set\n"
              "OPENAI_API_KEY or ANTHROPIC_API_KEY for a real base-model comparison.\n")

    print(f"Brief: {args.prompt}\nSeed: {args.seed} · arms: {', '.join(args.arms)}\n")
    pair = build_pair(
        args.prompt, args.seed, arms=tuple(args.arms),
        candidates=args.candidates, provider=provider,
        progress=lambda message: print(message, flush=True),
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(args.output_dir or f".harness-data/play_ab/{stamp}")
    written = write_scenes(pair, directory, args.prompt, args.seed)
    print(f"Scenes written to {directory}\n")

    order = presentation_order(pair, args.seed, args.blind)
    for name, entry in order:
        print(f"── {name} ──")
        if not args.blind:
            for line in scorecard_lines(entry):
                print(line)
        elif entry.scene is None:
            print(f"  produced no certified circuit: {entry.outcome.failure}")
        print()

    if args.no_play:
        _reveal(order, args.blind)
        return

    from .play import play_scene

    results: dict[str, object] = {}
    for name, entry in order:
        if entry.scene is None:
            continue
        scene = entry.scene.model_copy(update={
            "dynamics": apply_dynamics_preset(entry.scene.dynamics, args.dynamics),
        })
        print(f"Opening {name} in a window. Q or Escape closes it and moves to the next circuit.",
              flush=True)
        try:
            results[name] = play_scene(scene)
        except Exception as error:  # a viewer problem should name itself, not vanish
            print(f"  could not open a window: {error}\n"
                  f"  the circuit is still saved; drive it later with\n"
                  f"    harness play --scene {written.get(entry.arm)}", flush=True)
            results[name] = {"error": str(error)}
    _reveal(order, args.blind)
    print(json.dumps({"scenes": written, "sessions": results}, indent=2))


def _reveal(order, blind: bool) -> None:
    """After a blind session, say which arm authored which circuit."""
    if not blind:
        return
    from .play_ab import scorecard_lines

    print("── reveal ──")
    for name, entry in order:
        print(f"{name}: {entry.label}")
        for line in scorecard_lines(entry):
            print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness", description="Single-domain 2D racing-agent research harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    environment = subparsers.add_parser("environment")
    environment_subparsers = environment.add_subparsers(dest="environment_command", required=True)
    create = environment_subparsers.add_parser("create")
    create.add_argument("prompt")
    create.add_argument("--seed", type=int)
    run = subparsers.add_parser("run")
    run.add_argument("--environment", required=True)
    run.add_argument(
        "--policy", default="oracle-racing-line", type=canonical_policy_name,
        choices=tuple(built_in_policies()),
    )
    experiment = subparsers.add_parser("experiment")
    experiment.add_argument("prompt", nargs="?", default="Compare policies under action delay")
    experiment.add_argument("--environment", required=False)
    replay = subparsers.add_parser("replay", help="Open or export an engine-neutral native replay bundle")
    replay.add_argument("--run", required=True)
    replay.add_argument("--export", action="store_true", help="Export JSON without opening the desktop viewer")
    play = subparsers.add_parser("play", help="Drive the deterministic engine locally with WASD")
    play.add_argument(
        "--prompt",
        help=(
            "Compile and drive a natural-language circuit brief offline, e.g. "
            "\"slippery curvy track with a 90 degree bend in the top right and three aggressive npcs\". "
            "Overrides the archetype flags."
        ),
    )
    play.add_argument(
        "--scene",
        help="Drive a scene saved as JSON, e.g. one written by `play-ab`. Overrides every other flag.",
    )
    play.add_argument("--circuit", choices=("oval", "technical", "chicane"), default="technical")
    play.add_argument("--surface", choices=("asphalt", "clay", "ice"), default="asphalt")
    play.add_argument("--grip", type=float, default=1.0, help="Continuous grip multiplier from 0.3 to 1.2")
    play.add_argument("--obstacles", type=int, choices=range(7), default=2)
    play.add_argument(
        "--edge-barriers", action="store_true",
        help="Add continuous solid guardrails along both road edges",
    )
    play.add_argument("--npcs", type=int, choices=range(6), default=1)
    play.add_argument(
        "--npc-profile",
        choices=tuple(item.value for item in NpcProfile),
        default="racer",
        help="Opponent temperament for the archetype flags",
    )
    play.add_argument(
        "--npc-intelligence", type=float, default=None,
        help="Override opponent line quality from 0 (wandering) to 1 (apex-seeking)",
    )
    play.add_argument(
        "--npc-aggression", type=float, default=None,
        help="Override opponent willingness to commit to a pass, 0 to 1",
    )
    play.add_argument("--laps", type=int, choices=range(1, 5), default=1)
    play.add_argument("--npc-start", choices=("grid", "distributed"), default="grid")
    play.add_argument("--seed", type=int, default=17)
    play.add_argument(
        "--dynamics",
        choices=("balanced", "low_grip", "worn_tires", "heavy_car", "rear_bias", "high_drag", "high_downforce"),
        default="balanced",
        help="Deterministic vehicle/road condition preset",
    )
    play3d = subparsers.add_parser(
        "play3d", help="Drive the same circuit in the 3D runtime with switchable cameras",
    )
    for parser_with_scene_flags in (play3d,):
        parser_with_scene_flags.add_argument(
            "--prompt",
            help="Compile and drive a natural-language circuit brief offline; overrides the archetype flags",
        )
        parser_with_scene_flags.add_argument("--circuit", choices=("oval", "technical", "chicane"), default="technical")
        parser_with_scene_flags.add_argument("--surface", choices=("asphalt", "clay", "ice"), default="asphalt")
        parser_with_scene_flags.add_argument("--grip", type=float, default=1.0, help="Continuous grip multiplier from 0.3 to 1.2")
        parser_with_scene_flags.add_argument("--obstacles", type=int, choices=range(7), default=2)
        parser_with_scene_flags.add_argument(
            "--edge-barriers", action="store_true",
            help="Add continuous solid guardrails along both road edges",
        )
        parser_with_scene_flags.add_argument("--npcs", type=int, choices=range(6), default=1)
        parser_with_scene_flags.add_argument(
            "--npc-profile", choices=tuple(item.value for item in NpcProfile), default="racer",
            help="Opponent temperament for the archetype flags",
        )
        parser_with_scene_flags.add_argument(
            "--npc-intelligence", type=float, default=None,
            help="Override opponent line quality from 0 (wandering) to 1 (apex-seeking)",
        )
        parser_with_scene_flags.add_argument(
            "--npc-aggression", type=float, default=None,
            help="Override opponent willingness to commit to a pass, 0 to 1",
        )
        parser_with_scene_flags.add_argument("--laps", type=int, choices=range(1, 5), default=1)
        parser_with_scene_flags.add_argument("--npc-start", choices=("grid", "distributed"), default="grid")
        parser_with_scene_flags.add_argument("--seed", type=int, default=17)
        parser_with_scene_flags.add_argument(
            "--dynamics",
            choices=("balanced", "low_grip", "worn_tires", "heavy_car", "rear_bias", "high_drag", "high_downforce"),
            default="balanced", help="Deterministic vehicle/road condition preset",
        )
    play3d.add_argument(
        "--view", choices=tuple(item.value for item in ViewMode), default="third-person",
        help="Starting camera; C cycles between them in-game",
    )
    play3d.add_argument(
        "--elevation", choices=tuple(item.value for item in ElevationProfile), default="rolling",
        help="Vertical profile of the circuit",
    )
    play3d.add_argument("--amplitude", type=float, default=5.0, help="Peak-to-trough elevation in meters")
    play3d.add_argument("--hills", type=int, choices=range(1, 9), default=3, help="Crests per lap")
    play3d.add_argument("--banking", type=float, default=6.0, help="Maximum corner cross-slope in degrees")
    play3d.add_argument(
        "--crest-sharpness", type=float, default=None,
        help="Continuous elevation shape from 0 (smooth sine) to 1 (sharper compound crests)",
    )
    play3d.add_argument(
        "--road-detail", type=int, choices=range(1, 5), default=DEFAULT_ROAD_DETAIL,
        help="Render-only road subdivision; higher is smoother over crests and costs more",
    )
    play_ab = subparsers.add_parser(
        "play-ab",
        help="Drive the same brief as compiled by the base model and by the harness",
    )
    play_ab.add_argument("--prompt", required=True, help="The race brief both arms receive")
    play_ab.add_argument("--seed", type=int, default=17)
    play_ab.add_argument(
        "--arms", nargs="+", default=["oneshot", "harness"],
        choices=("oneshot", "selfjudge", "harness"),
        help="oneshot is the single-proposal base model; harness is the measured search",
    )
    play_ab.add_argument("--candidates", type=int, default=4, help="Proposals per search arm")
    play_ab.add_argument("--provider", default="auto", choices=("auto", "offline", "openai", "anthropic"))
    play_ab.add_argument(
        "--blind", action="store_true",
        help="Hide which arm authored which circuit until both have been driven",
    )
    play_ab.add_argument("--output-dir", help="Where to write the compiled scenes")
    play_ab.add_argument("--no-play", action="store_true", help="Compile and report without opening a window")
    play_ab.add_argument(
        "--dynamics", default="balanced",
        choices=("balanced", "low_grip", "worn_tires", "heavy_car", "rear_bias", "high_drag", "high_downforce"),
    )
    engine_check = subparsers.add_parser("engine-check", help="Run the offline engine conformance matrix")
    engine_check.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 43])
    subparsers.add_parser("demo")
    args = parser.parse_args()

    if args.command == "play" and args.scene:
        from .play import play_scene
        from .play_ab import load_scene

        scene = load_scene(args.scene)
        print(f"{scene.name} — replaying a saved scene from {args.scene}")
        print(f"  {scene.surface} at {scene.grip:.2f}x grip, {scene.laps} lap(s), "
              f"{scene.track_width:.0f}px wide, seed {scene.seed}")
        print(f"  brief: {scene.prompt}")
        print(json.dumps(play_scene(scene), indent=2))
        return
    if args.command == "play-ab":
        _play_ab(args)
        return
    if args.command == "play":
        from .play import play_scene

        plan, label = _plan_from_arguments(args)
        scene, certificate, notes = compile_certified_scene(label, plan, args.seed)
        scene = scene.model_copy(update={
            "dynamics": apply_dynamics_preset(scene.dynamics, args.dynamics),
        })
        _describe_circuit(scene, certificate, notes)
        print(json.dumps(play_scene(scene), indent=2))
        return
    if args.command == "play3d":
        from .play3d import play_scene_3d

        plan, label = _plan_from_arguments(args)
        elevation = ElevationSpec(
            profile=ElevationProfile(args.elevation),
            amplitude_m=args.amplitude,
            hill_count=args.hills,
            banking_degrees=args.banking,
            **({"crest_sharpness": args.crest_sharpness} if args.crest_sharpness is not None else {}),
        )
        scene, certificate, notes = compile_racing_3d_scene(label, plan, elevation, args.seed)
        scene = scene.model_copy(update={
            "dynamics": apply_dynamics_preset(scene.dynamics, args.dynamics),
        })
        _describe_circuit(scene, certificate, notes)
        print(json.dumps(
            play_scene_3d(scene, ViewMode(args.view), road_detail=args.road_detail), indent=2,
        ))
        return
    if args.command == "engine-check":
        from .diagnostics import engine_conformance_matrix

        result = engine_conformance_matrix(args.seeds)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return

    service = HarnessService()

    if args.command == "environment" and args.environment_command == "create":
        print(service.create_environment(args.prompt, args.seed).model_dump_json(indent=2))
        return
    if args.command == "run":
        print(service.run(RunRequest(environment_id=args.environment, policy_name=args.policy)).model_dump_json(indent=2))
        return
    if args.command == "experiment":
        environment_id = args.environment
        if not environment_id:
            environments = service.list_environments()
            if not environments:
                environment_id = service.create_environment("A technical asphalt circuit with two barriers and one opponent.").id
            else:
                environment_id = environments[0].id
        print(service.run_experiment(ExperimentRequest(environment_id=environment_id)).model_dump_json(indent=2))
        return
    if args.command == "replay":
        if args.export:
            print(service.export_replay_bundle(args.run))
        else:
            print(json.dumps(service.launch_native_viewer(args.run), indent=2))
        return
    if args.command == "demo":
        record = service.create_environment("A technical asphalt circuit with two barriers and one opponent.")
        run_record = service.run(RunRequest(environment_id=record.id, policy_name="oracle-racing-line"))
        experiment_record = service.run_experiment(ExperimentRequest(environment_id=record.id, seeds=[11, 29, 47]))
        print(json.dumps({"environment": record.id, "run": run_record.id, "experiment": experiment_record.id}, indent=2))


if __name__ == "__main__":
    main()

"""Offline conformance checks for the deterministic racing domain.

Two matrices run without a model or network call. The archetype matrix guards
the engine contract (deterministic compilation, geometry, oracle completion).
The grammar matrix guards prompt fidelity: it compiles briefs that name corner
angles, screen regions, opponent temperaments, and grip, then asserts the
compiled circuit actually carries those properties. Together they are the
regression net for "a brief describing a track and NPC behavior is reliable".
"""

from __future__ import annotations

from .models import NpcProfile, TrackRegion
from .racing import (
    RacingDesignDraft,
    RacingLineController,
    RacingWorld,
    compile_certified_scene,
    compile_racing_scene,
    validate_racing_scene,
    verify_racing_playability,
)
from .track_grammar import parse_track_prompt


# Each case pairs a brief with the properties the compiled circuit must carry.
GRAMMAR_CASES: tuple[dict[str, object], ...] = (
    {
        "brief": "a fast wide asphalt oval with no barriers and one opponent, 3 laps",
        "laps": 3, "surface": "asphalt", "profiles": ["racer"], "barriers": 0,
    },
    {
        "brief": "three aggressive npcs, slippery track, curvy with a 90 degree bend in the top right",
        "surface": "asphalt", "max_grip": .6, "profiles": ["aggressor"] * 3,
        "corner": {"angle": 90.0, "region": TrackRegion.TOP_RIGHT},
    },
    {
        "brief": "technical ice circuit with two barriers in the bottom left and two cautious rivals",
        "surface": "ice", "barriers": 2, "profiles": ["backmarker"] * 2,
    },
    {
        "brief": "very twisty wet circuit with a chicane on the right side and 2 blocking npcs",
        "surface": "asphalt", "max_grip": .6, "profiles": ["blocker"] * 2, "min_corners": 6,
    },
    {
        "brief": "clockwise dirt oval, 4 laps, no opponents",
        "surface": "clay", "laps": 4, "profiles": [], "direction": "clockwise",
    },
    {
        "brief": "a 120 degree corner in the bottom right and a 45 degree bend in the top left, four barriers",
        "barriers": 4, "corner": {"angle": 120.0, "region": TrackRegion.BOTTOM_RIGHT},
    },
    {
        "brief": "narrow icy track with six tight corners and one aggressive rival plus two slow backmarkers",
        "surface": "ice", "min_corners": 6, "profiles": ["aggressor", "backmarker", "backmarker"],
    },
    {
        "brief": "serpentine sticky circuit, 10 corners, three barriers, four aggressive npcs spread around the track",
        "min_corners": 9, "barriers": 3, "profiles": ["aggressor"] * 4,
        "npc_start_mode": "distributed",
    },
)


def engine_conformance_matrix(seeds: list[int]) -> dict[str, object]:
    rows = [
        *_archetype_rows(seeds),
        *_grammar_rows(seeds[:1] or [17]),
        *_elevation_rows(seeds[:1] or [17]),
    ]
    failures = [row for row in rows if not row["passed"]]
    return {
        "engine": "racing-2d-v5",
        "compiler": "track-grammar-v1",
        "engine_3d": "racing-3d-v1",
        "passed": not failures,
        "cases": len(rows),
        "failures": failures,
        "rows": rows,
    }


def _elevation_rows(seeds: list[int]) -> list[dict[str, object]]:
    """Guard the 3D engine and its one seam with the 2D engine.

    Every case checks the same three things: that a flat 3D world is bit-for-bit
    the 2D world, that the vertical profile closes over a lap and stays inside the
    drivable grade, and that the deterministic oracle can still finish the race
    once gradients and banking are applied.
    """
    from .models import ElevationProfile, ElevationSpec
    from .racing3d import verify_racing_3d_playability
    from .track3d import compile_track_surface, fit_drivable_elevation, validate_track_surface

    flat = ElevationSpec(profile=ElevationProfile.FLAT, amplitude_m=0, banking_degrees=0)
    rows: list[dict[str, object]] = []
    for circuit in ("oval", "technical", "chicane"):
        for surface in ("asphalt", "clay", "ice"):
            for profile in (ElevationProfile.ROLLING, ElevationProfile.HILLY):
                for seed in seeds:
                    prompt = f"offline 3d conformance {circuit} {surface} {profile.value} seed {seed}"
                    draft = RacingDesignDraft(
                        title=f"{circuit.title()} {profile.value.title()} 3D conformance",
                        rationale="Elevation, banking, traffic, and barriers together.",
                        circuit=circuit, surface=surface, obstacle_count=2, npc_count=2,
                        laps=1, npc_start_mode="grid", npc_profile=NpcProfile.RACER,
                    )
                    planar = compile_racing_scene(prompt, draft, seed)
                    requested = ElevationSpec(
                        profile=profile, amplitude_m=6.0, hill_count=3, banking_degrees=8.0,
                    )
                    fitted, fit_notes = fit_drivable_elevation(planar, requested)
                    scene = planar.model_copy(update={"elevation": fitted})
                    surface_profile = compile_track_surface(scene)
                    surface_findings = validate_track_surface(surface_profile)
                    certificate = verify_racing_3d_playability(scene)
                    reduces_to_2d = _flat_3d_matches_2d(planar, flat)
                    rows.append({
                        "matrix": "elevation",
                        "circuit": circuit,
                        "surface": surface,
                        "profile": profile.value,
                        "seed": seed,
                        "passed": (
                            reduces_to_2d and not surface_findings and certificate.playable
                            and abs(surface_profile.seam_step) < 1e-6
                        ),
                        "flat_3d_matches_2d": reduces_to_2d,
                        "seam_step_pixels": surface_profile.seam_step,
                        "relief_pixels": round(surface_profile.relief_pixels, 1),
                        "steepest_grade_degrees": round(surface_profile.steepest_grade_degrees, 1),
                        "steepest_bank_degrees": round(surface_profile.steepest_bank_degrees, 1),
                        "surface_findings": surface_findings,
                        "oracle_playable": certificate.playable,
                        "oracle_steps": certificate.route_steps,
                        "failure": certificate.failure,
                        "relaxations": fit_notes,
                    })
    return rows


def _flat_3d_matches_2d(planar_scene, flat) -> bool:
    """Run both engines on one scene and compare the shared state exactly."""
    from .racing3d import Racing3DWorld

    flat_scene = planar_scene.model_copy(update={"elevation": flat})
    planar = RacingWorld.from_scene(planar_scene)
    spatial = Racing3DWorld.from_scene(flat_scene)
    planar_controller, spatial_controller = RacingLineController(), RacingLineController()
    planar_controller.reset(planar_scene, planar_scene.seed)
    spatial_controller.reset(flat_scene, flat_scene.seed)
    for _ in range(260):
        if planar.terminated or spatial.terminated:
            break
        planar.step(*planar_controller.act(planar.observe()))
        spatial.step(*spatial_controller.act(spatial.observe()))
    planar_state = planar.snapshot()
    spatial_state = spatial.snapshot()
    return {key: spatial_state.get(key) for key in planar_state} == planar_state


def _archetype_rows(seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for circuit in ("oval", "technical", "chicane"):
        for surface in ("asphalt", "clay", "ice"):
            for seed in seeds:
                prompt = f"offline conformance {circuit} {surface} seed {seed}"
                draft = RacingDesignDraft(
                    title=f"{circuit.title()} {surface.title()} conformance",
                    rationale="Maximum traffic and barriers exercise the deterministic engine contract.",
                    circuit=circuit,
                    surface=surface,
                    obstacle_count=4,
                    npc_count=3,
                    laps=1,
                    npc_start_mode="grid",
                    npc_profile=NpcProfile.AGGRESSOR,
                )
                first = compile_racing_scene(prompt, draft, seed)
                second = compile_racing_scene(prompt, draft, seed)
                deterministic = first.model_dump() == second.model_dump()
                validation = validate_racing_scene(first)
                certificate = verify_racing_playability(first)
                report = first.track_report
                closed = report is not None and report.closure_error_pixels <= .5
                rows.append({
                    "matrix": "archetype",
                    "circuit": circuit,
                    "surface": surface,
                    "seed": seed,
                    "passed": (
                        deterministic and closed
                        and validation == ["Racing domain contract passed."]
                        and certificate.playable
                    ),
                    "deterministic_compile": deterministic,
                    "closed_loop": closed,
                    "sector_count": first.sector_count,
                    "validation": validation,
                    "oracle_playable": certificate.playable,
                    "oracle_steps": certificate.route_steps,
                    "failure": certificate.failure,
                })
    return rows


def _grammar_rows(seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in GRAMMAR_CASES:
        for seed in seeds:
            brief = str(case["brief"])
            plan = parse_track_prompt(brief)
            mismatches: list[str] = []
            try:
                scene, certificate, notes = compile_certified_scene(brief, plan, seed)
            except ValueError as error:
                rows.append({
                    "matrix": "grammar", "brief": brief, "seed": seed, "passed": False,
                    "mismatches": [str(error)],
                })
                continue
            report = scene.track_report
            assert report is not None
            repeat, _, _ = compile_certified_scene(brief, parse_track_prompt(brief), seed)
            if repeat.model_dump() != scene.model_dump():
                mismatches.append("recompiling the same brief produced a different scene")
            if report.closure_error_pixels > .5:
                mismatches.append(f"loop did not close ({report.closure_error_pixels}px)")
            if report.angle_fidelity_degrees > .5:
                mismatches.append(f"turn angles drifted {report.angle_fidelity_degrees} degrees")
            for key, actual in (
                ("surface", scene.surface), ("laps", scene.laps),
                ("direction", report.direction), ("npc_start_mode", scene.npc_start_mode),
            ):
                if key in case and case[key] != actual:
                    mismatches.append(f"{key} was {actual!r}, brief asked for {case[key]!r}")
            if "max_grip" in case and scene.grip > float(case["max_grip"]):
                mismatches.append(f"grip was {scene.grip}, brief asked for slippery")
            if "profiles" in case:
                actual_profiles = sorted(item.profile.value for item in scene.npc_behaviors)
                if actual_profiles != sorted(str(item) for item in case["profiles"]):
                    mismatches.append(f"opponents were {actual_profiles}, brief asked for {case['profiles']}")
            if "barriers" in case:
                barriers = sum(1 for entity in scene.entities if entity.kind == "obstacle")
                # The certification ladder may surrender barriers; it must say so.
                if barriers != case["barriers"] and not notes:
                    mismatches.append(f"compiled {barriers} barriers, brief asked for {case['barriers']}")
            if "min_corners" in case and len(report.corners) < int(case["min_corners"]):
                mismatches.append(f"compiled {len(report.corners)} corners, brief implies at least {case['min_corners']}")
            if "corner" in case:
                wanted = case["corner"]
                matched = [
                    corner for corner in report.corners
                    if abs(corner.achieved_angle_degrees - float(wanted["angle"])) <= .5
                    and corner.achieved_region == wanted["region"]
                ]
                if not matched:
                    mismatches.append(
                        f"no {wanted['angle']} degree corner landed in {wanted['region'].value}"
                    )
            if validate_racing_scene(scene) != ["Racing domain contract passed."]:
                mismatches.append("compiled scene failed the domain contract")
            if not certificate.playable:
                mismatches.append(f"oracle could not finish: {certificate.failure}")
            rows.append({
                "matrix": "grammar",
                "brief": brief,
                "seed": seed,
                "passed": not mismatches,
                "mismatches": mismatches,
                "corners": len(report.corners),
                "sector_count": scene.sector_count,
                "grip": scene.grip,
                "angle_fidelity_degrees": report.angle_fidelity_degrees,
                "region_fidelity": report.region_fidelity,
                "closure_error_pixels": report.closure_error_pixels,
                "oracle_steps": certificate.route_steps,
                "relaxations": [*report.relaxations, *notes],
            })
    return rows

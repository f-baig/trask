"""Prompt-to-circuit fidelity and opponent-behavior differentiation.

These tests protect the two properties that make the harness usable: a brief
that names track geometry and NPC temperament compiles into a circuit that
actually has them, and two opponents with different temperaments actually drive
differently. Every case is offline and deterministic.
"""

from __future__ import annotations

import math

import pytest

from harness.models import Action, CornerRadius, NpcProfile, StraightLength, TrackRegion, Vec2
from harness.racing import (
    CAR_RADIUS, SCENE_BOUNDS, RacingWorld, _nearest_point_index, _rect_center,
    compile_certified_scene, compile_racing_scene, racing_public_context, racing_track_map,
    validate_racing_scene,
)
from harness.track_grammar import (
    BarrierSpec, CornerSpec, MIN_STRAIGHT_PIXELS, NpcSpec, TrackPlan, archetype_plan,
    compile_certified_track, compile_track, minimum_corner_radius, parse_track_prompt,
    track_plan_schema, validate_track_geometry,
)


def plan(**overrides) -> TrackPlan:
    base = TrackPlan(
        title="Grammar test circuit",
        rationale="A circuit authored directly in the corner grammar.",
        corners=[CornerSpec(angle_degrees=90) for _ in range(4)],
    )
    return base.model_copy(update=overrides)


def green_flag(world: RacingWorld) -> RacingWorld:
    world.countdown_ticks_remaining = 0
    return world


def test_track_plan_schema_is_a_structured_output_object() -> None:
    schema = track_plan_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert {"corners", "npcs", "barriers", "grip", "direction", "loop_shape"} <= set(schema["properties"])
    barrier_schema = schema["properties"]["barriers"]["items"]
    assert barrier_schema["properties"]["shape"]["enum"] == ["circle", "box", "oriented-box"]
    assert "shape" in barrier_schema["required"]


def test_literal_circle_compiles_to_a_zero_corner_constant_radius_loop() -> None:
    plan = parse_track_prompt("a legitimate circular track with no corners")
    track, findings = compile_certified_track(plan, SCENE_BOUNDS, CAR_RADIUS)

    assert findings == []
    assert plan.loop_shape == "circle"
    assert plan.corners == []
    assert track.report.loop_shape == "circle"
    assert track.report.corners == []
    assert track.corner_indices == ()
    assert track.report.closure_error_pixels == 0
    center_x = sum(point.x for point in track.centerline) / len(track.centerline)
    center_y = sum(point.y for point in track.centerline) / len(track.centerline)
    radii = [math.hypot(point.x - center_x, point.y - center_y) for point in track.centerline]
    assert max(radii) - min(radii) < .02


def test_edge_barrier_language_authors_guardrails_not_discrete_obstacles() -> None:
    parsed = parse_track_prompt(
        "A technical circuit with visible edge barriers along both road edges"
    )
    assert parsed.edge_barriers is True
    assert parsed.barriers == []


def test_prompt_can_place_the_start_line_and_player_in_the_grid() -> None:
    parsed = parse_track_prompt(
        "A clay circuit with the start/finish line in the top right and player starts P4"
    )
    assert parsed.start_region == TrackRegion.TOP_RIGHT
    assert parsed.player_grid_position == 4
    assert len(parsed.npcs) == 3, "P4 requires three actual grid cars ahead of the player"


@pytest.mark.parametrize("circuit", ["oval", "technical", "chicane"])
def test_archetypes_compile_to_distinct_closed_circuits(circuit: str) -> None:
    track, findings = compile_certified_track(archetype_plan(circuit), SCENE_BOUNDS, CAR_RADIUS)
    assert findings == []
    assert track.report.closure_error_pixels <= .5
    assert track.report.angle_fidelity_degrees <= .5
    others = [
        compile_certified_track(archetype_plan(other), SCENE_BOUNDS, CAR_RADIUS)[0]
        for other in ("oval", "technical", "chicane") if other != circuit
    ]
    signature = [(point.x, point.y) for point in track.centerline]
    assert all(signature != [(point.x, point.y) for point in item.centerline] for item in others)


def test_compilation_is_byte_deterministic() -> None:
    first = compile_track(plan(), SCENE_BOUNDS, CAR_RADIUS)
    second = compile_track(plan(), SCENE_BOUNDS, CAR_RADIUS)
    assert first.report.model_dump() == second.report.model_dump()
    assert [(point.x, point.y) for point in first.centerline] == [
        (point.x, point.y) for point in second.centerline
    ]
    assert first.sector_indices == second.sector_indices


@pytest.mark.parametrize("angle", [30.0, 45.0, 90.0, 120.0, 150.0])
def test_requested_turn_angle_is_honored_exactly(angle: float) -> None:
    """An authored angle is geometry, not a hint; closure is absorbed elsewhere."""
    track = compile_track(
        plan(corners=[CornerSpec(angle_degrees=angle, region=TrackRegion.TOP_RIGHT),
                      CornerSpec(), CornerSpec(), CornerSpec()]),
        SCENE_BOUNDS, CAR_RADIUS,
    )
    authored = track.report.corners[0]
    assert authored.requested_angle_degrees == angle
    assert authored.achieved_angle_degrees == pytest.approx(angle, abs=.01)
    # The remaining corners absorb whatever rotation is left over.
    signed = sum(
        (-1 if corner.direction == "left" else 1) * corner.achieved_angle_degrees
        for corner in track.report.corners
    )
    assert abs(abs(signed) - 360.0) < .01


@pytest.mark.parametrize("region", [
    TrackRegion.TOP_LEFT, TrackRegion.TOP_RIGHT, TrackRegion.BOTTOM_LEFT,
    TrackRegion.BOTTOM_RIGHT, TrackRegion.TOP_CENTER, TrackRegion.LEFT,
])
def test_a_corner_lands_in_the_requested_screen_region(region: TrackRegion) -> None:
    track = compile_track(
        plan(corners=[CornerSpec(angle_degrees=90, region=region),
                      CornerSpec(), CornerSpec(), CornerSpec()]),
        SCENE_BOUNDS, CAR_RADIUS,
    )
    assert track.report.corners[0].achieved_region == region
    assert track.report.region_fidelity == 1.0


def test_ninety_degree_bend_in_the_top_right_from_a_natural_brief() -> None:
    """The end-to-end path for the brief this grammar exists to satisfy."""
    brief = "three aggressive npcs, slippery track, curvy with a 90 degree bend in the top right"
    scene, certificate, _ = compile_certified_scene(brief, parse_track_prompt(brief), seed=17)
    report = scene.track_report
    assert report is not None
    assert certificate.playable
    assert validate_racing_scene(scene) == ["Racing domain contract passed."]
    assert scene.grip <= .6, "a slippery brief must lower grip"
    assert scene.surface == "asphalt", "slipperiness is a grip condition, not a new surface"
    assert [item.profile for item in scene.npc_behaviors] == [NpcProfile.AGGRESSOR] * 3
    assert len(report.corners) >= 6, "a curvy brief needs more than an oval's corners"
    bends = [
        corner for corner in report.corners
        if corner.achieved_region == TrackRegion.TOP_RIGHT
        and abs(corner.achieved_angle_degrees - 90) < .5
    ]
    assert bends, [
        (corner.achieved_angle_degrees, corner.achieved_region.value)
        for corner in report.corners
    ]


def test_region_word_is_not_read_as_a_turn_direction() -> None:
    """"in the top right" locates a corner; it does not say which way it turns."""
    located = parse_track_prompt("a 90 degree bend in the top right")
    assert located.corners[0].direction is None
    handed = parse_track_prompt("a 90 degree right hand corner")
    assert handed.corners[0].direction == "right"


def test_a_chicane_compiles_to_an_opposed_pair_of_kinks() -> None:
    parsed = parse_track_prompt("a circuit with a chicane on the right side")
    kinks = [corner for corner in parsed.corners if corner.label and "chicane" in corner.label]
    assert len(kinks) == 2
    assert {kink.direction for kink in kinks} == {"left", "right"}


@pytest.mark.parametrize(("brief", "expected"), [
    ("no opponents at all", []),
    ("one aggressive rival", [NpcProfile.AGGRESSOR]),
    ("three aggressive npcs", [NpcProfile.AGGRESSOR] * 3),
    ("two blocking cars", [NpcProfile.BLOCKER] * 2),
    ("four slow backmarkers", [NpcProfile.BACKMARKER] * 4),
    ("2 opponents", [NpcProfile.RACER] * 2),
])
def test_opponent_count_and_temperament_are_read_from_the_brief(
    brief: str, expected: list[NpcProfile],
) -> None:
    assert [npc.profile for npc in parse_track_prompt(brief).npcs] == expected


@pytest.mark.parametrize(("brief", "ceiling"), [
    ("a slippery circuit", .6),
    ("a wet track", .6),
    ("a treacherous ice circuit", .45),
    ("a greasy low grip surface", .6),
])
def test_slipperiness_is_a_continuous_grip_axis(brief: str, ceiling: float) -> None:
    assert parse_track_prompt(brief).grip <= ceiling


def test_high_grip_briefs_raise_grip_above_baseline() -> None:
    assert parse_track_prompt("a sticky high grip circuit").grip > 1.0


def test_geometry_validation_rejects_a_self_overlapping_corridor() -> None:
    """A corridor that merges with itself makes on-track and progress undefined."""
    # Two lanes of road only twelve pixels apart cannot both be drivable.
    lane = [Vec2(x=200.0 + index * 20, y=300.0) for index in range(20)]
    doubled = [*lane, *reversed([Vec2(x=point.x, y=point.y + 12) for point in lane])]
    findings = validate_track_geometry(doubled, 132.0, SCENE_BOUNDS, CAR_RADIUS)
    assert any("overlaps itself" in finding or "uniform" in finding for finding in findings)


def test_impossible_corner_set_is_relaxed_and_reported_rather_than_silently_bent() -> None:
    """A hairpin too tight for the corridor is opened up, and the report says so."""
    track, findings = compile_certified_track(
        plan(
            track_width=152.0,
            corners=[
                CornerSpec(angle_degrees=170, radius=CornerRadius.HAIRPIN,
                           exit_straight=StraightLength.LONG, region=TrackRegion.TOP_LEFT),
                CornerSpec(), CornerSpec(), CornerSpec(),
            ],
        ),
        SCENE_BOUNDS, CAR_RADIUS,
    )
    assert findings == []
    assert track.report.relaxations, "a relaxed plan must record what changed"
    assert 2 * minimum_corner_radius(track.centerline) > 152.0


def test_compiled_geometry_is_uniformly_sampled_and_inside_bounds() -> None:
    track = compile_track(plan(corners=[CornerSpec() for _ in range(7)]), SCENE_BOUNDS, CAR_RADIUS)
    count = len(track.centerline)
    spacings = [
        math.hypot(
            track.centerline[(index + 1) % count].x - track.centerline[index].x,
            track.centerline[(index + 1) % count].y - track.centerline[index].y,
        )
        for index in range(count)
    ]
    assert max(spacings) / min(spacings) < 1.2, "index-based lookahead assumes even sampling"
    assert validate_track_geometry(track.centerline, 132.0, SCENE_BOUNDS, CAR_RADIUS) == []
    margin = 132.0 / 2 + CAR_RADIUS
    assert all(
        margin <= point.x <= SCENE_BOUNDS.width - margin
        and margin <= point.y <= SCENE_BOUNDS.height - margin
        for point in track.centerline
    )


def test_straight_lengths_stay_positive_for_every_corner_count() -> None:
    for corner_count in range(3, 11):
        track = compile_track(
            plan(corners=[CornerSpec() for _ in range(corner_count)]), SCENE_BOUNDS, CAR_RADIUS,
        )
        assert track.report.closure_error_pixels <= .5, corner_count
        assert track.report.longest_straight_pixels >= MIN_STRAIGHT_PIXELS * .5, corner_count
        assert validate_track_geometry(track.centerline, 132.0, SCENE_BOUNDS, CAR_RADIUS) == [], corner_count


def test_sector_gates_avoid_corners_and_end_at_the_finish_line() -> None:
    track = compile_track(plan(corners=[CornerSpec() for _ in range(6)]), SCENE_BOUNDS, CAR_RADIUS)
    assert track.sector_indices[-1] == 0, "the finish line is always index zero"
    assert len(set(track.sector_indices)) == len(track.sector_indices)
    corner_curvature = max(track.curvature_at(index) for index in track.corner_indices)
    for gate in track.sector_indices:
        assert track.curvature_at(gate) < corner_curvature


def test_opponent_temperament_changes_measured_lap_pace() -> None:
    """Behavior is not cosmetic: profiles produce different distance covered."""
    covered: dict[str, float] = {}
    for profile in (NpcProfile.BACKMARKER, NpcProfile.RACER, NpcProfile.AGGRESSOR):
        scene = compile_racing_scene(
            "pace probe",
            plan(npcs=[NpcSpec(profile=profile)], barriers=[]),
            seed=11,
        )
        world = green_flag(RacingWorld.from_scene(scene))
        world.terminate_on_opponent_win = False
        for _ in range(220):
            world.step(Action())
        # Cumulative race distance, not an index delta: a fast car laps inside the
        # probe window and an index delta silently wraps to a tiny number.
        covered[profile.value] = world.opponents[0].progress_samples
    assert covered["backmarker"] < covered["racer"] < covered["aggressor"], covered


def test_defending_opponent_never_blocks_the_certified_racing_line() -> None:
    """Aggression must stay compatible with the scene's own verification.

    The deterministic oracle certifies every circuit by driving lane offset zero,
    so a defender that covered the centerline would make aggressive-traffic
    briefs fail their own playability check and be discarded.
    """
    scene = compile_racing_scene(
        "defensive traffic",
        plan(npcs=[NpcSpec(profile=NpcProfile.AGGRESSOR) for _ in range(3)], barriers=[]),
        seed=11,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    phases: set[str] = set()
    for _ in range(600):
        if world.terminated:
            break
        world.step(Action())
        for opponent in world.opponents:
            phases.add(opponent.overtake_phase)
            assert abs(opponent.lane_offset) >= 20.0 or opponent.overtake_phase != "defending", (
                opponent.entity_id, opponent.lane_offset, opponent.overtake_phase,
            )
    assert phases <= {"cruise", "passing", "merge", "defending"}


def test_track_map_is_published_to_the_player_and_hides_no_reward_state() -> None:
    scene = compile_racing_scene("map probe", plan(), seed=5)
    world = green_flag(RacingWorld.from_scene(scene))
    context = racing_public_context(scene, world.observe())
    corners = context["track_map"]
    assert corners == racing_track_map(scene)
    assert corners and all(
        {"corner", "entry_progress_percent", "direction", "turn_degrees",
         "recommended_entry_speed", "screen_region"} <= item.keys()
        for item in corners
    )
    upcoming = context["upcoming_corner"]
    assert upcoming is not None and upcoming["distance_pixels"] >= 0
    # Route knowledge is public; creator-side fidelity accounting is not.
    payload = str(context)
    assert "relaxations" not in payload
    assert "requested_angle" not in payload and "requested_region" not in payload
    assert "privileged" not in payload and "objective_index" not in payload


def test_nearby_opponent_telemetry_exposes_temperament() -> None:
    scene = compile_racing_scene(
        "telemetry probe", plan(npcs=[NpcSpec(profile=NpcProfile.AGGRESSOR)], barriers=[]), seed=5,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    opponents = [
        entity for entity in world.observe().local_entities if entity.get("kind") == "npc"
    ]
    assert opponents, "the grid opponent should be within observation range"
    assert opponents[0]["profile"] == "aggressor"
    assert opponents[0]["aggression"] > .8
    assert opponents[0]["defends"] is True


def test_barrier_lane_side_is_published_so_a_driver_knows_which_edge_narrows() -> None:
    scene = compile_racing_scene(
        "barrier probe", plan(barriers=[BarrierSpec(side="right")], npcs=[]), seed=5,
    )
    barriers = [entity for entity in scene.entities if entity.kind == "obstacle"]
    assert len(barriers) == 1
    world = green_flag(RacingWorld.from_scene(scene))
    # Stand the player on the centerline beside the barrier so it is in range.
    centre = _rect_center(barriers[0].rect)
    world.player = scene.track_centerline[
        _nearest_point_index(scene.track_centerline, centre)
    ].model_copy()
    observed = [
        entity for entity in world.observe().local_entities
        if entity.get("kind") == "obstacle"
    ]
    assert observed, "a barrier beside the player must appear in local entities"
    assert abs(observed[0]["lane_offset"]) > 20, observed[0]


def test_scene_serializes_behavior_and_fidelity_for_replay_and_research() -> None:
    scene = compile_racing_scene(
        "serialization", plan(npcs=[NpcSpec(profile=NpcProfile.BLOCKER)], grip=.7), seed=3,
    )
    restored = type(scene).model_validate_json(scene.model_dump_json())
    assert restored.grip == .7
    assert restored.npc_behaviors[0].profile == NpcProfile.BLOCKER
    assert restored.track_report is not None
    assert restored.sector_count == scene.sector_count
    assert restored.track_report.angle_fidelity_degrees == scene.track_report.angle_fidelity_degrees

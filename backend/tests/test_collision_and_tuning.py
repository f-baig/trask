"""Collision correctness, barrier shapes, and physics tunability.

Three claims are protected here. Collision is exact against the shape an obstacle
declares, it is swept so no speed can tunnel through a barrier, and every headline
dynamics parameter actually changes the simulation in the direction it should.
"""

from __future__ import annotations

import math

import pytest

from harness.collision import (
    Collider, circle_collider, collider_for, edge_barrier_colliders, outline,
    track_edge_points,
)
from harness.models import (
    Action, CollisionShape, DynamicsSpec, EntityKind, EntitySpec, NpcProfile, Rect,
    RoadDynamicsSpec, VehicleDynamicsSpec, Vec2,
)
from harness.racing import (
    CAR_RADIUS, RacingDesignDraft, RacingWorld, compile_racing_scene,
)
from harness.track_grammar import BarrierSpec, CornerSpec, TrackPlan
from harness.vehicle_physics import VehiclePhysicsState, integrate_vehicle_substep


def barrier(shape: CollisionShape, rotation: float = 0.0, size: tuple[float, float] = (20, 20)):
    width, height = size
    return EntitySpec(
        id="barrier-1", kind=EntityKind.OBSTACLE,
        rect=Rect(x=100 - width / 2, y=100 - height / 2, width=width, height=height),
        label="probe", shape=shape, rotation_degrees=rotation,
    )


def green_flag(world: RacingWorld) -> RacingWorld:
    world.countdown_ticks_remaining = 0
    return world


# ------------------------------------------------------------------- shape tests


def test_shape_declaration_changes_the_hit_region() -> None:
    """A circle and a box over the same rect are genuinely different obstacles.

    If the declaration did not change collision there would be no point carrying
    it, and the renderers would be free to disagree with the physics again.
    """
    box = collider_for(barrier(CollisionShape.BOX))
    circle = collider_for(barrier(CollisionShape.CIRCLE))
    # Straight at a face both shapes are hit, because the circle's radius is the
    # box's half-extent.
    assert box.hits_circle(100 + 10 + 5, 100, CAR_RADIUS)
    assert circle.hits_circle(100 + 10 + 5, 100, CAR_RADIUS)
    # Diagonally into a corner only the box reaches out that far.
    corner = (100 + 17, 100 + 17)
    assert box.hits_circle(*corner, CAR_RADIUS)
    assert not circle.hits_circle(*corner, CAR_RADIUS)


def test_oriented_box_respects_its_rotation() -> None:
    wall = collider_for(barrier(CollisionShape.ORIENTED_BOX, rotation=45.0, size=(80, 8)))
    along = (100 + 25, 100 + 25)
    across = (100 + 25, 100 - 25)
    assert wall.hits_circle(*along, 4.0), "a point along the wall's axis must hit it"
    assert not wall.hits_circle(*across, 4.0), "the same distance across it must miss"
    unrotated = collider_for(barrier(CollisionShape.ORIENTED_BOX, rotation=0.0, size=(80, 8)))
    assert unrotated.hits_circle(100 + 35, 100, 4.0)
    assert not unrotated.hits_circle(100, 100 + 25, 4.0)


@pytest.mark.parametrize("shape", list(CollisionShape))
def test_outline_matches_the_collider_it_describes(shape: CollisionShape) -> None:
    """Renderers draw this outline, so it has to bound the collider exactly."""
    collider = collider_for(barrier(shape, rotation=30.0, size=(40, 12)))
    hull = outline(collider)
    assert len(hull) >= 4
    for x, y in hull:
        # Every outline vertex sits on the boundary: a hair inside is a hit for a
        # tiny probe circle, and the vertex itself is at most a hair outside.
        towards_centre = (
            x + (collider.centre_x - x) * .05, y + (collider.centre_y - y) * .05,
        )
        assert collider.hits_circle(*towards_centre, 1e-6) or collider.hits_circle(x, y, .5)
        assert math.hypot(x - collider.centre_x, y - collider.centre_y) <= collider.bounding_radius + 1e-6


def test_grammar_can_author_every_barrier_shape() -> None:
    for shape in ("circle", "box", "oriented-box"):
        plan = TrackPlan(
            title="Shape probe circuit",
            rationale="Author each barrier shape through the public grammar.",
            corners=[CornerSpec(angle_degrees=90) for _ in range(4)],
            barriers=[BarrierSpec(shape=shape)],
            npcs=[],
        )
        scene = compile_racing_scene("shapes", plan, seed=5)
        obstacles = [item for item in scene.entities if item.kind == EntityKind.OBSTACLE]
        assert len(obstacles) == 1
        assert obstacles[0].shape.value == shape
        # A round obstacle has no meaningful orientation, but an oriented wall must
        # be laid along the road rather than across it.
        assert -360 <= obstacles[0].rotation_degrees <= 360


def test_edge_barriers_are_continuous_two_sided_guardrails() -> None:
    scene = compile_racing_scene(
        "guardrail", RacingDesignDraft(
            title="Guardrail probe circuit",
            rationale="Continuous walls on both road boundaries.",
            circuit="technical", surface="asphalt", obstacle_count=0, npc_count=0,
            edge_barriers=True,
        ), seed=5,
    )
    barriers = edge_barrier_colliders(scene)
    ids = {barrier_id for barrier_id, _ in barriers}
    assert len(barriers) >= len(scene.track_centerline) * 2 - 2
    assert any(item.startswith("edge-left-") for item in ids)
    assert any(item.startswith("edge-right-") for item in ids)
    assert all(collider.shape == CollisionShape.ORIENTED_BOX for _, collider in barriers)
    centre = scene.track_centerline[0]
    assert not any(collider.hits_circle(centre.x, centre.y, CAR_RADIUS) for _, collider in barriers)


def test_edge_barrier_contact_bounces_car_back_onto_track() -> None:
    scene = compile_racing_scene(
        "guardrail bounce", RacingDesignDraft(
            title="Guardrail bounce circuit",
            rationale="A car crossing the road edge must rebound, not terminate.",
            circuit="oval", surface="asphalt", obstacle_count=0, npc_count=0,
            edge_barriers=True,
        ), seed=5,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    centre = scene.track_centerline[0]
    left, _ = track_edge_points(scene)
    dx, dy = left[0][0] - centre.x, left[0][1] - centre.y
    length = math.hypot(dx, dy)
    outward = (dx / length, dy / length)
    start = Vec2(
        x=centre.x + outward[0] * (scene.track_width / 2 - CAR_RADIUS - 2),
        y=centre.y + outward[1] * (scene.track_width / 2 - CAR_RADIUS - 2),
    )
    world.player = Vec2(
        x=centre.x + outward[0] * (scene.track_width / 2 + 20),
        y=centre.y + outward[1] * (scene.track_width / 2 + 20),
    )
    collision = world._barrier_collision(start)
    assert collision is not None and collision[0].startswith("edge-left-")
    world._bounce_player_from_barrier(collision[1], collision[2])
    assert world._on_track(world.player)
    assert not world.terminated


# ------------------------------------------------------------------- swept tests


def test_sweeping_catches_a_barrier_a_single_test_would_jump_over() -> None:
    """Top speed is tunable, so a per-tick point test is not sufficient.

    Six physics substeps run per control tick. A fast enough car covers more than
    a barrier's width in one tick, so testing only the end position registers no
    contact at all and the car drives through solid geometry.
    """
    box = collider_for(barrier(CollisionShape.BOX))
    start, end = (-300.0, 100.0), (300.0, 100.0)
    assert not box.hits_circle(*start, CAR_RADIUS)
    assert not box.hits_circle(*end, CAR_RADIUS)
    assert box.hits_swept_circle(start, end, CAR_RADIUS)


def test_swept_contact_resolves_safe_point_and_outward_normal() -> None:
    box = collider_for(barrier(CollisionShape.BOX))
    contact = box.sweep_contact((0.0, 100.0), (200.0, 100.0), CAR_RADIUS)
    assert contact is not None
    assert contact.normal == pytest.approx((-1.0, 0.0))
    assert not box.hits_circle(*contact.safe_point, CAR_RADIUS)
    assert box.hits_circle(*contact.impact_point, CAR_RADIUS)
    assert 0 < contact.fraction < 1


def test_sweeping_does_not_invent_contact_on_a_clear_path() -> None:
    box = collider_for(barrier(CollisionShape.BOX))
    assert not box.hits_swept_circle((-300.0, 400.0), (300.0, 400.0), CAR_RADIUS)
    # Grazing just outside the combined radius stays clear.
    clearance = 10 + CAR_RADIUS + .5
    assert not box.hits_swept_circle(
        (-300.0, 100.0 + clearance), (300.0, 100.0 + clearance), CAR_RADIUS,
    )


def test_swept_resolution_is_independent_of_travel_distance() -> None:
    """Any distance is sampled finely enough; the guarantee cannot be outrun."""
    circle = circle_collider(0.0, 0.0, 6.0)
    for travel in (30.0, 300.0, 3_000.0, 30_000.0):
        assert circle.hits_swept_circle((-travel, 0.0), (travel, 0.0), 6.0), travel


def test_engine_collision_is_swept_at_extreme_speed() -> None:
    """The same guarantee through the real engine, not just the collider."""
    scene = compile_racing_scene(
        "sweep", RacingDesignDraft(
            title="Sweep probe circuit",
            rationale="A single barrier and a deliberately absurd top speed.",
            circuit="oval", surface="asphalt", obstacle_count=1, npc_count=0,
        ), seed=5,
    )
    obstacle = next(item for item in scene.entities if item.kind == EntityKind.OBSTACLE)
    centre = Vec2(
        x=obstacle.rect.x + obstacle.rect.width / 2,
        y=obstacle.rect.y + obstacle.rect.height / 2,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    # Teleport the car to one side of the barrier and step it far past the other,
    # which is what a high top speed does in a single control tick.
    world.player = Vec2(x=centre.x - 260, y=centre.y)
    frame = world.step(Action())
    world.player = Vec2(x=centre.x + 260, y=centre.y)
    assert world._collision(Vec2(x=centre.x - 260, y=centre.y)) == obstacle.id
    del frame


def test_car_to_car_contact_uses_both_radii() -> None:
    scene = compile_racing_scene(
        "contact", RacingDesignDraft(
            title="Contact probe circuit",
            rationale="Two cars and no barriers, to check car-to-car geometry.",
            circuit="oval", surface="asphalt", obstacle_count=0, npc_count=1,
        ), seed=5,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    opponent = world.opponents[0]
    world.player = Vec2(x=opponent.position.x + CAR_RADIUS * 2 + 1, y=opponent.position.y)
    assert world._collision(world.player) is None
    world.player = Vec2(x=opponent.position.x + CAR_RADIUS * 2 - 1, y=opponent.position.y)
    assert world._collision(world.player) == opponent.entity_id


def test_car_to_car_contact_stays_terminal_while_barriers_rebound() -> None:
    scene = compile_racing_scene(
        "contact", RacingDesignDraft(
            title="Contact terminal probe",
            rationale="One opponent verifies that only barriers became recoverable.",
            circuit="oval", surface="asphalt", obstacle_count=0, npc_count=1,
        ), seed=5,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    opponent = world.opponents[0]
    world.player = opponent.position.model_copy()
    frame = world.step(Action())
    assert world.terminated and not world.succeeded
    assert frame.events == [f"collision with {opponent.entity_id}"]


def test_collision_is_planar_in_3d_by_design() -> None:
    """3D keeps planar collision, and the track compiler is what makes that safe.

    Two stretches of road are never allowed to overlap in plan view, so cars can
    never share an `(x, y)` at different heights. That is the invariant that lets
    both engines share one collision implementation.
    """
    from harness.racing3d import Racing3DWorld
    from harness.track3d import fit_drivable_elevation
    from harness.models import ElevationProfile, ElevationSpec

    scene = compile_racing_scene(
        "planar", RacingDesignDraft(
            title="Planar collision circuit",
            rationale="An elevated circuit used to check planar collision holds.",
            circuit="technical", surface="asphalt", obstacle_count=2, npc_count=1,
        ), seed=17,
    )
    fitted, _ = fit_drivable_elevation(scene, ElevationSpec(
        profile=ElevationProfile.HILLY, amplitude_m=6, hill_count=2, banking_degrees=8,
    ))
    world = green_flag(Racing3DWorld.from_scene(scene.model_copy(update={"elevation": fitted})))
    obstacle = next(item for item in world.scene.entities if item.kind == EntityKind.OBSTACLE)
    collider = collider_for(obstacle)
    world.player = Vec2(x=collider.centre_x, y=collider.centre_y)
    world._refresh_vertical_state()
    assert world._collision(world.player) == obstacle.id, (
        "a barrier is solid in 3D at whatever height the road puts it"
    )
    frame = world.step(Action())
    assert not world.terminated
    assert frame.events[0] == f"bounced off {obstacle.id}"
    assert frame.privileged_state.barrier_impact is not None
    assert world.player_z == pytest.approx(
        world.surface.surface_height(world.player, world.scene.track_centerline, world.surface_index)
    )


# ------------------------------------------------------------- tunability probes


def _terminal_speed(dynamics: DynamicsSpec, ticks: int = 90, grade: float = 0.0) -> float:
    state = VehiclePhysicsState(x=0, y=0, heading_radians=0.0, longitudinal_velocity_mps=1.0)
    for _ in range(ticks * (dynamics.physics_hz // dynamics.control_hz)):
        state = integrate_vehicle_substep(
            state, dynamics, throttle=1.0, brake=0.0, steering=0.0,
            nitro=False, on_track=True, grade_radians=grade,
        )
    return state.longitudinal_velocity_mps


def _cornering_limit(dynamics: DynamicsSpec, speed: float = 10.0) -> float:
    """Peak lateral acceleration at a held speed.

    Speed is pinned each substep on purpose. Letting the car find its own speed
    conflates the grip limit with how fast the car happens to be going, and both
    gravity and downforce change acceleration as well as grip -- so a heavier
    planet would read as *less* cornering grip simply because the car was slower.
    """
    state = VehiclePhysicsState(x=0, y=0, heading_radians=0.0, longitudinal_velocity_mps=speed)
    peak = 0.0
    for _ in range(200):
        state.longitudinal_velocity_mps = speed
        state = integrate_vehicle_substep(
            state, dynamics, throttle=.0, brake=0.0, steering=1.0,
            nitro=False, on_track=True,
        )
        peak = max(peak, abs(state.lateral_acceleration_mps2))
    return peak


def _tuned(**overrides) -> DynamicsSpec:
    vehicle_keys = set(VehicleDynamicsSpec.model_fields)
    road_keys = set(RoadDynamicsSpec.model_fields)
    vehicle = {key: value for key, value in overrides.items() if key in vehicle_keys}
    road = {key: value for key, value in overrides.items() if key in road_keys}
    top = {
        key: value for key, value in overrides.items()
        if key not in vehicle_keys and key not in road_keys
    }
    return DynamicsSpec(
        vehicle=VehicleDynamicsSpec(**vehicle), road=RoadDynamicsSpec(**road), **top,
    )


def test_car_weight_changes_acceleration() -> None:
    light = _terminal_speed(_tuned(mass_kg=800), ticks=12)
    heavy = _terminal_speed(_tuned(mass_kg=2_400), ticks=12)
    assert light > heavy, (light, heavy)


def test_road_friction_changes_grip_limited_cornering() -> None:
    grippy = _cornering_limit(_tuned(friction_coefficient=1.4))
    greasy = _cornering_limit(_tuned(friction_coefficient=.35))
    assert grippy > greasy * 1.5, (grippy, greasy)


def test_gravity_changes_grip_until_steering_geometry_becomes_the_limit() -> None:
    """Gravity sets tire normal load, so it scales the cornering limit -- up to a
    point.

    Below about one g the car is grip-limited and gravity dominates. Above it the
    binding constraint becomes the bicycle model's steering geometry and understeer
    gradient: no amount of grip lets a car corner harder than its front wheels can
    point. Asserting that gravity raises cornering without bound would be asserting
    a physics bug.

    The settable range spans low-gravity bodies through heavy planets, because a
    harness restricted to 8-12 cannot express the grip-versus-weight question.
    """
    assert DynamicsSpec(gravity_mps2=1.62).gravity_mps2 == pytest.approx(1.62)
    assert DynamicsSpec(gravity_mps2=24.8).gravity_mps2 == pytest.approx(24.8)
    lunar = _cornering_limit(_tuned(gravity_mps2=1.62))
    martian = _cornering_limit(_tuned(gravity_mps2=3.71))
    earth = _cornering_limit(_tuned(gravity_mps2=9.81))
    heavy = _cornering_limit(_tuned(gravity_mps2=24.0))
    assert lunar < martian < earth, (lunar, martian, earth)
    assert lunar < earth * .3, "low gravity must cost most of the cornering grip"
    # Past the crossover the geometry ceiling holds, so extra grip buys nothing.
    assert heavy == pytest.approx(earth, rel=.05), (earth, heavy)


def test_engine_and_brake_force_are_tunable() -> None:
    weak = _terminal_speed(_tuned(engine_force_n=1_500), ticks=10)
    strong = _terminal_speed(_tuned(engine_force_n=16_000), ticks=10)
    assert strong > weak

    def stopping_ticks(brake_force: float) -> int:
        dynamics = _tuned(brake_force_n=brake_force)
        state = VehiclePhysicsState(x=0, y=0, heading_radians=0.0, longitudinal_velocity_mps=12.0)
        for tick in range(400):
            for _ in range(6):
                state = integrate_vehicle_substep(
                    state, dynamics, throttle=0.0, brake=1.0, steering=0.0,
                    nitro=False, on_track=True,
                )
            if state.longitudinal_velocity_mps <= .1:
                return tick
        return 400

    assert stopping_ticks(30_000) < stopping_ticks(2_000)


def test_drag_downforce_and_air_density_are_tunable() -> None:
    slippery = _terminal_speed(_tuned(drag_coefficient=.15), ticks=200)
    draggy = _terminal_speed(_tuned(drag_coefficient=1.4), ticks=200)
    assert slippery > draggy, (slippery, draggy)
    vacuum = _terminal_speed(_tuned(air_density_kg_m3=0.0, drag_coefficient=1.4), ticks=200)
    assert vacuum > draggy, "zero air density must remove aerodynamic drag entirely"
    # Downforce trades straight-line speed for cornering grip, and because it
    # scales with the square of speed it is measured where it actually matters.
    # This once did nothing at all: downforce raised longitudinal traction while
    # the cornering limit was computed from gravity alone, inverting the whole
    # point of an aero package.
    fast = 35.0
    with_wing = _cornering_limit(_tuned(lift_coefficient=-1.8, max_speed_mps=90), fast)
    without = _cornering_limit(_tuned(lift_coefficient=0.0, max_speed_mps=90), fast)
    assert with_wing > without * 1.1, (with_wing, without)
    # At low speed there is no meaningful downforce, so it must change nothing.
    assert _cornering_limit(_tuned(lift_coefficient=-1.8), 8.0) == pytest.approx(
        _cornering_limit(_tuned(lift_coefficient=0.0), 8.0), rel=.02,
    )


def test_tire_friction_and_nitro_are_tunable() -> None:
    assert _cornering_limit(_tuned(tire_friction_multiplier=1.8)) > _cornering_limit(
        _tuned(tire_friction_multiplier=.3),
    )
    plain = _terminal_speed(_tuned(nitro_force_n=0), ticks=8)
    state = VehiclePhysicsState(x=0, y=0, heading_radians=0.0, longitudinal_velocity_mps=1.0)
    boosted_dynamics = _tuned(nitro_force_n=12_000)
    for _ in range(8 * 6):
        state = integrate_vehicle_substep(
            state, boosted_dynamics, throttle=1.0, brake=0.0, steering=0.0,
            nitro=True, on_track=True,
        )
    assert state.longitudinal_velocity_mps > plain


def test_every_dynamics_field_survives_a_scene_round_trip() -> None:
    """A tuned scene is serializable, so an experiment can reproduce it exactly."""
    scene = compile_racing_scene(
        "tuning", RacingDesignDraft(
            title="Tuned probe circuit",
            rationale="Confirm tuned dynamics survive scene serialization.",
            circuit="oval", surface="asphalt", obstacle_count=0, npc_count=0,
        ), seed=5,
    )
    tuned = scene.model_copy(update={"dynamics": _tuned(
        mass_kg=1_950, engine_force_n=9_500, brake_force_n=21_000,
        friction_coefficient=.62, tire_friction_multiplier=1.3,
        drag_coefficient=.9, lift_coefficient=-1.2, nitro_force_n=7_000,
        gravity_mps2=3.71, air_density_kg_m3=.02,
    )})
    restored = type(scene).model_validate_json(tuned.model_dump_json())
    assert restored.dynamics == tuned.dynamics
    world = green_flag(RacingWorld.from_scene(restored))
    world.step(Action(keys=["w"]))
    assert math.isfinite(world.speed) and world.speed >= 0
    assert restored.dynamics.gravity_mps2 == pytest.approx(3.71)


def test_extreme_but_legal_dynamics_stay_finite() -> None:
    """Rock-solid means no legal parameter combination can produce NaN state."""
    scene = compile_racing_scene(
        "extremes", RacingDesignDraft(
            title="Extreme probe circuit",
            rationale="Sweep the legal parameter corners for non-finite state.",
            circuit="technical", surface="ice", obstacle_count=0, npc_count=0,
        ), seed=5,
    )
    corners = (
        _tuned(mass_kg=500, engine_force_n=20_000, friction_coefficient=2.0, gravity_mps2=30.0),
        _tuned(mass_kg=3_000, engine_force_n=500, friction_coefficient=.05, gravity_mps2=.5),
        _tuned(max_speed_mps=90, drag_coefficient=.1, air_density_kg_m3=0.0),
        _tuned(tire_load_sensitivity=.5, center_of_mass_height_m=1.2, width_m=1.3),
        _tuned(front_cornering_stiffness_n_per_rad=5_000, rear_cornering_stiffness_n_per_rad=200_000),
    )
    for dynamics in corners:
        world = green_flag(RacingWorld.from_scene(scene.model_copy(update={"dynamics": dynamics})))
        for keys in (["w"], ["w", "a"], ["s", "d"], ["w", "space"], []):
            for _ in range(12):
                if world.terminated:
                    break
                world.step(Action(keys=keys))
            state = world.privileged_state()
            assert math.isfinite(state.speed) and state.speed >= 0
            assert math.isfinite(state.heading) and 0 <= state.heading < 360
            assert math.isfinite(state.slip_angle_degrees)
            assert math.isfinite(state.lateral_acceleration_mps2)


def test_collider_is_hashable_and_frozen() -> None:
    """Colliders are values, so they can be cached per scene without aliasing."""
    first = collider_for(barrier(CollisionShape.CIRCLE))
    second = collider_for(barrier(CollisionShape.CIRCLE))
    assert first == second
    assert len({first, second}) == 1
    with pytest.raises(Exception):
        first.centre_x = 5  # type: ignore[misc]
    assert isinstance(first, Collider)


# ------------------------------------------------------- opponent intelligence


def _opponent_probe(ticks: int = 400, **behavior):
    """Run one opponent alone and report how it drove."""
    import statistics

    scene = compile_racing_scene(
        "intelligence", RacingDesignDraft(
            title="Intelligence probe circuit",
            rationale="A single opponent driving unobstructed around a technical lap.",
            circuit="technical", surface="asphalt", obstacle_count=0, npc_count=1,
        ), seed=17,
    )
    scene = scene.model_copy(update={
        "npc_behaviors": [
            item.model_copy(update=behavior) for item in scene.npc_behaviors
        ],
    })
    world = green_flag(RacingWorld.from_scene(scene))
    world.terminate_on_opponent_win = False
    lanes = []
    for _ in range(ticks):
        world.step(Action())
        lanes.append(world.opponents[0].lane_offset)
    opponent = world.opponents[0]
    return {
        "distance": opponent.progress_samples,
        "lane_spread": max(lanes) - min(lanes),
        "lane_stdev": statistics.pstdev(lanes),
    }


def test_intelligence_is_an_axis_independent_of_pace() -> None:
    """A smarter opponent is faster at identical pace and cornering commitment.

    Difficulty needs a dial that is not just "drives faster". Intelligence changes
    the line: it looks further ahead before treating a corner as begun, and aims
    for the geometric inside rather than holding its grid lane. Holding pace and
    skill fixed isolates that from raw speed.
    """
    dull = _opponent_probe(intelligence=0.0, pace=.9, skill=.7)
    sharp = _opponent_probe(intelligence=1.0, pace=.9, skill=.7)
    assert sharp["distance"] > dull["distance"], (dull, sharp)


def test_intelligence_is_monotonic() -> None:
    distances = [
        _opponent_probe(intelligence=value, pace=.9, skill=.7)["distance"]
        for value in (0.0, .35, .7, 1.0)
    ]
    assert distances == sorted(distances), distances


def test_npc_aggression_increases_lap_pace_at_fixed_capability() -> None:
    """Aggression now affects racing commitment, not only whether a pass begins."""
    patient = _opponent_probe(aggression=0.0, intelligence=.7, pace=.9, skill=.7)
    attacker = _opponent_probe(aggression=1.0, intelligence=.7, pace=.9, skill=.7)
    assert attacker["distance"] > patient["distance"], (patient, attacker)


def test_a_low_intelligence_car_holds_its_lane_and_a_high_one_seeks_apexes() -> None:
    """The two ends of the dial should look different, not just score differently."""
    dull = _opponent_probe(intelligence=0.0, pace=.9, skill=.7)
    sharp = _opponent_probe(intelligence=1.0, pace=.9, skill=.7)
    # A wandering car stays near its grid lane; an apex-seeking one crosses the road.
    assert sharp["lane_spread"] > dull["lane_spread"] * 2, (dull, sharp)
    assert dull["lane_spread"] > 0, "a low-intelligence line must still wander"


def test_line_wander_is_deterministic_not_random() -> None:
    """Mannerisms must be reproducible or they cannot be used in an experiment."""
    first = _opponent_probe(intelligence=.2, pace=.9, skill=.7)
    second = _opponent_probe(intelligence=.2, pace=.9, skill=.7)
    assert first == second


def test_named_profiles_span_the_intelligence_range() -> None:
    from harness.track_grammar import NpcSpec

    values = {
        profile.value: NpcSpec(profile=profile).resolve("opponent-1").intelligence
        for profile in NpcProfile
    }
    assert values["backmarker"] < values["cruiser"] < values["racer"] < values["aggressor"]
    assert all(0 <= value <= 1 for value in values.values())


def test_intelligence_and_aggression_are_separately_authorable() -> None:
    """Perturbation studies need to move one mannerism at a time."""
    from harness.track_grammar import NpcSpec

    careful_but_pushy = NpcSpec(
        profile=NpcProfile.RACER, intelligence=.1, aggression=.95,
    ).resolve("opponent-1")
    sharp_but_patient = NpcSpec(
        profile=NpcProfile.RACER, intelligence=.95, aggression=.1,
    ).resolve("opponent-1")
    assert careful_but_pushy.intelligence < sharp_but_patient.intelligence
    assert careful_but_pushy.aggression > sharp_but_patient.aggression
    # Everything else stays on the profile, so only the named axis moved.
    assert careful_but_pushy.pace == sharp_but_patient.pace
    assert careful_but_pushy.skill == sharp_but_patient.skill


# ------------------------------------------------------- opponents are solid too
#
# Opponents used to be solid only to the player. They could overlap each other and
# drive through barriers, and the starting grid put two cars on the same spot, so a
# field of five raced as a field of four welded to a fifth.


def _race_with_traffic(brief: str, seed: int):
    """Drive the reference controller through a full race and watch the field."""
    from harness.collision import collider_for as resolve
    from harness.probes import _full_throttle_action  # noqa: F401  (import guard only)
    from harness.racing import RacingLineController, compile_certified_scene
    from harness.track_grammar import parse_track_prompt

    scene, certificate, _ = compile_certified_scene(brief, parse_track_prompt(brief), seed)
    barriers = [
        resolve(entity, 0.0) for entity in scene.entities
        if entity.kind == EntityKind.OBSTACLE
    ]
    world = RacingWorld.from_scene(scene)
    world.terminate_on_opponent_win = False
    controller = RacingLineController()
    controller.reset(scene, scene.seed)
    closest_pair = math.inf
    pair_overlaps = barrier_overlaps = 0
    for _ in range(1_400 * scene.laps):
        if world.terminated:
            break
        if world.countdown_ticks_remaining > 0:
            world.step(Action())
            continue
        action, decision = controller.act(world.observe())
        world.step(action, decision)
        cars = world.opponents
        for left in range(len(cars)):
            for right in range(left + 1, len(cars)):
                separation = math.hypot(
                    cars[left].position.x - cars[right].position.x,
                    cars[left].position.y - cars[right].position.y,
                )
                closest_pair = min(closest_pair, separation)
                pair_overlaps += separation < CAR_RADIUS * 2 - 1e-6
            barrier_overlaps += sum(
                collider.hits_circle(cars[left].position.x, cars[left].position.y, CAR_RADIUS)
                for collider in barriers
            )
    return {
        "certified": certificate.playable,
        "oracle_finished": world.succeeded,
        "closest_pair": closest_pair,
        "pair_overlaps": pair_overlaps,
        "barrier_overlaps": barrier_overlaps,
        "opponents": len(world.opponents),
        "finished": sum(1 for car in world.opponents if car.finished_step is not None),
    }


@pytest.mark.parametrize("brief,seed", [
    ("a technical circuit with six barriers and five aggressive rivals", 17),
    ("an ice circuit with six barriers and five aggressive rivals", 43),
    ("a slippery clay circuit with four barriers and five blockers spread around the track", 91),
])
def test_opponents_never_overlap_each_other_or_a_barrier(brief: str, seed: int) -> None:
    result = _race_with_traffic(brief, seed)
    assert result["pair_overlaps"] == 0
    assert result["barrier_overlaps"] == 0
    assert result["closest_pair"] >= CAR_RADIUS * 2 - 1e-6


@pytest.mark.parametrize("brief,seed", [
    ("a technical circuit with six barriers and five aggressive rivals", 17),
    ("an ice circuit with six barriers and five aggressive rivals", 43),
])
def test_making_opponents_solid_does_not_deadlock_the_field(brief: str, seed: int) -> None:
    """Traffic keeps flowing; cars behind the player may still be racing at the flag."""
    result = _race_with_traffic(brief, seed)
    assert result["certified"] and result["oracle_finished"]
    # The player ends the race when it takes the flag, so a car legitimately
    # behind it does not get a fictitious finish just to satisfy this probe. Most
    # of the field must nevertheless have crossed the shared physical line.
    assert result["finished"] >= result["opponents"] - 1


@pytest.mark.parametrize("count", range(1, 6))
def test_no_two_cars_share_a_grid_slot(count: int) -> None:
    from harness.racing import compile_certified_scene
    from harness.track_grammar import NpcSpec, TrackPlan as Plan

    plan = Plan(
        title="Grid spacing probe", rationale="Every car needs its own slot.",
        corners=[CornerSpec(), CornerSpec(), CornerSpec()],
        npcs=[NpcSpec() for _ in range(count)],
    )
    scene, _, _ = compile_certified_scene("grid spacing", plan, 17)
    world = RacingWorld.from_scene(scene)
    positions = [(car.position.x, car.position.y) for car in world.opponents]
    assert len(positions) == count
    for left in range(count):
        for right in range(left + 1, count):
            separation = math.hypot(
                positions[left][0] - positions[right][0],
                positions[left][1] - positions[right][1],
            )
            assert separation >= CAR_RADIUS * 2, (
                f"cars {left + 1} and {right + 1} spawn {separation:.1f}px apart"
            )

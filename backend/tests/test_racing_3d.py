"""The 3D engine, its surface, and the agent-facing view component.

The first test is the most important one in this file: with a flat surface the 3D
engine must reduce *exactly* to the 2D engine. That is what makes "the same game
in 3D" a checkable claim rather than a description, and it guards the one seam
between them — the `road_attitude` hook — against silently changing the 2D game.
"""

from __future__ import annotations

import math

import pytest

from harness.models import (
    Action, ActionName, ElevationProfile, ElevationSpec, NpcProfile, Vec2,
)
from harness.policy_protocol import CameraContract, VisualFrame
from harness.racing import (
    RacingDesignDraft, RacingLineController, RacingWorld, compile_racing_scene,
)
from harness.racing3d import (
    Racing3DWorld, compile_racing_3d_scene, verify_racing_3d_playability,
)
from harness.track3d import (
    MAX_DRIVABLE_GRADE_DEGREES, compile_track_surface, drivable_grade_limit,
    fit_drivable_elevation, validate_track_surface,
)
from harness.track_grammar import archetype_plan
from harness.vehicle_physics import (
    VehiclePhysicsState, _banking_gain, integrate_vehicle_substep,
)
from harness.view3d import (
    NEAR_PLANE, ViewMode, _clip_near, _clip_viewport, _IN_CAR_MODES, _Projector,
    _edge_barrier_faces, camera_for, render_policy_view, render_view_surface,
)
from harness.reflex.runtime import ReflexRuntime
from harness.reflex.visual_3d import PerspectiveVisionSense
from harness.reflex.tools import tool_schemas
from harness.policies import PerspectiveVisualTickPolicy, built_in_policies


FLAT = ElevationSpec(profile=ElevationProfile.FLAT, amplitude_m=0, banking_degrees=0)


def draft(circuit: str = "technical", surface: str = "asphalt", obstacles: int = 2, npcs: int = 2):
    return RacingDesignDraft(
        title=f"{circuit.title()} 3D test circuit",
        rationale="A deterministic circuit used to exercise the 3D contract.",
        circuit=circuit, surface=surface, obstacle_count=obstacles, npc_count=npcs,
    )


def scene_for(circuit: str = "technical", surface: str = "asphalt", **kwargs):
    return compile_racing_scene("3d test", draft(circuit, surface, **kwargs), seed=17)


def elevated(
    circuit: str = "technical", surface: str = "asphalt",
    obstacles: int = 2, npcs: int = 2, **elevation,
):
    scene = scene_for(circuit, surface, obstacles=obstacles, npcs=npcs)
    spec = ElevationSpec(**{
        "profile": ElevationProfile.ROLLING, "amplitude_m": 5.0,
        "hill_count": 2, "banking_degrees": 7.0, **elevation,
    })
    fitted, _ = fit_drivable_elevation(scene, spec)
    return scene.model_copy(update={"elevation": fitted})


def green_flag(world):
    world.countdown_ticks_remaining = 0
    return world


def test_npc_win_terminates_an_elevated_race_too() -> None:
    world = green_flag(Racing3DWorld.from_scene(elevated(obstacles=0, npcs=1)))
    opponent = world.opponents[0]
    before_finish = (world.scene.start_line_index - 1) % len(world.scene.track_centerline)
    opponent.position = world.scene.track_centerline[before_finish].model_copy()
    opponent.track_index = before_finish
    opponent.target_index = world.scene.start_line_index
    opponent.speed = 30.0
    opponent.progress_samples = len(world.scene.track_centerline) * world.scene.laps
    frame = world.step(Action())
    assert world.terminated and not world.succeeded
    assert frame.events == [f"{opponent.entity_id} finished first"]


def test_3d_reflex_uses_the_perspective_visual_contract_only() -> None:
    """A 3D scene selects perspective image cues, never 2D cone or telemetry fields."""
    world = Racing3DWorld.from_scene(elevated())
    runtime = ReflexRuntime(world.scene, vision_only=True)
    values = runtime.observe_visual(world)
    assert runtime.visual_mode == "3d"
    assert runtime.visible_fields == frozenset({
        "vision_track_offset", "vision_track_heading", "vision_bend_ahead",
        "vision_bend_severity", "vision_visible_depth", "vision_left_gap",
        "vision_right_gap", "vision_road_contact", "vision_recovery_direction",
        "vision_road_horizon", "vision_horizon_shift",
        "vision_crest_risk", "vision_confidence", "speed",
    })
    assert runtime.visible_fields <= values.keys()
    assert "speed" in runtime.visible_fields
    assert "vision_center_near" not in runtime.visible_fields
    assert "inspect_perspective_road" in {tool["name"] for tool in tool_schemas(visual_mode="3d")}
    assert "inspect_cone" not in {tool["name"] for tool in tool_schemas(visual_mode="3d")}
    installed = runtime.install(
        name="perspective", reads=["vision_track_offset", "speed"],
        source=(
            "def control(sense, ctrl, out):\n"
            "    out.steer(-sense.vision_track_offset)\n"
            "    out.throttle(0.4 if sense.speed < 4 else 0.1)\n"
        ),
    )
    assert installed["installed"], installed


@pytest.mark.parametrize("surface", ["asphalt", "clay", "ice"])
def test_perspective_sensor_calibrates_road_colour_from_its_camera_frame(surface: str) -> None:
    """Valid non-asphalt 3D environments remain visible to the default player."""
    world = Racing3DWorld.from_scene(elevated(surface=surface, obstacles=0, npcs=0))
    values = PerspectiveVisionSense().update(
        render_policy_view(world, mode=ViewMode.FIRST_PERSON),
    )

    assert values["vision_road_contact"]
    assert values["vision_confidence"] >= .7


def test_3d_visual_control_calibration_is_episode_local_and_camera_only() -> None:
    world = Racing3DWorld.from_scene(elevated())
    runtime = ReflexRuntime(world.scene, vision_only=True)
    runtime.observe_visual(world)
    report = runtime.calibrate_perspective_controls(world, ticks=2)
    assert set(report["results"]) == {"left", "straight", "right"}
    assert "episode-local" in report["validity"]
    for result in report["results"].values():
            assert set(result) == {"after", "delta", "road_contact"}
            assert "vision_track_offset" in result["delta"]
            assert "speed" in result["after"]


def test_direct_3d_visual_policy_renders_first_person_without_telemetry() -> None:
    world = Racing3DWorld.from_scene(elevated())
    policy = PerspectiveVisualTickPolicy()
    policy.reset(world.scene, world.scene.seed)
    frame = policy.render_frame(world)
    assert frame.viewpoint == ViewMode.FIRST_PERSON
    assert frame.ego_anchor == "camera-relative"
    assert "vision-3d-direct-every-tick" in built_in_policies()
    assert "vision-3d-direct-short" in built_in_policies()
    assert "vision-3d-direct-short-features" in built_in_policies()
    assert "vision-3d-predictive-skills" in built_in_policies()


# --------------------------------------------------------------- the 2D contract


@pytest.mark.parametrize("circuit", ["oval", "technical", "chicane"])
@pytest.mark.parametrize("surface", ["asphalt", "clay", "ice"])
def test_flat_3d_engine_reproduces_the_2d_engine_exactly(circuit: str, surface: str) -> None:
    """A flat 3D world and the 2D world are the same simulation, tick for tick.

    Both run the identical controller against the identical scene. Any divergence
    means the elevation hook leaked into the planar physics, which would make the
    2D game a different game than it was before 3D existed.
    """
    scene = scene_for(circuit, surface)
    flat_scene = scene.model_copy(update={"elevation": FLAT})
    planar = RacingWorld.from_scene(scene)
    spatial = Racing3DWorld.from_scene(flat_scene)
    planar_controller, spatial_controller = RacingLineController(), RacingLineController()
    planar_controller.reset(scene, scene.seed)
    spatial_controller.reset(flat_scene, flat_scene.seed)
    for step in range(400):
        if planar.terminated or spatial.terminated:
            break
        planar.step(*planar_controller.act(planar.observe()))
        spatial.step(*spatial_controller.act(spatial.observe()))
        assert planar.player.x == spatial.player.x, step
        assert planar.player.y == spatial.player.y, step
        assert planar.heading == spatial.heading, step
        assert planar.speed == spatial.speed, step
    planar_state = planar.snapshot()
    spatial_state = spatial.snapshot()
    assert {key: spatial_state[key] for key in planar_state} == planar_state
    assert planar.terminated == spatial.terminated
    assert planar.objective_index == spatial.objective_index


def test_the_2d_engine_reports_a_flat_road() -> None:
    """The planar engine's attitude hook is the seam, and it must stay flat."""
    scene = scene_for()
    world = RacingWorld.from_scene(scene)
    for point in scene.track_centerline[:12]:
        assert world.road_attitude(point) == (0.0, 0.0)


def test_omitting_slope_matches_passing_zero_slope() -> None:
    """Default arguments must reproduce the planar integrator bit for bit."""
    scene = scene_for()
    state = VehiclePhysicsState(x=0, y=0, heading_radians=0.3, longitudinal_velocity_mps=9.0)
    common = {"throttle": 1.0, "brake": 0.0, "steering": .5, "nitro": False, "on_track": True}
    without = integrate_vehicle_substep(state, scene.dynamics, **common)
    with_zero = integrate_vehicle_substep(
        state, scene.dynamics, **common, grade_radians=0.0, bank_radians=0.0,
    )
    assert without == with_zero


# ------------------------------------------------------------------ the surface


@pytest.mark.parametrize("profile", list(ElevationProfile))
@pytest.mark.parametrize("hills", [1, 3, 5, 8])
def test_elevation_closes_exactly_over_one_lap(profile: ElevationProfile, hills: int) -> None:
    """A circuit is a loop, so its height profile must be exactly periodic.

    A profile that gained even a little height per lap would be an invisible
    cliff at the start/finish line, hit once every lap.
    """
    surface = compile_track_surface(scene_for(), ElevationSpec(
        profile=profile, amplitude_m=8.0, hill_count=hills, banking_degrees=6,
    ))
    assert abs(surface.seam_step) < 1e-6


def test_elevation_is_deterministic() -> None:
    scene = scene_for()
    spec = ElevationSpec(profile=ElevationProfile.HILLY, amplitude_m=6, hill_count=3)
    first = compile_track_surface(scene, spec)
    second = compile_track_surface(scene, spec)
    assert first.heights == second.heights
    assert first.banks == second.banks
    assert first.grades == second.grades


def test_profile_names_only_supply_continuous_shape_defaults() -> None:
    rolling = ElevationSpec(profile=ElevationProfile.ROLLING)
    alpine = ElevationSpec(profile=ElevationProfile.ALPINE)
    explicit = ElevationSpec(profile=ElevationProfile.ALPINE, crest_sharpness=.333)
    assert rolling.crest_sharpness == pytest.approx(.18)
    assert alpine.crest_sharpness == pytest.approx(.78)
    assert explicit.crest_sharpness == pytest.approx(.333)

    scene = scene_for()
    smooth = compile_track_surface(scene, ElevationSpec(
        profile=ElevationProfile.HILLY, amplitude_m=6, hill_count=3,
        crest_sharpness=.47,
    ))
    slightly_sharper = compile_track_surface(scene, ElevationSpec(
        profile=ElevationProfile.HILLY, amplitude_m=6, hill_count=3,
        crest_sharpness=.48,
    ))
    assert smooth.heights != slightly_sharper.heights


def test_3d_edge_guardrail_faces_follow_the_elevated_surface() -> None:
    scene = scene_for(obstacles=0, npcs=0).model_copy(update={"edge_barriers": True})
    scene = scene.model_copy(update={"elevation": ElevationSpec(
        profile=ElevationProfile.HILLY, amplitude_m=5, hill_count=2,
        banking_degrees=8, crest_sharpness=.61,
    )})
    world = Racing3DWorld.from_scene(scene)
    faces = _edge_barrier_faces(world, list(range(len(scene.track_centerline))))
    assert len(faces) == len(scene.track_centerline) * 2 * 3
    assert max(point[2] for face, _, _ in faces for point in face) > 20


def test_climbable_gradient_depends_on_available_grip() -> None:
    """Climbing is traction-limited, so an ice circuit cannot be as steep.

    Holding the car's weight against a slope spends `sin(grade)` of the available
    friction. Treating the limit as one global number let low-grip circuits
    compile into a climb the oracle could only creep up until it ran out of steps.
    """
    limits = {
        surface: drivable_grade_limit(scene_for(surface=surface))
        for surface in ("asphalt", "clay", "ice")
    }
    assert limits["ice"] < limits["clay"] <= limits["asphalt"], limits
    assert all(0 < limit <= MAX_DRIVABLE_GRADE_DEGREES for limit in limits.values())
    steep = ElevationSpec(profile=ElevationProfile.ROLLING, amplitude_m=6, hill_count=3)
    ice_scene = scene_for(surface="ice")
    assert validate_track_surface(compile_track_surface(ice_scene, steep)) != []
    fitted, notes = fit_drivable_elevation(ice_scene, steep)
    assert validate_track_surface(compile_track_surface(ice_scene, fitted)) == []
    assert notes


def test_undrivable_gradients_are_rejected_then_fitted() -> None:
    scene = scene_for()
    absurd = ElevationSpec(
        profile=ElevationProfile.ALPINE, amplitude_m=40, hill_count=8, banking_degrees=20,
    )
    assert validate_track_surface(compile_track_surface(scene, absurd)) != []
    fitted, notes = fit_drivable_elevation(scene, absurd)
    assert validate_track_surface(compile_track_surface(scene, fitted)) == []
    assert notes, "a reduced elevation request must say what it gave up"


def test_fitting_stretches_crests_before_it_flattens_them() -> None:
    """Relief is what makes a 3D circuit worth driving, so it is kept if possible."""
    scene = scene_for()
    requested = ElevationSpec(
        profile=ElevationProfile.ROLLING, amplitude_m=7, hill_count=6, banking_degrees=6,
    )
    fitted, _ = fit_drivable_elevation(scene, requested)
    assert fitted.hill_count < requested.hill_count
    assert fitted.amplitude_m == requested.amplitude_m


def test_banking_leans_into_corners_and_stays_flat_on_straights() -> None:
    scene = scene_for(circuit="oval")
    surface = compile_track_surface(scene, ElevationSpec(
        profile=ElevationProfile.FLAT, amplitude_m=0, banking_degrees=9,
    ))
    curvature = [
        abs(_sample_curvature(scene.track_centerline, index))
        for index in range(surface.sample_count)
    ]
    banked = max(range(surface.sample_count), key=lambda index: curvature[index])
    flat = min(range(surface.sample_count), key=lambda index: curvature[index])
    assert abs(surface.bank_at_index(banked)) > abs(surface.bank_at_index(flat))
    assert surface.steepest_bank_degrees <= 9.0 + 1e-6


def test_road_height_follows_the_bank_across_the_corridor() -> None:
    scene = scene_for(circuit="oval")
    surface = compile_track_surface(scene, ElevationSpec(
        profile=ElevationProfile.FLAT, amplitude_m=0, banking_degrees=12,
    ))
    index = max(
        range(surface.sample_count),
        key=lambda candidate: abs(surface.bank_at_index(candidate)),
    )
    left, right = surface.edge_heights(index)
    assert abs(left - right) > 1.0, "a banked corner must have one edge higher"


def _sample_curvature(centerline, index: int) -> float:
    count = len(centerline)
    before, current, after = (
        centerline[(index - 2) % count], centerline[index], centerline[(index + 2) % count],
    )
    first = math.degrees(math.atan2(current.y - before.y, current.x - before.x))
    second = math.degrees(math.atan2(after.y - current.y, after.x - current.x))
    return (second - first + 180) % 360 - 180


# ------------------------------------------------------------------ 3D physics


def test_gradient_slows_the_climb_and_speeds_the_descent() -> None:
    """Gravity along the road is real force, not a rendering trick."""
    scene = scene_for(obstacles=0, npcs=0)
    outcomes: dict[str, float] = {}
    for label, grade in (("uphill", math.radians(12)), ("level", 0.0), ("downhill", math.radians(-12))):
        state = VehiclePhysicsState(x=0, y=0, heading_radians=0.0, longitudinal_velocity_mps=6.0)
        for _ in range(60):
            state = integrate_vehicle_substep(
                state, scene.dynamics, throttle=1.0, brake=0.0, steering=0.0,
                nitro=False, on_track=True, grade_radians=grade,
            )
        outcomes[label] = state.longitudinal_velocity_mps
    assert outcomes["uphill"] < outcomes["level"] < outcomes["downhill"], outcomes


def test_banking_raises_the_cornering_limit_and_is_bounded() -> None:
    assert _banking_gain(0.0, 1.0) == 1.0
    assert _banking_gain(math.radians(8), 1.0) > 1.0
    assert _banking_gain(math.radians(16), 1.0) > _banking_gain(math.radians(8), 1.0)
    # Banking helps regardless of which way the road leans, because a circuit is
    # compiled banked in the direction its corner turns.
    assert _banking_gain(math.radians(-10), 1.0) == _banking_gain(math.radians(10), 1.0)
    assert _banking_gain(math.radians(60), 1.6) <= 1.75


def test_cars_sit_on_the_road_surface() -> None:
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    controller = RacingLineController()
    controller.reset(world.scene, world.scene.seed)
    for _ in range(150):
        if world.terminated:
            break
        world.step(*controller.act(world.observe()))
        expected = world.surface.surface_height(
            world.player, world.scene.track_centerline, world.surface_index,
        )
        assert world.player_z == pytest.approx(expected, abs=1e-6)
        for pose in world.opponent_poses.values():
            assert math.isfinite(pose.z)


def test_chassis_attitude_tracks_the_road_and_the_load() -> None:
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    controller = RacingLineController()
    controller.reset(world.scene, world.scene.seed)
    grades, pitches = [], []
    for _ in range(200):
        if world.terminated:
            break
        world.step(*controller.act(world.observe()))
        grades.append(world.grade_degrees)
        pitches.append(world.pitch_degrees)
        assert abs(world.pitch_degrees - world.grade_degrees) <= 6.001
        assert abs(world.roll_degrees - world.bank_degrees) <= 8.001
    assert max(grades) - min(grades) > 4, "an elevated circuit must change gradient"
    assert max(pitches) - min(pitches) > 4


def test_3d_rollouts_are_deterministic() -> None:
    scene = elevated()
    fingerprints = []
    for _ in range(2):
        world = Racing3DWorld.from_scene(scene)
        controller = RacingLineController()
        controller.reset(scene, scene.seed)
        for _ in range(220):
            if world.terminated:
                break
            world.step(*controller.act(world.observe()))
        fingerprints.append(world.snapshot())
    assert fingerprints[0] == fingerprints[1]


def test_restore_recomputes_height_from_the_road() -> None:
    """A replayed height can never disagree with the road under the car."""
    scene = elevated()
    world = green_flag(Racing3DWorld.from_scene(scene))
    controller = RacingLineController()
    controller.reset(scene, scene.seed)
    for _ in range(90):
        world.step(*controller.act(world.observe()))
    saved = world.snapshot()
    forked = Racing3DWorld.from_scene(scene)
    forked.restore(saved)
    assert forked.player_z == pytest.approx(world.player_z, abs=1e-9)
    assert forked.pitch_degrees == pytest.approx(world.pitch_degrees, abs=1e-9)
    assert forked.roll_degrees == pytest.approx(world.roll_degrees, abs=1e-9)


@pytest.mark.parametrize("circuit", ["oval", "technical", "chicane"])
@pytest.mark.parametrize("surface", ["asphalt", "clay", "ice"])
def test_the_oracle_completes_every_elevated_circuit(circuit: str, surface: str) -> None:
    certificate = verify_racing_3d_playability(elevated(circuit, surface))
    assert certificate.playable, certificate.failure
    assert certificate.verifier == "racing-oracle-replay-3d-v1"


def test_3d_scene_compilation_certifies_and_reports() -> None:
    scene, certificate, notes = compile_racing_3d_scene(
        "3d compile", archetype_plan("chicane", "ice"),
        ElevationSpec(profile=ElevationProfile.ALPINE, amplitude_m=30, hill_count=7),
        seed=17,
    )
    assert certificate.playable
    assert scene.elevation is not None
    assert validate_track_surface(compile_track_surface(scene)) == []
    assert notes, "a fitted elevation must be reported"


def test_shared_rules_still_apply_in_3d() -> None:
    """Nitro, countdown, and opponent behavior are the 2D implementations."""
    scene = elevated(npcs=3)
    scene = scene.model_copy(update={
        "npc_behaviors": [
            item.model_copy(update={"profile": NpcProfile.AGGRESSOR, "aggression": .9})
            for item in scene.npc_behaviors
        ],
    })
    world = Racing3DWorld.from_scene(scene)
    assert world.countdown_ticks_remaining == scene.dynamics.control_hz * 3
    assert world.nitro == 0
    frame = world.step(Action(keys=["w"]))
    assert frame.action == ActionName.IDLE, "the countdown must freeze the grid in 3D too"
    assert all(pose is not None for pose in world.opponent_poses.values())


# ------------------------------------------------------------ the view component


@pytest.mark.parametrize("mode", list(ViewMode))
def test_camera_is_a_pure_function_of_world_state(mode: ViewMode) -> None:
    """A policy frame must be reproducible, so cameras carry no memory."""
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    controller = RacingLineController()
    controller.reset(world.scene, world.scene.seed)
    for _ in range(80):
        world.step(*controller.act(world.observe()))
    before = world.snapshot()
    first = camera_for(world, mode)
    second = camera_for(world, mode)
    assert first == second
    assert world.snapshot() == before, "building a camera must not mutate the world"


@pytest.mark.parametrize("heading", [0.0, 45.0, 90.0, 180.0, 270.0, 330.0])
@pytest.mark.parametrize("mode", list(ViewMode))
def test_the_view_is_not_mirrored(mode: ViewMode, heading: float) -> None:
    """Something on the car's right must appear on the right of the frame.

    World axes are x east, y south, z up. Screen right is `up x forward`; using
    `forward x up` points it north instead, which mirrors the whole image. That
    presents as inverted steering -- the car turns the way it was asked, but the
    picture shows it turning the other way -- so it is easy to misdiagnose as a
    controls bug rather than a rendering one.
    """
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    world.heading = heading
    world.player = Vec2(x=460.0, y=320.0)
    world._refresh_vertical_state()
    yaw = math.radians(heading)
    # The driver's right hand is the forward direction rotated a quarter turn.
    right_x, right_y = -math.sin(yaw), math.cos(yaw)
    camera = camera_for(world, mode)
    right, up, forward = camera.basis()
    projector = _Projector(
        eye=camera.eye, right=right, up=up, forward=forward,
        focal=200.0, centre_x=0.0, centre_y=0.0,
    )
    on_right = projector.project(projector.to_view(
        (world.player.x + right_x * 90, world.player.y + right_y * 90, world.player_z + 8),
    ))[0]
    on_left = projector.project(projector.to_view(
        (world.player.x - right_x * 90, world.player.y - right_y * 90, world.player_z + 8),
    ))[0]
    assert on_right > on_left, (mode.value, heading, on_right, on_left)


@pytest.mark.parametrize("mode", list(ViewMode))
def test_camera_basis_is_orthonormal(mode: ViewMode) -> None:
    """Includes the overhead camera, where the usual up-vector cross collapses."""
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    right, up, forward = camera_for(world, mode).basis()
    for axis in (right, up, forward):
        assert sum(component * component for component in axis) == pytest.approx(1.0, abs=1e-9)
    for first, second in ((right, up), (right, forward), (up, forward)):
        assert sum(a * b for a, b in zip(first, second)) == pytest.approx(0.0, abs=1e-9)


def test_overhead_camera_looks_down_and_chase_cameras_look_forward() -> None:
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    assert camera_for(world, ViewMode.OVERHEAD_3D).pitch_degrees < -70
    assert camera_for(world, ViewMode.THIRD_PERSON).distance_behind_pixels > 0
    assert camera_for(world, ViewMode.FIRST_PERSON).distance_behind_pixels == 0
    assert ViewMode.FIRST_PERSON in _IN_CAR_MODES and ViewMode.HOOD in _IN_CAR_MODES
    # An in-car eye is physically inside the cabin, which is why those modes skip
    # drawing the ego bodywork.
    world_pose = world.player_pose()
    body_top = world.dynamics.vehicle.width_m * world.dynamics.pixels_per_meter * .7
    assert 0 < camera_for(world, ViewMode.FIRST_PERSON).eye_height_pixels < body_top * 3
    del world_pose


def test_near_plane_clipping_removes_geometry_behind_the_eye() -> None:
    """Unclipped, a vertex behind the eye projects inverted and smears the frame."""
    straddling = [(0.0, 0.0, -20.0), (10.0, 0.0, 40.0), (10.0, 10.0, 40.0)]
    clipped = _clip_near(straddling)
    assert clipped and all(point[2] >= NEAR_PLANE - 1e-9 for point in clipped)
    assert _clip_near([(0.0, 0.0, -5.0), (1.0, 0.0, -6.0), (1.0, 1.0, -7.0)]) == []
    ahead = [(0.0, 0.0, 50.0), (10.0, 0.0, 50.0), (10.0, 10.0, 50.0)]
    assert _clip_near(ahead) == ahead


def test_viewport_clipping_bounds_every_coordinate() -> None:
    covering = _clip_viewport([(-9000, -9000), (9000, -9000), (9000, 9000), (-9000, 9000)], 320, 200)
    assert covering
    assert all(-3 <= x <= 323 and -3 <= y <= 203 for x, y in covering)
    assert _clip_viewport([(400, 400), (500, 400), (500, 500)], 320, 200) == []


@pytest.mark.parametrize("mode", list(ViewMode))
def test_policy_view_declares_how_it_was_rendered(mode: ViewMode) -> None:
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    frame = render_policy_view(world, mode, 240, 160)
    assert frame.viewpoint == mode.value
    assert frame.orientation == "camera-up" and frame.ego_anchor == "camera-relative"
    assert frame.camera is not None and frame.camera.mode == mode.value
    assert frame.camera.pixels_per_meter == world.dynamics.pixels_per_meter
    assert frame.camera.near_plane_pixels == NEAR_PLANE
    assert 0 < frame.camera.horizontal_fov_degrees < 180
    assert frame.width == 240 and frame.height == 160
    assert len(frame.data_base64) > 500


def test_perspective_frames_must_carry_a_matching_camera_contract() -> None:
    contract = CameraContract(
        mode="third-person", vertical_fov_degrees=58, horizontal_fov_degrees=83,
        eye_height_pixels=30, pitch_degrees=-6, near_plane_pixels=5, pixels_per_meter=8,
    )
    with pytest.raises(ValueError):
        VisualFrame(
            media_type="image/png", data_base64="x", width=8, height=8,
            viewpoint="third-person", orientation="camera-up", ego_anchor="camera-relative",
        )
    with pytest.raises(ValueError):
        VisualFrame(
            media_type="image/png", data_base64="x", width=8, height=8,
            viewpoint="overhead", camera=contract,
        )
    with pytest.raises(ValueError):
        VisualFrame(
            media_type="image/png", data_base64="x", width=8, height=8,
            viewpoint="third-person", camera=contract,
        )
    ok = VisualFrame(
        media_type="image/png", data_base64="x", width=8, height=8,
        viewpoint="third-person", orientation="camera-up", ego_anchor="camera-relative",
        camera=contract,
    )
    assert ok.camera is contract


def test_camera_modes_render_distinguishable_frames() -> None:
    """Distinct viewing angles must actually produce distinct images."""
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    controller = RacingLineController()
    controller.reset(world.scene, world.scene.seed)
    for _ in range(120):
        world.step(*controller.act(world.observe()))
    payloads = {
        mode: render_policy_view(world, mode, 200, 130).data_base64 for mode in ViewMode
    }
    assert len(set(payloads.values())) == len(ViewMode), "camera modes collapsed to one image"


def test_rendering_draws_road_terrain_and_sky() -> None:
    """A frame that is one flat colour means the pipeline silently drew nothing."""
    import pygame

    pygame.init()
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    controller = RacingLineController()
    controller.reset(world.scene, world.scene.seed)
    for _ in range(120):
        world.step(*controller.act(world.observe()))
    surface = render_view_surface(world, ViewMode.THIRD_PERSON, 240, 160)
    sampled = {
        surface.get_at((x, y))[:3]
        for x in range(0, 240, 8) for y in range(0, 160, 8)
    }
    assert len(sampled) > 8, sampled
    assert all(
        all(0 <= channel <= 255 for channel in colour) for colour in sampled
    )


def test_rendering_never_mutates_the_simulation() -> None:
    world = green_flag(Racing3DWorld.from_scene(elevated()))
    controller = RacingLineController()
    controller.reset(world.scene, world.scene.seed)
    for _ in range(60):
        world.step(*controller.act(world.observe()))
    before = world.snapshot()
    for mode in ViewMode:
        render_policy_view(world, mode, 120, 80)
    assert world.snapshot() == before


def test_steep_surfaces_are_refused_by_the_world_rather_than_simulated() -> None:
    scene = scene_for()
    absurd = scene.model_copy(update={
        "elevation": ElevationSpec(
            profile=ElevationProfile.ALPINE, amplitude_m=40, hill_count=8,
        ),
    })
    with pytest.raises(ValueError, match="undrivable surface"):
        Racing3DWorld.from_scene(absurd)
    assert MAX_DRIVABLE_GRADE_DEGREES > 0


# -- serving the 3D view to a viewer that is not a pygame window ------------------------


def test_the_harness_renders_a_3d_frame_as_png(tmp_path) -> None:
    """The browser gets the harness's own renderer rather than a second implementation.

    A camera here is a pure function of world state, so a WebGL scene in the frontend could
    only add a second set of bugs to keep in sync with the physics. Serving PNG bytes keeps
    one renderer for the desktop viewer, the policy view, and the cockpit.
    """
    from harness.models import RunRequest
    from harness.service import HarnessService
    from harness.store import HarnessStore

    service = HarnessService(store=HarnessStore(tmp_path))
    environment = service.create_environment(
        "technical circuit with two barriers", seed=17, provider="offline", dimensions="3d",
        elevation=ElevationSpec(profile=ElevationProfile.HILLY, amplitude_m=7.0, hill_count=3, banking_degrees=9.0),
    )
    assert environment.scene.elevation is not None

    at_the_grid = service.render_environment_view3d(environment.id, "third-person", 480, 300)
    assert at_the_grid[:4] == b"\x89PNG", "an image endpoint has to return an image"
    assert len(at_the_grid) > 1_000

    run = service.run(RunRequest(
        environment_id=environment.id, policy_name="oracle-racing-line", max_steps=600,
    ))
    assert run.frames
    mid = service.render_run_view3d(run.id, len(run.frames) // 2, "third-person", 480, 300)
    assert mid[:4] == b"\x89PNG"
    assert mid != at_the_grid, "a later tick must not render the starting grid"

    # Every camera has to work, since the cockpit offers all of them.
    for camera in ("first-person", "hood", "third-person-far", "overhead-3d"):
        assert service.render_run_view3d(run.id, 40, camera, 320, 200)[:4] == b"\x89PNG"

    # The compiled surface is reused between requests; rebuilding it per frame would make
    # scrubbing a replay unusable.
    assert len(service._view3d_worlds) <= 8
    before = service._view3d_worlds[f"run:{run.id}"]
    service.render_run_view3d(run.id, 41, "third-person", 320, 200)
    assert service._view3d_worlds[f"run:{run.id}"] is before

    with pytest.raises(ValueError, match="Unknown camera"):
        service.render_run_view3d(run.id, 0, "helicopter", 320, 200)


def test_a_planar_scene_says_why_it_has_no_perspective_view(tmp_path) -> None:
    """Refusing with a reason, rather than rendering a flat plane and calling it 3D."""
    from harness.service import HarnessService
    from harness.store import HarnessStore

    service = HarnessService(store=HarnessStore(tmp_path))
    environment = service.create_environment(
        "technical circuit with two barriers", seed=17, provider="offline",
    )
    with pytest.raises(ValueError, match="planar"):
        service.render_environment_view3d(environment.id, "third-person", 320, 200)

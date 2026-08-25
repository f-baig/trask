"""Appearance, and the line between appearance and physics.

The whole reason colour lives in its own plan is that it must never be able to reach
the simulator. These check both directions of that: that a repaint changes the render
and nothing else, and that a repaint cannot satisfy a requirement about how the
circuit drives.
"""

from __future__ import annotations

import pytest

from harness.generation_spec import Assertion, _Context, _EVALUATORS
from harness.models import SceneSpec
from harness.racing import compile_certified_scene
from harness.track_grammar import (
    CornerRadius, CornerSpec, NpcSpec, StraightLength, TrackPlan, parse_track_prompt,
)
from harness.visual import SceneryBand, VisualPlan, rgb, to_hex


def _plan(visual: VisualPlan | None = None) -> TrackPlan:
    return TrackPlan(
        title="Palette Circuit", rationale="a circuit for checking the visual plan",
        corners=[
            CornerSpec(radius=CornerRadius.OPEN, exit_straight=StraightLength.LONG, label=f"c{index}")
            for index in range(4)
        ],
        npcs=[NpcSpec(profile="racer", label="rival")],
        visual=visual or VisualPlan(),
    )


def _scene(visual: VisualPlan | None = None) -> SceneSpec:
    scene, _certificate, _notes = compile_certified_scene("palette", _plan(visual), 4)
    return scene


def _check(scene: SceneSpec, kind: str, target, tolerance: float = 0.0):
    assertion = Assertion(id="a", kind=kind, target=target, tolerance=tolerance, label=kind)
    return _EVALUATORS[kind](assertion, _Context(scene, None))


# --- the separation --------------------------------------------------------------


def test_a_repaint_changes_nothing_the_simulator_reads() -> None:
    """Colour is not a dial. A recoloured circuit must be the same circuit.

    If this ever fails, a brief asking for a blue track has silently become a brief
    asking for different handling, and every fidelity number about it is wrong.
    """
    plain = _scene()
    painted = _scene(VisualPlan(road="purple", terrain="green", barrier="blue", kerbs=False))
    assert plain.grip == painted.grip
    assert plain.track_width == painted.track_width
    assert plain.surface == painted.surface
    assert plain.track_centerline == painted.track_centerline
    assert [item.model_dump() for item in plain.entities] == [
        item.model_dump() for item in painted.entities
    ]
    assert plain.dynamics == painted.dynamics


def test_a_scene_with_no_visual_plan_keeps_the_surface_palette() -> None:
    """Additive by construction: saying nothing about colour changes nothing."""
    resolved = _scene().visual.resolved("asphalt")
    assert resolved["road"] == "#343a40"
    assert resolved["terrain"] == "#3f6746"
    assert resolved["kerbs"] is True


def test_the_visual_plan_survives_compilation() -> None:
    scene = _scene(VisualPlan(
        road="#7a44b5", kerbs=False,
        scenery=[SceneryBand(label="river", color="blue", region="bottom-center")],
    ))
    assert scene.visual.road == "#7a44b5"
    assert scene.visual.kerbs is False
    assert len(scene.visual.scenery) == 1


# --- colour parsing --------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("#ff0000", "#ff0000"), ("FF0000", "#ff0000"), ("black", "#101215"),
    ("Neon Pink", "#ff2d95"), ("  blue  ", "#2f6fd0"),
])
def test_colours_are_read_in_either_form(value: str, expected: str) -> None:
    """Briefs say "blue" far more often than they say "#2f6fd0"."""
    assert to_hex(value) == expected


def test_an_unresolvable_colour_is_not_silently_black() -> None:
    """Scoring a named colour the table lacks against black failed real requirements."""
    assert to_hex("chartreuse-ish") is None
    assert to_hex("") is None
    scene = _scene(VisualPlan(road="purple"))
    satisfied, _actual, message = _check(scene, "road_colour", "chartreuse-ish")
    assert not satisfied
    assert "not a colour this harness can resolve" in message


def test_a_malformed_colour_falls_back_rather_than_failing_the_plan() -> None:
    """Nobody should lose a compiled circuit over a bad swatch."""
    plan = VisualPlan(road="not-a-colour", barrier="#zzzzzz")
    assert plan.road is None and plan.barrier is None
    assert plan.resolved("asphalt")["road"] == "#343a40"


# --- verification ----------------------------------------------------------------


@pytest.mark.parametrize("kind,slot,asked", [
    ("road_colour", "road", "purple"),
    ("terrain_colour", "terrain", "green"),
    ("barrier_colour", "barrier", "blue"),
    ("player_car_colour", "player_car", "white"),
    ("opponent_car_colour", "opponent_car", "black"),
    ("sky_colour", "sky", "navy"),
])
def test_each_palette_request_is_verifiable(kind: str, slot: str, asked: str) -> None:
    scene = _scene(VisualPlan(**{slot: asked}))
    satisfied, _actual, message = _check(scene, kind, asked)
    assert satisfied, message


def test_the_wrong_colour_is_reported_as_wrong() -> None:
    scene = _scene(VisualPlan(barrier="red"))
    satisfied, actual, message = _check(scene, "barrier_colour", "blue")
    assert not satisfied
    assert actual == "#d23b2f"
    assert "asked for something like blue" in message


def test_near_shades_of_the_same_colour_still_count() -> None:
    """"Blue" is a request, not a swatch, so azure has to satisfy it."""
    scene = _scene(VisualPlan(barrier="azure"))
    satisfied, _actual, _message = _check(scene, "barrier_colour", "blue")
    assert satisfied


def test_turning_the_kerbs_off_is_checkable() -> None:
    off = _scene(VisualPlan(kerbs=False))
    assert _check(off, "kerbs_present", False)[0]
    assert not _check(off, "kerbs_present", True)[0]
    assert _check(_scene(), "kerbs_present", True)[0]


def test_scenery_bands_are_counted() -> None:
    scene = _scene(VisualPlan(scenery=[
        SceneryBand(label="river", color="blue", region="bottom-center"),
        SceneryBand(label="sand", color="sand", region="top-left"),
    ]))
    assert _check(scene, "scenery_count", 2)[0]
    assert not _check(scene, "scenery_count", 1)[0]


def test_painting_a_circuit_cannot_satisfy_a_physics_requirement() -> None:
    """A blue track is not a slippery track, and the verifier must not confuse them."""
    scene = _scene(VisualPlan(road="blue", terrain="blue"))
    satisfied, actual, _message = _check(scene, "grip_max", 0.5)
    assert not satisfied
    assert actual == 1.0


# --- rendering -------------------------------------------------------------------


def test_the_renderer_reads_the_plan_rather_than_the_surface_table() -> None:
    pygame = pytest.importorskip("pygame")
    from harness.models import ElevationProfile, ElevationSpec
    from harness.racing3d import Racing3DWorld, compile_racing_3d_scene
    from harness.view3d import ViewMode, ensure_headless_video, render_view_surface

    flat = ElevationSpec(
        profile=ElevationProfile.FLAT, amplitude_m=0.0, hill_count=1, banking_degrees=0.0,
    )
    ensure_headless_video()

    def frame(visual: VisualPlan):
        scene, _certificate, _notes = compile_racing_3d_scene("palette", _plan(visual), flat, 4)
        world = Racing3DWorld.from_scene(scene)
        return pygame.surfarray.array3d(
            render_view_surface(world, ViewMode.THIRD_PERSON, 128, 72)
        )

    plain = frame(VisualPlan())
    painted = frame(VisualPlan(road="#7a44b5", terrain="#3f9a4e"))
    assert not (plain == painted).all(), "a recolour has to reach the rasterizer"


def test_the_named_colour_table_round_trips_through_rgb() -> None:
    assert rgb("black") == (16, 18, 21)
    assert rgb("#ffffff") == (255, 255, 255)

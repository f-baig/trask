"""Tests for the player-side optical-flow overlay tool.

The flow solver is checked against translations whose answer is known exactly, so a
failure localizes to the estimator rather than to a judgement about racing. The
integration tests then assert the physical signature each viewpoint must produce: a
world-fixed overhead view moves only where cars are, and an ego-normalized cone view
moves everywhere the car can see.
"""

import base64
import io

import numpy
import pytest

from harness.models import Action
from harness.motion import (
    ARROW_COLOR, MINIMUM_DRAWN_MAGNITUDE, MotionOverlay, MotionUnavailable,
    cell_geometry, dense_optical_flow, grid_average,
)
from harness.policies import AnthropicRacingPolicy, _configured_motion_overlay
from harness.policy_protocol import VisualFrame
from harness.providers import ActionSegment, PlayerPlan, ProviderUsage
from harness.racing import RacingDesignDraft, RacingWorld, compile_racing_scene


def draft(obstacles: int = 2, npcs: int = 2) -> RacingDesignDraft:
    return RacingDesignDraft(
        title="Motion overlay circuit",
        rationale="A deterministic circuit used to exercise the motion overlay.",
        circuit="technical", surface="asphalt", obstacle_count=obstacles, npc_count=npcs,
    )


def texture(height: int = 320, width: int = 480, seed: int = 3):
    """A band-limited random field: textured enough to track, smooth enough to solve.

    Lucas-Kanade linearizes the image, so white noise is not a valid test signal —
    it has no scale at which a gradient predicts its neighbour.
    """
    generator = numpy.random.default_rng(seed)
    rows, columns = numpy.mgrid[0:height, 0:width]
    waves = (
        numpy.sin(columns / 23.0) * 0.3
        + numpy.cos(rows / 17.0) * 0.3
        + numpy.sin((columns + rows) / 41.0) * 0.25
    )
    noise = generator.normal(0, 0.05, (height, width))
    kernel = numpy.ones((9, 9)) / 81.0
    smoothed = numpy.apply_along_axis(
        lambda column: numpy.convolve(column, kernel[0], mode="same"), 0,
        numpy.apply_along_axis(
            lambda row: numpy.convolve(row, kernel[0], mode="same"), 1, noise,
        ),
    )
    return ((waves + smoothed + 1) / 2).astype(numpy.float32)


def advanced_world(view: str, monkeypatch: pytest.MonkeyPatch, ticks: int = 46) -> RacingWorld:
    """A world already up to speed, because a stationary car has no flow to measure."""
    monkeypatch.setenv("RACING_POLICY_VIEW", view)
    scene = compile_racing_scene("motion", draft(), seed=17)
    world = RacingWorld.from_scene(scene)
    for _ in range(ticks):
        world.step(Action(keys=["w"]))
    assert world.speed > 1, "the test needs a moving car"
    return world


@pytest.mark.parametrize("dx,dy", [(0, 0), (4, -3), (-9, 6), (14, 0), (0, -12), (22, -8)])
def test_flow_recovers_a_known_translation(dx: int, dy: int) -> None:
    previous = texture()
    current = numpy.roll(previous, shift=(dy, dx), axis=(0, 1))
    cells = grid_average(dense_optical_flow(previous, current))
    # The median rejects the wrap-around seam that np.roll introduces at two edges;
    # the interior of the field is what the arrows are drawn from.
    assert numpy.median(cells[:, :, 0]) == pytest.approx(dx, abs=0.5)
    assert numpy.median(cells[:, :, 1]) == pytest.approx(dy, abs=0.5)


def test_identical_frames_measure_no_motion() -> None:
    still = texture()
    flow = dense_optical_flow(still, still)
    assert numpy.abs(flow).max() == 0.0


def test_flow_is_a_pure_function_of_the_frame_pair() -> None:
    previous, current = texture(), numpy.roll(texture(), shift=(2, -5), axis=(0, 1))
    first = dense_optical_flow(previous, current)
    second = dense_optical_flow(previous, current)
    assert numpy.array_equal(first, second)


def test_flat_regions_report_no_measurement_rather_than_noise() -> None:
    """A featureless region has no recoverable motion, and must not invent one."""
    previous = numpy.full((320, 480), 0.5, dtype=numpy.float32)
    current = numpy.roll(texture(), shift=(0, 7), axis=1)
    current[:, :] = 0.5
    assert numpy.abs(dense_optical_flow(previous, current)).max() == 0.0


@pytest.mark.parametrize("grid", [(16, 16), (8, 12), (5, 7)])
def test_grid_average_covers_every_pixel_exactly_once(grid: tuple[int, int]) -> None:
    flow = numpy.ones((320, 480, 2), dtype=numpy.float32)
    cells = grid_average(flow, grid)
    assert cells.shape == (grid[0], grid[1], 2)
    # A constant field must average to that constant in every cell, which fails if
    # any cell is empty, double-counted, or divided by the wrong pixel count.
    assert numpy.allclose(cells, 1.0)
    centers = cell_geometry(480, 320, grid)
    assert (len(centers), len(centers[0])) == grid


def test_grid_cannot_be_finer_than_the_frame() -> None:
    with pytest.raises(ValueError, match="motion grid"):
        grid_average(numpy.zeros((8, 8, 2), dtype=numpy.float32), (16, 16))


def test_first_frame_is_returned_unannotated() -> None:
    """One frame is not a motion measurement, and must not be dressed up as one."""
    frame = _rendered_frame()
    tool = MotionOverlay()
    result = tool.annotate(frame)
    assert result is frame
    assert result.motion_overlay is False
    assert tool.frames_seen == 1 and tool.pairs_measured == 0


def test_annotated_frame_declares_how_to_read_its_arrows(monkeypatch: pytest.MonkeyPatch) -> None:
    world = advanced_world("overhead", monkeypatch)
    tool = MotionOverlay()
    tool.annotate(world.render_policy_frame())
    world.step(Action(keys=["w"]))
    annotated = tool.annotate(world.render_policy_frame())

    assert annotated.motion_overlay and annotated.motion_base == "grayscale"
    assert annotated.motion_overlay_semantics == "grid-averaged-optical-flow"
    assert annotated.motion_grid == [16, 16]
    assert annotated.motion_arrow_scale > 0 and annotated.motion_arrow_max_pixels > 0
    # The overlay must not change the sensor it annotates.
    assert (annotated.viewpoint, annotated.orientation) == ("overhead", "north-up")
    assert (annotated.width, annotated.height) == (480, 320)
    assert tool.pairs_measured == 1


def test_overlay_is_deterministic_down_to_the_encoded_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _frame_pair("overhead", monkeypatch)
    renders = []
    for _ in range(2):
        tool = MotionOverlay()
        tool.annotate(frames[0])
        renders.append(tool.annotate(frames[1]).data_base64)
    assert renders[0] == renders[1]


def test_one_annotated_frame_costs_one_frame_of_vision_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point is replacing a frame stack with one frame of the same price.

    Vision tokens are billed against image dimensions, not encoded size, so equal
    dimensions is the claim that matters. Arrows do inflate the PNG — high-frequency
    amber over flat gray is exactly what a filter predictor is worst at — but that is
    transport bytes, and it stays inside one frame either way.
    """
    frames = _frame_pair("forward-cone", monkeypatch)
    tool = MotionOverlay()
    tool.annotate(frames[0])
    annotated = tool.annotate(frames[1])
    assert (annotated.width, annotated.height) == (frames[1].width, frames[1].height)
    assert len(base64.b64decode(annotated.data_base64)) < 60_000


def test_overhead_view_shows_motion_only_where_cars_are(monkeypatch: pytest.MonkeyPatch) -> None:
    """A world-fixed camera must not report the static circuit as moving."""
    cells = _measured_cells("overhead", monkeypatch)
    magnitude = numpy.hypot(cells[:, :, 0], cells[:, :, 1])
    moving = int((magnitude >= MINIMUM_DRAWN_MAGNITUDE).sum())
    assert 0 < moving < magnitude.size // 2


def test_ego_view_shows_the_whole_scene_flowing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ego-normalized camera moves the world, so most visible cells must flow."""
    cells = _measured_cells("forward-cone", monkeypatch)
    magnitude = numpy.hypot(cells[:, :, 0], cells[:, :, 1])
    assert int((magnitude >= MINIMUM_DRAWN_MAGNITUDE).sum()) > magnitude.size // 3


def test_unobserved_cone_pixels_get_no_arrows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-FOV black is unknown. Pyramid bleed used to fill it with confident flow."""
    import pygame

    frames = _frame_pair("forward-cone", monkeypatch)
    tool = MotionOverlay()
    tool.annotate(frames[0])
    annotated = tool.annotate(frames[1])
    image = pygame.image.load(io.BytesIO(base64.b64decode(annotated.data_base64)))
    # The lower-left corner is behind the car's field of view in every scene.
    corner = [
        image.get_at((x, y))[:3]
        for x in range(4, 40) for y in range(annotated.height - 40, annotated.height - 4)
    ]
    assert set(corner) == {(0, 0, 0)}
    assert any(
        image.get_at((x, y))[:3] == ARROW_COLOR
        for x in range(annotated.width) for y in range(annotated.height // 4)
    ), "the visible part of the cone should carry arrows"


def test_color_base_keeps_the_entity_palette(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grayscale is cheaper but discards hue, so the color base has to stay available."""
    import pygame

    frames = _frame_pair("overhead", monkeypatch)
    surfaces = {}
    for label, tool in (("gray", MotionOverlay()), ("color", MotionOverlay(color_base=True))):
        tool.annotate(frames[0])
        annotated = tool.annotate(frames[1])
        assert annotated.motion_base == ("color" if label == "color" else "grayscale")
        surfaces[label] = pygame.image.load(io.BytesIO(base64.b64decode(annotated.data_base64)))

    def saturated(surface) -> int:
        return sum(
            1
            for x in range(0, surface.get_width(), 3)
            for y in range(0, surface.get_height(), 3)
            if max(surface.get_at((x, y))[:3]) - min(surface.get_at((x, y))[:3]) > 40
            and surface.get_at((x, y))[:3] != ARROW_COLOR
        )

    assert saturated(surfaces["color"]) > saturated(surfaces["gray"]) * 5


def test_rgb_frames_are_rejected_rather_than_silently_skipped() -> None:
    raw = VisualFrame(
        media_type="application/x-rgb", data_base64="", width=480, height=320,
    )
    with pytest.raises(MotionUnavailable, match="encoded image frame"):
        MotionOverlay().annotate(raw)


@pytest.mark.parametrize("setting,expected", [
    ("0", None), ("off", None), ("", None),
    ("1", "grayscale"), ("gray", "grayscale"), ("on", "grayscale"),
    ("color", "color"), ("rgb", "color"),
])
def test_env_switch_selects_the_overlay_base(
    monkeypatch: pytest.MonkeyPatch, setting: str, expected: str | None,
) -> None:
    monkeypatch.setenv("RACING_MOTION_OVERLAY", setting)
    tool = _configured_motion_overlay()
    if expected is None:
        assert tool is None
    else:
        assert tool is not None and tool.color_base is (expected == "color")


def test_an_unreadable_overlay_setting_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must not silently disable the tool the experiment is measuring."""
    monkeypatch.setenv("RACING_MOTION_OVERLAY", "grayscsale")
    with pytest.raises(ValueError, match="RACING_MOTION_OVERLAY"):
        _configured_motion_overlay()


def test_policy_sends_one_motion_frame_instead_of_a_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[list[VisualFrame]] = []

    def capture_plan(*args, visual_frames=None, **kwargs):
        sent.append(list(visual_frames or []))
        return PlayerPlan(
            subgoal="read the motion field", summary="one tick, then observe again",
            confidence=1, actions=[ActionSegment(action="idle", steps=1)],
        ), ProviderUsage(provider="test", model="test", output_tokens=1)

    monkeypatch.setenv("RACING_MOTION_OVERLAY", "1")
    # An explicit stack request loses to the overlay: paying for both would send the
    # same motion twice.
    monkeypatch.setenv("RACING_VISUAL_HISTORY", "4")
    monkeypatch.setattr("harness.policies.plan_racing_actions", capture_plan)
    world = advanced_world("overhead", monkeypatch)
    policy = AnthropicRacingPolicy()
    policy.configure_episode(20, decision_budget=6)
    policy.reset(world.scene, world.scene.seed)
    assert policy.visual_history_size == 1

    for step in range(4):
        policy.act_visual(
            world.observe().model_copy(update={"step": step}), world.render_policy_frame(),
        )
        world.step(Action(keys=["w"]))

    assert [len(batch) for batch in sent] == [1, 1, 1, 1]
    assert [batch[0].motion_overlay for batch in sent] == [False, True, True, True]
    assert policy.motion_overlay is not None and policy.motion_overlay.pairs_measured == 3


def test_policy_leaves_frames_alone_when_the_overlay_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RACING_MOTION_OVERLAY", raising=False)
    world = advanced_world("overhead", monkeypatch)
    policy = AnthropicRacingPolicy()
    policy.reset(world.scene, world.scene.seed)
    assert policy.motion_overlay is None


def test_frame_contract_rejects_unexplained_arrows() -> None:
    """A frame may not claim a motion overlay without saying how to read it."""
    base = {
        "media_type": "image/png", "data_base64": "x", "width": 480, "height": 320,
        "motion_overlay": True, "motion_overlay_semantics": "grid-averaged-optical-flow",
        "motion_grid": [16, 16], "motion_arrow_scale": 2.4,
        "motion_arrow_max_pixels": 18.0, "motion_base": "grayscale",
        "motion_interval_ticks": 1,
    }
    VisualFrame.model_validate(base)
    for missing in (
        "motion_overlay_semantics", "motion_grid", "motion_arrow_scale",
        "motion_base", "motion_interval_ticks",
    ):
        with pytest.raises(ValueError):
            VisualFrame.model_validate({**base, missing: None})
    with pytest.raises(ValueError, match="only motion frames"):
        VisualFrame.model_validate({**base, "motion_overlay": False})


def _rendered_frame() -> VisualFrame:
    scene = compile_racing_scene("motion", draft(), seed=17)
    return RacingWorld.from_scene(scene).render_policy_frame()


def _frame_pair(view: str, monkeypatch: pytest.MonkeyPatch) -> tuple[VisualFrame, VisualFrame]:
    world = advanced_world(view, monkeypatch)
    first = world.render_policy_frame()
    world.step(Action(keys=["w"]))
    return first, world.render_policy_frame()


def _measured_cells(view: str, monkeypatch: pytest.MonkeyPatch):
    from harness.motion import _grayscale, _surface_from_frame

    first, second = _frame_pair(view, monkeypatch)
    grayscales = [_grayscale(_surface_from_frame(frame)) for frame in (first, second)]
    return grid_average(dense_optical_flow(*grayscales))


def test_overlay_reports_the_interval_it_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrows across thirty ticks mean something different from arrows across one.

    An asynchronous scheduler only renders when it issues a decision, so the frame has
    to carry the span rather than let the prompt assert a tick.
    """
    frames = _frame_pair("forward-cone", monkeypatch)
    tool = MotionOverlay()
    tool.annotate(frames[0])
    assert tool.annotate(frames[1]).motion_interval_ticks == 1
    tool = MotionOverlay()
    tool.annotate(frames[0])
    assert tool.annotate(frames[1], interval_ticks=29).motion_interval_ticks == 29

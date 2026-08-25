"""Grid-averaged optical flow: encode motion into one frame, not a frame stack.

A single rendered frame is ambiguous about velocity. A car at the apex of a corner
looks identical whether it is accelerating out or sliding in, so a policy given one
image has to infer motion from telemetry it cannot localize in the picture. The
existing answer is `RACING_VISUAL_HISTORY`, which sends two to four images and lets
the model diff them itself. That multiplies vision tokens and latency by the stack
depth, and the four-frame probe bought nothing on the ice chicane.

This module does the differencing in the harness instead. It measures dense optical
flow between the previous and current policy frame, averages it over a coarse grid,
and draws one arrow per cell on the current grayscale frame. The result is a single
image whose token cost is the same as an unannotated frame but which carries the
motion field explicitly.

The flow solver is pyramidal Lucas-Kanade in NumPy rather than a call into OpenCV.
That is a deliberate constraint: the OpenCV wheels vendor their own SDL2, which
collides with the pygame-ce SDL2 this harness renders through. Everything here is
float arithmetic over fixed-size arrays with no sampling, so the same frame pair
always produces the same annotated PNG — the determinism the rest of the harness
relies on holds through the preprocessor.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

from .policy_protocol import VisualFrame


GRID_ROWS = 16
GRID_COLUMNS = 16
PYRAMID_LEVELS = 4
"""Four levels resolve roughly 24 pixels of displacement on a 480x320 frame.

Three levels measured a 22-pixel translation as 14, because the coarsest level still
had to explain 2.75 pixels per iteration against a one-pixel step clamp. Four costs
nothing measurable — each added level is a quarter of the pixels of the one below.
"""
SOLVER_ITERATIONS = 4
AGGREGATION_RADIUS = 6
"""Half-width in pixels of the Lucas-Kanade least-squares window, per level."""
ARROW_PIXELS_PER_FLOW_PIXEL = 2.4
MINIMUM_DRAWN_MAGNITUDE = 0.35
"""Flow below this, in pixels per frame, is drawn as a dot instead of an arrow."""
UNOBSERVED_LUMINANCE = 0.04
"""Cells this dark are unknown, not still, and are left blank.

The forward-cone sensor renders everything outside the field of view black, and the
pyramid leaks coarse-level flow into regions that have no gradient to correct it. So
an unmasked overlay drew a confident arrow field across the wedge the car cannot see.
Three marks now mean three different things: an arrow is measured motion, a dot is a
measurement of no motion, and blank is no observation.
"""
SMALLEST_PYRAMID_EDGE = 24
REGULARIZATION = 1e-3
MAXIMUM_ITERATION_STEP = 1.0
"""Per-iteration displacement clamp, in pixels at the level being solved."""

ARROW_COLOR = (255, 196, 46)
STILL_COLOR = (128, 118, 92)


class MotionUnavailable(RuntimeError):
    """The motion overlay tool cannot run in this process."""


@dataclass
class MotionOverlay:
    """A player-side tool: frames in, one motion-annotated frame out.

    The tool is stateful because flow needs two frames and a policy only ever holds
    one. It keeps the previous grayscale frame, so the caller passes each observation
    straight through and gets back either the untouched frame (first call, nothing to
    measure yet) or a single frame carrying the motion field.

    There is no `reset`: build one per episode. A carried-over tool would measure its
    first flow across a scene change, and the arrows would describe a discontinuity
    rather than any motion the car made.
    """

    grid: tuple[int, int] = (GRID_ROWS, GRID_COLUMNS)
    arrow_scale: float = ARROW_PIXELS_PER_FLOW_PIXEL
    color_base: bool = False
    """Draw arrows over the color frame instead of its grayscale conversion."""
    pairs_measured: int = 0
    frames_seen: int = 0
    _previous_grayscale: object | None = field(default=None, repr=False)

    def annotate(self, frame: VisualFrame, interval_ticks: int = 1) -> VisualFrame:
        """Return `frame` with per-cell motion arrows, or unchanged on the first call.

        An unchanged first frame is the honest answer rather than a convenience: a
        frame with no predecessor has no measured motion, and drawing a grid of zero
        arrows would tell the policy that the world is stationary.
        """
        numpy = _numpy()
        surface = _surface_from_frame(frame)
        grayscale = _grayscale(surface)
        previous, self._previous_grayscale = self._previous_grayscale, grayscale
        self.frames_seen += 1
        if previous is None or previous.shape != grayscale.shape:
            return frame
        flow = dense_optical_flow(previous, grayscale)
        cells = grid_average(flow, self.grid)
        observed = grid_average(grayscale[:, :, None], self.grid)[:, :, 0] >= UNOBSERVED_LUMINANCE
        base = surface if self.color_base else _grayscale_surface(grayscale)
        annotated = _draw_flow_arrows(base, cells, self.arrow_scale, observed)
        self.pairs_measured += 1
        # Revalidating rather than copying keeps the frame contract enforced: a
        # motion frame that failed to declare its own semantics would reach the
        # model as an unexplained field of arrows.
        return VisualFrame.model_validate({
            **frame.model_dump(),
            "media_type": "image/png",
            "data_base64": _encode_png(annotated),
            "motion_overlay": True,
            "motion_overlay_semantics": "grid-averaged-optical-flow",
            "motion_grid": list(self.grid),
            "motion_arrow_scale": self.arrow_scale,
            "motion_arrow_max_pixels": _maximum_arrow_pixels(
                grayscale.shape, self.grid, numpy,
            ),
            "motion_base": "color" if self.color_base else "grayscale",
            "motion_interval_ticks": max(1, interval_ticks),
        })


def dense_optical_flow(
    previous, current, *, levels: int = PYRAMID_LEVELS,
    iterations: int = SOLVER_ITERATIONS, radius: int = AGGREGATION_RADIUS,
):
    """Per-pixel displacement carrying `previous` onto `current`, in pixels.

    Coarse-to-fine is not an optimization here, it is a correctness requirement.
    Lucas-Kanade linearizes the image around each pixel, so it only resolves motion
    smaller than its own gradient support; a car crossing thirty pixels in one
    control tick is invisible to a single-level solve. Each level halves the
    displacement it has to explain until the finest level is only refining.
    """
    numpy = _numpy()
    previous_pyramid = _pyramid(previous, levels)
    current_pyramid = _pyramid(current, levels)
    flow = numpy.zeros(previous_pyramid[-1].shape + (2,), dtype=numpy.float32)
    for level in reversed(range(len(previous_pyramid))):
        reference, target = previous_pyramid[level], current_pyramid[level]
        if flow.shape[:2] != reference.shape:
            flow = _upsample_flow(flow, reference.shape) * 2.0
        for _ in range(iterations):
            flow = flow + _lucas_kanade_step(reference, target, flow, radius)
    return flow


def grid_average(flow, grid: tuple[int, int] = (GRID_ROWS, GRID_COLUMNS)):
    """Mean `(dx, dy)` per grid cell, shaped `(rows, columns, 2)`.

    Cell edges are computed from the frame size rather than assumed to divide it, so
    a 480x320 frame and a 640x360 frame both produce exactly `grid` cells.
    """
    numpy = _numpy()
    rows, columns = grid
    height, width = flow.shape[:2]
    if rows < 1 or columns < 1:
        raise ValueError("motion grid needs at least one row and one column")
    if height < rows or width < columns:
        raise ValueError(
            f"a {width}x{height} frame cannot be divided into a {columns}x{rows} motion grid"
        )
    row_edges = _cell_edges(height, rows, numpy)
    column_edges = _cell_edges(width, columns, numpy)
    summed = numpy.add.reduceat(flow, row_edges, axis=0)
    summed = numpy.add.reduceat(summed, column_edges, axis=1)
    row_spans = numpy.diff(numpy.append(row_edges, height)).astype(numpy.float32)
    column_spans = numpy.diff(numpy.append(column_edges, width)).astype(numpy.float32)
    pixels = numpy.outer(row_spans, column_spans)[:, :, None]
    return (summed / pixels).astype(numpy.float32)


def cell_geometry(
    width: int, height: int, grid: tuple[int, int] = (GRID_ROWS, GRID_COLUMNS),
) -> list[list[tuple[float, float]]]:
    """Center of every grid cell in frame pixels, for drawing and for tests."""
    numpy = _numpy()
    rows, columns = grid
    row_edges = list(_cell_edges(height, rows, numpy)) + [height]
    column_edges = list(_cell_edges(width, columns, numpy)) + [width]
    return [
        [
            ((column_edges[column] + column_edges[column + 1] - 1) / 2,
             (row_edges[row] + row_edges[row + 1] - 1) / 2)
            for column in range(columns)
        ]
        for row in range(rows)
    ]


def _cell_edges(extent: int, divisions: int, numpy):
    return (numpy.arange(divisions) * extent) // divisions


def _lucas_kanade_step(reference, target, flow, radius: int):
    """One Gauss-Newton refinement of `flow` at a single pyramid level."""
    numpy = _numpy()
    temporal = _warp(target, flow) - reference
    gradient_y, gradient_x = numpy.gradient(reference)
    a11 = _box_sum(gradient_x * gradient_x, radius)
    a12 = _box_sum(gradient_x * gradient_y, radius)
    a22 = _box_sum(gradient_y * gradient_y, radius)
    b1 = -_box_sum(gradient_x * temporal, radius)
    b2 = -_box_sum(gradient_y * temporal, radius)
    # A textureless cell has a singular structure tensor and no recoverable motion.
    # Regularizing towards zero reports "no measurement" as no arrow, which is the
    # truth; inverting it anyway would fill flat asphalt with noise arrows.
    determinant = a11 * a22 - a12 * a12 + REGULARIZATION
    step = numpy.stack([
        (a22 * b1 - a12 * b2) / determinant,
        (a11 * b2 - a12 * b1) / determinant,
    ], axis=-1)
    return numpy.clip(step, -MAXIMUM_ITERATION_STEP, MAXIMUM_ITERATION_STEP)


def _warp(image, flow):
    """Bilinear sample of `image` at each pixel displaced by `flow`."""
    numpy = _numpy()
    height, width = image.shape
    rows, columns = numpy.meshgrid(
        numpy.arange(height, dtype=numpy.float32),
        numpy.arange(width, dtype=numpy.float32),
        indexing="ij",
    )
    sample_x = numpy.clip(columns + flow[:, :, 0], 0, width - 1)
    sample_y = numpy.clip(rows + flow[:, :, 1], 0, height - 1)
    left = numpy.floor(sample_x).astype(numpy.int32)
    top = numpy.floor(sample_y).astype(numpy.int32)
    right = numpy.minimum(left + 1, width - 1)
    bottom = numpy.minimum(top + 1, height - 1)
    weight_x = (sample_x - left).astype(numpy.float32)
    weight_y = (sample_y - top).astype(numpy.float32)
    upper = image[top, left] * (1 - weight_x) + image[top, right] * weight_x
    lower = image[bottom, left] * (1 - weight_x) + image[bottom, right] * weight_x
    return upper * (1 - weight_y) + lower * weight_y


def _box_sum(image, radius: int):
    """Sum over every `(2*radius+1)` square window, in two cumulative passes.

    Edge-replicated padding keeps the output the same shape as the input, so cells
    against the frame border are measured rather than dropped.
    """
    numpy = _numpy()
    padded = numpy.pad(image, radius, mode="edge")
    span = 2 * radius + 1
    for axis in (0, 1):
        cumulative = numpy.cumsum(padded, axis=axis)
        zero_shape = list(cumulative.shape)
        zero_shape[axis] = 1
        cumulative = numpy.concatenate(
            [numpy.zeros(zero_shape, dtype=cumulative.dtype), cumulative], axis=axis,
        )
        head = [slice(None)] * cumulative.ndim
        tail = [slice(None)] * cumulative.ndim
        head[axis] = slice(span, None)
        tail[axis] = slice(None, -span)
        padded = cumulative[tuple(head)] - cumulative[tuple(tail)]
    return padded


def _pyramid(image, levels: int) -> list:
    """Level 0 is full resolution; the last level is the coarsest."""
    result = [image]
    for _ in range(max(0, levels - 1)):
        height, width = result[-1].shape
        if min(height, width) // 2 < SMALLEST_PYRAMID_EDGE:
            break
        result.append(_downsample(result[-1]))
    return result


def _downsample(image):
    return (_box_sum(image, 1) / 9.0)[::2, ::2]


def _upsample_flow(flow, shape: tuple[int, int]):
    """Invert the decimation exactly, so no level boundary shifts the field."""
    numpy = _numpy()
    height, width = shape
    rows = numpy.minimum(numpy.arange(height) // 2, flow.shape[0] - 1)
    columns = numpy.minimum(numpy.arange(width) // 2, flow.shape[1] - 1)
    return flow[rows][:, columns]


def _maximum_arrow_pixels(shape: tuple[int, int], grid: tuple[int, int], numpy) -> float:
    """Arrows saturate at the cell half-diagonal so neighbours stay distinguishable."""
    height, width = shape
    return round(float(numpy.hypot(width / grid[1], height / grid[0])) / 2, 2)


def _draw_flow_arrows(base, cells, arrow_scale: float, observed=None):
    """One arrow per cell, drawn from the cell center along its mean flow."""
    pygame, numpy = _pygame(), _numpy()
    surface = base.copy()
    width, height = surface.get_size()
    limit = _maximum_arrow_pixels((height, width), cells.shape[:2], numpy)
    centers = cell_geometry(width, height, cells.shape[:2])
    for row, row_centers in enumerate(centers):
        for column, (center_x, center_y) in enumerate(row_centers):
            if observed is not None and not observed[row, column]:
                continue
            dx, dy = (float(value) for value in cells[row, column])
            magnitude = float(numpy.hypot(dx, dy))
            if magnitude < MINIMUM_DRAWN_MAGNITUDE:
                pygame.draw.circle(surface, STILL_COLOR, (round(center_x), round(center_y)), 1)
                continue
            length = min(magnitude * arrow_scale, limit)
            unit_x, unit_y = dx / magnitude, dy / magnitude
            start = (round(center_x), round(center_y))
            tip = (round(center_x + unit_x * length), round(center_y + unit_y * length))
            pygame.draw.line(surface, ARROW_COLOR, start, tip, 1)
            head = max(2.0, min(4.0, length * 0.45))
            for sign in (1, -1):
                # Rotate the unit vector by +/-150 degrees to splay the head back
                # along the shaft, so direction stays readable at two pixels long.
                barb_x = unit_x * -0.866 - unit_y * sign * 0.5
                barb_y = unit_y * -0.866 + unit_x * sign * 0.5
                pygame.draw.line(surface, ARROW_COLOR, tip, (
                    round(tip[0] + barb_x * head), round(tip[1] + barb_y * head),
                ), 1)
    return surface


def _grayscale(surface):
    """Rec. 601 luma of a surface, shaped `(height, width)` in the range 0..1."""
    pygame, numpy = _pygame(), _numpy()
    pixels = pygame.surfarray.array3d(surface).astype(numpy.float32)
    luma = (
        pixels[:, :, 0] * 0.299 + pixels[:, :, 1] * 0.587 + pixels[:, :, 2] * 0.114
    ) / 255.0
    return numpy.ascontiguousarray(luma.T)


def _grayscale_surface(grayscale):
    pygame, numpy = _pygame(), _numpy()
    channel = numpy.clip(grayscale * 255.0, 0, 255).astype(numpy.uint8)
    return pygame.surfarray.make_surface(numpy.repeat(channel.T[:, :, None], 3, axis=2))


def _surface_from_frame(frame: VisualFrame):
    pygame = _pygame()
    if frame.media_type not in {"image/png", "image/jpeg"}:
        raise MotionUnavailable(
            f"the motion overlay needs an encoded image frame, not {frame.media_type}"
        )
    buffer = io.BytesIO(base64.b64decode(frame.data_base64))
    return pygame.image.load(buffer, "policy-frame.png")


def _encode_png(surface) -> str:
    pygame = _pygame()
    buffer = io.BytesIO()
    pygame.image.save(surface, buffer, "motion-overlay.png")
    return base64.b64encode(buffer.getvalue()).decode()


def _numpy():
    try:
        import numpy
    except ImportError as error:  # pragma: no cover - dependency guard
        raise MotionUnavailable(
            "the motion overlay requires numpy; install the 'native' extra"
        ) from error
    return numpy


def _pygame():
    try:
        import pygame
        import pygame.surfarray  # noqa: F401  - not imported by the package itself
    except ImportError as error:  # pragma: no cover - dependency guard
        raise MotionUnavailable(
            "the motion overlay requires pygame-ce; install the 'native' extra"
        ) from error
    return pygame

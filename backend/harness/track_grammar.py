"""Authorable circuit grammar for the deterministic racing engine.

A brief describes a circuit in human terms ("curvy, slippery, with a 90-degree
bend in the top right"). This module turns that intent into a typed `TrackPlan`,
compiles the plan into exact closed geometry, and publishes the residual between
what was requested and what was built.

Compilation is a similarity transform of a path-space turtle walk, so requested
turn angles are honored exactly and the loop closes by construction:

1. resolve corner angles so the signed turns sum to exactly one revolution;
2. walk straights and circular arcs, solving straight lengths for position closure;
3. choose the global rotation that best satisfies requested screen regions;
4. scale and center into the drawable box, then resample to uniform arclength;
5. validate corridor geometry and report every relaxation.

No step consults a random number generator, so a plan compiles byte-identically
on every machine. The compiler never silently absorbs an impossible request: it
relaxes deterministically and records what it changed in the `TrackReport`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .visual import VisualPlan
from .models import (
    CornerRadius, CornerReport, NpcBehaviorSpec, NpcProfile, Rect, StraightLength,
    TrackRegion, TrackReport, Vec2,
)


# A closed loop turns through exactly one revolution. Angles below the minimum
# read as noise on the racing line; angles above the maximum are hairpins whose
# arc chord no longer fits a bounded box.
FULL_REVOLUTION = 360.0
MIN_CORNER_ANGLE = 12.0
MAX_CORNER_ANGLE = 172.0
MIN_STRAIGHT_PIXELS = 44.0
MAX_TOTAL_CORNERS = 14
CENTERLINE_SPACING = 22.0
# Per-grid-slot reduction in default pace and skill, so a field is a running order
# rather than a train of identical cars. Capped small enough that temperament
# still dominates: four slots apart is a 12% pace difference.
FIELD_PACE_STAGGER = .03

# Nominal pre-fit dimensions. The whole path is uniformly scaled into the
# drawable box afterwards, so only their ratios matter.
_RADIUS_PIXELS = {
    CornerRadius.HAIRPIN: 74.0,
    CornerRadius.TIGHT: 104.0,
    CornerRadius.MEDIUM: 152.0,
    CornerRadius.OPEN: 215.0,
    CornerRadius.SWEEPING: 300.0,
}
_STRAIGHT_PIXELS = {
    StraightLength.SHORT: 95.0,
    StraightLength.MEDIUM: 200.0,
    StraightLength.LONG: 345.0,
}

# Screen anchors for each addressable region, normalized inside the drawable
# box. Corners live on the loop's outer ring, so `center` is never a candidate
# when classifying an achieved region.
_REGION_TARGETS: dict[TrackRegion, tuple[float, float]] = {
    TrackRegion.TOP_LEFT: (.20, .20),
    TrackRegion.TOP_CENTER: (.50, .13),
    TrackRegion.TOP_RIGHT: (.80, .20),
    TrackRegion.LEFT: (.13, .50),
    TrackRegion.CENTER: (.50, .50),
    TrackRegion.RIGHT: (.87, .50),
    TrackRegion.BOTTOM_LEFT: (.20, .80),
    TrackRegion.BOTTOM_CENTER: (.50, .87),
    TrackRegion.BOTTOM_RIGHT: (.80, .80),
}
_RING_REGIONS = tuple(region for region in _REGION_TARGETS if region != TrackRegion.CENTER)
# Clockwise walk of the outer ring, used to measure how far an achieved corner
# drifted from the region a brief asked for.
_RING_ORDER = (
    TrackRegion.RIGHT, TrackRegion.BOTTOM_RIGHT, TrackRegion.BOTTOM_CENTER,
    TrackRegion.BOTTOM_LEFT, TrackRegion.LEFT, TrackRegion.TOP_LEFT,
    TrackRegion.TOP_CENTER, TrackRegion.TOP_RIGHT,
)

# Named opponent temperaments. Every profile expands to explicit numbers so the
# engine never branches on a label.
_NPC_PROFILES: dict[NpcProfile, dict[str, float | bool]] = {
    NpcProfile.BACKMARKER: {"pace": .62, "skill": .34, "aggression": .12, "intelligence": .20, "defends": False, "uses_nitro": False},
    NpcProfile.CRUISER: {"pace": .78, "skill": .55, "aggression": .28, "intelligence": .45, "defends": False, "uses_nitro": True},
    NpcProfile.RACER: {"pace": .90, "skill": .72, "aggression": .52, "intelligence": .70, "defends": False, "uses_nitro": True},
    NpcProfile.AGGRESSOR: {"pace": .99, "skill": .86, "aggression": .93, "intelligence": .90, "defends": True, "uses_nitro": True},
    NpcProfile.BLOCKER: {"pace": .80, "skill": .66, "aggression": .74, "intelligence": .60, "defends": True, "uses_nitro": False},
}


class CornerSpec(BaseModel):
    """One authored corner. Omitted fields are solved by the compiler."""

    direction: Literal["left", "right"] | None = None
    """Turn direction; defaults to the circuit direction."""
    angle_degrees: float | None = Field(default=None, ge=MIN_CORNER_ANGLE, le=MAX_CORNER_ANGLE)
    """Requested heading change. Left unset, the compiler solves it for closure."""
    radius: CornerRadius = CornerRadius.MEDIUM
    region: TrackRegion = TrackRegion.AUTO
    exit_straight: StraightLength = StraightLength.MEDIUM
    label: str | None = Field(default=None, max_length=48)


class BarrierSpec(BaseModel):
    """A lane-edge barrier. Barriers never occlude the certified center line."""

    region: TrackRegion = TrackRegion.AUTO
    side: Literal["left", "right", "auto"] = "auto"
    shape: Literal["circle", "box", "oriented-box"] = "circle"
    """Collision shape. `circle` is a tyre stack or bollard; `oriented-box` is a
    wall section laid along the road; `box` is axis-aligned. The same declaration
    drives collision and both renderers."""
    label: str | None = Field(default=None, max_length=48)


class NpcSpec(BaseModel):
    """One opponent car; the profile supplies defaults each field may override."""

    profile: NpcProfile = NpcProfile.RACER
    pace: float | None = Field(default=None, ge=.35, le=1.05)
    skill: float | None = Field(default=None, ge=0, le=1)
    aggression: float | None = Field(default=None, ge=0, le=1)
    intelligence: float | None = Field(default=None, ge=0, le=1)
    defends: bool | None = None
    uses_nitro: bool | None = None
    label: str | None = Field(default=None, max_length=48)

    def resolve(self, entity_id: str, grid_index: int = 0) -> NpcBehaviorSpec:
        """Expand the profile into explicit numbers for one car on the grid.

        Cars further back get a slightly lower default pace and skill. A field of
        identical opponents drives as one train that never changes order, which
        makes a race read as scenery; a small spread gives a running order the
        player can actually move through. Explicitly authored values are never
        staggered, so "three aggressive npcs" still means three aggressors.
        """
        defaults = _NPC_PROFILES[self.profile]
        stagger = 1.0 - FIELD_PACE_STAGGER * min(4, max(0, grid_index))
        return NpcBehaviorSpec(
            entity_id=entity_id,
            profile=self.profile,
            pace=float(
                self.pace if self.pace is not None
                else min(1.05, float(defaults["pace"]) * stagger)
            ),
            skill=float(
                self.skill if self.skill is not None
                else max(0.0, min(1.0, float(defaults["skill"]) * stagger))
            ),
            aggression=float(self.aggression if self.aggression is not None else defaults["aggression"]),
            intelligence=float(
                self.intelligence if self.intelligence is not None
                else max(0.0, min(1.0, float(defaults["intelligence"]) * stagger))
            ),
            defends=bool(self.defends if self.defends is not None else defaults["defends"]),
            uses_nitro=bool(self.uses_nitro if self.uses_nitro is not None else defaults["uses_nitro"]),
        )


class TrackPlan(BaseModel):
    """The complete typed configuration an environment creator may author."""

    title: str = Field(min_length=3, max_length=64)
    rationale: str = Field(min_length=8, max_length=360)
    direction: Literal["clockwise", "counterclockwise"] = "counterclockwise"
    loop_shape: Literal["cornered", "circle"] = "cornered"
    """`circle` is a true constant-radius loop, not a four-corner approximation."""
    corners: list[CornerSpec] = Field(default_factory=list, max_length=10)
    surface: Literal["asphalt", "clay", "ice"] = "asphalt"
    grip: float = Field(default=1.0, ge=.3, le=1.2)
    """Continuous grip multiplier layered on the surface preset; <1 is slippery."""
    track_width: float = Field(default=132.0, ge=110.0, le=170.0)
    edge_barriers: bool = False
    """Whether both road boundaries carry continuous solid guardrails."""
    laps: int = Field(default=1, ge=1, le=10)
    start_region: TrackRegion = TrackRegion.AUTO
    """Requested map region for the start/finish gate; ``auto`` uses the safest straight."""
    player_grid_position: int = Field(default=1, ge=1, le=6)
    """Player's place in the starting grid: 1 is pole, later places sit behind it."""
    barriers: list[BarrierSpec] = Field(default_factory=list, max_length=6)
    npcs: list[NpcSpec] = Field(default_factory=list, max_length=5)
    npc_start_mode: Literal["grid", "distributed"] = "grid"
    visual: VisualPlan = Field(default_factory=VisualPlan)
    """The look, carried alongside the geometry but never read by the compiler."""

    @model_validator(mode="after")
    def reject_impossible_rotation(self) -> "TrackPlan":
        if self.loop_shape == "cornered" and len(self.corners) < 3:
            raise ValueError("A cornered circuit needs at least three turn entries")
        if self.loop_shape == "circle" and self.corners:
            raise ValueError("A circular circuit has no authored corners")
        # A plan whose explicit angles cannot be closed is still compilable, but
        # only after the compiler adds counter-direction kinks. Catch the case
        # where a single corner already exceeds a full revolution.
        if any(
            corner.angle_degrees is not None and corner.angle_degrees >= FULL_REVOLUTION
            for corner in self.corners
        ):
            raise ValueError("A single corner cannot turn through a full revolution")
        # A requested P4 is a promise of three cars ahead, not merely an offset
        # for the player sprite. Fill unnamed places with ordinary racers so the
        # visible lineup, running order, and finish order agree.
        if self.npc_start_mode == "grid" and self.player_grid_position > len(self.npcs) + 1:
            self.npcs.extend(
                NpcSpec(label=f"grid opponent {number}")
                for number in range(len(self.npcs) + 1, self.player_grid_position)
            )
        return self


@dataclass(frozen=True)
class _ResolvedCorner:
    index: int
    signed_angle: float
    radius: float
    region: TrackRegion
    requested_angle: float | None
    requested_radius: CornerRadius
    entry_straight: float
    origin: Literal["requested", "closure-filler"]


@dataclass(frozen=True)
class _PathElement:
    """A straight or a circular arc, parameterized by arclength."""

    kind: Literal["straight", "arc"]
    length: float
    origin: tuple[float, float]
    heading: float
    radius: float = 0.0
    sweep: float = 0.0

    def point_at(self, distance: float) -> tuple[float, float]:
        if self.kind == "straight":
            direction = _unit(self.heading)
            return (
                self.origin[0] + direction[0] * distance,
                self.origin[1] + direction[1] * distance,
            )
        side = 1.0 if self.sweep >= 0 else -1.0
        normal = _unit(self.heading + side * 90.0)
        center = (self.origin[0] + normal[0] * self.radius, self.origin[1] + normal[1] * self.radius)
        travelled = math.degrees(distance / self.radius) * side
        spoke = _unit(self.heading - side * 90.0 + travelled)
        return (center[0] + spoke[0] * self.radius, center[1] + spoke[1] * self.radius)


@dataclass(frozen=True)
class CompiledTrack:
    """Uniformly sampled closed geometry plus its fidelity report."""

    centerline: list[Vec2]
    track_width: float
    sector_indices: tuple[int, ...]
    corner_indices: tuple[int, ...]
    report: TrackReport

    def progress_for_region(self, region: TrackRegion, bounds: Rect) -> float:
        """Return the lap fraction whose centerline sample sits in `region`."""
        if region == TrackRegion.AUTO:
            return 0.0
        target_x, target_y = _REGION_TARGETS[region]
        anchor = (bounds.x + target_x * bounds.width, bounds.y + target_y * bounds.height)
        best = min(
            range(len(self.centerline)),
            key=lambda index: (
                (self.centerline[index].x - anchor[0]) ** 2
                + (self.centerline[index].y - anchor[1]) ** 2
            ),
        )
        return best / len(self.centerline)

    def curvature_at(self, index: int) -> float:
        """Absolute heading change in degrees across the sample at `index`."""
        points = self.centerline
        count = len(points)
        before = points[(index - 2) % count]
        current = points[index % count]
        after = points[(index + 2) % count]
        return abs(_angle_delta(_bearing(before, current), _bearing(current, after)))


def compile_track(plan: TrackPlan, bounds: Rect, car_radius: float) -> CompiledTrack:
    """Compile a plan into a closed, uniformly sampled, drivable centerline."""
    if plan.loop_shape == "circle":
        return _compile_circle(plan, bounds, car_radius)
    relaxations: list[str] = []
    corners = _resolve_corners(plan, relaxations)
    margin = plan.track_width / 2 + car_radius + 6.0
    box = Rect(
        x=bounds.x + margin, y=bounds.y + margin,
        width=max(1.0, bounds.width - 2 * margin), height=max(1.0, bounds.height - 2 * margin),
    )
    # First pass finds the orientation. Second pass then lengthens the straights
    # that run along the canvas's long axis, so a wide box yields a genuine oval
    # with usable straights instead of a square loop that wastes the width.
    probe, _, _ = _walk_closed_path(corners, [])
    rotation, _, _ = _choose_placement(probe, corners, box)
    elements, closure_error, radius_scale = _walk_closed_path(
        corners, relaxations, aspect=box.width / box.height, rotation=rotation,
    )
    if radius_scale < 1.0:
        corners = [
            _ResolvedCorner(**{**corner.__dict__, "radius": corner.radius * radius_scale})
            for corner in corners
        ]
    rotation, scale, offset = _choose_placement(elements, corners, box)
    raw_length = sum(element.length for element in elements)
    total_length = raw_length * scale
    sample_count = max(64, min(320, round(total_length / CENTERLINE_SPACING)))
    raw = _sample_elements(elements, sample_count)
    placed = [
        Vec2(
            x=round(offset[0] + (point[0] * math.cos(math.radians(rotation)) - point[1] * math.sin(math.radians(rotation))) * scale, 3),
            y=round(offset[1] + (point[0] * math.sin(math.radians(rotation)) + point[1] * math.cos(math.radians(rotation))) * scale, 3),
        )
        for point in raw
    ]
    apex_offsets = _apex_offsets(elements)
    # Anchor index zero to the middle of the longest straight so the start/finish
    # gate, the grid, and the first checkpoint never land inside a corner.
    shift = _longest_straight_sample(elements, sample_count)
    centerline = placed[shift:] + placed[:shift]
    corner_indices = tuple(sorted(
        (round(apex / raw_length * sample_count) - shift) % sample_count
        for apex in apex_offsets
    ))
    sector_count = _sector_count(total_length, len(corners))
    sector_indices = _sector_indices(centerline, sector_count)
    report = _build_report(
        plan=plan, corners=corners, centerline=centerline, elements=elements, scale=scale,
        rotation=rotation, offset=offset, shift=shift, sample_count=sample_count,
        box=box, closure_error=closure_error, sector_count=sector_count,
        relaxations=relaxations,
    )
    return CompiledTrack(
        centerline=centerline, track_width=plan.track_width,
        sector_indices=sector_indices, corner_indices=corner_indices, report=report,
    )


def _compile_circle(plan: TrackPlan, bounds: Rect, car_radius: float) -> CompiledTrack:
    """Compile the dedicated no-corner primitive into a constant-radius loop.

    This intentionally bypasses the corner/closure solver. A true circle has
    continuous curvature and no apexes, so four large arcs would be both
    geometrically and semantically false.
    """
    margin = plan.track_width / 2 + car_radius + 6.0
    box = Rect(
        x=bounds.x + margin, y=bounds.y + margin,
        width=max(1.0, bounds.width - 2 * margin), height=max(1.0, bounds.height - 2 * margin),
    )
    radius = min(box.width, box.height) / 2
    circumference = 2 * math.pi * radius
    sample_count = max(64, min(320, round(circumference / CENTERLINE_SPACING)))
    center_x, center_y = box.x + box.width / 2, box.y + box.height / 2
    # In screen coordinates y grows down: increasing polar angle is clockwise.
    direction = 1.0 if plan.direction == "clockwise" else -1.0
    start_angle = -math.pi / 2
    centerline = [
        Vec2(
            x=round(center_x + radius * math.cos(start_angle + direction * 2 * math.pi * index / sample_count), 3),
            y=round(center_y + radius * math.sin(start_angle + direction * 2 * math.pi * index / sample_count), 3),
        )
        for index in range(sample_count)
    ]
    sector_count = _sector_count(circumference, 0)
    report = TrackReport(
        compiler="track-grammar-v1",
        loop_shape="circle",
        direction=plan.direction,
        corners=[],
        length_pixels=round(circumference, 1),
        longest_straight_pixels=0.0,
        minimum_radius_pixels=round(radius, 1),
        sector_count=sector_count,
        closure_error_pixels=0.0,
        centerline_spacing_pixels=round(circumference / sample_count, 2),
        angle_fidelity_degrees=0.0,
        region_fidelity=1.0,
    )
    return CompiledTrack(
        centerline=centerline, track_width=plan.track_width,
        sector_indices=_sector_indices(centerline, sector_count), corner_indices=(), report=report,
    )


def compile_certified_track(
    plan: TrackPlan, bounds: Rect, car_radius: float,
) -> tuple[CompiledTrack, list[str]]:
    """Compile a plan, walking a bounded relaxation ladder until it validates.

    An authored corner set can be closed and still be undrivable: a hairpin
    whose radius is under half a corridor width merges the road with itself.
    Rather than fail the brief or silently distort it, the compiler retries with
    progressively weaker structural assumptions and records each step, so the
    creator agent's fidelity remains measurable in the returned report.
    """
    attempts = list(_repair_ladder(plan))
    last: tuple[CompiledTrack, list[str]] | None = None
    for candidate, notes in attempts:
        track = compile_track(candidate, bounds, car_radius)
        findings = validate_track_geometry(
            track.centerline, candidate.track_width, bounds, car_radius,
        )
        report = track.report.model_copy(update={
            "relaxations": [*notes, *track.report.relaxations],
        })
        resolved = CompiledTrack(
            centerline=track.centerline, track_width=candidate.track_width,
            sector_indices=track.sector_indices, corner_indices=track.corner_indices,
            report=report,
        )
        if not findings:
            return resolved, []
        last = (resolved, findings)
    assert last is not None
    return last


def compile_drawn_track(
    normalized_points: list[Vec2], bounds: Rect, car_radius: float,
    track_width: float = 118.0,
) -> CompiledTrack:
    """Turn a freehand closed stroke into certified, uniformly sampled geometry.

    The drawing is a centerline, not a road polygon. Compilation closes it,
    smooths pointer jitter, fits it inside the same safety margin as authored
    tracks, and rejects self-overlap rather than inventing a different route.
    """
    if len(normalized_points) < 8:
        raise ValueError("Draw at least eight points before compiling the circuit")
    if any(not (0 <= point.x <= 1 and 0 <= point.y <= 1) for point in normalized_points):
        raise ValueError("Drawing points must stay inside the canvas")

    raw = _dedupe_drawn_points(normalized_points)
    if len(raw) < 8:
        raise ValueError("The stroke is too short; draw one complete loop")
    width = max(110.0, min(170.0, float(track_width)))
    # First normalize input density. Browser pointer events otherwise make the
    # same shape compile differently on a fast and a slow machine.
    raw = _resample_closed(raw, min(128, max(48, len(raw))))
    last_findings: list[str] = []
    for smoothing_rounds in range(2, 7):
        # Chaikin removes pointer noise but keeps the radius of a hand-drawn
        # right angle close to its original, tiny sampling interval. Relax the
        # low-resolution loop first, then round it: ordinary rectangles and
        # diamonds become driveable sweeping turns without joining separate arms
        # of a genuinely crossed or cramped drawing.
        smoothed = _relax_drawn_curve(raw, smoothing_rounds * 6)
        for _ in range(smoothing_rounds):
            smoothed = _chaikin_closed(smoothed)
        fitted = _fit_drawn_points(smoothed, bounds, width, car_radius)
        perimeter = _closed_length(fitted)
        sample_count = max(64, min(320, round(perimeter / CENTERLINE_SPACING)))
        centerline = _resample_closed(fitted, sample_count)
        shift = min(range(sample_count), key=lambda index: abs(_angle_delta(
            _bearing(centerline[(index - 2) % sample_count], centerline[index]),
            _bearing(centerline[index], centerline[(index + 2) % sample_count]),
        )))
        centerline = centerline[shift:] + centerline[:shift]
        findings = validate_track_geometry(centerline, width, bounds, car_radius)
        if findings:
            last_findings = findings
            continue
        total = _closed_length(centerline)
        sector_count = _sector_count(total, 4)
        sectors = _sector_indices(centerline, sector_count)
        spacing = total / len(centerline)
        signed_area = sum(
            point.x * centerline[(index + 1) % len(centerline)].y
            - centerline[(index + 1) % len(centerline)].x * point.y
            for index, point in enumerate(centerline)
        ) / 2
        report = TrackReport(
            compiler="drawing-compiler-v1",
            loop_shape="drawn",
            direction="clockwise" if signed_area > 0 else "counterclockwise",
            corners=[], length_pixels=round(total, 1),
            longest_straight_pixels=round(_longest_drawn_straight(centerline), 1),
            minimum_radius_pixels=round(minimum_corner_radius(centerline), 1),
            sector_count=sector_count, closure_error_pixels=0,
            centerline_spacing_pixels=round(spacing, 2),
            relaxations=[
                "Closed the freehand stroke and smoothed pointer jitter into a racing centerline."
            ],
        )
        return CompiledTrack(
            centerline=centerline, track_width=width,
            sector_indices=sectors, corner_indices=(), report=report,
        )
    detail = "; ".join(dict.fromkeys(last_findings)) or "the stroke could not form a road"
    raise ValueError(
        "This drawing cannot become a playable circuit without changing its shape: "
        f"{detail}. Leave more space between nearby sections and round very tight corners."
    )


def _dedupe_drawn_points(points: list[Vec2]) -> list[Vec2]:
    kept: list[Vec2] = []
    for point in points:
        if not kept or math.hypot(point.x - kept[-1].x, point.y - kept[-1].y) >= .004:
            kept.append(point.model_copy())
    if len(kept) > 1 and math.hypot(kept[0].x - kept[-1].x, kept[0].y - kept[-1].y) < .004:
        kept.pop()
    return kept


def _closed_length(points: list[Vec2]) -> float:
    return sum(math.hypot(
        points[(index + 1) % len(points)].x - point.x,
        points[(index + 1) % len(points)].y - point.y,
    ) for index, point in enumerate(points))


def _resample_closed(points: list[Vec2], count: int) -> list[Vec2]:
    lengths = [0.0]
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        lengths.append(lengths[-1] + math.hypot(following.x - point.x, following.y - point.y))
    total = lengths[-1]
    if total <= 1e-6:
        raise ValueError("The drawing needs a visible loop, not a single point")
    result: list[Vec2] = []
    segment = 0
    for sample in range(count):
        distance = total * sample / count
        while segment + 1 < len(lengths) and lengths[segment + 1] < distance:
            segment += 1
        start = points[segment % len(points)]
        end = points[(segment + 1) % len(points)]
        span = max(1e-9, lengths[segment + 1] - lengths[segment])
        fraction = (distance - lengths[segment]) / span
        result.append(Vec2(
            x=round(start.x + (end.x - start.x) * fraction, 4),
            y=round(start.y + (end.y - start.y) * fraction, 4),
        ))
    return result


def _chaikin_closed(points: list[Vec2]) -> list[Vec2]:
    smoothed: list[Vec2] = []
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        smoothed.extend((
            Vec2(x=.75 * point.x + .25 * following.x, y=.75 * point.y + .25 * following.y),
            Vec2(x=.25 * point.x + .75 * following.x, y=.25 * point.y + .75 * following.y),
        ))
    return smoothed


def _relax_drawn_curve(points: list[Vec2], rounds: int) -> list[Vec2]:
    """Open freehand corners before high-resolution interpolation.

    Applying this to the normalized, low-resolution stroke keeps the smoothing
    scale tied to the visible drawing rather than to browser pointer frequency.
    A short cyclic diffusion pass rounds sharp corners but retains the circuit's
    overall topology; normal geometry validation still rejects crossings and
    road corridors that would overlap.
    """
    relaxed = points
    for _ in range(rounds):
        relaxed = [
            Vec2(
                x=(.2 * relaxed[(index - 1) % len(relaxed)].x
                   + .6 * point.x
                   + .2 * relaxed[(index + 1) % len(relaxed)].x),
                y=(.2 * relaxed[(index - 1) % len(relaxed)].y
                   + .6 * point.y
                   + .2 * relaxed[(index + 1) % len(relaxed)].y),
            )
            for index, point in enumerate(relaxed)
        ]
    return relaxed


def _fit_drawn_points(points: list[Vec2], bounds: Rect, width: float, car_radius: float) -> list[Vec2]:
    margin = width / 2 + car_radius + 8
    available_width = bounds.width - 2 * margin
    available_height = bounds.height - 2 * margin
    min_x, max_x = min(point.x for point in points), max(point.x for point in points)
    min_y, max_y = min(point.y for point in points), max(point.y for point in points)
    source_width, source_height = max_x - min_x, max_y - min_y
    if source_width < .12 or source_height < .12:
        raise ValueError("The drawing needs a loop with visible width and height")
    scale = min(available_width / source_width, available_height / source_height)
    used_width, used_height = source_width * scale, source_height * scale
    offset_x = bounds.x + margin + (available_width - used_width) / 2
    offset_y = bounds.y + margin + (available_height - used_height) / 2
    return [Vec2(
        x=round(offset_x + (point.x - min_x) * scale, 3),
        y=round(offset_y + (point.y - min_y) * scale, 3),
    ) for point in points]


def _longest_drawn_straight(points: list[Vec2]) -> float:
    spacing = _closed_length(points) / len(points)
    longest = run = 0
    for index in range(len(points) * 2):
        turn = abs(_angle_delta(
            _bearing(points[(index - 2) % len(points)], points[index % len(points)]),
            _bearing(points[index % len(points)], points[(index + 2) % len(points)]),
        ))
        run = run + 1 if turn < 4 else 0
        longest = max(longest, min(run, len(points)))
    return longest * spacing


def _repair_ladder(plan: TrackPlan):
    """Deterministic sequence of progressively relaxed plans.

    Each rung surrenders a little more of the authored shape, cheapest first:
    corner sharpness, then straight length, then extra linking corners, and only
    last the authored turn angles themselves.
    """
    yield plan, []
    if plan.loop_shape == "circle":
        return
    yield _shorten_straights(plan), [
        "Shortened the longest straights so the circuit fits the canvas.",
    ]
    for floor in (CornerRadius.TIGHT, CornerRadius.MEDIUM, CornerRadius.OPEN):
        opened = _open_tight_corners(plan, floor)
        note = f"Opened every corner to at least a {floor.value} radius so the corridor cannot merge with itself."
        yield opened, [note]
        combined = _shorten_straights(opened)
        yield combined, [note, "Shortened the longest straights so the circuit fits the canvas."]
        for extra in (1, 2, 3):
            yield _add_sweeps(combined, extra), [
                note,
                f"Shortened the longest straights and added {extra} linking sweep(s) to round "
                "out the return loop.",
            ]
    widest = _add_sweeps(_shorten_straights(_open_tight_corners(plan, CornerRadius.OPEN)), 3)
    yield _soften_extreme_angles(widest), [
        "Reduced the sharpest authored corner; the requested angle could not be closed into a "
        "drivable corridor at this track width.",
    ]


def _open_tight_corners(plan: TrackPlan, floor: CornerRadius = CornerRadius.TIGHT) -> TrackPlan:
    """Raise every corner to at least `floor`, leaving opener corners alone."""
    order = (
        CornerRadius.HAIRPIN, CornerRadius.TIGHT, CornerRadius.MEDIUM,
        CornerRadius.OPEN, CornerRadius.SWEEPING,
    )
    floor_index = order.index(floor)
    return plan.model_copy(update={"corners": [
        corner.model_copy(update={
            "radius": order[max(order.index(corner.radius), floor_index)],
        })
        for corner in plan.corners
    ]})


def _shorten_straights(plan: TrackPlan) -> TrackPlan:
    demotion = {
        StraightLength.LONG: StraightLength.MEDIUM,
        StraightLength.MEDIUM: StraightLength.SHORT,
    }
    return plan.model_copy(update={"corners": [
        corner.model_copy(update={
            "exit_straight": demotion.get(corner.exit_straight, corner.exit_straight),
        })
        for corner in plan.corners
    ]})


def _add_sweeps(plan: TrackPlan, extra: int) -> TrackPlan:
    room = max(0, 10 - len(plan.corners))
    additions = [
        CornerSpec(radius=CornerRadius.MEDIUM, exit_straight=StraightLength.MEDIUM,
                   label=f"linking sweep {index + 1}")
        for index in range(min(extra, room))
    ]
    if not additions:
        return plan
    return plan.model_copy(update={"corners": [*plan.corners, *additions]})


def _soften_extreme_angles(plan: TrackPlan) -> TrackPlan:
    sharpest = max(
        (corner.angle_degrees or 0 for corner in plan.corners), default=0,
    )
    if sharpest <= 100:
        return plan
    return plan.model_copy(update={"corners": [
        corner.model_copy(update={
            "angle_degrees": max(MIN_CORNER_ANGLE, corner.angle_degrees * .62),
        })
        if corner.angle_degrees == sharpest else corner
        for corner in plan.corners
    ]})


def validate_track_geometry(
    centerline: list[Vec2], track_width: float, bounds: Rect, car_radius: float,
) -> list[str]:
    """Reject geometry the runtime could not fairly simulate or render."""
    findings: list[str] = []
    count = len(centerline)
    if count < 32:
        findings.append("Compiled centerline needs at least 32 uniform samples")
        return findings
    spacings = [
        math.hypot(
            centerline[(index + 1) % count].x - centerline[index].x,
            centerline[(index + 1) % count].y - centerline[index].y,
        )
        for index in range(count)
    ]
    if min(spacings) <= 1e-6:
        findings.append("Centerline cannot contain zero-length segments")
    elif max(spacings) / min(spacings) > 1.6:
        findings.append("Centerline sampling is not uniform enough for index-based lookahead")
    margin = track_width / 2 + car_radius
    if any(
        not (margin <= point.x <= bounds.x + bounds.width - margin)
        or not (margin <= point.y <= bounds.y + bounds.height - margin)
        for point in centerline
    ):
        findings.append("Circuit does not leave a safe track margin inside the scene bounds")
    # A corner whose diameter is narrower than the corridor merges the road with
    # itself: the two arms overlap, so a car can cut the corner and `_on_track`
    # cannot tell them apart. The pairwise sweep below deliberately excludes
    # points a short arclength apart, which is exactly where this happens, so the
    # radius is checked directly.
    required_diameter = track_width + 2 * car_radius
    if 2 * minimum_corner_radius(centerline) < required_diameter:
        findings.append(
            "A corner is tighter than the corridor is wide; the road merges with itself"
        )
    # Two stretches of road that pass closer than a full corridor width would
    # merge: `_on_track`, forward progress, and gate crossing all stop being
    # well defined. Corner arcs are excluded by arclength, not by index.
    spacing = sum(spacings) / count
    separation_window = max(12, round(300.0 / max(1e-6, spacing)))
    required = track_width + 2 * car_radius
    for index in range(count):
        for other in range(index + separation_window, count - separation_window + index + 1):
            if other >= count:
                break
            distance = math.hypot(
                centerline[other].x - centerline[index].x,
                centerline[other].y - centerline[index].y,
            )
            if distance < required:
                findings.append("Circuit corridor overlaps itself; corners are packed too tightly")
                return findings
    return findings


def minimum_corner_radius(centerline: list[Vec2]) -> float:
    """Estimate the tightest radius on a uniformly sampled closed centerline."""
    count = len(centerline)
    spacing = sum(
        math.hypot(
            centerline[(index + 1) % count].x - centerline[index].x,
            centerline[(index + 1) % count].y - centerline[index].y,
        )
        for index in range(count)
    ) / count
    tightest = float("inf")
    for index in range(count):
        turn = abs(_angle_delta(
            _bearing(centerline[(index - 2) % count], centerline[index]),
            _bearing(centerline[index], centerline[(index + 2) % count]),
        ))
        if turn < 1e-6:
            continue
        tightest = min(tightest, 2 * spacing / math.radians(turn))
    return tightest if math.isfinite(tightest) else 1e9


def archetype_plan(circuit: str, surface: str = "asphalt") -> TrackPlan:
    """The three legacy archetypes, expressed in the general grammar."""
    if circuit == "oval":
        corners = [
            CornerSpec(angle_degrees=90, radius=CornerRadius.OPEN, exit_straight=StraightLength.LONG,
                       region=TrackRegion.BOTTOM_RIGHT, label="turn 1"),
            CornerSpec(angle_degrees=90, radius=CornerRadius.OPEN, exit_straight=StraightLength.LONG,
                       region=TrackRegion.TOP_RIGHT, label="turn 2"),
            CornerSpec(angle_degrees=90, radius=CornerRadius.OPEN, exit_straight=StraightLength.LONG,
                       region=TrackRegion.TOP_LEFT, label="turn 3"),
            CornerSpec(angle_degrees=90, radius=CornerRadius.OPEN, exit_straight=StraightLength.LONG,
                       region=TrackRegion.BOTTOM_LEFT, label="turn 4"),
        ]
        title = "North Loop"
    elif circuit == "chicane":
        corners = [
            CornerSpec(angle_degrees=85, radius=CornerRadius.MEDIUM, exit_straight=StraightLength.SHORT,
                       region=TrackRegion.BOTTOM_RIGHT, label="turn 1"),
            CornerSpec(angle_degrees=55, radius=CornerRadius.TIGHT, exit_straight=StraightLength.SHORT,
                       region=TrackRegion.RIGHT, label="chicane entry"),
            CornerSpec(direction="right", angle_degrees=45, radius=CornerRadius.TIGHT,
                       exit_straight=StraightLength.SHORT, region=TrackRegion.TOP_RIGHT, label="chicane exit"),
            CornerSpec(angle_degrees=95, radius=CornerRadius.MEDIUM, exit_straight=StraightLength.MEDIUM,
                       region=TrackRegion.TOP_CENTER, label="turn 4"),
            CornerSpec(angle_degrees=80, radius=CornerRadius.MEDIUM, exit_straight=StraightLength.MEDIUM,
                       region=TrackRegion.TOP_LEFT, label="turn 5"),
            CornerSpec(angle_degrees=90, radius=CornerRadius.MEDIUM, exit_straight=StraightLength.LONG,
                       region=TrackRegion.BOTTOM_LEFT, label="turn 6"),
        ]
        title = "Orange Gate Raceway"
    else:
        corners = [
            CornerSpec(angle_degrees=75, radius=CornerRadius.MEDIUM, exit_straight=StraightLength.MEDIUM,
                       region=TrackRegion.BOTTOM_RIGHT, label="turn 1"),
            CornerSpec(angle_degrees=65, radius=CornerRadius.TIGHT, exit_straight=StraightLength.SHORT,
                       region=TrackRegion.RIGHT, label="turn 2"),
            CornerSpec(angle_degrees=60, radius=CornerRadius.TIGHT, exit_straight=StraightLength.SHORT,
                       region=TrackRegion.TOP_RIGHT, label="turn 3"),
            CornerSpec(angle_degrees=55, radius=CornerRadius.MEDIUM, exit_straight=StraightLength.MEDIUM,
                       region=TrackRegion.TOP_CENTER, label="turn 4"),
            CornerSpec(angle_degrees=50, radius=CornerRadius.MEDIUM, exit_straight=StraightLength.SHORT,
                       region=TrackRegion.TOP_LEFT, label="turn 5"),
            CornerSpec(angle_degrees=55, radius=CornerRadius.TIGHT, exit_straight=StraightLength.MEDIUM,
                       region=TrackRegion.BOTTOM_LEFT, label="turn 6"),
        ]
        title = "Switchback Circuit"
    return TrackPlan(
        title=title,
        rationale=f"The {circuit} archetype expressed in the general corner grammar on {surface}.",
        corners=corners, surface=surface,
    )


def _resolve_corners(plan: TrackPlan, relaxations: list[str]) -> list[_ResolvedCorner]:
    """Fix every turn angle so the signed rotation is exactly one revolution."""
    circuit_sign = -1.0 if plan.direction == "counterclockwise" else 1.0
    required_total = circuit_sign * FULL_REVOLUTION
    authored: list[dict] = []
    for corner in plan.corners:
        direction = corner.direction or ("left" if circuit_sign < 0 else "right")
        sign = -1.0 if direction == "left" else 1.0
        authored.append({
            "direction": direction, "sign": sign, "angle": corner.angle_degrees,
            "radius": corner.radius, "region": corner.region,
            "entry_straight": _STRAIGHT_PIXELS[corner.exit_straight],
            "origin": "requested",
        })
    # `entry_straight` is the straight preceding a corner, which is the previous
    # corner's requested exit. Rotate the authored exits by one to match.
    exits = [item["entry_straight"] for item in authored]
    for index, item in enumerate(authored):
        item["entry_straight"] = exits[index - 1]

    fixed = sum(item["sign"] * item["angle"] for item in authored if item["angle"] is not None)
    solved = [item for item in authored if item["angle"] is None]
    residual = required_total - fixed
    if solved:
        share = residual / len(solved)
        for item in solved:
            magnitude = min(MAX_CORNER_ANGLE, max(MIN_CORNER_ANGLE, abs(share)))
            item["sign"] = math.copysign(1.0, share) if share != 0 else item["sign"]
            item["angle"] = magnitude
        fixed = sum(item["sign"] * item["angle"] for item in authored)
        residual = required_total - fixed

    # Whatever rotation is still missing becomes explicit filler corners rather
    # than a silent distortion of the authored angles.
    guard = 0
    while abs(residual) > 1e-6 and len(authored) < MAX_TOTAL_CORNERS and guard < MAX_TOTAL_CORNERS:
        guard += 1
        magnitude = min(MAX_CORNER_ANGLE, max(MIN_CORNER_ANGLE, abs(residual)))
        sign = math.copysign(1.0, residual)
        authored.append({
            "direction": "left" if sign < 0 else "right", "sign": sign, "angle": magnitude,
            "radius": CornerRadius.MEDIUM, "region": TrackRegion.AUTO,
            "entry_straight": _STRAIGHT_PIXELS[StraightLength.MEDIUM],
            "origin": "closure-filler",
        })
        residual = required_total - sum(item["sign"] * item["angle"] for item in authored)
    if abs(residual) > 1e-6:
        # Only reachable when the authored angles alone overshoot a revolution by
        # more than the filler budget. Scale them and say so.
        total = sum(item["sign"] * item["angle"] for item in authored)
        factor = required_total / total if abs(total) > 1e-9 else 1.0
        for item in authored:
            item["angle"] = min(MAX_CORNER_ANGLE, max(MIN_CORNER_ANGLE, item["angle"] * abs(factor)))
        drift = required_total - sum(item["sign"] * item["angle"] for item in authored)
        authored[0]["angle"] = max(
            MIN_CORNER_ANGLE,
            min(MAX_CORNER_ANGLE, authored[0]["angle"] + drift * authored[0]["sign"]),
        )
        relaxations.append(
            f"Scaled authored turn angles by {abs(factor):.2f} because they exceeded one "
            "revolution by more than the filler-corner budget."
        )
    filler = sum(1 for item in authored if item["origin"] == "closure-filler")
    if filler:
        counter = sum(
            1 for item in authored
            if item["origin"] == "closure-filler" and item["sign"] != circuit_sign
        )
        relaxations.append(
            f"Added {filler} closure corner(s)"
            + (f", {counter} against the circuit direction," if counter else "")
            + " so the signed turns sum to one revolution."
        )
    return [
        _ResolvedCorner(
            index=index,
            signed_angle=item["sign"] * item["angle"],
            radius=_RADIUS_PIXELS[item["radius"]],
            region=item["region"],
            requested_angle=(
                plan.corners[index].angle_degrees if index < len(plan.corners) else None
            ),
            requested_radius=item["radius"],
            entry_straight=item["entry_straight"],
            origin=item["origin"],
        )
        for index, item in enumerate(authored)
    ]


def _walk_closed_path(
    corners: list[_ResolvedCorner], relaxations: list[str],
    aspect: float = 1.0, rotation: float = 0.0,
) -> tuple[list[_PathElement], float, float]:
    """Solve straight lengths (and if needed radii) for exact position closure.

    Heading closure already holds because the signed angles sum to a revolution.
    Position closure is two scalar equations in the straight lengths, so the
    solve is the least-squares problem `min ||L - requested||` subject to
    `U·L = -arc_sum` and `L >= MIN_STRAIGHT_PIXELS`. An active-set pass handles
    the inequality exactly: pin the straights that want to go short and re-solve
    on the rest, which terminates in at most one round per straight. Shrinking
    every radius enlarges the feasible set when even that has no solution.
    """
    headings: list[float] = []
    heading = 0.0
    for corner in corners:
        headings.append(heading)
        heading += corner.signed_angle
    requested = [
        corner.entry_straight * _aspect_gain(headings[index] + rotation, aspect)
        for index, corner in enumerate(corners)
    ]
    radius_scale = 1.0
    for _ in range(30):
        arc_sum = _arc_displacement(corners, headings, radius_scale)
        lengths = _solve_straights(requested, headings, arc_sum)
        if lengths is not None:
            closure_error = math.hypot(*_closure_residual(lengths, headings, arc_sum))
            return _elements_for(corners, lengths, headings, radius_scale), closure_error, radius_scale
        radius_scale *= .88
    if radius_scale < 1.0:
        relaxations.append(
            f"Shrank every corner radius to {radius_scale:.2f}x of the requested size while "
            "searching for a closed loop."
        )
    relaxations.append(
        "Could not close the loop with positive straights at any radius; this corner set is "
        "geometrically impossible and the plan will be rejected."
    )
    arc_sum = _arc_displacement(corners, headings, radius_scale)
    lengths = [max(MIN_STRAIGHT_PIXELS, value) for value in requested]
    return (
        _elements_for(corners, lengths, headings, radius_scale),
        math.hypot(*_closure_residual(lengths, headings, arc_sum)),
        radius_scale,
    )


def _arc_displacement(
    corners: list[_ResolvedCorner], headings: list[float], radius_scale: float,
) -> tuple[float, float]:
    """Total displacement contributed by every corner arc's chord."""
    total_x = total_y = 0.0
    for index, corner in enumerate(corners):
        chord = 2 * corner.radius * radius_scale * math.sin(math.radians(abs(corner.signed_angle)) / 2)
        direction = _unit(headings[index] + corner.signed_angle / 2)
        total_x += chord * direction[0]
        total_y += chord * direction[1]
    return (total_x, total_y)


def _solve_straights(
    requested: list[float], headings: list[float], arc_sum: tuple[float, float],
) -> list[float] | None:
    """Exactly closing straight lengths, or None when none exist."""
    count = len(requested)
    pinned: set[int] = set()
    for _ in range(count + 1):
        free = [index for index in range(count) if index not in pinned]
        if len(free) < 2:
            return None
        lengths = [
            MIN_STRAIGHT_PIXELS if index in pinned else requested[index]
            for index in range(count)
        ]
        residual = _closure_residual(lengths, headings, arc_sum)
        corrected = _minimum_norm_step(lengths, headings, residual, free)
        if corrected is None:
            return None
        violated = [index for index in free if corrected[index] < MIN_STRAIGHT_PIXELS - 1e-9]
        if not violated:
            return corrected
        pinned.update(violated)
    return None


def _aspect_gain(world_heading: float, aspect: float) -> float:
    """Stretch straights along the canvas's long axis without touching angles."""
    if abs(aspect - 1.0) < 1e-9:
        return 1.0
    radians = math.radians(world_heading)
    if aspect > 1.0:
        return 1.0 + (aspect - 1.0) * math.cos(radians) ** 2
    return 1.0 + (1.0 / aspect - 1.0) * math.sin(radians) ** 2


def _closure_residual(
    lengths: list[float], headings: list[float], arc_sum: tuple[float, float],
) -> tuple[float, float]:
    return (
        arc_sum[0] + sum(length * _unit(heading)[0] for length, heading in zip(lengths, headings)),
        arc_sum[1] + sum(length * _unit(heading)[1] for length, heading in zip(lengths, headings)),
    )


def _minimum_norm_step(
    lengths: list[float], headings: list[float], residual: tuple[float, float],
    free: list[int],
) -> list[float] | None:
    """Cancel `residual` with the smallest correction to the free straights only.

    Returns None when the free directions are collinear, because then no
    correction can cancel a residual off that line.
    """
    directions = [_unit(heading) for heading in headings]
    a11 = sum(directions[index][0] ** 2 for index in free)
    a12 = sum(directions[index][0] * directions[index][1] for index in free)
    a22 = sum(directions[index][1] ** 2 for index in free)
    determinant = a11 * a22 - a12 * a12
    if abs(determinant) < 1e-9:
        return None
    inverse = (a22 / determinant, -a12 / determinant, a11 / determinant)
    multiplier = (
        inverse[0] * residual[0] + inverse[1] * residual[1],
        inverse[1] * residual[0] + inverse[2] * residual[1],
    )
    corrected = list(lengths)
    for index in free:
        corrected[index] -= (
            directions[index][0] * multiplier[0] + directions[index][1] * multiplier[1]
        )
    return corrected


def _elements_for(
    corners: list[_ResolvedCorner], lengths: list[float], headings: list[float],
    radius_scale: float,
) -> list[_PathElement]:
    elements: list[_PathElement] = []
    position = (0.0, 0.0)
    for index, corner in enumerate(corners):
        heading = headings[index]
        straight = _PathElement(
            kind="straight", length=lengths[index], origin=position, heading=heading,
        )
        elements.append(straight)
        position = straight.point_at(straight.length)
        radius = corner.radius * radius_scale
        arc = _PathElement(
            kind="arc", length=radius * math.radians(abs(corner.signed_angle)),
            origin=position, heading=heading, radius=radius, sweep=corner.signed_angle,
        )
        elements.append(arc)
        position = arc.point_at(arc.length)
    return elements


def _sample_elements(elements: list[_PathElement], count: int) -> list[tuple[float, float]]:
    total = sum(element.length for element in elements)
    boundaries: list[float] = []
    running = 0.0
    for element in elements:
        boundaries.append(running)
        running += element.length
    samples: list[tuple[float, float]] = []
    cursor = 0
    for index in range(count):
        distance = total * index / count
        while cursor + 1 < len(elements) and distance >= boundaries[cursor + 1]:
            cursor += 1
        samples.append(elements[cursor].point_at(distance - boundaries[cursor]))
    return samples


def _apex_offsets(elements: list[_PathElement]) -> list[float]:
    """Arclength of every corner's mid-arc point."""
    offsets: list[float] = []
    running = 0.0
    for element in elements:
        if element.kind == "arc":
            offsets.append(running + element.length / 2)
        running += element.length
    return offsets


def _longest_straight_sample(elements: list[_PathElement], count: int) -> int:
    total = sum(element.length for element in elements)
    best_offset, best_length, running = 0.0, -1.0, 0.0
    for element in elements:
        if element.kind == "straight" and element.length > best_length:
            best_length, best_offset = element.length, running + element.length / 2
        running += element.length
    return round(best_offset / total * count) % count


def _choose_placement(
    elements: list[_PathElement], corners: list[_ResolvedCorner], box: Rect,
) -> tuple[float, float, tuple[float, float]]:
    """Pick the rotation, scale, and offset that best honor requested regions."""
    probe = _sample_elements(elements, 240)
    apexes = [
        _point_at_arclength(elements, offset) for offset in _apex_offsets(elements)
    ]
    constrained = [
        (index, corner.region) for index, corner in enumerate(corners)
        if corner.region != TrackRegion.AUTO and index < len(apexes)
    ]
    candidates: list[tuple[float, float, tuple[float, float], int, float]] = []
    for step in range(180):
        rotation = step * 2.0
        radians = math.radians(rotation)
        cos_r, sin_r = math.cos(radians), math.sin(radians)
        rotated = [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for x, y in probe]
        min_x = min(point[0] for point in rotated)
        max_x = max(point[0] for point in rotated)
        min_y = min(point[1] for point in rotated)
        max_y = max(point[1] for point in rotated)
        span_x, span_y = max(1e-6, max_x - min_x), max(1e-6, max_y - min_y)
        scale = min(box.width / span_x, box.height / span_y)
        offset = (
            box.x + box.width / 2 - (min_x + max_x) / 2 * scale,
            box.y + box.height / 2 - (min_y + max_y) / 2 * scale,
        )
        penalty, misses = 0.0, 0
        for index, region in constrained:
            x, y = apexes[index]
            normalized = (
                (offset[0] + (x * cos_r - y * sin_r) * scale - box.x) / box.width,
                (offset[1] + (x * sin_r + y * cos_r) * scale - box.y) / box.height,
            )
            target_x, target_y = _REGION_TARGETS[region]
            penalty += (normalized[0] - target_x) ** 2 + (normalized[1] - target_y) ** 2
            misses += int(_nearest_ring_region(normalized) != region)
        candidates.append((rotation, scale, offset, misses, penalty))
    best_scale = max(item[1] for item in candidates)
    # Satisfying the brief's requested regions strictly outranks filling the
    # canvas, but a rotation that shrinks the circuit badly is never worth it.
    usable = [item for item in candidates if item[1] >= best_scale * .6] or candidates
    rotation, scale, offset, _, _ = min(
        usable, key=lambda item: (item[3], round(item[4], 6), -round(item[1], 6), item[0]),
    )
    return rotation, scale, offset


def _nearest_ring_region(normalized: tuple[float, float]) -> TrackRegion:
    """Classify a box-normalized point into the closest on-ring screen region."""
    return min(
        _RING_REGIONS,
        key=lambda region: (
            (normalized[0] - _REGION_TARGETS[region][0]) ** 2
            + (normalized[1] - _REGION_TARGETS[region][1]) ** 2
        ),
    )


def _point_at_arclength(elements: list[_PathElement], distance: float) -> tuple[float, float]:
    running = 0.0
    for element in elements:
        if distance <= running + element.length or element is elements[-1]:
            return element.point_at(max(0.0, min(element.length, distance - running)))
        running += element.length
    return elements[-1].point_at(elements[-1].length)


def _sector_count(total_length: float, corner_count: int) -> int:
    """Roughly one ordered gate per corner, including the finish line."""
    return max(3, min(9, max(round(total_length / 560.0), min(7, corner_count))))


def _sector_indices(centerline: list[Vec2], sector_count: int) -> tuple[int, ...]:
    """Even lap fractions, nudged onto the straightest nearby sample.

    A gate inside a corner is crossed at an angle and reads as ambiguous to both
    a human and a policy, so each gate slides to the flattest sample in a small
    window around its nominal position.
    """
    count = len(centerline)

    def curvature(index: int) -> float:
        before = centerline[(index - 2) % count]
        current = centerline[index % count]
        after = centerline[(index + 2) % count]
        return abs(_angle_delta(_bearing(before, current), _bearing(current, after)))

    window = max(1, count // (sector_count * 4))
    chosen: list[int] = []
    for sector in range(1, sector_count):
        nominal = round(count * sector / sector_count)
        best = min(
            range(nominal - window, nominal + window + 1),
            key=lambda index: (round(curvature(index % count), 4), abs(index - nominal)),
        )
        candidate = best % count
        if candidate in chosen or candidate == 0:
            candidate = nominal % count
        chosen.append(candidate)
    # Index zero is the finish line and is always last in crossing order.
    return (*chosen, 0)


def _build_report(
    *, plan: TrackPlan, corners: list[_ResolvedCorner], centerline: list[Vec2],
    elements: list[_PathElement], scale: float, rotation: float,
    offset: tuple[float, float], shift: int, sample_count: int, box: Rect,
    closure_error: float, sector_count: int, relaxations: list[str],
) -> TrackReport:
    radians = math.radians(rotation)
    cos_r, sin_r = math.cos(radians), math.sin(radians)
    total = sum(element.length for element in elements)
    apex_offsets = _apex_offsets(elements)
    corner_reports: list[CornerReport] = []
    angle_error = 0.0
    achieved_by_request: dict[TrackRegion, list[TrackRegion]] = {}
    for index, corner in enumerate(corners):
        local = _point_at_arclength(elements, apex_offsets[index])
        apex = Vec2(
            x=round(offset[0] + (local[0] * cos_r - local[1] * sin_r) * scale, 2),
            y=round(offset[1] + (local[0] * sin_r + local[1] * cos_r) * scale, 2),
        )
        achieved_region = _nearest_ring_region(
            ((apex.x - box.x) / box.width, (apex.y - box.y) / box.height),
        )
        if corner.region != TrackRegion.AUTO:
            achieved_by_request.setdefault(corner.region, []).append(achieved_region)
        achieved_angle = abs(corner.signed_angle)
        if corner.requested_angle is not None:
            angle_error = max(angle_error, abs(achieved_angle - corner.requested_angle))
        entry_index = (round(apex_offsets[index] / total * sample_count) - shift) % sample_count
        corner_reports.append(CornerReport(
            index=index,
            direction="left" if corner.signed_angle < 0 else "right",
            requested_angle_degrees=corner.requested_angle,
            achieved_angle_degrees=round(achieved_angle, 2),
            requested_region=corner.region,
            achieved_region=achieved_region,
            requested_radius=corner.requested_radius,
            achieved_radius_pixels=round(corner.radius * scale, 1),
            entry_progress_percent=round(entry_index / sample_count * 100, 1),
            apex=apex,
            origin=corner.origin,
        ))
    longest_straight = max(
        (element.length for element in elements if element.kind == "straight"), default=0.0,
    )
    count = len(centerline)
    spacing = sum(
        math.hypot(
            centerline[(index + 1) % count].x - centerline[index].x,
            centerline[(index + 1) % count].y - centerline[index].y,
        )
        for index in range(count)
    ) / count
    return TrackReport(
        loop_shape="cornered",
        direction=plan.direction,
        corners=corner_reports,
        length_pixels=round(total * scale, 1),
        longest_straight_pixels=round(longest_straight * scale, 1),
        minimum_radius_pixels=round(minimum_corner_radius(centerline), 1),
        sector_count=sector_count,
        closure_error_pixels=round(closure_error * scale, 4),
        centerline_spacing_pixels=round(spacing, 2),
        angle_fidelity_degrees=round(angle_error, 2),
        region_fidelity=_region_fidelity(achieved_by_request),
        relaxations=relaxations,
    )


def _region_fidelity(achieved_by_request: dict[TrackRegion, list[TrackRegion]]) -> float:
    """Score requested regions, not individual corners.

    "Put a chicane on the right side" asks for one feature in one region, and a
    chicane is two corners. Scoring each corner separately would cap that brief
    at 50% no matter how well it was satisfied, so a requested region counts as
    honored when one of its corners lands exactly in it and none of them stray
    more than one region away.
    """
    if not achieved_by_request:
        return 1.0
    honored = 0
    for requested, achieved in achieved_by_request.items():
        exact = any(region == requested for region in achieved)
        contained = all(_ring_distance(region, requested) <= 1 for region in achieved)
        honored += int(exact and contained)
    return round(honored / len(achieved_by_request), 3)


def _ring_distance(left: TrackRegion, right: TrackRegion) -> int:
    """Cyclic separation between two on-ring regions, in region steps."""
    if left not in _RING_ORDER or right not in _RING_ORDER:
        return 0 if left == right else len(_RING_ORDER)
    delta = abs(_RING_ORDER.index(left) - _RING_ORDER.index(right))
    return min(delta, len(_RING_ORDER) - delta)


def _unit(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return (math.cos(radians), math.sin(radians))


def _bearing(origin: Vec2, target: Vec2) -> float:
    return math.degrees(math.atan2(target.y - origin.y, target.x - origin.x)) % 360


def _angle_delta(current: float, target: float) -> float:
    return (target - current + 180) % 360 - 180


def track_plan_schema() -> dict:
    """Structured-output schema handed to the environment-creator model."""
    corner = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["left", "right"]},
            "angle_degrees": {"type": "number"},
            "radius": {"type": "string", "enum": [item.value for item in CornerRadius]},
            "region": {"type": "string", "enum": [item.value for item in TrackRegion]},
            "exit_straight": {"type": "string", "enum": [item.value for item in StraightLength]},
            "label": {"type": "string"},
        },
        "required": ["direction", "angle_degrees", "radius", "region", "exit_straight", "label"],
        "additionalProperties": False,
    }
    barrier = {
        "type": "object",
        "properties": {
            "region": {"type": "string", "enum": [item.value for item in TrackRegion]},
            "side": {"type": "string", "enum": ["left", "right", "auto"]},
            "shape": {"type": "string", "enum": ["circle", "box", "oriented-box"]},
            "label": {"type": "string"},
        },
        "required": ["region", "side", "shape", "label"],
        "additionalProperties": False,
    }
    npc = {
        "type": "object",
        "properties": {
            "profile": {"type": "string", "enum": [item.value for item in NpcProfile]},
            "pace": {"type": "number"},
            "skill": {"type": "number"},
            "aggression": {"type": "number"},
            "intelligence": {"type": "number"},
            "defends": {"type": "boolean"},
            "uses_nitro": {"type": "boolean"},
            "label": {"type": "string"},
        },
        "required": [
            "profile", "pace", "skill", "aggression", "intelligence", "defends",
            "uses_nitro", "label",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "rationale": {"type": "string"},
            "direction": {"type": "string", "enum": ["clockwise", "counterclockwise"]},
            "loop_shape": {"type": "string", "enum": ["cornered", "circle"]},
            "corners": {"type": "array", "items": corner},
            "surface": {"type": "string", "enum": ["asphalt", "clay", "ice"]},
            "grip": {"type": "number"},
            "track_width": {"type": "number"},
            "edge_barriers": {"type": "boolean"},
            "laps": {"type": "integer"},
            "start_region": {"type": "string", "enum": [item.value for item in TrackRegion]},
            "player_grid_position": {"type": "integer", "minimum": 1, "maximum": 6},
            "barriers": {"type": "array", "items": barrier},
            "npcs": {"type": "array", "items": npc},
            "npc_start_mode": {"type": "string", "enum": ["grid", "distributed"]},
        },
        "required": [
            "title", "rationale", "direction", "loop_shape", "corners", "surface", "grip",
            "track_width", "edge_barriers", "laps", "start_region", "player_grid_position",
            "barriers", "npcs", "npc_start_mode",
        ],
        "additionalProperties": False,
    }


_UNSUPPORTED_REQUESTS: tuple[tuple[tuple[str, ...], str], ...] = (
    # Colour used to be listed here. It is a real dial now — `VisualPlan` carries the
    # road, ground, barrier, car, and sky palettes plus coloured ground bands — so
    # reporting a repaint as impossible would be telling the user we cannot do
    # something the harness verifies it did.
    (("lighting", "headlight", "shadow", "sunbeam", "god ray"),
     "lighting and shadows — the sky can be recoloured, but nothing casts light"),
    (("pit", "pitlane", "pit-stop", "refuel", "fuel", "tyre change", "tire change", "damage"),
     "pit stops, fuel, and damage"),
    (("jump", "ramp", "tunnel", "bridge", "loop-the-loop", "vertical loop", "chicane wall"),
     "jumps, tunnels, and bridges — the road is a surface with height, not a volume"),
    (("crowd", "spectator", "grandstand", "music", "sound", "commentary", "weather change"),
     "spectators, audio, and changing weather"),
    (("rain during", "gets wetter", "dries out", "changing grip", "degrading"),
     "conditions that change during a race — grip is fixed for the whole run"),
)


def unsupported_requests(prompt: str) -> list[str]:
    """Parts of a brief the corner grammar has no field for.

    The compiler always produces a valid circuit, so a request for something outside the
    grammar does not fail — it silently produces a circuit that ignores that part and gets
    named after it. From the outside that is indistinguishable from the prompt being thrown
    away at random, which is exactly how it was reported. Naming the gap is cheap and it is
    the difference between "this harness is random" and "this harness does not do colours".

    Deliberately conservative: only phrases with no corresponding field at all. Anything the
    grammar does read — surface, slipperiness, corner angles and regions, corridor width, laps,
    barriers, opponent counts and temperaments, direction, elevation — is not listed here.
    """
    text = " ".join(prompt.lower().split())
    found: list[str] = []
    for words, description in _UNSUPPORTED_REQUESTS:
        if any(word in text for word in words) and description not in found:
            found.append(description)
    return found


_WORD_NUMBERS = {
    "no": 0, "zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_REGION_PHRASES: tuple[tuple[str, TrackRegion], ...] = (
    ("top left", TrackRegion.TOP_LEFT), ("upper left", TrackRegion.TOP_LEFT),
    ("north west", TrackRegion.TOP_LEFT), ("northwest", TrackRegion.TOP_LEFT),
    ("top right", TrackRegion.TOP_RIGHT), ("upper right", TrackRegion.TOP_RIGHT),
    ("north east", TrackRegion.TOP_RIGHT), ("northeast", TrackRegion.TOP_RIGHT),
    ("bottom left", TrackRegion.BOTTOM_LEFT), ("lower left", TrackRegion.BOTTOM_LEFT),
    ("south west", TrackRegion.BOTTOM_LEFT), ("southwest", TrackRegion.BOTTOM_LEFT),
    ("bottom right", TrackRegion.BOTTOM_RIGHT), ("lower right", TrackRegion.BOTTOM_RIGHT),
    ("south east", TrackRegion.BOTTOM_RIGHT), ("southeast", TrackRegion.BOTTOM_RIGHT),
    ("top center", TrackRegion.TOP_CENTER), ("top centre", TrackRegion.TOP_CENTER),
    ("top middle", TrackRegion.TOP_CENTER), ("north", TrackRegion.TOP_CENTER),
    ("bottom center", TrackRegion.BOTTOM_CENTER), ("bottom centre", TrackRegion.BOTTOM_CENTER),
    ("bottom middle", TrackRegion.BOTTOM_CENTER), ("south", TrackRegion.BOTTOM_CENTER),
    ("left side", TrackRegion.LEFT), ("west", TrackRegion.LEFT),
    ("right side", TrackRegion.RIGHT), ("east", TrackRegion.RIGHT),
)
_CORNER_NOUNS = (
    "hairpin", "chicane", "bend", "corner", "turn", "kink", "sweeper", "switchback",
)


def parse_track_prompt(prompt: str) -> TrackPlan:
    """Deterministically read a natural-language brief into a typed plan.

    This is the offline generator and the behavior floor for the model-backed
    creator: every phrase it understands is a phrase the harness can satisfy
    without an API key, which keeps prompt fidelity testable.
    """
    text = " ".join(prompt.lower().split())
    surface, grip, surface_note = _parse_surface(text)
    direction = "clockwise" if "clockwise" in text and not any(
        phrase in text for phrase in ("counterclockwise", "counter-clockwise", "anticlockwise")
    ) else "counterclockwise"
    circle = any(phrase in text for phrase in (
        "circle track", "circular track", "circle circuit", "circular circuit",
        "circular loop", "round loop", "legitimate circle", "no corners",
    ))
    corners = [] if circle else _parse_corners(text, direction)
    npcs = _parse_npcs(text)
    edge_barriers = any(phrase in text for phrase in (
        "edge barrier", "edge barriers", "guardrail", "guardrails",
        "safety wall", "safety walls", "barriers around", "barriers along the edge",
        "barriers along both edges", "walled circuit",
    ))
    # Edge-wall language must not also create a stray bollard. Strip only the
    # phrases that describe guardrails before the independent obstacle parser;
    # an explicit "and two obstacles" clause remains intact.
    discrete_barrier_text = re.sub(
        r"\b(?:edge barriers?|guardrails?|safety walls?|barriers? (?:around|along (?:the|both) (?:road )?edges?))\b",
        "", text,
    )
    barriers = [
        BarrierSpec(region=region, label=f"track barrier {index + 1}")
        for index, region in enumerate(_parse_barrier_regions(discrete_barrier_text))
    ]
    laps = max(1, min(10, _parse_count(text, ("lap",), default=1) or 1))
    start_region = _parse_start_region(text)
    player_grid_position = _parse_player_grid_position(text)
    width = 152.0 if any(word in text for word in ("wide", "broad")) else 118.0 if any(
        word in text for word in ("narrow", "tight track", "claustrophobic")
    ) else 132.0
    descriptor = ", ".join(filter(None, (
        surface_note,
        "true circular centerline" if circle else f"{len(corners)} authored corner(s)",
        f"{len(npcs)} opponent(s)" if npcs else None,
        f"{len(barriers)} barrier(s)" if barriers else None,
        "continuous edge barriers" if edge_barriers else None,
    )))
    return TrackPlan(
        title=_parse_title(text, surface),
        rationale=f"Compiled from the brief: {descriptor}.",
        direction=direction, loop_shape="circle" if circle else "cornered", corners=corners,
        surface=surface, grip=grip,
        track_width=width, edge_barriers=edge_barriers, laps=laps,
        start_region=start_region, player_grid_position=player_grid_position,
        barriers=barriers, npcs=npcs,
        npc_start_mode="distributed" if any(
            phrase in text for phrase in ("distributed", "spread out", "rolling start", "spread around")
        ) else "grid",
    )


def _parse_start_region(text: str) -> TrackRegion:
    """Read an explicitly located start/finish line without borrowing a corner's region."""
    match = re.search(
        r"(?:start(?:ing)?(?:(?:\s*(?:/|&|-)\s*|\s+)finish)?|finish(?:\s+line)?|grid)\s*"
        r"(?:line|area|position)?\s*(?:at|in|on|near)?\s*(?:the\s+)?"
        r"(top left|upper left|north west|northwest|top right|upper right|north east|northeast|"
        r"bottom left|lower left|south west|southwest|bottom right|lower right|south east|southeast|"
        r"top center|top centre|top middle|bottom center|bottom centre|bottom middle|left side|right side|west|east|north|south)",
        text,
    )
    if not match:
        return TrackRegion.AUTO
    phrase = match.group(1)
    return next((region for words, region in _REGION_PHRASES if words == phrase), TrackRegion.AUTO)


def _parse_player_grid_position(text: str) -> int:
    """Read natural grid language such as ``start P4`` or ``player in 3rd``."""
    match = re.search(
        r"(?:player\s*(?:starts?|at|in)|start\s*(?:from|at)?|grid\s*(?:position|slot)?)\s*"
        r"(?:p|position\s*)?(\d+)(?:st|nd|rd|th)?\b",
        text,
    )
    if match:
        return max(1, min(6, int(match.group(1))))
    ordinal_words = {
        "pole": 1, "first": 1, "second": 2, "third": 3,
        "fourth": 4, "fifth": 5, "sixth": 6,
    }
    for word, position in ordinal_words.items():
        if re.search(rf"(?:player\s*(?:starts?|at|in)|start\s*(?:from|at)?|grid)\s+{word}\b", text):
            return position
    return 1


def _parse_surface(text: str) -> tuple[str, float, str]:
    if any(word in text for word in ("ice", "icy", "snow", "frozen", "glacier")):
        surface, grip = "ice", 1.0
    elif any(word in text for word in ("dirt", "clay", "gravel", "desert", "sand", "rally")):
        surface, grip = "clay", 1.0
    elif any(word in text for word in ("wet", "rain", "damp", "slick", "slippery", "greasy", "oily", "low grip", "low-grip")):
        # Slipperiness on a paved circuit is a grip condition, not a new surface.
        surface, grip = "asphalt", .55
    else:
        surface, grip = "asphalt", 1.0
    if any(word in text for word in ("slippery", "slick", "greasy", "oily", "low grip", "low-grip", "no grip")):
        grip = min(grip, .55)
    if any(word in text for word in ("very slippery", "extremely slippery", "treacherous")):
        grip = min(grip, .42)
    if any(word in text for word in ("high grip", "sticky", "grippy", "maximum grip")):
        grip = 1.15
    note = f"{surface} surface" + ("" if grip == 1.0 else f" at {grip:.2f}x grip")
    return surface, round(grip, 2), note


def _parse_corners(text: str, circuit_direction: str) -> list[CornerSpec]:
    """Extract explicitly described corners, then pad to a complete circuit."""
    circuit_hand = "left" if circuit_direction == "counterclockwise" else "right"
    explicit: list[CornerSpec] = []
    for match in re.finditer(
        r"(?:(\d{2,3})\s*(?:-|\s)?degree[s]?\s+)?"
        r"(hairpin|chicane|sweeper|switchback|bend|corner|turn|kink)"
        r"([^.;]{0,48})",
        text,
    ):
        angle_text, noun, tail = match.group(1), match.group(2), match.group(3) or ""
        leading = text[max(0, match.start() - 40):match.start()]
        context = f"{leading} {noun} {tail}"
        angle = float(angle_text) if angle_text else _default_angle(noun)
        angle = max(MIN_CORNER_ANGLE, min(MAX_CORNER_ANGLE, angle))
        direction = _parse_corner_direction(leading, tail)
        explicit.append(CornerSpec(
            direction=direction, angle_degrees=angle,
            radius=_default_radius(noun, context),
            region=_parse_region(context),
            exit_straight=_default_straight(noun, context),
            label=f"{angle:.0f}-degree {noun}",
        ))
        if noun == "chicane":
            # A chicane is a pair of opposed kinks, so the exit must turn back
            # against the entry even when the brief never stated a handedness.
            entry_hand = direction or circuit_hand
            explicit[-1] = explicit[-1].model_copy(update={"direction": entry_hand})
            explicit.append(CornerSpec(
                direction="right" if entry_hand == "left" else "left",
                angle_degrees=max(MIN_CORNER_ANGLE, angle * .8), radius=CornerRadius.TIGHT,
                region=_parse_region(context), exit_straight=StraightLength.SHORT,
                label="chicane exit",
            ))
        if len(explicit) >= 8:
            break
    corners = explicit[:10]
    target = _target_corner_count(text, len(corners))
    filler_radius, filler_straight = _filler_shape(text)
    while len(corners) < target:
        corners.append(CornerSpec(
            radius=filler_radius, exit_straight=filler_straight,
            label=f"sweep {len(corners) + 1}",
        ))
    return corners[:10]


def _target_corner_count(text: str, explicit: int) -> int:
    if any(word in text for word in ("very curvy", "very twisty", "serpentine", "labyrinth")):
        target = 9
    elif any(word in text for word in ("curvy", "twisty", "technical", "winding", "sinuous")):
        target = 7
    elif any(word in text for word in ("oval", "fast", "flowing", "high speed", "high-speed", "simple")):
        target = 4
    else:
        target = 5
    requested = _parse_count(text, _CORNER_NOUNS, default=0)
    if requested >= 3:
        target = max(target, min(10, requested))
    return max(3, max(explicit, target))


def _filler_shape(text: str) -> tuple[CornerRadius, StraightLength]:
    if any(word in text for word in ("tight", "technical", "twisty", "narrow")):
        return CornerRadius.TIGHT, StraightLength.SHORT
    if any(word in text for word in ("fast", "flowing", "high speed", "high-speed", "oval", "sweeping")):
        return CornerRadius.SWEEPING, StraightLength.LONG
    return CornerRadius.MEDIUM, StraightLength.MEDIUM


def _default_angle(noun: str) -> float:
    return {
        "hairpin": 165.0, "switchback": 150.0, "chicane": 45.0, "kink": 30.0,
        "sweeper": 55.0, "bend": 75.0, "corner": 85.0, "turn": 85.0,
    }.get(noun, 85.0)


def _default_radius(noun: str, context: str) -> CornerRadius:
    if noun in {"hairpin", "switchback"} or "tight" in context or "sharp" in context:
        return CornerRadius.HAIRPIN if noun in {"hairpin", "switchback"} else CornerRadius.TIGHT
    if noun in {"sweeper", "kink"} or any(word in context for word in ("fast", "sweeping", "open", "gentle")):
        return CornerRadius.SWEEPING if noun == "sweeper" else CornerRadius.OPEN
    return CornerRadius.MEDIUM


def _default_straight(noun: str, context: str) -> StraightLength:
    if "long straight" in context or "back straight" in context or "main straight" in context:
        return StraightLength.LONG
    if noun in {"chicane", "kink"} or "short straight" in context:
        return StraightLength.SHORT
    return StraightLength.MEDIUM


def _parse_corner_direction(leading: str, tail: str) -> Literal["left", "right"] | None:
    """Read turn handedness without mistaking a screen region for a direction.

    "a 90 degree bend in the top right" locates the corner; it does not say
    which way it turns. Region phrases are removed before looking for a turn
    word, so an unstated handedness correctly falls back to the circuit
    direction instead of being invented from the region.
    """
    def strip_regions(fragment: str) -> str:
        for phrase, _ in _REGION_PHRASES:
            fragment = fragment.replace(phrase, " ")
        return re.sub(r"\b(?:on|in|at|toward|towards)\s+the\s+(?:left|right)\b", " ", fragment)

    before, after = strip_regions(leading), strip_regions(tail)
    for fragment in (before, after):
        has_left = re.search(r"\bleft\b|\blefthand\b|\bleft-hand\b", fragment) is not None
        has_right = re.search(r"\bright\b|\brighthand\b|\bright-hand\b", fragment) is not None
        if has_left and not has_right:
            return "left"
        if has_right and not has_left:
            return "right"
    return None


def _parse_region(context: str) -> TrackRegion:
    for phrase, region in _REGION_PHRASES:
        if phrase in context:
            return region
    return TrackRegion.AUTO


def _parse_npcs(text: str) -> list[NpcSpec]:
    """Read opponent count and temperament from the brief."""
    profiles: list[NpcProfile] = []
    adjectives: tuple[tuple[tuple[str, ...], NpcProfile], ...] = (
        (("aggressive", "ruthless", "hostile", "attacking", "pushy"), NpcProfile.AGGRESSOR),
        (("blocking", "defensive", "obstructive", "blocker"), NpcProfile.BLOCKER),
        (("slow", "backmarker", "timid", "cautious", "gentle"), NpcProfile.BACKMARKER),
        (("cruising", "casual", "relaxed", "steady"), NpcProfile.CRUISER),
    )
    total = _parse_count(text, ("npc", "opponent", "rival", "car", "traffic", "ai", "bot"), default=1)
    for words, profile in adjectives:
        for word in words:
            if word not in text:
                continue
            count = _adjective_count(text, word)
            profiles.extend([profile] * count)
            break
    if not profiles:
        profiles = [NpcProfile.RACER] * total
    elif total > len(profiles):
        profiles.extend([NpcProfile.RACER] * (total - len(profiles)))
    profiles = profiles[:5]
    return [
        NpcSpec(profile=profile, label=f"{profile.value} {index + 1}")
        for index, profile in enumerate(profiles)
    ]


def _adjective_count(text: str, adjective: str) -> int:
    """How many opponents an adjective governs, e.g. 'three aggressive rivals'.

    The temperament word can be the noun itself, as in "two blockers". Without
    the optional plural the count failed to bind to it and every such brief
    silently compiled a single opponent.
    """
    match = re.search(rf"(\w+)\s+{adjective}s?\b", text)
    if match:
        word = match.group(1)
        if word.isdigit():
            return max(1, min(5, int(word)))
        if word in _WORD_NUMBERS:
            return max(1, min(5, _WORD_NUMBERS[word]))
    following = re.search(rf"{adjective}s?\s+(\w+)", text)
    if following and following.group(1).rstrip("s") in {"npc", "opponent", "rival", "car", "bot", "driver"}:
        return 1
    return 1


def _parse_barrier_regions(text: str) -> list[TrackRegion]:
    count = _parse_count(text, ("obstacle", "barrier", "cone", "block"), default=0)
    if not count:
        return []
    regions: list[TrackRegion] = []
    for phrase, region in _REGION_PHRASES:
        if re.search(rf"{phrase}[^.;]{{0,24}}\b(barrier|obstacle|cone)", text) or re.search(
            rf"\b(barrier|obstacle|cone)[^.;]{{0,24}}{phrase}", text,
        ):
            if region not in regions:
                regions.append(region)
    while len(regions) < count:
        regions.append(TrackRegion.AUTO)
    return regions[:6]


def _parse_count(text: str, nouns: tuple[str, ...], default: int) -> int:
    """Read '3 barriers', 'three barriers', or 'no barriers' from the brief."""
    for noun in nouns:
        if re.search(rf"\bno\s+(?:\w+\s+){{0,2}}{noun}", text):
            return 0
    for noun in nouns:
        match = re.search(rf"(\d+|{'|'.join(_WORD_NUMBERS)})\s+(?:\w+\s+){{0,2}}{noun}s?\b", text)
        if match:
            token = match.group(1)
            return int(token) if token.isdigit() else _WORD_NUMBERS[token]
    for noun in nouns:
        if re.search(rf"\b{noun}s?\b", text):
            return max(default, 1)
    return default


def _parse_title(text: str, surface: str) -> str:
    if "hairpin" in text:
        base = "Hairpin Pass"
    elif "chicane" in text:
        base = "Orange Gate Raceway"
    elif any(word in text for word in ("curvy", "twisty", "technical", "winding")):
        base = "Switchback Circuit"
    elif any(word in text for word in ("oval", "fast", "flowing")):
        base = "North Loop"
    else:
        base = "Commissioned Circuit"
    prefix = {"ice": "Frozen ", "clay": "Dust ", "asphalt": ""}[surface]
    return f"{prefix}{base}"[:64]

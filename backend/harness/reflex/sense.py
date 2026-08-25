"""Normalized local channels: everything a controller or a wake condition may read.

Every field is scaled by something the car can measure about itself or the road under
it, because that is what makes a generated controller portable. `LANE_GAIN = 0.15` in
`lowlevel.py` is not really a gain — it is a gain for one corridor width, one control
rate, and one tire model, which is why it had to be tuned against a baseline and why the
tuning does not survive a change of surface. A controller written against `lane`
(corridor-relative), `speed` (car lengths per second), and `grip_used` (1.0 is the
friction limit on any surface) has a chance of working on the next generated circuit.

Two deliberate omissions. There is no lap progress and no corner map: this is the local
channel set, the same privilege boundary `LOCAL_OBSERVATION_FIELDS` draws in
`lowlevel.py`, so a controller cannot steer by geometry it has not reached. And there is
no reward — the runtime does not tell the controller how it is scored.

`FIELDS` is the single source of truth. The install gate checks declared reads against
it and the agent's system prompt is generated from it, so the catalog cannot drift out
of sync with what a controller can actually read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..models import ObservationPacket, SceneSpec, Vec2


NO_HAZARD_SECONDS = 99.0
"""What `ttc` reads when nothing solid is ahead. Finite, so comparisons stay total."""

MAX_FREE_AHEAD_CARLENGTHS = 12.0
"""How far the corridor ray-march looks before giving up. Beyond this, `free_ahead`
saturates: a controller only needs to know the road ahead is open, not how open."""

CORRIDOR_SEARCH_WINDOW = 24
"""Centerline indices either side of the tracked one considered for distance queries.

Bounded rather than global for the same reason `Racing3DWorld` bounds its surface
lookup: two arms of a chicane pass close together, and a global nearest-point query is
free to jump between them and flip the sign of everything local.
"""

MARCH_WINDOW = 6
"""Indices considered per step of the `free_ahead` ray-march.

The probe advances about a quarter of a car length per step, which is one or two
centerline samples, so a narrow window is enough — and it keeps the whole march to a few
hundred distance computations per tick.
"""


FIELDS: dict[str, tuple[str, str]] = {
    "lane": ("[-1,1]", "lateral position; -1 left corridor edge, +1 right edge, 0 centre. Steering right increases it"),
    "lane_rate": ("1/s", "rate of change of lane; positive is drifting right"),
    "heading_error": ("deg", "degrees you must turn RIGHT to align with the road; negative means turn left. Steering right decreases it"),
    "speed": ("cl/s", "forward speed in car lengths per second"),
    "speed_limit": ("cl/s", "local physics-derived speed envelope from curvature, braking distance, grip, and the car's maximum speed"),
    "curvature": ("1/cl", "signed road curvature at the car; positive means the road turns right"),
    "grip_used": ("frac", "how much of the tires' friction circle is in use; 1.0 is the limit on any surface, and past it the car slides"),
    "grip_headroom": ("frac", "1 - grip_used, floored at 0"),
    "free_ahead": ("cl", f"drivable car lengths straight ahead before the corridor edge, saturating at {MAX_FREE_AHEAD_CARLENGTHS:.0f}"),
    "ttc": ("s", f"seconds to contact with the nearest solid thing ahead; {NO_HAZARD_SECONDS:.0f} when clear"),
    "hazard_bearing": ("deg", "bearing of that hazard relative to the car; 0 is dead ahead"),
    "half_width": ("cl", "safe corridor half-width in car lengths"),
    "on_track": ("bool", "inside the safe corridor"),
    "slip": ("deg", "body slip angle; large magnitude means the car is sliding"),
    "yaw_rate": ("deg/s", "rate of rotation; positive is rotating right"),
    "steer_angle": ("[-1,1]", "actual front wheel angle over its maximum; positive is right"),
    "grade": ("deg", "uphill road grade; always 0 in 2D"),
    "bank": ("deg", "road cross-slope; always 0 in 2D"),
    "nitro_ready": ("bool", "tank full, so boost may be requested"),
    "target_lane": ("[-1,1]", "lane the active target sits at, on the same scale as lane"),
    "target_error": ("cl", "distance from the car to the active target"),
    "target_reached": ("bool", "the active target has been reached or passed"),
    "tick": ("ticks", "control ticks since the episode began"),
    "vision_lane": ("[-1,1]", "pixel-derived road-centre offset in the forward cone; positive means road centre is right"),
    "vision_turn": ("[-1,1]", "pixel-derived road bend in the forward cone; positive means the visible road bends right"),
    "vision_flow": ("px/tick", "apparent road motion measured by optical flow between the last two cone screenshots"),
    "vision_road_visible": ("bool", "a road segment was detected ahead in the cone screenshot"),
    "vision_center_near": ("[-1,1]", "road-centre offset from pixels near the ego anchor; positive means road is right"),
    "vision_center_mid": ("[-1,1]", "road-centre offset from pixels halfway up the cone"),
    "vision_center_far": ("[-1,1]", "road-centre offset from pixels far up the cone"),
    "vision_turn_ahead": ("[-2,2]", "signed pixel-derived change in road-centre direction from near to far; positive means the visible road bends right"),
    "vision_turn_severity": ("[0,2]", "pixel-derived magnitude of the road bend across the cone; high means commit to a tight turn early"),
    "vision_lookahead_depth": ("[0,1]", "fraction of the forward cone in which a plausible road corridor remains visible"),
    "vision_road_width": ("frac", "visible near-road width divided by image width"),
    "vision_left_gap": ("frac", "pixel-derived distance from ego column to the left road edge"),
    "vision_right_gap": ("frac", "pixel-derived distance from ego column to the right road edge"),
    "vision_confidence": ("[0,1]", "fraction of near, middle, and far cone scans that found a plausible road segment"),
    "vision_road_lost": ("bool", "no reliable road corridor was detected in the cone screenshot"),
    "vision_flow_rotation": ("px/tick", "mean horizontal optical-flow component on detected road pixels; signed image-right"),
    "vision_ego_road_contact": ("bool", "pixels immediately ahead of the ego anchor still match the road"),
    "vision_recovery_direction": ("[-1,1]", "pixel-derived direction to the closest visible road when ego contact is lost"),
    "vision_center_rate": ("frac/tick", "change in near road-centre offset between cone frames"),
    "vision_turn_delta": ("frac/tick", "change in visible road bend between cone frames"),
    "vision_edge_closing_rate": ("frac/tick", "change in the smaller visible edge gap; negative means an edge is closing"),
    "vision_confidence_trend": ("frac/tick", "change in road detection confidence; negative means the visual corridor is degrading"),
    # Perspective-only visual contract. These are not populated by the 2D cone arm and
    # are never available unless ReflexRuntime selected an elevated 3D scene.
    "vision_track_offset": ("[-1,1]", "pixel-derived lateral road-centre offset near the first-person camera; positive means visible road is right"),
    "vision_track_heading": ("[-2,2]", "pixel-derived near-to-middle visible road direction in the perspective image; positive means it leads right"),
    "vision_bend_ahead": ("[-2,2]", "signed pixel-derived visible bend across perspective image depth; positive means right"),
    "vision_bend_severity": ("[0,2]", "pixel-derived magnitude of the visible perspective bend; use with image speed to begin braking earlier"),
    "vision_visible_depth": ("[0,1]", "fraction of the first-person image containing a plausible road corridor"),
    "vision_road_contact": ("bool", "pixels immediately ahead of the camera still match the detected road"),
    "vision_road_horizon": ("[0,1] image y", "highest plausible road row in the camera image; a visual road-plane cue, not world elevation"),
    "vision_horizon_shift": ("image frac/tick", "change in the visible road horizon between screenshots; a camera-derived crest or dip cue"),
    "vision_crest_risk": ("[0,1]", "camera-derived risk that the road disappears behind a crest or visual occlusion; higher means slow before committing"),
}


def catalog_text(fields=None) -> str:
    """The field list as the agent's system prompt shows it."""
    fields = tuple(fields or FIELDS)
    width = max(len(name) for name in fields)
    return "\n".join(
        f"  sense.{name:<{width}}  {unit:<8} {description}"
        for name in fields for unit, description in [FIELDS[name]]
    )


class SenseView:
    """Attribute access to exactly the fields a controller declared, and no others.

    The install gate already rejects an undeclared field statically, so this is the
    backstop rather than the primary check. It exists because a silent `None` at tick
    400 is the worst possible failure and an explicit error is the cheapest fix.
    """

    __slots__ = ("_values", "_allowed")

    def __init__(self, values: dict[str, float], allowed: frozenset[str] | None = None):
        self._values = values
        self._allowed = allowed

    def __getattr__(self, name: str):
        if name not in FIELDS:
            raise AttributeError(
                f"sense.{name} is not a field; available fields are {sorted(FIELDS)}"
            )
        if self._allowed is not None and name not in self._allowed:
            raise AttributeError(
                f"sense.{name} was not declared in this controller's reads "
                f"(declared: {sorted(self._allowed)})"
            )
        return self._values[name]

    def as_dict(self) -> dict[str, float]:
        return dict(self._values)


@dataclass
class Target:
    """Where the agent wants the car to go, anchored where it was set.

    Two kinds, because the useful distinction is whether the target can be *reached*.
    `hold_lane` never completes, which is what an agent wants on a long straight;
    `lane_point` completes, which is what wakes the agent at the end of a corner.

    A point target is anchored to the centerline index the car occupied when the target
    was set, so "eight car lengths ahead, on the inside" stays put as the car drives to
    it instead of receding forever.
    """

    kind: str = "hold_lane"
    lane: float = 0.0
    ahead_cl: float = 8.0
    tolerance_cl: float = 1.5
    anchor_index: int | None = None
    target_index: int | None = None
    note: str = ""

    def as_dict(self) -> dict:
        payload = {"kind": self.kind, "lane": round(self.lane, 3)}
        if self.kind == "lane_point":
            payload.update({
                "ahead_cl": round(self.ahead_cl, 2),
                "tolerance_cl": round(self.tolerance_cl, 2),
            })
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class SenseMemory:
    """The little state channel computation needs across ticks.

    Owned by the runtime, snapshotted with it, and reset on a fork, so a rehearsal
    starts from the same continuity state the live episode had.
    """

    track_index: int | None = None
    previous_lane: float | None = None
    tick: int = 0
    curvature_fast: float = 0.0
    curvature_slow: float = 0.0
    curvature_var: float = 0.0
    geometry_samples: int = 0
    last_geometry_wake: int = -(10**6)
    history: list[dict] = field(default_factory=list)


def compute_sense(
    world, observation: ObservationPacket, memory: SenseMemory, target: Target | None,
) -> dict[str, float]:
    """One channel row. Called once per tick, and once per rehearsed tick."""
    scene: SceneSpec = world.scene
    points = scene.track_centerline
    dynamics = observation.dynamics
    px_per_cl = _pixels_per_carlength(dynamics)
    control_hz = dynamics.control_hz
    position = observation.proprioception

    index = _tracked_index(points, position, memory.track_index)
    memory.track_index = index
    current, before, after = points[index], points[(index - 1) % len(points)], points[(index + 1) % len(points)]
    tangent_x, tangent_y = after.x - before.x, after.y - before.y
    tangent_length = max(1.0, math.hypot(tangent_x, tangent_y))
    lateral_px = (
        (position.x - current.x) * (-tangent_y / tangent_length)
        + (position.y - current.y) * (tangent_x / tangent_length)
    )
    safe_half_px = max(1.0, scene.track_width / 2 - _car_radius())
    lane = _clamp(lateral_px / safe_half_px, -2.0, 2.0)
    lane_rate = 0.0 if memory.previous_lane is None else (lane - memory.previous_lane) * control_hz
    memory.previous_lane = lane

    road_heading = _bearing(before, after)
    heading_error = _angle_delta(observation.heading, road_heading)
    speed_cl_s = observation.speed * control_hz / px_per_cl
    curvature = _curvature(points, index, px_per_cl)
    grade_radians, bank_radians = _attitude(world, position)

    grip_used = _grip_used(
        dynamics, observation, grade_radians, bank_radians, abs(lane) <= 1.0,
    )
    hazard_seconds, hazard_bearing = _time_to_contact(observation, control_hz)
    free_ahead = _free_ahead(
        points, position, observation.heading, index, safe_half_px, px_per_cl,
    )
    speed_limit = _speed_limit(
        dynamics, curvature=curvature, free_ahead_cl=free_ahead, px_per_cl=px_per_cl,
    )
    target_lane, target_error, target_reached = _target_state(
        target, points, index, position, px_per_cl,
    )

    memory.tick += 1
    return {
        "lane": round(lane, 4),
        "lane_rate": round(lane_rate, 4),
        "heading_error": round(heading_error, 2),
        "speed": round(speed_cl_s, 4),
        "speed_limit": round(speed_limit, 4),
        "curvature": round(curvature, 5),
        "grip_used": round(grip_used, 4),
        "grip_headroom": round(max(0.0, 1.0 - grip_used), 4),
        "free_ahead": round(free_ahead, 3),
        "ttc": round(hazard_seconds, 3),
        "hazard_bearing": round(hazard_bearing, 1),
        "half_width": round(safe_half_px / px_per_cl, 3),
        "on_track": abs(lane) <= 1.0,
        "slip": observation.slip_angle_degrees,
        "yaw_rate": observation.yaw_rate_degrees_per_second,
        "steer_angle": round(
            observation.steering_angle_degrees
            / max(1e-6, dynamics.vehicle.max_steering_angle_degrees), 4,
        ),
        "grade": round(math.degrees(grade_radians), 2),
        "bank": round(math.degrees(bank_radians), 2),
        "nitro_ready": bool(observation.nitro_ready),
        "target_lane": round(target_lane, 3),
        "target_error": round(target_error, 3),
        "target_reached": target_reached,
        "tick": memory.tick,
        "vision_lane": 0.0,
        "vision_turn": 0.0,
        "vision_flow": 0.0,
        "vision_road_visible": False,
        "vision_center_near": 0.0, "vision_center_mid": 0.0, "vision_center_far": 0.0,
        "vision_turn_ahead": 0.0, "vision_turn_severity": 0.0, "vision_lookahead_depth": 0.0,
        "vision_road_width": 0.0, "vision_left_gap": 0.0, "vision_right_gap": 0.0,
        "vision_confidence": 0.0, "vision_road_lost": True, "vision_flow_rotation": 0.0,
        "vision_ego_road_contact": False, "vision_recovery_direction": 0.0, "vision_center_rate": 0.0,
        "vision_turn_delta": 0.0, "vision_edge_closing_rate": 0.0, "vision_confidence_trend": 0.0,
        "vision_track_offset": 0.0, "vision_track_heading": 0.0,
        "vision_bend_ahead": 0.0, "vision_bend_severity": 0.0,
        "vision_visible_depth": 0.0, "vision_road_contact": False,
        "vision_road_horizon": 1.0,
        "vision_horizon_shift": 0.0, "vision_crest_risk": 1.0,
    }


def anchor_target(target: Target, world, observation: ObservationPacket, memory: SenseMemory) -> Target:
    """Pin a point target to the centerline index the car occupies right now."""
    if target.kind != "lane_point":
        return target
    points = world.scene.track_centerline
    px_per_cl = _pixels_per_carlength(observation.dynamics)
    index = _tracked_index(points, observation.proprioception, memory.track_index)
    spacing = _spacing(points)
    steps = max(1, round(target.ahead_cl * px_per_cl / spacing))
    target.anchor_index = index
    target.target_index = (index + steps) % len(points)
    return target


def _target_state(
    target: Target | None, points: list[Vec2], index: int, position: Vec2, px_per_cl: float,
) -> tuple[float, float, bool]:
    if target is None:
        return 0.0, 0.0, False
    if target.kind != "lane_point" or target.target_index is None:
        return target.lane, 0.0, False
    point = points[target.target_index]
    distance_cl = math.hypot(point.x - position.x, point.y - position.y) / px_per_cl
    passed = _cyclic_delta(index, target.target_index, len(points)) <= 0
    return target.lane, distance_cl, bool(passed or distance_cl <= target.tolerance_cl)


def _curvature(points: list[Vec2], index: int, px_per_cl: float) -> float:
    """Signed curvature at the car, in inverse car lengths.

    Measured over roughly a car length either side rather than between adjacent
    samples, because the compiled centerline is finely sampled and an adjacent-point
    estimate is mostly quantization noise.
    """
    spacing = _spacing(points)
    step = max(1, round(px_per_cl / max(1e-6, spacing)))
    count = len(points)
    behind = points[(index - step) % count]
    ahead = points[(index + step) % count]
    entry = _bearing(behind, points[index])
    exit_heading = _bearing(points[index], ahead)
    turn_degrees = _angle_delta(entry, exit_heading)
    arc_px = math.hypot(points[index].x - behind.x, points[index].y - behind.y) + math.hypot(
        ahead.x - points[index].x, ahead.y - points[index].y,
    )
    if arc_px <= 1e-6:
        return 0.0
    return math.radians(turn_degrees) / (arc_px / px_per_cl)


def _free_ahead(
    points: list[Vec2], position: Vec2, heading: float, index: int,
    safe_half_px: float, px_per_cl: float,
) -> float:
    """March along the car's heading until it would leave the corridor.

    This is the channel that tells a controller a corner is coming without handing it
    the corner: it is the same information a forward camera carries, measured instead
    of inferred.
    """
    step_px = px_per_cl * 0.25
    radians = math.radians(heading)
    step_x, step_y = math.cos(radians) * step_px, math.sin(radians) * step_px
    x, y = position.x, position.y
    travelled = 0.0
    limit = MAX_FREE_AHEAD_CARLENGTHS * px_per_cl
    hint = index
    while travelled < limit:
        x, y = x + step_x, y + step_y
        travelled += step_px
        probe = Vec2(x=x, y=y)
        # The index hint has to travel with the probe. Measuring every probe against a
        # window fixed at the car makes the ray appear to leave the corridor as soon as it
        # outruns the window, which reads as a wall a fixed distance ahead everywhere.
        hint = _local_index(points, probe, hint, MARCH_WINDOW)
        if _windowed_distance(points, probe, hint, MARCH_WINDOW) > safe_half_px:
            break
    return min(MAX_FREE_AHEAD_CARLENGTHS, travelled / px_per_cl)


def _time_to_contact(observation: ObservationPacket, control_hz: int) -> tuple[float, float]:
    """Seconds to the nearest solid thing ahead, from public nearby-entity telemetry.

    The closing speed used is the car's own speed projected at the hazard, which
    over-estimates time to contact against an opponent closing head-on and
    under-estimates it against one driving away. It is a proximity alarm rather than a
    prediction, and a controller should treat it as one.
    """
    nearest_seconds, nearest_bearing = NO_HAZARD_SECONDS, 0.0
    for entity in observation.local_entities:
        if entity.get("kind") not in {"obstacle", "npc"}:
            continue
        bearing = float(entity.get("bearing", 0.0))
        if abs(bearing) > 70:
            continue
        gap_px = max(0.0, float(entity.get("distance", 0.0)) - _car_radius())
        closing = observation.speed * math.cos(math.radians(bearing))
        if closing <= 0.05:
            continue
        seconds = gap_px / closing / control_hz
        if seconds < nearest_seconds:
            nearest_seconds, nearest_bearing = seconds, bearing
    return min(nearest_seconds, NO_HAZARD_SECONDS), nearest_bearing


def _speed_limit(dynamics, *, curvature: float, free_ahead_cl: float, px_per_cl: float) -> float:
    """A local pace envelope, not an oracle racing line.

    It uses only the road under the car plus the same forward corridor ray used by
    ``free_ahead``.  The controller still owns the aggression trade-off; this saves it
    from rediscovering vehicle units and braking-distance arithmetic on every circuit.
    """
    vehicle, road = dynamics.vehicle, dynamics.road
    car_length_m = px_per_cl / max(1e-6, dynamics.pixels_per_meter)
    max_speed = vehicle.max_speed_mps / max(1e-6, car_length_m)
    normal = dynamics.gravity_mps2
    lateral_accel = max(
        0.1,
        road.friction_coefficient * vehicle.tire_friction_multiplier
        * road.lateral_grip_multiplier * normal,
    )
    # v²/R, represented in car-length units.  Straight segments retain the
    # vehicle cap; tight local curvature lowers the envelope immediately.
    curve_limit = math.sqrt(
        lateral_accel / max(1e-6, car_length_m * abs(curvature))
    )
    brake_accel = max(
        0.1,
        min(
            vehicle.brake_force_n / vehicle.mass_kg,
            road.friction_coefficient * vehicle.tire_friction_multiplier * normal,
        ),
    )
    distance_m = max(0.25, free_ahead_cl) * car_length_m
    braking_limit = math.sqrt(2.0 * brake_accel * distance_m) / max(1e-6, car_length_m)
    return min(max_speed, curve_limit, braking_limit)


def _attitude(world, position: Vec2) -> tuple[float, float]:
    attitude = getattr(world, "road_attitude", None)
    if attitude is None:
        return 0.0, 0.0
    grade, bank = attitude(position)
    return float(grade), float(bank)


def _grip_used(dynamics, observation, grade_radians: float, bank_radians: float, on_track: bool) -> float:
    """Fraction of the tires' friction circle in use, normalized the way the engine is.

    Each axis is divided by its own limit and then combined, rather than dividing the
    total acceleration by one scalar. That matters because the lateral and longitudinal
    limits genuinely differ: `lateral_mu` carries `lateral_grip_multiplier`, which is 1.0
    on asphalt but 0.78 on clay and 0.68 on ice. A single-scalar version of this channel
    read 1.5 during clean lapping, which makes the one channel that is supposed to mean
    "at the limit on any surface" mean nothing at all.

    Downforce is included, since it is a pure function of speed and the aero package and is
    exactly what makes the limit speed-dependent. Load sensitivity and the banking gain are
    not, so at high lateral load this reads slightly low and on a banked corner slightly
    high. It approximates the engine's friction circle; it is not a second copy of it.
    """
    vehicle, road = dynamics.vehicle, dynamics.road
    speed_mps = math.hypot(observation.longitudinal_speed_mps, observation.lateral_speed_mps)
    dynamic_pressure = 0.5 * dynamics.air_density_kg_m3 * speed_mps * speed_mps
    downforce = max(0.0, -dynamic_pressure * vehicle.lift_coefficient * vehicle.frontal_area_m2)
    normal_per_mass = (
        dynamics.gravity_mps2 * math.cos(grade_radians) * math.cos(bank_radians)
        + downforce / vehicle.mass_kg
    )
    road_mu = road.friction_coefficient if on_track else road.off_track_friction_coefficient
    longitudinal_limit = max(0.1, road_mu * vehicle.tire_friction_multiplier * normal_per_mass)
    lateral_limit = max(0.1, longitudinal_limit * road.lateral_grip_multiplier)
    return min(4.0, math.hypot(
        observation.longitudinal_acceleration_mps2 / longitudinal_limit,
        observation.lateral_acceleration_mps2 / lateral_limit,
    ))


def _pixels_per_carlength(dynamics) -> float:
    return max(1.0, dynamics.vehicle.length_m * dynamics.pixels_per_meter)


def _car_radius() -> float:
    from ..racing import CAR_RADIUS

    return CAR_RADIUS


def _spacing(points: list[Vec2]) -> float:
    return max(1e-6, math.hypot(points[1].x - points[0].x, points[1].y - points[0].y))


def _windowed_distance(points: list[Vec2], point: Vec2, hint: int, window: int) -> float:
    count = len(points)
    best = float("inf")
    for offset in range(-window, window):
        start = points[(hint + offset) % count]
        end = points[(hint + offset + 1) % count]
        best = min(best, _segment_distance(point, start, end))
    return best


def _local_index(points: list[Vec2], point: Vec2, hint: int, window: int) -> int:
    count = len(points)
    return min(
        ((hint + offset) % count for offset in range(-window, window + 1)),
        key=lambda i: math.hypot(points[i].x - point.x, points[i].y - point.y),
    )


def _segment_distance(point: Vec2, start: Vec2, end: Vec2) -> float:
    dx, dy = end.x - start.x, end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(point.x - start.x, point.y - start.y)
    t = max(0.0, min(1.0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared))
    return math.hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy))


def _tracked_index(points: list[Vec2], position: Vec2, hint: int | None) -> int:
    """Nearest centerline index, searched locally when a previous index is known."""
    count = len(points)
    if hint is None:
        return min(range(count), key=lambda i: math.hypot(points[i].x - position.x, points[i].y - position.y))
    candidates = range(hint - CORRIDOR_SEARCH_WINDOW, hint + CORRIDOR_SEARCH_WINDOW + 1)
    return min(
        (i % count for i in candidates),
        key=lambda i: math.hypot(points[i].x - position.x, points[i].y - position.y),
    )


def _cyclic_delta(origin: int, target: int, size: int) -> int:
    forward = (target - origin) % size
    return forward if forward <= size // 2 else forward - size


def _bearing(origin: Vec2, target: Vec2) -> float:
    return math.degrees(math.atan2(target.y - origin.y, target.x - origin.x))


def _angle_delta(current: float, target: float) -> float:
    return (target - current + 180) % 360 - 180


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

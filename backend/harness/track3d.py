"""The vertical half of a circuit: elevation, banking, and surface normals.

The 2D compiler owns the plan view — where the road goes. This module adds the
third dimension on top of that existing centerline rather than replacing it, so
both engines race the same corners, the same barriers, and the same gates.

Two properties make the result safe to simulate:

- **Exact periodicity.** Height is a sum of harmonics of the lap fraction, so
  `height(0)` and `height(1)` are the same number by construction. A circuit
  cannot develop a step at the start/finish seam no matter what parameters are
  chosen, which would otherwise be an invisible cliff the car falls off once per
  lap.
- **Bounded gradients.** Amplitude is expressed in meters and converted through
  the scene's own `pixels_per_meter`, then the compiled profile is checked
  against a maximum drivable grade. A hill the car cannot climb is a rejected
  surface, not a stuck rollout.

Banking is derived from the plan view's curvature: the road leans into its
corners, is flat on straights, and transitions smoothly between them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .models import ElevationProfile, ElevationSpec, SceneSpec, Vec2


# The absolute ceiling on road gradient, regardless of how much grip a surface
# has. Steeper roads are rejected rather than silently trapping a rollout.
MAX_DRIVABLE_GRADE_DEGREES = 16.0
# Fraction of available traction a circuit may spend on merely holding its own
# weight uphill. Spending all of it means the car creeps at zero net acceleration
# and runs out of step budget rather than failing outright, so it is capped well
# below one.
UPHILL_TRACTION_BUDGET = .42
@dataclass(frozen=True)
class TrackSurface3D:
    """Per-sample elevation, banking, and derived slope for one circuit.

    Everything is indexed by centerline sample so a lookup is an array read plus
    a neighbour interpolation, cheap enough to call once per physics substep.
    All lengths are in pixels, matching the plan-view geometry.
    """

    heights: tuple[float, ...]
    banks: tuple[float, ...]
    grades: tuple[float, ...]
    spacing: float
    half_width: float
    spec: ElevationSpec
    seam_step: float = 0.0
    """Height the profile gains over exactly one lap; must be zero to be closed."""
    grade_limit_degrees: float = MAX_DRIVABLE_GRADE_DEGREES
    """Steepest climb this specific car and surface can drive, not a global bound."""

    @property
    def sample_count(self) -> int:
        return len(self.heights)

    @property
    def relief_pixels(self) -> float:
        return max(self.heights) - min(self.heights)

    @property
    def steepest_grade_degrees(self) -> float:
        return math.degrees(max(abs(grade) for grade in self.grades))

    @property
    def steepest_bank_degrees(self) -> float:
        return math.degrees(max(abs(bank) for bank in self.banks))

    def height_at_index(self, index: int) -> float:
        return self.heights[index % self.sample_count]

    def bank_at_index(self, index: int) -> float:
        return self.banks[index % self.sample_count]

    def grade_at_index(self, index: int) -> float:
        return self.grades[index % self.sample_count]

    def height_at_position(self, position: float) -> float:
        """Height at a fractional sample position, smoothly.

        The stored heights are a sampling of a continuous profile, so linear
        interpolation between them leaves a visible crease at every crest. A
        Catmull-Rom pass through the same samples restores the curvature the
        profile actually has, which is what lets a renderer add detail between
        samples instead of subdividing a flat plane into smaller flat planes.
        """
        return _catmull_rom(self.heights, position)

    def bank_at_position(self, position: float) -> float:
        return _catmull_rom(self.banks, position)

    def attitude_at_index(self, index: int) -> tuple[float, float]:
        """Uphill grade and cross-slope bank at one sample, in radians."""
        wrapped = index % self.sample_count
        return self.grades[wrapped], self.banks[wrapped]

    def surface_height(self, point: Vec2, centerline: list[Vec2], index: int) -> float:
        """Road height under a point, including its lateral offset on a bank.

        The car sits on the road plane, so being off to one side of a banked
        corner genuinely raises or lowers it. Height is interpolated along the
        track between samples so crests are smooth rather than stepped.
        """
        count = self.sample_count
        current = centerline[index % count]
        following = centerline[(index + 1) % count]
        forward_x, forward_y = following.x - current.x, following.y - current.y
        length_squared = forward_x * forward_x + forward_y * forward_y
        if length_squared <= 1e-9:
            fraction = 0.0
        else:
            fraction = (
                (point.x - current.x) * forward_x + (point.y - current.y) * forward_y
            ) / length_squared
            fraction = max(0.0, min(1.0, fraction))
        # The same smooth profile the renderer draws, so a car never floats above
        # or sinks into a crest relative to the road it is standing on.
        centre_height = self.height_at_position(index + fraction)
        bank = self.bank_at_position(index + fraction)
        lateral = _signed_lateral_offset(point, current, following)
        return centre_height + math.tan(bank) * lateral

    def edge_heights(self, index: int) -> tuple[float, float]:
        """Height of the left and right road edges at one sample."""
        centre = self.heights[index % self.sample_count]
        rise = math.tan(self.banks[index % self.sample_count]) * self.half_width
        return centre + rise, centre - rise

    def normal_at_index(self, index: int) -> tuple[float, float, float]:
        """Unit surface normal in world axes, for shading and camera framing."""
        grade, bank = self.attitude_at_index(index)
        # Start from world up, tip it back by the grade and sideways by the bank.
        return _normalize((
            -math.sin(bank),
            -math.sin(grade),
            math.cos(grade) * math.cos(bank),
        ))


def compile_track_surface(
    scene: SceneSpec, elevation: ElevationSpec | None = None,
) -> TrackSurface3D:
    """Build the vertical profile for a compiled 2D scene, deterministically."""
    spec = elevation or scene.elevation or ElevationSpec(profile=ElevationProfile.FLAT)
    centerline = scene.track_centerline
    count = len(centerline)
    spacing = _mean_spacing(centerline)
    amplitude = spec.amplitude_m * scene.dynamics.pixels_per_meter
    detail = spec.crest_sharpness
    # Phases are fixed multiples of the seed rather than random draws, so a scene
    # compiles to the same hills on every machine and in every process.
    phase = (scene.seed % 360) / 360 * math.tau
    # Geometry is controlled entirely by the stored numeric parameter. Profile
    # labels never select one of a few canned waveforms: they only supply a
    # default `crest_sharpness` during model validation. The nonlinear third
    # harmonic comes in gradually as sharpness rises, keeping the control
    # continuous throughout its range.
    weights = (1.0, .65 * detail, .28 * detail * detail)
    heights = [
        _profile_height(index / count, spec.hill_count, phase, weights, amplitude)
        for index in range(count)
    ]
    if spec.is_flat:
        heights = [0.0] * count
    # One full lap on from the origin. Integer harmonics make this exactly the
    # starting height, so a step here can only come from a future change that
    # breaks periodicity -- which the grade check alone could never see.
    seam_step = 0.0 if spec.is_flat else (
        _profile_height(1.0, spec.hill_count, phase, weights, amplitude) - heights[0]
    )

    banks = _compile_banking(centerline, spec)
    grades = _compile_grades(heights, spacing)
    # `banking_degrees` is a stated maximum, so the rounding applied for
    # cross-platform determinism must not be able to round a corner past it.
    bank_limit = math.radians(spec.banking_degrees)
    return TrackSurface3D(
        heights=tuple(round(value, 4) for value in heights),
        banks=tuple(
            max(-bank_limit, min(bank_limit, round(value, 6))) for value in banks
        ),
        grades=tuple(round(value, 6) for value in grades),
        spacing=spacing,
        half_width=scene.track_width / 2,
        spec=spec,
        seam_step=seam_step,
        grade_limit_degrees=drivable_grade_limit(scene),
    )


def drivable_grade_limit(scene: SceneSpec) -> float:
    """Steepest gradient this car can climb on this surface, in degrees.

    Climbing is traction-limited, not a property of the road alone: holding the
    car's weight against a slope costs `sin(grade)` of the available friction, so
    an ice circuit runs out of grip on a hill an asphalt one takes at full
    throttle. Treating the limit as a single global number is what let low-grip
    circuits compile into a climb the oracle could only creep up.
    """
    vehicle, road = scene.dynamics.vehicle, scene.dynamics.road
    gravity = scene.dynamics.gravity_mps2
    engine_limit = vehicle.engine_force_n / max(1e-6, vehicle.mass_kg * gravity)
    traction_limit = road.friction_coefficient * vehicle.tire_friction_multiplier
    usable = min(engine_limit, traction_limit) * UPHILL_TRACTION_BUDGET
    return min(
        MAX_DRIVABLE_GRADE_DEGREES,
        math.degrees(math.asin(max(.01, min(.9, usable)))),
    )


def _profile_height(
    lap_fraction: float, hill_count: int, phase: float,
    weights: tuple[float, float, float], amplitude: float,
) -> float:
    """Elevation at a lap fraction as a sum of harmonics of the lap itself.

    Because every term is a harmonic of the lap, the profile is exactly periodic:
    substituting `lap_fraction + 1` adds a whole number of turns to each angle.
    """
    angle = math.tau * hill_count * lap_fraction + phase
    value = (
        weights[0] * math.sin(angle)
        + weights[1] * math.sin(2 * angle + phase * .5)
        + weights[2] * math.sin(3 * angle + phase * 1.5)
    )
    return amplitude / 2 * value / max(1e-6, sum(weights))


def validate_track_surface(surface: TrackSurface3D) -> list[str]:
    """Reject a vertical profile the car could not actually drive."""
    findings: list[str] = []
    if surface.sample_count < 8:
        findings.append("Elevation profile needs at least eight samples")
        return findings
    if not all(math.isfinite(value) for value in surface.heights):
        findings.append("Elevation profile must contain only finite heights")
    if not all(math.isfinite(value) for value in (*surface.banks, *surface.grades)):
        findings.append("Elevation profile must contain only finite slopes")
    # Periodicity is the one property the grade check cannot see: a profile could
    # climb steadily and gently and still be a cliff once per lap. Adjacent
    # samples across the seam are a normal slope and are covered by the grades.
    if abs(surface.seam_step) > 1e-6:
        findings.append(
            f"Elevation profile gains {surface.seam_step:.3f}px over a lap instead of closing"
        )
    if surface.steepest_grade_degrees > surface.grade_limit_degrees + 1e-6:
        findings.append(
            f"Circuit climbs at {surface.steepest_grade_degrees:.1f} degrees, steeper than the "
            f"{surface.grade_limit_degrees:.1f}-degree limit this car has grip for"
        )
    if surface.steepest_bank_degrees > 24.0:
        findings.append("Corner banking exceeds the drivable cross-slope limit")
    return findings


_ELEVATION_WORDS = (
    "3d", "three-dimensional", "elevated", "elevation", "hill", "hilly", "hills",
    "mountain", "mountainous", "alpine", "gradient", "incline", "inclined", "slope",
    "sloped", "sloping", "uphill", "downhill", "climb", "descent", "crest", "undulating",
    "rolling", "banked", "banking", "camber", "cambered", "valley", "ridge",
)
_STEEP_WORDS = ("alpine", "mountain", "mountainous", "steep", "dramatic", "extreme", "huge", "big")
_GENTLE_WORDS = ("gentle", "slight", "mild", "subtle", "shallow", "rolling", "undulating")
_BANK_WORDS = ("banked", "banking", "camber", "cambered", "bowl", "oval")


def parse_elevation_prompt(prompt: str) -> ElevationSpec | None:
    """Read elevation intent out of a brief, or return None for a planar request.

    The corner grammar reads surface, grip, corners, laps, barriers and opponents from a brief
    but has never read the vertical profile, so "an elevated loop with banked corners" compiled
    a flat circuit and looked like the request had been thrown away. Elevation is now parsed the
    same deterministic way as everything else the brief can ask for: local code owns the mapping
    from words to a spec, and the compiler still fits it to a drivable grade afterwards.
    """
    text = " ".join(prompt.lower().split())
    if not any(word in text for word in _ELEVATION_WORDS):
        return None
    strong_relief = bool(re.search(
        r"\b(?:high|large|major|substantial|significant)\s+"
        r"(?:elevation|relief|vertical|height|(?:elevation\s+)?differentials?)\b",
        text,
    ))
    steep = any(word in text for word in _STEEP_WORDS) or strong_relief
    gentle = any(word in text for word in _GENTLE_WORDS) and not steep
    banked = any(word in text for word in _BANK_WORDS)
    if steep:
        profile, amplitude, hills = ElevationProfile.ALPINE, 14.0, 4
    elif gentle:
        profile, amplitude, hills = ElevationProfile.ROLLING, 4.0, 2
    else:
        profile, amplitude, hills = ElevationProfile.HILLY, 8.0, 3
    return ElevationSpec(
        profile=profile, amplitude_m=amplitude, hill_count=hills,
        crest_sharpness=.78 if steep else .18 if gentle else .48,
        # Banking is only raised when it was asked for: a cross-slope changes the cornering
        # limit, so applying it to every hilly circuit would quietly alter the driving problem.
        banking_degrees=14.0 if banked else 6.0,
    )


def fit_drivable_elevation(scene: SceneSpec, spec: ElevationSpec) -> tuple[ElevationSpec, list[str]]:
    """Scale an elevation request down until the circuit is drivable.

    Amplitude and hill count interact with circuit length: three big crests fit
    comfortably on a long lap and are a wall on a short one. Grade is set by the
    ratio of the two, so crests are stretched out before height is given up --
    lengthening a hill keeps the relief that makes a 3D circuit worth driving,
    while shrinking it just flattens the track.
    """
    notes: list[str] = []
    if validate_track_surface(compile_track_surface(scene, spec)) == []:
        return spec, notes
    for hills in range(spec.hill_count, 0, -1):
        candidate = spec.model_copy(update={"hill_count": hills})
        if validate_track_surface(compile_track_surface(scene, candidate)) == []:
            notes.append(
                f"Stretched the elevation profile from {spec.hill_count} to {hills} crest(s) so "
                "the climb stays inside the drivable grade."
            )
            return candidate, notes
    stretched = spec.model_copy(update={"hill_count": 1})
    amplitude = stretched.amplitude_m
    for _ in range(16):
        amplitude *= .8
        candidate = stretched.model_copy(update={"amplitude_m": round(amplitude, 3)})
        if validate_track_surface(compile_track_surface(scene, candidate)) == []:
            notes.append(
                f"Reduced the elevation profile to one {candidate.amplitude_m:.1f} m crest; "
                f"{spec.amplitude_m:.1f} m over {spec.hill_count} crest(s) exceeded the drivable "
                "grade on a circuit this short."
            )
            return candidate, notes
        if amplitude <= .2:
            break
    notes.append(
        f"Flattened the elevation profile: no {spec.profile.value} variant of "
        f"{spec.amplitude_m:.1f} m fits inside the drivable grade on this circuit."
    )
    return spec.model_copy(update={"amplitude_m": 0.0, "profile": ElevationProfile.FLAT}), notes


def _compile_banking(centerline: list[Vec2], spec: ElevationSpec) -> list[float]:
    """Lean the road into its corners, proportional to plan-view curvature."""
    count = len(centerline)
    maximum = math.radians(spec.banking_degrees)
    if maximum <= 0:
        return [0.0] * count
    raw: list[float] = []
    for index in range(count):
        before = centerline[(index - 2) % count]
        current = centerline[index]
        after = centerline[(index + 2) % count]
        turn = _signed_angle_delta(
            _bearing(before, current), _bearing(current, after),
        )
        # Normalize against a sharp corner's per-sample turn so the tightest
        # corner on any circuit reaches the requested maximum bank.
        raw.append(max(-1.0, min(1.0, turn / 14.0)))
    # A three-pass moving average keeps banking continuous through corner entry
    # and exit; an abrupt cross-slope change would snap the car sideways.
    smoothed = raw
    for _ in range(3):
        smoothed = [
            (smoothed[(index - 1) % count] + 2 * smoothed[index] + smoothed[(index + 1) % count]) / 4
            for index in range(count)
        ]
    return [value * maximum for value in smoothed]


def _compile_grades(heights: list[float], spacing: float) -> list[float]:
    """Uphill slope at each sample from a central difference of the heights."""
    count = len(heights)
    return [
        math.atan2(
            heights[(index + 1) % count] - heights[(index - 1) % count],
            2 * max(1e-6, spacing),
        )
        for index in range(count)
    ]


def _catmull_rom(values: tuple[float, ...], position: float) -> float:
    """Interpolate a periodic sample array smoothly at a fractional index."""
    count = len(values)
    if count == 0:
        return 0.0
    base = math.floor(position)
    fraction = position - base
    before = values[(base - 1) % count]
    start = values[base % count]
    end = values[(base + 1) % count]
    after = values[(base + 2) % count]
    return .5 * (
        2 * start
        + (-before + end) * fraction
        + (2 * before - 5 * start + 4 * end - after) * fraction ** 2
        + (-before + 3 * start - 3 * end + after) * fraction ** 3
    )


def _mean_spacing(centerline: list[Vec2]) -> float:
    count = len(centerline)
    return sum(
        math.hypot(
            centerline[(index + 1) % count].x - centerline[index].x,
            centerline[(index + 1) % count].y - centerline[index].y,
        )
        for index in range(count)
    ) / max(1, count)


def _signed_lateral_offset(point: Vec2, current: Vec2, following: Vec2) -> float:
    forward_x, forward_y = following.x - current.x, following.y - current.y
    length = max(1e-6, math.hypot(forward_x, forward_y))
    return (
        (point.x - current.x) * (-forward_y / length)
        + (point.y - current.y) * (forward_x / length)
    )


def _bearing(origin: Vec2, target: Vec2) -> float:
    return math.degrees(math.atan2(target.y - origin.y, target.x - origin.x)) % 360


def _signed_angle_delta(current: float, target: float) -> float:
    return (target - current + 180) % 360 - 180


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-9:
        return (0.0, 0.0, 1.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)

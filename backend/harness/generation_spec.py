"""Checkable specifications for environment generation.

A brief is graded against a list of typed assertions rather than an opinion. Each
assertion names one thing the brief asked for, how it is read off a compiled
scene, and the tolerance it is allowed. Two properties matter:

- The same scorer grades every generator arm, so a comparison is not a matter of
  which arm reports more confidently about itself.
- An assertion may be marked `held_out`. Held-out assertions are never shown to a
  generator's search loop, which is what separates "the harness optimized our
  metric" from "the harness produced a better circuit".
"""

from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from .models import SceneSpec
from .probes import ProbeReport
from .track_grammar import parse_track_prompt


class Assertion(BaseModel):
    """One checkable claim about a compiled scene."""

    id: str
    kind: str
    target: Any = None
    tolerance: float = 0
    held_out: bool = False
    """Excluded from every generator-visible score, used only in the final grade."""
    weight: float = 1.0
    label: str = ""

    def describe(self) -> str:
        return self.label or f"{self.kind}={self.target!r}"


class AssertionResult(BaseModel):
    id: str
    kind: str
    label: str
    satisfied: bool
    achieved: Any = None
    residual: float = 0
    """Normalized shortfall; 0 when satisfied, larger the further away it landed."""
    message: str = ""
    held_out: bool = False


class SpecScore(BaseModel):
    results: list[AssertionResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def satisfied(self) -> int:
        return sum(1 for result in self.results if result.satisfied)

    @property
    def conjunction(self) -> bool:
        """The headline criterion: every assertion holds at once."""
        return bool(self.results) and all(result.satisfied for result in self.results)

    @property
    def satisfaction_rate(self) -> float:
        return round(self.satisfied / self.total, 4) if self.results else 0.0

    @property
    def weighted_residual(self) -> float:
        """Search objective; lower is better. Ties break toward fewer misses."""
        if not self.results:
            return 0.0
        return round(sum(result.residual for result in self.results), 5)

    def failures(self) -> list[AssertionResult]:
        return [result for result in self.results if not result.satisfied]

    def feedback(self, limit: int = 5) -> list[str]:
        """Concrete measured residuals to hand back to a generator."""
        ordered = sorted(self.failures(), key=lambda result: -result.residual)
        return [result.message for result in ordered[:limit]]

    def summary(self) -> str:
        return f"{self.satisfied}/{self.total} assertions" + (
            "" if self.conjunction else f"; missed: {'; '.join(self.feedback(3))}"
        )


class GenerationSpec(BaseModel):
    """The full grading contract for one brief."""

    prompt: str
    case_id: str = ""
    case_class: str = ""
    assertions: list[Assertion] = Field(default_factory=list)

    def visible(self) -> "GenerationSpec":
        """The part of the contract a search loop is allowed to optimize."""
        return self.model_copy(update={
            "assertions": [item for item in self.assertions if not item.held_out],
        })

    def held_out(self) -> "GenerationSpec":
        return self.model_copy(update={
            "assertions": [item for item in self.assertions if item.held_out],
        })


# --- evaluators ---------------------------------------------------------------


class _Context:
    def __init__(self, scene: SceneSpec, probes: ProbeReport | None) -> None:
        self.scene = scene
        self.report = scene.track_report
        self.probes = probes


Evaluator = Callable[[Assertion, _Context], tuple[bool, Any, str]]
_EVALUATORS: dict[str, Evaluator] = {}


def _evaluator(kind: str) -> Callable[[Evaluator], Evaluator]:
    def register(function: Evaluator) -> Evaluator:
        _EVALUATORS[kind] = function
        return function
    return register


def _match(assertion: Assertion, actual: Any, noun: str) -> tuple[bool, Any, str]:
    satisfied = actual == assertion.target
    return satisfied, actual, (
        "" if satisfied else f"{noun} is {actual!r}, the brief asked for {assertion.target!r}"
    )


def _at_most(assertion: Assertion, actual: float, noun: str) -> tuple[bool, Any, str]:
    satisfied = actual <= float(assertion.target) + assertion.tolerance
    return satisfied, actual, (
        "" if satisfied else f"{noun} is {actual}, the brief needs at most {assertion.target}"
    )


def _at_least(assertion: Assertion, actual: float, noun: str) -> tuple[bool, Any, str]:
    satisfied = actual >= float(assertion.target) - assertion.tolerance
    return satisfied, actual, (
        "" if satisfied else f"{noun} is {actual}, the brief needs at least {assertion.target}"
    )


def _near(assertion: Assertion, actual: float | None, noun: str) -> tuple[bool, Any, str]:
    if actual is None:
        return False, None, f"{noun} could not be measured on this scene"
    satisfied = abs(actual - float(assertion.target)) <= assertion.tolerance
    return satisfied, actual, (
        "" if satisfied
        else f"{noun} is {actual}, the brief asked for {assertion.target} (±{assertion.tolerance})"
    )


@_evaluator("surface")
def _surface(assertion, context):
    return _match(assertion, context.scene.surface, "surface")


@_evaluator("laps")
def _laps(assertion, context):
    return _match(assertion, context.scene.laps, "lap count")


@_evaluator("direction")
def _direction(assertion, context):
    return _match(assertion, context.report.direction if context.report else None, "circuit direction")


@_evaluator("npc_start_mode")
def _npc_start_mode(assertion, context):
    return _match(assertion, context.scene.npc_start_mode, "NPC start mode")


@_evaluator("start_line_region")
def _start_line_region(assertion, context):
    actual = context.scene.start_line_region.value
    return _match(assertion, actual, "start/finish region")


@_evaluator("player_grid_position")
def _player_grid_position(assertion, context):
    return _match(assertion, context.scene.player_grid_position, "player grid position")


@_evaluator("grip_max")
def _grip_max(assertion, context):
    return _at_most(assertion, context.scene.grip, "grip")


@_evaluator("grip_min")
def _grip_min(assertion, context):
    return _at_least(assertion, context.scene.grip, "grip")


@_evaluator("grip_target")
def _grip_target(assertion, context):
    return _near(assertion, context.scene.grip, "grip")


@_evaluator("track_width_max")
def _width_max(assertion, context):
    return _at_most(assertion, context.scene.track_width, "corridor width")


@_evaluator("track_width_min")
def _width_min(assertion, context):
    return _at_least(assertion, context.scene.track_width, "corridor width")


@_evaluator("npc_count")
def _npc_count(assertion, context):
    actual = sum(1 for entity in context.scene.entities if entity.kind == "npc")
    return _match(assertion, actual, "opponent count")


@_evaluator("npc_profiles")
def _npc_profiles(assertion, context):
    actual = sorted(behavior.profile.value for behavior in context.scene.npc_behaviors)
    wanted = sorted(str(item) for item in (assertion.target or []))
    satisfied = actual == wanted
    return satisfied, actual, (
        "" if satisfied else f"opponent temperaments are {actual}, the brief asked for {wanted}"
    )


@_evaluator("barrier_count")
def _barrier_count(assertion, context):
    actual = sum(1 for entity in context.scene.entities if entity.kind == "obstacle")
    return _match(assertion, actual, "barrier count")


@_evaluator("elevation_profile")
def _elevation_profile(assertion, context):
    actual = context.scene.elevation.profile.value if context.scene.elevation else "flat"
    return _match(assertion, actual, "elevation profile")


@_evaluator("elevation_amplitude_min")
def _elevation_amplitude_min(assertion, context):
    actual = context.scene.elevation.amplitude_m if context.scene.elevation else 0.0
    return _at_least(assertion, actual, "elevation amplitude in meters")


@_evaluator("elevation_hill_count")
def _elevation_hill_count(assertion, context):
    actual = context.scene.elevation.hill_count if context.scene.elevation else 0
    return _match(assertion, actual, "elevation crest count")


@_evaluator("corner_count_min")
def _corner_count_min(assertion, context):
    actual = len(context.report.corners) if context.report else 0
    return _at_least(assertion, actual, "corner count")


@_evaluator("loop_shape")
def _loop_shape(assertion, context):
    actual = context.report.loop_shape if context.report else "unknown"
    return _match(assertion, actual, "loop shape")


@_evaluator("corner_in_region")
def _corner_in_region(assertion, context):
    """A corner of roughly this angle must actually land in this screen region."""
    wanted = dict(assertion.target or {})
    angle, region = float(wanted.get("angle", 90)), str(wanted.get("region", "auto"))
    corners = context.report.corners if context.report else []
    matched = [
        corner for corner in corners
        if abs(corner.achieved_angle_degrees - angle) <= (assertion.tolerance or 5)
        and corner.achieved_region.value == region
    ]
    if matched:
        return True, region, ""
    in_region = [
        round(corner.achieved_angle_degrees, 1) for corner in corners
        if corner.achieved_region.value == region
    ]
    nearest = sorted(corners, key=lambda corner: abs(corner.achieved_angle_degrees - angle))
    detail = (
        f"angles in {region} are {in_region}" if in_region
        else f"no corner landed in {region}; the closest {angle}° corner is in "
        + (nearest[0].achieved_region.value if nearest else "no region")
    )
    return False, in_region, f"missing a {angle}° corner in {region}: {detail}"


def _colour_distance(left: str, right: str) -> float:
    """Straight RGB distance, normalized to 0-1.

    Crude on purpose. The question a brief asks is "is the barrier blue", not "is it
    within 3 JND of this swatch", and a perceptual space would imply a precision the
    request never had. What matters is that "blue" and "red" are far apart and "blue"
    and "azure" are close, which this gets right.
    """
    from .visual import rgb

    one, two = rgb(left), rgb(right)
    return sum(abs(a - b) for a, b in zip(one, two)) / (3 * 255)


def _visual_slot(assertion, context, slot: str, noun: str):
    """A palette entry landed near the colour the brief named."""
    from .visual import to_hex

    palette = context.scene.visual.resolved(context.scene.surface)
    actual = palette.get(slot)
    if not isinstance(actual, str):
        return False, actual, f"{noun} has no colour on this scene"
    wanted = to_hex(str(assertion.target))
    if wanted is None:
        # An unresolvable target is a check that cannot be settled, not a failure of
        # the circuit. Saying so beats scoring the scene against black.
        return False, actual, f"{assertion.target!r} is not a colour this harness can resolve"
    distance = _colour_distance(actual, wanted)
    tolerance = assertion.tolerance or .18
    satisfied = distance <= tolerance
    return satisfied, actual, (
        "" if satisfied
        else f"{noun} is {actual}, the brief asked for something like {assertion.target}"
    )


@_evaluator("road_colour")
def _road_colour(assertion, context):
    return _visual_slot(assertion, context, "road", "the road")


@_evaluator("terrain_colour")
def _terrain_colour(assertion, context):
    return _visual_slot(assertion, context, "terrain", "the surrounding ground")


@_evaluator("barrier_colour")
def _barrier_colour(assertion, context):
    return _visual_slot(assertion, context, "barrier", "the barriers")


@_evaluator("player_car_colour")
def _player_car_colour(assertion, context):
    return _visual_slot(assertion, context, "player_car", "the player's car")


@_evaluator("opponent_car_colour")
def _opponent_car_colour(assertion, context):
    return _visual_slot(assertion, context, "opponent_car", "the opponent cars")


@_evaluator("sky_colour")
def _sky_colour(assertion, context):
    return _visual_slot(assertion, context, "sky", "the sky")


@_evaluator("kerbs_present")
def _kerbs_present(assertion, context):
    actual = bool(context.scene.visual.kerbs)
    satisfied = actual == bool(assertion.target)
    return satisfied, actual, (
        "" if satisfied
        else ("the red-and-white edge striping is still drawn" if actual
              else "the circuit has no edge striping")
    )


@_evaluator("scenery_count")
def _scenery_count(assertion, context):
    actual = len(context.scene.visual.scenery)
    return _match(assertion, actual, "scenery band count")


@_evaluator("angle_fidelity_max")
def _angle_fidelity(assertion, context):
    actual = context.report.angle_fidelity_degrees if context.report else 999
    return _at_most(assertion, actual, "turn-angle error")


@_evaluator("closure_error_max")
def _closure(assertion, context):
    actual = context.report.closure_error_pixels if context.report else 999
    return _at_most(assertion, actual, "loop closure error")


@_evaluator("min_radius_max")
def _min_radius_max(assertion, context):
    actual = context.report.minimum_radius_pixels if context.report else 9_999
    return _at_most(assertion, actual, "tightest corner radius")


@_evaluator("min_radius_min")
def _min_radius_min(assertion, context):
    """Nothing on the circuit is tighter than this — the "no hairpins" direction.

    Every bound here needs its opposite. A brief asking for sweeping curves and
    nothing tight has no way to say so with only `min_radius_max`, and a reader
    forced to choose between a wrong check and no check will pick the wrong one.
    """
    actual = context.report.minimum_radius_pixels if context.report else 0
    return _at_least(assertion, actual, "tightest corner radius")


@_evaluator("corner_count_max")
def _corner_count_max(assertion, context):
    actual = len(context.report.corners) if context.report else 99
    return _at_most(assertion, actual, "corner count")


@_evaluator("longest_straight_min")
def _longest_straight(assertion, context):
    actual = context.report.longest_straight_pixels if context.report else 0
    return _at_least(assertion, actual, "longest straight")


@_evaluator("longest_straight_max")
def _longest_straight_max(assertion, context):
    actual = context.report.longest_straight_pixels if context.report else 9_999
    return _at_most(assertion, actual, "longest straight")


def _probes(context: _Context) -> ProbeReport:
    if context.probes is None:
        raise ValueError("This assertion needs a probe report but none was measured.")
    return context.probes


@_evaluator("oracle_finishes")
def _oracle_finishes(assertion, context):
    actual = _probes(context).oracle_finished
    satisfied = actual == bool(assertion.target)
    return satisfied, actual, (
        "" if satisfied
        else f"reference driver outcome was finished={actual} ({_probes(context).oracle_failure})"
    )


@_evaluator("naive_finishes")
def _naive_finishes(assertion, context):
    actual = _probes(context).naive_finished
    satisfied = actual == bool(assertion.target)
    return satisfied, actual, (
        "" if satisfied
        else f"an unbraked driver finished={actual}, the brief implies {bool(assertion.target)}"
    )


@_evaluator("naive_off_track_min")
def _naive_off_track_min(assertion, context):
    return _at_least(assertion, _probes(context).naive_off_track_ticks, "unbraked-driver off-track ticks")


@_evaluator("naive_off_track_max")
def _naive_off_track_max(assertion, context):
    """A careless driver mostly stays on the road — the "forgiving" direction."""
    return _at_most(assertion, _probes(context).naive_off_track_ticks, "unbraked-driver off-track ticks")


@_evaluator("oracle_seconds")
def _oracle_seconds(assertion, context):
    return _near(assertion, _probes(context).oracle_seconds, "reference lap/race time in seconds")


@_evaluator("oracle_seconds_max")
def _oracle_seconds_max(assertion, context):
    return _at_most(assertion, _probes(context).oracle_seconds, "reference race time in seconds")


@_evaluator("off_track_ticks_max")
def _off_track_max(assertion, context):
    return _at_most(assertion, _probes(context).off_track_ticks, "reference-driver off-track ticks")


@_evaluator("order_changes_min")
def _order_changes(assertion, context):
    return _at_least(assertion, _probes(context).order_changes, "position changes in the field")


@_evaluator("field_spread_max")
def _field_spread(assertion, context):
    actual = _probes(context).field_spread_seconds
    if actual is None:
        return False, None, (
            f"only {_probes(context).opponents_finished} opponents finished, so the field "
            "spread is undefined"
        )
    return _at_most(assertion, actual, "opponent finish spread in seconds")


@_evaluator("brake_fraction_min")
def _brake_min(assertion, context):
    return _at_least(assertion, _probes(context).brake_fraction, "share of the race spent braking")


@_evaluator("brake_fraction_max")
def _brake_max(assertion, context):
    return _at_most(assertion, _probes(context).brake_fraction, "share of the race spent braking")


@_evaluator("mean_speed_min")
def _mean_speed_min(assertion, context):
    return _at_least(assertion, _probes(context).oracle_mean_speed, "reference mean speed")


@_evaluator("mean_speed_max")
def _mean_speed_max(assertion, context):
    return _at_most(assertion, _probes(context).oracle_mean_speed, "reference mean speed")


def needs_probes(spec: GenerationSpec) -> bool:
    return any(
        assertion.kind in _PROBE_KINDS for assertion in spec.assertions
    )


_PROBE_KINDS = frozenset({
    "oracle_finishes", "naive_finishes", "naive_off_track_min", "naive_off_track_max",
    "oracle_seconds", "oracle_seconds_max", "off_track_ticks_max", "order_changes_min",
    "field_spread_max", "brake_fraction_min", "brake_fraction_max", "mean_speed_min",
    "mean_speed_max",
})


def _residual(assertion: Assertion, satisfied: bool, achieved: Any) -> float:
    """Normalized distance from satisfaction, used to rank candidates."""
    if satisfied:
        return 0.0
    target = assertion.target
    if isinstance(target, (int, float)) and isinstance(achieved, (int, float)) and not isinstance(target, bool):
        scale = max(abs(float(target)), 1.0)
        gap = max(0.0, abs(float(achieved) - float(target)) - assertion.tolerance)
        return round(assertion.weight * min(2.0, gap / scale), 5)
    return round(assertion.weight * 1.0, 5)


def score(spec: GenerationSpec, scene: SceneSpec | None, probes: ProbeReport | None) -> SpecScore:
    """Grade one compiled scene against one specification.

    A scene that failed to compile at all fails every assertion rather than being
    excluded, because "produced nothing" is a generation outcome, not a missing
    data point.
    """
    results: list[AssertionResult] = []
    for assertion in spec.assertions:
        if scene is None:
            results.append(AssertionResult(
                id=assertion.id, kind=assertion.kind, label=assertion.describe(),
                satisfied=False, residual=assertion.weight,
                message=f"no scene was produced, so {assertion.describe()} is unmet",
                held_out=assertion.held_out,
            ))
            continue
        evaluator = _EVALUATORS.get(assertion.kind)
        if evaluator is None:
            raise KeyError(f"Unknown assertion kind: {assertion.kind}")
        context = _Context(scene, probes)
        try:
            satisfied, achieved, message = evaluator(assertion, context)
        except ValueError as error:
            satisfied, achieved, message = False, None, str(error)
        results.append(AssertionResult(
            id=assertion.id, kind=assertion.kind, label=assertion.describe(),
            satisfied=satisfied, achieved=achieved,
            residual=_residual(assertion, satisfied, achieved),
            message=message or f"{assertion.describe()} holds",
            held_out=assertion.held_out,
        ))
    return SpecScore(results=results)


# --- deterministic extraction -------------------------------------------------
#
# The search-side spec is derived from the brief by local code, never by a model
# call. That keeps the harness's objective reproducible and stops the generator
# from grading its own homework by reinterpreting the brief.

_DEFAULT_PLAN = None


def _default_plan():
    global _DEFAULT_PLAN
    if _DEFAULT_PLAN is None:
        _DEFAULT_PLAN = parse_track_prompt("a circuit")
    return _DEFAULT_PLAN


def extract_spec(prompt: str) -> GenerationSpec:
    """Read a brief into assertions the harness can search against.

    Only phrases the brief actually evidences become assertions: a field is
    asserted when the deterministic parser moved it away from the value it takes
    for a featureless brief. Silent defaults are not constraints.
    """
    text = " ".join(prompt.lower().split())
    plan = parse_track_prompt(prompt)
    default = _default_plan()
    assertions: list[Assertion] = []

    def add(kind: str, target: Any, label: str, tolerance: float = 0) -> None:
        assertions.append(Assertion(
            id=f"{kind}-{len(assertions)}", kind=kind, target=target,
            tolerance=tolerance, label=label,
        ))

    # Drivability is implicit in every brief.
    add("oracle_finishes", True, "the circuit can be completed by the reference driver")

    if plan.surface != default.surface:
        add("surface", plan.surface, f"surface is {plan.surface}")
    if plan.grip < default.grip:
        add("grip_max", round(plan.grip + .05, 2), f"grip is at most {round(plan.grip + .05, 2)}")
    if plan.grip > default.grip:
        add("grip_min", round(plan.grip - .05, 2), f"grip is at least {round(plan.grip - .05, 2)}")
    if plan.laps != default.laps:
        add("laps", plan.laps, f"{plan.laps} laps")
    if plan.direction != default.direction:
        add("direction", plan.direction, f"runs {plan.direction}")
    if plan.loop_shape != default.loop_shape:
        add("loop_shape", plan.loop_shape, "a true circular no-corner centerline")
    if plan.npc_start_mode != default.npc_start_mode:
        add("npc_start_mode", plan.npc_start_mode, f"{plan.npc_start_mode} start")
    if plan.start_region != default.start_region:
        add("start_line_region", plan.start_region.value, f"start/finish in {plan.start_region.value}")
    if plan.player_grid_position != default.player_grid_position:
        add("player_grid_position", plan.player_grid_position, f"player starts P{plan.player_grid_position}")
    if plan.track_width != default.track_width:
        if plan.track_width < default.track_width:
            add("track_width_max", plan.track_width + 6, "a narrow corridor")
        else:
            add("track_width_min", plan.track_width - 6, "a wide corridor")
    profiles = sorted(npc.profile.value for npc in plan.npcs)
    default_profiles = sorted(npc.profile.value for npc in default.npcs)
    # A featureless brief still parses to a token opponent, so an unchanged field
    # is not evidence that the brief asked for traffic.
    if plan.npcs and profiles != default_profiles:
        add("npc_count", len(plan.npcs), f"{len(plan.npcs)} opponents")
        add("npc_profiles", profiles, "the requested opponent temperaments")
    if plan.barriers:
        add("barrier_count", len(plan.barriers), f"{len(plan.barriers)} barriers")

    # The vertical surface is compiled outside TrackPlan, but it is still part of the
    # environment contract. Including it here lets a 3D search prefer a planar layout long
    # enough to preserve the requested relief rather than accepting one the grade fitter had
    # to flatten into a nominally 3D scene.
    from .track3d import parse_elevation_prompt

    elevation = parse_elevation_prompt(prompt)
    if elevation is not None:
        add("elevation_profile", elevation.profile.value,
            f"a {elevation.profile.value} elevation profile")
        add("elevation_amplitude_min", max(.5, elevation.amplitude_m * .5),
            "at least half of the requested vertical relief")
        # Hill count is a generator default, not a user requirement, unless the
        # brief actually names a count. The 3D grade fitter may reduce the
        # number of crests to keep a circuit drivable; grading that adaptation as
        # a miss would falsely report that a request for *high relief* failed.
        if re.search(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight)\s+"
            r"(?:elevation\s+)?(?:hills?|crests?|peaks?|climbs?)\b",
            text,
        ):
            add("elevation_hill_count", elevation.hill_count,
                f"{elevation.hill_count} elevation crests")
    for corner in plan.corners:
        if corner.angle_degrees is not None and corner.region.value != "auto":
            add("corner_in_region",
                {"angle": corner.angle_degrees, "region": corner.region.value},
                f"a {corner.angle_degrees:g}° corner in {corner.region.value}",
                tolerance=5)

    assertions.extend(_outcome_assertions(text, len(assertions)))
    return GenerationSpec(prompt=prompt, assertions=assertions)


def _outcome_assertions(text: str, offset: int) -> list[Assertion]:
    """Assertions that can only be settled by running the simulator."""
    found: list[Assertion] = []

    def add(kind: str, target: Any, label: str, tolerance: float = 0) -> None:
        found.append(Assertion(
            id=f"{kind}-{offset + len(found)}", kind=kind, target=target,
            tolerance=tolerance, label=label,
        ))

    seconds = re.search(
        r"(?:lap|race|circuit)[^.]{0,40}?(\d{1,3})\s*(?:-|\s)?second"
        r"|(\d{1,3})\s*(?:-|\s)?second[^.]{0,20}?(?:lap|race)", text)
    if seconds:
        target = float(next(group for group in seconds.groups() if group))
        add("oracle_seconds", target, f"a reference race time near {target:g}s",
            tolerance=max(3.0, target * .12))

    spread = re.search(r"within\s+(\d{1,3})\s*(?:-|\s)?second", text)
    if spread:
        target = float(spread.group(1))
        add("field_spread_max", target, f"the opponents finish within {target:g}s of each other")
    elif any(phrase in text for phrase in ("photo finish", "close finish", "tight finish")):
        add("field_spread_max", 3.0, "the opponents finish within 3s of each other")

    if any(phrase in text for phrase in (
        "hard but fair", "difficult but fair", "challenging but fair", "punishing",
        "demands braking", "punishes", "unforgiving",
    )):
        add("oracle_finishes", True, "a competent driver can still finish it")
        add("naive_off_track_min", 40, "an unbraked driver runs wide repeatedly")

    if any(phrase in text for phrase in (
        "overtaking", "overtake", "passing opportunit", "wheel to wheel", "wheel-to-wheel",
        "changes of position", "position changes",
    )):
        add("order_changes_min", 2, "the race produces real position changes")

    if any(phrase in text for phrase in ("brake-heavy", "braking zones", "stop and go", "stop-start")):
        add("brake_fraction_min", .3, "the circuit is braking-dominated")

    if any(phrase in text for phrase in ("flat out", "flat-out", "full throttle", "no braking")):
        add("brake_fraction_max", .1, "the circuit can be taken nearly flat out")

    return found

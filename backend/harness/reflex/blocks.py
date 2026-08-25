"""Instrumented control helpers, plus the arithmetic a sandboxed controller cannot import.

Two reasons a generated controller should call `ctrl.pid` rather than write the
arithmetic itself, and the second one is the real one.

**The harness owns the state.** A block's memory is keyed by its name and lives here, so
controller state is snapshotted with the runtime, restored on a fork, and printed in the
recorder. State kept in Python closures inside the controller would make `try_controller`
lie and would be invisible when something goes wrong.

**The harness sees the error signal.** Because the caller hands over the error, every
block can be instrumented for free: sign changes over a window, time spent at the output
clamp, mean error. That is where `unstable` comes from. Detecting an oscillating
controller is usually posed as an inference problem; here it is a counter, and it costs
nothing, because the loop primitive was supplied rather than written.

This module may not import `sense`, and there is a test asserting it. A block sees a
number and returns a number, so no amount of growth here can turn into a driver that
knows what a corner is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


OSCILLATION_WINDOW = 20
"""Ticks of error history kept per block — two seconds at 10 Hz."""

DEADBAND_FRACTION = 0.02
"""Errors smaller than this fraction of the window's peak do not count as a sign change.

Without it, a converged loop sitting on top of its target flips sign on floating-point
noise every tick and reports itself as violently unstable.
"""

NOISE_FLOOR = 1e-4
"""An absolute floor under the relative deadband, for a loop that has fully converged.

A relative deadband cannot help when the whole signal is noise: an error alternating
between ±1e-9 has a peak of 1e-9 and clears any fraction of it. No actuator in any unit
system responds to 1e-4, so oscillation below that is convergence, not instability.
"""


@dataclass
class BlockState:
    integral: float = 0.0
    previous_error: float | None = None
    previous_value: float | None = None
    output: float = 0.0
    latched: bool = False
    debounce_count: int = 0
    ticks: int = 0
    duty: float = 0.0
    errors: list[float] = field(default_factory=list)
    clamped_ticks: int = 0
    terms: dict = field(default_factory=dict)

    def note_error(self, error: float) -> None:
        self.errors.append(error)
        if len(self.errors) > OSCILLATION_WINDOW:
            self.errors.pop(0)

    @property
    def sign_changes(self) -> int:
        """Sign changes of the error over the window, ignoring noise near zero."""
        peak = max((abs(value) for value in self.errors), default=0.0)
        floor = max(peak * DEADBAND_FRACTION, NOISE_FLOOR)
        signs = [1 if value > floor else -1 if value < -floor else 0 for value in self.errors]
        meaningful = [sign for sign in signs if sign != 0]
        return sum(
            1 for previous, current in zip(meaningful, meaningful[1:]) if previous != current
        )

    @property
    def mean_error(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0

    def report(self) -> dict:
        return {
            "ticks": self.ticks,
            "sign_changes": self.sign_changes,
            "mean_error": round(self.mean_error, 4),
            "clamped_fraction": round(self.clamped_ticks / self.ticks, 3) if self.ticks else 0.0,
            "last_output": round(self.output, 4),
            **({"terms": self.terms} if self.terms else {}),
        }


class Params:
    """Controller parameters, addressed as `ctrl.p.name`.

    Separate from the source so retuning is a `patch_params` call that skips the install
    gate entirely. Most revisions are a gain or a target speed, and regenerating a whole
    controller body to change one number is the expensive way to do it.
    """

    __slots__ = ("_values",)

    def __init__(self, values: dict[str, float]):
        self._values = dict(values)

    def __getattr__(self, name: str) -> float:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(
                f"ctrl.p.{name} is not a declared parameter; declared: {sorted(self._values)}"
            ) from None

    def as_dict(self) -> dict[str, float]:
        return dict(self._values)


class ControlBlocks:
    """The `ctrl` argument: parameters, stateful blocks, and safe math."""

    def __init__(self, params: dict[str, float] | None = None):
        self.p = Params(params or {})
        self._blocks: dict[str, BlockState] = {}
        self.dt = 0.1
        """Seconds per control tick, set by the runtime from the scene's control rate."""

    # -- bookkeeping the runtime drives ------------------------------------------------

    def begin_tick(self, dt: float) -> None:
        self.dt = dt

    def reports(self) -> dict[str, dict]:
        return {name: state.report() for name, state in self._blocks.items()}

    def max_sign_changes(self) -> tuple[str | None, int]:
        """The worst-oscillating block, which is what the `unstable` condition reads."""
        worst_name, worst = None, 0
        for name, state in self._blocks.items():
            changes = state.sign_changes
            if changes > worst:
                worst_name, worst = name, changes
        return worst_name, worst

    def state(self, name: str) -> BlockState:
        if name not in self._blocks:
            self._blocks[name] = BlockState()
        return self._blocks[name]

    # -- feedback blocks ---------------------------------------------------------------

    def pid(
        self, name: str, error: float, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
        i_clamp: float = 1.0, limit: float = 1.0,
    ) -> float:
        """Proportional-integral-derivative on an error the caller computes."""
        state = self.state(name)
        state.ticks += 1
        state.note_error(error)
        derivative = 0.0 if state.previous_error is None else (error - state.previous_error) / self.dt
        state.previous_error = error
        if ki:
            state.integral = _clamp(state.integral + error * self.dt, -i_clamp, i_clamp)
        raw = kp * error + ki * state.integral + kd * derivative
        output = _clamp(raw, -limit, limit)
        if output != raw:
            state.clamped_ticks += 1
        state.output = output
        state.terms = {
            "p": round(kp * error, 4), "i": round(ki * state.integral, 4),
            "d": round(kd * derivative, 4), "error": round(error, 4),
        }
        return output

    def stanley(
        self, name: str, cross_track: float, heading_error_degrees: float,
        k: float = 1.0, v_floor: float = 1.0, speed: float = 1.0, limit: float = 1.0,
    ) -> float:
        """Align with the road, plus a cross-track term that softens with speed.

        The classic path-tracking law, offered because it is a common shape and not
        because it is the right one here. It is instrumented on the combined error like
        any other block.
        """
        state = self.state(name)
        state.ticks += 1
        cross_term = math.degrees(math.atan2(k * cross_track, max(v_floor, speed)))
        combined = heading_error_degrees + cross_term
        state.note_error(combined)
        raw = combined / 45.0
        output = _clamp(raw, -limit, limit)
        if output != raw:
            state.clamped_ticks += 1
        state.output = output
        state.terms = {
            "heading": round(heading_error_degrees, 2), "cross": round(cross_term, 2),
            "combined_degrees": round(combined, 2),
        }
        return output

    def pursuit(
        self, name: str, lateral_offset: float, lookahead: float, wheelbase: float = 0.6,
        limit: float = 1.0,
    ) -> float:
        """Geometric pure pursuit: steer to an aim point `lookahead` ahead.

        All three arguments are in car lengths, so the returned curvature command is in
        inverse car lengths before normalization.
        """
        state = self.state(name)
        state.ticks += 1
        state.note_error(lateral_offset)
        distance = max(0.2, lookahead)
        curvature = 2.0 * lateral_offset / (distance * distance)
        raw = math.atan2(curvature * wheelbase, 1.0) / math.radians(20.0)
        output = _clamp(raw, -limit, limit)
        if output != raw:
            state.clamped_ticks += 1
        state.output = output
        state.terms = {"curvature": round(curvature, 4), "lookahead": round(distance, 2)}
        return output

    # -- filters and logic -------------------------------------------------------------

    def ewma(self, name: str, value: float, tau: float = 0.5) -> float:
        state = self.state(name)
        state.ticks += 1
        alpha = 1.0 if tau <= 0 else min(1.0, self.dt / tau)
        state.output = value if state.previous_value is None else (
            state.output + alpha * (value - state.output)
        )
        state.previous_value = value
        return state.output

    def deriv(self, name: str, value: float) -> float:
        state = self.state(name)
        state.ticks += 1
        previous = state.previous_value
        state.previous_value = value
        return 0.0 if previous is None else (value - previous) / self.dt

    def integral(self, name: str, value: float, clamp: float = 10.0) -> float:
        state = self.state(name)
        state.ticks += 1
        state.integral = _clamp(state.integral + value * self.dt, -clamp, clamp)
        return state.integral

    def rate_limit(self, name: str, value: float, per_second: float = 5.0) -> float:
        state = self.state(name)
        state.ticks += 1
        maximum = abs(per_second) * self.dt
        current = value if state.previous_value is None else _clamp(
            value, state.previous_value - maximum, state.previous_value + maximum,
        )
        state.previous_value = current
        state.output = current
        return current

    def hysteresis(self, name: str, value: float, enter: float = 0.5, exit: float = 0.2) -> bool:
        """A boolean that resists flapping: needs `enter` to set, falls below `exit` to clear."""
        state = self.state(name)
        state.ticks += 1
        state.latched = value >= enter if not state.latched else value > exit
        return state.latched

    def latch(self, name: str, set_when: bool, clear_when: bool = False) -> bool:
        state = self.state(name)
        state.ticks += 1
        if clear_when:
            state.latched = False
        elif set_when:
            state.latched = True
        return state.latched

    def debounce(self, name: str, condition: bool, ticks: int = 3) -> bool:
        """True only once `condition` has held for `ticks` consecutive calls."""
        state = self.state(name)
        state.ticks += 1
        state.debounce_count = state.debounce_count + 1 if condition else 0
        return state.debounce_count >= max(1, ticks)

    def clock(self, name: str) -> float:
        """Seconds since this block first ran — for time-based state machines."""
        state = self.state(name)
        state.ticks += 1
        return (state.ticks - 1) * self.dt

    def once(self, name: str) -> bool:
        """True on the first call only. Cheap one-shot initialization."""
        state = self.state(name)
        state.ticks += 1
        return state.ticks == 1

    def memo(self, name: str, value: float | None = None) -> float:
        """Read the last stored value, or store a new one and return it.

        The escape hatch for a controller that needs state no block covers, without
        letting it keep that state where the runtime cannot see it.
        """
        state = self.state(name)
        if value is not None:
            state.previous_value = float(value)
        return 0.0 if state.previous_value is None else state.previous_value

    # -- math, since the sandbox has no imports ----------------------------------------

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return _clamp(value, low, high)

    @staticmethod
    def sign(value: float) -> float:
        return 0.0 if value == 0 else math.copysign(1.0, value)

    @staticmethod
    def hypot(x: float, y: float) -> float:
        return math.hypot(x, y)

    @staticmethod
    def sqrt(value: float) -> float:
        return math.sqrt(max(0.0, value))

    @staticmethod
    def atan2(y: float, x: float) -> float:
        return math.atan2(y, x)

    @staticmethod
    def degrees(radians: float) -> float:
        return math.degrees(radians)

    @staticmethod
    def radians(degrees: float) -> float:
        return math.radians(degrees)

    @staticmethod
    def cos(radians: float) -> float:
        return math.cos(radians)

    @staticmethod
    def sin(radians: float) -> float:
        return math.sin(radians)

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * _clamp(t, 0.0, 1.0)


HELPERS = (
    "pid(name, error, kp=, ki=, kd=, i_clamp=, limit=) -> [-limit, limit]",
    "stanley(name, cross_track, heading_error_degrees, k=, v_floor=, speed=)",
    "pursuit(name, lateral_offset, lookahead, wheelbase=)  # all in car lengths",
    "ewma(name, value, tau=seconds) / deriv(name, value) / integral(name, value, clamp=)",
    "rate_limit(name, value, per_second=)",
    "hysteresis(name, value, enter=, exit=) -> bool  # resists flapping",
    "latch(name, set_when, clear_when=) -> bool",
    "debounce(name, condition, ticks=) -> bool",
    "clock(name) -> seconds since first call / once(name) -> bool",
    "memo(name, value=None) -> float  # store or read one number",
    "clamp, sign, hypot, sqrt, atan2, degrees, radians, cos, sin, lerp",
)


def helper_text() -> str:
    return "\n".join(f"  ctrl.{line}" for line in HELPERS)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

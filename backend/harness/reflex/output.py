"""The command stage: continuous commands in, held keys out.

The cut this module implements is the whole reason the harness is still involved at tick
rate. **The agent owns everything between a channel and a continuous command; the harness
owns everything between a continuous command and the keyboard.** Gains, targets, feedback
laws, when to brake: the agent's. Turning -0.34 into `a` or nothing, respecting the
steering slew rate, refusing `w` and `s` together, and knowing that nitro needs straight
throttle on track with a full tank: the harness's.

That is where the hard-coded version broke, which is why the line is drawn here.
`lowlevel.py:48` records it: the action space is three steering states with a five-degree
deadband, so a gain that turns a half-pixel lane error into a seven-degree signal does not
steer proportionally — it chatters left and right every tick. Fifteen reversals a lap, and
a crash at speed 6. Nothing about that is racing knowledge; it is what happens when a
continuous command meets a discrete actuator, so the harness should own it, fix it once,
and report what it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Action, ActionName


DISCRETIZERS = ("deadband", "hysteresis", "pwm")

STEER_ENTER = 0.35
"""Command magnitude at which `deadband` and `hysteresis` first hold a steering key."""
STEER_EXIT = 0.18
"""Where `hysteresis` releases it. The gap is what stops the chatter."""
THROTTLE_DEADBAND = 0.08


@dataclass
class OutputState:
    """Discretizer memory, owned here so it snapshots with the runtime."""

    steer_key: str | None = None
    pwm_accumulator: float = 0.0
    reversals: int = 0
    steer_key_ticks: int = 0
    ticks: int = 0
    boost_refusals: int = 0
    clamped_ticks: int = 0

    def report(self) -> dict:
        return {
            "ticks": self.ticks,
            "steer_reversals": self.reversals,
            "steer_duty": round(self.steer_key_ticks / self.ticks, 3) if self.ticks else 0.0,
            "command_clamped_ticks": self.clamped_ticks,
            "boost_refusals": self.boost_refusals,
        }


@dataclass
class CommandOut:
    """The `out` argument. A controller calls `steer`, `throttle`, and maybe `boost`."""

    state: OutputState
    nitro_ready: bool = False
    nitro_active: bool = False
    on_track: bool = True
    discretizer_name: str = "hysteresis"
    steer_command: float = 0.0
    throttle_command: float = 0.0
    boost_requested: bool = False
    notes: list[str] = field(default_factory=list)

    def discretizer(self, name: str) -> None:
        if name not in DISCRETIZERS:
            raise ValueError(f"discretizer must be one of {DISCRETIZERS}; got {name!r}")
        self.discretizer_name = name

    def steer(self, command: float) -> None:
        """-1 is full left, +1 is full right. Values outside are clipped, not an error."""
        value = float(command)
        clipped = _clamp(value, -1.0, 1.0)
        if clipped != value:
            self.state.clamped_ticks += 1
            self.notes.append(f"steer clipped from {value:+.2f}")
        self.steer_command = clipped

    def throttle(self, command: float) -> None:
        """Positive accelerates, negative brakes. Braking and steering compose."""
        value = float(command)
        clipped = _clamp(value, -1.0, 1.0)
        if clipped != value:
            self.state.clamped_ticks += 1
            self.notes.append(f"throttle clipped from {value:+.2f}")
        self.throttle_command = clipped

    def boost(self, requested: bool = True) -> None:
        """Request nitro. Applied only when the engine's own rules allow it."""
        self.boost_requested = bool(requested)

    # -- the harness half --------------------------------------------------------------

    def resolve(self) -> Action:
        """Turn the commands into held keys for this tick."""
        self.state.ticks += 1
        steer_key = self._steer_key()
        if steer_key is not None and self.state.steer_key is not None and steer_key != self.state.steer_key:
            self.state.reversals += 1
        self.state.steer_key = steer_key
        if steer_key is not None:
            self.state.steer_key_ticks += 1

        keys: list[str] = []
        longitudinal = (
            "w" if self.throttle_command > THROTTLE_DEADBAND
            else "s" if self.throttle_command < -THROTTLE_DEADBAND else None
        )
        if longitudinal:
            keys.append(longitudinal)
        if steer_key:
            keys.append(steer_key)
        if self.boost_requested:
            # The engine burns nitro only on straight-line throttle, on track, with a
            # charged tank. Asking anyway is not an error, but it is reported, because a
            # controller that boosts every tick and never moves needs to know why.
            legal = (
                longitudinal == "w" and steer_key is None and self.on_track
                and (self.nitro_ready or self.nitro_active)
            )
            if legal:
                keys.append("space")
            else:
                self.state.boost_refusals += 1
                self.notes.append("boost refused: needs straight throttle on track with a full tank")
        return Action(name=_primary_name(keys), keys=keys)

    def _steer_key(self) -> str | None:
        magnitude = abs(self.steer_command)
        side = "d" if self.steer_command > 0 else "a"
        if self.discretizer_name == "deadband":
            return side if magnitude > STEER_ENTER else None
        if self.discretizer_name == "hysteresis":
            # A held key stays held while the command still asks for that side above the
            # exit threshold, and swapping sides requires passing through neutral first.
            # Without that second rule a command alternating either side of zero still
            # reverses every tick, which is the chatter this discretizer exists to remove.
            held = self.state.steer_key
            if held is not None:
                signed = self.steer_command if held == "d" else -self.steer_command
                return held if signed > STEER_EXIT else None
            return side if magnitude > STEER_ENTER else None
        # pwm: hold the key on a fraction of ticks equal to the command magnitude, so the
        # *average* steering angle tracks a fractional command the three-state action
        # space cannot express directly. Highest resolution, noisiest trace.
        self.state.pwm_accumulator += magnitude
        if self.state.pwm_accumulator >= 1.0:
            self.state.pwm_accumulator -= 1.0
            return side
        return None


def _primary_name(keys: list[str]) -> ActionName:
    """The label the replay records. Steering wins because it is what a viewer sees."""
    if "a" in keys:
        return ActionName.LEFT
    if "d" in keys:
        return ActionName.RIGHT
    if "space" in keys:
        return ActionName.NITRO
    if "w" in keys:
        return ActionName.FORWARD
    if "s" in keys:
        return ActionName.BACKWARD
    return ActionName.IDLE


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

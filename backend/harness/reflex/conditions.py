"""Cheap per-tick wake conditions: when the harness stops trusting the controller.

A condition is either a named event the runtime already computes, or a threshold
comparison on a channel. Both are evaluated with a handful of arithmetic operations per
tick, so a dozen conditions cost nothing at 10 Hz and the agent is free to declare as many
as it finds useful.

Deliberately not a general expression language. A condition needs four properties that
arbitrary Python would cost: it must be cheap, the runtime must be able to see which
channels it reads, it must report *how close it came* to firing rather than only a
boolean, and it must be checkable against a recorded window. The margin is the part worth
paying for — "your collision condition never fired because ttc bottomed out at 0.83
against your 0.8 threshold" is a complete explanation of a crash, and no boolean gives it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sense import FIELDS


NAMED_EVENTS = {
    "target_reached": "the active target was reached or passed",
    "off_track": "the car left the safe corridor",
    "unstable": "a control block's error changed sign repeatedly — the loop is oscillating",
    "geometry_changed": "local curvature, grade, or bank moved well off this episode's baseline",
    "controller_failed": "the controller raised, overran its tick budget, or emitted a non-finite command",
    "no_progress": "the car remained nearly stationary for too long while a controller was active",
    "deadline": "the directive's tick bound elapsed",
}

ALWAYS_ARMED = ("controller_failed", "off_track", "no_progress", "geometry_changed", "deadline")
"""Armed whether the agent asks for them or not.

Not a safety policy — the harness still has no idea how to drive. These are the four
situations in which the agent's *own* conditions have stopped being the right question:
its controller is broken, the car is somewhere the channels no longer describe well, the
road is not the road the controller was written for, or it has been asleep too long.
"""

BOOLEAN_FIELDS = frozenset({"on_track", "target_reached", "nitro_ready"})
NUMERIC_FIELDS = frozenset(FIELDS) - BOOLEAN_FIELDS

_COMPARISON = re.compile(
    r"^\s*(?P<abs>abs\s*\(\s*)?(?P<field>[a-z_]+)\s*(?(abs)\)\s*)"
    r"(?P<op><=|>=|<|>|==|!=)\s*(?P<value>-?\d+(?:\.\d+)?)\s*$"
)

OPERATORS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class ConditionError(ValueError):
    """A condition that cannot be parsed, with the alternatives spelled out."""


@dataclass
class WakeCondition:
    """One reason to wake the agent."""

    when: str
    for_ticks: int = 1
    hold: str | None = None
    """A controller to switch to while the agent is being consulted.

    This exists because of latency. A decision costs on the order of a dozen ticks, and
    the controller keeps driving throughout, which is fine for `target_reached` and
    useless for an imminent collision — the reasoning lands after the impact. So a
    condition may name another controller *the agent wrote* to hold the car while it
    thinks. The harness supplies no emergency behaviour of its own.
    """
    streak: int = field(default=0, repr=False)
    fired_count: int = field(default=0, repr=False)
    closest_margin: float = field(default=float("inf"), repr=False)
    rearmed: bool = field(default=True, repr=False)
    """Whether this condition may fire again.

    Conditions are edge-triggered: one fires on the transition into the situation and
    stays quiet until the situation clears. Level-triggering looks equivalent and is not.
    A car stuck off track on ice satisfies `off_track` on every one of the next thousand
    ticks, and a level-triggered condition asks for a thousand wakes — burning the whole
    call budget re-reporting a fact the agent already acted on.
    """

    # Parsed form, filled by `compile_condition`.
    event: str | None = field(default=None, repr=False)
    field_name: str | None = field(default=None, repr=False)
    operator: str | None = field(default=None, repr=False)
    threshold: float = field(default=0.0, repr=False)
    absolute: bool = field(default=False, repr=False)

    def evaluate(self, values: dict, signals: dict) -> bool:
        """True when this condition has held for `for_ticks` consecutive ticks."""
        if self.event is not None:
            active = bool(signals.get(self.event, False))
            if self.event == "off_track":
                active = not bool(values.get("on_track", True))
            elif self.event == "target_reached":
                active = bool(values.get("target_reached", False))
        else:
            raw = float(values[self.field_name])
            current = abs(raw) if self.absolute else raw
            active = OPERATORS[self.operator](current, self.threshold)
            self.closest_margin = min(self.closest_margin, abs(current - self.threshold))
        self.streak = self.streak + 1 if active else 0
        if not active:
            self.rearmed = True
            return False
        if self.streak >= self.for_ticks and self.rearmed:
            self.rearmed = False
            self.fired_count += 1
            return True
        return False

    def describe(self) -> str:
        suffix = f" for {self.for_ticks} ticks" if self.for_ticks > 1 else ""
        hold = f" (hold: {self.hold})" if self.hold else ""
        return f"{self.when}{suffix}{hold}"

    def report(self) -> dict:
        payload: dict = {"when": self.when, "fired": self.fired_count}
        if self.field_name is not None and self.closest_margin != float("inf"):
            payload["closest_margin"] = round(self.closest_margin, 3)
        return payload


def compile_condition(specification: str | dict) -> WakeCondition:
    """Parse one condition. Accepts `"ttc < 1.0"` or `{"when": ..., "for_ticks": 2}`."""
    if isinstance(specification, str):
        specification = {"when": specification}
    when = str(specification.get("when", "")).strip()
    if not when:
        raise ConditionError("a condition needs a `when`")
    condition = WakeCondition(
        when=when,
        for_ticks=max(1, int(specification.get("for_ticks", 1) or 1)),
        hold=specification.get("hold") or None,
    )
    if when in NAMED_EVENTS:
        condition.event = when
        return condition
    match = _COMPARISON.match(when)
    if not match:
        raise ConditionError(
            f"cannot parse {when!r}. A condition is either a named event "
            f"({sorted(NAMED_EVENTS)}) or a comparison like 'ttc < 1.0' or "
            f"'abs(lane) > 0.8' over a channel ({sorted(NUMERIC_FIELDS)})"
        )
    field_name = match.group("field")
    if field_name not in FIELDS:
        raise ConditionError(
            f"{field_name!r} is not a channel. Available: {sorted(NUMERIC_FIELDS)}"
        )
    if field_name in BOOLEAN_FIELDS:
        raise ConditionError(
            f"{field_name!r} is a boolean channel; compare it by name instead "
            f"(for example 'off_track' rather than 'on_track == 0')"
        )
    condition.field_name = field_name
    condition.operator = match.group("op")
    condition.threshold = float(match.group("value"))
    condition.absolute = bool(match.group("abs"))
    return condition


@dataclass
class ConditionSet:
    """The agent's conditions plus the always-armed ones, evaluated together."""

    declared: list[WakeCondition] = field(default_factory=list)
    armed: list[WakeCondition] = field(default_factory=list)

    @classmethod
    def build(cls, specifications: list) -> "ConditionSet":
        declared = [compile_condition(item) for item in specifications or []]
        chosen = {condition.when for condition in declared}
        armed = [
            WakeCondition(when=event, event=event)
            for event in ALWAYS_ARMED if event not in chosen
        ]
        return cls(declared=declared, armed=armed)

    @property
    def all(self) -> list[WakeCondition]:
        return [*self.declared, *self.armed]

    def evaluate(self, values: dict, signals: dict) -> list[WakeCondition]:
        """Every condition that fired this tick, declared ones first."""
        return [condition for condition in self.all if condition.evaluate(values, signals)]

    def describe(self) -> list[str]:
        return [condition.describe() for condition in self.declared] + [
            f"{condition.when} (always armed)" for condition in self.armed
        ]

    def report(self) -> list[dict]:
        return [condition.report() for condition in self.all if condition.fired_count or (
            condition.closest_margin != float("inf")
        )]

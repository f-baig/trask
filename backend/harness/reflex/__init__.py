"""Reflex harness: the player writes the tick-rate controller, the harness runs it.

See `docs/reflex-harness.md`. The player agent chooses a target, writes a sandboxed
micro-controller, and declares the conditions that should wake it. This package owns
the parts that are not driving: normalized channels (`sense`), instrumented control
helpers (`blocks`), the command-to-keyboard stage (`output`), an install gate
(`sandbox`), cheap wake conditions (`conditions`), a wake payload chosen by cause
(`perception`), and the per-tick runtime that binds them (`runtime`).

No module here contains a driving policy. `blocks` may not import `sense`, which is
what keeps the helper library from growing into a controller.
"""

from .blocks import ControlBlocks
from .conditions import ALWAYS_ARMED, ConditionSet, WakeCondition
from .output import CommandOut, DISCRETIZERS
from .runtime import ControllerRecord, ReflexRuntime, TryReport
from .sandbox import GateReport, InstallError, compile_controller
from .sense import FIELDS, SenseMemory, SenseView, Target, compute_sense

__all__ = [
    "ALWAYS_ARMED",
    "CommandOut",
    "ConditionSet",
    "ControlBlocks",
    "ControllerRecord",
    "DISCRETIZERS",
    "FIELDS",
    "GateReport",
    "InstallError",
    "ReflexRuntime",
    "SenseMemory",
    "SenseView",
    "Target",
    "TryReport",
    "WakeCondition",
    "compile_controller",
    "compute_sense",
]

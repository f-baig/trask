"""Compile and gate an agent-written controller before it is allowed to drive.

The threat model is mistakes, not malice: this is a local research harness running code
its own operator's agent wrote. The gate exists so a bad controller cannot corrupt an
experiment or produce an unreadable replay, not to contain an adversary.

What it is actually for is turning runtime failures into install-time messages. An agent
that is told `line 7: import is not available inside a controller; use ctrl.sqrt` fixes it
in one turn. An agent that gets a `NameError` at tick 300 has to reason backwards from a
crashed race, and usually re-installs something equally broken.

Loops are banned outright. That is not only a sandbox convenience: with no loops, a
controller's execution time is bounded by its own source length, which is what makes the
per-tick budget a real guarantee rather than a hope. A controller that wants iteration is
asking for something the tick budget cannot afford anyway.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from time import perf_counter

from .blocks import ControlBlocks
from .output import CommandOut, OutputState
from .sense import FIELDS, SenseView


class InstallError(ValueError):
    """A controller that cannot be installed, with a message the agent can act on."""


ALLOWED_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round,
    "float": float, "int": int, "bool": bool,
}

ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Load, ast.Store,
    ast.Expr, ast.Assign, ast.AugAssign, ast.Return, ast.If, ast.Pass,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Attribute,
    ast.Name, ast.Constant, ast.IfExp, ast.Tuple, ast.List, ast.Dict,
    ast.keyword, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

FORBIDDEN_MESSAGES = {
    ast.Import: "import is not available inside a controller; the arithmetic you need is on ctrl",
    ast.ImportFrom: "import is not available inside a controller; the arithmetic you need is on ctrl",
    ast.While: "loops are not available inside a controller; it must return within one tick",
    ast.For: "loops are not available inside a controller; it must return within one tick",
    ast.Lambda: "lambda is not available; write the expression inline",
    ast.ClassDef: "class definitions are not available inside a controller",
    ast.Global: "global is not available; keep state in ctrl.memo or a ctrl block",
    ast.Nonlocal: "nonlocal is not available; keep state in ctrl.memo or a ctrl block",
    ast.Try: "try/except is not available; the runtime reports failures for you",
    ast.Raise: "raise is not available; return a safe command instead",
    ast.With: "with is not available inside a controller",
    ast.Delete: "del is not available inside a controller",
    ast.ListComp: "comprehensions are not available; a controller handles one tick of scalars",
    ast.DictComp: "comprehensions are not available; a controller handles one tick of scalars",
    ast.SetComp: "comprehensions are not available; a controller handles one tick of scalars",
    ast.GeneratorExp: "generators are not available inside a controller",
    ast.Await: "await is not available inside a controller",
    ast.Yield: "yield is not available inside a controller",
    ast.Subscript: "indexing is not available; every channel is a scalar on sense",
}

CTRL_METHODS = frozenset({
    "pid", "stanley", "pursuit", "ewma", "deriv", "integral", "rate_limit",
    "hysteresis", "latch", "debounce", "clock", "once", "memo",
    "clamp", "sign", "hypot", "sqrt", "atan2", "degrees", "radians", "cos", "sin", "lerp",
})
OUT_METHODS = frozenset({"steer", "throttle", "boost", "discretizer"})
ARGUMENT_NAMES = ("sense", "ctrl", "out")


@dataclass
class GateReport:
    """What the gate found, in the form the agent is shown."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fuzz_samples: int = 0
    max_tick_ms: float = 0.0
    mean_tick_ms: float = 0.0
    mirror_ok: bool | None = None

    def as_dict(self) -> dict:
        payload: dict[str, object] = {"ok": self.ok}
        if self.errors:
            payload["errors"] = self.errors
        if self.warnings:
            payload["warnings"] = self.warnings
        if self.ok:
            payload.update({
                "fuzz_samples": self.fuzz_samples,
                "max_tick_ms": round(self.max_tick_ms, 3),
                "mean_tick_ms": round(self.mean_tick_ms, 3),
                "mirror_test": (
                    "passed" if self.mirror_ok else "not applicable" if self.mirror_ok is None
                    else "FAILED — steering does not mirror when the track mirrors, which is "
                         "usually a sign error"
                ),
            })
        return payload


@dataclass
class CompiledController:
    name: str
    version: int
    source: str
    reads: frozenset[str]
    params: dict[str, float]
    safe_action: dict
    function: object
    parent: str | None = None

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"

    def run(self, sense: SenseView, blocks: ControlBlocks, out: CommandOut) -> None:
        self.function(sense, blocks, out)


def compile_controller(
    *, name: str, source: str, reads: list[str], params: dict[str, float],
    safe_action: dict | None = None, version: int = 1, parent: str | None = None,
) -> CompiledController:
    """Parse, whitelist, and bind a controller. Raises `InstallError` with a reason."""
    unknown = [field_name for field_name in reads if field_name not in FIELDS]
    if unknown:
        raise InstallError(
            f"reads contains {unknown}, which are not channels. Available: {sorted(FIELDS)}"
        )
    if not reads:
        raise InstallError("reads is empty; declare the channels the controller uses")
    for key, value in (params or {}).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InstallError(f"param {key!r} must be a number, got {value!r}")

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise InstallError(f"line {error.lineno}: {error.msg}") from None

    # Forbidden constructs first, then shape. A module-level `import` is both an import
    # and a second top-level statement, and "import is not available; use ctrl.sqrt" is
    # the message that gets fixed in one turn.
    _scan_forbidden(tree)
    _check_shape(tree)
    _Whitelist(frozenset(reads), frozenset(params or {})).visit(tree)

    globals_dict: dict = {"__builtins__": dict(ALLOWED_BUILTINS)}
    exec(compile(tree, filename=f"<controller {name}>", mode="exec"), globals_dict)  # noqa: S102
    function = globals_dict.get("control")
    if function is None:
        raise InstallError("the controller must define a function named `control`")
    return CompiledController(
        name=name, version=version, source=source, reads=frozenset(reads),
        params=dict(params or {}), safe_action=dict(safe_action or {}),
        function=function, parent=parent,
    )


def gate_controller(
    controller: CompiledController, *, samples: list[dict] | None = None,
    budget_ms: float = 2.0, fuzz_count: int = 256, seed: int = 17,
) -> GateReport:
    """Run the controller against synthetic and recorded channel rows before it drives.

    Three checks. It must not raise, it must not emit a non-finite command, and it must
    fit the tick budget. Then a mirror test: flip every lateral channel and the steering
    command should flip with it. A controller that fails the mirror test usually has a
    sign error, which is the most common way a generated control law fails, and catching
    it costs microseconds instead of a race.
    """
    report = GateReport(ok=True)
    vectors = _fuzz_vectors(fuzz_count, seed) + list(samples or [])
    durations: list[float] = []
    steering: dict[int, float] = {}

    for position, values in enumerate(vectors):
        try:
            command, elapsed = _evaluate(controller, values)
        except Exception as error:  # noqa: BLE001 - any failure is an install failure
            report.ok = False
            report.errors.append(
                f"raised {type(error).__name__}: {error} on channels "
                f"{_brief(values, controller.reads)}"
            )
            break
        durations.append(elapsed)
        if not math.isfinite(command["steer"]) or not math.isfinite(command["throttle"]):
            report.ok = False
            report.errors.append(
                f"produced a non-finite command on channels {_brief(values, controller.reads)}"
            )
            break
        steering[position] = command["steer"]

    if durations:
        ordered = sorted(durations)
        median_ms = ordered[len(ordered) // 2] * 1000
        report.fuzz_samples = len(durations)
        report.max_tick_ms = ordered[-1] * 1000
        report.mean_tick_ms = median_ms
        # Judged on the median, not the maximum. A single tick's wall-clock reading measures
        # the operating system as much as the controller: on a busy machine this same
        # controller timed 0.006 ms typically and 22 ms once, so a max-based gate rejected a
        # third of perfectly good installs. A genuinely slow controller is slow every tick and
        # moves the median; a descheduled process moves only the tail.
        if median_ms > budget_ms:
            report.ok = False
            report.errors.append(
                f"median tick cost {median_ms:.2f} ms against a {budget_ms:.1f} ms budget"
            )
        elif report.max_tick_ms > budget_ms * 20:
            report.warnings.append(
                f"one sampled tick took {report.max_tick_ms:.1f} ms against a median of "
                f"{median_ms:.3f} ms — almost certainly the host, but worth knowing"
            )

    if report.ok:
        report.mirror_ok = _mirror_test(controller, vectors[:64])
        if report.mirror_ok is False:
            report.warnings.append(
                "steering does not mirror when the lateral channels are mirrored; check "
                "the sign of the cross-track term"
            )
        if not any(abs(value) > 1e-9 for value in steering.values()):
            report.warnings.append("steer was zero on every sampled channel row")
    return report


def _evaluate(controller: CompiledController, values: dict) -> tuple[dict, float]:
    """One isolated tick, with fresh block state, returning the raw commands."""
    blocks = ControlBlocks(controller.params)
    blocks.begin_tick(0.1)
    out = CommandOut(state=OutputState(), nitro_ready=bool(values.get("nitro_ready", False)))
    started = perf_counter()
    controller.run(SenseView(values, controller.reads), blocks, out)
    elapsed = perf_counter() - started
    return {"steer": out.steer_command, "throttle": out.throttle_command}, elapsed


MIRRORED = (
    "lane", "lane_rate", "heading_error", "curvature", "bank", "yaw_rate",
    "steer_angle", "hazard_bearing", "target_lane", "vision_track_offset",
    "vision_track_heading", "vision_bend_ahead", "vision_recovery_direction",
)


def _mirror_test(controller: CompiledController, vectors: list[dict]) -> bool | None:
    """Mirror the track and the steering command should mirror too."""
    if not any(field_name in controller.reads for field_name in MIRRORED):
        return None
    for values in vectors:
        mirrored = dict(values)
        for field_name in MIRRORED:
            mirrored[field_name] = -values[field_name]
        left, _ = _evaluate(controller, values)
        right, _ = _evaluate(controller, mirrored)
        if abs(left["steer"] + right["steer"]) > 1e-6 + 1e-3 * abs(left["steer"]):
            return False
    return True


def _fuzz_vectors(count: int, seed: int) -> list[dict]:
    """Deterministic pseudo-random channel rows, plus the extremes.

    A tiny linear congruential generator rather than `random`, so an install report is
    reproducible without touching global random state that a replay depends on.
    """
    state = seed or 1
    def next_unit() -> float:
        nonlocal state
        state = (state * 1103515245 + 12345) % (2**31)
        return state / (2**31)

    ranges = {
        "lane": (-2.0, 2.0), "lane_rate": (-4.0, 4.0), "heading_error": (-180.0, 180.0),
            "speed": (0.0, 12.0), "speed_limit": (0.05, 12.0),
            "curvature": (-0.6, 0.6), "grip_used": (0.0, 2.0),
        "grip_headroom": (0.0, 1.0), "free_ahead": (0.0, 12.0), "ttc": (0.0, 99.0),
        "hazard_bearing": (-70.0, 70.0), "half_width": (0.3, 4.0), "slip": (-90.0, 90.0),
        "yaw_rate": (-200.0, 200.0), "steer_angle": (-1.0, 1.0), "grade": (-20.0, 20.0),
        "bank": (-22.0, 22.0), "target_lane": (-1.0, 1.0), "target_error": (0.0, 40.0),
        "tick": (0.0, 5000.0),
        "vision_lane": (-2.0, 2.0), "vision_turn": (-2.0, 2.0), "vision_flow": (0.0, 30.0),
        "vision_center_near": (-2.0, 2.0), "vision_center_mid": (-2.0, 2.0), "vision_center_far": (-2.0, 2.0),
        "vision_turn_ahead": (-2.0, 2.0), "vision_turn_severity": (0.0, 2.0), "vision_lookahead_depth": (0.0, 1.0),
        "vision_road_width": (0.0, 1.0), "vision_left_gap": (0.0, 1.0), "vision_right_gap": (0.0, 1.0),
        "vision_confidence": (0.0, 1.0), "vision_flow_rotation": (-30.0, 30.0),
        "vision_recovery_direction": (-1.0, 1.0), "vision_center_rate": (-1.0, 1.0), "vision_turn_delta": (-1.0, 1.0), "vision_edge_closing_rate": (-1.0, 1.0), "vision_confidence_trend": (-1.0, 1.0),
        "vision_track_offset": (-2.0, 2.0), "vision_track_heading": (-2.0, 2.0),
        "vision_bend_ahead": (-2.0, 2.0), "vision_bend_severity": (0.0, 2.0),
        "vision_visible_depth": (0.0, 1.0),
        "vision_road_horizon": (0.0, 1.0), "vision_horizon_shift": (-1.0, 1.0),
        "vision_crest_risk": (0.0, 1.0),
    }
    vectors: list[dict] = []
    for _ in range(max(0, count - 3)):
        row: dict = {}
        for field_name in FIELDS:
            if field_name in {"on_track", "target_reached", "nitro_ready", "vision_road_visible", "vision_road_lost", "vision_ego_road_contact", "vision_road_contact"}:
                row[field_name] = next_unit() > 0.5
            else:
                low, high = ranges[field_name]
                row[field_name] = low + (high - low) * next_unit()
        vectors.append(row)
    # The extremes matter more than the middle: a divide-by-zero hides at speed 0.
    for corner in (0.0, 1.0, -1.0):
        row = {}
        for field_name in FIELDS:
            if field_name in {"on_track", "target_reached", "nitro_ready", "vision_road_visible", "vision_road_lost", "vision_ego_road_contact", "vision_road_contact"}:
                row[field_name] = corner > 0
            else:
                low, high = ranges[field_name]
                row[field_name] = 0.0 if corner == 0.0 else (high if corner > 0 else low)
        vectors.append(row)
    return vectors


def _brief(values: dict, reads: frozenset[str]) -> dict:
    return {name: round(values[name], 3) if isinstance(values[name], float) else values[name]
            for name in sorted(reads)}


def _scan_forbidden(tree: ast.Module) -> None:
    """Report the specific banned construct before anything more generic."""
    for node in ast.walk(tree):
        for forbidden, message in FORBIDDEN_MESSAGES.items():
            if isinstance(node, forbidden):
                raise InstallError(f"line {getattr(node, 'lineno', 0)}: {message}")


def _check_shape(tree: ast.Module) -> None:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise InstallError(
            "a controller is exactly one function: def control(sense, ctrl, out)"
        )
    function = tree.body[0]
    if function.name != "control":
        raise InstallError(f"the function must be named `control`, not {function.name!r}")
    argument_names = [argument.arg for argument in function.args.args]
    if argument_names != list(ARGUMENT_NAMES):
        raise InstallError(
            f"control must take exactly {ARGUMENT_NAMES}; got {tuple(argument_names)}"
        )
    if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise InstallError("control takes no *args, **kwargs, or keyword-only arguments")
    if function.decorator_list:
        raise InstallError("decorators are not available inside a controller")


class _Whitelist(ast.NodeVisitor):
    def __init__(self, reads: frozenset[str], params: frozenset[str]):
        self.reads = reads
        self.params = params
        self.locals: set[str] = set()
        self.inner: set[int] = set()
        """Attribute nodes already covered by an enclosing one.

        `ctrl.p.kp` is two nested Attribute nodes, and the inner `ctrl.p` is not a
        complete path. Validating it on its own rejects every parameter read.
        """

    def visit(self, node: ast.AST):
        for forbidden, message in FORBIDDEN_MESSAGES.items():
            if isinstance(node, forbidden):
                raise InstallError(f"line {getattr(node, 'lineno', 0)}: {message}")
        if not isinstance(node, ALLOWED_NODES):
            raise InstallError(
                f"line {getattr(node, 'lineno', 0)}: {type(node).__name__} is not "
                "available inside a controller"
            )
        return super().visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise InstallError(f"line {node.lineno}: private attributes are not available")
        if id(node) in self.inner:
            return self.generic_visit(node)
        root, path = _attribute_path(node)
        current = node.value
        while isinstance(current, ast.Attribute):
            self.inner.add(id(current))
            current = current.value
        if root not in ARGUMENT_NAMES:
            raise InstallError(
                f"line {node.lineno}: attribute access is only available on "
                f"{ARGUMENT_NAMES}, not on {root!r}"
            )
        if root == "sense":
            if len(path) != 1:
                raise InstallError(f"line {node.lineno}: sense fields are scalars, not objects")
            if path[0] not in FIELDS:
                raise InstallError(
                    f"line {node.lineno}: sense.{path[0]} is not a channel. "
                    f"Available: {sorted(FIELDS)}"
                )
            if path[0] not in self.reads:
                raise InstallError(
                    f"line {node.lineno}: sense.{path[0]} is not in this controller's "
                    f"reads. Add it to reads, or use one of {sorted(self.reads)}"
                )
        elif root == "ctrl":
            if path[0] == "p":
                if len(path) != 2:
                    raise InstallError(f"line {node.lineno}: use ctrl.p.<param>")
                if path[1] not in self.params:
                    raise InstallError(
                        f"line {node.lineno}: ctrl.p.{path[1]} is not a declared param. "
                        f"Declared: {sorted(self.params)}"
                    )
            elif len(path) != 1 or path[0] not in CTRL_METHODS:
                raise InstallError(
                    f"line {node.lineno}: ctrl.{'.'.join(path)} is not a helper. "
                    f"Available: {sorted(CTRL_METHODS)}"
                )
        elif len(path) != 1 or path[0] not in OUT_METHODS:
            raise InstallError(
                f"line {node.lineno}: out.{'.'.join(path)} is not available. "
                f"Available: {sorted(OUT_METHODS)}"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise InstallError(f"line {node.lineno}: names may not start with an underscore")
        if isinstance(node.ctx, ast.Store):
            self.locals.add(node.id)
            return
        if node.id in ARGUMENT_NAMES or node.id in self.locals or node.id in ALLOWED_BUILTINS:
            return
        raise InstallError(
            f"line {node.lineno}: {node.id!r} is not defined. A controller may use "
            f"{ARGUMENT_NAMES}, its own local variables, and {sorted(ALLOWED_BUILTINS)}"
        )

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_BUILTINS:
                raise InstallError(
                    f"line {node.lineno}: {node.func.id}() is not available; "
                    f"callable builtins are {sorted(ALLOWED_BUILTINS)}"
                )
        elif isinstance(node.func, ast.Attribute):
            root, _ = _attribute_path(node.func)
            if root == "sense":
                raise InstallError(
                    f"line {node.lineno}: sense fields are values, not methods"
                )
        else:
            raise InstallError(f"line {node.lineno}: only named calls are available")
        self.generic_visit(node)


def _attribute_path(node: ast.Attribute) -> tuple[str, list[str]]:
    path = [node.attr]
    current: ast.AST = node.value
    while isinstance(current, ast.Attribute):
        path.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise InstallError(
            f"line {getattr(node, 'lineno', 0)}: attribute access must start from "
            f"{ARGUMENT_NAMES}"
        )
    return current.id, list(reversed(path))

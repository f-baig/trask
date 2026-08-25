"""The per-tick runtime: sense, run the controller, decide whether to wake the agent.

This is the loop that replaces `LocalIntentController`. The difference is not the shape —
both close a loop at tick rate from local channels — it is who wrote the law. Here the
harness computes channels, hands them to code the agent installed, turns the returned
commands into keys, and watches for the conditions the agent declared plus four it always
watches for. It contains no control law of its own, and if no controller is installed the
car does not move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

from ..models import Action, ActionName
from .blocks import ControlBlocks, Params
from .conditions import ConditionSet, WakeCondition
from .output import CommandOut, OutputState
from .sandbox import CompiledController, GateReport, InstallError, compile_controller, gate_controller
from .sense import SenseMemory, SenseView, Target, anchor_target, compute_sense


DEFAULT_DEADLINE_TICKS = 80
MAX_DIRECTIVE_TICKS = 80
NO_PROGRESS_TICKS = 25
DEFAULT_TICK_BUDGET_MS = 2.0
UNSTABLE_SIGN_CHANGES = 6
"""Sign changes within one block's 20-tick window before `unstable` fires.

Six is three full oscillations in two seconds. The tuned hard-coded controller logged
fifteen reversals a lap at the gain that crashed it and none at the gain that worked, so
the useful threshold sits well below the failure and above ordinary corner-to-corner
correction.
"""
OVERRUN_TICKS_BEFORE_FAILURE = 5
"""Consecutive over-budget ticks before the controller is blamed rather than the host."""
RECORDER_CAPACITY = 600
DIAGNOSTIC_TRACE_CAPACITY = 4_000
MIN_WAKE_GAP = 3
"""Ticks before a non-critical condition may wake the agent again."""


@dataclass
class ControllerRecord:
    """One installed version, plus its live block state."""

    controller: CompiledController
    blocks: ControlBlocks
    gate: GateReport
    ticks_driven: int = 0

    @property
    def label(self) -> str:
        return self.controller.label


@dataclass
class TryReport:
    """What happened when a candidate drove a fork of the current situation."""

    controller: str
    is_active: bool = False
    ticks: int = 0
    terminated: bool = False
    succeeded: bool = False
    reason: str | None = None
    off_track_ticks: int = 0
    mean_speed: float = 0.0
    max_grip_used: float = 0.0
    checkpoints_reached: int = 0
    fired: list[str] = field(default_factory=list)
    worst_block: str | None = None
    worst_sign_changes: int = 0
    failure: str | None = None

    # Scoring. A rehearsal that finishes yields a projected race time measured from the
    # start of the episode, which is the only figure comparable across rehearsals taken at
    # different points in the race.
    control_hz: int = 10
    projected_finish_tick: int | None = None
    best_finish_tick: int | None = None
    budget_remaining: int = 0
    visual: dict = field(default_factory=dict)

    @property
    def is_new_best(self) -> bool:
        return (
            self.projected_finish_tick is not None
            and (self.best_finish_tick is None or self.projected_finish_tick < self.best_finish_tick)
        )

    def _score(self) -> dict:
        """The comparison, phrased so it cannot be read as anything but "lower is better".

        Finishing is easy to treat as success and stop there, so the score never reports a
        finish without also reporting the gap to the best finish so far. Both ticks and
        seconds, because a tick count is a harness unit and a second is not.
        """
        if self.projected_finish_tick is None:
            return {
                "lap_time": None,
                "note": (
                    f"did not finish inside {self.ticks} rehearsed ticks, so there is no lap "
                    "time to compare. Rehearse for more ticks, or this controller is too "
                    "slow or stuck."
                ),
            }
        seconds = self.projected_finish_tick / max(1, self.control_hz)
        score: dict = {
            "lap_time_ticks": self.projected_finish_tick,
            "lap_time_seconds": round(seconds, 1),
            "objective": "finish in the FEWEST ticks. Lower is better.",
        }
        if self.best_finish_tick is None:
            score["standing"] = "first finish of the episode; this is now your best"
        elif self.is_new_best:
            gain = self.best_finish_tick - self.projected_finish_tick
            score["standing"] = (
                f"NEW BEST — {gain} ticks ({gain / max(1, self.control_hz):.1f}s) faster "
                f"than your previous best of {self.best_finish_tick}"
            )
        else:
            loss = self.projected_finish_tick - self.best_finish_tick
            score["standing"] = (
                f"SLOWER than your best of {self.best_finish_tick} ticks, by {loss} ticks "
                f"({loss / max(1, self.control_hz):.1f}s). Keep the faster settings."
            )
        return score

    def as_dict(self) -> dict:
        return {
            "controller": self.controller,
            "is_active": self.is_active,
            **({} if self.is_active else {
                "warning": "this controller is NOT the one driving; call "
                           "activate_controller to make it drive",
            }),
            "score": self._score(),
            "rehearsals_left_this_wake": self.budget_remaining,
            "ticks_survived": self.ticks,
            "terminated": self.terminated,
            "finished": self.succeeded,
            "reason": self.reason,
            "off_track_ticks": self.off_track_ticks,
            "mean_speed_cl_s": round(self.mean_speed, 2),
            "max_grip_used": round(self.max_grip_used, 3),
            "checkpoints_reached": self.checkpoints_reached,
            "conditions_fired": self.fired[:12],
            "worst_oscillation": (
                f"{self.worst_block}: {self.worst_sign_changes} sign changes"
                if self.worst_block else "none"
            ),
            **({"failure": self.failure} if self.failure else {}),
        }


class _Novelty:
    """Fast-versus-slow drift on one channel, for `geometry_changed`.

    A fixed curvature threshold cannot answer "is this substantially different
    geometry", because a hairpin on a circuit of hairpins is not news. Comparing a
    half-second average against a six-second one, scaled by the channel's own spread,
    is relative to what this episode has actually been like — and it self-resets, so
    sustained new geometry stops firing once it becomes the norm.
    """

    __slots__ = ("fast", "slow", "variance", "samples", "floor")

    def __init__(self, floor: float):
        self.fast = self.slow = self.variance = 0.0
        self.samples = 0
        self.floor = floor

    def update(self, value: float, dt: float) -> float:
        if self.samples == 0:
            self.fast = self.slow = value
        alpha_fast = min(1.0, dt / 0.5)
        alpha_slow = min(1.0, dt / 6.0)
        self.fast += alpha_fast * (value - self.fast)
        self.slow += alpha_slow * (value - self.slow)
        deviation = (value - self.slow) ** 2
        self.variance += alpha_slow * (deviation - self.variance)
        self.samples += 1
        if self.samples < 30:
            return 0.0
        spread = 3.0 * math.sqrt(max(0.0, self.variance)) + self.floor
        return abs(self.fast - self.slow) / spread


class ReflexRuntime:
    """Owns the controllers, the channels, the conditions, and the recorder."""

    def __init__(
        self, scene, *, tick_budget_ms: float = DEFAULT_TICK_BUDGET_MS,
        vision_only: bool = False, visual_mode: str | None = None,
    ):
        self.scene = scene
        self.vision_only = vision_only
        self.visual_mode = visual_mode or (
            "3d" if vision_only and getattr(scene, "elevation", None) is not None
            and not scene.elevation.is_flat else "2d"
        )
        self.visible_fields = (
            # Deliberately small controller ABI. Extra pixel measurements remain
            # harness diagnostics, but handing every derivative to the authoring
            # model made it write reactive, contradictory control laws.
            self._visual_fields() | {"speed"}
            if vision_only else None
        )
        self.vision = None
        self.latest_frame = None
        if vision_only:
            if self.visual_mode == "3d":
                from .visual_3d import PerspectiveVisionSense
                self.vision = PerspectiveVisionSense()
            else:
                from .visual_2d import ConeVisionSense
                self.vision = ConeVisionSense()
        self.dt = 1.0 / scene.dynamics.control_hz
        self.tick_budget_ms = tick_budget_ms
        self.memory = SenseMemory()
        self.controllers: dict[str, ControllerRecord] = {}
        self.active: str | None = None
        self.holding: str | None = None
        """Set while a `hold` controller covers a pending wake."""
        self.output_state = OutputState()
        self.target: Target | None = None
        self.conditions = ConditionSet.build([])
        self.deadline_ticks = DEFAULT_DEADLINE_TICKS
        self.ticks_since_wake = 0
        self.pending: list[WakeCondition] = []
        self.recorder: list[dict] = []
        # Kept separately from the compact model-facing recorder.  This is evaluator-only
        # evidence: enough to explain an early miss even when a later timeout would evict it
        # from `recorder`.
        self.diagnostic_trace: list[dict] = []
        self.firings: dict[str, int] = {}
        """Conditions that asked for a wake, whether or not one happened."""
        self.wake_causes: dict[str, int] = {}
        """Conditions that actually cost a model call. Only the episode can know this: a
        request can be refused by the wake budget, and counting requests as wakes made a
        car stuck off track look like it had woken the agent a thousand times."""
        self.failures: list[str] = []
        self.last_failure: str | None = None
        self.last_sense: dict = {}
        self.last_notes: list[str] = []
        self.controller_ticks = 0
        self.idle_ticks = 0
        self.no_progress_ticks = 0
        self.fatal: str | None = None
        self.overrun_streak = 0
        self.best_finish_tick: int | None = None
        """Fewest ticks any rehearsal has projected for finishing the race.

        The agent's own record, and nothing else. Comparing against the fixture
        controller's time would hand it the answer to how fast the circuit can be driven,
        which is the measurement, not the input.
        """
        self.rehearsal_log: list[dict] = []
        self.rehearsal_budget = 5
        self.rehearsals_used = 0
        self.visual_calibrations: list[dict] = []
        self._novelty = {
            "curvature": _Novelty(0.02), "grade": _Novelty(1.0), "bank": _Novelty(1.0),
        }
        self._version_counter: dict[str, int] = {}

    def _visual_fields(self) -> frozenset[str]:
        if self.visual_mode == "3d":
            from .visual_3d import FIELDS
        else:
            from .visual_2d import FIELDS
        return frozenset(FIELDS)

    # -- installation ------------------------------------------------------------------

    def install(
        self, *, name: str, source: str, reads: list[str], params: dict | None = None,
        safe_action: dict | None = None, activate: bool = True,
    ) -> dict:
        """Compile, gate, and (by default) activate a controller.

        A controller that fails the gate is not installed at all, so a bad install leaves
        the previous one driving rather than stopping the car.
        """
        forbidden = sorted(set(reads) - self.visible_fields) if self.visible_fields is not None else []
        if forbidden:
            return {"installed": False, "gate": {"ok": False, "errors": [
                f"vision-only mode permits only {sorted(self.visible_fields)}; attempted {forbidden}"
            ]}}
        version = self._version_counter.get(name, 0) + 1
        parent = self.controllers[name].label if name in self.controllers else None
        try:
            compiled = compile_controller(
                name=name, source=source, reads=list(reads), params=dict(params or {}),
                safe_action=safe_action, version=version, parent=parent,
            )
        except InstallError as error:
            return {"installed": False, "gate": {"ok": False, "errors": [str(error)]}}
        report = gate_controller(
            compiled, samples=self._recent_channel_rows(), budget_ms=self.tick_budget_ms,
        )
        if not report.ok:
            return {"installed": False, "gate": report.as_dict()}
        self._version_counter[name] = version
        self.controllers[name] = ControllerRecord(
            controller=compiled, blocks=ControlBlocks(compiled.params), gate=report,
        )
        if activate:
            self.active = name
        return {
            "installed": True, "controller": compiled.label,
            "parent": parent, "active": self.active == name, "gate": report.as_dict(),
        }

    def activate(self, name: str) -> dict:
        """Make an already-installed controller the one that drives.

        Needed because rehearsing with `activate: false` and then never activating is an
        easy and expensive mistake: the good controller passes its rehearsal while the bad
        one keeps driving. Re-sending the whole source just to flip a flag is worse.
        """
        record = self.controllers.get(name)
        if record is None:
            return {
                "activated": False,
                "error": f"no controller named {name!r}; installed: {sorted(self.controllers)}",
            }
        self.active = name
        return {"activated": True, "controller": record.label}

    def patch_params(self, name: str, params: dict) -> dict:
        """Retune without recompiling. Block state is kept, so integrators stay continuous."""
        record = self.controllers.get(name)
        if record is None:
            return {"patched": False, "error": f"no controller named {name!r}"}
        unknown = sorted(set(params) - set(record.controller.params))
        if unknown:
            return {
                "patched": False,
                "error": f"{unknown} are not declared params of {name!r}; "
                         f"declared: {sorted(record.controller.params)}",
            }
        merged = {**record.controller.params, **{k: float(v) for k, v in params.items()}}
        record.controller.params = merged
        record.blocks.p = Params(merged)
        return {"patched": True, "controller": record.label, "params": merged}

    def set_target(self, specification: dict, world=None, observation=None) -> dict:
        if self.vision_only:
            return {"error": "set_target is unavailable in vision-only mode; use the forward-cone screenshot and visual wake conditions."}
        kind = str(specification.get("kind", "hold_lane"))
        if kind not in {"hold_lane", "lane_point"}:
            return {"error": "target kind must be 'hold_lane' or 'lane_point'"}
        target = Target(
            kind=kind,
            lane=float(specification.get("lane", 0.0)),
            ahead_cl=float(specification.get("ahead_cl", 8.0)),
            tolerance_cl=float(specification.get("tolerance_cl", 1.5)),
            note=str(specification.get("note", "")),
        )
        if world is not None and observation is not None:
            target = anchor_target(target, world, observation, self.memory)
        self.target = target
        return {"target": target.as_dict()}

    def set_conditions(self, specifications: list, deadline_ticks: int | None = None) -> dict:
        try:
            self.conditions = ConditionSet.build(specifications)
        except ValueError as error:
            return {"accepted": False, "error": str(error)}
        if deadline_ticks is not None:
            self.deadline_ticks = min(MAX_DIRECTIVE_TICKS, max(1, int(deadline_ticks)))
        return {
            "accepted": True, "conditions": self.conditions.describe(),
            "deadline_ticks": self.deadline_ticks,
        }

    # -- the tick ----------------------------------------------------------------------

    def observe_visual(self, world):
        if self.visual_mode == "3d":
            # The player sees the real 3D driving camera on an elevated scene.  The same
            # screenshot-only adapter below extracts its contract; no surface/physics state
            # crosses this boundary.
            from ..view3d import ViewMode, render_policy_view
            raw = render_policy_view(world, mode=ViewMode.FIRST_PERSON)
        else:
            from ..vision import render_racing_forward_cone
            raw = render_racing_forward_cone(world)
        self.latest_frame = raw
        values = self.vision.update(raw)
        if self.vision_only:
            observation = world.observe()
            if self.visual_mode == "3d":
                # The perspective contract and its existing skill library use
                # physical speed. Keep generated controllers on that same ABI.
                values["speed"] = observation.speed
            else:
                car_length_px = world.scene.dynamics.vehicle.length_m * world.scene.dynamics.pixels_per_meter
                values["speed"] = observation.speed * world.scene.dynamics.control_hz / car_length_px
        return values

    def model_frame(self):
        """A wake screenshot with an optical-flow overlay, never a telemetry overlay."""
        if not self.vision_only or self.latest_frame is None:
            return self.latest_frame
        from ..motion import MotionOverlay
        if not hasattr(self, "_wake_motion_overlay"):
            self._wake_motion_overlay = MotionOverlay(color_base=True)
            self._wake_motion_tick = self.last_sense.get("tick", 1)
        interval = max(1, self.last_sense.get("tick", 1) - self._wake_motion_tick)
        self._wake_motion_tick = self.last_sense.get("tick", 1)
        return self._wake_motion_overlay.annotate(self.latest_frame, interval_ticks=interval)

    def visual_road_profile(self) -> dict:
        """Return the current screenshot's road profile without interpreting it as a route."""
        if not self.vision_only:
            return {"error": "visual road inspection is available only to a vision-only controller."}
        return {
            "profile": self.last_sense.get("vision_profile", []),
            "bend": {
                "image_side": "right" if self.last_sense.get("vision_turn_ahead", 0.0) > 0 else "left",
                "severity": round(float(self.last_sense.get("vision_turn_severity", 0.0)), 4),
                "visible_depth": round(float(self.last_sense.get("vision_lookahead_depth", 0.0)), 4),
            },
            "mode": self.visual_mode,
            "note": "pixel-derived road centreline in image coordinates; it is not a world path or steering command.",
        }

    def calibrate_perspective_controls(self, world, ticks: int = 8) -> dict:
        """Fork and measure small left/right/straight control pulses.

        This reports camera-derived fields plus the permanently exposed physical speed.
        It is episode-local evidence, not a stored vehicle model: friction, drag, or
        camera changes require a fresh probe from the current visual state.
        """
        if not self.vision_only or self.visual_mode != "3d":
            return {"error": "perspective calibration is available only in 3D vision-only mode."}
        ticks = max(2, min(12, int(ticks)))
        actions = {
            "left": Action(name=ActionName.LEFT, keys=["w", "a"]),
            "straight": Action(name=ActionName.FORWARD, keys=["w"]),
            "right": Action(name=ActionName.RIGHT, keys=["w", "d"]),
        }
        fields = ("vision_track_offset", "vision_track_heading", "vision_bend_ahead", "vision_road_horizon", "vision_crest_risk", "speed")
        results: dict[str, dict] = {}
        try:
            snapshot = world.snapshot()
            for label, action in actions.items():
                fork = type(world).from_scene(world.scene)
                fork.restore(snapshot)
                # The live episode is still on the grid at its first wake. Advance only
                # the fork through the frozen countdown before measuring input response;
                # otherwise every left/straight/right probe is identically stationary.
                while fork.countdown_ticks_remaining > 0:
                    fork.step(Action(name=ActionName.IDLE))
                probe = ReflexRuntime(
                    self.scene, tick_budget_ms=self.tick_budget_ms, vision_only=True,
                    visual_mode=self.visual_mode,
                )
                before = probe.observe_visual(fork)
                for _ in range(ticks):
                    fork.step(action)
                    after = probe.observe_visual(fork)
                results[label] = {
                    "after": {field: round(float(after[field]), 4) for field in fields},
                    "delta": {field: round(float(after[field]) - float(before[field]), 4) for field in fields},
                    "road_contact": bool(after["vision_road_contact"]),
                }
        except Exception as error:  # noqa: BLE001
            return {"error": f"could not run visual calibration: {type(error).__name__}: {error}"}
        report = {
            "at_tick": int(self.last_sense.get("tick", 0)), "pulse_ticks": ticks,
            "results": results,
            "validity": "episode-local camera response only; recalibrate after handling, surface, drag, or visual-context changes.",
            "note": (
                "Camera fields are measured from forked first-person screenshots. "
                + "Physical speed is the sole engine value returned."
            ),
        }
        self.visual_calibrations.append(report)
        return report

    def cone_profile(self) -> dict:
        """Compatibility name for the original 2D inspection tool."""
        return self.visual_road_profile()

    def tick(self, world, observation=None) -> Action:
        """One control tick: channels in, held keys out, conditions checked."""
        values = self.observe_visual(world) if self.vision_only else compute_sense(world, observation, self.memory, self.target)
        self.last_sense = values
        name = self.holding or self.active
        record = self.controllers.get(name) if name else None
        failure: str | None = None
        notes: list[str] = []

        if record is None:
            # Nothing installed. The harness has no idea how to drive, so the car sits
            # still and the deadline condition wakes the agent.
            action = Action(name=ActionName.IDLE)
            steer_command = throttle_command = 0.0
            self.idle_ticks += 1
        else:
            out = CommandOut(
                state=self.output_state, nitro_ready=False if self.vision_only else bool(observation.nitro_ready),
                nitro_active=False if self.vision_only else bool(observation.nitro_active),
                on_track=bool(values["on_track"]),
            )
            record.blocks.begin_tick(self.dt)
            started = perf_counter()
            try:
                record.controller.run(SenseView(values, record.controller.reads), record.blocks, out)
            except Exception as error:  # noqa: BLE001 - the runtime reports, never crashes
                failure = f"{type(error).__name__}: {error}"
            elapsed_ms = (perf_counter() - started) * 1000
            # A sustained overrun is the controller; a single one is the host descheduling the
            # process. Failing on the first reading would roll back a working controller
            # mid-race because something else on the machine got busy for a millisecond.
            self.overrun_streak = self.overrun_streak + 1 if elapsed_ms > self.tick_budget_ms * 4 else 0
            if failure is None and self.overrun_streak >= OVERRUN_TICKS_BEFORE_FAILURE:
                failure = (
                    f"overran the tick budget on {self.overrun_streak} consecutive ticks "
                    f"({elapsed_ms:.1f} ms)"
                )
            if failure is None and not (
                math.isfinite(out.steer_command) and math.isfinite(out.throttle_command)
            ):
                failure = "emitted a non-finite command"
            if failure is not None:
                action = self._on_failure(name, failure)
                steer_command = throttle_command = 0.0
            else:
                action = out.resolve()
                steer_command, throttle_command = out.steer_command, out.throttle_command
                notes = out.notes
                record.ticks_driven += 1
                self.controller_ticks += 1

        self.last_notes = notes
        self.ticks_since_wake += 1
        signals = self._signals(
            values, record, failure, countdown_active=world.countdown_ticks_remaining > 0,
        )
        fired = self.conditions.evaluate(values, signals)
        self._record(values, name, steer_command, throttle_command, action, fired)
        if fired:
            self._raise_wake(fired)
        return action

    def _signals(
        self, values: dict, record: ControllerRecord | None, failure: str | None,
        *, countdown_active: bool,
    ) -> dict:
        """The named events, all from quantities already computed this tick."""
        novelty = 0.0 if self.vision_only else max(
            self._novelty[channel].update(float(values[channel]), self.dt)
            for channel in self._novelty
        )
        worst_block, sign_changes = (
            record.blocks.max_sign_changes() if record is not None else (None, 0)
        )
        # A race has no reason to sit still.  This is intentionally a local
        # watchdog rather than a hidden progress oracle: it only sees the measured
        # speed the controller itself sees, and grants it 2.5 seconds to recover.
        barely_moving = (
            not countdown_active and record is not None and float(
                values.get("speed", values.get("vision_flow", 0.0))
                if self.vision_only else values["speed"]
            ) < (0.25 if self.vision_only else 0.12)
        )
        self.no_progress_ticks = self.no_progress_ticks + 1 if barely_moving else 0
        return {
            "controller_failed": failure is not None,
            "no_progress": self.no_progress_ticks >= NO_PROGRESS_TICKS,
            "unstable": sign_changes >= UNSTABLE_SIGN_CHANGES,
            "geometry_changed": (not self.vision_only and
                novelty > 1.0 and self.memory.tick - self.memory.last_geometry_wake > 25
            ),
            "deadline": self.ticks_since_wake >= self.deadline_ticks,
            "_worst_block": worst_block,
            "_sign_changes": sign_changes,
            "_novelty": round(novelty, 2),
        }

    def _on_failure(self, name: str | None, failure: str) -> Action:
        """Roll back to the agent's own previous controller, then to its safe action.

        Never to a harness controller. Substituting one here would reinstate exactly the
        hard-coded policy this design removes, and it would do it in the situations that
        matter most.
        """
        self.last_failure = failure
        self.failures.append(f"{name}: {failure}")
        record = self.controllers.get(name) if name else None
        if record is not None:
            parent = record.controller.parent
            if parent and parent.split("@")[0] in self.controllers and parent != record.label:
                self.active = parent.split("@")[0]
                return Action(name=ActionName.IDLE)
            safe = record.controller.safe_action or {}
            keys: list[str] = []
            throttle = float(safe.get("throttle", 0.0) or 0.0)
            if throttle > 0.08:
                keys.append("w")
            elif throttle < -0.08:
                keys.append("s")
            steer = safe.get("steer")
            if steer in {"left", "a"}:
                keys.append("a")
            elif steer in {"right", "d"}:
                keys.append("d")
            elif steer == "hold" and self.output_state.steer_key:
                keys.append(self.output_state.steer_key)
            if keys:
                from .output import _primary_name

                return Action(name=_primary_name(keys), keys=keys)
        self.fatal = f"controller failed with no fallback available ({failure})"
        return Action(name=ActionName.IDLE)

    def _raise_wake(self, fired: list[WakeCondition]) -> None:
        critical = [item for item in fired if item.hold or item.when == "controller_failed"]
        if not critical and self.ticks_since_wake < MIN_WAKE_GAP and self.pending:
            return
        self.pending = fired
        for item in fired:
            self.firings[item.when] = self.firings.get(item.when, 0) + 1
            if item.when == "geometry_changed":
                self.memory.last_geometry_wake = self.memory.tick
        hold = next((item.hold for item in fired if item.hold), None)
        if hold and hold in self.controllers:
            self.holding = hold

    @property
    def wake_requested(self) -> bool:
        return bool(self.pending) or self.fatal is not None

    def wake_causes_pending(self) -> list[str]:
        return [item.when for item in self.pending]

    def note_wake(self, causes: list[str]) -> None:
        """Record that these causes actually cost a model call."""
        for cause in causes:
            self.wake_causes[cause] = self.wake_causes.get(cause, 0) + 1

    def clear_wake(self) -> None:
        """Called once the agent has finished a turn."""
        self.pending = []
        self.holding = None
        self.ticks_since_wake = 0

    # -- recorder ----------------------------------------------------------------------

    def _record(
        self, values: dict, name: str | None, steer: float, throttle: float,
        action: Action, fired: list[WakeCondition],
    ) -> None:
        row = {
            "tick": values["tick"],
            "controller": name,
            "lane": values.get("vision_lane", values.get("lane", 0.0)),
            "heading_error": values.get("vision_turn", values.get("heading_error", 0.0)),
            "speed": values.get("vision_flow", values.get("speed", 0.0)),
            "curvature": values.get("vision_turn", values.get("curvature", 0.0)),
            "grip_used": values.get("grip_used", 0.0),
            "free_ahead": values.get("free_ahead", 0.0),
            "ttc": values.get("ttc") if values.get("ttc", 99) < 90 else None,
            "target_error": values.get("target_error", 0.0),
            "steer": round(steer, 3),
            "throttle": round(throttle, 3),
            "keys": "".join(action.keys) or "-",
            "fired": [item.when for item in fired] or None,
        }
        self.recorder.append(row)
        if len(self.recorder) > RECORDER_CAPACITY:
            del self.recorder[0]
        diagnostic = {
            "tick": values["tick"], "controller": name,
            "action": action.name.value, "keys": row["keys"],
            "steer": row["steer"], "throttle": row["throttle"],
            "fired": row["fired"],
            "vision": {
                field: round(float(values[field]), 4)
                for field in (self.visible_fields or ()) if field in values
            } if self.vision_only else {},
            "notes": list(self.last_notes),
        }
        self.diagnostic_trace.append(diagnostic)
        if len(self.diagnostic_trace) > DIAGNOSTIC_TRACE_CAPACITY:
            del self.diagnostic_trace[0]

    def _recent_channel_rows(self) -> list[dict]:
        """The live channel row, so the gate also checks the situation the car is in.

        Only the current row: the recorder keeps a decimated subset of fields rather than
        whole channel vectors, and a gate fed partial rows would fail on missing keys
        instead of on the controller.
        """
        return [dict(self.last_sense)] if self.last_sense else []

    def window(self, ticks: int = 60, rows: int = 24) -> list[dict]:
        """Decimated recorder rows, preserving extremes rather than sampling by stride.

        Stride sampling is the obvious implementation and it destroys the thing usually
        being debugged: a controller oscillating at half the control rate aliases into a
        smooth line. Bucketing and keeping each bucket's extreme lane error and steering
        keeps the oscillation visible at a fraction of the rows.
        """
        window = self.recorder[-max(1, ticks):]
        if len(window) <= rows:
            return window
        bucket_size = math.ceil(len(window) / rows)
        decimated: list[dict] = []
        for start in range(0, len(window), bucket_size):
            bucket = window[start:start + bucket_size]
            extreme = max(bucket, key=lambda row: abs(row["lane"]))
            summary = dict(extreme)
            summary["ticks"] = f"{bucket[0]['tick']}-{bucket[-1]['tick']}"
            summary["lane_range"] = [
                round(min(row["lane"] for row in bucket), 3),
                round(max(row["lane"] for row in bucket), 3),
            ]
            summary["steer_range"] = [
                round(min(row["steer"] for row in bucket), 3),
                round(max(row["steer"] for row in bucket), 3),
            ]
            fired = sorted({name for row in bucket if row["fired"] for name in row["fired"]})
            summary["fired"] = fired or None
            decimated.append(summary)
        return decimated

    def blocks_report(self) -> dict:
        record = self.controllers.get(self.active) if self.active else None
        if record is None:
            return {}
        return record.blocks.reports()

    # -- rehearsal ---------------------------------------------------------------------

    def try_controller(self, world, name: str, ticks: int = 300) -> TryReport:
        """Drive a fork of the current situation with `name`, with no model in the loop.

        `snapshot`/`restore` already exist and back `service.fork_run`, and the simulator
        is deterministic, so this costs no model calls and no real ticks. It is what turns
        writing a controller from generation into testing: the agent can find out that its
        controller crashes at the next hairpin before driving into it.

        It is a test of the situation the car is in now, not a generalization guarantee.
        """
        record = self.controllers.get(name)
        if record is None:
            return TryReport(controller=name, failure=f"no controller named {name!r}")
        if self.rehearsals_used >= self.rehearsal_budget:
            return TryReport(
                controller=record.label, is_active=self.active == name,
                control_hz=self.scene.dynamics.control_hz,
                best_finish_tick=self.best_finish_tick,
                failure=(
                    f"rehearsal budget for this wake is spent ({self.rehearsal_budget}). "
                    "Activate your best controller and resume."
                ),
            )
        self.rehearsals_used += 1
        try:
            fork = type(world).from_scene(world.scene)
            fork.restore(world.snapshot())
        except Exception as error:  # noqa: BLE001
            return TryReport(controller=name, failure=f"could not fork the world: {error}")

        rehearsal = ReflexRuntime(
            self.scene, tick_budget_ms=self.tick_budget_ms, vision_only=self.vision_only,
            visual_mode=self.visual_mode,
        )
        rehearsal.memory.track_index = self.memory.track_index
        rehearsal.target = self.target
        rehearsal.deadline_ticks = 10**6
        rehearsal.conditions = ConditionSet.build(
            [condition.when for condition in self.conditions.declared]
        )
        rehearsal.controllers[name] = ControllerRecord(
            controller=record.controller,
            blocks=ControlBlocks(record.controller.params),
            gate=record.gate,
        )
        rehearsal.active = name

        report = TryReport(
            controller=record.label, is_active=self.active == name,
            control_hz=self.scene.dynamics.control_hz,
            best_finish_tick=self.best_finish_tick,
        )
        started_at = self.memory.tick
        speeds: list[float] = []
        visual_rows: list[dict] = []
        for _ in range(max(1, ticks)):
            if fork.terminated:
                break
            countdown = fork.countdown_ticks_remaining > 0
            action = rehearsal.tick(fork, None if self.vision_only else fork.observe())
            fork.step(action)
            if countdown:
                # The frozen grid is not part of anyone's lap time, and the episode runner
                # does not count it either. Counting it here inflated every projected lap by
                # the countdown, which made a rehearsal look 30 ticks pessimistic against the
                # race it had actually predicted correctly.
                continue
            report.ticks += 1
            row = rehearsal.last_sense
            if self.vision_only:
                visual_row = dict(row)
                visual_row["action"] = action.name.value
                visual_rows.append(visual_row)
            if not self.vision_only:
                speeds.append(float(row["speed"]))
                report.max_grip_used = max(report.max_grip_used, float(row["grip_used"]))
                if not row["on_track"]:
                    report.off_track_ticks += 1
            for condition in rehearsal.pending:
                marker = f"{condition.when}@{row['tick']}"
                if marker not in report.fired:
                    report.fired.append(marker)
            rehearsal.clear_wake()
        report.terminated = fork.terminated
        report.succeeded = fork.succeeded
        report.reason = fork.reason
        report.checkpoints_reached = fork.objective_index
        report.mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        report.worst_block, report.worst_sign_changes = (
            rehearsal.controllers[name].blocks.max_sign_changes()
        )
        if report.succeeded:
            # Measured from the start of the episode, not from the fork, or two rehearsals
            # taken at different points in the race would not be comparable.
            report.projected_finish_tick = started_at + report.ticks
        report.budget_remaining = max(0, self.rehearsal_budget - self.rehearsals_used)
        self.rehearsal_log.append({
            "at_tick": started_at, "controller": record.label,
            "params": dict(record.controller.params),
            "finished": report.succeeded,
            "lap_time_ticks": report.projected_finish_tick,
            "previous_best": self.best_finish_tick,
        })
        if report.is_new_best:
            self.best_finish_tick = report.projected_finish_tick
        if self.vision_only and visual_rows:
            report.visual = {
                "road_contact_losses": sum(not bool(row["vision_ego_road_contact"]) for row in visual_rows),
                "min_left_gap": round(min(float(row["vision_left_gap"]) for row in visual_rows), 3),
                "min_right_gap": round(min(float(row["vision_right_gap"]) for row in visual_rows), 3),
                "min_confidence": round(min(float(row["vision_confidence"]) for row in visual_rows), 3),
                "visual_trace": [{key: row[key] for key in ("tick", "action", "vision_center_near", "vision_center_far", "vision_turn_ahead", "vision_turn_severity", "vision_lookahead_depth", "vision_left_gap", "vision_right_gap", "vision_confidence", "vision_ego_road_contact")} for row in visual_rows[::max(1, len(visual_rows) // 12)]],
            }
        return report

    # -- reporting ---------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "active_controller": (
                self.controllers[self.active].label if self.active in self.controllers else None
            ),
            "installed": [record.label for record in self.controllers.values()],
            "target": self.target.as_dict() if self.target else None,
            "conditions": self.conditions.describe(),
            "deadline_ticks": self.deadline_ticks,
            "ticks_since_wake": self.ticks_since_wake,
        }

    def summary(self) -> dict:
        return {
            "perception": "forward-cone screenshots + optical flow" if self.vision_only else "telemetry",
            "best_rehearsed_lap_ticks": self.best_finish_tick,
            "rehearsals": self.rehearsal_log,
            "visual_calibrations": self.visual_calibrations,
            "controller_ticks": self.controller_ticks,
            "idle_ticks": self.idle_ticks,
            "wake_causes": dict(sorted(
                self.wake_causes.items(), key=lambda item: -item[1],
            )),
            "condition_firings": dict(sorted(
                self.firings.items(), key=lambda item: -item[1],
            )),
            "controllers": [
                {
                    "label": record.label, "ticks": record.ticks_driven,
                    "reads": list(record.controller.reads),
                    "params": dict(record.controller.params),
                    "safe_action": dict(record.controller.safe_action),
                    "source": record.controller.source,
                    "gate": record.gate.as_dict(),
                }
                for record in self.controllers.values()
            ],
            "output": self.output_state.report(),
            "condition_margins": self.conditions.report(),
            "failures": self.failures[:8],
        }

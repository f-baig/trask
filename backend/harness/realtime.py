"""Run an episode with the planner off the simulator's critical path.

The synchronous loop in `service.run` stops the world while the model thinks. That is
the right default for a reproducible benchmark, but it quietly hands the policy an
infinite time budget: a decision that takes three seconds costs exactly as much as one
that takes fifty milliseconds, because the car waits either way. A driver measured
that way can look competent while being far too slow to drive anything.

This scheduler instead advances the simulator on a clock and lets a worker thread
compute the next chunk in parallel. While a decision is in flight the car keeps
executing whatever it was last told to do, so latency shows up where it actually
hurts — as ticks driven on stale input.

Three clocks, because "real time" and "reproducible" are different requests:

  wall      One tick per 1/control_hz of real time, with the planner on a worker
            thread. What a human would watch, and the only mode with real concurrency.
  measured  A decision lands after as many ticks as its measured latency corresponds
            to. The same episode `wall` would produce, without burning the wall clock
            idling — so it needs no thread at all: charging the ticks after the fact is
            equivalent to advancing them during, and it is exactly reproducible from
            the recorded latencies.
  fixed     Every decision lands after exactly N ticks, whatever the API did. The
            schedule becomes a pure function of the model's outputs, so the only
            remaining nondeterminism is the model itself.

The tick cost is deliberately not gated on the worker finishing outside `wall` mode. An
earlier version was, and because the tick loop is not paced there, the simulator raced
ahead while the operating system got round to scheduling the thread — charging a
zero-latency decision six ticks, and a real three-second call the entire step budget.

The engine is untouched and still deterministic. What varies here is when inputs reach
it, and that is now an explicit, recorded quantity rather than an artifact of however
fast the API happened to be.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .models import Action, ActionName, DecisionRecord, FrameRecord
from .policies import DecisionRequest, PolicySessionError
from .providers import ProviderError


CLOCKS = ("measured", "wall", "fixed")
STARVATION_MODES = ("hold", "coast")


class TerminalAwareExecutor(ThreadPoolExecutor):
    """Do not hold a completed race open for an obsolete planner response.

    Python's ordinary executor context waits for running futures on exit. That is useful
    for batch work, but wrong for a real-time episode: after the final lap there is no
    future simulator state to apply the response to. Pending calls are cancelled when
    possible; an already-running provider call is pure with respect to policy state and
    is allowed to unwind without delaying run finalization.
    """

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.shutdown(wait=False, cancel_futures=True)
        return False


@dataclass
class InFlight:
    """A decision the world is currently driving without."""

    request: DecisionRequest
    submitted_tick: int
    future: Future | None = None
    """Set only under the `wall` clock, the one mode with real concurrency."""
    result: tuple | None = None
    ready_tick: int | None = None


@dataclass
class RealtimeReport:
    ticks: int = 0
    decisions: int = 0
    starved_ticks: int = 0
    """Ticks driven with an empty queue, on held or coasted input."""
    stale_tick_total: int = 0
    """Summed age, in ticks, of every decision at the moment it was applied."""
    late_decisions: int = 0
    """Decisions the model delivered slower than a `fixed` clock allowed for."""
    superseded_decisions: int = 0
    """Answers discarded because a newer decision had already been applied.

    Concurrent requests can finish out of order. Applying an older plan on top of a
    newer one would drive the car backwards in time, so the stale answer is dropped —
    and counted, because a high count means the pipeline is too deep to be useful.
    """
    decision_ticks: list[int] = field(default_factory=list)
    rejected_decisions: int = 0
    """Answers rejected because their predicted activation state was obsolete."""

    @property
    def mean_decision_ticks(self) -> float:
        return (
            sum(self.decision_ticks) / len(self.decision_ticks)
            if self.decision_ticks else 0.0
        )

    @property
    def fresh_input_fraction(self) -> float:
        return 0.0 if not self.ticks else round(1 - self.starved_ticks / self.ticks, 3)

    def as_dict(self) -> dict:
        return {
            "ticks": self.ticks,
            "decisions": self.decisions,
            "starved_ticks": self.starved_ticks,
            "fresh_input_fraction": self.fresh_input_fraction,
            "mean_decision_ticks": round(self.mean_decision_ticks, 2),
            "max_decision_ticks": max(self.decision_ticks, default=0),
            "decision_ticks": list(self.decision_ticks),
            "late_decisions": self.late_decisions,
            "superseded_decisions": self.superseded_decisions,
            "rejected_decisions": self.rejected_decisions,
            "ticks_per_decision": (
                round(self.ticks / self.decisions, 1) if self.decisions else None
            ),
        }


def run_realtime_episode(
    world, policy, *, max_steps: int, clock: str = "measured",
    latency_ticks: int = 3, starve: str = "hold", pipeline_depth: int = 1,
    decision_budget: int | None = None,
    action_delay: bool = False, guidance: str | None = None,
    progress: Callable[[int, Action, RealtimeReport], None] | None = None,
) -> dict:
    """Drive `world` with `policy` while the planner runs concurrently.

    Returns the episode outcome plus a latency report. The policy must expose the
    three-phase decision interface (`prepare_decision`, `execute_decision`,
    `apply_decision`); a policy that only implements `act` has nothing to overlap and
    belongs on the synchronous path.
    """
    if clock not in CLOCKS:
        raise ValueError(f"clock must be one of {CLOCKS}; got {clock!r}")
    if starve not in STARVATION_MODES:
        raise ValueError(f"starve must be one of {STARVATION_MODES}; got {starve!r}")
    if pipeline_depth < 1:
        raise ValueError(f"pipeline_depth must be at least 1; got {pipeline_depth}")
    if pipeline_depth > 1 and clock != "wall":
        # Overlapping latencies are a wall-clock phenomenon. The other clocks simulate
        # concurrency by charging ticks after an inline call, which cannot represent
        # two calls running at once without a real clock to overlap them on.
        raise ValueError("pipeline_depth above 1 requires the wall clock")
    plans = all(
        hasattr(policy, required)
        for required in ("prepare_decision", "execute_decision", "apply_decision")
    )
    controls = hasattr(policy, "tick_action")
    if not plans and not controls:
        raise TypeError(
            f"{type(policy).__name__} cannot run on the realtime scheduler: it needs "
            "either the three-phase decision interface or a tick_action controller"
        )
    if not hasattr(policy, "record_executed"):
        raise TypeError(
            f"{type(policy).__name__} cannot run on the realtime scheduler: it has no record_executed"
        )

    tick_seconds = 1 / world.dynamics.control_hz
    report = RealtimeReport()
    queue: list[Action] = []
    pending: list[InFlight] = []
    frames: list[FrameRecord] = []
    failure: str | None = None
    held = Action(name=ActionName.IDLE)
    active_steps = 0
    last_observed_tick = 0
    last_submit_tick = -(10**9)
    last_applied_submit_tick = -1
    next_submit_tick = 0
    submitted_decisions = 0
    expected_ticks = max(1, latency_ticks)
    """Running estimate of a decision's cost, used to space out submissions."""
    next_tick_at = time.monotonic()

    policy.reset(world.scene, world.scene.seed)
    if hasattr(policy, "configure_episode"):
        policy.configure_episode(max_steps, decision_budget)
    if guidance and hasattr(policy, "set_episode_guidance"):
        policy.set_episode_guidance(guidance)

    with TerminalAwareExecutor(max_workers=pipeline_depth, thread_name_prefix="racelab-planner") as pool:
        try:
            while not world.terminated and active_steps < max_steps:
                if world.countdown_ticks_remaining > 0:
                    frames.append(world.step(Action()))
                    continue

                # Submissions are spaced by latency/depth rather than fired together.
                # Issuing the whole pipeline at once makes N answers land in a burst and
                # then leaves a full latency of silence, which is the starvation the
                # pipeline exists to remove.
                stagger = max(1, expected_ticks // pipeline_depth)
                decision_interval = max(0, int(getattr(policy, "decision_interval_ticks", 0)))
                cadence_ready = (
                    report.ticks >= next_submit_tick if decision_interval
                    else report.ticks - last_submit_tick >= stagger
                )
                # A feedback controller remains useful after its planning allowance
                # is spent. Stop asking the model and let the last installed skill
                # finish the episode instead of turning a call cap into a fatal error.
                controller_driven = controls or hasattr(policy, "tick_action_visual")
                budget_ready = (
                    not controller_driven
                    or decision_budget is None
                    or submitted_decisions < decision_budget
                )
                if plans and budget_ready and cadence_ready and len(pending) < pipeline_depth:
                    observation = world.observe()
                    frame = None
                    if hasattr(policy, "observe_frame") and hasattr(world, "render_policy_frame"):
                        frame = policy.observe_frame(
                            _render_policy_frame(policy, world),
                            interval_ticks=max(1, report.ticks - last_observed_tick),
                        )
                        last_observed_tick = report.ticks
                    if hasattr(policy, "set_prediction_horizon"):
                        policy.set_prediction_horizon(expected_ticks, tick_seconds, report.ticks)
                    request = policy.prepare_decision(observation, frame)
                    pending.append(_issue(
                        policy, request, report, pool, tick_seconds, clock, latency_ticks,
                    ))
                    submitted_decisions += 1
                    last_submit_tick = report.ticks

                pending = [_collect(item, report, clock) for item in pending]
                landed = [
                    item for item in pending
                    if item.ready_tick is not None and report.ticks >= item.ready_tick
                ]
                pending = [item for item in pending if item not in landed]
                for item in sorted(landed, key=lambda entry: entry.submitted_tick):
                    if item.submitted_tick < last_applied_submit_tick:
                        report.superseded_decisions += 1
                        continue
                    plan, usage = item.result
                    age = report.ticks - item.submitted_tick
                    accepted, rejection_reason = True, ""
                    if hasattr(policy, "validate_decision"):
                        activation_frame = _render_policy_frame(policy, world)
                        accepted, rejection_reason = policy.validate_decision(
                            item.request, plan, world.observe(), activation_frame,
                        )
                    if accepted:
                        queue, _ = policy.apply_decision(item.request, plan, usage)
                    else:
                        report.rejected_decisions += 1
                        policy.reject_decision(
                            item.request, plan, usage, rejection_reason,
                        )
                    last_applied_submit_tick = item.submitted_tick
                    report.decisions += 1
                    report.decision_ticks.append(age)
                    report.stale_tick_total += age
                    expected_ticks = max(1, round(report.mean_decision_ticks))
                    if decision_interval:
                        next_submit_tick = report.ticks + decision_interval
                    if hasattr(policy, "min_action_horizon"):
                        # Tell the policy how far ahead it must cover, so the next plan
                        # is long enough to bridge its own latency.
                        policy.min_action_horizon = max(1, expected_ticks // pipeline_depth)

                if hasattr(policy, "tick_action_visual"):
                    action = policy.tick_action_visual(
                        world.observe(), _render_policy_frame(policy, world),
                    )
                    held = action
                elif controls:
                    # A fast controller closes the loop every tick from the current
                    # intent, so no tick is ever uncontrolled. Latency stops costing
                    # ticks-without-input and starts costing ticks on a stale intent,
                    # which `mean_decision_ticks` already reports.
                    action = policy.tick_action(world.observe())
                    held = action
                elif queue:
                    action = queue.pop(0)
                    held = action
                else:
                    # No fresh input has arrived. Holding the last keys is what a
                    # physical keyboard does when nobody presses anything new;
                    # coasting is the safer counterfactual.
                    action = held if starve == "hold" else Action(name=ActionName.IDLE)
                    report.starved_ticks += 1

                policy.record_executed(action)
                frames.append(world.step(action, DecisionRecord(
                    action=action.name, subgoal=policy.subgoal,
                    confidence=policy.confidence,
                    summary=f"Realtime {clock} clock. {policy.summary[:200]}",
                    candidates=[action.name],
                ), action_delay=action_delay))
                report.ticks += 1
                active_steps += 1
                if progress is not None:
                    progress(report.ticks, action, report)
                if clock == "wall" and not world.terminated:
                    next_tick_at += tick_seconds
                    remaining = next_tick_at - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    else:
                        # The renderer and physics themselves fell behind real time.
                        # Resync rather than accumulate an unpayable debt.
                        next_tick_at = time.monotonic()
        except (PolicySessionError, ProviderError) as error:
            failure = str(error)
        finally:
            for item in pending:
                if item.future is not None:
                    item.future.cancel()

    return {
        "succeeded": world.succeeded,
        "terminated": world.terminated,
        "reason": failure or world.reason or "step budget exhausted",
        "policy_failure": failure,
        "steps": len(frames),
        "checkpoints_reached": world.objective_index,
        "position": world.player_position,
        "field_size": world.field_size,
        "clock": clock,
        "starve": starve,
        "pipeline_depth": pipeline_depth,
        "terse": bool(getattr(policy, "terse", False)),
        "latency_ticks": latency_ticks if clock == "fixed" else None,
        "control_hz": world.dynamics.control_hz,
        "realtime": report.as_dict(),
        "policy_realtime": (
            policy.realtime_metrics() if hasattr(policy, "realtime_metrics") else {}
        ),
        "frames": frames,
    }


def _render_policy_frame(policy, world):
    """Use the policy's declared camera rather than the world's generic view."""
    return policy.render_frame(world) if hasattr(policy, "render_frame") else world.render_policy_frame()


def _issue(
    policy, request: DecisionRequest, report: RealtimeReport, pool: ThreadPoolExecutor,
    tick_seconds: float, clock: str, latency_ticks: int,
) -> InFlight:
    """Start a decision, and decide up front which tick it will land on.

    Under `wall` the answer genuinely arrives whenever it arrives, so the call goes to
    a worker and the world keeps ticking in real time. The other clocks are simulating
    that concurrency rather than performing it: the call is made inline and then
    charged a tick cost, which advances the world by exactly as much as running it in
    parallel would have. That is equivalent, cheaper, and reproducible.
    """
    pending = InFlight(request=request, submitted_tick=report.ticks)
    if clock == "wall":
        pending.future = pool.submit(policy.execute_decision, request)
        return pending
    pending.result = policy.execute_decision(request)
    _, usage = pending.result
    measured_ticks = math.ceil(max(0, usage.latency_ms) / (tick_seconds * 1000))
    if clock == "fixed":
        if measured_ticks > latency_ticks:
            # The model was slower than the budget this clock pretends it had. Charging
            # the budget anyway is the point of `fixed`, but the gap gets recorded so
            # the fiction is visible in the report.
            report.late_decisions += 1
        cost_ticks = latency_ticks
    else:
        cost_ticks = measured_ticks
    pending.ready_tick = pending.submitted_tick + cost_ticks
    return pending


def _collect(pending: InFlight, report: RealtimeReport, clock: str) -> InFlight:
    """Resolve a `wall`-clock future once the worker has actually answered."""
    if pending.result is not None or pending.future is None:
        return pending
    if not pending.future.done():
        return pending
    pending.result = pending.future.result()
    pending.ready_tick = report.ticks
    return pending

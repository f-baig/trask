"""Run a whole episode: the controller drives, and the agent is woken by events.

The wake loop is the point of the architecture, so it is worth being exact about what this
runner does and does not measure.

The simulator does not advance during a model call. Instead the runner *charges* the call a
tick cost and drives those ticks on whatever was installed before the wake — which is the
same equivalence `realtime.py` relies on for its `measured` clock: charging the ticks after
the fact advances the world by exactly as much as running the call in parallel would have,
and it is reproducible. The estimate for a call's cost is the previous call's measured
latency, because a cost has to be charged before the call whose latency is not yet known.

With `--latency zero` nothing is charged, which is the clean architectural baseline: it
measures whether agent-written control works at all, with latency set aside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..models import Action, FrameRecord
from .agent import AgentTurn, TurnUsage, run_agent_turn
from .perception import render_payload, wake_payload
from .runtime import ReflexRuntime
from .tools import first_wake_prompt, system_prompt


@dataclass
class EpisodeReport:
    succeeded: bool = False
    terminated: bool = False
    reason: str | None = None
    ticks: int = 0
    checkpoints_reached: int = 0
    position: int | None = None
    field_size: int = 0
    wakes: int = 0
    wake_budget_exhausted: bool = False
    stale_ticks: int = 0
    """Ticks driven on a pre-wake controller while a decision was in flight."""
    usage: TurnUsage = field(default_factory=TurnUsage)
    turns: list[AgentTurn] = field(default_factory=list)
    runtime_summary: dict = field(default_factory=dict)
    diagnostics_payload: dict = field(default_factory=dict, repr=False)
    fatal: str | None = None
    frames: list[FrameRecord] = field(default_factory=list)
    """Every simulator frame, countdown included, for the replay viewer.

    Kept out of `as_dict` deliberately: a few hundred frames of privileged state dwarf the
    report they would be embedded in. `replay_bundle` turns them into the same artifact
    `harness replay` produces.
    """

    def as_dict(self) -> dict:
        return {
            "succeeded": self.succeeded,
            "terminated": self.terminated,
            "reason": self.reason,
            "ticks": self.ticks,
            "checkpoints_reached": self.checkpoints_reached,
            "position": self.position,
            "field_size": self.field_size,
            "wakes": self.wakes,
            "wake_budget_exhausted": self.wake_budget_exhausted,
            "ticks_per_wake": round(self.ticks / self.wakes, 1) if self.wakes else None,
            "stale_ticks": self.stale_ticks,
            **self.usage.as_dict(),
            "runtime": self.runtime_summary,
            "turns": [turn.as_dict() for turn in self.turns],
            **({"fatal": self.fatal} if self.fatal else {}),
        }

    def diagnostics(self, runtime: ReflexRuntime) -> dict:
        """Evaluator-side provenance for explaining a controller failure.

        This artifact is never put back into the model conversation.  It records what the
        model installed and the pixel-derived inputs/actions that followed, so an off-track
        result can be attributed to a specific authored branch rather than guessed from a
        rendered path.
        """
        return {
            "schema_version": 1,
            "perception": "forward-cone screenshots + optical flow" if runtime.vision_only else "telemetry",
            "episode": self.as_dict(),
            "turns": [
                {
                    "tick": turn.tick,
                    "woke_because": turn.causes,
                    "resumed": turn.resumed,
                    "error": turn.error,
                    "said": turn.said,
                    "tool_calls": turn.tool_calls,
                    "usage": turn.usage.as_dict(),
                }
                for turn in self.turns
            ],
            "final_runtime": runtime.summary(),
            "control_trace": runtime.diagnostic_trace,
        }


def run_reflex_episode(
    world, *, model: str, max_steps: int = 900, max_wakes: int = 12,
    latency: str = "measured", latency_ticks: int = 12, verbose: bool = False,
    max_round_trips: int = 10, rehearsal_budget: int = 5, progress=None,
    vision_only: bool = False, visual_mode: str | None = None,
) -> EpisodeReport:
    """Drive `world` with an agent-authored reflex controller."""
    scene = world.scene
    runtime = ReflexRuntime(
        scene, vision_only=vision_only, visual_mode=visual_mode,
    )
    runtime.rehearsal_budget = rehearsal_budget
    report = EpisodeReport(field_size=world.field_size)
    system = system_prompt(
        scene, vision_only=vision_only, visual_mode=runtime.visual_mode,
    )
    history: list[dict] = []
    tick_ms = 1_000 / scene.dynamics.control_hz
    charged_ticks = 0 if latency == "zero" else latency_ticks

    # The first wake happens before the flag, while the countdown freezes the grid: the
    # car will not move at all until something is installed, so there is nothing to
    # observe and no reason to wait.
    runtime.last_sense = runtime.observe_visual(world) if vision_only else _seed_sense(runtime, world)
    # A perspective controller cannot assume that a previous car/surface/camera response
    # still applies. Run the fixed, camera-only pulse test at the flag for every 3D episode
    # and retain it only in this episode's diagnostics.
    if vision_only and runtime.visual_mode == "3d":
        runtime.calibrate_perspective_controls(world)
    if verbose:
        print("  [wake 1] first decision, nothing installed", flush=True)
    turn = run_agent_turn(
        runtime=runtime, world=world, observation=None if vision_only else world.observe(), system=system,
        prompt=first_wake_prompt(runtime), model=model, causes=[], history=history,
        # The first wake is authorship — write, rehearse, retune, rehearse again — and it
        # is the one turn where running out of round trips costs the whole episode, because
        # nothing is installed yet.
        max_round_trips=max_round_trips + 8, verbose=verbose, frame=runtime.model_frame() if vision_only else None,
    )
    _absorb(report, turn)
    runtime.clear_wake()
    if turn.error:
        report.reason = f"first decision failed: {turn.error}"
        report.terminated = True
        return report

    while not world.terminated and report.ticks < max_steps:
        if world.countdown_ticks_remaining > 0:
            report.frames.append(world.step(Action()))
            continue
        action = runtime.tick(world, None if vision_only else world.observe())
        report.frames.append(world.step(action))
        report.ticks += 1
        if progress is not None:
            progress(report.ticks, action, runtime)
        if runtime.fatal:
            report.fatal = runtime.fatal
            break
        if not runtime.wake_requested:
            continue
        if report.wakes >= max_wakes:
            # Not a silent degradation: the agent's own controller keeps driving and the
            # report says the budget ran out. Terminating here would throw away the most
            # informative part of the episode, which is how far a controller gets unsupervised.
            report.wake_budget_exhausted = True
            runtime.clear_wake()
            continue

        causes = runtime.wake_causes_pending()
        runtime.note_wake(causes)
        runtime.rehearsals_used = 0
        if verbose:
            print(f"  [wake {report.wakes + 1}] tick {report.ticks}: {', '.join(causes)}", flush=True)
        # Charge the decision its tick cost before making it, driving those ticks on what
        # was already installed — or on the `hold` controller a condition named.
        for _ in range(charged_ticks):
            if world.terminated or report.ticks >= max_steps:
                break
            stale_action = runtime.tick(world, None if vision_only else world.observe())
            report.frames.append(world.step(stale_action))
            report.ticks += 1
            report.stale_ticks += 1
        if world.terminated:
            break
        payload = wake_payload(runtime, causes)
        prompt = render_payload(payload)
        if not report.turns[-1].resumed:
            # Worth saying plainly: a turn that ran out of round trips left no target and no
            # conditions behind, and an agent that cannot see that will do it again.
            prompt = (
                "NOTE: your previous turn ended without calling resume, so the car kept "
                "driving on whatever was already active.\n\n" + prompt
            )
        turn = run_agent_turn(
            runtime=runtime, world=world, observation=None if vision_only else world.observe(), system=system,
            prompt=prompt, model=model, causes=causes, history=history,
            max_round_trips=max_round_trips, verbose=verbose, frame=runtime.model_frame() if vision_only else None,
        )
        _absorb(report, turn)
        runtime.clear_wake()
        if latency == "measured" and turn.usage.latency_ms:
            charged_ticks = max(1, math.ceil(turn.usage.latency_ms / tick_ms))
        if turn.error:
            report.reason = f"decision failed: {turn.error}"
            break

    report.succeeded = world.succeeded
    report.terminated = world.terminated
    report.reason = report.reason or world.reason or "step budget exhausted"
    report.checkpoints_reached = world.objective_index
    report.position = world.player_position
    report.runtime_summary = runtime.summary()
    report.diagnostics_payload = report.diagnostics(runtime)
    return report


def replay_bundle(scene, report: EpisodeReport, *, run_id: str, policy_name: str):
    """Package a reflex episode as the artifact every RaceLab renderer already reads.

    Deliberately the same `ReplayBundle` that `harness replay` exports, rather than a
    reflex-specific format: a run driven by an agent-written controller should be watchable
    in the existing viewer, and comparable frame by frame against every other arm.
    """
    from ..models import RunStatus
    from ..rendering import ReplayBundle, ReplayMetadata

    return ReplayBundle.from_frames(scene, report.frames, metadata=ReplayMetadata(
        run_id=run_id, policy_name=policy_name,
        status=(RunStatus.SUCCEEDED if report.succeeded else RunStatus.FAILED).value,
        seed=scene.seed,
        total_reward=round(sum(frame.reward for frame in report.frames), 3),
    ))


def _seed_sense(runtime: ReflexRuntime, world) -> dict:
    """A channel row for the first wake, before any tick has happened."""
    from .sense import compute_sense

    return compute_sense(world, world.observe(), runtime.memory, runtime.target)


def _absorb(report: EpisodeReport, turn: AgentTurn) -> None:
    report.wakes += 1
    report.turns.append(turn)
    report.usage.calls += turn.usage.calls
    report.usage.input_tokens += turn.usage.input_tokens
    report.usage.output_tokens += turn.usage.output_tokens
    report.usage.cache_read_input_tokens += turn.usage.cache_read_input_tokens
    report.usage.cache_creation_input_tokens += turn.usage.cache_creation_input_tokens
    report.usage.latency_ms += turn.usage.latency_ms

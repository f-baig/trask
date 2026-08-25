"""Player adapters for the single racing domain."""

from __future__ import annotations

import random
import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Protocol

from .lowlevel import (
    LOCAL_OBSERVATION_FIELDS, ConstantIntentPolicy, Intent, LocalIntentController,
)
from .models import Action, ActionName, DecisionRecord, ObservationPacket, ProviderTurnUsage, SceneSpec
from .motion import MotionOverlay
from .providers import (
    configured_model, plan_predictive_driving_skill, plan_racing_actions,
    plan_racing_intent, plan_racing_strategy,
    review_racing_action,
)
from .policy_protocol import PolicyAction, PolicyCapabilities, PolicyReset, PolicyStep, VisualFrame
from .racing import (
    RacingIntentController, RacingLineController, racing_local_state,
    racing_public_context, racing_strategy_context,
)


STARTING_TARGET_SPEED = 4.0
"""Pace the fast layer creeps at before any strategy has arrived."""


def _visible_skill_failure(dimension: str, skill: str, state: dict) -> dict:
    """Label a road-loss transition using only the player's public visual contract."""
    severity = float(state.get("bend_severity", state.get("turn_severity", 0.0)))
    depth = float(state.get("visible_depth", 0.0))
    speed = float(state.get("speed", 0.0))
    offset = float(state.get("road_offset", state.get("center_near", 0.0)))
    if severity >= 0.7:
        circumstance = "during a severe visible bend"
    elif depth and depth < 0.45:
        circumstance = "while visible road depth was short"
    elif abs(offset) >= 0.65:
        circumstance = "after a large camera-visible lateral error"
    else:
        circumstance = "without one dominant visible precursor"
    return {
        "skill": skill,
        "dimension": dimension,
        "failure": "road_contact_lost",
        "label": f"{skill} lost camera-visible road contact {circumstance}",
        "evidence_source": ["camera_frame", "scalar_speed", "image_features", "skill_history"],
        "public_evidence": {
            "speed": round(speed, 3),
            "bend_severity": round(severity, 3),
            "visible_depth": round(depth, 3),
            "lateral_error": round(offset, 3),
        },
        "status": "pending",
    }


# Policy identifiers are part of the experiment record, CLI, and coordinator prompt.  Keep
# the information boundary visible in the identifier itself so a telemetry-assisted driver
# cannot be mistaken for a camera-only one in a chart or replay tree.
LEGACY_POLICY_ALIASES = {
    "racing-line": "oracle-racing-line",
    "racing-agent": "telemetry-direct",
    "racing-agent-strategy": "telemetry-strategy",
    "racing-agent-hierarchical": "telemetry-hierarchical",
    "racing-agent-reflex": "telemetry-reflex",
    "racing-agent-reflex-vision": "vision-reflex-sim-rehearsal",
    "racing-agent-cone-visual": "vision-2d-direct",
    "racing-agent-2d-predictive-skills": "vision-2d-predictive-skills",
    "racing-agent-3d-visual-tick": "vision-3d-direct-every-tick",
    "racing-agent-3d-visual-short": "vision-3d-direct-short",
    "racing-agent-3d-visual-short-speed-road": "vision-3d-direct-short-features",
    "racing-agent-3d-predictive-skills": "vision-3d-predictive-skills",
    "constant-intent": "baseline-constant-intent",
    "wanderer": "baseline-random",
    "external-player": "external-telemetry-player",
}

POLICY_DISPLAY_NAMES = {
    "vision-2d-predictive-skills": "Vision Controller Agent · predictive skills · 2D",
    "vision-2d-direct": "Vision Action Agent · 2D",
    "vision-reflex-sim-rehearsal": "Vision Controller Agent · 2D",
    "vision-3d-direct-every-tick": "Vision Action Agent · every tick · 3D",
    "vision-3d-direct-short": "Vision Action Agent · short horizon · 3D",
    "vision-3d-direct-short-features": "Vision Action Agent · road features · 3D",
    "vision-3d-predictive-skills": "Vision Controller Agent · predictive skills · 3D",
    "telemetry-direct": "Diagnostic · telemetry action agent",
    "telemetry-strategy": "Diagnostic · telemetry strategy agent",
    "telemetry-hierarchical": "Diagnostic · telemetry hierarchical agent",
    "telemetry-reflex": "Diagnostic · telemetry controller agent",
    "oracle-racing-line": "Oracle Racing Line",
    "baseline-constant-intent": "Fixed Controller Baseline",
    "baseline-random": "Random Action Baseline",
}


def canonical_policy_name(name: str) -> str:
    """Resolve a historical identifier without keeping ambiguous names in new records."""
    return LEGACY_POLICY_ALIASES.get(name, name)


def policy_display_name(name: str) -> str:
    """Stable researcher-facing name without changing serialized policy identifiers."""
    canonical = canonical_policy_name(name)
    return POLICY_DISPLAY_NAMES.get(canonical, canonical.replace("_", " ").replace("-", " ").title())


class PlayerPolicy(Protocol):
    name: str

    def reset(self, scene: SceneSpec, seed: int) -> None: ...

    def act(self, observation: ObservationPacket) -> tuple[Action, DecisionRecord]: ...


class PolicySessionError(RuntimeError):
    """A policy session ended before the simulator episode did."""


class PolicyBudgetExhausted(PolicySessionError):
    """The configured policy-call or token allowance was consumed."""


@dataclass(frozen=True)
class DecisionRequest:
    """One planner call's inputs, frozen at the moment it was issued.

    An asynchronous scheduler keeps advancing the simulator while a decision is in
    flight, so by the time the answer arrives the observation it was computed from is
    stale. Carrying the request explicitly makes that staleness visible and
    measurable instead of implicit.
    """

    observation: ObservationPacket
    frame: VisualFrame | None
    frames: list[VisualFrame]
    context: dict
    max_tokens: int
    operator_guidance: str | None = None


@dataclass
class WandererPolicy:
    """A deterministic random-control failure baseline."""

    name: str = "baseline-random"
    rng: random.Random = random.Random(0)

    def reset(self, scene: SceneSpec, seed: int) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: ObservationPacket) -> tuple[Action, DecisionRecord]:
        action = self.rng.choice([ActionName.FORWARD, ActionName.BACKWARD, ActionName.LEFT, ActionName.RIGHT])
        return Action(name=action), DecisionRecord(
            action=action, subgoal="drive without a plan", confidence=.2,
            summary="Sampling a racing control.", candidates=[action],
        )


@dataclass
class AnthropicRacingPolicy:
    """Domain-specific Claude driver; returned controls are executed directly."""

    name: str = "telemetry-direct"
    scene: SceneSpec | None = None
    actions: list[Action] | None = None
    subgoal: str = "acquire racing line"
    summary: str = "Awaiting public racing telemetry."
    confidence: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: int = 0
    turn_usages: list[ProviderTurnUsage] | None = None
    output_token_budget: int = 30_000
    max_turns: int = 100
    action_horizon: int = 6
    terse: bool = False
    """Ask for control ticks without commentary, which is most of the output tokens."""
    min_action_horizon: int = 0
    """Floor on queued ticks, set by a scheduler that knows its own latency.

    The risk-based horizon shrink below is right when replanning is free: get into
    trouble, replan next tick. Under real-time latency it inverts — cutting the queue
    to one tick when the next decision is thirteen ticks away leaves the car
    uncontrolled for twelve of them, precisely when it is already in trouble.
    """
    recent_trace: list[dict[str, object]] | None = None
    last_action: ActionName | None = None
    last_keys: tuple[str, ...] = ()
    # The chunk the model last authored, and what the scheduler actually did with it.
    # Without this the model cannot tell a correction that ran for four ticks from
    # one the harness threw away after one, so it keeps re-issuing plans that never
    # execute. It matters more the longer a decision takes.
    planned_segments: list[dict[str, object]] | None = None
    planned_ticks: int = 0
    queued_ticks: int = 0
    executed_since_plan: list[dict[str, object]] | None = None
    chunk_interrupted: bool = False
    track_index: int | None = None
    visual_history: list[VisualFrame] | None = None
    visual_history_size: int = 1
    motion_overlay: MotionOverlay | None = None
    interrupt_guard: bool = False
    planning_turns: int = 0
    interruptions: int = 0
    episode_guidance: str | None = None

    def set_episode_guidance(self, guidance: str) -> None:
        self.episode_guidance = guidance.strip()[:600] or None

    def configure_episode(self, max_steps: int, decision_budget: int | None = None) -> None:
        """Allow the fastest legal cadence unless an experiment sets an explicit cap."""
        self.max_turns = decision_budget or max_steps
        self.output_token_budget = max(30_000, self.max_turns * 220)

    def reset(self, scene: SceneSpec, seed: int) -> None:
        self.scene, self.actions, self.turn_usages, self.recent_trace = scene, None, [], []
        self.visual_history = []
        self.visual_history_size = max(1, min(4, int(os.environ.get("RACING_VISUAL_HISTORY", "1"))))
        self.motion_overlay = _configured_motion_overlay()
        if self.motion_overlay is not None:
            # The overlay exists to replace a frame stack, not to ride on top of one.
            # Sending both would pay the stack's token cost and then also hand the
            # model a second, redundant account of the same motion.
            self.visual_history_size = 1
        self.interrupt_guard = os.environ.get("RACING_INTERRUPT_GUARD", "0").lower() in {"1", "true", "yes", "on"}
        self.planning_turns = self.interruptions = 0
        self.last_action, self.track_index = None, None
        self.last_keys = ()
        self.planned_segments, self.executed_since_plan = None, []
        self.planned_ticks = self.queued_ticks = 0
        self.chunk_interrupted = False
        self.episode_guidance = None
        self.input_tokens = self.output_tokens = self.latency_ms = 0
        self.uncached_input_tokens = self.cache_creation_input_tokens = self.cache_read_input_tokens = 0

    def act(self, observation: ObservationPacket) -> tuple[Action, DecisionRecord]:
        return self._act(observation, None)

    def act_visual(self, observation: ObservationPacket, frame: VisualFrame) -> tuple[Action, DecisionRecord]:
        return self._act(observation, self.observe_frame(frame))

    def observe_frame(self, frame: VisualFrame, interval_ticks: int = 1) -> VisualFrame:
        """Apply frame-level tools and record history; returns what the model sees.

        `interval_ticks` is how many simulator ticks passed since the previous frame
        handed to this method. The synchronous loop observes every tick, so it is one.
        An asynchronous scheduler only observes when it issues a decision, and the
        motion overlay has to say so: arrows measured across thirty ticks mean
        something different from arrows measured across one.
        """
        assert self.visual_history is not None
        if self.motion_overlay is not None:
            frame = self.motion_overlay.annotate(frame, interval_ticks=interval_ticks)
        self.visual_history.append(frame)
        self.visual_history = self.visual_history[-self.visual_history_size:]
        return frame

    def _act(self, observation: ObservationPacket, visual_frame: VisualFrame | None) -> tuple[Action, DecisionRecord]:
        assert self.scene is not None
        assert self.recent_trace is not None
        self.recent_trace.append({
            "step": observation.step,
            "x": round(observation.proprioception.x, 1),
            "y": round(observation.proprioception.y, 1),
            "heading": round(observation.heading, 1),
            "speed": round(observation.speed, 1),
            "checkpoint": observation.checkpoint_index,
            "requested_action": self.last_action.value if self.last_action else None,
            # W+A and A alone are different physics, so the name of the dominant
            # intent is not enough to tell what the car was actually asked to do.
            "held_keys": list(self.last_keys),
        })
        self.recent_trace = self.recent_trace[-8:]
        provider_usage_record = None
        guard_note = ""
        remaining = self.output_token_budget - self.output_tokens
        context = None
        if self.interrupt_guard and self.actions and visual_frame is not None:
            if remaining <= 0:
                raise PolicyBudgetExhausted(
                    f"policy output-token budget exhausted after {self.output_tokens} tokens; no fallback action was executed"
                )
            context = self._context(observation)
            review, guard_usage = review_racing_action(
                context, [action.name.value for action in self.actions], visual_frame,
                max_tokens=min(80, remaining),
            )
            provider_usage_record = self._record_usage(guard_usage)
            guard_note = f" Interrupt critic: {review.reason[:160]}"
            if review.interrupt:
                context["safety_interrupt"] = {
                    "rejected_action": self.actions[0].name.value,
                    "reason": review.reason,
                    "confidence": review.confidence,
                }
                self.actions = []
                self.chunk_interrupted = True
                self.interruptions += 1
        if not self.actions:
            self.actions, provider_usage_record = self.plan_chunk(
                observation, visual_frame, context=context,
            )
        action = self.actions.pop(0) if self.actions else Action(name=ActionName.IDLE)
        self.record_executed(action)
        return action, DecisionRecord(
            action=action.name, subgoal=self.subgoal, confidence=self.confidence,
            summary=f"Direct racing control. {self.summary[:240]}{guard_note}", candidates=[action.name], provider_usage=provider_usage_record,
        )

    def plan_chunk(
        self, observation: ObservationPacket, visual_frame: VisualFrame | None,
        context: dict | None = None,
    ) -> tuple[list[Action], ProviderTurnUsage]:
        """One planner call, start to finish: observation in, control ticks out."""
        request = self.prepare_decision(observation, visual_frame, context=context)
        plan, usage = self.execute_decision(request)
        return self.apply_decision(request, plan, usage)

    def prepare_decision(
        self, observation: ObservationPacket, visual_frame: VisualFrame | None,
        context: dict | None = None,
    ) -> DecisionRequest:
        """Everything a decision needs, snapshotted off live policy state.

        Split out so an asynchronous scheduler can hand `execute_decision` to a worker
        thread while the simulator keeps advancing. The split is where it is because of
        thread safety, not taste: reading the chunk ledger and writing it back both
        happen here on the caller's thread, so the worker touches no shared state and
        cannot race the ticks executing underneath it.
        """
        assert self.scene is not None
        if self.planning_turns >= self.max_turns:
            raise PolicyBudgetExhausted(
                f"policy call budget exhausted after {self.max_turns} decisions; no fallback action was executed"
            )
        remaining = self.output_token_budget - self.output_tokens
        if remaining <= 0:
            raise PolicyBudgetExhausted(
                f"policy output-token budget exhausted after {self.output_tokens} tokens; no fallback action was executed"
            )
        return DecisionRequest(
            observation=observation, frame=visual_frame,
            frames=list(self.visual_history or []),
            context=context or self._context(observation),
            max_tokens=min(220, remaining),
            operator_guidance=self.episode_guidance,
        )

    def execute_decision(self, request: DecisionRequest):
        """The model call alone. Pure with respect to policy state, so it may run
        on any thread."""
        return plan_racing_actions(
            request.context, max_tokens=request.max_tokens,
            visual_frame=request.frame, visual_frames=request.frames,
            terse=self.terse, operator_guidance=request.operator_guidance,
        )

    def apply_decision(
        self, request: DecisionRequest, plan, usage,
    ) -> tuple[list[Action], ProviderTurnUsage]:
        """Turn a returned plan into a queue of ticks and update the ledgers."""
        expanded = [
            Action(name=ActionName(segment.action), keys=segment.keys)
            for segment in plan.actions for _ in range(segment.steps)
        ]
        nearby_npcs = [
            entity for entity in request.observation.local_entities
            if entity.get("kind") == "npc" and float(entity.get("distance", 999)) < 100
        ]
        track_state = request.context["track_state"]
        high_control_risk = (
            abs(float(track_state["centerline_heading_error"])) > 8
            or abs(float(track_state["signed_lane_offset"])) > float(track_state["safe_lane_half_width"]) * .6
        )
        horizon = self.action_horizon if self.interrupt_guard else 1 if high_control_risk else 2 if nearby_npcs else self.action_horizon
        horizon = max(horizon, self.min_action_horizon)
        queued = expanded[:horizon]
        self.subgoal, self.summary, self.confidence = plan.subgoal, plan.summary, plan.confidence
        self.planning_turns += 1
        # Reset the chunk ledger only once the replacement plan exists, so a failed
        # call cannot erase the account of what the previous chunk did.
        self.planned_segments = [
            {"action": segment.action, "steps": segment.steps, "keys": list(segment.keys)}
            for segment in plan.actions
        ]
        self.planned_ticks, self.queued_ticks = len(expanded), len(queued)
        self.executed_since_plan, self.chunk_interrupted = [], False
        return queued, self._record_usage(usage)

    def record_executed(self, action: Action) -> None:
        """Note a tick that actually reached the simulator."""
        assert self.executed_since_plan is not None
        self.last_action, self.last_keys = action.name, tuple(action.keys)
        self.executed_since_plan.append(
            {"action": action.name.value, "keys": list(action.keys)}
        )

    def _context(self, observation: ObservationPacket) -> dict:
        assert self.scene is not None
        context = racing_public_context(
            self.scene, observation, recent_trace=self.recent_trace,
            track_index_hint=self.track_index,
            previous_chunk=self._previous_chunk(),
            control_budget=self._control_budget(),
        )
        self.track_index = int(context["track_state"]["centerline_index"])
        return context

    def _control_budget(self) -> dict | None:
        """Tell the model how far ahead it has to cover, when that is not one tick."""
        if self.min_action_horizon <= 1:
            return None
        return {
            "ticks_until_your_next_decision": self.min_action_horizon,
            "guidance": (
                "The car executes your ticks and then holds the last keys until your next "
                "decision arrives. Cover at least this many ticks or the car drives itself "
                "on stale input. This overrides the general instruction to return two to "
                "six ticks."
            ),
        }

    def _previous_chunk(self) -> dict | None:
        """What the model last asked for, and what the scheduler did with it."""
        if self.planned_segments is None:
            return None
        executed = list(self.executed_since_plan or [])
        if self.chunk_interrupted:
            ended = "cut short by the safety critic"
        elif self.queued_ticks < self.planned_ticks:
            ended = (
                f"the action horizon queued only {self.queued_ticks} of the "
                f"{self.planned_ticks} ticks you requested"
            )
        else:
            ended = "ran to completion"
        return {
            "requested": self.planned_segments,
            "requested_ticks": self.planned_ticks,
            "queued_ticks": self.queued_ticks,
            "executed": executed,
            "executed_ticks": len(executed),
            "unexecuted_ticks": max(0, self.planned_ticks - len(executed)),
            "ended_because": ended,
        }

    def _record_usage(self, usage) -> ProviderTurnUsage:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.uncached_input_tokens += usage.uncached_input_tokens
        self.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self.cache_read_input_tokens += usage.cache_read_input_tokens
        self.latency_ms += usage.latency_ms
        assert self.turn_usages is not None
        record = ProviderTurnUsage(
            turn=len(self.turn_usages) + 1, provider=usage.provider, model=usage.model,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            uncached_input_tokens=usage.uncached_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            output_budget=self.output_token_budget,
            remaining_output_budget=max(0, self.output_token_budget - self.output_tokens),
            latency_ms=usage.latency_ms,
        )
        self.turn_usages.append(record)
        return record


@dataclass
class ConeVisualRefreshPolicy(AnthropicRacingPolicy):
    """A strict visual-only short-horizon player, refreshed from cone frames."""

    name: str = "vision-2d-direct"
    summary: str = "Awaiting a forward-cone screenshot."
    refresh_ticks: int = 6
    visual_controls: list[dict] | None = None
    ticks_since_frame: int = 0

    def reset(self, scene: SceneSpec, seed: int) -> None:
        super().reset(scene, seed)
        self.actions, self.visual_controls, self.ticks_since_frame = [], [], self.refresh_ticks
        # The image is sampled when the model is about to plan, not every simulator
        # tick.  This makes optical flow genuinely describe consecutive decisions.
        self.motion_overlay = MotionOverlay(color_base=True)
        self.visual_history, self.visual_history_size = [], 1

    def render_frame(self, world) -> VisualFrame:
        from .vision import render_racing_forward_cone
        return render_racing_forward_cone(world)

    def act_visual(self, observation: ObservationPacket, frame: VisualFrame) -> tuple[Action, DecisionRecord]:
        # `observation` is required by the generic policy protocol, but is never
        # read here or passed to the provider. This is intentionally auditable.
        assert self.actions is not None and self.visual_controls is not None
        usage = None
        if not self.actions:
            if self.planning_turns >= self.max_turns:
                raise PolicyBudgetExhausted(f"visual call budget exhausted after {self.max_turns} decisions")
            visual = self.motion_overlay.annotate(frame, interval_ticks=max(1, self.ticks_since_frame))
            from .providers import plan_cone_visual_actions
            assert self.scene is not None
            car_length_px = (
                self.scene.dynamics.vehicle.length_m
                * self.scene.dynamics.pixels_per_meter
            )
            speed_car_lengths_s = (
                observation.speed * self.scene.dynamics.control_hz / car_length_px
            )
            plan, usage = plan_cone_visual_actions(
                visual, previous_controls=self.visual_controls,
                max_tokens=min(220, self.output_token_budget - self.output_tokens),
                speed=speed_car_lengths_s,
                operator_guidance=self.episode_guidance,
            )
            expanded = [
                Action(name=ActionName(segment.action), keys=segment.keys)
                for segment in plan.actions for _ in range(segment.steps)
            ]
            self.actions = expanded[:self.refresh_ticks]
            self.subgoal, self.summary, self.confidence = plan.subgoal, plan.summary, plan.confidence
            self.planning_turns += 1
            self.ticks_since_frame = 0
            usage = self._record_usage(usage)
        action = self.actions.pop(0) if self.actions else Action(name=ActionName.IDLE)
        self.visual_controls.append({"action": action.name.value, "keys": list(action.keys)})
        self.visual_controls = self.visual_controls[-12:]
        self.ticks_since_frame += 1
        return action, DecisionRecord(
            action=action.name, subgoal=self.subgoal, confidence=self.confidence,
            summary="Cone visual-only refresh. " + self.summary[:200], candidates=[action.name], provider_usage=usage,
        )


@dataclass
class PerspectiveVisualTickPolicy(ConeVisualRefreshPolicy):
    """Direct, one-model-call-per-tick player for a first-person 3D camera."""

    name: str = "vision-3d-direct-every-tick"
    summary: str = "Awaiting a first-person 3D screenshot."
    refresh_ticks: int = 1
    max_visual_tokens: int = 90
    expose_road_geometry: bool = False
    road_sensor: object | None = None

    def reset(self, scene: SceneSpec, seed: int) -> None:
        super().reset(scene, seed)
        self.motion_overlay = None
        self.road_sensor = None

    def road_geometry(self, frame: VisualFrame) -> dict | None:
        if not self.expose_road_geometry:
            return None
        if self.road_sensor is None:
            from .reflex.visual_3d import PerspectiveVisionSense
            self.road_sensor = PerspectiveVisionSense()
        sensed = self.road_sensor.update(frame)
        return {
            key: sensed[key] for key in (
                "vision_track_offset", "vision_track_heading", "vision_bend_ahead",
                "vision_bend_severity", "vision_visible_depth", "vision_left_gap",
                "vision_right_gap", "vision_road_contact", "vision_recovery_direction",
                "vision_confidence",
            )
        }

    def render_frame(self, world) -> VisualFrame:
        from .view3d import ViewMode, render_policy_view
        return render_policy_view(world, mode=ViewMode.FIRST_PERSON)

    def act_visual(self, observation: ObservationPacket, frame: VisualFrame) -> tuple[Action, DecisionRecord]:
        assert self.actions is not None and self.visual_controls is not None
        usage = None
        if not self.actions:
            if self.planning_turns >= self.max_turns:
                raise PolicyBudgetExhausted(f"3D visual call budget exhausted after {self.max_turns} decisions")
            from .providers import plan_perspective_visual_actions
            plan, usage = plan_perspective_visual_actions(
                frame, previous_controls=self.visual_controls,
                max_tokens=min(self.max_visual_tokens, self.output_token_budget - self.output_tokens),
                speed_mps=observation.speed,
                road_geometry=self.road_geometry(frame),
                operator_guidance=self.episode_guidance,
            )
            segment = plan.actions[0]
            self.actions = [Action(name=ActionName(segment.action), keys=segment.keys)]
            self.subgoal, self.summary, self.confidence = plan.subgoal, plan.summary, plan.confidence
            self.planning_turns += 1
            usage = self._record_usage(usage)
        action = self.actions.pop(0) if self.actions else Action(name=ActionName.IDLE)
        self.visual_controls.append({"action": action.name.value, "keys": list(action.keys)})
        self.visual_controls = self.visual_controls[-12:]
        return action, DecisionRecord(
            action=action.name, subgoal=self.subgoal, confidence=self.confidence,
            summary="3D visual-only tick control. " + self.summary[:200], candidates=[action.name], provider_usage=usage,
        )


@dataclass
class PerspectiveVisualShortPolicy(PerspectiveVisualTickPolicy):
    """Direct 3D visual player that commits only a short image-conditioned chunk."""

    name: str = "vision-3d-direct-short"
    refresh_ticks: int = 4
    max_visual_tokens: int = 120
    countdown_ticks: int = 30

    def reset(self, scene: SceneSpec, seed: int) -> None:
        super().reset(scene, seed)
        # The deterministic pre-race countdown ignores all keys.  Do not charge a
        # screenshot decision while the player has no agency; this scheduler state
        # is never included in the model prompt or visual contract.
        self.countdown_ticks = 30

    def act_visual(self, observation: ObservationPacket, frame: VisualFrame) -> tuple[Action, DecisionRecord]:
        assert self.actions is not None and self.visual_controls is not None
        usage = None
        if self.countdown_ticks > 0:
            self.countdown_ticks -= 1
            action = Action(name=ActionName.IDLE)
            return action, DecisionRecord(
                action=action.name, subgoal="await the race start", confidence=1.0,
                summary="Pre-race input is disabled.", candidates=[action.name], provider_usage=None,
            )
        if not self.actions:
            if self.planning_turns >= self.max_turns:
                raise PolicyBudgetExhausted(f"3D visual call budget exhausted after {self.max_turns} decisions")
            from .providers import plan_perspective_visual_actions
            plan, usage = plan_perspective_visual_actions(
                frame, previous_controls=self.visual_controls,
                max_tokens=min(self.max_visual_tokens, self.output_token_budget - self.output_tokens),
                max_actions=self.refresh_ticks,
                speed_mps=observation.speed,
                road_geometry=self.road_geometry(frame),
                operator_guidance=self.episode_guidance,
            )
            selected = [
                Action(name=ActionName(segment.action), keys=segment.keys)
                for segment in plan.actions
            ][:self.refresh_ticks]
            # A straight throttle answer can safely be held until the next visual
            # refresh.  Steering is different: holding w+a/w+d blind for four ticks
            # compounds yaw and is exactly what caused the 3D player to miss the
            # opening bend.  A one-action steering answer therefore requests the
            # next screenshot immediately.  This is scheduling, not a controller:
            # the harness never changes a model-selected key state.
            steering = bool(selected and ({"a", "d"} & set(selected[-1].keys)))
            if selected and (len(selected) > 1 or not steering):
                selected.extend([selected[-1]] * (self.refresh_ticks - len(selected)))
            self.actions = selected
            self.subgoal, self.summary, self.confidence = plan.subgoal, plan.summary, plan.confidence
            self.planning_turns += 1
            usage = self._record_usage(usage)
        action = self.actions.pop(0) if self.actions else Action(name=ActionName.IDLE)
        self.visual_controls.append({"action": action.name.value, "keys": list(action.keys)})
        self.visual_controls = self.visual_controls[-12:]
        return action, DecisionRecord(
            action=action.name, subgoal=self.subgoal, confidence=self.confidence,
            summary="3D visual-only short control. " + self.summary[:200], candidates=[action.name], provider_usage=usage,
        )


@dataclass
class PerspectiveVisualShortSpeedRoadPolicy(PerspectiveVisualShortPolicy):
    """Short-horizon player with physical speed plus screenshot-derived road geometry."""

    name: str = "vision-3d-direct-short-features"
    expose_road_geometry: bool = True


@dataclass
class PredictiveConeSkillPolicy(AnthropicRacingPolicy):
    """The evaluated 2D predictive-overlap arm as a first-class player policy."""

    name: str = "vision-2d-predictive-skills"
    summary: str = "Camera-grounded stabilize skill active while the first plan is in flight."
    realtime_clock: str = "wall"
    decision_interval_ticks: int = 70
    initial_latency_ticks: int = 18
    prediction_horizon_ticks: int = 18
    control_hz: int = 10
    driver: object | None = None
    prediction_diagnostics: list[dict] | None = None
    skill_failure_candidates: list[dict] | None = None
    previous_road_contact: bool | None = None
    rejected_predictions: int = 0
    aggression: float = .78

    def set_aggression(self, value: float) -> None:
        self.aggression = max(0.0, min(1.0, float(value)))

    def reset(self, scene: SceneSpec, seed: int) -> None:
        from .pipeline2d import ConeSkillDriver

        super().reset(scene, seed)
        self.motion_overlay = None
        self.visual_history_size = 1
        self.driver = ConeSkillDriver(scene, aggression=self.aggression)
        self.prediction_diagnostics = []
        self.skill_failure_candidates = []
        self.previous_road_contact = None
        self.rejected_predictions = 0
        self.control_hz = scene.dynamics.control_hz

    def render_frame(self, world) -> VisualFrame:
        from .vision import render_racing_forward_cone

        return render_racing_forward_cone(world)

    def set_prediction_horizon(
        self, ticks: int, tick_seconds: float | None = None, submitted_tick: int | None = None,
    ) -> None:
        self.prediction_horizon_ticks = max(1, min(80, int(ticks)))

    @staticmethod
    def _speed(observation: ObservationPacket) -> float:
        dynamics = observation.dynamics
        car_length = dynamics.vehicle.length_m * dynamics.pixels_per_meter
        return observation.speed * dynamics.control_hz / max(1e-6, car_length)

    def _sense(self, observation: ObservationPacket, frame: VisualFrame) -> dict:
        assert self.driver is not None
        state, _ = self.driver.observe_frame(frame, self._speed(observation))
        return state

    def prepare_decision(
        self, observation: ObservationPacket, visual_frame: VisualFrame | None,
        context: dict | None = None,
    ) -> DecisionRequest:
        from dataclasses import asdict

        if visual_frame is None:
            raise PolicySessionError("predictive 2D skills require a forward-cone frame")
        assert self.driver is not None
        public = {
            "public_state": self._sense(observation, visual_frame),
            "active_skill": asdict(self.driver.active),
            "recent_controls": list(self.driver.recent_controls[-8:]),
            "activation_horizon_ticks": self.prediction_horizon_ticks,
            "control_hz": self.control_hz,
            "driving_aggression": self.aggression,
        }
        return super().prepare_decision(observation, visual_frame, context=public)

    def execute_decision(self, request: DecisionRequest):
        from .providers import plan_cone_driving_skill

        context = request.context
        return plan_cone_driving_skill(
            request.frame, public_state=context["public_state"],
            active_skill=context["active_skill"], recent_controls=context["recent_controls"],
            activation_horizon_ticks=context["activation_horizon_ticks"],
            control_hz=context["control_hz"],
            driving_aggression=context["driving_aggression"],
            max_tokens=min(190, request.max_tokens),
        )

    def validate_decision(
        self, request: DecisionRequest, plan, observation: ObservationPacket,
        frame: VisualFrame | None,
    ) -> tuple[bool, str]:
        from .pipeline2d import prediction_matches

        if frame is None:
            return False, "no activation frame"
        accepted, diagnostic = prediction_matches(plan, self._sense(observation, frame))
        diagnostic.update({"skill": plan.skill, "reason": (
            "prediction within public-state tolerances" if accepted
            else "activation state diverged from the model prediction"
        )})
        assert self.prediction_diagnostics is not None
        self.prediction_diagnostics.append(diagnostic)
        return accepted, diagnostic["reason"]

    def apply_decision(self, request: DecisionRequest, plan, usage):
        assert self.driver is not None
        self.driver.install(plan, request.observation.step, aggression=self.aggression)
        self.planning_turns += 1
        self.subgoal = f"run {plan.skill} for the predicted activation state"
        self.summary = plan.summary
        self.confidence = .7
        self.planned_segments = [{
            "action": "skill", "steps": 0, "skill": plan.skill,
            "target_speed": plan.target_speed, "target_offset": plan.target_offset,
            "turn_direction": plan.turn_direction,
            "aggression": self.aggression,
        }]
        self.planned_ticks = self.queued_ticks = 0
        self.executed_since_plan, self.chunk_interrupted = [], False
        return [], self._record_usage(usage)

    def reject_decision(self, request: DecisionRequest, plan, usage, reason: str) -> None:
        self.planning_turns += 1
        self.rejected_predictions += 1
        self.summary = f"Rejected {plan.skill}: {reason}; continuing the active skill."
        self._record_usage(usage)

    def tick_action_visual(self, observation: ObservationPacket, frame: VisualFrame) -> Action:
        assert self.driver is not None
        state = self._sense(observation, frame)
        contact = bool(state["road_contact"])
        if self.previous_road_contact is True and not contact:
            assert self.skill_failure_candidates is not None
            candidate = _visible_skill_failure("2d", self.driver.active.name, state)
            candidate["tick"] = observation.step
            self.skill_failure_candidates.append(candidate)
            self.skill_failure_candidates = self.skill_failure_candidates[-32:]
        self.previous_road_contact = contact
        return self.driver.tick_state(state)

    def tick_action(self, observation: ObservationPacket) -> Action:
        raise PolicySessionError("predictive 2D skills require a fresh cone frame every tick")

    def realtime_metrics(self) -> dict:
        from .context_loader import player_context_provenance

        diagnostics = list(self.prediction_diagnostics or [])
        return {
            "prediction_attempts": len(diagnostics),
            "prediction_accepts": sum(bool(item["accepted"]) for item in diagnostics),
            "prediction_rejections": self.rejected_predictions,
            "skill_activations": list(getattr(self.driver, "activations", []) or []),
            "prediction_diagnostics": diagnostics,
            "player_aggression": self.aggression,
            "last_control_terms": dict(getattr(self.driver, "last_control_terms", {}) or {}),
            "skill_failure_candidates": list(self.skill_failure_candidates or []),
            "context_pack": player_context_provenance("2d"),
        }

    def act_visual(self, observation: ObservationPacket, frame: VisualFrame):
        request = self.prepare_decision(observation, frame)
        plan, usage = self.execute_decision(request)
        self.apply_decision(request, plan, usage)
        action = self.tick_action_visual(observation, frame)
        self.record_executed(action)
        return action, DecisionRecord(
            action=action.name, subgoal=self.subgoal, confidence=self.confidence,
            summary="Blocking predictive-skill compatibility path. " + self.summary[:180],
            candidates=[action.name], provider_usage=self.turn_usages[-1],
        )


@dataclass
class PredictiveVisualSkillPolicy(AnthropicRacingPolicy):
    """Predict response-time visual state, then activate a feedback skill.

    This is the latency-compensated arm.  The model sees a first-person frame,
    physical speed, prior requested keys, and pixel-derived road measurements.
    The active primitive receives the same contract on each control tick.  No
    route geometry or other engine telemetry enters either layer.
    """

    name: str = "vision-3d-predictive-skills"
    summary: str = "Camera-grounded stabilize skill active while the first plan is in flight."
    realtime_clock: str = "wall"
    """This policy is only meaningful when planning is off the simulator thread."""
    decision_interval_ticks: int = 70
    """Wait seven seconds after a response lands before refreshing its high-level skill."""
    initial_latency_ticks: int = 18
    """Conservative first-call prediction horizon at the 10 Hz control rate."""
    prediction_horizon_ticks: int = 12
    control_hz: int = 10
    sensor: object | None = None
    skill_controller: object | None = None
    active_skill: object | None = None
    latest_sense: dict | None = None
    visual_controls: list[dict] | None = None
    prediction_diagnostics: list[dict] | None = None
    skill_activations: list[dict] | None = None
    skill_failure_candidates: list[dict] | None = None
    previous_road_contact: bool | None = None
    rejected_predictions: int = 0
    last_control_terms: dict | None = None
    aggression: float = .78

    def set_aggression(self, value: float) -> None:
        self.aggression = max(0.0, min(1.0, float(value)))

    def reset(self, scene: SceneSpec, seed: int) -> None:
        from .predictive import SKILL_DEFAULTS, VisualSkillController
        from .reflex.visual_3d import PerspectiveVisionSense

        super().reset(scene, seed)
        self.sensor = PerspectiveVisionSense()
        self.skill_controller = VisualSkillController()
        default = SKILL_DEFAULTS["stabilize"]
        self.active_skill = type(default)(
            name=default.name, target_speed=default.target_speed,
            target_offset=default.target_offset, turn_direction=default.turn_direction,
            aggression=self.aggression,
        )
        self.latest_sense = None
        self.visual_controls, self.prediction_diagnostics, self.skill_activations = [], [], []
        self.skill_failure_candidates = []
        self.previous_road_contact = None
        self.rejected_predictions = 0
        self.last_control_terms = None
        self.control_hz = scene.dynamics.control_hz

    def render_frame(self, world) -> VisualFrame:
        from .view3d import ViewMode, render_policy_view

        return render_policy_view(world, mode=ViewMode.FIRST_PERSON)

    def set_prediction_horizon(
        self, ticks: int, tick_seconds: float | None = None, submitted_tick: int | None = None,
    ) -> None:
        self.prediction_horizon_ticks = max(1, min(80, int(ticks)))

    def _sense(self, frame: VisualFrame) -> dict:
        assert self.sensor is not None
        self.latest_sense = self.sensor.update(frame)
        return self.latest_sense

    def prepare_decision(
        self, observation: ObservationPacket, visual_frame: VisualFrame | None,
        context: dict | None = None,
    ) -> DecisionRequest:
        from .predictive import public_visual_state

        if visual_frame is None:
            raise PolicySessionError("predictive visual skills require a first-person frame")
        sense = self._sense(visual_frame)
        assert self.active_skill is not None
        public = {
            "public_state": public_visual_state(observation.speed, sense),
            "active_skill": asdict(self.active_skill),
            "previous_controls": list((self.visual_controls or [])[-8:]),
            "activation_horizon_ticks": self.prediction_horizon_ticks,
            "control_hz": self.control_hz,
            "driving_aggression": self.aggression,
        }
        return super().prepare_decision(observation, visual_frame, context=public)

    def execute_decision(self, request: DecisionRequest):
        context = request.context
        return plan_predictive_driving_skill(
            request.frame,
            public_state=context["public_state"],
            active_skill=context["active_skill"],
            previous_controls=context["previous_controls"],
            activation_horizon_ticks=context["activation_horizon_ticks"],
            control_hz=context["control_hz"],
            driving_aggression=context["driving_aggression"],
            max_tokens=min(180, request.max_tokens),
        )

    def validate_decision(
        self, request: DecisionRequest, plan, observation: ObservationPacket,
        frame: VisualFrame | None,
    ) -> tuple[bool, str]:
        """Compare the model's activation prediction to current public state."""
        from .predictive import public_visual_state

        if frame is None:
            return False, "no activation frame"
        actual = public_visual_state(observation.speed, self._sense(frame))
        predicted = plan.predicted.model_dump()
        errors = {
            "speed": abs(actual["speed"] - predicted["speed"]),
            "road_offset": abs(actual["road_offset"] - predicted["road_offset"]),
            "bend_ahead": abs(actual["bend_ahead"] - predicted["bend_ahead"]),
        }
        tolerances = {
            # A language-model projection is deliberately coarse. These floors
            # reject qualitatively obsolete plans without demanding localization.
            "speed": max(1.25, float(plan.speed_tolerance)),
            "road_offset": max(0.65, float(plan.offset_tolerance)),
            "bend_ahead": max(0.55, float(plan.bend_tolerance)),
        }
        # Bend magnitude is noisy under perspective and hills; the useful coarse
        # prediction is its class. Reject an answer only when a strong visible
        # turn has reversed direction, not when its estimated magnitude changed.
        opposite_bend = (
            predicted["bend_ahead"] > 0.35 and actual["bend_ahead"] < -0.35
        ) or (
            predicted["bend_ahead"] < -0.35 and actual["bend_ahead"] > 0.35
        )
        # A low-confidence crest can briefly hide the ego road patch. Treat that
        # as unknown, not proof that the car left the road.
        contact_match = (
            actual["road_contact"] == predicted["road_contact"]
            or actual["confidence"] < 0.5
        )
        accepted = (
            contact_match
            and errors["speed"] <= tolerances["speed"]
            and errors["road_offset"] <= tolerances["road_offset"]
            and not opposite_bend
        )
        reason = (
            "prediction within public-state tolerances" if accepted
            else "activation state diverged from the model prediction"
        )
        assert self.prediction_diagnostics is not None
        self.prediction_diagnostics.append({
            "submitted_state": request.context["public_state"],
            "predicted_state": predicted,
            "actual_state": actual,
            "errors": {key: round(value, 4) for key, value in errors.items()},
            "tolerances": tolerances,
            "contact_match": contact_match,
            "accepted": accepted,
            "skill": plan.skill,
            "reason": reason,
        })
        return accepted, reason

    def apply_decision(self, request: DecisionRequest, plan, usage):
        from .predictive import DrivingSkill

        self.active_skill = DrivingSkill(
            name=plan.skill, target_speed=plan.target_speed,
            target_offset=plan.target_offset, turn_direction=plan.turn_direction,
            aggression=self.aggression,
        ).clamped()
        self.planning_turns += 1
        self.subgoal = f"run {self.active_skill.name} for the predicted activation state"
        self.summary = plan.summary
        self.confidence = 0.7
        assert self.skill_activations is not None
        self.skill_activations.append({
            "turn": self.planning_turns, **asdict(self.active_skill),
            "prediction": plan.predicted.model_dump(),
        })
        self.planned_segments = [{"action": "skill", "steps": 0, **asdict(self.active_skill)}]
        self.planned_ticks = self.queued_ticks = 0
        self.executed_since_plan, self.chunk_interrupted = [], False
        return [], self._record_usage(usage)

    def reject_decision(self, request: DecisionRequest, plan, usage, reason: str) -> None:
        """Charge a rejected call without replacing the still-live primitive."""
        self.planning_turns += 1
        self.rejected_predictions += 1
        self.summary = f"Rejected {plan.skill}: {reason}; continuing {self.active_skill.name}."
        self._record_usage(usage)

    def tick_action_visual(self, observation: ObservationPacket, frame: VisualFrame) -> Action:
        assert self.skill_controller is not None and self.active_skill is not None
        sense = self._sense(frame)
        from .predictive import public_visual_state

        public = public_visual_state(observation.speed, sense)
        contact = bool(public["road_contact"])
        if self.previous_road_contact is True and not contact:
            assert self.skill_failure_candidates is not None
            candidate = _visible_skill_failure("3d", self.active_skill.name, public)
            candidate["tick"] = observation.step
            self.skill_failure_candidates.append(candidate)
            self.skill_failure_candidates = self.skill_failure_candidates[-32:]
        self.previous_road_contact = contact
        action, terms = self.skill_controller.act(
            self.active_skill, observation.speed, sense,
        )
        self.last_control_terms = terms
        assert self.visual_controls is not None
        self.visual_controls.append({
            "action": action.name.value, "keys": list(action.keys),
            "skill": self.active_skill.name,
        })
        self.visual_controls = self.visual_controls[-16:]
        return action

    def tick_action(self, observation: ObservationPacket) -> Action:
        raise PolicySessionError("predictive visual skills require a fresh frame every tick")

    def realtime_metrics(self) -> dict:
        from .context_loader import player_context_provenance

        diagnostics = list(self.prediction_diagnostics or [])
        return {
            "prediction_attempts": len(diagnostics),
            "prediction_accepts": sum(bool(item["accepted"]) for item in diagnostics),
            "prediction_rejections": self.rejected_predictions,
            "skill_activations": list(self.skill_activations or []),
            "prediction_diagnostics": diagnostics,
            "player_aggression": self.aggression,
            "last_control_terms": dict(self.last_control_terms or {}),
            "skill_failure_candidates": list(self.skill_failure_candidates or []),
            "context_pack": player_context_provenance("3d"),
        }

    def act_visual(self, observation: ObservationPacket, frame: VisualFrame):
        """Blocking compatibility path; realtime.py provides the intended overlap."""
        request = self.prepare_decision(observation, frame)
        plan, usage = self.execute_decision(request)
        self.apply_decision(request, plan, usage)
        action = self.tick_action_visual(observation, frame)
        self.record_executed(action)
        return action, DecisionRecord(
            action=action.name, subgoal=self.subgoal, confidence=self.confidence,
            summary="Blocking predictive-skill compatibility path. " + self.summary[:180],
            candidates=[action.name], provider_usage=self.turn_usages[-1],
        )


@dataclass
class AnthropicHierarchicalRacingPolicy(AnthropicRacingPolicy):
    """Claude chooses a lane and a speed; a controller holds it every tick.

    Inherits the budget, usage, frame and chunk machinery from the direct driver so
    the two are measured on the same ledger, and replaces what a decision *is*: two
    numbers refreshed every dozen ticks rather than a control sequence that runs out.
    Because the fast layer acts every tick from the current intent, this policy never
    starves — the cost of latency moves from ticks without input to ticks spent
    holding a stale lane.
    """

    name: str = "telemetry-hierarchical"
    intent: Intent | None = None
    controller: LocalIntentController | None = None
    intents_applied: int = 0
    last_control_terms: dict | None = None

    def reset(self, scene: SceneSpec, seed: int) -> None:
        super().reset(scene, seed)
        self.controller = LocalIntentController()
        self.intent, self.last_control_terms = None, None
        self.intents_applied = 0

    def execute_decision(self, request: DecisionRequest):
        return plan_racing_intent(
            request.context, visual_frame=request.frame, visual_frames=request.frames,
            operator_guidance=request.operator_guidance,
        )

    def apply_decision(self, request: DecisionRequest, plan, usage):
        """Store the new intent. The action queue stays empty by design."""
        self.intent = Intent(
            target_speed=float(plan.target_speed), lane_offset=float(plan.lane_offset),
        )
        self.intents_applied += 1
        self.planning_turns += 1
        self.subgoal = f"hold lane {self.intent.lane_offset:+.0f} at {self.intent.target_speed:.1f}"
        self.summary = (
            f"Strategy layer chose lane {self.intent.lane_offset:+.1f} and speed "
            f"{self.intent.target_speed:.1f}; the controller holds it every tick."
        )
        self.confidence = 0.6
        self.planned_segments = [{
            "action": "intent", "steps": 0,
            "target_speed": self.intent.target_speed, "lane_offset": self.intent.lane_offset,
        }]
        self.planned_ticks = self.queued_ticks = 0
        self.executed_since_plan, self.chunk_interrupted = [], False
        return [], self._record_usage(usage)

    def tick_action(self, observation: ObservationPacket) -> Action:
        """Close the steering loop at tick rate from local proprioception only."""
        assert self.scene is not None and self.controller is not None
        full = racing_local_state(self.scene, observation, self.track_index)
        self.track_index = int(full["centerline_index"])
        if self.intent is None:
            # No strategy has arrived yet. Creeping straight is the honest default: it
            # is what the fast layer can justify without any instruction.
            intent = Intent(target_speed=STARTING_TARGET_SPEED, lane_offset=0.0)
        else:
            intent = self.intent
        action, terms = self.controller.act(
            intent, observation.speed,
            {key: full[key] for key in LOCAL_OBSERVATION_FIELDS},
        )
        self.last_control_terms = terms
        return action

    def _control_budget(self) -> dict | None:
        """The strategy layer is never asked to cover ticks; the controller does."""
        return None


@dataclass
class AnthropicStrategyRacingPolicy:
    """One pre-race Claude strategy call with deterministic intent execution."""

    name: str = "telemetry-strategy"
    scene: SceneSpec | None = None
    controller: RacingIntentController | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: int = 0
    turn_usages: list[ProviderTurnUsage] | None = None
    output_token_budget: int = 900
    max_turns: int = 1

    def reset(self, scene: SceneSpec, seed: int) -> None:
        self.scene, self.controller, self.turn_usages = scene, None, []
        self.input_tokens = self.output_tokens = self.latency_ms = 0
        self.uncached_input_tokens = self.cache_creation_input_tokens = self.cache_read_input_tokens = 0

    def act(self, observation: ObservationPacket) -> tuple[Action, DecisionRecord]:
        assert self.scene is not None
        usage_record = None
        if self.controller is None:
            strategy, usage = plan_racing_strategy(racing_strategy_context(self.scene), max_tokens=self.output_token_budget)
            intents = [intent.model_dump() for intent in strategy.sectors]
            self.controller = RacingIntentController(self.scene, intents, strategy.summary)
            self.input_tokens, self.output_tokens, self.latency_ms = usage.input_tokens, usage.output_tokens, usage.latency_ms
            self.uncached_input_tokens = usage.uncached_input_tokens
            self.cache_creation_input_tokens = usage.cache_creation_input_tokens
            self.cache_read_input_tokens = usage.cache_read_input_tokens
            usage_record = ProviderTurnUsage(
                turn=1, provider=usage.provider, model=usage.model,
                input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                uncached_input_tokens=usage.uncached_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                output_budget=self.output_token_budget,
                remaining_output_budget=max(0, self.output_token_budget - usage.output_tokens),
                latency_ms=usage.latency_ms,
            )
            assert self.turn_usages is not None
            self.turn_usages.append(usage_record)
        action, decision = self.controller.act(observation)
        if usage_record is not None:
            decision = decision.model_copy(update={"provider_usage": usage_record})
        return action, decision


@dataclass
class HttpKeyboardPolicy:
    """Third-party session adapter: scene/observation in, held WASD state out."""

    endpoint: str
    name: str = "external-telemetry-player"
    timeout_seconds: float = 30.0
    episode_id: str | None = None
    queued_actions: list[Action] | None = None
    previous_keys: list[str] | None = None

    def reset(self, scene: SceneSpec, seed: int) -> None:
        self.episode_id = f"episode-{uuid.uuid4().hex[:12]}"
        self.queued_actions, self.previous_keys = [], []
        payload = PolicyReset(
            episode_id=self.episode_id, seed=seed, scene=scene,
            capabilities=PolicyCapabilities(observation_modalities=["rgb", "telemetry"]),
        )
        self._post("reset", payload.model_dump(mode="json"))

    def act(self, observation: ObservationPacket) -> tuple[Action, DecisionRecord]:
        return self._act(observation, None)

    def act_visual(self, observation: ObservationPacket, frame: VisualFrame) -> tuple[Action, DecisionRecord]:
        return self._act(observation, frame)

    def _act(self, observation: ObservationPacket, frame: VisualFrame | None) -> tuple[Action, DecisionRecord]:
        assert self.episode_id is not None and self.queued_actions is not None
        if not self.queued_actions:
            payload = PolicyStep(
                episode_id=self.episode_id, observation=observation,
                frame=frame,
                previous_keys=self.previous_keys or [],
            )
            response = PolicyAction.model_validate(self._post("act", payload.model_dump(mode="json")))
            if response.episode_id != self.episode_id:
                raise PolicySessionError("external policy returned the wrong episode_id")
            keys = response.control.keys
            self.previous_keys = list(keys)
            self.queued_actions = [Action(keys=keys) for _ in range(response.control.repeat)]
        action = self.queued_actions.pop(0)
        primary = _primary_name(action.keys)
        return action, DecisionRecord(
            action=primary, subgoal="external keyboard policy", confidence=1,
            summary=f"External policy holds {action.keys or ['no keys']}.", candidates=[primary],
        )

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.endpoint.rstrip('/')}/{path}",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            raise PolicySessionError(f"external policy unavailable during {path}: {error}") from error
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise PolicySessionError(f"external policy returned invalid JSON during {path}") from error


def _configured_motion_overlay() -> MotionOverlay | None:
    """Build the optical-flow tool if the episode asked for it.

    `RACING_MOTION_OVERLAY=1` draws the arrows on the grayscale frame, which is the
    cheapest encoding but discards the palette the driver prompt relies on to tell an
    opponent from a barrier. `=color` keeps the color frame underneath and pays a few
    more tokens for it, so the two costs can be measured against each other.
    """
    setting = os.environ.get("RACING_MOTION_OVERLAY", "0").strip().lower()
    if setting in {"", "0", "false", "no", "off"}:
        return None
    if setting not in {"1", "true", "yes", "on", "gray", "grayscale", "color", "colour", "rgb"}:
        raise ValueError(
            f"RACING_MOTION_OVERLAY must be off, 1/gray, or color; got {setting!r}"
        )
    return MotionOverlay(color_base=setting in {"color", "colour", "rgb"})


def _primary_name(keys: list[str]) -> ActionName:
    if "space" in keys and "w" in keys and "a" not in keys and "d" not in keys:
        return ActionName.NITRO
    if "a" in keys:
        return ActionName.LEFT
    if "d" in keys:
        return ActionName.RIGHT
    if "w" in keys:
        return ActionName.FORWARD
    if "s" in keys:
        return ActionName.BACKWARD
    return ActionName.IDLE


@dataclass
class ReflexRacingDriver:
    """The reflex harness as a selectable driver.

    Deliberately not a `PlayerPolicy`: it has no per-tick `act`, because the entire point is
    that the model does not act per tick. It writes a controller, declares what should wake
    it, and hands control back. `service.run` detects `run_episode` and delegates the whole
    episode rather than pumping this object once per tick.
    """

    name: str = "telemetry-reflex"
    model: str | None = None
    max_wakes: int = 6
    rehearsal_budget: int = 5
    latency: str = "measured"
    verbose: bool = False
    subgoal: str = "write a controller, then let it drive"
    summary: str = "Reflex driver: occasional authorship, tick-rate control by generated code."
    confidence: float = 0.6
    vision_only: bool = False
    visual_mode: str | None = None

    def run_episode(self, world, *, max_steps: int):
        from .reflex.episode import run_reflex_episode

        return run_reflex_episode(
            world,
            model=self.model or configured_model("ANTHROPIC_PLAYER_MODEL", "claude-sonnet-5"),
            max_steps=max_steps, max_wakes=self.max_wakes,
            rehearsal_budget=self.rehearsal_budget, latency=self.latency,
            verbose=self.verbose, vision_only=self.vision_only,
            visual_mode=self.visual_mode,
        )


def built_in_policies() -> dict[str, PlayerPolicy]:
    policies: dict[str, PlayerPolicy] = {
        "oracle-racing-line": RacingLineController(),
        "telemetry-direct": AnthropicRacingPolicy(),
        "telemetry-strategy": AnthropicStrategyRacingPolicy(),
        "telemetry-hierarchical": AnthropicHierarchicalRacingPolicy(),
        "telemetry-reflex": ReflexRacingDriver(),
        "vision-reflex-sim-rehearsal": ReflexRacingDriver(
            name="vision-reflex-sim-rehearsal", vision_only=True,
            summary="Forward-cone reflex driver with physical speed as its only engine telemetry.",
        ),
        "vision-2d-predictive-skills": PredictiveConeSkillPolicy(),
        "vision-2d-direct": ConeVisualRefreshPolicy(),
        "vision-3d-direct-every-tick": PerspectiveVisualTickPolicy(),
        "vision-3d-direct-short": PerspectiveVisualShortPolicy(),
        "vision-3d-direct-short-features": PerspectiveVisualShortSpeedRoadPolicy(),
        "vision-3d-predictive-skills": PredictiveVisualSkillPolicy(),
        "baseline-constant-intent": ConstantIntentPolicy(),
        "baseline-random": WandererPolicy(),
    }
    if os.environ.get("EXTERNAL_PLAYER_URL"):
        policies["external-telemetry-player"] = HttpKeyboardPolicy(os.environ["EXTERNAL_PLAYER_URL"])
    return policies

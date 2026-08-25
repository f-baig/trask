"""Per-tick motor control from local proprioception, for hierarchical driving.

A language model on a 10 Hz engine cannot close a steering loop. Even terse and
pipelined, a decision costs about thirteen ticks, so between decisions the car is
either frozen on stale keys or coasting. The fix is the standard one from real
autonomous racing stacks: split the problem. The slow layer chooses *where and how
fast* — a lane and a target speed. The fast layer holds that choice every tick.

The privilege boundary is the whole design constraint here. `RacingIntentController`
already exists and is deliberately not reused: it steers toward
`_offset_track_point(scene.track_centerline, nearest + lookahead, lane)`, so it knows
the geometry of corners it cannot see. Handing that to a forward-cone policy would
make the experiment measure a centerline follower, because the cone view exists
precisely to withhold global route knowledge.

So this controller reads only `racing_local_state`, which is the same function that
fills the `track_state` the prompt receives. It cannot look ahead, cannot know a
corner is coming, and cannot choose a racing line. It can hold a lane and a speed.
Everything that requires seeing the future stays the model's job, and any success at
cornering remains attributable to the model.

What that leaves out is deliberate for a first version: no hazard reaction. A barrier
is avoided by the model choosing a lane around it, not by the controller swerving. If
barrier contacts and recovery time become the failure mode, that is a real result about the split
rather than something to paper over inside the fast layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Action, ActionName, DecisionRecord


LOCAL_OBSERVATION_FIELDS = frozenset({
    "signed_lane_offset", "centerline_heading_error", "safe_lane_half_width", "on_track",
})
"""Every field the fast layer is allowed to read, and nothing else.

Checked at runtime rather than trusted. A controller that quietly started reading
`centerline_index` would be sampling global route position, and the whole
attribution argument above would silently stop holding.
"""

STEER_DEADBAND_DEGREES = 5.0
LANE_GAIN = 0.15
"""Cross-track gain, in the Stanley sense: how hard a lane error becomes steering.

Tuned against the free constant-intent baseline, and the value matters more than it
looks. The action space is three discrete steering states with a five-degree
deadband, so a gain that turns a half-pixel lane error into a seven-degree signal
does not steer proportionally — it chatters left and right every tick. At 1.6 the
controller logged fifteen steering reversals a lap and crashed outright at speed 6;
at 0.15 it logs none and finishes the same lap in 309 ticks instead of 391.
"""
SPEED_FLOOR = 2.5
"""Speed used in the Stanley denominator, so a near-stopped car still corrects."""
SPEED_DEADBAND = 0.35
BRAKE_MARGIN = 0.7
MAXIMUM_TARGET_SPEED = 10.0
MINIMUM_TARGET_SPEED = 0.0


@dataclass(frozen=True)
class Intent:
    """What the slow layer decides: a lane to hold and a speed to hold it at.

    `lane_offset` is in pixels right of the road centre, matching the sign of
    `signed_lane_offset` — steering right increases both.
    """

    target_speed: float
    lane_offset: float

    def clamped(self, safe_half_width: float) -> "Intent":
        limit = max(0.0, safe_half_width)
        return Intent(
            target_speed=max(MINIMUM_TARGET_SPEED, min(MAXIMUM_TARGET_SPEED, self.target_speed)),
            lane_offset=max(-limit, min(limit, self.lane_offset)),
        )


@dataclass
class LocalIntentController:
    """Hold an intent using local proprioception only.

    Steering is a Stanley law: align with the road, plus a cross-track term that
    scales down with speed so a lane correction does not become a spin at pace. The
    longitudinal law is a deadbanded three-state controller, because the action space
    is keys rather than a continuous pedal.
    """

    lane_gain: float = LANE_GAIN
    steer_deadband_degrees: float = STEER_DEADBAND_DEGREES

    def act(self, intent: Intent, speed: float, local: dict) -> tuple[Action, dict]:
        """Return the keys for this tick, plus the terms that produced them."""
        unknown = set(local) - LOCAL_OBSERVATION_FIELDS
        if unknown:
            raise ValueError(
                f"the fast controller may not read {sorted(unknown)}; it would be "
                "using route knowledge the cone-view policy never receives"
            )
        missing = LOCAL_OBSERVATION_FIELDS - set(local)
        if missing:
            raise ValueError(f"local state is missing {sorted(missing)}")

        target = intent.clamped(float(local["safe_lane_half_width"]))
        heading_error = float(local["centerline_heading_error"])
        # Positive lane error means the car sits right of where it should be, and the
        # correction is to steer left, so it enters with a negative sign.
        lane_error = float(local["signed_lane_offset"]) - target.lane_offset
        cross_track = math.degrees(math.atan2(
            self.lane_gain * lane_error, max(SPEED_FLOOR, speed),
        ))
        steer_signal = heading_error - cross_track

        braking = speed > target.target_speed + BRAKE_MARGIN
        throttling = speed < target.target_speed - SPEED_DEADBAND
        longitudinal = "s" if braking else "w" if throttling else None
        if steer_signal > self.steer_deadband_degrees:
            name, keys = ActionName.RIGHT, [*filter(None, [longitudinal]), "d"]
        elif steer_signal < -self.steer_deadband_degrees:
            name, keys = ActionName.LEFT, [*filter(None, [longitudinal]), "a"]
        elif braking:
            name, keys = ActionName.BACKWARD, ["s"]
        elif throttling:
            name, keys = ActionName.FORWARD, ["w"]
        else:
            name, keys = ActionName.IDLE, []
        return Action(name=name, keys=keys), {
            "steer_signal": round(steer_signal, 1),
            "heading_error": round(heading_error, 1),
            "lane_error": round(lane_error, 1),
            "target_speed": round(target.target_speed, 1),
            "target_lane": round(target.lane_offset, 1),
        }


@dataclass
class ConstantIntentPolicy:
    """The fast layer alone, holding a fixed intent, with no model in the loop.

    This is the control arm and it is not optional. A hierarchical driver that
    completes a lap proves nothing on its own, because a lane-keeping controller can
    complete a lap unaided — the question is only ever whether the model's choice of
    lane and speed beats a constant one. Without this arm, the controller's
    competence would be reported as the model's.
    """

    name: str = "baseline-constant-intent"
    target_speed: float = 6.0
    lane_offset: float = 0.0
    controller: LocalIntentController | None = None
    scene: object | None = None
    track_index: int | None = None
    subgoal: str = "hold a constant lane and speed"
    summary: str = "Fast controller only; no model decides anything."
    confidence: float = 1.0
    planning_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    terse: bool = False
    min_action_horizon: int = 0

    def reset(self, scene, seed: int) -> None:
        self.scene, self.controller, self.track_index = scene, LocalIntentController(), None

    def tick_action(self, observation) -> Action:
        from .racing import racing_local_state

        assert self.scene is not None and self.controller is not None
        full = racing_local_state(self.scene, observation, self.track_index)
        self.track_index = int(full["centerline_index"])
        action, _ = self.controller.act(
            Intent(self.target_speed, self.lane_offset), observation.speed,
            {key: full[key] for key in LOCAL_OBSERVATION_FIELDS},
        )
        return action

    def record_executed(self, action: Action) -> None:
        return None

    def act(self, observation) -> tuple[Action, DecisionRecord]:
        action = self.tick_action(observation)
        return action, DecisionRecord(
            action=action.name, subgoal=self.subgoal, confidence=self.confidence,
            summary=self.summary, candidates=[action.name],
        )

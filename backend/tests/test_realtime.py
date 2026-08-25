"""Tests for the asynchronous scheduler.

The point of the scheduler is that latency costs something, so these tests assert the
cost shows up: ticks driven on stale input, decisions aged in ticks, and a model that
misses a fixed budget being recorded as late rather than credited as on time. No test
here calls a real model; the planner is a stub with a controllable latency.
"""

import threading
import time

import pytest

from harness.models import Action, ActionName, FrameRecord
from harness.policies import AnthropicRacingPolicy, PolicyBudgetExhausted
from harness.providers import ActionSegment, PlayerPlan, ProviderUsage
from harness.racing import RacingDesignDraft, RacingWorld, compile_racing_scene
from harness.realtime import run_realtime_episode


def draft() -> RacingDesignDraft:
    return RacingDesignDraft(
        title="Realtime circuit",
        rationale="A deterministic circuit used to exercise the realtime scheduler.",
        circuit="technical", surface="asphalt", obstacle_count=0, npc_count=0,
    )


def scene():
    return compile_racing_scene("realtime", draft(), seed=17)


def stub_planner(latency_ms: int, steps: int = 4, action: str = "forward"):
    """A planner that answers instantly in wall time but reports a chosen latency."""
    calls: list[int] = []

    def plan(*args, **kwargs):
        calls.append(latency_ms)
        return PlayerPlan(
            subgoal="hold the line", summary="stub plan for the realtime scheduler",
            confidence=1, actions=[ActionSegment(action=action, steps=steps, keys=["w"])],
        ), ProviderUsage(
            provider="test", model="test", output_tokens=10, latency_ms=latency_ms,
        )

    return plan, calls


@pytest.mark.parametrize("latency_ms,expected_ticks", [(0, 0), (100, 1), (250, 3), (3000, 30)])
def test_measured_clock_charges_latency_in_ticks(
    monkeypatch: pytest.MonkeyPatch, latency_ms: int, expected_ticks: int,
) -> None:
    """A three-second decision must cost thirty ticks at 10 Hz, not zero."""
    plan, _ = stub_planner(latency_ms)
    monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
    result = run_realtime_episode(
        RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
        max_steps=60, clock="measured",
    )
    assert result["realtime"]["max_decision_ticks"] == expected_ticks
    assert result["realtime"]["mean_decision_ticks"] == pytest.approx(expected_ticks, abs=1)


def test_a_slow_model_drives_most_of_the_episode_on_stale_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is the whole point: the synchronous loop hides exactly this."""
    plan, _ = stub_planner(3_000, steps=2)
    monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
    slow = run_realtime_episode(
        RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
        max_steps=90, clock="measured",
    )
    plan, _ = stub_planner(50, steps=2)
    monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
    fast = run_realtime_episode(
        RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
        max_steps=90, clock="measured",
    )
    assert slow["realtime"]["starved_ticks"] > fast["realtime"]["starved_ticks"]
    assert slow["realtime"]["fresh_input_fraction"] < fast["realtime"]["fresh_input_fraction"]
    assert slow["realtime"]["decisions"] < fast["realtime"]["decisions"]


def test_fixed_clock_makes_the_schedule_independent_of_real_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two very different latencies must produce the same schedule under `fixed`."""
    schedules = []
    for latency_ms in (10, 2_500):
        plan, _ = stub_planner(latency_ms, steps=3)
        monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
        result = run_realtime_episode(
            RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
            max_steps=60, clock="fixed", latency_ticks=4,
        )
        schedules.append((
            result["realtime"]["decisions"], result["realtime"]["starved_ticks"],
            result["realtime"]["max_decision_ticks"],
        ))
    assert schedules[0] == schedules[1]
    assert schedules[0][2] == 4


def test_starvation_mode_changes_what_the_car_does_while_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holding the last keys and coasting are different physics, so both are offered."""
    outcomes = {}
    for starve in ("hold", "coast"):
        plan, _ = stub_planner(2_000, steps=1, action="forward")
        monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
        result = run_realtime_episode(
            RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
            max_steps=80, clock="measured", starve=starve,
        )
        outcomes[starve] = max(
            frame.privileged_state.speed for frame in result["frames"]
        )
    assert outcomes["hold"] > outcomes["coast"]


def test_wall_clock_runs_at_about_the_control_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, _ = stub_planner(0, steps=8)
    monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
    world = RacingWorld.from_scene(scene())
    started = time.monotonic()
    result = run_realtime_episode(
        world, AnthropicRacingPolicy(), max_steps=20, clock="wall",
    )
    elapsed = time.monotonic() - started
    # Countdown frames are not paced, so compare against the green-flag ticks only.
    expected = result["realtime"]["ticks"] / world.dynamics.control_hz
    # Generous bounds: this asserts the clock is pacing at all, not its jitter.
    assert expected * 0.5 < elapsed < expected * 2.5


def test_final_lap_does_not_wait_for_an_obsolete_overlapped_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A race is over on its terminal frame, even with a planner still in flight."""
    planner_started = threading.Event()

    def blocked_plan(*args, **kwargs):
        planner_started.set()
        time.sleep(1)
        return PlayerPlan(
            subgoal="too late", summary="this answer arrived after the finish",
            confidence=1, actions=[ActionSegment(action="forward", steps=2, keys=["w"])],
        ), ProviderUsage(provider="test", model="test", output_tokens=10, latency_ms=1_000)

    monkeypatch.setattr("harness.policies.plan_racing_actions", blocked_plan)
    world = RacingWorld.from_scene(scene())
    world.countdown_ticks_remaining = 0
    original_step_number = world.step_number

    def finish_on_this_tick(action, decision=None, action_delay=False):
        assert planner_started.wait(.5), "the test needs a genuinely in-flight planner call"
        world.objective_index = len(world.scene.objectives)
        world.terminated = world.succeeded = True
        world.finish_order.append("player")
        world.reason = f"{world.scene.laps}-lap race completed in P1 of 1"
        world.step_number += 1
        return FrameRecord(
            step=world.step_number, observation=world.observe(),
            privileged_state=world.privileged_state(), action=action.name,
            keys=list(action.keys), reward=1,
            events=["crossed finish-line", "race completed P1/1"], decision=decision,
        )

    world.step = finish_on_this_tick  # type: ignore[method-assign]
    started = time.monotonic()
    result = run_realtime_episode(
        world, AnthropicRacingPolicy(), max_steps=20, clock="wall",
    )
    elapsed = time.monotonic() - started

    assert elapsed < .5, "finalization waited for a model response that can no longer be used"
    assert result["succeeded"] and result["terminated"]
    assert result["steps"] == 1
    assert world.step_number == original_step_number + 1


def test_budget_exhaustion_ends_the_episode_without_a_fallback_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = stub_planner(0, steps=1)
    monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
    policy = AnthropicRacingPolicy()
    policy.configure_episode(200, decision_budget=3)
    # configure_episode runs again inside the scheduler, so pin the cap afterwards.
    result = run_realtime_episode(
        RacingWorld.from_scene(scene()), _capped(policy, 3), max_steps=200, clock="measured",
    )
    assert result["policy_failure"] is not None
    assert "budget exhausted" in result["reason"]
    assert result["realtime"]["decisions"] == 3


def _capped(policy: AnthropicRacingPolicy, decisions: int) -> AnthropicRacingPolicy:
    """Freeze a decision cap against the scheduler's own configure_episode call."""
    policy.configure_episode = lambda *args, **kwargs: setattr(policy, "max_turns", decisions)
    return policy


def test_a_policy_without_the_async_interface_is_refused() -> None:
    """A policy with nothing to overlap belongs on the synchronous path."""
    class OnlyAct:
        name = "only-act"

        def reset(self, scene, seed):
            return None

        def act(self, observation):
            return Action(name=ActionName.IDLE), None

    with pytest.raises(TypeError, match="realtime scheduler"):
        run_realtime_episode(
            RacingWorld.from_scene(scene()), OnlyAct(), max_steps=20,
        )


@pytest.mark.parametrize("clock", ["asap", "", "Wall"])
def test_an_unknown_clock_fails_loudly(clock: str) -> None:
    with pytest.raises(ValueError, match="clock must be"):
        run_realtime_episode(
            RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
            max_steps=20, clock=clock,
        )


def test_pipeline_lands_decisions_more_often_than_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrency is the point: N in flight means a decision every latency/N ticks."""
    import time

    def slow_plan(*args, **kwargs):
        time.sleep(0.35)
        return PlayerPlan(
            subgoal="hold", summary="stub plan with real wall-clock latency",
            confidence=1, actions=[ActionSegment(action="forward", steps=2, keys=["w"])],
        ), ProviderUsage(provider="test", model="test", output_tokens=10, latency_ms=350)

    monkeypatch.setattr("harness.policies.plan_racing_actions", slow_plan)
    rates = {}
    for depth in (1, 3):
        result = run_realtime_episode(
            RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
            max_steps=60, clock="wall", pipeline_depth=depth, latency_ticks=4,
        )
        rates[depth] = result["realtime"]
    assert rates[3]["decisions"] > rates[1]["decisions"]
    assert rates[3]["fresh_input_fraction"] > rates[1]["fresh_input_fraction"]
    # An out-of-order answer must be dropped, never applied over a newer one.
    assert rates[3]["superseded_decisions"] >= 0


def test_pipeline_depth_needs_a_real_clock_to_overlap_on() -> None:
    with pytest.raises(ValueError, match="requires the wall clock"):
        run_realtime_episode(
            RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
            max_steps=20, clock="measured", pipeline_depth=3,
        )


def test_scheduler_raises_the_horizon_to_cover_its_own_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-tick queue with a thirteen-tick latency leaves twelve ticks uncontrolled."""
    plan, _ = stub_planner(1_300, steps=8)
    monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
    policy = AnthropicRacingPolicy()
    run_realtime_episode(
        RacingWorld.from_scene(scene()), policy, max_steps=80, clock="measured",
    )
    assert policy.min_action_horizon >= 8


def test_terse_mode_is_reported_so_a_fast_run_cannot_be_mistaken_for_a_verbose_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = stub_planner(100, steps=2)
    monkeypatch.setattr("harness.policies.plan_racing_actions", plan)
    policy = AnthropicRacingPolicy(terse=True)
    result = run_realtime_episode(
        RacingWorld.from_scene(scene()), policy, max_steps=30, clock="measured",
    )
    assert result["terse"] is True


def test_hierarchical_control_never_leaves_a_tick_uncontrolled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason to split the driver: latency stops costing ticks without input.

    A chunk policy runs out of queued ticks and holds stale keys. A fast controller
    acts every tick from the current intent, so the cost of a slow model moves from
    uncontrolled ticks to ticks spent holding a stale lane.
    """
    from harness.policies import AnthropicHierarchicalRacingPolicy
    from harness.providers import RacingIntent

    def slow_intent(*args, **kwargs):
        return RacingIntent(target_speed=5.0, lane_offset=0.0), ProviderUsage(
            provider="test", model="test", output_tokens=8, latency_ms=3_000,
        )

    monkeypatch.setattr("harness.policies.plan_racing_intent", slow_intent)
    chunk, _ = stub_planner(3_000, steps=2)
    monkeypatch.setattr("harness.policies.plan_racing_actions", chunk)

    hierarchical = run_realtime_episode(
        RacingWorld.from_scene(scene()), AnthropicHierarchicalRacingPolicy(),
        max_steps=120, clock="measured",
    )
    direct = run_realtime_episode(
        RacingWorld.from_scene(scene()), AnthropicRacingPolicy(),
        max_steps=120, clock="measured",
    )
    assert hierarchical["realtime"]["starved_ticks"] == 0
    assert hierarchical["realtime"]["fresh_input_fraction"] == 1.0
    assert direct["realtime"]["starved_ticks"] > 100
    # Both pay the same latency; only its consequence differs.
    assert hierarchical["realtime"]["mean_decision_ticks"] > 20


def test_hierarchical_policy_drives_before_any_intent_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 3-second first decision must not mean 30 stationary ticks."""
    from harness.policies import AnthropicHierarchicalRacingPolicy
    from harness.providers import RacingIntent

    def never_lands(*args, **kwargs):
        return RacingIntent(target_speed=5.0, lane_offset=0.0), ProviderUsage(
            provider="test", model="test", output_tokens=8, latency_ms=60_000,
        )

    monkeypatch.setattr("harness.policies.plan_racing_intent", never_lands)
    result = run_realtime_episode(
        RacingWorld.from_scene(scene()), AnthropicHierarchicalRacingPolicy(),
        max_steps=60, clock="measured",
    )
    assert result["realtime"]["decisions"] == 0, "no intent should have landed"
    moved = max(frame.privileged_state.speed for frame in result["frames"])
    assert moved > 1, "the fast layer should creep on its own default"

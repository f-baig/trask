"""Tests for the reflex harness: agent-authored tick-rate control.

The claims worth checking here are not lap times. They are that a bad controller is
rejected with a message rather than crashing a race, that the channels are actually
normalized so a controller transfers between scenes, that the harness detects the events
it says it detects, that a rehearsal predicts the episode it claims to predict, and that
no driving competence has quietly moved into the harness.

The controllers in this file are fixtures, not the harness's policy. They live here for
the same reason `baseline-constant-intent` exists: a result is only interesting relative to what a
fixed controller achieves unaided.
"""

from __future__ import annotations

import inspect

import pytest

from harness.models import Action, ActionName
from harness.racing import RacingWorld, compile_racing_scene
from harness.reflex import blocks as blocks_module
from harness.reflex.blocks import ControlBlocks
from harness.reflex.conditions import ConditionError, ConditionSet, compile_condition
from harness.reflex.output import CommandOut, OutputState
from harness.reflex import runtime as runtime_module
from harness.reflex.runtime import ReflexRuntime
from harness.reflex.sandbox import InstallError, compile_controller, gate_controller
from harness.reflex.sense import FIELDS, compute_sense
from harness.track_grammar import parse_track_prompt


EASY = "A technical asphalt circuit with two barriers and one opponent."
ICE = "A narrow slippery ice circuit with three hairpins and two barriers."

KEEPER = """
def control(sense, ctrl, out):
    lane_error = sense.lane - ctrl.p.target_lane
    steer = ctrl.pid("lane", -lane_error, kp=ctrl.p.kp, kd=ctrl.p.kd)
    heading = ctrl.pid("heading", sense.heading_error / 45.0, kp=0.7)
    out.discretizer("hysteresis")
    out.steer(steer + heading)
    out.throttle(ctrl.pid("speed", ctrl.p.target_speed - sense.speed, kp=0.6))
"""

KEEPER_READS = ["lane", "heading_error", "speed"]
KEEPER_PARAMS = {"target_lane": 0.0, "kp": 0.9, "kd": 0.08, "target_speed": 2.4}


def scene_for(brief: str, seed: int = 17):
    return compile_racing_scene(brief, parse_track_prompt(brief), seed=seed)


def keeper(**overrides):
    params = {**KEEPER_PARAMS, **overrides}
    return compile_controller(
        name="keep", source=KEEPER, reads=KEEPER_READS, params=params,
        safe_action={"steer": "hold", "throttle": -0.6},
    )


def runtime_with_keeper(world, **overrides) -> ReflexRuntime:
    runtime = ReflexRuntime(world.scene)
    result = runtime.install(
        name="keep", source=KEEPER, reads=KEEPER_READS,
        params={**KEEPER_PARAMS, **overrides},
        safe_action={"steer": "hold", "throttle": -0.6},
    )
    assert result["installed"], result
    return runtime


def test_visual_players_always_expose_speed_and_no_other_engine_field() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = ReflexRuntime(world.scene, vision_only=True, visual_mode="2d")
    values = runtime.observe_visual(world)
    assert "speed" in runtime.visible_fields
    assert "speed" in values
    assert runtime.visible_fields - {"speed"} == runtime._visual_fields()
    assert not ({"lane", "heading_error", "curvature", "grip_used", "on_track"} & runtime.visible_fields)


# -- the install gate ------------------------------------------------------------------


@pytest.mark.parametrize(("source", "expected"), [
    ("import math\ndef control(sense, ctrl, out):\n    out.steer(0)", "import is not available"),
    ("def control(sense, ctrl, out):\n    while True:\n        out.steer(0)", "loops are not available"),
    ("def control(sense, ctrl, out):\n    for i in [1]:\n        out.steer(0)", "loops are not available"),
    ("def control(sense, ctrl, out):\n    out.steer(sense.nonsense)", "not a channel"),
    ("def control(sense, ctrl, out):\n    out.steer(sense.curvature)", "not in this controller's reads"),
    ("def drive(sense, ctrl, out):\n    out.steer(0)", "must be named `control`"),
    ("def control(sense, out):\n    out.steer(0)", "must take exactly"),
    ("def control(sense, ctrl, out):\n    out.turbo(1)", "out.turbo is not available"),
    ("def control(sense, ctrl, out):\n    out.steer(ctrl.p.missing)", "not a declared param"),
    ("def control(sense, ctrl, out):\n    out.steer(open('x'))", r"open\(\) is not available"),
    ("def control(sense, ctrl, out):\n    out.steer(ctrl.magic(1))", "is not a helper"),
    ("def control(sense, ctrl, out):\n    out.steer(sense.__class__)", "private attributes"),
])
def test_gate_rejects_with_an_actionable_message(source: str, expected: str) -> None:
    """Every rejection names the line and the alternative.

    The point of the gate is not safety, it is that an agent told `import is not
    available; use ctrl.sqrt` fixes it in one turn, where an agent that gets a NameError
    at tick 300 has to reason backwards from a crashed race.
    """
    with pytest.raises(InstallError, match=expected):
        compile_controller(
            name="bad", source=source, reads=["lane", "speed"], params={"kp": 1.0},
        )


def test_gate_accepts_a_reasonable_controller_and_reports_its_cost() -> None:
    report = gate_controller(keeper())
    assert report.ok, report.as_dict()
    assert report.fuzz_samples > 200
    assert report.max_tick_ms < 2.0
    assert report.mirror_ok is True


def test_gate_catches_a_sign_error_with_the_mirror_test() -> None:
    """Mirror the track and the steering should mirror. This is the most common failure."""
    wrong_sign = KEEPER.replace("steer + heading", "steer + abs(heading)")
    report = gate_controller(compile_controller(
        name="flipped", source=wrong_sign, reads=KEEPER_READS, params=KEEPER_PARAMS,
    ))
    assert report.mirror_ok is False
    assert any("mirror" in warning for warning in report.warnings)


def test_the_timing_gate_judges_the_controller_not_the_host() -> None:
    """A busy machine must not reject a fast controller.

    Found by the full suite failing while the same test passed alone: this controller costs
    0.006 ms a tick, and under load the host produced single readings of 13–22 ms, so a
    max-based gate rejected roughly a third of installs. The median is the statistic that
    describes the code; the tail describes the operating system.
    """
    import threading

    controller = keeper()
    stop = threading.Event()

    def burn() -> None:
        while not stop.is_set():
            sum(index * index for index in range(20_000))

    workers = [threading.Thread(target=burn, daemon=True) for _ in range(8)]
    for worker in workers:
        worker.start()
    try:
        reports = [gate_controller(controller) for _ in range(12)]
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=2)

    assert all(report.ok for report in reports), [
        report.errors for report in reports if not report.ok
    ]
    assert all(report.mean_tick_ms < 2.0 for report in reports)


def test_a_single_slow_tick_does_not_fail_a_working_controller() -> None:
    """The live path has the same asymmetry: one overrun is the host, five in a row is the code."""
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world)
    runtime.tick_budget_ms = 0.0  # Every tick now counts as an overrun.
    for _ in range(runtime_module.OVERRUN_TICKS_BEFORE_FAILURE - 1):
        world.step(runtime.tick(world, world.observe()))
    assert runtime.last_failure is None, "a short burst of slow ticks is not a controller fault"
    world.step(runtime.tick(world, world.observe()))
    assert runtime.last_failure and "consecutive" in runtime.last_failure


def test_gate_rejects_a_controller_that_divides_by_zero_at_rest() -> None:
    """A controller only fails at speed zero, which the extremes in the fuzz set cover."""
    source = (
        "def control(sense, ctrl, out):\n"
        "    out.steer(sense.lane / sense.speed)\n"
    )
    report = gate_controller(compile_controller(
        name="fragile", source=source, reads=["lane", "speed"], params={},
    ))
    assert not report.ok
    assert any("ZeroDivisionError" in error for error in report.errors)


def test_a_failed_install_leaves_the_previous_controller_driving() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world)
    result = runtime.install(name="keep", source="def control(x):\n    pass", reads=["lane"])
    assert result["installed"] is False
    assert runtime.controllers["keep"].controller.version == 1
    assert runtime.active == "keep"


# -- blocks ----------------------------------------------------------------------------


def test_blocks_may_not_import_channels() -> None:
    """The structural rule that stops the helper library becoming a driver.

    A block sees a number and returns a number. If it could read the channel catalog it
    could know where the road is, and the library would slowly become the policy.
    """
    source = inspect.getsource(blocks_module)
    assert "from .sense" not in source
    assert "import sense" not in source


def test_pid_counts_the_oscillation_that_wakes_the_agent() -> None:
    blocks = ControlBlocks({})
    blocks.begin_tick(0.1)
    for tick in range(20):
        blocks.pid("lane", 1.0 if tick % 2 == 0 else -1.0, kp=1.0)
    name, changes = blocks.max_sign_changes()
    assert name == "lane"
    assert changes >= 15


def test_a_converged_loop_is_not_reported_as_unstable() -> None:
    """Floating-point noise on top of a target must not read as violent oscillation."""
    blocks = ControlBlocks({})
    blocks.begin_tick(0.1)
    for tick in range(20):
        blocks.pid("lane", 1e-9 if tick % 2 == 0 else -1e-9, kp=1.0)
    assert blocks.max_sign_changes()[1] == 0


def test_pid_reports_its_clamped_fraction() -> None:
    blocks = ControlBlocks({})
    blocks.begin_tick(0.1)
    for _ in range(10):
        blocks.pid("saturated", 50.0, kp=1.0, limit=1.0)
    assert blocks.reports()["saturated"]["clamped_fraction"] == 1.0


# -- the output stage ------------------------------------------------------------------


def chatter_trace(discretizer: str) -> int:
    """Drive the signal that broke the tuned controller and count key reversals."""
    state = OutputState()
    for tick in range(40):
        out = CommandOut(state=state)
        out.discretizer(discretizer)
        out.steer(0.4 if tick % 2 == 0 else -0.4)
        out.resolve()
    return state.reversals


def test_hysteresis_removes_the_chatter_that_deadband_produces() -> None:
    """`lowlevel.py:48` records fifteen reversals a lap and a crash. This is that failure.

    The discretizer is the harness's, not the agent's, because chatter is a fact about
    pushing a continuous command through a three-state actuator rather than anything
    about racing.
    """
    assert chatter_trace("deadband") > 15
    assert chatter_trace("hysteresis") == 0


def test_pwm_tracks_a_fractional_command_the_action_space_cannot_express() -> None:
    state = OutputState()
    held = 0
    for _ in range(100):
        out = CommandOut(state=state)
        out.discretizer("pwm")
        out.steer(0.3)
        if "d" in out.resolve().keys:
            held += 1
    assert 28 <= held <= 32


def test_nitro_is_refused_unless_the_engine_would_allow_it() -> None:
    """The engine burns nitro only on straight throttle, on track, with a full tank."""
    out = CommandOut(state=OutputState(), nitro_ready=True, on_track=True)
    out.throttle(1.0)
    out.steer(0.0)
    out.boost(True)
    assert "space" in out.resolve().keys

    turning = CommandOut(state=OutputState(), nitro_ready=True, on_track=True)
    turning.throttle(1.0)
    turning.steer(1.0)
    turning.boost(True)
    assert "space" not in turning.resolve().keys
    assert turning.state.boost_refusals == 1


def test_conflicting_keys_are_impossible_by_construction() -> None:
    out = CommandOut(state=OutputState())
    out.throttle(-1.0)
    out.steer(1.0)
    action = out.resolve()
    assert set(action.keys) == {"s", "d"}
    assert action.name is ActionName.RIGHT


# -- conditions ------------------------------------------------------------------------


def test_conditions_parse_both_forms() -> None:
    assert compile_condition("target_reached").event == "target_reached"
    threshold = compile_condition("ttc < 1.5")
    assert (threshold.field_name, threshold.operator, threshold.threshold) == ("ttc", "<", 1.5)
    absolute = compile_condition({"when": "abs(lane) > 0.85", "for_ticks": 3})
    assert absolute.absolute and absolute.for_ticks == 3


@pytest.mark.parametrize("bad", ["ttc <", "nonsense > 1", "lane", "on_track == 0"])
def test_unparseable_conditions_explain_themselves(bad: str) -> None:
    with pytest.raises(ConditionError):
        compile_condition(bad)


def test_for_ticks_suppresses_a_single_noisy_tick() -> None:
    condition = compile_condition({"when": "ttc < 1.0", "for_ticks": 3})
    values = {"ttc": 0.5}
    assert not condition.evaluate(values, {})
    assert not condition.evaluate(values, {})
    assert condition.evaluate(values, {})


def test_conditions_are_edge_triggered_not_level_triggered() -> None:
    """A sustained situation is one wake, not one per tick.

    This was a real failure on an ice circuit: a car stuck off track satisfied `off_track`
    on every one of the next thousand ticks and asked for a thousand wakes, burning the
    call budget re-reporting a fact the agent had already acted on.
    """
    condition = compile_condition("off_track")
    values = {"on_track": False}
    assert condition.evaluate(values, {})
    for _ in range(50):
        assert not condition.evaluate(values, {}), "a held condition must not re-fire"
    condition.evaluate({"on_track": True}, {})
    assert condition.evaluate(values, {}), "clearing re-arms it"
    assert condition.fired_count == 2


def test_a_condition_reports_how_close_it_came() -> None:
    """The margin is the reason conditions are not opaque predicates.

    "Your collision condition never fired because ttc bottomed out at 0.83 against your
    0.8 threshold" is a complete explanation of a crash; a boolean is not.
    """
    condition = compile_condition("ttc < 0.8")
    for value in (3.0, 1.4, 0.83, 2.0):
        condition.evaluate({"ttc": value}, {})
    assert condition.fired_count == 0
    assert condition.report()["closest_margin"] == pytest.approx(0.03, abs=1e-6)


def test_the_always_armed_conditions_cannot_be_dropped() -> None:
    conditions = ConditionSet.build(["target_reached"])
    armed = {condition.when for condition in conditions.armed}
    assert armed == {"controller_failed", "off_track", "no_progress", "geometry_changed", "deadline"}


# -- channels --------------------------------------------------------------------------


def test_channels_are_normalized_and_local() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = ReflexRuntime(world.scene)
    values = compute_sense(world, world.observe(), runtime.memory, None)
    assert set(values) == set(FIELDS)
    assert abs(values["lane"]) <= 1.0
    assert values["on_track"] is True
    assert values["grade"] == 0.0 and values["bank"] == 0.0, "2D is a flat plane"
    assert 0.0 < values["half_width"] < 4.0, "corridor half-width in car lengths"
    assert values["ttc"] > 0
    assert not {"progress", "next_corner", "centerline_index"} & set(values), (
        "the channel set is local: no lap position and no corner map"
    )


def test_free_ahead_shrinks_before_a_corner() -> None:
    """The channel that lets a controller anticipate without being handed the route.

    An earlier version measured every probe against a centerline window fixed at the car,
    so the ray appeared to leave the corridor as soon as it outran the window — a wall at
    a constant distance everywhere, on a straight and in a hairpin alike.
    """
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world, target_speed=2.4)
    seen = []
    for _ in range(200):
        if world.terminated:
            break
        world.step(runtime.tick(world, world.observe()))
        seen.append(runtime.last_sense["free_ahead"])
    assert max(seen) > 4.0, "an open road has to read as open"
    assert min(seen) < max(seen) / 2, "a corner has to read as materially more closed"


def test_grip_used_is_comparable_across_surfaces() -> None:
    """1.0 means saturated tires on ice and on asphalt alike, which is what makes a
    controller written in these units portable."""
    for brief in (EASY, ICE):
        world = RacingWorld.from_scene(scene_for(brief))
        runtime = runtime_with_keeper(world, target_speed=3.0)
        peak = 0.0
        for _ in range(160):
            if world.terminated:
                break
            world.step(runtime.tick(world, world.observe()))
            peak = max(peak, runtime.last_sense["grip_used"])
        assert 0.0 <= peak < 4.0


# -- the runtime -----------------------------------------------------------------------


def test_nothing_is_installed_and_the_car_does_not_move() -> None:
    """No default controller ships. The harness has no driving competence to fall back on."""
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = ReflexRuntime(world.scene)
    for _ in range(40):
        action = runtime.tick(world, world.observe())
        assert action.keys == []
        world.step(action)
    assert world.speed == 0.0
    assert runtime.idle_ticks == 40


def test_the_deadline_wakes_an_agent_that_installed_nothing() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = ReflexRuntime(world.scene)
    runtime.set_conditions([], deadline_ticks=5)
    for _ in range(6):
        world.step(runtime.tick(world, world.observe()))
    assert "deadline" in runtime.wake_causes_pending()


def test_pace_deadline_is_capped_and_stationary_controller_wakes() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = ReflexRuntime(world.scene)
    runtime.set_conditions([], deadline_ticks=10**6)
    assert runtime.deadline_ticks == 80
    installed = runtime.install(
        name="wait", reads=["speed"], params={},
        source="def control(sense, ctrl, out):\n    sense.speed\n    out.throttle(0.0)\n",
    )
    assert installed["installed"]
    while world.countdown_ticks_remaining > 0:
        world.step(Action())
    for _ in range(25):
        world.step(runtime.tick(world, world.observe()))
    assert "no_progress" in runtime.wake_causes_pending()


def test_an_installed_controller_drives_the_car() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world)
    while world.countdown_ticks_remaining > 0:
        world.step(runtime.tick(world, world.observe()))
    for _ in range(80):
        if world.terminated:
            break
        world.step(runtime.tick(world, world.observe()))
    assert world.speed > 0.5
    assert runtime.controller_ticks > 50


def test_a_controller_that_fails_falls_back_to_the_agents_own_safe_action() -> None:
    """Never to a harness controller: that would reinstate the hard-coded policy exactly
    where it matters most."""
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = ReflexRuntime(world.scene)
    installed = runtime.install(
        name="breaks", reads=["lane"], params={},
        safe_action={"steer": "hold", "throttle": -0.6},
        source="def control(sense, ctrl, out):\n    out.steer(ctrl.p.gone)\n",
    )
    # The gate catches this one, which is the point — so break it after installation.
    assert installed["installed"] is False

    runtime.install(
        name="fine", reads=["lane"], params={},
        safe_action={"steer": "none", "throttle": -0.6},
        source="def control(sense, ctrl, out):\n    out.steer(sense.lane)\n",
    )

    def explode(sense, ctrl, out):
        raise RuntimeError("boom")

    runtime.controllers["fine"].controller.function = explode
    action = runtime.tick(world, world.observe())
    assert action.keys == ["s"], "the agent's declared safe action, not a harness policy"
    assert "controller_failed" in runtime.wake_causes_pending()
    assert runtime.failures


def test_a_non_finite_command_never_reaches_the_simulator() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = ReflexRuntime(world.scene)
    runtime.install(
        name="nan", reads=["lane"], params={}, safe_action={"throttle": 0.0},
        source="def control(sense, ctrl, out):\n    out.steer(sense.lane)\n",
    )

    def poison(sense, ctrl, out):
        out.steer_command = float("nan")

    runtime.controllers["nan"].controller.function = poison
    runtime.tick(world, world.observe())
    assert runtime.last_failure == "emitted a non-finite command"


def test_patch_params_retunes_without_recompiling() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world)
    assert runtime.patch_params("keep", {"target_speed": 1.2})["params"]["target_speed"] == 1.2
    assert runtime.patch_params("keep", {"nope": 1})["patched"] is False
    assert runtime.controllers["keep"].controller.version == 1, "a patch is not a recompile"


def test_geometry_change_fires_on_novel_curvature_not_on_familiar_corners() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world, target_speed=3.0)
    runtime.set_conditions(["geometry_changed"], deadline_ticks=10**6)
    fired = 0
    for _ in range(400):
        if world.terminated:
            break
        world.step(runtime.tick(world, world.observe()))
        if "geometry_changed" in runtime.wake_causes_pending():
            fired += 1
        runtime.clear_wake()
    assert fired < 20, "a corner on a circuit of corners is not news"


# -- rehearsal -------------------------------------------------------------------------


def test_rehearsal_does_not_disturb_the_live_world() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world)
    for _ in range(40):
        world.step(runtime.tick(world, world.observe()))
    before = world.snapshot()
    report = runtime.try_controller(world, "keep", ticks=120)
    assert report.ticks > 0
    assert world.snapshot() == before, "a rehearsal is a fork, not a detour"


def test_rehearsal_predicts_the_episode_it_claims_to_predict() -> None:
    """Rehearsal fidelity: the fork must reproduce what the live world then does.

    Without this the primitive is worse than useless — an agent would tune against a
    simulator that disagrees with the one it is racing in.
    """
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world)
    while world.countdown_ticks_remaining > 0:
        world.step(runtime.tick(world, world.observe()))

    rehearsal = runtime.try_controller(world, "keep", ticks=60)

    live = ReflexRuntime(world.scene)
    live.memory.track_index = runtime.memory.track_index
    live.install(
        name="keep", source=KEEPER, reads=KEEPER_READS, params=dict(KEEPER_PARAMS),
        safe_action={"steer": "hold", "throttle": -0.6},
    )
    speeds = []
    for _ in range(60):
        if world.terminated:
            break
        world.step(live.tick(world, world.observe()))
        speeds.append(live.last_sense["speed"])
    observed = sum(speeds) / len(speeds)
    assert rehearsal.mean_speed == pytest.approx(observed, rel=0.02)


def test_a_rehearsal_scores_the_lap_against_the_agents_own_best() -> None:
    """Finishing is easy to mistake for success, so a finish never reports alone.

    The comparison is against the agent's own previous best and nothing else. Comparing
    against the fixture controller's time would hand it the answer to how fast the circuit
    can be driven, which is the measurement rather than the input.
    """
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world, target_speed=2.4)
    while world.countdown_ticks_remaining > 0:
        world.step(runtime.tick(world, world.observe()))

    slow = runtime.try_controller(world, "keep", ticks=900)
    assert slow.succeeded and slow.projected_finish_tick is not None
    assert slow.is_new_best
    assert "first finish" in slow.as_dict()["score"]["standing"]
    baseline = runtime.best_finish_tick

    runtime.patch_params("keep", {"target_speed": 3.0})
    quick = runtime.try_controller(world, "keep", ticks=900)
    assert quick.projected_finish_tick < baseline, "3.0 cl/s must lap faster than 2.4"
    assert "NEW BEST" in quick.as_dict()["score"]["standing"]

    runtime.patch_params("keep", {"target_speed": 2.4})
    regression = runtime.try_controller(world, "keep", ticks=900)
    assert "SLOWER" in regression.as_dict()["score"]["standing"]
    assert runtime.best_finish_tick == quick.projected_finish_tick, "a slower lap is not a best"


def test_a_projected_lap_time_is_comparable_to_the_real_race() -> None:
    """The frozen countdown grid belongs to nobody's lap time.

    Counting it in a rehearsal but not in the episode inflated every projection by the
    countdown, so a rehearsal that had predicted the race correctly looked 30 ticks
    pessimistic — and the agent was optimizing against a number that did not mean what the
    result meant.
    """
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world, target_speed=3.0)
    rehearsed = runtime.try_controller(world, "keep", ticks=900)
    assert rehearsed.succeeded

    live = runtime_with_keeper(world, target_speed=3.0)
    driven = 0
    while not world.terminated and driven < 900:
        if world.countdown_ticks_remaining > 0:
            world.step(live.tick(world, world.observe()))
            continue
        world.step(live.tick(world, world.observe()))
        driven += 1
    assert world.succeeded
    assert abs(rehearsed.projected_finish_tick - driven) <= 4


def test_an_unfinished_rehearsal_says_why_it_cannot_be_scored() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world, target_speed=2.4)
    score = runtime.try_controller(world, "keep", ticks=40).as_dict()["score"]
    assert score["lap_time"] is None
    assert "did not finish" in score["note"]


def test_the_rehearsal_budget_is_enforced_per_wake() -> None:
    """Small on purpose: rehearsals should be spent lowering the lap time, not confirming."""
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world)
    runtime.rehearsal_budget = 2
    assert runtime.try_controller(world, "keep", ticks=20).failure is None
    assert runtime.try_controller(world, "keep", ticks=20).failure is None
    spent = runtime.try_controller(world, "keep", ticks=20)
    assert spent.failure is not None and "budget" in spent.failure
    runtime.rehearsals_used = 0
    assert runtime.try_controller(world, "keep", ticks=20).failure is None


def test_a_rehearsal_reports_where_a_controller_fails() -> None:
    """Too fast on ice, which is the whole point of being able to try before driving."""
    world = RacingWorld.from_scene(scene_for(ICE))
    runtime = runtime_with_keeper(world, target_speed=9.0)
    while world.countdown_ticks_remaining > 0:
        world.step(runtime.tick(world, world.observe()))
    report = runtime.try_controller(world, "keep", ticks=600)
    assert report.off_track_ticks > 0 or report.terminated, report.as_dict()


# -- portability -----------------------------------------------------------------------


def test_one_controller_runs_unchanged_on_two_surfaces() -> None:
    """The claim the normalized units exist to buy."""
    for brief in (EASY, ICE):
        world = RacingWorld.from_scene(scene_for(brief))
        runtime = runtime_with_keeper(world, target_speed=2.2)
        while world.countdown_ticks_remaining > 0:
            world.step(runtime.tick(world, world.observe()))
        for _ in range(200):
            if world.terminated:
                break
            world.step(runtime.tick(world, world.observe()))
        assert runtime.controller_ticks > 100
        assert not runtime.failures


def test_the_same_controller_drives_a_flat_3d_circuit_identically() -> None:
    """A strict extension of the existing flat-3D equivalence assertion.

    With no elevation the 3D engine reproduces the 2D engine bit for bit, so a controller
    written in normalized units must emit the identical key sequence. If it does not, some
    channel is reading a quantity that is not actually scale-free.
    """
    from harness.models import ElevationProfile, ElevationSpec
    from harness.racing3d import Racing3DWorld

    scene = scene_for(EASY)
    flat = ElevationSpec(profile=ElevationProfile.FLAT, amplitude_m=0.0, banking_degrees=0.0)

    planar = RacingWorld.from_scene(scene)
    planar_runtime = runtime_with_keeper(planar)
    solid = Racing3DWorld.from_scene(scene, elevation=flat)
    solid_runtime = runtime_with_keeper(solid)

    for _ in range(150):
        if planar.terminated or solid.terminated:
            break
        planar_action = planar_runtime.tick(planar, planar.observe())
        solid_action = solid_runtime.tick(solid, solid.observe())
        assert planar_action.keys == solid_action.keys
        planar.step(planar_action)
        solid.step(solid_action)


# -- the service path ------------------------------------------------------------------


def test_a_reflex_run_is_a_first_class_run(tmp_path) -> None:
    """`service.run` must produce the same artifact for a reflex episode as for any other.

    Otherwise the reflex driver is a script with a private output format, and none of the
    things that make a run useful — the replay artifact, the desktop viewer, the run tree —
    reach it. The driver here is a stub so this costs no model calls; what is under test is
    the record mapping, not the agent.
    """
    from harness.models import RunRequest
    from harness.reflex.agent import AgentTurn
    from harness.reflex.episode import EpisodeReport
    from harness.service import HarnessService
    from harness.store import HarnessStore

    class StubReflexDriver:
        """Shaped like the real driver: `run_episode`, and deliberately no `act`."""

        name = "telemetry-reflex"

        def run_episode(self, world, *, max_steps: int) -> EpisodeReport:
            runtime = ReflexRuntime(world.scene)
            runtime.install(
                name="keep", source=KEEPER, reads=KEEPER_READS, params=dict(KEEPER_PARAMS),
                safe_action={"steer": "hold", "throttle": -0.6},
            )
            runtime.set_conditions([], deadline_ticks=10**6)
            report = EpisodeReport(field_size=world.field_size)
            while not world.terminated and report.ticks < min(max_steps, 400):
                if world.countdown_ticks_remaining > 0:
                    report.frames.append(world.step(Action()))
                    continue
                report.frames.append(world.step(runtime.tick(world, world.observe())))
                runtime.clear_wake()
                report.ticks += 1
            report.succeeded, report.terminated = world.succeeded, world.terminated
            report.reason = world.reason
            report.wakes = 2
            report.turns = [
                AgentTurn(causes=[], tick=1, resumed=True),
                AgentTurn(causes=["geometry_changed"], tick=60, resumed=True),
            ]
            report.usage.calls, report.usage.input_tokens, report.usage.output_tokens = 7, 900, 120
            report.runtime_summary = runtime.summary()
            return report

    service = HarnessService(store=HarnessStore(tmp_path))
    service.policies["telemetry-reflex"] = StubReflexDriver()
    environment = service.create_environment(EASY, seed=17, provider="offline")
    run = service.run(RunRequest(
        environment_id=environment.id, policy_name="telemetry-reflex", max_steps=400,
    ))

    assert run.frames, "a reflex run must record frames like any other run"
    assert run.status.value in {"succeeded", "failed", "timeout"}
    assert run.player_turns == 7, "player-turn accounting must count provider calls, not reflex wakes"
    assert run.input_tokens == 900 and run.output_tokens == 120
    assert run.artifacts and run.artifacts[0].kind == "replay"
    assert service.get_run(run.id) is not None, "it has to be in the store to be reopened"

    bundle = service.get_replay_bundle(run.id)
    assert len(bundle.frames) == len(run.frames) == len(bundle.timeline)
    assert bundle.renderer.transport == "replay-bundle/v2"
    wake_frames = [
        frame for frame in run.frames if frame.decision and frame.decision.subgoal.startswith("woke")
    ]
    assert wake_frames, "wake ticks must be labelled so the UI can show where the model acted"
    assert all(frame.decision is not None for frame in run.frames)


def test_a_schema_invalid_creator_plan_costs_one_retry_not_the_dispatch(tmp_path) -> None:
    """The creator can fail on its own grammar, and that is a retry.

    Found live: a plan with a 49-character corner label raises before there is any geometry
    to compile, and the ladder only caught compile failures — so one over-long string aborted
    a whole coordinator dispatch.
    """
    # Patched on `faithful`, which is the module that actually calls the creator now that
    # generation runs through the contract pipeline. The guarantee under test is unchanged:
    # a plan that breaks the grammar's own field limits is one bounded retry, not a dead
    # dispatch. Only the seam it is enforced at has moved.
    import harness.faithful as faithful_module
    from harness.providers import ProviderError
    from harness.service import HarnessService
    from harness.store import HarnessStore

    attempts = {"count": 0}
    authored = faithful_module.author

    def flaky(spec, provider="auto", feedback=None, repair=None, precedents=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ProviderError(
                "Racing creator returned an invalid track plan: corners.1.label "
                "String should have at most 48 characters"
            )
        return authored(spec, "offline", feedback=feedback, repair=repair, precedents=precedents)

    faithful_module.author = flaky
    try:
        service = HarnessService(store=HarnessStore(tmp_path))
        steps: list[tuple[str, str]] = []
        environment = service.create_environment(
            EASY, seed=17, provider="offline", on_step=lambda stage, detail: steps.append((stage, detail)),
        )
    finally:
        faithful_module.author = authored

    assert attempts["count"] == 2, "the rejected plan must be retried, not fatal"
    assert environment.playability_certificate and environment.playability_certificate.playable
    assert any(stage == "rejected" for stage, _ in steps), "the rejection has to be reported"


def test_a_reflex_run_refuses_to_fork(tmp_path) -> None:
    """Honest failure beats a fork that silently drops the controller it was driving."""
    from harness.models import ForkRequest, RunRequest
    from harness.service import HarnessService
    from harness.store import HarnessStore

    service = HarnessService(store=HarnessStore(tmp_path))
    environment = service.create_environment(EASY, seed=17, provider="offline")
    parent = service.run(RunRequest(
        environment_id=environment.id, policy_name="oracle-racing-line", max_steps=120,
    ))
    service.policies["telemetry-reflex"].name = "telemetry-reflex"
    parent.policy_name = "telemetry-reflex"
    service.store.save_run(parent)
    with pytest.raises(ValueError, match="cannot resume a forked prefix"):
        service.fork_run(parent.id, ForkRequest(fork_step=40, perturbation="low_grip"))


def test_every_controller_authorship_call_becomes_a_timeline_entry() -> None:
    from types import SimpleNamespace

    from harness.service import HarnessService

    turns = [SimpleNamespace(tick=17, tool_calls=[{
        "name": "install_controller",
        "input": {
            "name": "hairpin", "source": "def control(sense, ctrl, out):\n    out.brake(1)",
            "reads": ["bend", "speed"], "params": {"entry": 0.4},
        },
        "result": {
            "installed": True, "controller": "hairpin@2", "active": True,
            "gate": {"ok": True, "errors": []},
        },
    }, {
        "name": "install_controller",
        "input": {"name": "bad", "source": "import os", "reads": ["speed"]},
        "result": {"installed": False, "gate": {"ok": False, "errors": ["import is unavailable"]}},
    }])]

    writes = HarnessService._controller_writes(turns, countdown_frames=30, end_frame_step=90)
    assert [item.name for item in writes] == ["hairpin", "bad"]
    assert writes[0].frame_step == 47 and writes[0].label == "hairpin@2" and writes[0].active
    assert writes[0].effective_from_frame_step == 48
    assert writes[0].effective_until_frame_step == 90
    assert not writes[1].installed and writes[1].errors == ["import is unavailable"]
    assert "out.brake" in writes[0].source


def test_controller_timeline_uses_only_the_version_that_reaches_a_live_tick() -> None:
    from types import SimpleNamespace

    from harness.service import HarnessService

    def install(name: str, version: int) -> dict:
        return {
            "name": "install_controller",
            "input": {"name": name, "source": "def control(sense, ctrl, out):\n    out.throttle(1)"},
            "result": {
                "installed": True, "controller": f"{name}@{version}", "active": True,
                "gate": {"ok": True, "errors": []},
            },
        }

    turns = [
        SimpleNamespace(tick=0, tool_calls=[install("draft", 1), install("launch", 1)]),
        SimpleNamespace(tick=17, tool_calls=[install("corner", 1)]),
    ]
    writes = HarnessService._controller_writes(
        turns, countdown_frames=30, end_frame_step=70,
    )
    draft, launch, corner = writes
    assert not draft.active and draft.effective_from_frame_step is None
    assert (launch.effective_from_frame_step, launch.effective_until_frame_step) == (31, 47)
    assert (corner.effective_from_frame_step, corner.effective_until_frame_step) == (48, 70)


# -- the recorder ----------------------------------------------------------------------


def test_decimation_preserves_the_oscillation_it_is_meant_to_show() -> None:
    """Stride sampling would alias a 5 Hz reversal into a smooth line — the exact thing
    being debugged."""
    runtime = ReflexRuntime(RacingWorld.from_scene(scene_for(EASY)).scene)
    for tick in range(200):
        runtime.recorder.append({
            "tick": tick, "controller": "osc",
            "lane": 0.9 if tick % 2 == 0 else -0.9,
            "heading_error": 0.0, "speed": 3.0, "curvature": 0.0, "grip_used": 0.5,
            "free_ahead": 6.0, "ttc": None, "target_error": 0.0,
            "steer": 1.0 if tick % 2 == 0 else -1.0, "throttle": 0.5,
            "keys": "w", "fired": None,
        })
    rows = runtime.window(200, 20)
    assert len(rows) <= 20
    spans = [row["lane_range"] for row in rows]
    assert all(low < -0.5 and high > 0.5 for low, high in spans), (
        "each bucket must keep both extremes"
    )


def test_the_summary_reports_wake_causes_and_chatter() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    runtime = runtime_with_keeper(world)
    for _ in range(60):
        if world.terminated:
            break
        world.step(runtime.tick(world, world.observe()))
        runtime.clear_wake()
    summary = runtime.summary()
    assert summary["controller_ticks"] > 0
    assert "steer_reversals" in summary["output"]
    assert isinstance(summary["wake_causes"], dict)

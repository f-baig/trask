"""Tests for hierarchical control: the fast layer and its attribution baseline.

The load-bearing claims here are about privilege and attribution, not lap times. The
fast controller must read only what a cone-view prompt reads, and it must be strong
enough on its own that any credit given to the model is credit the controller could
not have earned by itself.
"""

import pytest

from harness.lowlevel import (
    LOCAL_OBSERVATION_FIELDS, ConstantIntentPolicy, Intent, LocalIntentController,
)
from harness.models import ActionName
from harness.racing import (
    RacingDesignDraft, RacingWorld, compile_racing_scene, racing_local_state,
    racing_public_context,
)
from harness.realtime import run_realtime_episode
from harness.track_grammar import parse_track_prompt


EASY = "A technical asphalt circuit with two barriers and one opponent."
HARD = "A narrow slippery ice circuit with three hairpins and four barriers."


def scene_for(brief: str):
    return compile_racing_scene(brief, parse_track_prompt(brief), seed=17)


def local(world) -> dict:
    full = racing_local_state(world.scene, world.observe())
    return {key: full[key] for key in LOCAL_OBSERVATION_FIELDS}


def test_controller_may_not_read_route_knowledge() -> None:
    """The privilege boundary is enforced, not merely documented.

    A controller that quietly began reading `centerline_index` would be sampling
    global route position, which is exactly what the forward-cone view withholds.
    """
    world = RacingWorld.from_scene(scene_for(EASY))
    smuggled = {**local(world), "centerline_index": 4}
    with pytest.raises(ValueError, match="route knowledge"):
        LocalIntentController().act(Intent(5.0, 0.0), 5.0, smuggled)


def test_controller_requires_the_whole_local_state() -> None:
    world = RacingWorld.from_scene(scene_for(EASY))
    partial = {key: value for key, value in local(world).items() if key != "on_track"}
    with pytest.raises(ValueError, match="missing"):
        LocalIntentController().act(Intent(5.0, 0.0), 5.0, partial)


def test_the_prompt_and_the_controller_read_the_same_numbers() -> None:
    """One function fills both, so the boundary is checkable by construction."""
    world = RacingWorld.from_scene(scene_for(EASY))
    observation = world.observe()
    prompt_state = racing_public_context(world.scene, observation)["track_state"]
    controller_state = racing_local_state(world.scene, observation)
    for field in LOCAL_OBSERVATION_FIELDS:
        assert prompt_state[field] == controller_state[field]


@pytest.mark.parametrize("lane_error,expected", [
    (30.0, ActionName.LEFT),    # sitting right of target, so correct leftwards
    (-30.0, ActionName.RIGHT),
])
def test_steering_corrects_toward_the_requested_lane(lane_error: float, expected: ActionName) -> None:
    """Signs were established empirically: steering right raises signed_lane_offset."""
    state = {
        "signed_lane_offset": lane_error, "centerline_heading_error": 0.0,
        "safe_lane_half_width": 55.0, "on_track": True,
    }
    action, terms = LocalIntentController().act(Intent(5.0, 0.0), 5.0, state)
    assert action.name is expected
    assert terms["lane_error"] == pytest.approx(lane_error)


def test_speed_control_is_deadbanded_in_three_states() -> None:
    state = {
        "signed_lane_offset": 0.0, "centerline_heading_error": 0.0,
        "safe_lane_half_width": 55.0, "on_track": True,
    }
    controller = LocalIntentController()
    assert controller.act(Intent(6.0, 0.0), 2.0, state)[0].keys == ["w"]
    assert controller.act(Intent(2.0, 0.0), 6.0, state)[0].keys == ["s"]
    assert controller.act(Intent(5.0, 0.0), 5.0, state)[0].keys == []


def test_intent_is_clamped_into_the_drivable_corridor() -> None:
    clamped = Intent(target_speed=99.0, lane_offset=900.0).clamped(40.0)
    assert clamped.target_speed == 10.0 and clamped.lane_offset == 40.0
    assert Intent(-5.0, -900.0).clamped(40.0) == Intent(0.0, -40.0)


def test_the_controller_alone_completes_a_lap() -> None:
    """The attribution baseline. If this failed, any model success would be unearned.

    It is also why the fast layer is tuned against this arm rather than against a
    model run: a deliberately weak controller would manufacture a win for the model.
    """
    world = RacingWorld.from_scene(scene_for(EASY))
    world.terminate_on_opponent_win = False
    result = run_realtime_episode(
        world,
        ConstantIntentPolicy(target_speed=6.0), max_steps=1500, clock="measured",
    )
    assert result["succeeded"], result["reason"]
    assert result["realtime"]["decisions"] == 0, "no model may be consulted"


def test_lane_chatter_stays_out_of_the_tuned_controller() -> None:
    """A gain that turns a half-pixel error into a steer command reverses every tick.

    At gain 1.6 this circuit logged fifteen reversals a lap and crashed at speed 6.
    """
    result = run_realtime_episode(
        RacingWorld.from_scene(scene_for(EASY)),
        ConstantIntentPolicy(target_speed=6.0), max_steps=1500, clock="measured",
    )
    names = [frame.action.value for frame in result["frames"]]
    reversals = sum(1 for a, b in zip(names, names[1:]) if {a, b} == {"left", "right"})
    assert reversals <= 2, f"{reversals} steering reversals suggests the lane gain is chattering"


def test_the_hard_circuit_actually_requires_anticipation() -> None:
    """A benchmark where flat out wins cannot measure anticipation at all.

    On the easy circuit the fastest constant intent beats the oracle, so there is no
    headroom for a model to add anything. This asserts the hard circuit is different:
    flat out fails, and a moderate constant speed finishes.
    """
    flat_out_world = RacingWorld.from_scene(scene_for(HARD))
    flat_out_world.terminate_on_opponent_win = False
    flat_out = run_realtime_episode(
        flat_out_world,
        ConstantIntentPolicy(target_speed=9.0), max_steps=1500, clock="measured",
    )
    moderate_world = RacingWorld.from_scene(scene_for(HARD))
    moderate_world.terminate_on_opponent_win = False
    moderate = run_realtime_episode(
        moderate_world,
        ConstantIntentPolicy(target_speed=4.0), max_steps=1500, clock="measured",
    )
    assert not flat_out["succeeded"], "flat out should not survive the hairpins"
    assert moderate["succeeded"], moderate["reason"]


def test_a_controller_only_policy_needs_no_planner_interface() -> None:
    """The baseline arm has nothing to plan, and must still be schedulable."""
    result = run_realtime_episode(
        RacingWorld.from_scene(scene_for(EASY)),
        ConstantIntentPolicy(target_speed=5.0), max_steps=40, clock="measured",
    )
    assert result["realtime"]["starved_ticks"] == 0
    assert result["realtime"]["fresh_input_fraction"] == 1.0

"""Tests for the matched 2D controller-pipeline evaluation machinery."""

import pytest

from harness.pipeline2d import (
    ConeSkill, ConeSkillDriver, GeneratedConeDriver, prediction_matches, public_cone_state,
)
from harness.policies import PredictiveConeSkillPolicy
from harness.providers import ConeSkillPlan, PredictedConeState, ProviderUsage
from harness.racing import RacingDesignDraft, RacingWorld, compile_racing_scene
from harness.reflex.vision_sense import ConeVisionSense
from harness.realtime import run_realtime_episode
from harness.track_grammar import archetype_plan
from harness.vision import render_racing_forward_cone


def scene():
    return compile_racing_scene(
        "2D pipeline test",
        RacingDesignDraft(
            title="2D pipeline test", rationale="Exercise visual pipeline adapters.",
            circuit="oval", surface="asphalt", obstacle_count=0, npc_count=0,
        ), seed=17,
    )


def test_generated_driver_installs_only_declared_visual_fields() -> None:
    class Plan:
        source = (
            "def control(sense, ctrl, out):\n"
            "    out.discretizer('hysteresis')\n"
            "    out.steer(0.8 * sense.vision_center_near)\n"
            "    out.throttle(0.7 if sense.speed < 1.2 else -0.2)\n"
        )
        reads = ["vision_center_near", "speed"]
        summary = "hold the visible road"

    driver = GeneratedConeDriver(scene())
    installed, reason = driver.install(Plan(), 0)
    assert installed, reason
    assert driver.current_source == Plan.source


def test_skill_driver_uses_only_compact_visual_state_plus_speed() -> None:
    world = RacingWorld.from_scene(scene())
    driver = ConeSkillDriver(world.scene)
    state, frame = driver.observe(world)
    assert set(state) == {
        "speed", "center_near", "center_far", "turn_ahead", "turn_severity",
        "visible_depth", "road_contact", "confidence",
    }
    assert frame.viewpoint == "forward-cone"


@pytest.mark.parametrize("surface", ["asphalt", "clay", "ice"])
def test_cone_sensor_calibrates_the_road_from_pixels_for_every_surface(surface: str) -> None:
    """The shipping camera policy cannot assume that every generated road is asphalt."""
    plan = archetype_plan("oval", surface)
    world = RacingWorld.from_scene(compile_racing_scene(f"{surface} cone", plan, seed=17))
    values = ConeVisionSense().update(render_racing_forward_cone(world))

    assert values["vision_ego_road_contact"]
    assert values["vision_confidence"] >= .8


def test_aggression_increases_2d_skill_pace_without_changing_visual_contract() -> None:
    state = {
        "speed": 1.0, "center_near": 0.0, "center_far": 0.0,
        "turn_ahead": 0.4, "turn_severity": 0.65, "visible_depth": 1.0,
        "road_contact": True, "confidence": 1.0,
    }
    patient = ConeSkillDriver(scene(), aggression=0.0)
    attacker = ConeSkillDriver(scene(), aggression=1.0)
    patient.active = ConeSkill("take_turn", 1.5, turn_direction=1, aggression=0.0)
    attacker.active = ConeSkill("take_turn", 1.5, turn_direction=1, aggression=1.0)
    patient.tick_state(state)
    attacker.tick_state(state)
    assert attacker.last_control_terms["target_speed"] > patient.last_control_terms["target_speed"]
    assert attacker.last_control_terms["steer_signal"] > patient.last_control_terms["steer_signal"]


def test_prediction_rejects_a_reversed_turn() -> None:
    plan = ConeSkillPlan(
        skill="take_turn", target_speed=0.9, target_offset=0,
        turn_direction=1,
        predicted=PredictedConeState(
            speed=1.0, center_near=0.1, turn_ahead=1.0, road_contact=True,
        ),
        speed_tolerance=0.5, lateral_tolerance=0.6, summary="take right turn",
    )
    accepted, diagnostic = prediction_matches(plan, {
        "speed": 1.0, "center_near": 0.1, "turn_ahead": -1.0,
        "road_contact": True, "confidence": 1.0,
    })
    assert not accepted
    assert diagnostic["opposite_turn"]


def test_public_state_does_not_admit_route_or_pose_fields() -> None:
    state = public_cone_state({
        "speed": 1.0, "vision_center_near": 0.2, "vision_center_far": 0.3,
        "vision_turn_ahead": 0.1, "vision_turn_severity": 0.2,
        "vision_lookahead_depth": 0.8, "vision_ego_road_contact": True,
        "vision_confidence": 1.0, "x": 200, "heading": 90, "progress": 0.4,
    })
    assert not ({"x", "heading", "progress"} & set(state))


def test_cone_sensor_contract_bounds_public_geometry() -> None:
    # The prediction schema uses these exact bounds. A sensor value outside
    # them would make an otherwise correct activation prediction impossible.
    state = public_cone_state({
        "speed": 1.0, "vision_center_near": 2.0, "vision_center_far": -2.0,
        "vision_turn_ahead": 2.0, "vision_turn_severity": 2.0,
        "vision_lookahead_depth": 1.0, "vision_ego_road_contact": False,
        "vision_confidence": 0.0,
    })
    assert -2 <= state["center_near"] <= 2
    assert -2 <= state["center_far"] <= 2


def test_predictive_cone_policy_runs_the_evaluated_skill_pipeline(monkeypatch) -> None:
    captured: list[dict] = []

    def plan(*args, **kwargs):
        captured.append(kwargs)
        state = kwargs["public_state"]
        return ConeSkillPlan(
            skill="follow_lane", target_speed=1.3, target_offset=0,
            turn_direction=0,
            predicted=PredictedConeState(
                speed=state["speed"], center_near=state["center_near"],
                turn_ahead=state["turn_ahead"], road_contact=state["road_contact"],
            ),
            speed_tolerance=2, lateral_tolerance=2, summary="follow visible lane",
        ), ProviderUsage(provider="test", model="test", latency_ms=1)

    monkeypatch.setattr("harness.providers.plan_cone_driving_skill", plan)
    result = run_realtime_episode(
        RacingWorld.from_scene(scene()), PredictiveConeSkillPolicy(),
        max_steps=8, clock="fixed", latency_ticks=1, decision_budget=1,
    )

    assert captured
    assert set(captured[0]["public_state"]) == {
        "speed", "center_near", "center_far", "turn_ahead", "turn_severity",
        "visible_depth", "road_contact", "confidence",
    }
    assert result["realtime"]["starved_ticks"] == 0
    assert result["policy_realtime"]["skill_activations"]
    assert captured[0]["driving_aggression"] == .78

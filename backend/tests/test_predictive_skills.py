"""Tests for latency-compensated visual skill selection."""

from __future__ import annotations

import time

from harness.models import ActionName, ElevationProfile, ElevationSpec, RunRequest
from harness.policies import PredictiveVisualSkillPolicy, _visible_skill_failure
from harness.pipeline3d import GeneratedPerspectiveDriver, prediction_matches
from harness.predictive import DrivingSkill, VisualSkillController
from harness.providers import (
    PredictedVisualState, PredictiveSkillPlan, ProviderUsage,
)
from harness.racing import RacingDesignDraft, compile_racing_scene
from harness.racing3d import Racing3DWorld
from harness.realtime import run_realtime_episode
from harness.service import HarnessService
from harness.store import HarnessStore


def world3d() -> Racing3DWorld:
    scene = compile_racing_scene(
        "predictive visual test",
        RacingDesignDraft(
            title="Predictive visual test", rationale="Exercise the camera-only pipeline.",
            circuit="oval", surface="asphalt", obstacle_count=0, npc_count=0,
        ),
        seed=17,
    ).model_copy(update={
        "elevation": ElevationSpec(
            profile=ElevationProfile.ROLLING, amplitude_m=1.0,
            hill_count=2, banking_degrees=2.0,
        ),
    })
    return Racing3DWorld.from_scene(scene)


def test_skill_controller_steers_toward_pixel_derived_road() -> None:
    controller = VisualSkillController()
    sense = {
        "vision_road_contact": True, "vision_confidence": 1.0,
        "vision_track_offset": 0.8, "vision_track_heading": 0.1,
        "vision_bend_ahead": 0.2, "vision_bend_severity": 0.2,
        "vision_crest_risk": 0.0, "vision_recovery_direction": 0.0,
    }
    action, _ = controller.act(DrivingSkill("take_turn", 4.0, turn_direction=1), 3.0, sense)
    assert action.name is ActionName.RIGHT
    assert "d" in action.keys


def test_skill_controller_recovers_using_image_direction_only() -> None:
    controller = VisualSkillController()
    sense = {
        "vision_road_contact": False, "vision_confidence": 0.2,
        "vision_track_offset": 0.0, "vision_track_heading": 0.0,
        "vision_bend_ahead": 0.0, "vision_bend_severity": 0.0,
        "vision_crest_risk": 0.0, "vision_recovery_direction": -0.7,
    }
    action, terms = controller.act(DrivingSkill("follow_lane", 8.0), 4.0, sense)
    assert action.name is ActionName.LEFT
    assert terms["target_speed"] == 1.2


def test_skill_failure_label_contains_only_public_visual_evidence() -> None:
    failure = _visible_skill_failure("3d", "take_turn", {
        "speed": 6.2, "bend_severity": 0.8, "visible_depth": 0.4,
        "road_offset": 0.7, "road_contact": False,
    })
    assert failure["status"] == "pending"
    assert failure["skill"] == "take_turn"
    assert set(failure["evidence_source"]) == {
        "camera_frame", "scalar_speed", "image_features", "skill_history",
    }
    assert set(failure["public_evidence"]) == {
        "speed", "bend_severity", "visible_depth", "lateral_error",
    }


def test_aggression_increases_3d_skill_pace_and_corner_commitment() -> None:
    controller = VisualSkillController()
    sense = {
        "vision_road_contact": True, "vision_confidence": 1.0,
        "vision_track_offset": 0.0, "vision_track_heading": 0.0,
        "vision_bend_ahead": 0.5, "vision_bend_severity": 0.6,
        "vision_crest_risk": 0.0, "vision_visible_depth": 1.0,
        "vision_recovery_direction": 0.0, "vision_left_gap": 0.5,
        "vision_right_gap": 0.5,
    }
    _, patient = controller.act(
        DrivingSkill("take_turn", 6.0, turn_direction=1, aggression=0.0), 3.0, sense,
    )
    _, attacker = controller.act(
        DrivingSkill("take_turn", 6.0, turn_direction=1, aggression=1.0), 3.0, sense,
    )
    assert attacker["target_speed"] > patient["target_speed"]
    assert attacker["steer_signal"] > patient["steer_signal"]


def test_generated_perspective_driver_accepts_only_camera_fields_plus_speed() -> None:
    class Plan:
        source = (
            "def control(sense, ctrl, out):\n"
            "    out.discretizer('hysteresis')\n"
            "    out.steer(0.7 * sense.vision_track_offset + 0.3 * sense.vision_track_heading)\n"
            "    out.throttle(0.7 if sense.speed < 4.0 else -0.4)\n"
        )
        reads = ["vision_track_offset", "vision_track_heading", "speed"]
        summary = "hold the camera-visible road"

    driver = GeneratedPerspectiveDriver(world3d().scene)
    installed, reason = driver.install(Plan(), 0)
    assert installed, reason


def test_perspective_prediction_rejects_reversed_visible_bend() -> None:
    class Plan:
        predicted = PredictedVisualState(
            speed=3.0, road_offset=0.1, bend_ahead=1.2, road_contact=True,
        )
        speed_tolerance = 1.5
        offset_tolerance = 0.8
        bend_tolerance = 0.8

    accepted, diagnostic = prediction_matches(Plan(), {
        "speed": 3.0, "road_offset": 0.1, "bend_ahead": -1.2,
        "road_contact": True, "confidence": 1.0,
    })
    assert not accepted
    assert diagnostic["opposite_bend"]


def test_skills_enforce_operational_speed_ranges() -> None:
    assert DrivingSkill("follow_lane", 0.1).clamped().target_speed == 3.5
    assert DrivingSkill("take_turn", 0.3).clamped().target_speed == 2.5
    assert DrivingSkill("recover_track", 9.0).clamped().target_speed == 3.0


def test_realtime_pipeline_selects_skills_without_starving(monkeypatch) -> None:
    captured: list[dict] = []

    def plan(*args, **kwargs):
        captured.append(kwargs)
        state = kwargs["public_state"]
        return PredictiveSkillPlan(
            predicted=PredictedVisualState(
                speed=state["speed"], road_offset=state["road_offset"],
                bend_ahead=state["bend_ahead"], road_contact=state["road_contact"],
            ),
            skill="follow_lane", target_speed=4.0, target_offset=0.0,
            turn_direction=0, speed_tolerance=5.0, offset_tolerance=2.0,
            bend_tolerance=2.0, summary="follow the visible road",
        ), ProviderUsage(
            provider="test", model="test", input_tokens=20,
            output_tokens=10, latency_ms=200,
        )

    monkeypatch.setattr("harness.policies.plan_predictive_driving_skill", plan)
    policy = PredictiveVisualSkillPolicy()
    result = run_realtime_episode(
        world3d(), policy, max_steps=18, clock="fixed", latency_ticks=2,
    )
    assert captured
    assert set(captured[0]["public_state"]) == {
        "speed", "road_offset", "road_heading", "bend_ahead",
        "bend_severity", "visible_depth", "road_contact", "crest_risk", "confidence",
    }
    assert result["realtime"]["starved_ticks"] == 0
    assert result["policy_realtime"]["skill_activations"]
    assert policy.active_skill.name == "follow_lane"
    assert captured[0]["driving_aggression"] == .78


def test_obsolete_prediction_is_rejected_without_replacing_skill(monkeypatch) -> None:
    def plan(*args, **kwargs):
        state = kwargs["public_state"]
        return PredictiveSkillPlan(
            predicted=PredictedVisualState(
                speed=12.0, road_offset=state["road_offset"],
                bend_ahead=state["bend_ahead"], road_contact=state["road_contact"],
            ),
            skill="take_hairpin", target_speed=2.0, target_offset=0.0,
            turn_direction=1, speed_tolerance=0.5, offset_tolerance=2.0,
            bend_tolerance=2.0, summary="take a predicted hairpin",
        ), ProviderUsage(provider="test", model="test", latency_ms=100)

    monkeypatch.setattr("harness.policies.plan_predictive_driving_skill", plan)
    policy = PredictiveVisualSkillPolicy()
    result = run_realtime_episode(
        world3d(), policy, max_steps=8, clock="fixed", latency_ticks=1,
    )
    assert result["realtime"]["rejected_decisions"] > 0
    assert result["policy_realtime"]["prediction_accepts"] == 0
    assert policy.active_skill.name == "stabilize"


def test_wall_clock_keeps_visual_controller_running_during_model_call(monkeypatch) -> None:
    """The microcontroller must tick while the provider worker is still sleeping."""
    def slow_plan(*args, **kwargs):
        time.sleep(0.25)
        state = kwargs["public_state"]
        return PredictiveSkillPlan(
            predicted=PredictedVisualState(
                speed=state["speed"], road_offset=state["road_offset"],
                bend_ahead=state["bend_ahead"], road_contact=state["road_contact"],
            ),
            skill="follow_lane", target_speed=4.0, target_offset=0.0,
            turn_direction=0, speed_tolerance=5.0, offset_tolerance=2.0,
            bend_tolerance=2.0, summary="follow after real provider latency",
        ), ProviderUsage(provider="test", model="test", latency_ms=250)

    monkeypatch.setattr("harness.policies.plan_predictive_driving_skill", slow_plan)
    result = run_realtime_episode(
        world3d(), PredictiveVisualSkillPolicy(), max_steps=6, clock="wall",
        decision_budget=1,
    )
    assert result["realtime"]["max_decision_ticks"] >= 2
    assert result["realtime"]["starved_ticks"] == 0
    assert max(frame.privileged_state.speed for frame in result["frames"]) > 0


def test_service_routes_predictive_skills_through_real_wall_clock(tmp_path, monkeypatch) -> None:
    def plan(*args, **kwargs):
        state = kwargs["public_state"]
        return PredictiveSkillPlan(
            predicted=PredictedVisualState(
                speed=state["speed"], road_offset=state["road_offset"],
                bend_ahead=state["bend_ahead"], road_contact=state["road_contact"],
            ),
            skill="follow_lane", target_speed=4.0, target_offset=0.0,
            turn_direction=0, speed_tolerance=5.0, offset_tolerance=2.0,
            bend_tolerance=2.0, summary="service wall-clock skill",
        ), ProviderUsage(provider="test", model="test", latency_ms=1)

    monkeypatch.setattr("harness.policies.plan_predictive_driving_skill", plan)
    service = HarnessService(store=HarnessStore(tmp_path))
    environment = service.create_environment(
        "small elevated oval with no opponents", seed=17, provider="offline",
        dimensions="3d", elevation=ElevationSpec(
            profile=ElevationProfile.ROLLING, amplitude_m=1.0,
            hill_count=2, banking_degrees=2.0,
        ),
    )
    run = service.run(RunRequest(
        environment_id=environment.id,
        policy_name="vision-3d-predictive-skills",
        max_steps=10, policy_decision_budget=1,
    ))
    assert run.realtime_metrics["clock"] == "wall"
    assert run.realtime_metrics["control_hz"] == 10
    assert run.realtime_metrics["starved_ticks"] == 0
    assert run.player_turns == 1
    assert run.player_aggression == .78

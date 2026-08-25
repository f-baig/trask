import json
import base64
import io
import math
from pathlib import Path

import pytest

from harness.collision import collider_for
from harness.execution import RolloutDag, RolloutNode, SlurmExecutor
from harness.models import Action, ActionName, DecisionRecord, ForkRequest, ObservationPacket, ResourceRequest, RunRequest, RunStatus, SceneSpec, Vec2
from harness.policies import AnthropicRacingPolicy, HttpKeyboardPolicy, canonical_policy_name
from harness.policy_protocol import KeyboardState, VisualFrame
from harness.providers import ActionSegment, InterruptDecision, PlayerPlan, ProviderUsage, anthropic_json, plan_racing_actions, review_racing_action
from harness.racing import (
    CAR_RADIUS, ENGINE_ID, RacingDesignDraft, RacingIntentController, RacingLineController, RacingWorld, compile_racing_scene,
    racing_physics_context, racing_public_context, racing_strategy_context, racing_track_map,
    validate_racing_scene, verify_racing_playability,
)
from harness.providers import interrupt_decision_schema, player_plan_schema, race_strategy_schema
from harness.service import HarnessService
from harness.store import HarnessStore


def make_service(tmp_path: Path) -> HarnessService:
    return HarnessService(HarnessStore(tmp_path / "data"))


def green_flag(world: RacingWorld) -> RacingWorld:
    world.countdown_ticks_remaining = 0
    return world


def draft(circuit: str = "technical", obstacles: int = 4, npcs: int = 3) -> RacingDesignDraft:
    return RacingDesignDraft(
        title=f"{circuit.title()} test circuit",
        rationale="A deterministic circuit used to exercise the racing contract.",
        circuit=circuit, surface="asphalt", obstacle_count=obstacles, npc_count=npcs,
    )


def test_racing_structured_output_schemas_are_objects() -> None:
    assert player_plan_schema()["type"] == "object"
    assert interrupt_decision_schema()["type"] == "object"
    assert race_strategy_schema()["type"] == "object"


@pytest.mark.parametrize("circuit", ["oval", "technical", "chicane"])
@pytest.mark.parametrize("surface", ["asphalt", "clay", "ice"])
def test_every_circuit_and_surface_replays_a_certified_lap(circuit: str, surface: str) -> None:
    design = draft(circuit)
    design.surface = surface
    scene = compile_racing_scene(circuit, design, seed=17)
    certificate = verify_racing_playability(scene)
    assert validate_racing_scene(scene) == ["Racing domain contract passed."]
    assert certificate.playable
    # Sector count now follows compiled track length rather than a fixed four,
    # but the gates are still crossed in order and the lap still ends at the line.
    assert certificate.objective_trace == [
        *(f"cross:sector-{number}" for number in range(1, scene.sector_count)),
        "cross:finish-line",
    ]


def test_service_exposes_one_game_engine_and_racing_policies(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    environment = service.create_environment(
        "A technical asphalt race with four barriers and no opponents", seed=17,
    )
    # Historical identifiers remain accepted, but a new record is always canonical.
    run = service.run(RunRequest(environment_id=environment.id, policy_name="racing-line", max_steps=1_400))
    assert service.runtime.id == ENGINE_ID
    assert set(service.policies) == {
        "oracle-racing-line", "telemetry-direct", "telemetry-strategy",
        "telemetry-hierarchical", "telemetry-reflex", "vision-reflex-sim-rehearsal",
        "vision-2d-predictive-skills", "vision-2d-direct", "vision-3d-direct-every-tick",
        "vision-3d-direct-short", "vision-3d-direct-short-features",
        "vision-3d-predictive-skills",
        "baseline-constant-intent", "baseline-random",
    }
    assert canonical_policy_name("racing-agent") == "telemetry-direct"
    assert service._require_policy("racing-agent") is service.policies["telemetry-direct"]
    assert run.policy_name == "oracle-racing-line"
    assert environment.scene.domain_pack_version == ENGINE_ID
    assert environment.baseline_solved and environment.playability_certificate and environment.playability_certificate.playable
    assert run.status == RunStatus.SUCCEEDED
    assert run.frames[-1].events[0] == "crossed finish-line"
    assert run.frames[-1].events[1].startswith("race completed P")
    assert run.frames[-1].privileged_state.objective_index == len(environment.scene.objectives)
    assert run.frames[-1].privileged_state.lap == environment.scene.laps


def test_player_observation_is_public_racing_telemetry_only() -> None:
    scene = compile_racing_scene("test", draft(), seed=9)
    world = RacingWorld.from_scene(scene)
    observation = world.observe()
    payload = observation.model_dump()
    context = racing_public_context(scene, observation)
    assert {"heading", "speed", "checkpoint_index", "proprioception"} <= payload.keys()
    assert "reward" not in payload and "lap" not in payload and "privileged_state" not in payload
    assert context["tool_surface"] == "racing-line-v2"
    assert {"progress_percent", "signed_lane_offset", "centerline_heading_error"} <= context["track_state"].keys()
    assert all({"distance", "absolute_heading", "heading_error"} <= point.keys() for point in context["racing_line_lookahead"])
    assert "inventory" not in str(context) and "environment agent" not in str(context)


def test_physics_context_matches_ice_runtime_one_tick_outcomes() -> None:
    import math

    scene = compile_racing_scene(
        "ice physics", draft(circuit="chicane", obstacles=0, npcs=0).model_copy(update={"surface": "ice"}), seed=17,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    world.speed = 6.2
    world.longitudinal_velocity_mps = world.speed * world.dynamics.control_hz / world.dynamics.pixels_per_meter
    observation = world.observe()
    physics = racing_physics_context(scene, observation)

    assert physics["model"] == "transient-bicycle-v1"
    assert physics["integration"] == {
        "physics_hz": 60, "control_hz": 10, "substeps_per_control": 6,
        "method": "fixed-step semi-implicit transient bicycle",
    }
    assert physics["limits"]["max_steering_angle_degrees"] == 20
    assert physics["vehicle"]["mass_kg"] == 1_180
    assert physics["road"]["friction_coefficient"] == .32
    assert physics["aerodynamics"]["drag_coefficient"] == .32
    assert physics["braking_from_current_speed"]["ticks_to_stop"] > 0

    start_heading = world.heading
    start = world.player.model_copy()
    predicted = physics["next_tick_outcomes"]["left"]
    world.step(Action(name=ActionName.LEFT))
    assert world.heading == pytest.approx((start_heading + predicted["heading_delta_degrees"]) % 360, abs=.02)
    assert world.speed == pytest.approx(predicted["next_speed"], abs=.02)
    assert math.hypot(world.player.x - start.x, world.player.y - start.y) == pytest.approx(
        predicted["travel_distance_this_tick"], abs=.02,
    )


def test_physics_context_uses_active_off_track_speed_cap() -> None:
    scene = compile_racing_scene(
        "terrain physics", draft(circuit="chicane", obstacles=0, npcs=0).model_copy(update={"surface": "ice"}), seed=17,
    )
    world = RacingWorld.from_scene(scene)
    world.player = Vec2(x=20, y=20)
    world.speed = 3
    physics = racing_physics_context(scene, world.observe())
    assert not physics["currently_on_track"]
    assert physics["off_track_behavior"]["speed_cap"] == 3
    assert physics["next_tick_outcomes"]["forward"]["next_speed"] <= 3


def test_direct_policy_exhaustion_terminates_instead_of_silently_idling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def one_tick_plan(*args, **kwargs):
        return PlayerPlan(
            subgoal="hold position", summary="single permitted control", confidence=1,
            actions=[ActionSegment(action="idle", steps=1)],
        ), ProviderUsage(provider="test", model="test", output_tokens=1)

    monkeypatch.setattr("harness.policies.plan_racing_actions", one_tick_plan)
    service = make_service(tmp_path)
    environment = service.create_environment("An oval with no opponents", seed=7)
    service.policies["limited-direct"] = AnthropicRacingPolicy(name="limited-direct")
    run = service.run(RunRequest(
        environment_id=environment.id, policy_name="limited-direct",
        max_steps=20, policy_decision_budget=1,
    ))
    assert run.status == RunStatus.FAILED
    assert len(run.frames) == 31
    assert all(frame.decision is None for frame in run.frames[:30])
    assert run.result_reason == "policy call budget exhausted after 1 decisions; no fallback action was executed"


def test_default_policy_budget_allows_one_planner_call_per_simulator_tick() -> None:
    policy = AnthropicRacingPolicy()
    policy.configure_episode(400)
    assert policy.max_turns == 400
    assert policy.output_token_budget == 88_000


def test_direct_policy_reobserves_each_tick_under_high_control_risk(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def long_plan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return PlayerPlan(
            subgoal="test risky cadence", summary="return a deliberately long plan", confidence=1,
            actions=[ActionSegment(action="forward", steps=4)],
        ), ProviderUsage(provider="test", model="test", output_tokens=1)

    monkeypatch.setattr("harness.policies.plan_racing_actions", long_plan)
    scene = compile_racing_scene("risk", draft(obstacles=0, npcs=0), seed=7)
    world = green_flag(RacingWorld.from_scene(scene))
    observation = world.observe().model_copy(update={"heading": (world.heading + 90) % 360})
    policy = AnthropicRacingPolicy()
    policy.configure_episode(20, decision_budget=5)
    policy.reset(scene, scene.seed)
    policy.act(observation)
    policy.act(observation.model_copy(update={"step": 1}))
    assert calls == 2


def test_interrupt_guard_reviews_each_queued_tick_and_replans_on_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    plan_calls = 0
    review_calls = 0
    planner_contexts: list[dict] = []

    def chunk_plan(*args, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        planner_contexts.append(args[0])
        action = "forward" if plan_calls == 1 else "left"
        return PlayerPlan(
            subgoal="guarded chunk", summary="plan several simulator ticks", confidence=1,
            actions=[ActionSegment(action=action, steps=3)],
        ), ProviderUsage(provider="test", model="sonnet", output_tokens=5)

    def review(*args, **kwargs):
        nonlocal review_calls
        review_calls += 1
        interrupt = review_calls == 2
        return InterruptDecision(
            interrupt=interrupt,
            reason="queued action remains safe" if not interrupt else "road changed; replan now",
            confidence=1,
        ), ProviderUsage(provider="test", model="haiku", output_tokens=2)

    monkeypatch.setenv("RACING_INTERRUPT_GUARD", "1")
    monkeypatch.setattr("harness.policies.plan_racing_actions", chunk_plan)
    monkeypatch.setattr("harness.policies.review_racing_action", review)
    scene = compile_racing_scene("guard", draft(obstacles=0, npcs=0), seed=7)
    world = green_flag(RacingWorld.from_scene(scene))
    frame = VisualFrame(media_type="image/png", data_base64="frame", width=480, height=320)
    policy = AnthropicRacingPolicy()
    policy.configure_episode(20, decision_budget=2)
    policy.reset(scene, scene.seed)

    first, _ = policy.act_visual(world.observe(), frame)
    second, _ = policy.act_visual(world.observe().model_copy(update={"step": 1}), frame)
    third, _ = policy.act_visual(world.observe().model_copy(update={"step": 2}), frame)

    assert [first.name, second.name, third.name] == [ActionName.FORWARD, ActionName.FORWARD, ActionName.LEFT]
    assert plan_calls == 2 and review_calls == 2 and policy.interruptions == 1
    assert policy.planning_turns == 2
    assert [usage.model for usage in policy.turn_usages or []] == ["sonnet", "haiku", "haiku", "sonnet"]
    assert planner_contexts[1]["safety_interrupt"]["rejected_action"] == "forward"
    assert planner_contexts[1]["safety_interrupt"]["reason"] == "road changed; replan now"


def test_steering_from_rest_does_not_inject_hidden_throttle() -> None:
    scene = compile_racing_scene("test", draft(obstacles=0, npcs=0), seed=9)
    world = green_flag(RacingWorld.from_scene(scene))
    start = world.player.model_copy()
    heading = world.heading
    decision = DecisionRecord(action=ActionName.RIGHT, subgoal="align", confidence=1, summary="align")
    world.step(Action(name=ActionName.RIGHT), decision)
    assert world.player == start
    assert world.speed == 0
    assert world.heading == heading
    assert world.steering_angle_radians > 0


def test_compound_wasd_applies_throttle_and_steering_together() -> None:
    scene = compile_racing_scene("test", draft(obstacles=0, npcs=0), seed=9)
    world = green_flag(RacingWorld.from_scene(scene))
    start_heading = world.heading
    decision = DecisionRecord(action=ActionName.LEFT, subgoal="keyboard", confidence=1, summary="W+A")
    frame = world.step(Action(keys=["w", "a"]), decision)
    assert world.speed > 0
    assert world.heading != start_heading
    assert frame.keys == ["w", "a"] and frame.action == ActionName.LEFT


def test_model_policy_can_emit_compound_physical_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    def compound_plan(*args, **kwargs):
        return PlayerPlan(
            subgoal="powered left turn", summary="hold throttle while steering", confidence=1,
            actions=[ActionSegment(action="left", keys=["w", "a"], steps=1)],
        ), ProviderUsage(provider="test", model="test")

    monkeypatch.setattr("harness.policies.plan_racing_actions", compound_plan)
    scene = compile_racing_scene("compound model control", draft(obstacles=0, npcs=0), seed=9)
    policy = AnthropicRacingPolicy()
    policy.configure_episode(10, 1)
    policy.reset(scene, scene.seed)
    action, decision = policy.act(RacingWorld.from_scene(scene).observe())
    assert action.keys == ["w", "a"]
    assert action.name == decision.action == ActionName.LEFT
    with pytest.raises(ValueError, match="simultaneously"):
        ActionSegment(action="left", keys=["a", "d"], steps=1)


def test_external_policy_contract_supports_held_compound_keys() -> None:
    class FakePolicy(HttpKeyboardPolicy):
        requests: list[str] = []

        def _post(self, path: str, payload: dict) -> dict:
            self.requests.append(path)
            if path == "reset":
                return {}
            return {
                "protocol": "racelab-policy/v3", "episode_id": payload["episode_id"],
                "control": {"keys": ["w", "a"], "repeat": 2},
            }

    scene = compile_racing_scene("test", draft(obstacles=0, npcs=0), seed=9)
    world = RacingWorld.from_scene(scene)
    policy = FakePolicy("http://policy.invalid")
    policy.reset(scene, scene.seed)
    frame = world.render_policy_frame()
    assert frame.media_type == "image/png" and frame.width == 480 and frame.height == 320
    first, _ = policy.act_visual(world.observe(), frame)
    second, _ = policy.act(world.observe())
    assert first.keys == second.keys == ["w", "a"]
    assert policy.requests == ["reset", "act"]
    with pytest.raises(ValueError):
        KeyboardState(keys=["w", "s"])


def test_external_policy_session_can_own_a_complete_lap(tmp_path: Path) -> None:
    class ConformingPolicy(HttpKeyboardPolicy):
        controller: RacingLineController

        def _post(self, path: str, payload: dict) -> dict:
            if path == "reset":
                scene = SceneSpec.model_validate(payload["scene"])
                self.controller = RacingLineController()
                self.controller.reset(scene, payload["seed"])
                return {}
            assert payload["frame"]["media_type"] == "image/png"
            observation = ObservationPacket.model_validate(payload["observation"])
            action, _ = self.controller.act(observation)
            keys = action.keys or {
                ActionName.FORWARD: ["w"], ActionName.BACKWARD: ["s"],
                ActionName.LEFT: ["a"], ActionName.RIGHT: ["d"], ActionName.IDLE: [],
            }[action.name]
            return {
                "protocol": "racelab-policy/v3", "episode_id": payload["episode_id"],
                "control": {"keys": keys, "repeat": 1},
            }

    service = make_service(tmp_path)
    environment = service.create_environment("A technical ice circuit with no opponents", seed=17)
    service.policies["conforming-external"] = ConformingPolicy("http://policy.invalid", name="conforming-external")
    run = service.run(RunRequest(
        environment_id=environment.id, policy_name="conforming-external", max_steps=1_400,
    ))
    assert run.status == RunStatus.SUCCEEDED
    assert all(frame.keys in (["w"], ["a"], ["d"], ["s"], ["w", "a"], ["w", "d"], ["s", "a"], ["s", "d"], []) for frame in run.frames)


def test_anthropic_visual_request_contains_real_image_block(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps({
                "content": [{"type": "text", "text": '{"action":"forward"}'}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }).encode()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result, _ = anthropic_json(
        system="drive", prompt="act", model="claude-sonnet-5", max_tokens=20,
        json_schema={"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False},
        image_media_type="image/png", image_data_base64="iVBORw0KGgo=",
    )
    assert result == {"action": "forward"}
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/png"
    assert content[1] == {"type": "text", "text": "act"}


def test_gpt_model_uses_openai_strict_json_and_records_cached_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class Response:
        is_error = False
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": '{"action":"forward"}'}}],
                "usage": {
                    "prompt_tokens": 24, "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 10},
                },
            }

    def fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["body"] = json
        return Response()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("httpx.post", fake_post)
    result, usage = anthropic_json(
        system="drive", prompt="act", model="gpt-5-nano", max_tokens=20,
        json_schema={"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False},
        image_media_type="image/png", image_data_base64="iVBORw0KGgo=",
    )
    assert result == {"action": "forward"}
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["body"]["response_format"]["json_schema"]["strict"] is False
    assert captured["body"]["messages"][1]["content"][0]["type"] == "image_url"
    assert usage.provider == "openai"
    assert usage.input_tokens == 24 and usage.uncached_input_tokens == 14
    assert usage.cache_read_input_tokens == 10 and usage.output_tokens == 3


def test_anthropic_cache_marks_only_stable_system_and_tracks_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self) -> bytes:
            return json.dumps({
                "content": [{"type": "text", "text": '{"action":"forward"}'}],
                "usage": {
                    "input_tokens": 200,
                    "cache_creation_input_tokens": 1_100,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 2,
                },
            }).encode()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _, usage = anthropic_json(
        system="stable driver protocol", prompt="changing frame telemetry",
        model="claude-sonnet-5", max_tokens=20, cache_system=True,
        json_schema={"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False},
    )
    assert captured["system"] == [{
        "type": "text", "text": "stable driver protocol",
        "cache_control": {"type": "ephemeral"},
    }]
    assert captured["messages"][0]["content"] == "changing frame telemetry"
    assert usage.input_tokens == 1_300
    assert usage.uncached_input_tokens == 200
    assert usage.cache_creation_input_tokens == 1_100
    assert usage.cache_read_input_tokens == 0


def test_anthropic_prompt_cache_has_environment_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self) -> bytes:
            return json.dumps({
                "content": [{"type": "text", "text": '{"action":"forward"}'}],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }).encode()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHE", "0")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    anthropic_json(
        system="stable", prompt="dynamic", model="claude-sonnet-5", max_tokens=20,
        cache_system=True,
        json_schema={"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False},
    )
    assert captured["system"] == "stable"


def test_anthropic_visual_stack_preserves_oldest_to_newest_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self) -> bytes:
            return json.dumps({
                "content": [{"type": "text", "text": '{"action":"forward"}'}],
                "usage": {},
            }).encode()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    frames = [
        VisualFrame(media_type="image/png", data_base64=value, width=480, height=320)
        for value in ("oldest", "middle", "current")
    ]
    anthropic_json(
        system="drive", prompt="act", model="claude-sonnet-5", max_tokens=20,
        json_schema={"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False},
        image_frames=frames,
    )
    content = captured["messages"][0]["content"]
    assert [block["source"]["data"] for block in content[:-1]] == ["oldest", "middle", "current"]
    assert content[-1]["type"] == "text"


def test_anthropic_structured_output_retries_once_and_accounts_for_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict] = []
    responses = iter([
        {
            "content": [{"type": "text", "text": '{"action":'}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
        {
            "content": [{"type": "text", "text": '{"action":"forward"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 3},
        },
    ])

    class Response:
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self) -> bytes: return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return Response(next(responses))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result, usage = anthropic_json(
        system="drive", prompt="act", model="claude-sonnet-5", max_tokens=220,
        json_schema={"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False},
    )
    assert result == {"action": "forward"}
    assert [request["max_tokens"] for request in requests] == [220, 440]
    assert usage.input_tokens == 21 and usage.output_tokens == 23


def test_anthropic_read_timeout_retries_same_decision_once(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self) -> bytes:
            return json.dumps({
                "content": [{"type": "text", "text": '{"action":"forward"}'}],
                "usage": {"input_tokens": 7, "output_tokens": 2},
            }).encode()

    def fake_urlopen(request, timeout):
        nonlocal requests
        requests += 1
        if requests == 1:
            raise TimeoutError("read timed out")
        return Response()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result, usage = anthropic_json(
        system="drive", prompt="act", model="claude-sonnet-5", max_tokens=20,
        json_schema={"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False},
    )
    assert requests == 2 and result == {"action": "forward"}
    assert usage.input_tokens == 7 and usage.output_tokens == 2


def test_interrupt_review_bounds_verbose_model_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_anthropic_json(**kwargs):
        return {"interrupt": False, "reason": "safe " * 100, "confidence": 1.4}, ProviderUsage(provider="test", model="haiku")

    monkeypatch.setattr("harness.providers.anthropic_json", fake_anthropic_json)
    context = {
        "telemetry": {"speed": 5},
        "track_state": {
            "progress_percent": 10, "signed_lane_offset": 0,
            "centerline_heading_error": 0, "safe_lane_half_width": 55,
        },
        "nearby": [],
    }
    frame = VisualFrame(media_type="image/png", data_base64="frame", width=480, height=320)
    review, _ = review_racing_action(context, ["forward"], frame)
    assert not review.interrupt and len(review.reason) == 240 and review.confidence == 1


def test_direct_policy_visual_history_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    histories: list[list[str]] = []

    def capture_plan(*args, visual_frames=None, **kwargs):
        histories.append([frame.data_base64 for frame in visual_frames or []])
        return PlayerPlan(
            subgoal="observe temporal motion", summary="one tick to force another observation", confidence=1,
            actions=[ActionSegment(action="idle", steps=1)],
        ), ProviderUsage(provider="test", model="test", output_tokens=1)

    monkeypatch.setenv("RACING_VISUAL_HISTORY", "4")
    monkeypatch.setattr("harness.policies.plan_racing_actions", capture_plan)
    scene = compile_racing_scene("history", draft(obstacles=0, npcs=0), seed=7)
    world = RacingWorld.from_scene(scene)
    policy = AnthropicRacingPolicy()
    policy.configure_episode(20, decision_budget=5)
    policy.reset(scene, scene.seed)
    for step, value in enumerate(("one", "two", "three", "four", "five")):
        frame = VisualFrame(media_type="image/png", data_base64=value, width=480, height=320)
        policy.act_visual(world.observe().model_copy(update={"step": step}), frame)
    assert histories[-1] == ["two", "three", "four", "five"]


def test_forward_cone_frame_hides_global_and_rear_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import pygame

    monkeypatch.setenv("RACING_POLICY_VIEW", "forward-cone")
    scene = compile_racing_scene("cone", draft(obstacles=4, npcs=3), seed=17)
    world = RacingWorld.from_scene(scene)
    frame = world.render_policy_frame()
    assert frame.viewpoint == "forward-cone"
    assert frame.horizontal_fov_degrees == 120 and frame.range_pixels == 330
    assert frame.orientation == "ego-forward-up" and frame.ego_anchor == "bottom-center"
    image = pygame.image.load(io.BytesIO(base64.b64decode(frame.data_base64)))
    # Both rear corners and a point directly behind the player are sensor-black.
    assert image.get_at((10, frame.height - 10))[:3] == (0, 0, 0)
    assert image.get_at((frame.width - 10, frame.height - 10))[:3] == (0, 0, 0)
    assert image.get_at((frame.width // 2, frame.height - 2))[:3] == (0, 0, 0)
    # The player marker at the cone apex remains visible and forward-oriented.
    assert image.get_at((frame.width // 2, frame.height - 28))[:3] != (0, 0, 0)


def test_forward_cone_heading_guide_projects_current_heading(monkeypatch: pytest.MonkeyPatch) -> None:
    import pygame

    monkeypatch.setenv("RACING_POLICY_VIEW", "forward-cone")
    monkeypatch.setenv("RACING_HEADING_GUIDE", "1")
    scene = compile_racing_scene("guide", draft(obstacles=0, npcs=0), seed=17)
    world = RacingWorld.from_scene(scene)
    frame = world.render_policy_frame()
    image = pygame.image.load(io.BytesIO(base64.b64decode(frame.data_base64)))
    assert frame.heading_guide
    assert frame.heading_guide_semantics == "current-ego-heading"
    # A dashed guide segment lies on the vertical ego-forward axis.
    assert image.get_at((frame.width // 2, frame.height - 55))[:3] == (255, 222, 64)


def test_overhead_track_has_round_continuous_road_at_centerline_joins(monkeypatch: pytest.MonkeyPatch) -> None:
    import math
    import pygame

    monkeypatch.setenv("RACING_POLICY_VIEW", "overhead")
    scene = compile_racing_scene("continuous-render", draft(circuit="chicane", obstacles=0, npcs=0), seed=17)
    world = RacingWorld.from_scene(scene)
    frame = world.render_policy_frame()
    image = pygame.image.load(io.BytesIO(base64.b64decode(frame.data_base64)))
    road_color = (42, 45, 52)
    sx, sy = frame.width / scene.bounds.width, frame.height / scene.bounds.height
    player_pixel = (round(world.player.x * sx), round(world.player.y * sy))
    road_radius = round(scene.track_width * (sx + sy) / 2) // 2 - 3
    # Skip the spawn neighborhood because the player marker intentionally covers it.
    for point in scene.track_centerline:
        if math.hypot(point.x - world.player.x, point.y - world.player.y) < 65:
            continue
        center = (round(point.x * sx), round(point.y * sy))
        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            sample = (
                round(center[0] + math.cos(radians) * road_radius),
                round(center[1] + math.sin(radians) * road_radius),
            )
            if math.hypot(sample[0] - player_pixel[0], sample[1] - player_pixel[1]) < 22:
                continue
            assert image.get_at(sample)[:3] == road_color


def test_surface_palettes_have_strong_road_terrain_separation() -> None:
    import math
    from harness.vision import SURFACE_PALETTES

    for ground, road, edge in SURFACE_PALETTES.values():
        color_distance = math.sqrt(sum((road[channel] - ground[channel]) ** 2 for channel in range(3)))
        assert color_distance >= 75
        assert sum(edge) > sum(road)


def test_cone_model_context_omits_global_racing_line(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_anthropic_json(**kwargs):
        captured.update(kwargs)
        return {
            "subgoal": "follow visible road", "summary": "steer from cone pixels", "confidence": 1,
            "actions": [{"action": "left", "steps": 1}],
        }, ProviderUsage(provider="test", model="test")

    monkeypatch.setattr("harness.providers.anthropic_json", fake_anthropic_json)
    context = {
        "telemetry": {"x": 123, "y": 456, "heading_degrees": 270, "speed": 6},
        "active_checkpoint": "sector two", "nearby": [], "recent_trajectory": [],
        "controls": {"forward": "throttle"},
        "physics": {
            "braking_from_current_speed": {"ticks_to_stop": 3, "distance_until_stopped": 5.2},
            "next_tick_outcomes": {
                "left": {"heading_delta_degrees": -8, "travel_distance_this_tick": 6.2},
                "backward": {"heading_delta_degrees": 0, "travel_distance_this_tick": 3.8},
            },
        },
        "safety_interrupt": {
            "rejected_action": "left", "reason": "left steers toward barrier", "confidence": 1,
        },
        "racing_line_lookahead": [{"x": 10, "y": 20, "heading_error": 99}],
        "track_state": {
            "progress_percent": 42.5,
            "on_track": False,
            "signed_lane_offset": 12, "centerline_heading_error": -12,
            "safe_lane_half_width": 55,
        },
    }
    frame = VisualFrame(
        media_type="image/png", data_base64="iVBORw0KGgo=", width=480, height=320,
        viewpoint="forward-cone", orientation="ego-forward-up", ego_anchor="bottom-center",
        horizontal_fov_degrees=120, range_pixels=330,
    )
    plan, _ = plan_racing_actions(context, visual_frame=frame)
    assert plan.actions[0].action == "left"
    assert "racing_line_lookahead" not in captured["prompt"]
    assert '"local_heading_error":-12' in captured["prompt"]
    assert '"track_completion_percent":42.5' in captured["prompt"]
    assert '"on_track":false' in captured["prompt"]
    assert '"orientation":"ego-forward-up"' in captured["system"]
    assert '"ego_anchor":"bottom-center"' in captured["system"]
    assert '"rejected_action":"left"' in captured["prompt"]
    assert '"reason":"left steers toward barrier"' in captured["prompt"]
    assert '"distance_until_stopped":5.2' in captured["prompt"]
    assert '"next_tick_outcome_columns":["heading_delta_degrees","next_speed","forward_displacement","lateral_displacement"]' in captured["prompt"]
    assert '"left":[-8' in captured["prompt"]
    assert "avoid imminent collision, remain on drivable road" in captured["system"]
    assert "outside terrain is recoverable, not terminal" in captured["system"]
    assert "steer immediately" not in captured["system"]
    assert '"heading_error":99' not in captured["prompt"]
    assert '"x":123' not in captured["prompt"] and '"y":456' not in captured["prompt"]


def test_continuity_hint_prevents_chicane_track_index_jump() -> None:
    scene = compile_racing_scene("chicane", draft(circuit="chicane"), seed=17)
    hint = 30
    distant_arm = scene.track_centerline[70]
    observation = RacingWorld.from_scene(scene).observe().model_copy(update={"proprioception": distant_arm})
    context = racing_public_context(scene, observation, track_index_hint=hint)
    selected = context["track_state"]["centerline_index"]
    signed_delta = (selected - hint + len(scene.track_centerline) // 2) % len(scene.track_centerline) - len(scene.track_centerline) // 2
    assert -4 <= signed_delta <= 12


def test_sector_strategy_is_public_and_locally_executable() -> None:
    scene = compile_racing_scene("test", draft(obstacles=0, npcs=0), seed=9)
    context = racing_strategy_context(scene)
    assert context["tool_surface"] == "racing-strategy-v1"
    assert len(context["sectors"]) == 12
    assert "player" not in str(context) and "reward" not in str(context)
    controller = RacingIntentController(
        scene,
        [{"sector": sector, "target_speed": 6.0, "lane_offset": 0.0} for sector in range(12)],
        "test strategy",
    )
    world = RacingWorld.from_scene(scene)
    for _ in range(1_400):
        if world.terminated:
            break
        action, decision = controller.act(world.observe())
        world.step(action, decision)
    assert world.succeeded


def test_strategy_provider_schema_defers_exact_cardinality_to_local_validation() -> None:
    sectors = race_strategy_schema()["properties"]["sectors"]
    assert sectors["type"] == "array"
    assert "minItems" not in sectors
    assert "maxItems" not in sectors


def test_public_observation_tracks_live_npc_position() -> None:
    scene = compile_racing_scene("traffic", draft(obstacles=0, npcs=1), seed=9)
    world = RacingWorld.from_scene(scene)
    world.opponents[0].position = Vec2(x=world.player.x + 20, y=world.player.y)
    opponent = next(item for item in world.observe().local_entities if item["kind"] == "npc")
    assert opponent["distance"] == 20.0
    assert opponent["lane_offset"] == 34.0
    assert "track_steps_ahead" in opponent and "speed" in opponent


@pytest.mark.parametrize("circuit", ["oval", "technical", "chicane"])
def test_npcs_pass_a_stopped_player_without_finish_line_oscillation(circuit: str) -> None:
    scene = compile_racing_scene(
        "stopped-player traffic regression", draft(circuit=circuit, obstacles=0, npcs=3), seed=17,
    )
    world = green_flag(RacingWorld.from_scene(scene))
    world.terminate_on_opponent_win = False
    phases_seen = {opponent.entity_id: set() for opponent in world.opponents}
    lane_changes = {opponent.entity_id: 0.0 for opponent in world.opponents}
    previous_indices = {opponent.entity_id: opponent.track_index for opponent in world.opponents}
    previous_headings = {opponent.entity_id: opponent.heading for opponent in world.opponents}
    seam_crossed = set()
    minimum_gap = float("inf")

    for _ in range(500):
        world.step(Action(name=ActionName.IDLE))
        assert not world.terminated, world.reason
        for opponent in world.opponents:
            phases_seen[opponent.entity_id].add(opponent.overtake_phase)
            lane_changes[opponent.entity_id] = max(
                lane_changes[opponent.entity_id],
                abs(opponent.lane_offset - opponent.base_lane_offset),
            )
            progress_delta = (opponent.track_index - previous_indices[opponent.entity_id]) % len(scene.track_centerline)
            assert progress_delta <= 12
            if previous_indices[opponent.entity_id] > len(scene.track_centerline) - 8 and opponent.track_index < 8:
                seam_crossed.add(opponent.entity_id)
            previous_indices[opponent.entity_id] = opponent.track_index
            heading_delta = abs((opponent.heading - previous_headings[opponent.entity_id] + 180) % 360 - 180)
            assert heading_delta <= 22.5 + 1e-6
            previous_headings[opponent.entity_id] = opponent.heading
            minimum_gap = min(minimum_gap, math.hypot(
                opponent.position.x - world.player.x,
                opponent.position.y - world.player.y,
            ))

    assert seam_crossed == set(phases_seen)
    assert all("passing" in phases for phases in phases_seen.values())
    assert all(change >= 9.5 for change in lane_changes.values())
    assert minimum_gap >= 2 * 11.0


def test_npc_overtake_state_round_trips_through_snapshot() -> None:
    scene = compile_racing_scene("overtake fork", draft(circuit="oval", obstacles=0, npcs=1), seed=17)
    world = green_flag(RacingWorld.from_scene(scene))
    # This is a long traffic-state probe, not a user-facing competitive race.
    world.terminate_on_opponent_win = False
    for _ in range(400):
        world.step(Action(name=ActionName.IDLE))
        if world.opponents[0].overtake_phase == "passing":
            break
    assert world.opponents[0].overtake_phase == "passing"
    restored = green_flag(RacingWorld.from_scene(scene))
    restored.restore(world.snapshot())
    assert restored.snapshot() == world.snapshot()


def test_npc_passes_a_player_stopped_away_from_finish_line() -> None:
    scene = compile_racing_scene("mid-track stop", draft(circuit="technical", obstacles=0, npcs=1), seed=17)
    world = green_flag(RacingWorld.from_scene(scene))
    world.terminate_on_opponent_win = False
    world.player = scene.track_centerline[35].model_copy()
    phases = set()
    minimum_gap = float("inf")
    for _ in range(500):
        world.step(Action(name=ActionName.IDLE))
        opponent = world.opponents[0]
        phases.add(opponent.overtake_phase)
        minimum_gap = min(minimum_gap, math.hypot(
            opponent.position.x - world.player.x,
            opponent.position.y - world.player.y,
        ))
    assert not world.terminated
    assert {"passing", "merge"} <= phases
    assert minimum_gap >= 2 * 11.0


@pytest.mark.parametrize("circuit", ["oval", "technical", "chicane"])
@pytest.mark.parametrize("surface", ["asphalt", "clay", "ice"])
def test_sector_intent_completes_every_circuit_and_surface_with_traffic(
    circuit: str, surface: str,
) -> None:
    """A grip-aware sector plan finishes on every circuit at maximum traffic.

    The pace is the same grip-derived corner target the deterministic oracle
    uses, so a model plan that reasons about grip as well as the oracle does is
    guaranteed a completable race on every circuit and surface, even with the
    maximum barrier and traffic load.
    """
    design = draft(circuit=circuit, obstacles=4, npcs=3)
    design.surface = surface
    scene = compile_racing_scene("traffic safety", design, seed=17)
    assert racing_track_map(scene), "compiled circuit must publish its corner map"
    grip = (
        scene.dynamics.road.friction_coefficient
        * scene.dynamics.road.lateral_grip_multiplier
        * scene.dynamics.vehicle.tire_friction_multiplier
    )
    pace = max(1.8, min(4.0, 4.0 * math.sqrt(grip)))
    intents = [
        {"sector": sector, "target_speed": pace, "lane_offset": 0.0} for sector in range(12)
    ]
    for action_delay in (False, True):
        controller = RacingIntentController(scene, intents, "steady strategy")
        world = RacingWorld.from_scene(scene)
        world.terminate_on_opponent_win = False
        for _ in range(1_400):
            if world.terminated:
                break
            action, decision = controller.act(world.observe())
            world.step(action, decision, action_delay=action_delay)
        assert world.succeeded, (circuit, surface, action_delay, world.reason)


@pytest.mark.parametrize("circuit", ["oval", "technical", "chicane"])
@pytest.mark.parametrize("surface", ["asphalt", "clay", "ice"])
def test_adversarial_sector_intent_terminates_cleanly(circuit: str, surface: str) -> None:
    """An adversarial plan may crash, but never corrupt the simulation.

    Demanding full-width lane swings and alternating speeds every sector in heavy
    traffic is a bad plan, and the harness deliberately does not rescue it: the
    controller executes what the model asked for. What is guaranteed is that the
    engine ends in a consistent, explainable state instead of raising or drifting
    into non-finite dynamics.
    """
    design = draft(circuit=circuit, obstacles=4, npcs=3)
    design.surface = surface
    scene = compile_racing_scene("traffic safety", design, seed=17)
    intents = [
        {"sector": sector, "target_speed": 3 if sector % 2 else 10, "lane_offset": -24 if sector % 2 else 24}
        for sector in range(12)
    ]
    for action_delay in (False, True):
        controller = RacingIntentController(scene, intents, "adversarial strategy")
        world = RacingWorld.from_scene(scene)
        for _ in range(1_400):
            if world.terminated:
                break
            action, decision = controller.act(world.observe())
            world.step(action, decision, action_delay=action_delay)
        assert world.succeeded or world.reason, (circuit, surface, action_delay)
        state = world.privileged_state()
        assert math.isfinite(state.speed) and state.speed >= 0
        assert math.isfinite(state.heading) and 0 <= state.heading < 360
        assert 0 <= state.objective_index <= len(scene.objectives)
        assert 0 <= world.nitro <= 100
        if world.terminated:
            # A terminated race must stay terminated rather than accept controls.
            with pytest.raises(RuntimeError):
                world.step(Action(name=ActionName.IDLE))


def test_sector_intent_executes_the_model_plan_instead_of_the_oracle() -> None:
    """The strategy policy must measure the model, not the racing-line oracle.

    A single opponent used to make the controller discard every model-authored
    intent and return the oracle's control, so `telemetry-strategy` silently
    benchmarked deterministic code in any scene with traffic.
    """
    scene = compile_racing_scene("intent fidelity", draft(obstacles=0, npcs=3), seed=17)
    slow = [{"sector": sector, "target_speed": 3.0, "lane_offset": 0.0} for sector in range(12)]
    fast = [{"sector": sector, "target_speed": 10.0, "lane_offset": 0.0} for sector in range(12)]
    speeds = []
    for intents in (slow, fast):
        controller = RacingIntentController(scene, intents, "pace probe")
        world = green_flag(RacingWorld.from_scene(scene))
        for _ in range(120):
            if world.terminated:
                break
            action, decision = controller.act(world.observe())
            world.step(action, decision)
        speeds.append(world.speed)
    assert speeds[0] < speeds[1] - 1.0, speeds


def test_leaving_track_applies_slowdown_and_allows_recovery() -> None:
    scene = compile_racing_scene("test", draft(obstacles=0, npcs=0), seed=2)
    world = green_flag(RacingWorld.from_scene(scene))
    world.player.x, world.player.y = 20, 20
    world.speed = 8
    decision = DecisionRecord(action=ActionName.IDLE, subgoal="test", confidence=1, summary="test")
    frame = world.step(Action(name=ActionName.IDLE), decision)
    assert not world.terminated and not world.succeeded and world.reason is None
    assert world.off_track and 0 < world.speed <= 3
    assert world.rolling_resistance_n > 0
    assert frame.reward == pytest.approx(-.021)
    assert frame.events == ["left track: terrain slowdown"]

    world.player = scene.track_centerline[20].model_copy()
    recovered = world.step(Action(name=ActionName.IDLE), decision)
    assert not world.off_track and recovered.events == ["returned to track"]


def test_barrier_collision_rebounds_and_race_continues() -> None:
    scene = compile_racing_scene("test", draft(obstacles=1, npcs=0), seed=2)
    world = green_flag(RacingWorld.from_scene(scene))
    barrier = next(entity for entity in scene.entities if entity.kind.value == "obstacle")
    world.player = Vec2(
        x=barrier.rect.x + barrier.rect.width / 2,
        y=barrier.rect.y + barrier.rect.height / 2,
    )
    frame = world.step(Action(name=ActionName.IDLE))
    collider = collider_for(barrier)
    assert not world.terminated and not world.succeeded and world.reason is None
    assert frame.events[0] == f"bounced off {barrier.id}"
    assert frame.privileged_state.barrier_impact is not None
    assert not collider.hits_circle(world.player.x, world.player.y, CAR_RADIUS)
    # The contact frame is not a disguised terminal state: the race accepts the
    # next control tick and clears its one-frame impact marker.
    following = world.step(Action(name=ActionName.IDLE))
    assert not world.terminated and following.privileged_state.barrier_impact is None


def test_checkpoint_order_cannot_be_skipped() -> None:
    scene = compile_racing_scene("test", draft(obstacles=0, npcs=0), seed=2)
    world = RacingWorld.from_scene(scene)
    finish = next(entity for entity in scene.entities if entity.id == "finish-line")
    world.player.x = finish.rect.x + finish.rect.width / 2
    world.player.y = finish.rect.y + finish.rect.height / 2
    decision = DecisionRecord(action=ActionName.IDLE, subgoal="test", confidence=1, summary="test")
    world.step(Action(name=ActionName.IDLE), decision)
    assert world.objective_index == 0
    assert not world.succeeded


def test_replay_bundle_declares_racing_contract(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    environment = service.create_environment("An oval asphalt race with no obstacles and no opponents", seed=4)
    run = service.run(RunRequest(environment_id=environment.id, policy_name="oracle-racing-line"))
    bundle = service.get_replay_bundle(run.id)
    assert bundle.backend.id == ENGINE_ID
    assert bundle.renderer_hint == "racing-topdown-2d"
    assert bundle.timeline[-1].state["lap"] == 1
    assert "heading" in bundle.timeline[-1].state and "speed" in bundle.timeline[-1].state


def test_racing_replay_can_branch_from_a_natural_language_condition(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    environment = service.create_environment("An oval asphalt race with one opponent", seed=22)
    parent = service.run(RunRequest(environment_id=environment.id, policy_name="oracle-racing-line"))
    condition = "Add steering delay from here"
    child = service.fork_run(parent.id, ForkRequest(fork_step=50, condition=condition))
    assert child.parent_run_id == parent.id and child.fork_step == 50
    assert child.frames[:50] == parent.frames[:50]
    assert child.perturbation == {
        "kind": "action_delay",
        "condition": condition,
        "summary": "Apply action delay from the selected tick.",
    }
    enriched = service.get_run(parent.id)
    assert enriched and enriched.fork_supported and not enriched.guidance_supported
    with pytest.raises(ValueError, match="cannot apply correction guidance"):
        service.fork_run(parent.id, ForkRequest(fork_step=50, condition="Brake earlier here"))


def test_operator_correction_reaches_every_post_fork_model_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = compile_racing_scene("test", draft(obstacles=0, npcs=0), seed=22)
    world = green_flag(RacingWorld.from_scene(scene))
    policy = AnthropicRacingPolicy()
    policy.configure_episode(20, decision_budget=4)
    policy.reset(scene, scene.seed)
    correction = "Brake before this right hander, then hold the inside lane."
    policy.set_episode_guidance(correction)
    received: list[str | None] = []

    def corrected_plan(*args, operator_guidance=None, **kwargs):
        received.append(operator_guidance)
        return PlayerPlan(
            subgoal="apply operator correction", summary="short corrected control",
            confidence=1, actions=[ActionSegment(action="idle", steps=1)],
        ), ProviderUsage(provider="test", model="test", output_tokens=1)

    monkeypatch.setattr("harness.policies.plan_racing_actions", corrected_plan)
    for _ in range(2):
        action, decision = policy.act(world.observe())
        world.step(action, decision)
    assert received == [correction, correction]


def test_slurm_remains_a_control_plane_handoff(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    environment = service.create_environment("A chicane clay race", seed=6)
    run = service.run(RunRequest(
        environment_id=environment.id, execution_backend="slurm",
        resources=ResourceRequest(cpu_cores=4, memory_mb=8192, gpu_count=1, wall_time_seconds=1800, queue="research"),
    ))
    assert run.status == RunStatus.PENDING
    assert run.execution.backend == "slurm"
    assert "#SBATCH --gpus=1" in str(run.execution.scheduler_metadata["script"])


def test_rollout_dag_is_still_executor_neutral() -> None:
    dag = RolloutDag(id="race-study", nodes=(
        RolloutNode(id="compile", run_id="r1"),
        RolloutNode(id="drive", run_id="r2", dependencies=("compile",)),
    ))
    assert [node.id for node in dag.topological_nodes()] == ["compile", "drive"]
    assert "harness-worker run --run-id r2" in SlurmExecutor().sbatch_script(run_id="r2", resources=ResourceRequest())


@pytest.mark.parametrize("subgoal,summary,confidence", [
    (":", "a real summary here", 0.5),
    ("a real subgoal", ":", 0.5),
    ("", "", 0.5),
    ("  ", " x ", 0.5),
    ("ok subgoal", "ok summary", 1.4),
    ("ok subgoal", "ok summary", "not-a-number"),
])
def test_unusable_commentary_does_not_end_the_episode(
    monkeypatch: pytest.MonkeyPatch, subgoal: str, summary: str, confidence: object,
) -> None:
    """Cosmetic fields must never terminate a race.

    A live Haiku episode died at tick 100 because the model wrote ":" as its summary.
    That is one character under a three-character floor on a string the controller
    never reads, and it killed the run — and would have spared the other arm of a
    paired comparison by luck, which is a coin flip masquerading as a result.
    """
    def unusable(**kwargs):
        return {
            "subgoal": subgoal, "summary": summary, "confidence": confidence,
            "actions": [{"action": "forward", "steps": 2}],
        }, ProviderUsage(provider="test", model="test", output_tokens=1)

    monkeypatch.setattr("harness.providers.anthropic_json", unusable)
    scene = compile_racing_scene("clamp", draft(obstacles=0, npcs=0), seed=7)
    world = green_flag(RacingWorld.from_scene(scene))
    plan, _ = plan_racing_actions(racing_public_context(scene, world.observe()))
    assert len(plan.subgoal) >= 3 and len(plan.summary) >= 3
    assert 0 <= plan.confidence <= 1
    assert [segment.action for segment in plan.actions] == ["forward"]


def test_player_schema_declares_string_floors_and_no_unsupported_bounds() -> None:
    """String bounds belong in the schema; numeric and array bounds are a 400.

    Structured output rejects the whole request for numeric minimum/maximum and for
    array minItems/maxItems. Adding them fails every call rather than being ignored,
    so this test pins the split: strings constrained at generation, everything else
    clamped on receipt.
    """
    for schema in (player_plan_schema(), interrupt_decision_schema()):
        properties = schema["properties"]
        assert properties["confidence"] == {"type": "number"}
        for definition in properties.values():
            assert "minimum" not in definition and "maximum" not in definition
            assert "minItems" not in definition and "maxItems" not in definition
    player = player_plan_schema()["properties"]
    assert player["subgoal"]["minLength"] == 3 and player["subgoal"]["maxLength"] == 320
    assert player["summary"]["minLength"] == 3 and player["summary"]["maxLength"] == 600
    reason = interrupt_decision_schema()["properties"]["reason"]
    assert reason["minLength"] == 3 and reason["maxLength"] == 240


def test_an_unusable_interrupt_reason_does_not_end_the_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unusable(**kwargs):
        return {"interrupt": False, "reason": ":", "confidence": 0.5}, ProviderUsage(
            provider="test", model="test", output_tokens=1,
        )

    monkeypatch.setattr("harness.providers.anthropic_json", unusable)
    scene = compile_racing_scene("clamp-critic", draft(obstacles=0, npcs=0), seed=7)
    world = green_flag(RacingWorld.from_scene(scene))
    frame = VisualFrame(media_type="image/png", data_base64="frame", width=480, height=320)
    decision, _ = review_racing_action(
        racing_public_context(scene, world.observe()), ["forward"], frame,
    )
    assert len(decision.reason) >= 3 and decision.interrupt is False

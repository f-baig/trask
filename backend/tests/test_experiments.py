"""The experiments flow: one circuit, a persisted chat about it, and the runs it launches.

The claims here are about bounding and persistence rather than about racing. A sentence must
not be able to launch fifty runs, a request the model over-answers is trimmed rather than
rejected, one failed cell must not lose the others, and what was asked about a circuit has to
still be there when the circuit is opened again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import providers
from harness.models import (
    AgentMessage, ArtifactLink, ElevationSpec, ExperimentAddress, ExperimentRecord, ExperimentRequest,
    PlayerAddress, RunStatus,
)
from harness.service import (
    HarnessService, default_vision_policy, experiment_policy_choices,
)
from harness.store import HarnessStore


BRIEF = "A technical asphalt circuit with two barriers and one opponent."


def service_with_circuit(tmp_path):
    service = HarnessService(store=HarnessStore(tmp_path))
    environment = service.create_environment(BRIEF, seed=17, provider="offline")
    return service, environment


def drain(events) -> list[dict]:
    return list(events)


def test_ordinary_experiments_fix_the_dimension_correct_predictive_player(tmp_path) -> None:
    service, environment = service_with_circuit(tmp_path)
    choices_2d = experiment_policy_choices(
        environment.scene, "compare the direct and reflex agents", service.policies,
    )
    assert choices_2d == ["vision-2d-predictive-skills"]
    assert default_vision_policy(environment.scene) == "vision-2d-predictive-skills"

    scene_3d = environment.scene.model_copy(update={"elevation": ElevationSpec()})
    choices_3d = experiment_policy_choices(scene_3d, "run the default agent", service.policies)
    assert choices_3d == ["vision-3d-predictive-skills"]
    assert default_vision_policy(scene_3d) == "vision-3d-predictive-skills"


def test_telemetry_controls_are_not_available_in_the_researcher_workflow(tmp_path) -> None:
    service, environment = service_with_circuit(tmp_path)
    choices = experiment_policy_choices(
        environment.scene, "compare telemetry direct with telemetry reflex", service.policies,
    )
    assert choices == ["vision-2d-predictive-skills"]


def test_conditions_are_trimmed_to_a_bounded_matrix(monkeypatch) -> None:
    """Policies times perturbations is what actually runs, so the product is what is capped.

    Trimming rather than rejecting: asking for one driver too many is a reasonable request to
    cut down, not a reason to launch nothing.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(providers, "anthropic_json", lambda **_: ({
        "plan": "everything at once",
        "policies": ["oracle-racing-line", "telemetry-direct", "baseline-random", "baseline-constant-intent", "telemetry-reflex"],
        "perturbations": ["none", "action_delay", "low_grip", "heavy_car", "high_drag"],
        "max_steps": 99_999,
    }, providers.ProviderUsage(provider="anthropic", model="test")))

    conditions, _ = providers.plan_run_conditions(
        message="run everything under everything", circuit={"name": "x"},
        policies=["oracle-racing-line", "telemetry-direct", "baseline-random", "baseline-constant-intent", "telemetry-reflex"],
        max_cells=6,
    )
    assert len(conditions.policies) <= 4
    assert len(conditions.perturbations) <= 4
    assert len(conditions.policies) * len(conditions.perturbations) <= 6
    assert conditions.max_steps == 2_000, "an absurd tick budget is clamped, not honoured"


def test_an_unknown_driver_cannot_be_requested(monkeypatch) -> None:
    """The model picks from lists the harness supplies, so a nonexistent driver is impossible.

    Also pins that a plan too terse for the field is replaced rather than allowed to fail the
    request: which runs to launch is still a usable answer without prose around it.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"plan": "p", "policies": ["oracle-racing-line"], "perturbations": ["none"], "max_steps": 400}, \
            providers.ProviderUsage(provider="anthropic", model="test")

    monkeypatch.setattr(providers, "anthropic_json", fake)
    providers.plan_run_conditions(
        message="use the teleport driver", circuit={}, policies=["oracle-racing-line", "baseline-random"],
    )
    enum = captured["json_schema"]["properties"]["policies"]["items"]["enum"]
    assert enum == ["oracle-racing-line", "baseline-random"]
    assert "maxItems" not in captured["json_schema"]["properties"]["policies"], (
        "array bounds are not in Anthropic's structured-output subset and are rejected"
    )


def test_the_offline_flow_runs_the_oracle_and_says_so(tmp_path) -> None:
    """With no key there is nothing to interpret a request with, so it does the honest thing."""
    service, environment = service_with_circuit(tmp_path)
    events = drain(service.dispatch_experiment_events(environment.id, "race everything"))
    kinds = [event["type"] for event in events]
    assert "token" in kinds and "done" in kinds
    plan = next(event["text"] for event in events if event["type"] == "token")
    assert "No model is configured" in plan
    progress = [event for event in events if event["type"] == "progress"]
    assert progress[0] == {
        "type": "progress", "scope": "experiment", "completed": 0, "total": 1,
        "elapsed_ms": 0, "eta_ms": None, "label": "Preparing first run",
    }
    assert progress[-1]["completed"] == progress[-1]["total"] == 1
    assert progress[-1]["eta_ms"] == 0
    assert progress[-1]["elapsed_ms"] >= 0
    done = events[-1]
    assert len(done["run_ids"]) == 1
    assert service.get_run(done["run_ids"][0]).policy_name == "oracle-racing-line"


def test_structured_experiment_can_sweep_aggression_and_tick_budget(tmp_path) -> None:
    """The visual experiment builder has first-class axes, not a chat-only approximation."""
    service, environment = service_with_circuit(tmp_path)
    original_run = service.run
    requested_budgets: list[int] = []

    def inspect_request(request, *args, **kwargs):
        requested_budgets.append(request.max_steps)
        return original_run(request, *args, **kwargs)

    service.run = inspect_request  # type: ignore[method-assign]
    record = service.run_experiment(ExperimentRequest(
        environment_id=environment.id, policies=["oracle-racing-line"],
        perturbations=["normal"], seeds=[environment.scene.seed], max_steps=135,
        player_aggressions=[.45, .78],
    ))
    runs = [service.get_run(run_id) for run_id in record.run_ids]
    assert len(runs) == 2
    assert record.player_aggressions == [.45, .78]
    assert {run.player_aggression for run in runs if run} == {.45, .78}
    assert requested_budgets == [135, 135]


def test_structured_experiment_rejects_a_seed_variant_as_a_second_matrix_source(tmp_path) -> None:
    """A seed variant is a cell within its parent experiment, not a new experiment grid."""
    service, environment = service_with_circuit(tmp_path)
    variant = environment.model_copy(update={
        "id": "env-seed-variant",
        "parent_environment_id": environment.id,
    })
    service.store.save_environment(variant)

    with pytest.raises(ValueError, match="source circuit"):
        service.run_experiment(ExperimentRequest(
            environment_id=variant.id, policies=["oracle-racing-line"],
            perturbations=["normal"], seeds=[variant.scene.seed], max_steps=135,
            player_aggressions=[.78],
        ))
    assert service.list_experiments() == []


def test_the_circuit_chat_persists_and_links_its_runs(tmp_path) -> None:
    """Reopening a circuit has to show what was asked about it and what that launched."""
    service, environment = service_with_circuit(tmp_path)
    drain(service.dispatch_experiment_events(environment.id, "run the oracle"))

    reopened = HarnessService(store=HarnessStore(tmp_path))
    messages = reopened.agent_messages("environment", environment.id)
    assert [item.speaker for item in messages] == ["user", "assistant"]
    assert messages[0].content == "run the oracle"
    assert messages[1].artifacts and messages[1].artifacts[0].kind == "run"
    assert [action.id for action in messages[1].actions] == ["conditions", "run:oracle-racing-line:none"]
    assert messages[1].actions[0].state == "done"
    assert messages[1].actions[1].state in {"done", "failed"}
    assert reopened.get_run(messages[1].artifacts[0].id) is not None
    assert all(item.environment_id == environment.id for item in messages), (
        "the chat is scoped to the circuit, so another circuit's chat cannot leak into it"
    )
    activity = reopened.agent_activity()
    assert activity[0].id == messages[1].id
    assert activity[0].actions[-1].artifact.id == messages[1].artifacts[0].id


def test_experiment_assistant_is_durable_before_the_player_run_finishes(tmp_path) -> None:
    service, environment = service_with_circuit(tmp_path)
    original_run = service.run
    observed_draft: list[str] = []

    def inspect_draft(*args, **kwargs):
        messages = service.agent_messages("environment", environment.id)
        assert [item.speaker for item in messages] == ["user", "assistant"]
        assert "Runs are still in progress" in messages[-1].content
        observed_draft.append(messages[-1].id)
        return original_run(*args, **kwargs)

    service.run = inspect_draft  # type: ignore[method-assign]
    drain(service.dispatch_experiment_events(environment.id, "run the oracle"))
    final = service.agent_messages("environment", environment.id)
    assert observed_draft and final[-1].id == observed_draft[0]
    assert "Runs are still in progress" not in final[-1].content


def test_one_failed_cell_does_not_lose_the_others(tmp_path, monkeypatch) -> None:
    """A driver that raises must cost its own run and nothing else."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(providers, "anthropic_json", lambda **_: ({
        "plan": "two drivers", "policies": ["exploding", "oracle-racing-line"],
        "perturbations": ["none"], "max_steps": 400,
    }, providers.ProviderUsage(provider="anthropic", model="test")))

    service, environment = service_with_circuit(tmp_path)

    class Exploding:
        name = "exploding"

        def reset(self, scene, seed):
            raise RuntimeError("this driver is broken")

        def act(self, observation):
            raise RuntimeError("unreachable")

    service.policies["exploding"] = Exploding()
    events = drain(service.dispatch_experiment_events(environment.id, "race both"))
    states = {
        event["label"].split(":")[0]: event["state"]
        for event in events if event["type"] == "step" and event["id"].startswith("run:")
    }
    assert any(state == "failed" for state in states.values())
    progress = [event for event in events if event["type"] == "progress"]
    assert [(item["completed"], item["total"]) for item in progress] == [(0, 2), (1, 2), (2, 2)]
    assert progress[1]["eta_ms"] >= 0
    assert events[-1]["run_ids"], "the healthy driver still has to produce a run"
    assert service.get_run(events[-1]["run_ids"][0]).policy_name == "oracle-racing-line"


def test_an_unknown_circuit_is_reported_not_swallowed(tmp_path) -> None:
    service = HarnessService(store=HarnessStore(tmp_path))
    events = drain(service.dispatch_experiment_events("race-nope", "run something"))
    assert events[-1]["type"] == "error"
    with pytest.raises(KeyError):
        service._require_environment("race-nope")


def test_deleting_a_run_cascades_to_forks_and_removes_artifacts(tmp_path) -> None:
    service, environment = service_with_circuit(tmp_path)
    events = drain(service.dispatch_experiment_events(environment.id, "run the oracle"))
    parent = service.store.get_run(events[-1]["run_ids"][0])
    assert parent is not None and parent.artifacts
    artifact_path = parent.artifacts[0].uri.removeprefix("file://")

    child = parent.model_copy(update={
        "id": "run-child", "parent_run_id": parent.id,
        "status": RunStatus.RUNNING, "artifacts": [],
    })
    grandchild = child.model_copy(update={
        "id": "run-grandchild", "parent_run_id": child.id,
        "status": RunStatus.SUCCEEDED,
    })
    service.store.save_run(child)
    service.store.save_run(grandchild)

    with pytest.raises(ValueError, match="Active runs cannot be deleted"):
        service.delete_run(parent.id)
    assert service.get_run(parent.id) is not None

    child.status = RunStatus.SUCCEEDED
    service.store.save_run(child)
    result = service.delete_run(parent.id)
    assert set(result["deleted_run_ids"]) == {parent.id, child.id, grandchild.id}
    assert all(service.get_run(run_id) is None for run_id in result["deleted_run_ids"])
    assert not Path(artifact_path).exists()
    messages = service.agent_messages("environment", environment.id)
    assert not any(artifact.kind == "run" for message in messages for artifact in message.artifacts)
    assert service.get_environment(environment.id) is not None


def test_deleting_an_experiment_keeps_circuit_and_unrelated_runs(tmp_path) -> None:
    service, environment = service_with_circuit(tmp_path)
    events = drain(service.dispatch_experiment_events(environment.id, "run the oracle"))
    run = service.store.get_run(events[-1]["run_ids"][0])
    assert run is not None and run.address is not None
    experiment_number = run.address.experiment

    unrelated = run.model_copy(update={
        "id": "run-unrelated", "parent_run_id": None, "artifacts": [],
        "address": PlayerAddress(
            experiment=experiment_number + 1,
            environment=run.address.environment,
            variant=run.address.variant,
            player=1,
        ),
    })
    service.store.save_run(unrelated)
    record = ExperimentRecord(
        id="exp-delete-me", name="Deletion test",
        address=ExperimentAddress(experiment=experiment_number),
        environment_id=environment.id, created_at=run.started_at,
        policies=[run.policy_name], perturbations=["normal"], seeds=[run.seed],
        run_ids=[run.id], summary=service._summarize([{
            "policy": run.policy_name, "perturbation": "normal",
            "environment_id": environment.id, "run_id": run.id,
            "status": run.status.value, "steps": len(run.frames),
            "reward": run.total_reward, "tokens": run.token_usage,
        }]),
    )
    service.store.save_experiment(record)

    result = service.delete_experiment(f"experiment-{experiment_number}", environment.id)
    assert result["deleted_run_ids"] == [run.id]
    assert result["deleted_experiment_ids"] == [record.id]
    assert service.get_run(run.id) is None
    assert service.get_run(unrelated.id) is not None
    assert service.store.get_experiment(record.id) is None
    assert service.get_environment(environment.id) is not None
    messages = service.agent_messages("environment", environment.id)
    assert not any(action.id.startswith("run:") for message in messages for action in message.actions)


def test_deleting_a_circuit_cascades_to_variants_runs_experiments_and_chat(tmp_path) -> None:
    service, environment = service_with_circuit(tmp_path)
    events = drain(service.dispatch_experiment_events(environment.id, "run the oracle"))
    parent_run = service.store.get_run(events[-1]["run_ids"][0])
    assert parent_run is not None

    child = environment.model_copy(update={
        "id": "env-derived", "parent_environment_id": environment.id,
        "origin": "derived test variant",
    })
    unrelated = environment.model_copy(update={
        "id": "env-unrelated", "parent_environment_id": None,
        "origin": "unrelated test circuit",
    })
    service.store.save_environment(child)
    service.store.save_environment(unrelated)
    child_run = parent_run.model_copy(update={
        "id": "run-derived", "environment_id": child.id,
        "parent_run_id": None, "status": RunStatus.RUNNING, "artifacts": [],
    })
    service.store.save_run(child_run)
    experiment = ExperimentRecord(
        id="exp-derived", name="Derived comparison", address=ExperimentAddress(experiment=99),
        environment_id=child.id, created_at=parent_run.started_at,
        policies=[parent_run.policy_name], perturbations=["normal"], seeds=[parent_run.seed],
        run_ids=[child_run.id], summary=service._summarize([]),
    )
    service.store.save_experiment(experiment)
    service.store.save_agent_message(AgentMessage(
        id="msg-main-environment-link", agent_role="main", speaker="assistant",
        content="Built a circuit.", created_at=parent_run.started_at,
        artifacts=[ArtifactLink(kind="environment", id=environment.id, label=environment.scene.name)],
    ))
    service.store.save_agent_message(AgentMessage(
        id="msg-derived-chat", agent_role="environment", environment_id=child.id,
        speaker="user", content="test the variant", created_at=parent_run.started_at,
    ))

    with pytest.raises(ValueError, match="Active runs cannot be deleted"):
        service.delete_environment(environment.id)
    assert service.store.get_environment(environment.id) is not None
    assert service.store.get_environment(child.id) is not None

    child_run.status = RunStatus.SUCCEEDED
    service.store.save_run(child_run)
    result = service.delete_environment(environment.id)
    assert set(result["deleted_environment_ids"]) == {environment.id, child.id}
    assert set(result["deleted_run_ids"]) == {parent_run.id, child_run.id}
    assert service.store.get_environment(environment.id) is None
    assert service.store.get_environment(child.id) is None
    assert service.store.get_environment(unrelated.id) is not None
    assert service.store.get_experiment(experiment.id) is None
    assert service.agent_messages("environment", child.id) == []
    main_messages = service.agent_messages("main")
    assert not any(
        artifact.kind == "environment" and artifact.id == environment.id
        for message in main_messages for artifact in message.artifacts
    )

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import sys
import uuid
import os
import time
from collections import defaultdict
from datetime import UTC, datetime
from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterator

from .artifacts import ArtifactStore, FileArtifactStore
from .execution import LocalExecutor, RolloutDag, RolloutExecutor, RolloutNode, SlurmExecutor, measure_usage
from .interventions import ForkIntervention, resolve_fork_condition
from .models import Action, AgentAction, AgentMessage, ArtifactLink, ControllerWrite, CoordinatorDispatch, ElevationSpec, EnvironmentAddress, EnvironmentRecord, ExperimentAddress, ExperimentRecord, ExperimentRequest, ExecutionState, ForkRequest, PlayerAddress, RunRecord, RunRequest, RunStatus, StudyPanelConfiguration, StudyPanelDashboard, StudyPanelUpdateRequest, TrackDrawing, TrackDrawingCreate
from .panels import DEFAULT_PANEL_IDS, PanelContext, catalog as panel_catalog, evaluate as evaluate_panels, validate_panel_ids
from .policies import (
    LEGACY_POLICY_ALIASES, PlayerPolicy, PolicySessionError, built_in_policies,
    canonical_policy_name, policy_display_name,
)
from .rendering import ReplayBundle, snapshot_from_frame
from .providers import COORDINATOR_TOOLS, ProviderError, chat_agent_reply, chat_agent_reply_stream
from .racing import (
    CAR_RADIUS, ENGINE_ID, SCENE_BOUNDS, RacingBackend, RacingWorld, compile_certified_scene,
    compile_racing_scene_from_track,
    validate_racing_scene, verify_racing_playability,
)
from .store import HarnessStore


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


VISION_POLICIES_2D = (
    "vision-2d-predictive-skills", "vision-2d-direct", "vision-reflex-sim-rehearsal",
)
VISION_POLICIES_3D = (
    "vision-3d-direct-short", "vision-3d-direct-short-features",
    "vision-3d-predictive-skills", "vision-3d-direct-every-tick",
)
CONTROL_POLICIES = ("oracle-racing-line", "baseline-constant-intent", "baseline-random")
DRAWING_REFERENCE = re.compile(
    r"/(drawing-[a-z0-9][a-z0-9-]{2,63})\b", re.IGNORECASE,
)


def _is_explicit_circuit_request(prompt: str) -> bool:
    """Keep an unambiguous build request from being mistaken for feedback."""
    return bool(re.match(
        r"^\s*(?:(?:okay|please|yes)[,!]?\s*)?(?:let'?s\s+)?"
        r"(?:make|build|generate|create|design|compile)\s+(?:me\s+)?(?:a|an|another|new)\b"
        r"|^\s*give\s+me\s+(?:a|an|another|new)\b"
        r"|^\s*(?:can|could|would|will)\s+we\s+(?:please\s+)?"
        r"(?:make|build|generate|create|design|compile|do)\s+(?:me\s+)?(?:a|an|another|new)\b",
        prompt,
        flags=re.IGNORECASE,
    ))


def _is_designless_build_continuation(prompt: str) -> bool:
    """True only for a short acknowledgement with no new circuit specification."""
    text = " ".join(prompt.lower().split()).strip(".!")
    if text in {"yes", "yeah", "yep", "go ahead", "please do", "do it", "try again"}:
        return True
    return bool(re.fullmatch(
        r"(?:okay[, ]*)?(?:make|build|generate|create)\s+(?:another|a new|new)"
        r"(?:\s+(?:one|circuit|track))?",
        text,
    ))


def _continuation_brief(prompt: str, history: list[dict]) -> str:
    """Resolve a bare confirmation to its one unambiguous preceding build request.

    This is deliberately narrow. General conversation remains model-mediated; only a
    designless "yes" or "make a new one" borrows the last explicit circuit brief.
    """
    if not _is_designless_build_continuation(prompt):
        return prompt
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        candidate = str(turn.get("content") or "").strip()
        if candidate and _is_explicit_circuit_request(candidate) and not _is_designless_build_continuation(candidate):
            return candidate
    return prompt


def default_vision_policy(scene) -> str:
    """Researcher-facing default; telemetry policies are diagnostic controls."""
    return (
        "vision-3d-predictive-skills"
        if scene.elevation and not scene.elevation.is_flat
        else "vision-2d-predictive-skills"
    )


def experiment_policy_choices(scene, message: str, registered: dict[str, Any]) -> list[str]:
    """The researcher-facing experiment surface always uses the predictive player.

    Other policies remain available to code-level evaluations and old replay records, but are
    not choices in the product workflow: changing the driver would confound condition, seed,
    and aggression comparisons.
    """
    del message
    policy = default_vision_policy(scene)
    return [policy] if policy in registered else []


class HarnessService:
    def __init__(self, store: HarnessStore | None = None, runtime: RacingBackend | None = None, artifact_store: ArtifactStore | None = None, executors: dict[str, RolloutExecutor] | None = None) -> None:
        self.store = store or HarnessStore()
        self.policies = built_in_policies()
        self.runtime = runtime or RacingBackend()
        self.artifact_store = artifact_store or FileArtifactStore(self.store.data_dir / "artifacts")
        self.executors = executors or {"local": LocalExecutor(), "slurm": SlurmExecutor()}
        self._view3d_worlds: dict[str, Any] = {}
        """Bounded cache of compiled elevation surfaces, keyed by what is being viewed."""

    def create_drawing(self, request: TrackDrawingCreate) -> TrackDrawing:
        slug = re.sub(r"[^a-z0-9]+", "-", request.name.lower()).strip("-")[:36] or "track"
        drawing = TrackDrawing(
            id=f"drawing-{slug}-{uuid.uuid4().hex[:4]}", name=request.name.strip(),
            points=request.points, created_at=timestamp(),
        )
        self.store.save_drawing(drawing)
        return drawing

    def list_drawings(self) -> list[TrackDrawing]:
        return self.store.list_drawings()

    def delete_drawing(self, drawing_id: str) -> dict[str, str]:
        if not self.store.delete_drawing(drawing_id):
            raise KeyError(f"Drawing not found: {drawing_id}")
        return {"deleted_drawing_id": drawing_id}

    def create_environment_from_drawing(
        self, drawing: TrackDrawing, prompt: str, *, dimensions: str = "2d",
        elevation: ElevationSpec | None = None, address: EnvironmentAddress | None = None,
        study_name: str | None = None,
    ) -> EnvironmentRecord:
        """Compile a saved centerline through the ordinary validation/certification gate."""
        from .track_grammar import compile_drawn_track, parse_track_prompt

        address = address or EnvironmentAddress(
            experiment=self._next_experiment_address().experiment, environment=1, variant=1,
        )
        # The random id suffix is an address, not part of the natural-language
        # options. Strip the `use /...` command before count parsing so an id
        # ending in "49" cannot be mistaken for "49 ... laps".
        options_prompt = DRAWING_REFERENCE.sub("", prompt).strip()
        plan = parse_track_prompt(options_prompt or "asphalt circuit").model_copy(update={
            "title": drawing.name,
            "rationale": f"Centerline compiled from /{drawing.id}.",
        })
        track = compile_drawn_track(drawing.points, SCENE_BOUNDS, CAR_RADIUS, plan.track_width)
        scene = compile_racing_scene_from_track(prompt, plan, track)
        validation = validate_racing_scene(scene)
        if validation != ["Racing domain contract passed."]:
            raise ValueError("Drawing compiler rejected the circuit: " + "; ".join(validation))
        certificate = verify_racing_playability(scene)
        if not certificate.playable:
            raise ValueError("The drawn circuit is valid but the reference driver could not finish it: " + str(certificate.failure))
        if dimensions == "3d":
            from .racing3d import certify_racing_3d_scene
            from .track3d import parse_elevation_prompt

            requested = elevation or parse_elevation_prompt(options_prompt) or ElevationSpec()
            scene, certificate, _ = certify_racing_3d_scene(scene, requested)
        scene = scene.model_copy(update={"id": f"{scene.id}-{uuid.uuid4().hex[:6]}"})
        record = EnvironmentRecord(
            id=scene.id, scene=scene, created_at=timestamp(), validation=validation,
            baseline_solved=self._dry_run(scene.id, scene),
            origin=f"drawing:{drawing.id}", generator_provider="drawing",
            generator_model="drawing-compiler-v1",
            generator_rationale=f"Saved centerline /{drawing.id} compiled and replay-certified.",
            playability_certificate=certificate, study_name=study_name, address=address,
        )
        self.store.save_environment(record)
        return record

    def create_environment(
        self,
        prompt: str,
        seed: int | None = None,
        parent_environment_id: str | None = None,
        origin: str | None = None,
        provider: str = "auto",
        study_name: str | None = None,
        address: EnvironmentAddress | None = None,
        dimensions: str = "2d",
        elevation: ElevationSpec | None = None,
        on_step: Callable[[str, str], None] | None = None,
        intent_prompt: str | None = None,
        conversation: list[dict[str, Any]] | None = None,
    ) -> EnvironmentRecord:
        """Compile a certified circuit from a brief.

        `intent_prompt` is what the *user* asked for, when that differs from the text a caller
        passes in. Everything that decides what the circuit should be reads this: comprehension,
        elevation intent, and fidelity. Grading a scene against a paraphrase measures the wrong
        thing, because it reports the paraphraser's invented details as though the user had
        asked for them.
        """
        address = address or EnvironmentAddress(experiment=self._next_experiment_address().experiment, environment=1, variant=1)
        # A brief that asks for hills is a 3D request whether or not the caller also set the
        # switch. Reading it from the text closes the gap that made "an elevated loop with
        # banked corners" compile a flat circuit and look like the prompt had been discarded.
        from .track3d import parse_elevation_prompt

        intent = intent_prompt or prompt
        spoken = parse_elevation_prompt(intent)
        elevated = dimensions == "3d" or spoken is not None
        if elevated and elevation is None:
            elevation = spoken
        # The circuit is built from the user's own words, not from a paraphrase of them.
        # Comprehension reads `intent` into an explicit list of requirements, the creator is
        # handed that list rather than the prose, and the same list is measured against the
        # compiled scene afterwards. There is no longer a private reading of the brief that
        # only the creator holds, which is what allowed a requirement to be dropped without
        # anything downstream being able to notice.
        from .faithful import generate_faithful

        compile_scene = compile_certified_scene
        if elevated:
            # A 3D scene is the planar scene with a surface fitted to it, so this path
            # certifies twice: once flat, then again over the gradients, which change lap
            # times and cornering enough to make a fine circuit uncompletable.
            from .racing3d import compile_racing_3d_scene

            def compile_scene(scene_prompt, plan, scene_seed):  # noqa: F811
                return compile_racing_3d_scene(scene_prompt, plan, elevation, scene_seed)

        outcome = generate_faithful(
            intent, seed, provider, on_step=on_step, compile_scene=compile_scene,
            precedent_lookup=self.store.precedents_for, conversation=conversation,
            dimensions="3d" if elevated else "2d",
        )
        if outcome.scene is None or outcome.certificate is None or outcome.plan is None:
            raise ValueError(
                "Environment creator could not produce a certified circuit: "
                + (outcome.failure or "no reason was recorded")
            )
        scene, certificate, design = outcome.scene, outcome.certificate, outcome.plan
        if on_step:
            for line in outcome.fidelity_lines():
                on_step("fidelity", line)
        validation = validate_racing_scene(scene)
        # A seed identifies a matched trial, not a globally unique artifact.
        # Paired generator ablations deliberately compile the same prompt and
        # seed more than once, so retain the stable seed prefix while giving
        # every accepted scene its own immutable record id.
        scene = scene.model_copy(update={"id": f"{scene.id}-{uuid.uuid4().hex[:6]}"})
        baseline_solved = self._dry_run(scene.id, scene)
        record = EnvironmentRecord(
            id=scene.id,
            scene=scene,
            created_at=timestamp(),
            validation=validation,
            baseline_solved=baseline_solved,
            parent_environment_id=parent_environment_id,
            origin=origin,
            generator_provider=design.provider,
            generator_model=design.model,
            generator_rationale=design.plan.rationale,
            generator_input_tokens=outcome.input_tokens,
            generator_output_tokens=outcome.output_tokens,
            generator_latency_ms=outcome.latency_ms,
            playability_certificate=certificate,
            prompt_spec=outcome.spec,
            fidelity=outcome.report,
            study_name=study_name,
            address=address,
        )
        self.store.save_environment(record)
        return record

    @staticmethod
    def fidelity_report(prompt: str, scene) -> list[str]:
        """Which parts of a brief the compiled circuit actually carries.

        `generation_spec` already reads a brief into typed assertions and checks them against a
        scene — that is the machinery the generation study uses to grade a creator. Reporting it
        back to whoever asked turns "this looks random" into a list of what landed and what did
        not, using the same measurement rather than a second opinion about it.

        Probe-backed assertions are skipped: they drive whole rollouts, and this runs inline
        while someone is waiting.
        """
        from .generation_spec import _PROBE_KINDS, extract_spec, score

        spec = extract_spec(prompt)
        checkable = [item for item in spec.assertions if item.kind not in _PROBE_KINDS]
        if not checkable:
            return []
        results = score(spec.model_copy(update={"assertions": checkable}), scene, None).results
        honoured = [item.label for item in results if item.satisfied]
        missed = [f"{item.label} (got {item.achieved})" for item in results if not item.satisfied]
        lines = []
        if honoured:
            lines.append("Honoured: " + "; ".join(honoured))
        if missed:
            lines.append("Not achieved: " + "; ".join(missed))
        return lines

    def list_environments(self) -> list[EnvironmentRecord]:
        return [self._certify_existing(record) for record in self.store.list_environments() if record.scene.domain_pack_version == ENGINE_ID]

    def study_panels(self, study_kind: str, study_id: str) -> StudyPanelDashboard:
        configuration = self._panel_configuration(study_kind, study_id)
        runs = self._runs_for_study(study_kind, study_id)
        study_name = self._study_name(study_kind, study_id)
        return StudyPanelDashboard(configuration=configuration, catalog=panel_catalog(), panels=evaluate_panels(configuration.panel_ids, PanelContext(study_name=study_name, runs=runs)))

    def update_study_panels(self, study_kind: str, study_id: str, request: StudyPanelUpdateRequest) -> StudyPanelDashboard:
        self._runs_for_study(study_kind, study_id)  # Validates the target before persisting configuration.
        configuration = StudyPanelConfiguration(study_kind=study_kind, study_id=study_id, panel_ids=validate_panel_ids(request.panel_ids))
        self.store.save_study_panels(configuration)
        return self.study_panels(study_kind, study_id)

    def _panel_configuration(self, study_kind: str, study_id: str) -> StudyPanelConfiguration:
        self._runs_for_study(study_kind, study_id)
        existing = self.store.get_study_panels(study_kind, study_id)
        if existing:
            return existing
        configuration = StudyPanelConfiguration(study_kind=study_kind, study_id=study_id, panel_ids=DEFAULT_PANEL_IDS)
        self.store.save_study_panels(configuration)
        return configuration

    def _runs_for_study(self, study_kind: str, study_id: str) -> list[RunRecord]:
        if study_kind == "comparison":
            record = self.store.get_experiment(study_id)
            if not record:
                raise KeyError(f"Study not found: {study_id}")
            return [self._require_run(run_id) for run_id in record.run_ids]
        raise ValueError("Study kind must be 'comparison'.")

    def _study_name(self, study_kind: str, study_id: str) -> str:
        if study_kind == "comparison":
            record = self.store.get_experiment(study_id)
            if record:
                return record.name
        raise KeyError(f"Study not found: {study_id}")

    def _next_experiment_address(self) -> ExperimentAddress:
        """Allocate an experiment sequence without deriving meaning from names."""
        used = [
            record.address.experiment
            for record in [*self.store.list_environments(), *self.store.list_runs(), *self.store.list_experiments(), *self.store.list_research_studies()]
            if record.address is not None
        ]
        next_number = max(used, default=0) + 1
        if next_number > 999:
            raise ValueError("Experiment address space is exhausted (EXP-999).")
        return ExperimentAddress(experiment=next_number)

    def _next_player_address(self, environment: EnvironmentRecord, parent: RunRecord | None) -> PlayerAddress:
        if environment.address is None:
            raise ValueError("Environment is missing its research address.")
        used = [
            run.address.player
            for run in self.store.list_runs(environment.id)
            if run.address and run.address.prefix.startswith(environment.address.prefix + "-PLAYER-")
        ]
        number = max(used, default=0) + 1
        if number > 999:
            raise ValueError(f"Player address space is exhausted for {environment.address.prefix}.")
        return PlayerAddress(
            experiment=environment.address.experiment,
            environment=environment.address.environment,
            variant=environment.address.variant,
            player=number,
        )

    def get_environment(self, environment_id: str) -> EnvironmentRecord | None:
        record = self.store.get_environment(environment_id)
        return self._certify_existing(record) if record and record.scene.domain_pack_version == ENGINE_ID else None

    def delete_environment(self, environment_id: str) -> dict[str, Any]:
        """Delete a circuit family and all records that cannot exist without it."""
        root = self.store.get_environment(environment_id)
        if root is None:
            raise KeyError(f"Environment not found: {environment_id}")
        all_environments = self.store.list_environments()
        environment_ids = {environment_id}
        changed = True
        while changed:
            changed = False
            for environment in all_environments:
                if environment.id not in environment_ids and environment.parent_environment_id in environment_ids:
                    environment_ids.add(environment.id)
                    changed = True

        run_ids: set[str] = set()
        for run in self.store.list_runs():
            if run.environment_id in environment_ids:
                run_ids.update(self.store.run_tree_ids(run.id))
        run_result = self._delete_run_records(sorted(run_ids), clear_experiment_actions=True)

        for study in self.store.list_research_studies():
            if study.family_id in environment_ids:
                self.store.delete_research_study(study.id)
                continue
            if environment_ids.intersection(study.scenario_ids):
                study.scenario_ids = [item for item in study.scenario_ids if item not in environment_ids]
                if not study.scenario_ids:
                    self.store.delete_research_study(study.id)
                else:
                    self.store.save_research_study(study)

        metadata_experiments = self.store.delete_environments(sorted(environment_ids))
        self._prune_environment_links(environment_ids)
        for deleted_id in environment_ids:
            self._view3d_worlds.pop(f"env:{deleted_id}", None)
        return {
            "deleted_environment_ids": sorted(environment_ids),
            "deleted_run_ids": run_result["deleted_run_ids"],
            "deleted_experiment_ids": sorted(set([
                *run_result["deleted_experiment_ids"], *metadata_experiments,
            ])),
        }

    def _prune_environment_links(self, deleted: set[str]) -> None:
        for message in self.store.list_agent_messages("main", None):
            linked = any(
                artifact.kind == "environment" and artifact.id in deleted
                for artifact in message.artifacts
            ) or any(
                action.artifact and action.artifact.kind == "environment" and action.artifact.id in deleted
                for action in message.actions
            )
            if not linked:
                continue
            message.artifacts = [
                artifact for artifact in message.artifacts
                if not (artifact.kind == "environment" and artifact.id in deleted)
            ]
            message.actions = [
                action for action in message.actions
                if not (
                    action.artifact and action.artifact.kind == "environment"
                    and action.artifact.id in deleted
                )
            ]
            self.store.save_agent_message(message)

    def get_domain_context(self, query: str) -> dict[str, Any]:
        lowered = query.lower()
        return {
            "version": ENGINE_ID,
            "game": "top-down circuit racing",
            "circuit_grammar": ["oval", "technical", "chicane"],
            "surfaces": ["asphalt", "clay", "ice"],
            "scene_elements": [item for item in ("barriers", "opponent cars", "control delay", "low grip", "worn tires", "heavy car", "rear weight bias", "high drag", "high downforce") if item.split()[0] in lowered] or ["barriers", "opponent cars"],
            "dynamics_conditions": ["low_grip", "worn_tires", "heavy_car", "rear_bias", "high_drag", "high_downforce"],
            "constraints": ["one closed certified racing line", "ordered sector checkpoints", "bounded obstacles and NPCs", "serialized transient bicycle dynamics", "deterministic replay verification"],
        }

    def agent_messages(self, role: str, environment_id: str | None = None) -> list[AgentMessage]:
        return self.store.list_agent_messages(role, environment_id)

    def agent_activity(self, limit: int = 40) -> list[AgentMessage]:
        return self.store.list_agent_activity(limit)

    @staticmethod
    def _capture_agent_action(actions: dict[str, AgentAction], event: dict) -> None:
        """Fold transient stream events into the durable action shown after reload."""
        kind = event.get("type")
        if kind not in {"step", "log"}:
            return
        action_id = str(event.get("id") or "work")
        current = actions.get(action_id) or AgentAction(
            id=action_id,
            label=action_id.replace("_", " ").replace(":", " · ").capitalize(),
            state="done" if kind == "log" else event.get("state", "running"),
        )
        if kind == "step":
            artifact = event.get("artifact")
            current = current.model_copy(update={
                "label": str(event.get("label") or current.label),
                "state": event.get("state", current.state),
                "artifact": ArtifactLink.model_validate(artifact) if artifact else current.artifact,
            })
        else:
            detail = f"{event.get('stage', 'detail')}: {event.get('detail', '')}".strip()
            current = current.model_copy(update={"logs": [*current.logs, detail]})
        actions[action_id] = current

    @staticmethod
    def _failed_actions(actions: dict[str, AgentAction]) -> list[AgentAction]:
        return [
            item.model_copy(update={"state": "failed"}) if item.state == "running" else item
            for item in actions.values()
        ]

    def dispatch_experiment_events(self, environment_id: str, message: str) -> Iterator[dict]:
        """Turn one request about a circuit into runs, reporting each as it happens.

        This is the whole experiments flow: a chat scoped to one compiled circuit, which the
        store already persists per environment, plus the ability to act on it. The chat and the
        runs share the environment id, so reopening a circuit later shows what was asked and
        what was launched.
        """
        events: Queue = Queue()

        def work() -> None:
            try:
                for event in self._experiment_flow(environment_id, message, events.put):
                    events.put(event)
            except Exception as error:  # noqa: BLE001 - the stream reports, never crashes
                events.put({"type": "error", "detail": str(error)})
            finally:
                events.put(None)

        Thread(target=work, name="racelab-experiment", daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                return
            yield item

    def _experiment_flow(
        self, environment_id: str, message: str, emit: Callable[[dict], None],
    ) -> Iterator[dict]:
        """Turn one request into runs, and leave a transcript either way.

        Wrapped so the assistant turn is persisted whatever happens. The reply used to be
        written only on the success path, so a provider outage while choosing conditions
        left the user's question sitting in the store with nothing under it — which on the
        next page load is indistinguishable from the conversation having been discarded.
        """
        self._require_environment(environment_id)
        self.store.save_agent_message(AgentMessage(
            id=f"msg-{uuid.uuid4().hex[:8]}", agent_role="environment",
            environment_id=environment_id, speaker="user", content=message,
            created_at=timestamp(),
        ))
        actions: dict[str, AgentAction] = {}
        assistant_message_id = f"msg-{uuid.uuid4().hex[:8]}"
        assistant_created_at = timestamp()

        def report(event: dict) -> None:
            self._capture_agent_action(actions, event)
            emit(event)

        try:
            yield from self._run_experiment_flow(
                environment_id, message, report, actions,
                assistant_message_id, assistant_created_at,
            )
        except Exception as error:  # noqa: BLE001 - reported to the reader, then re-raised
            self.store.save_agent_message(AgentMessage(
                id=assistant_message_id, agent_role="environment",
                environment_id=environment_id, speaker="assistant",
                content=f"That request could not be run: {str(error)[:400]}",
                created_at=assistant_created_at, actions=self._failed_actions(actions),
            ))
            raise

    def _run_experiment_flow(
        self, environment_id: str, message: str, emit: Callable[[dict], None],
        actions: dict[str, AgentAction], assistant_message_id: str,
        assistant_created_at: str,
    ) -> Iterator[dict]:
        from .providers import plan_run_conditions

        environment = self._require_environment(environment_id)
        scene = environment.scene
        emit({"type": "step", "id": "conditions", "label": "Choosing drivers and conditions", "state": "running"})
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
            conditions = None
        else:
            conditions, _ = plan_run_conditions(
                message=message,
                circuit={
                    "name": scene.name, "surface": scene.surface, "grip": scene.grip,
                    "laps": scene.laps, "corridor_px": scene.track_width,
                    "opponents": [item.profile.value for item in scene.npc_behaviors],
                    "dimensions": "3d" if scene.elevation and not scene.elevation.is_flat else "2d",
                },
                policies=experiment_policy_choices(scene, message, self.policies),
            )
        if conditions is None:
            # Without a key there is nothing to interpret the request with, so the offline
            # behaviour is the honest one: run the deterministic oracle and say so.
            plan_text = "No model is configured, so this runs the deterministic racing-line oracle."
            policies, perturbations, max_steps, player_aggression = (
                ["oracle-racing-line"], ["none"], 1_400, .78,
            )
        else:
            plan_text = conditions.plan
            for policy in conditions.policies:
                plan_text = plan_text.replace(policy, policy_display_name(policy))
            policies, perturbations, max_steps, player_aggression = (
                conditions.policies, conditions.perturbations, conditions.max_steps,
                conditions.player_aggression,
            )
        emit({"type": "token", "text": plan_text})
        cells = [(policy, perturbation) for policy in policies for perturbation in perturbations]
        emit({
            "type": "step", "id": "conditions", "state": "done",
            "label": f"{len(cells)} run{'' if len(cells) == 1 else 's'}: "
                     + ", ".join(
                         f"{policy_display_name(policy)}/{condition}"
                         for policy, condition in cells[:6]
                     ),
        })
        experiment_started = time.monotonic()
        emit({
            "type": "progress", "scope": "experiment", "completed": 0,
            "total": len(cells), "elapsed_ms": 0, "eta_ms": None,
            "label": "Preparing first run",
        })
        # Commit the assistant turn before the potentially long player runs. The same id is
        # updated at completion, so leaving the tab or reloading never exposes a user-only
        # transcript and never creates a duplicate assistant response.
        self.store.save_agent_message(AgentMessage(
            id=assistant_message_id, agent_role="environment",
            environment_id=environment_id, speaker="assistant",
            content=f"{plan_text}\n\nRuns are still in progress.",
            created_at=assistant_created_at, actions=list(actions.values()),
        ))

        artifacts: list[ArtifactLink] = []
        for cell_number, (policy, perturbation) in enumerate(cells, start=1):
            key = f"run:{policy}:{perturbation}"
            label = policy_display_name(policy) + (
                "" if perturbation == "none" else f" under {perturbation.replace('_', ' ')}"
            )
            emit({"type": "step", "id": key, "label": f"Driving with {label}", "state": "running"})
            try:
                run = self.run(
                    RunRequest(
                        environment_id=environment_id, policy_name=policy, max_steps=max_steps,
                        player_aggression=player_aggression,
                    ),
                    perturbation=None if perturbation == "none" else perturbation,
                    study_name=environment.study_name,
                )
            except Exception as error:  # noqa: BLE001 - one failed cell must not lose the rest
                emit({
                    "type": "step", "id": key, "state": "failed",
                    "label": f"{label}: {str(error)[:160]}",
                })
            else:
                replay_label = f"{policy_display_name(policy)} replay"
                artifacts.append(ArtifactLink(kind="run", id=run.id, label=replay_label))
                emit({
                    "type": "step", "id": key,
                    "state": "done" if run.status == RunStatus.SUCCEEDED else "failed",
                    "label": f"{label}: {run.result_reason or run.status.value} ({len(run.frames)} ticks)",
                    "artifact": {"kind": "run", "id": run.id, "label": replay_label},
                })
            elapsed_ms = round((time.monotonic() - experiment_started) * 1_000)
            eta_ms = round((elapsed_ms / cell_number) * (len(cells) - cell_number))
            emit({
                "type": "progress", "scope": "experiment", "completed": cell_number,
                "total": len(cells), "elapsed_ms": elapsed_ms, "eta_ms": eta_ms,
                "label": "Experiment complete" if cell_number == len(cells)
                         else f"Run {cell_number + 1} of {len(cells)} is next",
            })

        summary = plan_text + (
            f"\n\nLaunched {len(artifacts)} of {len(cells)} runs." if len(artifacts) != len(cells)
            else f"\n\nLaunched {len(artifacts)} run{'' if len(artifacts) == 1 else 's'}."
        )
        self.store.save_agent_message(AgentMessage(
            id=assistant_message_id, agent_role="environment",
            environment_id=environment_id, speaker="assistant", content=summary,
            created_at=assistant_created_at, artifacts=artifacts, actions=list(actions.values()),
        ))
        yield {
            "type": "done", "content": summary,
            "artifacts": [item.model_dump() for item in artifacts],
            "run_ids": [item.id for item in artifacts],
        }

    def stream_agent_message(
        self, role: str, message: str, environment_id: str | None = None,
    ) -> Iterator[dict]:
        """Stream a chat reply, persisting the finished message exactly as the blocking path does.

        The transcript is the record, so it is written once at the end rather than patched
        per delta: a stream that dies half-way leaves no half-message in the store.
        """
        if role not in {"main", "environment"}:
            raise ValueError("Only main and environment agent chats are available.")
        if role == "environment" and not environment_id:
            raise ValueError("An environment agent needs an environment id.")
        self.store.save_agent_message(AgentMessage(
            id=f"msg-{uuid.uuid4().hex[:8]}", agent_role=role, environment_id=environment_id,
            speaker="user", content=message, created_at=timestamp(),
        ))
        context = None
        artifacts: list[ArtifactLink] = []
        if role == "environment":
            environment = self._require_environment(environment_id)
            certificate = environment.playability_certificate
            context = {
                "id": environment.id, "name": environment.scene.name,
                "objectives": [item.model_dump() for item in environment.scene.objectives],
                "domain_search": self.get_domain_context(message),
                "playability": certificate.model_dump() if certificate else None,
            }
            artifacts.append(ArtifactLink(
                kind="environment", id=environment.id,
                label=f"{environment.address.prefix if environment.address else 'ENVIRONMENT'} · {environment.scene.name}",
            ))
        chunks: list[str] = []
        try:
            for kind, value in chat_agent_reply_stream(
                role=role, message=message, environment_context=context,
            ):
                if kind == "text":
                    chunks.append(value)
                    yield {"type": "token", "text": value}
        except Exception as error:  # noqa: BLE001 - a stream reports rather than 500s mid-body
            # Persisted before returning. A turn that vanishes on reload reads as the whole
            # conversation having been dropped, which is far worse than a recorded failure.
            self.store.save_agent_message(AgentMessage(
                id=f"msg-{uuid.uuid4().hex[:8]}", agent_role=role, environment_id=environment_id,
                speaker="assistant", created_at=timestamp(),
                content="".join(chunks).strip() or f"That reply failed: {str(error)[:300]}",
            ))
            yield {"type": "error", "detail": str(error)}
            return
        reply = "".join(chunks).strip() or "(no reply)"
        self.store.save_agent_message(AgentMessage(
            id=f"msg-{uuid.uuid4().hex[:8]}", agent_role=role, environment_id=environment_id,
            speaker="assistant", content=reply, created_at=timestamp(), artifacts=artifacts,
        ))
        yield {"type": "done", "content": reply, "artifacts": [item.model_dump() for item in artifacts]}

    def send_agent_message(self, role: str, message: str, environment_id: str | None = None) -> list[AgentMessage]:
        if role not in {"main", "environment"}:
            raise ValueError("Only main and environment agent chats are available.")
        if role == "environment" and not environment_id:
            raise ValueError("An environment agent needs an environment id.")
        user_message = AgentMessage(id=f"msg-{uuid.uuid4().hex[:8]}", agent_role=role, environment_id=environment_id, speaker="user", content=message, created_at=timestamp())
        self.store.save_agent_message(user_message)
        if role == "main":
            reply, _ = chat_agent_reply(role="main", message=message)
        else:
            environment = self._require_environment(environment_id)
            context = self.get_domain_context(message)
            certificate = environment.playability_certificate
            reply, _ = chat_agent_reply(
                role="environment", message=message,
                environment_context={
                    "id": environment.id, "name": environment.scene.name, "objectives": [item.model_dump() for item in environment.scene.objectives],
                    "domain_search": context, "playability": certificate.model_dump() if certificate else None,
                },
            )
        artifacts: list[ArtifactLink] = []
        if role == "environment" and environment_id:
            artifacts.append(ArtifactLink(kind="environment", id=environment_id, label=f"{environment.address.prefix if environment.address else 'ENVIRONMENT'} · {environment.scene.name}"))
            latest_run = next(iter(self.list_runs(environment_id)), None)
            if latest_run:
                artifacts.append(ArtifactLink(
                    kind="run", id=latest_run.id,
                    label=(
                        f"{latest_run.address.prefix if latest_run.address else 'PLAYER'} · "
                        f"{policy_display_name(latest_run.policy_name)}"
                    ),
                ))
        assistant_message = AgentMessage(id=f"msg-{uuid.uuid4().hex[:8]}", agent_role=role, environment_id=environment_id, speaker="assistant", content=reply, created_at=timestamp(), artifacts=artifacts)
        self.store.save_agent_message(assistant_message)
        return [user_message, assistant_message]

    def dispatch_coordinator_events(
        self, prompt: str, dimensions: str = "2d", elevation: ElevationSpec | None = None,
    ) -> Iterator[dict]:
        """Run the coordinator flow, yielding progress as it happens.

        The flow is a brief, then bounded creator attempts and certification — minutes of
        work behind one request. Returning only the final summary made the UI look frozen
        and hid where the time actually went, so the same work now reports each stage as it
        starts and finishes.

        The work runs on a thread and pushes onto a queue rather than being rewritten as a
        generator, because environment creation is a deep synchronous call tree with its own
        retry ladders. Threading it keeps one implementation of the flow
        instead of a streaming copy that can drift from the blocking one.
        """
        events: Queue = Queue()

        actions: dict[str, AgentAction] = {}

        def emit(payload: dict) -> None:
            self._capture_agent_action(actions, payload)
            events.put(payload)

        def work() -> None:
            try:
                result = self.dispatch_coordinator(
                    prompt, on_event=emit, dimensions=dimensions, elevation=elevation,
                )
                events.put({"type": "done", "result": result.model_dump(mode="json")})
            except Exception as error:  # noqa: BLE001 - the stream reports, never crashes
                stored = self.store.list_agent_messages("main", None)
                if stored and stored[-1].speaker == "user":
                    self.store.save_agent_message(AgentMessage(
                        id=f"msg-{uuid.uuid4().hex[:8]}", agent_role="main", speaker="assistant",
                        content=f"That request could not be completed: {str(error)[:400]}",
                        created_at=timestamp(), actions=self._failed_actions(actions),
                    ))
                events.put({"type": "error", "detail": str(error)})
            finally:
                events.put(None)

        Thread(target=work, name="racelab-coordinator", daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                return
            yield item

    def dispatch_coordinator(
        self, prompt: str, on_event: Callable[[dict], None] | None = None,
        dimensions: str = "2d", elevation: ElevationSpec | None = None,
    ) -> CoordinatorDispatch:
        """One coordinator turn: a reply, and a circuit only if one was actually asked for.

        The circuit used to be unconditional — every message compiled a track and raced a
        driver on it, so "hey what's up" produced a racetrack. Whether to build is the
        model's decision now, carried by a tool call in the same streamed response as the
        reply. Same call, so the decision cannot disagree with the prose the user just read.
        """
        downstream = on_event or (lambda payload: None)
        actions: dict[str, AgentAction] = {}

        def emit(payload: dict) -> None:
            self._capture_agent_action(actions, payload)
            downstream(payload)
        # Loaded before the user's turn is saved, so the model sees the conversation so far
        # without seeing the message it is about to answer twice. Without this the
        # coordinator was a fresh single-shot call every turn: it could not follow up, and
        # a reply like "keep going?" reached a model with no idea what it was continuing.
        history = self._coordinator_history()
        self.store.save_agent_message(AgentMessage(id=f"msg-{uuid.uuid4().hex[:8]}", agent_role="main", speaker="user", content=prompt, created_at=timestamp()))
        drawing_match = DRAWING_REFERENCE.search(prompt)
        drawing = self.store.get_drawing(drawing_match.group(1).lower()) if drawing_match else None
        if drawing_match and drawing is None:
            message = (
                f"I couldn't find saved drawing /{drawing_match.group(1)}. It may have been deleted "
                "or the reference was mistyped. Open Draw and use its Use button to insert a valid reference."
            )
            self.store.save_agent_message(AgentMessage(
                id=f"msg-{uuid.uuid4().hex[:8]}", agent_role="main", speaker="assistant",
                content=message, created_at=timestamp(), actions=self._failed_actions(actions),
            ))
            raise KeyError(message)
        assistant_message_id = f"msg-{uuid.uuid4().hex[:8]}"
        assistant_created_at = timestamp()
        # The coordinator's reply is the one part of this flow the user reads, so it
        # streams. Everything after it is work, and reports as steps instead.
        chunks: list[str] = []
        # With no model there is nobody to make the judgement, so the offline path keeps
        # its old contract and treats every dispatch as a build request. Deciding not to
        # build would be a decision nothing actually made.
        build_prompt = _continuation_brief(prompt, history)
        explicit_build = drawing is not None or _is_explicit_circuit_request(prompt) or _is_designless_build_continuation(prompt)
        requested = explicit_build or not (
            os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        )
        context = {
            "recent_build": self._recent_build(),
            "drawing_reference": f"/{drawing.id}" if drawing else None,
        }
        recorded: list[dict] = []
        if drawing:
            drawing_reply = (
                f"Using /{drawing.id} as the circuit centerline. I’ll smooth it, compile it, "
                "and replay-certify the resulting track now."
            )
            chunks.append(drawing_reply)
            emit({"type": "token", "text": drawing_reply})
        reply_stream = () if drawing else chat_agent_reply_stream(
            role="main", message=prompt, history=history, tools=COORDINATOR_TOOLS,
            environment_context=context, dimensions=dimensions,
        )
        for kind, value in reply_stream:
            if kind == "text":
                chunks.append(value)
                emit({"type": "token", "text": value})
            elif kind == "tool":
                call = json.loads(value)
                name, arguments = call.get("name"), call.get("input", {})
                if name == "build_circuit":
                    requested = True
                    reason = str(arguments.get("reason") or "").strip()
                    if reason:
                        emit({"type": "log", "id": "environment", "stage": "requested", "detail": reason})
                elif name == "record_feedback":
                    if explicit_build:
                        # This is a fresh build request, not a verdict on the previous
                        # circuit, even when its requirement ids are in short-term context.
                        continue
                    saved = self._record_feedback(arguments.get("confirmations") or [])
                    recorded.append({"call": call, "saved": saved})
                    emit({"type": "log", "id": "feedback", "stage": "learned", "detail": (
                        f"Recorded {saved} confirmed requirement reading(s) for future circuits."
                        if saved else "Noted, but nothing reusable to keep from that."
                    )})

        if recorded and not requested:
            # Recording resolves instantly, so the tool loop is closed properly and the
            # person gets an actual reply. Without the second leg a model that went straight
            # to the tool — which is exactly what it does when the whole message is feedback
            # — answered with silence.
            chunks.extend(self._close_feedback_turn(
                prompt, history, recorded, context, emit, dimensions=dimensions,
            ))
        reply = "".join(chunks).strip()

        if not requested:
            # A conversational turn. Nothing is compiled, nothing is raced, and the reply is
            # persisted so the transcript survives a reload like any other turn.
            self._save_coordinator_reply(
                reply, [], [], list(actions.values()),
                message_id=assistant_message_id, created_at=assistant_created_at,
            )
            return CoordinatorDispatch(summary=reply or "Talked it through.")

        # The prose response is complete before compilation begins. Persist it now,
        # then update this same message with the finished artifacts below.
        self._save_coordinator_reply(
            reply, [], [], list(actions.values()),
            message_id=assistant_message_id, created_at=assistant_created_at,
        )

        experiment_address = self._next_experiment_address()
        environment: EnvironmentRecord | None = None
        errors: list[str] = []
        # The circuit is built from the user's own words. A bare "yes" or "make a new
        # one" borrows one prior explicit brief; every other turn stays literal. The
        # coordinator's reply is
        # conversation for the person reading it, not a specification: comprehension reads
        # `prompt` into requirement ids and everything downstream measures against those,
        # so routing the request through a rewrite could only lose the exact phrases the
        # contract keys on ("ice", "90 degree", "top right", "hilly").
        #
        # Two of the stages report things the user asked about directly, so they are collected
        # here and shown in the reply rather than left folded inside the work log: what the
        # engine cannot express, and what it tried for and missed. Everything else stays a log.
        notes: list[str] = []

        def note(stage: str, detail: str) -> None:
            emit({"type": "log", "id": "environment", "stage": stage, "detail": detail})
            if stage == "unsupported" or detail.startswith("Not achieved"):
                if detail not in notes:
                    notes.append(detail)

        for attempt in range(1 if drawing else 2):
            emit({
                "type": "step", "id": "environment", "state": "running",
                "label": (
                    f"Compiling and certifying /{drawing.id}"
                    if drawing else
                    f"Circuit creator compiling and certifying (proposal {attempt + 1} of 2)"
                ),
            })
            try:
                if drawing:
                    environment = self.create_environment_from_drawing(
                        drawing, prompt, dimensions=dimensions, elevation=elevation,
                        study_name=f"Coordinator environment · /{drawing.id}",
                        address=EnvironmentAddress(
                            experiment=experiment_address.experiment, environment=1, variant=1,
                        ),
                    )
                else:
                    environment = self.create_environment(
                        build_prompt, provider="auto", study_name=f"Coordinator environment · {build_prompt[:42]}",
                        address=EnvironmentAddress(experiment=experiment_address.experiment, environment=1, variant=1),
                        dimensions=dimensions, elevation=elevation, intent_prompt=build_prompt,
                        on_step=note,
                        conversation=[*history, {"role": "user", "content": prompt}],
                    )
                break
            except ValueError as error:
                errors.append(str(error))
                emit({"type": "log", "id": "environment", "stage": "rejected", "detail": str(error)[:300]})
        if environment is None:
            emit({"type": "step", "id": "environment", "label": "No certified circuit", "state": "failed"})
            if drawing:
                detail = errors[-1] if errors else "The drawing compiler returned no detail."
                failure_reply = (
                    f"I couldn't compile /{drawing.id} into a playable circuit: {detail} "
                    "Try redrawing the congested portion with a wider turn or more space between nearby track sections."
                )
                emit({"type": "token", "text": "\n\n" + failure_reply})
                reply = (reply + "\n\n" + failure_reply).strip()
            # Persisted before raising, or a failed dispatch leaves the user's question in the
            # transcript with no answer under it — which on reload looks exactly like the
            # conversation was thrown away.
            self._save_coordinator_reply(
                reply, notes, [], self._failed_actions(actions),
                message_id=assistant_message_id, created_at=assistant_created_at,
            )
            if drawing:
                raise ValueError(f"Could not compile /{drawing.id}: " + " | ".join(errors))
            raise ValueError("Environment agent could not produce a certified scene after two bounded proposals: " + " | ".join(errors))
        emit({
            "type": "step", "id": "environment", "state": "done",
            "label": (
                f"Certified {environment.scene.name} — {environment.scene.surface} at "
                f"{environment.scene.grip:.2f}× grip, {environment.scene.laps} lap(s), "
                f"{len(environment.scene.npc_behaviors)} opponent(s)"
                # Labelled from the compiled scene, not the request: a brief that asked for
                # hills without the switch is still 3D, and saying otherwise is how "it ignored
                # my prompt" looked true even when it had not.
                + (", 3D" if environment.scene.elevation and not environment.scene.elevation.is_flat else ", 2D")
            ),
            "artifact": {"kind": "environment", "id": environment.id, "label": environment.scene.name},
        })
        if notes:
            emit({"type": "token", "text": "\n\n" + "\n".join(f"- {line}" for line in notes)})
        # The environment generator is itself the agent action. Record the tool
        # result without creating a second, unrelated chat-model call.
        self.store.save_agent_message(AgentMessage(
            id=f"msg-{uuid.uuid4().hex[:8]}", agent_role="environment", environment_id=environment.id, speaker="assistant",
            content=(
                f"Compiled from /{drawing.id} and certified with " if drawing else
                "Compiled from the request and certified with "
            ) + f"{' → '.join(environment.playability_certificate.objective_trace if environment.playability_certificate else [])}.",
            created_at=timestamp(), artifacts=[ArtifactLink(kind="environment", id=environment.id, label=f"{environment.address.prefix} · {environment.scene.name}")],
            actions=[AgentAction(
                id="environment", label=f"Certified {environment.scene.name}", state="done",
                artifact=ArtifactLink(kind="environment", id=environment.id, label=f"{environment.address.prefix} · {environment.scene.name}"),
            )],
        ))
        certificate = environment.playability_certificate
        assert certificate is not None
        summary = (
            f"Created {environment.address.prefix} · {environment.scene.name}; verifier trace: {' → '.join(certificate.objective_trace)}. "
            "Open Experiments to launch the predictive-skills player."
        )
        # What gets persisted is what the reader saw: the coordinator's actual reply plus
        # the fidelity notes appended under it, carrying the environment link. Storing the
        # internal summary instead meant reopening the tab replaced the conversation with
        # machine prose and dropped every link to the circuit it had just made.
        self._save_coordinator_reply(reply, notes, [
            ArtifactLink(kind="environment", id=environment.id, label=f"{environment.address.prefix} · {environment.scene.name}"),
        ], list(actions.values()), message_id=assistant_message_id, created_at=assistant_created_at)
        return CoordinatorDispatch(environment_id=environment.id, environment_name=environment.scene.name, certificate=certificate, summary=summary)

    def _save_coordinator_reply(
        self, reply: str, notes: list[str], artifacts: list[ArtifactLink],
        actions: list[AgentAction] | None = None,
        *, message_id: str | None = None, created_at: str | None = None,
    ) -> None:
        """Persist the coordinator turn exactly as the reader saw it stream."""
        content = reply + (("\n\n" + "\n".join(f"- {line}" for line in notes)) if notes else "")
        self.store.save_agent_message(AgentMessage(
            id=message_id or f"msg-{uuid.uuid4().hex[:8]}", agent_role="main", speaker="assistant",
            content=content.strip() or "Worked on that request.", created_at=created_at or timestamp(),
            artifacts=artifacts, actions=actions or [],
        ))

    def _close_feedback_turn(
        self, prompt: str, history: list[dict], recorded: list[dict],
        context: dict, emit: Callable[[dict], None], *, dimensions: str = "2d",
    ) -> list[str]:
        """Second leg of the tool loop: hand back the results and stream the real reply."""
        assistant_blocks = [{
            "type": "tool_use", "id": item["call"].get("id", ""),
            "name": item["call"].get("name", ""), "input": item["call"].get("input", {}),
        } for item in recorded]
        results = [{
            "type": "tool_result", "tool_use_id": item["call"].get("id", ""),
            "content": (
                f"Recorded. {item['saved']} confirmed reading(s) stored for future circuits."
                if item["saved"] else
                "Recorded. Nothing reusable to store from that one — either it was a "
                "correction rather than a confirmation, or the harness had already measured "
                "it as missed."
            ),
        } for item in recorded]
        chunks: list[str] = []
        try:
            for kind, value in chat_agent_reply_stream(
                role="main", message=results, environment_context=context,
                history=[*history, {"role": "user", "content": prompt},
                         {"role": "assistant", "content": assistant_blocks}],
                tools=COORDINATOR_TOOLS, dimensions=dimensions,
            ):
                if kind == "text":
                    chunks.append(value)
                    emit({"type": "token", "text": value})
        except ProviderError as error:
            # The feedback is already stored, so a failure here costs a sentence, not data.
            emit({"type": "log", "id": "feedback", "stage": "reply-failed", "detail": str(error)[:200]})
        return chunks

    def _latest_environment(self) -> EnvironmentRecord | None:
        """The circuit most recently built, which is the one feedback would be about."""
        records = [
            record for record in self.store.list_environments()
            if record.scene.domain_pack_version == ENGINE_ID and record.prompt_spec
        ]
        return max(records, key=lambda item: item.created_at, default=None)

    def _recent_build(self) -> dict | None:
        """Requirement ids for the last circuit, so an offhand comment can be attributed.

        Only the ids and statements, and only when a build has happened at all. The whole
        point of keeping precedents out of context is undone if the contract itself becomes
        permanently resident.
        """
        record = self._latest_environment()
        if record is None or record.prompt_spec is None:
            return None
        verdicts = {item.id: item for item in (record.fidelity.verdicts if record.fidelity else [])}
        return {
            "environment_id": record.id,
            "name": record.scene.name,
            "requirements": [
                f"{item.id} {item.statement}"
                + (" [measured as delivered]" if verdicts.get(item.id) and verdicts[item.id].satisfied
                   else " [the harness already knows it missed this]")
                for item in record.prompt_spec.requirements
            ],
        }

    def _record_feedback(self, confirmations: list[dict]) -> int:
        """Store the user's verdict on how their words were read, as precedents.

        Admission needs both gates, and the second one is applied here: `distil` keeps only
        requirements the simulator measured as satisfied *and* the person confirmed. A
        rejection is not stored as a precedent — there is nothing reusable in a reading that
        turned out to be wrong — but it still stops that pairing being learned as good.
        """
        from .precedents import distil

        record = self._latest_environment()
        if record is None or record.prompt_spec is None or record.fidelity is None:
            return 0
        approved = {
            str(item.get("id", "")).strip().upper(): bool(item.get("satisfied"))
            for item in confirmations if isinstance(item, dict) and item.get("id")
        }
        notes = {
            str(item.get("id", "")).strip().upper(): str(item.get("note") or "")
            for item in confirmations if isinstance(item, dict) and item.get("id")
        }
        found = distil(
            record.prompt_spec, record.scene, record.fidelity.verdicts,
            approved, notes, now=timestamp(),
        )
        for precedent in found:
            self.store.save_precedent(precedent)
        return len(found)

    def _coordinator_history(self, turns: int = 12) -> list[dict]:
        """The recent coordinator conversation, as provider messages.

        Bounded, and trimmed to start on a user turn: the API rejects a leading assistant
        message, and a window that happens to begin mid-exchange is how a bounded history
        turns into an intermittent 400 rather than a shorter memory.
        """
        stored = self.store.list_agent_messages("main", None)[-turns:]
        while stored and stored[0].speaker != "user":
            stored.pop(0)
        return [
            {"role": "user" if item.speaker == "user" else "assistant", "content": item.content}
            for item in stored if item.content.strip()
        ]

    def run(self, request: RunRequest, perturbation: str | None = None, parent: RunRecord | None = None, fork_step: int | None = None, study_name: str | None = None, address: PlayerAddress | None = None, guidance: str | None = None, intervention: dict[str, Any] | None = None) -> RunRecord:
        environment = self._require_environment(request.environment_id)
        executor = self._require_executor(request.execution_backend)
        policy_name = canonical_policy_name(request.policy_name)
        policy = self._require_policy(policy_name)
        supports_aggression = hasattr(policy, "set_aggression")
        if supports_aggression:
            policy.set_aggression(request.player_aggression)
        fork_supported = not hasattr(policy, "run_episode")
        guidance_supported = fork_supported and hasattr(policy, "set_episode_guidance")
        if guidance and not guidance_supported:
            raise ValueError(
                f"{policy_name} cannot apply operator correction guidance; choose a model policy that reports guidance_supported"
            )
        record = RunRecord(
            id=f"run-{uuid.uuid4().hex[:8]}",
            environment_id=environment.id,
            environment_version=environment.scene.version,
            policy_name=policy_name,
            seed=environment.scene.seed,
            status=RunStatus.PENDING,
            started_at=timestamp(),
            parent_run_id=parent.id if parent else None,
            fork_step=fork_step,
            perturbation=intervention or ({"kind": perturbation} if perturbation else None),
            study_name=study_name or (parent.study_name if parent else environment.study_name),
            address=address or self._next_player_address(environment, parent),
            fork_supported=fork_supported,
            guidance_supported=guidance_supported,
            # This is the assigned experimental treatment, not a claim that every
            # policy consumes it.  Recording it for fixed controls makes a matrix
            # auditable and prevents those rows from losing their pace axis.
            player_aggression=request.player_aggression,
        )
        record.execution = executor.submit(run_id=record.id, resources=request.resources)
        # Cluster work is represented by a durable control-plane record. A Slurm
        # worker can claim this run and call execute_queued_run on a compute node.
        if record.execution.backend != "local":
            self.store.save_run(record)
            return record
        record.status = RunStatus.RUNNING
        started = time.monotonic()
        world = self.runtime.create(environment.scene, perturbation)
        action_delay = perturbation == "action_delay"
        if parent and fork_step is not None:
            self._restore_parent_prefix(world, parent, fork_step)
            record.frames = parent.frames[:fork_step]
        max_steps = request.max_steps
        if hasattr(policy, "run_episode"):
            # The reflex driver owns its own episode loop, because a decision is a tool
            # conversation that installs a controller rather than one action for this tick.
            # It produces the same frames, so everything downstream — the replay artifact,
            # the desktop viewer, the run tree — is unchanged.
            if parent is not None or fork_step is not None:
                raise ValueError(
                    "the reflex driver cannot resume a forked prefix yet; fork a per-tick policy instead"
                )
            return self._run_reflex_episode(record, environment, policy, world, max_steps, started)
        realtime_clock = getattr(policy, "realtime_clock", None)
        if realtime_clock:
            # Predictive feedback policies must keep the simulator on its control
            # clock while the provider request runs on a worker. Sending them through
            # the ordinary act/step loop would silently freeze the race on every call.
            return self._run_realtime_policy(
                record, environment, policy, world, request, started,
                clock=realtime_clock, action_delay=action_delay, guidance=guidance,
            )
        if hasattr(policy, "configure_episode"):
            policy.configure_episode(max_steps, request.policy_decision_budget)
        policy.reset(environment.scene, environment.scene.seed)
        if guidance and hasattr(policy, "set_episode_guidance"):
            policy.set_episode_guidance(guidance)
        policy_failure: str | None = None
        active_steps = sum(
            1 for frame in record.frames
            if frame.privileged_state.countdown_ticks_remaining == 0
        )
        while not world.terminated and active_steps < max_steps:
            if world.countdown_ticks_remaining > 0:
                frame = world.step(Action())
                record.frames.append(frame)
                continue
            try:
                observation = world.observe()
                if hasattr(policy, "act_visual") and hasattr(world, "render_policy_frame"):
                    frame = policy.render_frame(world) if hasattr(policy, "render_frame") else world.render_policy_frame()
                    action, decision = policy.act_visual(observation, frame)
                else:
                    action, decision = policy.act(observation)
            except (PolicySessionError, ProviderError) as error:
                policy_failure = str(error)
                break
            frame = world.step(action, decision, action_delay=action_delay)
            record.frames.append(frame)
            record.total_reward += frame.reward
            active_steps += 1
        if policy_failure:
            record.status = RunStatus.FAILED
            record.result_reason = policy_failure
        else:
            record.status = RunStatus.SUCCEEDED if world.succeeded else RunStatus.TIMEOUT if not world.terminated else RunStatus.FAILED
            record.result_reason = world.reason or "step budget exhausted"
        record.completed_at = timestamp()
        record.total_reward = round(record.total_reward, 4)
        record.latency_ms = len(record.frames) * 4
        if hasattr(policy, "turn_usages"):
            provider_policy = policy
            record.token_usage = provider_policy.input_tokens + provider_policy.output_tokens
            record.input_tokens = provider_policy.input_tokens
            record.output_tokens = provider_policy.output_tokens
            record.uncached_input_tokens = provider_policy.uncached_input_tokens
            record.cache_creation_input_tokens = provider_policy.cache_creation_input_tokens
            record.cache_read_input_tokens = provider_policy.cache_read_input_tokens
            record.player_turns = len(provider_policy.turn_usages or [])
            record.output_token_budget = provider_policy.output_token_budget
            record.latency_ms += provider_policy.latency_ms
        return self._finalize_run(record, environment, started)

    def _run_realtime_policy(
        self, record: RunRecord, environment: EnvironmentRecord, policy: Any,
        world: Any, request: RunRequest, started: float, *, clock: str,
        action_delay: bool, guidance: str | None,
    ) -> RunRecord:
        """Run a feedback player without ever putting model latency on the tick thread."""
        from .realtime import run_realtime_episode

        prefix_active_steps = sum(
            frame.privileged_state.countdown_ticks_remaining == 0
            for frame in record.frames
        )
        remaining_steps = max(0, request.max_steps - prefix_active_steps)
        result = run_realtime_episode(
            world, policy, max_steps=remaining_steps, clock=clock,
            latency_ticks=int(getattr(policy, "initial_latency_ticks", 12)),
            decision_budget=request.policy_decision_budget,
            action_delay=action_delay, guidance=guidance,
        )
        record.frames.extend(result["frames"])
        record.total_reward = round(sum(frame.reward for frame in record.frames), 4)
        record.status = (
            RunStatus.SUCCEEDED if result["succeeded"]
            else RunStatus.FAILED if result["terminated"] or result["policy_failure"]
            else RunStatus.TIMEOUT
        )
        record.result_reason = result["reason"]
        record.completed_at = timestamp()
        record.realtime_metrics = {
            "clock": result["clock"],
            "control_hz": result["control_hz"],
            **result["realtime"],
            **result.get("policy_realtime", {}),
        }
        record.latency_ms = round((time.monotonic() - started) * 1_000)
        if hasattr(policy, "turn_usages"):
            record.token_usage = policy.input_tokens + policy.output_tokens
            record.input_tokens = policy.input_tokens
            record.output_tokens = policy.output_tokens
            record.uncached_input_tokens = policy.uncached_input_tokens
            record.cache_creation_input_tokens = policy.cache_creation_input_tokens
            record.cache_read_input_tokens = policy.cache_read_input_tokens
            record.player_turns = len(policy.turn_usages or [])
            record.output_token_budget = policy.output_token_budget
        return self._finalize_run(record, environment, started)

    def _finalize_run(self, record: RunRecord, environment: EnvironmentRecord, started: float) -> RunRecord:
        """Stamp execution accounting, store the replay artifact, and persist the run.

        Shared by the per-tick loop and the reflex loop so a reflex run is a first-class
        run: same artifact kind, same store, so `open-native-viewer` and the run tree work
        without knowing which loop produced it.
        """
        # The engine terminates on the exact final-gate tick. Keep this invariant at the
        # persistence boundary too: every policy path (ordinary, realtime, and reflex)
        # funnels through here, so an adapter cannot retain trailing frames or label an
        # incomplete lap sequence successful.
        expected_objectives = len(environment.scene.objectives)
        completed_at_frame = next((
            index for index, frame in enumerate(record.frames)
            if frame.privileged_state.objective_index == expected_objectives
            and frame.privileged_state.lap == environment.scene.laps
        ), None)
        if completed_at_frame is not None:
            if completed_at_frame + 1 < len(record.frames):
                record.frames = record.frames[:completed_at_frame + 1]
                record.total_reward = round(sum(frame.reward for frame in record.frames), 4)
            record.status = RunStatus.SUCCEEDED
            if not record.result_reason or "completed" not in record.result_reason:
                record.result_reason = f"{environment.scene.laps}-lap race completed"
        elif record.status == RunStatus.SUCCEEDED:
            record.status = RunStatus.FAILED
            record.result_reason = (
                "Invalid completion: run was marked successful before all "
                f"{environment.scene.laps} laps and ordered checkpoints were completed"
            )

        record.execution.state = ExecutionState.COMPLETED
        record.execution.resource_usage = measure_usage(started)
        record.execution.worker_id = record.execution.resource_usage.host
        replay_payload = ReplayBundle.from_run(environment.scene, record).model_dump_json(indent=None).encode()
        record.artifacts.append(self.artifact_store.put_bytes(kind="replay", key=record.id, payload=replay_payload, media_type="application/vnd.racelab.replay+json"))
        self.store.save_run(record)
        return record

    def _run_reflex_episode(
        self, record: RunRecord, environment: EnvironmentRecord, policy: Any, world: Any,
        max_steps: int, started: float,
    ) -> RunRecord:
        """Drive an episode where the model writes the controller instead of the controls."""
        from .models import DecisionRecord

        report = policy.run_episode(world, max_steps=max_steps)
        record.frames = report.frames
        record.total_reward = round(sum(frame.reward for frame in report.frames), 4)
        record.status = (
            RunStatus.SUCCEEDED if report.succeeded
            else RunStatus.FAILED if report.terminated else RunStatus.TIMEOUT
        )
        record.result_reason = report.reason or "step budget exhausted"
        record.completed_at = timestamp()
        record.token_usage = report.usage.input_tokens + report.usage.output_tokens
        record.input_tokens = report.usage.input_tokens
        record.output_tokens = report.usage.output_tokens
        record.cache_read_input_tokens = report.usage.cache_read_input_tokens
        record.cache_creation_input_tokens = report.usage.cache_creation_input_tokens
        record.uncached_input_tokens = max(
            0, report.usage.input_tokens - report.usage.cache_read_input_tokens
            - report.usage.cache_creation_input_tokens,
        )
        # `player_turns` is the provider-call count for ordinary player policies.
        # Keep that unit here as well: a reflex wake can contain several dependent
        # tool-loop round trips, and reporting only wakes would undercount the model
        # work in a direct-versus-reflex comparison.
        record.player_turns = report.usage.calls
        record.latency_ms = len(report.frames) * 4 + report.usage.latency_ms
        # The UI reads per-frame decision telemetry. Only wake ticks carry a model decision,
        # so label every frame with the controller that actually drove it and mark the wakes.
        # Frame indices include the countdown; the runtime's tick counter does not, because
        # nothing is controlled while the grid is frozen. Without the offset every wake would
        # be labelled about thirty frames early.
        countdown_frames = sum(
            1 for frame in report.frames if frame.privileged_state.countdown_ticks_remaining > 0
        )
        record.controller_writes = self._controller_writes(
            report.turns, countdown_frames,
            end_frame_step=report.frames[-1].step if report.frames else countdown_frames,
        )
        wake_ticks = {turn.tick + countdown_frames: turn for turn in report.turns}
        summary = (
            f"{report.wakes} wakes, {report.usage.calls} model calls. Best rehearsed lap: "
            f"{report.runtime_summary.get('best_rehearsed_lap_ticks')} ticks."
        )
        for index, frame in enumerate(record.frames, start=1):
            turn = wake_ticks.get(index)
            frame.decision = DecisionRecord(
                action=frame.action,
                subgoal=(
                    f"woke: {', '.join(turn.causes) or 'first decision'}" if turn
                    else "agent-written controller"
                ),
                confidence=1.0 if turn else 0.6,
                summary=(" ".join(turn.said)[:600] or summary) if turn else summary,
                candidates=[frame.action],
            )
        if report.diagnostics_payload:
            payload = json.dumps(report.diagnostics_payload, separators=(",", ":"), default=str).encode()
            record.artifacts.append(self.artifact_store.put_bytes(
                kind="reflex-diagnostics", key=record.id, payload=payload,
                media_type="application/vnd.racelab.reflex-diagnostics+json",
            ))
        return self._finalize_run(record, environment, started)

    def compile_experiment_dag(self, request: ExperimentRequest) -> RolloutDag:
        """Compile a rollout matrix into an executor-neutral dependency DAG."""
        from .providers import PERTURBATIONS

        unsupported = sorted(set(request.perturbations) - {"normal", *PERTURBATIONS})
        if unsupported:
            raise ValueError(f"Unsupported experiment condition: {', '.join(unsupported)}")
        experiment_id = f"dag-{uuid.uuid4().hex[:8]}"
        nodes: list[RolloutNode] = []
        aggressions = list(dict.fromkeys(request.player_aggressions or [request.player_aggression]))
        for policy in map(canonical_policy_name, request.policies):
            for perturbation in request.perturbations:
                for seed in request.seeds:
                    for aggression in aggressions:
                        cell = f"{policy}-{perturbation}-{seed}-a{aggression:g}".replace("_", "-")
                        nodes.append(RolloutNode(id=f"{experiment_id}-{cell}", run_id=f"planned-{cell}", resources=RunRequest(environment_id=request.environment_id).resources))
        return RolloutDag(id=experiment_id, nodes=tuple(nodes))

    def fork_run(self, run_id: str, request: ForkRequest) -> RunRecord:
        parent = self._require_run(run_id)
        if request.fork_step >= len(parent.frames):
            raise ValueError("Fork step must select a recorded frame.")
        policy = self._require_policy(parent.policy_name)
        if hasattr(policy, "run_episode"):
            raise ValueError("the reflex driver cannot resume a forked prefix yet; fork a per-tick policy instead")
        if request.condition is not None:
            resolved = resolve_fork_condition(request.condition)
        else:
            resolved = ForkIntervention(
                request.perturbation or "none", request.guidance,
                "Structured replay-fork intervention.",
            )
        guidance = request.guidance or resolved.guidance
        if guidance and not hasattr(policy, "set_episode_guidance"):
            raise ValueError(f"{parent.policy_name} can replay from a tick but cannot apply correction guidance")
        # Preserve the historical compact record for API clients that still send the
        # structured fields. Natural-language forks retain the original sentence and
        # its deterministic interpretation so the intervention is auditable.
        intervention = None if request.condition is None else {
            "kind": resolved.perturbation,
            "condition": request.condition,
            "summary": resolved.summary,
        }
        if intervention is not None and guidance:
            intervention["guidance"] = guidance
        return self.run(
            RunRequest(
                environment_id=parent.environment_id, policy_name=parent.policy_name,
                max_steps=max(len(parent.frames) + 280, 280),
                player_aggression=parent.player_aggression if parent.player_aggression is not None else .78,
            ),
            perturbation=None if resolved.perturbation == "none" else resolved.perturbation,
            parent=parent,
            fork_step=request.fork_step,
            guidance=guidance,
            intervention=intervention,
        )

    def list_runs(self, environment_id: str | None = None) -> list[RunRecord]:
        return [self._enrich_run(record) for record in self.store.list_runs(environment_id)]

    def delete_run(self, run_id: str) -> dict[str, Any]:
        """Delete one replay and every fork that depends on its recorded prefix."""
        run_ids = self.store.run_tree_ids(run_id)
        if not run_ids:
            raise KeyError(f"Run not found: {run_id}")
        return self._delete_run_records(run_ids, clear_experiment_actions=False)

    def delete_experiment(self, experiment_key: str, environment_id: str | None = None) -> dict[str, Any]:
        """Delete an experiment's runs without deleting its reusable circuit variants.

        Addressed experiments are global: a comparison may contain seed variants that are
        not visible under the currently selected circuit. Legacy groups predate addresses,
        so those remain scoped to the selected circuit and their recorded study name.
        """
        all_runs = self.store.list_runs()
        if experiment_key.startswith("experiment-"):
            try:
                number = int(experiment_key.removeprefix("experiment-"))
            except ValueError as error:
                raise ValueError(f"Invalid experiment key: {experiment_key}") from error
            roots = [run for run in all_runs if run.address and run.address.experiment == number]
        elif experiment_key.startswith("legacy-"):
            if not environment_id:
                raise ValueError("A circuit id is required to delete a legacy experiment.")
            legacy_name = experiment_key.removeprefix("legacy-")
            roots = [
                run for run in all_runs
                if run.environment_id == environment_id
                and not run.address
                and (run.study_name or "ad-hoc") == legacy_name
            ]
        else:
            record = self.store.get_experiment(experiment_key)
            roots = [run for run in all_runs if record and run.id in record.run_ids]
        if not roots:
            raise KeyError(f"Experiment not found: {experiment_key}")
        run_ids = set()
        for run in roots:
            run_ids.update(self.store.run_tree_ids(run.id))
        result = self._delete_run_records(sorted(run_ids), clear_experiment_actions=True)
        result["experiment_key"] = experiment_key
        return result

    def _delete_run_records(self, run_ids: list[str], *, clear_experiment_actions: bool) -> dict[str, Any]:
        deleted = set(run_ids)
        records = [record for record in self.store.list_runs() if record.id in deleted]
        active = [record.id for record in records if record.status in {RunStatus.PENDING, RunStatus.RUNNING}]
        if active:
            raise ValueError(
                "Active runs cannot be deleted. Wait for them to finish: " + ", ".join(active)
            )
        self.store.delete_runs(run_ids)

        deleted_experiments: list[str] = []
        for experiment in self.store.list_experiments():
            if not deleted.intersection(experiment.run_ids):
                continue
            remaining_ids = [run_id for run_id in experiment.run_ids if run_id not in deleted]
            if not remaining_ids or clear_experiment_actions:
                self.store.delete_experiment(experiment.id)
                deleted_experiments.append(experiment.id)
                continue
            rows = [
                row for row in experiment.summary.get("rows", [])
                if row.get("run_id") not in deleted
            ]
            experiment.run_ids = remaining_ids
            experiment.summary = self._summarize(rows)
            self.store.save_experiment(experiment)

        for study in self.store.list_research_studies():
            if not deleted.intersection(study.run_ids):
                continue
            study.run_ids = [run_id for run_id in study.run_ids if run_id not in deleted]
            study.fingerprints = [item for item in study.fingerprints if item.run_id not in deleted]
            study.interventions = [
                item for item in study.interventions
                if item.baseline_run_id not in deleted and item.counterfactual_run_id not in deleted
            ]
            self.store.save_research_study(study)

        self._prune_run_links(deleted, clear_experiment_actions=clear_experiment_actions)
        for record in records:
            self._view3d_worlds.pop(f"run:{record.id}", None)
            (self.store.data_dir / "replays" / f"{record.id}.json").unlink(missing_ok=True)
            for artifact in record.artifacts:
                try:
                    self.artifact_store.delete(artifact)
                except (OSError, RuntimeError, ValueError):
                    # Metadata deletion must still complete when a remote artifact store is
                    # temporarily unavailable. Its immutable orphan can be garbage-collected.
                    pass
        return {
            "deleted_run_ids": sorted(deleted),
            "deleted_experiment_ids": deleted_experiments,
        }

    def _prune_run_links(self, deleted: set[str], *, clear_experiment_actions: bool) -> None:
        for role, environment_id in [
            ("main", None),
            *[("environment", environment.id) for environment in self.store.list_environments()],
        ]:
            for message in self.store.list_agent_messages(role, environment_id):
                linked = {
                    artifact.id for artifact in message.artifacts
                    if artifact.kind == "run" and artifact.id in deleted
                }
                linked.update(
                    action.artifact.id for action in message.actions
                    if action.artifact and action.artifact.kind == "run" and action.artifact.id in deleted
                )
                if not linked:
                    continue
                message.artifacts = [
                    artifact for artifact in message.artifacts
                    if not (artifact.kind == "run" and artifact.id in deleted)
                ]
                if clear_experiment_actions:
                    message.actions = [action for action in message.actions if not action.id.startswith("run:")]
                else:
                    message.actions = [
                        action for action in message.actions
                        if not (action.artifact and action.artifact.kind == "run" and action.artifact.id in deleted)
                    ]
                self.store.save_agent_message(message)

    def get_run(self, run_id: str) -> RunRecord | None:
        record = self.store.get_run(run_id)
        return self._enrich_run(record) if record else None

    def _enrich_run(self, record: RunRecord) -> RunRecord:
        """Attach current policy capabilities and backfill old reflex diagnostics."""
        try:
            policy = self._require_policy(record.policy_name)
        except KeyError:
            policy = None
        fork_supported = policy is not None and not hasattr(policy, "run_episode")
        guidance_supported = fork_supported and hasattr(policy, "set_episode_guidance")
        controller_writes = record.controller_writes
        if not controller_writes:
            reference = next((item for item in record.artifacts if item.kind == "reflex-diagnostics"), None)
            if reference is not None:
                try:
                    payload = json.loads(self.artifact_store.get_bytes(reference))
                    countdown = sum(
                        1 for frame in record.frames
                        if frame.privileged_state.countdown_ticks_remaining > 0
                    )
                    controller_writes = self._controller_writes(
                        payload.get("turns", []), countdown,
                        end_frame_step=record.frames[-1].step if record.frames else countdown,
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    controller_writes = []
        return record.model_copy(update={
            "fork_supported": fork_supported,
            "guidance_supported": guidance_supported,
            "controller_writes": controller_writes,
        })

    @staticmethod
    def _controller_writes(
        turns: list, countdown_frames: int, end_frame_step: int | None = None,
    ) -> list[ControllerWrite]:
        """Map paused authorship turns onto the replay clock and live control intervals.

        A turn happens after control tick ``N`` has completed, so a controller activated in
        that paused turn can first affect ``N + 1``. Multiple activations in one turn cost no
        simulator ticks; only the final selected version can therefore become effective.
        """
        writes: list[ControllerWrite] = []
        current_label: str | None = None
        current_write: ControllerWrite | None = None
        for wake, turn in enumerate(turns, start=1):
            tick = int(turn.get("tick", 0) if isinstance(turn, dict) else turn.tick)
            calls = turn.get("tool_calls", []) if isinstance(turn, dict) else turn.tool_calls
            frame_step = max(0, tick + countdown_frames)
            prior_label = current_label
            for call in calls:
                name = call.get("name")
                result = call.get("result") or {}
                if name == "activate_controller" and result.get("activated"):
                    current_label = str(result.get("controller") or "") or current_label
                    continue
                if name != "install_controller":
                    continue
                arguments = call.get("input") or {}
                gate = result.get("gate") or {}
                write = ControllerWrite(
                    tick=max(0, tick), frame_step=frame_step, wake=wake,
                    name=str(arguments.get("name") or "unnamed"),
                    label=str(result.get("controller")) if result.get("controller") else None,
                    source=str(arguments.get("source") or ""),
                    reads=[str(item) for item in arguments.get("reads") or []],
                    params=dict(arguments.get("params") or {}),
                    installed=bool(result.get("installed")),
                    errors=[str(item) for item in gate.get("errors") or []],
                )
                writes.append(write)
                if result.get("installed") and result.get("active") and write.label:
                    current_label = write.label

            if current_label == prior_label:
                continue
            if current_write is not None and current_write.effective_from_frame_step is not None:
                # The previous version produced the already-recorded state at frame_step;
                # the newly selected version begins on the following simulator step.
                if current_write.effective_from_frame_step <= frame_step:
                    current_write.effective_until_frame_step = frame_step
                    current_write.active = True
                else:
                    current_write.effective_from_frame_step = None
            current_write = next(
                (item for item in reversed(writes) if item.label == current_label), None,
            )
            if current_write is not None:
                current_write.effective_from_frame_step = frame_step + 1

        if (
            current_write is not None
            and current_write.effective_from_frame_step is not None
            and end_frame_step is not None
        ):
            if current_write.effective_from_frame_step <= end_frame_step:
                current_write.effective_until_frame_step = end_frame_step
                current_write.active = True
            else:
                current_write.effective_from_frame_step = None
        return writes

    def render_environment_view3d(
        self, environment_id: str, camera: str, width: int, height: int,
        yaw: float = 45.0, pitch: float = 35.0, distance: float = 900.0, focus: str = "circuit",
    ) -> bytes:
        """A perspective frame of a 3D circuit, from a preset rig or a free orbit camera.

        `camera="free"` is what the environment tab uses: two angles and a radius around the
        circuit, which is the question an inspection view has to answer and no car-relative
        rig can, since a driving camera only ever looks where the car is pointed.
        """
        environment = self._require_environment(environment_id)
        world = self._elevated_world(environment.scene, f"env:{environment_id}")
        if camera == "free":
            return self._encode_free_view(world, width, height, yaw, pitch, distance, focus)
        return self._encode_view(world, camera, width, height)

    def render_run_view3d(
        self, run_id: str, step: int, camera: str, width: int, height: int,
    ) -> bytes:
        """A perspective frame of one recorded tick of a 3D run.

        The world is restored rather than re-simulated, so scrubbing to any tick costs one
        restore and one render — and `Racing3DWorld.restore` recomputes vertical pose from
        planar position, so the car sits on the road it recorded.
        """
        run = self._require_run(run_id)
        environment = self._require_environment(run.environment_id)
        world = self._elevated_world(environment.scene, f"run:{run_id}")
        if run.frames:
            index = max(0, min(step, len(run.frames) - 1))
            world.restore(snapshot_from_frame(run.frames[index]))
        return self._encode_view(world, camera, width, height)

    def _elevated_world(self, scene, cache_key: str):
        """A 3D world for this scene, kept between requests.

        Compiling the elevation surface is the expensive part and it is a pure function of the
        scene, so rebuilding it per frame would make scrubbing unusable. The cache is bounded
        because a long session would otherwise hold a surface for every run ever opened.
        """
        if scene.elevation is None or scene.elevation.is_flat:
            raise ValueError(
                "This scene is planar; the top-down view is the whole picture. "
                "Create an environment with dimensions=3d for a perspective view."
            )
        cached = self._view3d_worlds.get(cache_key)
        if cached is None:
            from .racing3d import Racing3DWorld

            cached = Racing3DWorld.from_scene(scene)
            if len(self._view3d_worlds) >= 8:
                self._view3d_worlds.pop(next(iter(self._view3d_worlds)))
            self._view3d_worlds[cache_key] = cached
        return cached

    @staticmethod
    def _encode_free_view(
        world, width: int, height: int, yaw: float, pitch: float, distance: float, focus: str,
    ) -> bytes:
        import io

        import pygame

        from .view3d import DEFAULT_ROAD_DETAIL, ensure_headless_video, orbit_camera, render_pose_surface

        ensure_headless_video()
        pose = orbit_camera(
            world, yaw_degrees=yaw, pitch_degrees=pitch, distance_pixels=distance,
            focus="car" if focus == "car" else "circuit",
        )
        # The far side of a circuit is exactly what an orbit camera is looking at, so the
        # driving-camera cull distance would erase most of the picture.
        surface = render_pose_surface(
            world, pose, max(160, min(1_280, width)), max(120, min(960, height)),
            DEFAULT_ROAD_DETAIL, draw_distance=pose.distance_behind_pixels + 2_400.0,
        )
        buffer = io.BytesIO()
        pygame.image.save(surface, buffer, "view3d.png")
        return buffer.getvalue()

    @staticmethod
    def _encode_view(world, camera: str, width: int, height: int) -> bytes:
        from .view3d import ViewMode, encode_view_png

        try:
            mode = ViewMode(camera)
        except ValueError:
            raise ValueError(
                f"Unknown camera {camera!r}; available: {[item.value for item in ViewMode]}"
            ) from None
        return encode_view_png(
            world, mode, max(160, min(1_280, width)), max(120, min(960, height)),
        )

    def get_replay_bundle(self, run_id: str) -> ReplayBundle:
        run = self._require_run(run_id)
        environment = self._require_environment(run.environment_id)
        return ReplayBundle.from_run(environment.scene, run)

    def export_replay_bundle(self, run_id: str) -> str:
        bundle = self.get_replay_bundle(run_id)
        target = self.store.data_dir / "replays" / f"{run_id}.json"
        return str(bundle.write_json(target.resolve()))

    def launch_native_viewer(self, run_id: str) -> dict[str, str]:
        """Export a portable bundle and open it in the separate desktop process."""
        path = self.export_replay_bundle(run_id)
        viewer_env = os.environ.copy()
        viewer_env.pop("SDL_VIDEODRIVER", None)
        subprocess.Popen(
            [sys.executable, "-m", "harness.native_viewer", "--bundle", path],
            start_new_session=True,
            env=viewer_env,
        )
        return {"status": "launched", "bundle_path": path, "renderer": "native-pygame-2d"}

    def run_experiment(self, request: ExperimentRequest) -> ExperimentRecord:
        from .providers import PERTURBATIONS

        source_environment = self._require_environment(request.environment_id)
        if source_environment.parent_environment_id:
            raise ValueError(
                "Experiment matrices must start from the source circuit, not one of its seed variants."
            )
        policies = list(dict.fromkeys(map(canonical_policy_name, request.policies)))
        perturbations = list(dict.fromkeys(request.perturbations))
        unsupported = sorted(set(perturbations) - {"normal", *PERTURBATIONS})
        if unsupported:
            raise ValueError(f"Unsupported experiment condition: {', '.join(unsupported)}")
        seeds = list(dict.fromkeys(request.seeds))
        aggressions = list(dict.fromkeys(request.player_aggressions or [request.player_aggression]))
        experiment_address = self._next_experiment_address()
        study_name = request.name or f"Study · {source_environment.scene.name}"
        # The source scene and derived seed variants remain visibly grouped by
        # the same named study in the environment and replay workspaces.
        if source_environment.study_name != study_name:
            source_environment.study_name = study_name
            source_environment.address = EnvironmentAddress(experiment=experiment_address.experiment, environment=1, variant=1)
            self.store.save_environment(source_environment)
        environments_by_seed: dict[int, EnvironmentRecord] = {}
        for variant_number, seed in enumerate(seeds, start=1):
            if source_environment.scene.seed == seed:
                environments_by_seed[seed] = source_environment
            else:
                environments_by_seed[seed] = self.create_environment(
                    source_environment.scene.prompt,
                    seed,
                    parent_environment_id=source_environment.id,
                    origin=f"experiment seed {seed}",
                    provider="offline",
                    study_name=study_name,
                    address=EnvironmentAddress(experiment=experiment_address.experiment, environment=1, variant=variant_number),
                )
        run_ids: list[str] = []
        rows: list[dict[str, Any]] = []
        for policy_number, policy in enumerate(policies, start=1):
            for perturbation_number, perturbation in enumerate(perturbations, start=1):
                for variant_number, seed in enumerate(seeds, start=1):
                    for aggression_number, aggression in enumerate(aggressions, start=1):
                        experiment_environment = environments_by_seed[seed]
                        run = self.run(
                            RunRequest(
                                environment_id=experiment_environment.id, policy_name=policy,
                                max_steps=request.max_steps, player_aggression=aggression,
                            ),
                            None if perturbation == "normal" else perturbation,
                            study_name=study_name,
                            address=PlayerAddress(
                                experiment=experiment_address.experiment, environment=1,
                                variant=experiment_environment.address.variant,
                                player=(((policy_number - 1) * len(perturbations) + perturbation_number - 1)
                                        * len(aggressions) + aggression_number),
                            ),
                        )
                        run_ids.append(run.id)
                        rows.append(
                            {
                                "policy": policy,
                                "perturbation": perturbation,
                                "player_aggression": aggression,
                                "environment_id": experiment_environment.id,
                                "run_id": run.id,
                                "status": run.status.value,
                                "steps": len(run.frames),
                                "reward": run.total_reward,
                                "tokens": run.token_usage,
                            }
                        )
        record = ExperimentRecord(
            id=f"exp-{uuid.uuid4().hex[:8]}",
            name=study_name,
            address=experiment_address,
            environment_id=request.environment_id,
            created_at=timestamp(),
            policies=policies,
            perturbations=perturbations,
            seeds=seeds,
            player_aggressions=aggressions,
            run_ids=run_ids,
            summary=self._summarize(rows),
        )
        self.store.save_experiment(record)
        return record

    def list_experiments(self, environment_id: str | None = None) -> list[ExperimentRecord]:
        return self.store.list_experiments(environment_id)

    def _dry_run(self, environment_id: str, scene) -> bool:
        world = self.runtime.create(scene)
        # This is a route-completion sanity check, not a competitive benchmark.
        # User-launched runs keep the default first-finisher termination.
        world.terminate_on_opponent_win = False
        policy = self._require_policy("oracle-racing-line")
        policy.reset(scene, scene.seed)
        for _ in range(1_400 * scene.laps):
            if world.terminated:
                break
            action, decision = policy.act(world.observe())
            world.step(action, decision)
        return world.succeeded

    def _restore_parent_prefix(self, world: RacingWorld, parent: RunRecord, fork_step: int) -> None:
        if fork_step == 0:
            return
        world.restore(snapshot_from_frame(parent.frames[fork_step - 1]))

    @staticmethod
    def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["policy"]].append(row)
            by_condition[row["perturbation"]].append(row)

        def aggregate(items: list[dict[str, Any]]) -> dict[str, float | int]:
            successes = [item for item in items if item["status"] == RunStatus.SUCCEEDED.value]
            return {
                "count": len(items),
                "success_rate": round(len(successes) / len(items), 3) if items else 0,
                "mean_steps": round(statistics.mean(item["steps"] for item in items), 1) if items else 0,
                "mean_reward": round(statistics.mean(item["reward"] for item in items), 3) if items else 0,
            }

        return {
            "rows": rows,
            "by_policy": {name: aggregate(items) for name, items in grouped.items()},
            "by_perturbation": {name: aggregate(items) for name, items in by_condition.items()},
            "total_runs": len(rows),
        }

    def _require_environment(self, environment_id: str) -> EnvironmentRecord:
        environment = self.store.get_environment(environment_id)
        if not environment:
            raise KeyError(f"Unknown environment: {environment_id}")
        if environment.scene.domain_pack_version != ENGINE_ID:
            raise KeyError(f"Environment {environment_id} belongs to a retired game domain")
        return environment

    def _certify_existing(self, record: EnvironmentRecord) -> EnvironmentRecord:
        if record.playability_certificate is None:
            # An elevated scene has to be certified over its gradients: a circuit the
            # oracle finishes flat can be uncompletable once it climbs.
            if record.scene.elevation is not None and not record.scene.elevation.is_flat:
                from .racing3d import verify_racing_3d_playability

                record.playability_certificate = verify_racing_3d_playability(record.scene)
            else:
                record.playability_certificate = verify_racing_playability(record.scene)
            self.store.save_environment(record)
        return record

    def _require_run(self, run_id: str) -> RunRecord:
        run = self.store.get_run(run_id)
        if not run:
            raise KeyError(f"Unknown run: {run_id}")
        return run

    def _require_policy(self, name: str) -> PlayerPolicy:
        canonical = canonical_policy_name(name)
        try:
            return self.policies[canonical]
        except KeyError as error:
            aliases = ", ".join(sorted(LEGACY_POLICY_ALIASES))
            raise KeyError(
                f"Unknown policy: {name}. Available: {', '.join(self.policies)}. "
                f"Legacy aliases also accepted: {aliases}"
            ) from error

    def _require_executor(self, name: str) -> RolloutExecutor:
        try:
            return self.executors[name]
        except KeyError as error:
            raise KeyError(f"Unknown execution backend: {name}. Available: {', '.join(self.executors)}") from error

    def execution_backends(self) -> list[dict[str, str | bool]]:
        return [
            {"id": executor.id, "kind": "control-plane handoff" if executor.id != "local" else "in-process worker", "available": True, "artifact_store": self.artifact_store.id}
            for executor in self.executors.values()
        ]

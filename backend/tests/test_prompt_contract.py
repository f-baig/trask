"""The contract pipeline: comprehension, verification, repair, and local solving.

These run offline. The model-backed reading of a brief cannot be asserted without a
key, but everything downstream of it can: that the vocabulary and the evaluator
registry agree, that a requirement is settled by measurement rather than assertion,
that a satisfied requirement cannot be traded away by the dial solver, and that a
brief which is not honoured is reported as not honoured.
"""

from __future__ import annotations

import pytest

from harness.comprehension import CHECK_VOCABULARY, comprehend, needs_probes
from harness.fidelity import repair_brief, scene_facts, verify
from harness.generation_spec import _EVALUATORS, _PROBE_KINDS
from harness.prompt_spec import PromptSpec, Requirement, RequirementCheck
from harness.racing import compile_certified_scene
from harness.track_grammar import parse_track_prompt

EASY = "a curvy circuit with two aggressive npcs"


def _scene(prompt: str = EASY, seed: int = 3):
    scene, certificate, _notes = compile_certified_scene(
        prompt, parse_track_prompt(prompt), seed,
    )
    return scene, certificate


def _spec(*checks: tuple[str, object]) -> PromptSpec:
    return PromptSpec(prompt=EASY, requirements=[
        Requirement(
            id=f"R{index + 1}", category="dynamics", statement=f"requirement {index + 1}",
            quote="the user said this", checks=[RequirementCheck(kind=kind, target=target)],
        )
        for index, (kind, target) in enumerate(checks)
    ])


def test_the_check_vocabulary_and_the_evaluator_registry_never_drift() -> None:
    """The model's output vocabulary IS the set of things the engine can settle.

    A registered evaluator with no vocabulary entry is capability the reader can never
    ask for, which is how eleven of them sat unused. A vocabulary entry with no
    evaluator is a check that can be requested and never measured.
    """
    assert set(CHECK_VOCABULARY) == set(_EVALUATORS)


def test_every_probe_backed_kind_is_declared_as_one() -> None:
    """A probe-backed check verified without a rollout must not read as satisfied."""
    scene, _certificate = _scene()
    spec = _spec(("oracle_finishes", True))
    assert needs_probes(spec)
    report = verify(spec, scene, probes=None, judge=False)
    verdict = report.verdicts[0]
    assert not verdict.satisfied
    assert verdict.method == "unverifiable"
    assert "rollout" in verdict.evidence


def test_a_requirement_holds_only_when_every_one_of_its_checks_holds() -> None:
    scene, _certificate = _scene()
    both = PromptSpec(prompt=EASY, requirements=[Requirement(
        id="R1", category="entity", statement="two aggressive opponents", quote="two aggressive npcs",
        checks=[
            RequirementCheck(kind="npc_count", target=2),
            RequirementCheck(kind="npc_profiles", target=["aggressor", "backmarker"]),
        ],
    )])
    report = verify(both, scene, probes=None, judge=False)
    assert not report.verdicts[0].satisfied, "one failing check must fail the requirement"
    assert not report.faithful


def test_verification_reports_the_users_own_words_back() -> None:
    """A miss is only actionable if it says what was asked, not what we inferred."""
    scene, _certificate = _scene()
    spec = _spec(("surface", "ice"))
    report = verify(spec, scene, probes=None, judge=False)
    assert not report.faithful
    assert any("the user said this" in line for line in report.lines())
    brief = repair_brief(spec, report)
    assert "R1" in brief and "the user asked" in brief


def test_a_repair_brief_names_what_must_keep_holding() -> None:
    """Repair is targeted. Without the held list, fixing one clause breaks two."""
    scene, _certificate = _scene()
    spec = _spec(("surface", "ice"), ("npc_count", 2))
    report = verify(spec, scene, probes=None, judge=False)
    brief = repair_brief(spec, report)
    assert "R2" in brief.split("must still hold in your revision:")[1]


def test_a_faithful_scene_produces_no_repair_brief() -> None:
    scene, _certificate = _scene()
    spec = _spec(("surface", "asphalt"), ("npc_count", 2))
    report = verify(spec, scene, probes=None, judge=False)
    assert report.faithful
    assert repair_brief(spec, report) == ""


def test_soft_preferences_do_not_break_faithfulness() -> None:
    scene, _certificate = _scene()
    spec = PromptSpec(prompt=EASY, requirements=[Requirement(
        id="R1", category="dynamics", statement="ideally icy", quote="ideally icy",
        priority="should", checks=[RequirementCheck(kind="surface", target="ice")],
    )])
    report = verify(spec, scene, probes=None, judge=False)
    assert not report.verdicts[0].satisfied
    assert report.faithful, "a missed soft preference is not an unfaithful circuit"


def test_the_judge_never_sees_the_creators_own_description() -> None:
    """The rationale is the claim under test, so it cannot be part of the evidence."""
    scene, _certificate = _scene()
    facts = scene_facts(scene)
    flat = repr(facts)
    assert scene.name not in flat
    assert "rationale" not in facts and "title" not in facts
    assert facts["surface"] == scene.surface
    assert facts["corner_count"] == len(scene.track_report.corners)


def test_the_dial_solver_never_trades_a_satisfied_requirement_away() -> None:
    """Grip must move to satisfy R1 without abandoning the width R2 already holds."""
    from harness.dials import solve

    prompt = "a circuit"
    plan = parse_track_prompt(prompt)
    scene, certificate, _notes = compile_certified_scene(prompt, plan, 5)
    spec = PromptSpec(prompt=prompt, requirements=[
        Requirement(id="R1", category="dynamics", statement="very low grip", quote="slippery",
                    checks=[RequirementCheck(kind="grip_max", target=0.5)]),
        Requirement(id="R2", category="layout", statement="a standard corridor", quote="normal",
                    checks=[RequirementCheck(kind="track_width_min", target=130.0)]),
    ])

    def revalidate(candidate, candidate_probes):
        return verify(spec, candidate, candidate_probes, judge=False)

    before = revalidate(scene, None)
    assert not before.faithful
    outcome = solve(spec, plan, scene, certificate, None, before, revalidate, seed=5)
    assert outcome.report.satisfied > before.satisfied, "the solver must close the grip gap"
    assert outcome.scene.grip <= 0.5
    assert outcome.scene.track_width >= 130.0, "R2 must survive the solve"


def test_the_dial_solver_declines_to_move_what_it_cannot_fix() -> None:
    """A failing requirement with no dial behind it must not perturb the scene."""
    from harness.dials import solve

    prompt = "a circuit"
    plan = parse_track_prompt(prompt)
    scene, certificate, _notes = compile_certified_scene(prompt, plan, 5)
    spec = _spec(("laps", 4))

    def revalidate(candidate, candidate_probes):
        return verify(spec, candidate, candidate_probes, judge=False)

    report = revalidate(scene, None)
    outcome = solve(spec, plan, scene, certificate, None, report, revalidate, seed=5)
    assert outcome.variants_tried == 0
    assert outcome.scene is scene


def test_offline_comprehension_still_produces_a_contract() -> None:
    """No key still yields requirement ids, so the pipeline runs in CI."""
    spec, usage = comprehend("an icy circuit with three laps", provider="offline")
    assert usage is None
    kinds = {check.kind for item in spec.requirements for check in item.checks}
    assert {"surface", "laps"} <= kinds
    assert all(item.id.startswith("R") for item in spec.requirements)


def test_openai_key_selects_model_backed_environment_comprehension(monkeypatch) -> None:
    """OpenAI setup must not silently fall through to the offline track grammar."""
    from harness import comprehension as comprehension_module
    from harness.providers import ProviderUsage

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(
        comprehension_module, "anthropic_json",
        lambda **kwargs: (
            calls.append(kwargs) or {
                "summary": "one icy circuit", "requirements": [],
                "unspecified": [], "unsupported": [],
            },
            ProviderUsage(provider="openai", model="gpt-5.6-luna"),
        ),
    )
    _spec, usage = comprehend("an icy circuit", provider="auto")
    assert calls and calls[0]["model"].startswith("gpt-")
    assert usage is not None and usage.provider == "openai"


def test_3d_comprehension_does_not_reject_elevation_as_a_flat_engine_feature(monkeypatch) -> None:
    """The reader must see the same dimensional capability as the eventual compiler."""
    from harness import comprehension as comprehension_module
    from harness.providers import ProviderUsage

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    calls: list[dict] = []
    monkeypatch.setattr(
        comprehension_module, "anthropic_json",
        lambda **kwargs: (
            calls.append(kwargs) or {
                "summary": "an elevated circuit", "requirements": [],
                "unspecified": [], "unsupported": [],
            },
            ProviderUsage(provider="openai", model="gpt-5.6-luna"),
        ),
    )
    comprehend("a 3d circuit with high elevation", dimensions="3d")
    system = calls[0]["system"]
    assert "3D CAPABILITY" in system
    assert "Treat them as requirements, not unsupported requests" in system
    assert "does NOT by itself request an elevation profile" in system
    assert "2D CAPABILITY" not in system


def test_comprehension_recognizes_a_literal_no_corner_circle(monkeypatch) -> None:
    """A circle is a dedicated zero-corner TrackPlan primitive, not an approximation."""
    from harness import comprehension as comprehension_module
    from harness.providers import ProviderUsage

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    calls: list[dict] = []
    monkeypatch.setattr(
        comprehension_module, "anthropic_json",
        lambda **kwargs: (
            calls.append(kwargs) or {
                "summary": "round loop", "requirements": [], "unspecified": [], "unsupported": [],
            },
            ProviderUsage(provider="openai", model="gpt-5.6-luna"),
        ),
    )
    comprehend("a circle with no corners", dimensions="3d")
    assert "`loop_shape=circle` requirement" in calls[0]["system"]


def test_high_elevation_language_is_described_as_alpine_to_the_brief_reader() -> None:
    """The reader's vocabulary matches the deterministic 3D elevation parser."""
    assert "high/large elevation differentials" in CHECK_VOCABULARY["elevation_profile"]


def test_unsupported_asks_are_shown_to_the_creator_not_hidden() -> None:
    """The creator must be told what not to chase, or it burns every retry on it."""
    spec = PromptSpec(
        prompt="a figure-eight", requirements=[],
        unsupported=["figure-eight, crossing, or self-intersecting layout"],
    )
    briefing = spec.briefing()
    assert "cannot do" in briefing
    assert "figure-eight" in briefing


@pytest.mark.parametrize("kind", sorted(_PROBE_KINDS))
def test_every_probe_kind_is_in_the_vocabulary(kind: str) -> None:
    assert kind in CHECK_VOCABULARY


# --- coordinator conversation state -------------------------------------------
#
# The coordinator lost its transcript on every refresh and every tab switch. Two
# separate causes, both worth pinning: it had no conversation memory at all, and what
# it persisted was its internal brief rather than the reply the reader actually saw.


def test_the_coordinator_remembers_the_conversation(tmp_path) -> None:
    """A follow-up like "keep going?" is meaningless without the turns before it."""
    import uuid

    from harness.models import AgentMessage
    from harness.service import HarnessService, timestamp
    from harness.store import HarnessStore

    service = HarnessService(store=HarnessStore(tmp_path))
    for speaker, content in (
        ("user", "an icy track please"),
        ("assistant", "Ice it is, with low grip."),
        ("user", "keep going?"),
    ):
        service.store.save_agent_message(AgentMessage(
            id=f"msg-{uuid.uuid4().hex[:8]}", agent_role="main", speaker=speaker,
            content=content, created_at=timestamp(),
        ))

    history = service._coordinator_history()
    assert [item["role"] for item in history] == ["user", "assistant", "user"]
    assert history[0]["content"] == "an icy track please"


def test_coordinator_history_never_starts_on_an_assistant_turn(tmp_path) -> None:
    """A window that opens mid-exchange is a provider 400, not a shorter memory."""
    import uuid

    from harness.models import AgentMessage
    from harness.service import HarnessService, timestamp
    from harness.store import HarnessStore

    service = HarnessService(store=HarnessStore(tmp_path))
    for index in range(6):
        for speaker in ("user", "assistant"):
            service.store.save_agent_message(AgentMessage(
                id=f"msg-{index}-{speaker}-{uuid.uuid4().hex[:6]}", agent_role="main",
                speaker=speaker, content=f"{speaker} {index}", created_at=f"2026-01-01T00:{index:02d}:0{0 if speaker == 'user' else 1}",
            ))
    history = service._coordinator_history(turns=5)
    assert history, "a bounded window must still carry something"
    assert history[0]["role"] == "user"


def test_the_coordinator_persists_the_reply_the_reader_saw(tmp_path) -> None:
    """Not the internal summary, and with the artifact links attached.

    Storing machine prose instead meant reopening the tab replaced the conversation and
    dropped every link to the circuit that had just been made.
    """
    from harness.models import AgentAction, ArtifactLink
    from harness.service import HarnessService
    from harness.store import HarnessStore

    service = HarnessService(store=HarnessStore(tmp_path))
    service._save_coordinator_reply(
        "Ice with three aggressive rivals, coming up.",
        ["Not achieved: R2 a hairpin bottom left"],
        [ArtifactLink(kind="environment", id="env-1", label="E1 · Frost Loop")],
        [AgentAction(id="environment", label="Certified Frost Loop", state="done")],
    )
    stored = service.store.list_agent_messages("main", None)
    assert len(stored) == 1
    assert stored[0].content.startswith("Ice with three aggressive rivals")
    assert "Not achieved: R2" in stored[0].content
    assert [item.id for item in stored[0].artifacts] == ["env-1"]
    assert stored[0].actions[0].label == "Certified Frost Loop"


def test_a_coordinator_reply_is_never_stored_empty(tmp_path) -> None:
    """An empty turn reads as a dropped conversation, which is the bug being fixed."""
    from harness.service import HarnessService
    from harness.store import HarnessStore

    service = HarnessService(store=HarnessStore(tmp_path))
    service._save_coordinator_reply("   ", [], [])
    assert service.store.list_agent_messages("main", None)[0].content.strip()


def test_a_greeting_does_not_build_a_racetrack(tmp_path, monkeypatch) -> None:
    """Whether to act is the model's call, carried by a tool call, not the control flow.

    Compiling a circuit takes minutes and runs a whole race. It used to happen on every
    coordinator message regardless of what was said, so "hey what's up" produced a
    racetrack — a expensive answer to a question nobody asked.
    """
    from harness import service as service_module
    from harness.service import HarnessService
    from harness.store import HarnessStore

    # A key has to look present, or the offline fallback builds unconditionally — which is
    # its own deliberate behaviour, pinned separately below.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    service = HarnessService(store=HarnessStore(tmp_path))
    built: list[str] = []
    monkeypatch.setattr(service, "create_environment", lambda *a, **k: built.append("x"))
    monkeypatch.setattr(
        service_module, "chat_agent_reply_stream",
        lambda **kwargs: iter([("text", "hey! not much. what are you working on?")]),
    )

    result = service.dispatch_coordinator("hey whats up")
    assert not built, "a greeting must not compile a circuit"
    assert not result.built
    assert result.environment_id is None
    assert "hey" in result.summary.lower()

    stored = service.store.list_agent_messages("main", None)
    assert [item.speaker for item in stored] == ["user", "assistant"]
    assert "what are you working on" in stored[1].content


def test_the_build_tool_is_what_triggers_a_circuit(tmp_path, monkeypatch) -> None:
    """And the decision rides in the same response as the reply, so they cannot disagree."""
    import json

    from harness import service as service_module
    from harness.service import HarnessService
    from harness.store import HarnessStore

    service = HarnessService(store=HarnessStore(tmp_path))
    monkeypatch.setattr(
        service_module, "chat_agent_reply_stream",
        lambda **kwargs: iter([
            ("text", "Building you an icy one now."),
            ("tool", json.dumps({"name": "build_circuit", "input": {"reason": "icy track"}})),
        ]),
    )
    seen: list[str] = []
    conversations: list[list[dict]] = []
    original = service.create_environment

    def spy(prompt, *args, **kwargs):
        seen.append(prompt)
        conversations.append(kwargs.get("conversation") or [])
        return original(prompt, *args, **kwargs)

    monkeypatch.setattr(service, "create_environment", spy)
    result = service.dispatch_coordinator("an icy track please")
    assert seen == ["an icy track please"], "the request reaches the compiler unrewritten"
    assert conversations == [[{"role": "user", "content": "an icy track please"}]]
    assert result.built and result.environment_id


def test_an_explicit_high_elevation_build_is_not_misread_as_feedback(tmp_path, monkeypatch) -> None:
    """Recent requirement context must not suppress an unambiguous new 3D request."""
    import json

    from harness import service as service_module
    from harness.service import HarnessService
    from harness.store import HarnessStore

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    service = HarnessService(store=HarnessStore(tmp_path))
    monkeypatch.setattr(
        service_module, "chat_agent_reply_stream",
        lambda **kwargs: iter([
            ("text", "I will build that circuit."),
            ("tool", json.dumps({"name": "record_feedback", "input": {"confirmations": []}})),
        ]),
    )
    original_create = service.create_environment

    def create_offline(*args, **kwargs):
        kwargs["provider"] = "offline"
        return original_create(*args, **kwargs)

    monkeypatch.setattr(service, "create_environment", create_offline)

    result = service.dispatch_coordinator(
        "let's make a simple square circuit with high elevation differentials"
    )
    record = service.get_environment(result.environment_id or "")
    assert result.built and record is not None
    assert record.scene.elevation is not None
    assert record.scene.elevation.profile.value == "alpine"
    assert record.fidelity is not None and record.fidelity.faithful


def test_designless_new_circuit_reuses_the_last_explicit_brief(tmp_path, monkeypatch) -> None:
    """A follow-up cannot compile the words 'okay make a new one' as a track."""
    from harness import service as service_module
    from harness.models import AgentMessage
    from harness.service import HarnessService
    from harness.store import HarnessStore

    brief = "let's make a simple square circuit with high elevation differentials"
    store = HarnessStore(tmp_path)
    store.save_agent_message(AgentMessage(
        id="previous-build", agent_role="main", speaker="user", content=brief,
        created_at="2026-08-24T12:00:00+00:00",
    ))
    service = HarnessService(store=store)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    monkeypatch.setattr(
        service_module, "chat_agent_reply_stream",
        lambda **kwargs: iter([("text", "Building another version of that circuit.")]),
    )
    original_create = service.create_environment
    seen: list[str] = []

    def create_offline(prompt, *args, **kwargs):
        seen.append(prompt)
        kwargs["provider"] = "offline"
        return original_create(prompt, *args, **kwargs)

    monkeypatch.setattr(service, "create_environment", create_offline)
    result = service.dispatch_coordinator("okay make a new one")
    record = service.get_environment(result.environment_id or "")

    assert result.built and seen == [brief]
    assert record is not None and record.scene.track_report is not None
    assert len(record.scene.track_report.corners) == 4
    assert record.scene.elevation is not None
    assert record.scene.elevation.profile.value == "alpine"
    assert record.fidelity is not None and record.fidelity.faithful


def test_high_relief_does_not_secretly_require_the_generator_default_crest_count() -> None:
    """Only an explicitly numbered hill request gets an exact crest-count check."""
    from harness.generation_spec import extract_spec

    unnumbered = extract_spec("a square circuit with high elevation differentials")
    numbered = extract_spec("a square circuit with four hills")
    assert "elevation_hill_count" not in [item.kind for item in unnumbered.assertions]
    assert "elevation_hill_count" in [item.kind for item in numbered.assertions]


def test_the_coordinator_tool_carries_no_paraphrase_of_the_request(tmp_path) -> None:
    """Anything but a log line here would compete with the user's own words."""
    from harness.providers import COORDINATOR_TOOLS

    tools = {tool["name"]: tool for tool in COORDINATOR_TOOLS}
    assert set(tools) == {"build_circuit", "record_feedback"}
    assert set(tools["build_circuit"]["input_schema"]["properties"]) == {"reason"}


def test_a_normal_can_we_do_a_track_request_forces_environment_creation() -> None:
    """A plain-language build request must not depend on an optional chat tool call."""
    from harness.service import _is_explicit_circuit_request

    assert _is_explicit_circuit_request(
        "Can we please do a legitimate circle track with no corners in 3D please?"
    )


def test_the_offline_path_still_builds_without_a_model(tmp_path, monkeypatch) -> None:
    """No key means no judgement to make, so the no-model contract is unchanged.

    Skipping the build offline would be a decision nothing actually made, and it would
    leave CI unable to exercise the coordinator flow at all.
    """
    from harness.service import HarnessService
    from harness.store import HarnessStore

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = HarnessService(store=HarnessStore(tmp_path))
    result = service.dispatch_coordinator("an icy circuit with two rivals")
    assert result.built and result.environment_id


# --- precedents: rewarding readings a person confirmed -------------------------


def _confirmed_setup():
    from harness.prompt_spec import PromptSpec, Requirement, RequirementCheck, RequirementVerdict

    prompt = "a slippery circuit with two aggressive rivals"
    scene, _certificate, _notes = compile_certified_scene(prompt, parse_track_prompt(prompt), 3)
    spec = PromptSpec(prompt=prompt, requirements=[
        Requirement(id="R1", category="dynamics", statement="the track is slippery",
                    quote="slippery", checks=[RequirementCheck(kind="grip_max", target=0.6)]),
        Requirement(id="R2", category="entity", statement="two rivals", quote="two aggressive rivals",
                    checks=[RequirementCheck(kind="npc_count", target=2)]),
    ])
    verdicts = [
        RequirementVerdict(id="R1", category="dynamics", statement="the track is slippery",
                           quote="slippery", satisfied=True, method="check", evidence="grip_max=0.55"),
        RequirementVerdict(id="R2", category="entity", statement="two rivals",
                           quote="two aggressive rivals", satisfied=True, method="check", evidence="npc_count=2"),
    ]
    return spec, scene, verdicts


def test_a_precedent_needs_both_the_measurement_and_the_person() -> None:
    """Measured-but-unconfirmed teaches nothing about whether the reading was right."""
    from harness.precedents import distil

    spec, scene, verdicts = _confirmed_setup()
    assert distil(spec, scene, verdicts, {}, now="t0") == []
    confirmed = distil(spec, scene, verdicts, {"R1": True, "R2": True}, now="t0")
    assert {item.check_kind for item in confirmed} == {"grip_max", "npc_count"}


def test_a_rejected_reading_is_never_stored_as_a_precedent() -> None:
    """There is nothing reusable in a reading the person said was wrong."""
    from harness.precedents import distil

    spec, scene, verdicts = _confirmed_setup()
    found = distil(spec, scene, verdicts, {"R1": False, "R2": True}, now="t0")
    assert [item.check_kind for item in found] == ["npc_count"]


def test_a_missed_requirement_is_never_stored_however_kind_the_user_was() -> None:
    from harness.prompt_spec import RequirementVerdict
    from harness.precedents import distil

    spec, scene, verdicts = _confirmed_setup()
    verdicts[0] = RequirementVerdict(
        id="R1", category="dynamics", statement="the track is slippery", quote="slippery",
        satisfied=False, method="check", evidence="grip is 1.0, needs at most 0.6",
    )
    found = distil(spec, scene, verdicts, {"R1": True, "R2": True}, now="t0")
    assert [item.check_kind for item in found] == ["npc_count"]


def test_a_precedent_records_the_settings_that_delivered_it(tmp_path) -> None:
    """The reusable part is the engine values, not the fact that someone was pleased."""
    from harness.precedents import distil

    spec, scene, verdicts = _confirmed_setup()
    grip = next(i for i in distil(spec, scene, verdicts, {"R1": True, "R2": True}, now="t0")
                if i.check_kind == "grip_max")
    assert "grip" in grip.settings and "surface" in grip.settings
    assert grip.settings["grip"] == scene.grip
    assert grip.quote == "slippery", "the user's own phrasing is what a later hint matches on"


def test_retrieval_costs_nothing_when_nothing_is_relevant(tmp_path) -> None:
    """The whole design is control-F, not a corpus in context."""
    from harness.precedents import distil, guidance
    from harness.prompt_spec import PromptSpec, Requirement, RequirementCheck
    from harness.store import HarnessStore

    store = HarnessStore(tmp_path)
    spec, scene, verdicts = _confirmed_setup()
    for precedent in distil(spec, scene, verdicts, {"R1": True, "R2": True}, now="t0"):
        store.save_precedent(precedent)

    unrelated = PromptSpec(prompt="four laps", requirements=[Requirement(
        id="R1", category="objective", statement="four laps", quote="four laps",
        checks=[RequirementCheck(kind="laps", target=4)])])
    assert guidance(unrelated, store.precedents_for) == ""

    similar = PromptSpec(prompt="make it slick", requirements=[Requirement(
        id="R1", category="dynamics", statement="slick", quote="slick",
        checks=[RequirementCheck(kind="grip_max", target=0.5)])])
    hint = guidance(similar, store.precedents_for)
    assert "grip_max" in hint and "slippery" in hint
    assert "npc_count" not in hint, "only kinds this contract contains may be recalled"


def test_precedents_are_ranked_by_closeness_to_what_is_being_asked(tmp_path) -> None:
    from harness.precedents import Precedent, guidance
    from harness.prompt_spec import PromptSpec, Requirement, RequirementCheck
    from harness.store import HarnessStore

    store = HarnessStore(tmp_path)
    for index, target in enumerate((0.45, 1.15)):
        store.save_precedent(Precedent(
            id=f"p{index}", created_at=f"t{index}", check_kind="grip_max", check_target=target,
            statement=f"grip {target}", quote=f"phrase-{target}", settings={"grip": target},
        ))
    spec = PromptSpec(prompt="slick", requirements=[Requirement(
        id="R1", category="dynamics", statement="slick", quote="slick",
        checks=[RequirementCheck(kind="grip_max", target=0.5)])])
    hint = guidance(spec, store.precedents_for, limit=1)
    assert "phrase-0.45" in hint and "phrase-1.15" not in hint

"""Invariants for probe measurement, specification scoring, and generation arms.

The generation comparison is only worth anything if the measuring instrument is
deterministic, the grader is separate from the thing being graded, and the search
arm cannot see the held-out half of a specification. Each of those is asserted
here rather than assumed.
"""

from __future__ import annotations

import pytest

from harness.generation import generate
from harness.generation_spec import Assertion, GenerationSpec, extract_spec, score
from harness.probes import measure
from harness.racing import _coerce_plan_payload, compile_certified_scene
from harness.track_grammar import TrackPlan, parse_track_prompt

LOCATED_BRIEF = "slippery curvy track with a 90 degree bend in the top right and three aggressive npcs"


def compile_brief(brief: str, seed: int = 17):
    return compile_certified_scene(brief, parse_track_prompt(brief), seed)[0]


@pytest.fixture(scope="module")
def located_scene():
    return compile_brief(LOCATED_BRIEF)


def test_probe_report_is_a_pure_function_of_the_scene(located_scene):
    """Attribution depends on this: same scene in, same measurement out."""
    assert measure(located_scene).model_dump() == measure(located_scene).model_dump()


def test_probes_are_stable_across_identical_compilations():
    first, second = compile_brief(LOCATED_BRIEF), compile_brief(LOCATED_BRIEF)
    assert measure(first).model_dump(exclude={"checked_seed"}) == measure(second).model_dump(exclude={"checked_seed"})


def test_reference_and_unbraked_drivers_are_measured_separately(located_scene):
    report = measure(located_scene)
    assert report.oracle_finished
    # The two drivers share aiming logic and differ only in speed control, so the
    # unbraked driver must be the one that runs wide.
    assert report.naive_off_track_ticks >= report.off_track_ticks
    assert 0 < report.brake_fraction < 1
    assert report.simulated_ticks > 0


def test_extraction_only_asserts_what_the_brief_evidences():
    spec = extract_spec("a circuit")
    assert [assertion.kind for assertion in spec.assertions] == ["oracle_finishes"]


def test_extraction_reads_geometry_traffic_and_grip():
    kinds = {assertion.kind: assertion for assertion in extract_spec(LOCATED_BRIEF).assertions}
    assert kinds["grip_max"].target <= .6
    assert kinds["npc_count"].target == 3
    assert kinds["npc_profiles"].target == ["aggressor"] * 3
    assert kinds["corner_in_region"].target == {"angle": 90.0, "region": "top-right"}


def test_extraction_reads_outcome_targets_that_need_a_simulator():
    laptime = {item.kind: item for item in extract_spec(
        "a one-lap circuit that takes a competent driver about 45 seconds").assertions}
    assert laptime["oracle_seconds"].target == 45.0
    assert laptime["oracle_seconds"].tolerance > 0
    spread = {item.kind: item for item in extract_spec(
        "three opponents that finish within 2 seconds of each other").assertions}
    assert spread["field_spread_max"].target == 2.0


def test_extraction_and_scoring_keep_3d_elevation_in_the_contract():
    from harness.racing3d import compile_racing_3d_scene
    from harness.track3d import parse_elevation_prompt

    brief = "a 3d circuit over gentle rolling elevation"
    spec = extract_spec(brief)
    vertical = [item for item in spec.assertions if item.kind.startswith("elevation_")]
    assert {item.kind for item in vertical} == {
        "elevation_profile", "elevation_amplitude_min", "elevation_hill_count",
    }
    scene = compile_racing_3d_scene(
        brief, parse_track_prompt(brief), parse_elevation_prompt(brief), 17,
    )[0]
    result = score(spec.model_copy(update={"assertions": vertical}), scene, None)
    assert result.conjunction


def test_generation_can_inject_the_3d_compiler_without_changing_the_arms():
    from harness.racing3d import compile_racing_3d_scene
    from harness.track3d import parse_elevation_prompt

    brief = "a compact 3d circuit over gentle rolling elevation"
    elevation = parse_elevation_prompt(brief)

    def compile_3d(prompt, plan, seed):
        return compile_racing_3d_scene(prompt, plan, elevation, seed)

    outcome = generate(
        brief, 17, arm="oneshot", provider="offline", compile_scene=compile_3d,
    )
    assert outcome.scene is not None
    assert outcome.scene.elevation is not None and not outcome.scene.elevation.is_flat


def test_scoring_a_missing_scene_fails_every_assertion():
    spec = extract_spec(LOCATED_BRIEF)
    result = score(spec, None, None)
    assert result.total == len(spec.assertions)
    assert result.satisfied == 0
    assert not result.conjunction


def test_scoring_reports_the_measured_value_on_a_miss(located_scene):
    spec = GenerationSpec(prompt="x", assertions=[Assertion(
        id="impossible", kind="oracle_seconds", target=2.0, tolerance=.1,
        label="a two second race",
    )])
    result = score(spec, located_scene, measure(located_scene))
    assert not result.conjunction
    assert "the brief asked for 2.0" in result.results[0].message
    assert result.results[0].achieved > 2.0


def test_held_out_assertions_are_partitioned_out_of_the_visible_spec():
    spec = GenerationSpec(prompt="x", assertions=[
        Assertion(id="a", kind="oracle_finishes", target=True),
        Assertion(id="b", kind="closure_error_max", target=.5, held_out=True),
    ])
    assert [item.id for item in spec.visible().assertions] == ["a"]
    assert [item.id for item in spec.held_out().assertions] == ["b"]


def test_search_never_scores_itself_against_held_out_assertions():
    """The generalization claim depends on the search being blind to these."""
    spec = GenerationSpec(prompt=LOCATED_BRIEF, assertions=[
        Assertion(id="visible", kind="oracle_finishes", target=True, label="drivable"),
        Assertion(id="secret", kind="oracle_seconds", target=1.0, tolerance=.1,
                  label="an impossible one second race", held_out=True),
    ])
    outcome = generate(LOCATED_BRIEF, 17, arm="harness", provider="offline", candidates=1, spec=spec)
    assert outcome.search_score is not None
    assert [result.id for result in outcome.search_score.results] == ["visible"]


def test_the_dial_solver_hits_a_target_the_authored_plan_missed():
    """A finish-gap target is solvable only by measuring, which is the whole claim."""
    brief = "a one-lap asphalt circuit with three opponents that finish within 2 seconds of each other"
    baseline = generate(brief, 17, arm="oneshot", provider="offline")
    searched = generate(brief, 17, arm="harness", provider="offline", candidates=1)
    assert baseline.scene is not None and searched.scene is not None
    spec = extract_spec(brief)
    baseline_score = score(spec, baseline.scene, measure(baseline.scene))
    searched_score = score(spec, searched.scene, measure(searched.scene))
    assert searched_score.satisfied > baseline_score.satisfied
    assert searched_score.conjunction


def test_solved_scenes_keep_the_constraints_the_brief_stated():
    """A dial move is rejected if it trades a stated constraint for a numeric one."""
    brief = ("a one-lap slippery circuit with three aggressive opponents that finish within "
             "2 seconds of each other")
    outcome = generate(brief, 43, arm="harness", provider="offline", candidates=1)
    assert outcome.scene is not None
    assert outcome.scene.grip <= .6
    assert sorted(item.profile.value for item in outcome.scene.npc_behaviors) == ["aggressor"] * 3


def test_offline_arms_are_deterministic():
    first = generate(LOCATED_BRIEF, 17, arm="harness", provider="offline", candidates=2)
    second = generate(LOCATED_BRIEF, 17, arm="harness", provider="offline", candidates=2)
    assert first.scene is not None and second.scene is not None
    assert first.scene.model_dump(exclude={"id"}) == second.scene.model_dump(exclude={"id"})


def test_a_verbose_creator_rationale_is_clamped_rather_than_discarded():
    """Free text is a range too; prose must not waste a whole generation."""
    payload = _coerce_plan_payload({
        "title": "T" * 200, "rationale": "R" * 900, "corners": [{}, {}, {}],
    })
    plan = TrackPlan.model_validate(payload)
    assert len(plan.title) == 64
    assert len(plan.rationale) == 360


def test_an_empty_creator_rationale_is_filled_rather_than_rejected():
    plan = TrackPlan.model_validate(_coerce_plan_payload({"corners": [{}, {}, {}]}))
    assert plan.title and plan.rationale


def test_creator_circle_payload_preserves_its_zero_corner_geometry():
    plan = TrackPlan.model_validate(_coerce_plan_payload({
        "title": "Circle", "rationale": "A literal circular racing loop.",
        "loop_shape": "circle", "corners": [],
    }))
    assert plan.loop_shape == "circle"
    assert plan.corners == []


@pytest.mark.parametrize("phrase,expected", [
    ("two blockers", ["blocker", "blocker"]),
    ("2 blockers", ["blocker", "blocker"]),
    ("two blocking rivals", ["blocker", "blocker"]),
    ("three aggressive npcs", ["aggressor"] * 3),
    ("a blocker", ["blocker"]),
])
def test_a_temperament_word_can_be_the_noun_it_counts(phrase, expected):
    """"two blockers" used to compile a single opponent, silently."""
    plan = parse_track_prompt(f"a circuit with {phrase}")
    assert sorted(npc.profile.value for npc in plan.npcs) == sorted(expected)

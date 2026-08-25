"""Pre-registered generation benchmark: briefs paired with hand-authored specs.

Every assertion here was written before any arm was run, and the tolerances are
part of the registration. Two rules keep the comparison honest:

- `held_out=True` assertions are never visible to any generator's search loop.
  They are the generalization claim: a harness that only moved the assertions it
  could see is a harness that optimized the metric, not the circuit.
- Nothing in this file is derived from `extract_spec`. The grader and the
  harness's own objective are separate objects, so the harness can be wrong
  about what the brief asked for and the grade will say so.

Case classes mirror the design taxonomy:
  `lexical`     one or two constraints a base model should already satisfy;
                included so a null result is visible rather than hidden.
  `conjunction` many in-distribution constraints at once.
  `outcome`     properties only a simulator rollout can settle.
  `numeric`     a number that has to be hit, not guessed.
"""

from __future__ import annotations

from harness.generation_spec import Assertion, GenerationSpec


def _spec(case_id: str, case_class: str, prompt: str, assertions: list[Assertion]) -> GenerationSpec:
    return GenerationSpec(
        prompt=prompt, case_id=case_id, case_class=case_class,
        assertions=[
            assertion.model_copy(update={"id": f"{case_id}-{index}-{assertion.kind}"})
            for index, assertion in enumerate(assertions)
        ],
    )


def _assertion(kind: str, target=None, label: str = "", tolerance: float = 0, held_out: bool = False) -> Assertion:
    return Assertion(id=kind, kind=kind, target=target, label=label or kind,
                     tolerance=tolerance, held_out=held_out)


SUITE: tuple[GenerationSpec, ...] = (
    _spec(
        "lex-oval", "lexical",
        "A fast asphalt oval with one opponent and no barriers.",
        [
            _assertion("surface", "asphalt", "asphalt surface"),
            _assertion("npc_count", 1, "exactly one opponent"),
            _assertion("barrier_count", 0, "no barriers"),
            _assertion("oracle_finishes", True, "the reference driver completes it"),
            _assertion("closure_error_max", .5, "the loop closes", held_out=True),
            _assertion("angle_fidelity_max", .5, "authored angles are honoured", held_out=True),
        ],
    ),
    _spec(
        "lex-located", "lexical",
        "A curvy asphalt circuit with a 90 degree bend in the top right.",
        [
            _assertion("corner_in_region", {"angle": 90, "region": "top-right"},
                       "a 90 degree corner in the top right", tolerance=5),
            _assertion("oracle_finishes", True, "the reference driver completes it"),
            _assertion("closure_error_max", .5, "the loop closes", held_out=True),
            _assertion("corner_count_min", 4, "at least four corners", held_out=True),
        ],
    ),
    _spec(
        "conj-clay", "conjunction",
        "A slippery clay circuit running clockwise over three laps, with a 120 degree hairpin "
        "in the bottom left, a 60 degree kink in the top center, four barriers, and two "
        "blockers spread around the track.",
        [
            _assertion("surface", "clay", "clay surface"),
            _assertion("grip_max", .6, "slippery, grip at most 0.6"),
            _assertion("direction", "clockwise", "runs clockwise"),
            _assertion("laps", 3, "three laps"),
            _assertion("corner_in_region", {"angle": 120, "region": "bottom-left"},
                       "a 120 degree hairpin in the bottom left", tolerance=5),
            _assertion("corner_in_region", {"angle": 60, "region": "top-center"},
                       "a 60 degree kink in the top center", tolerance=5),
            _assertion("barrier_count", 4, "four barriers"),
            _assertion("npc_count", 2, "two opponents"),
            _assertion("npc_profiles", ["blocker", "blocker"], "both opponents are blockers"),
            _assertion("npc_start_mode", "distributed", "traffic spread around the lap"),
            _assertion("oracle_finishes", True, "the reference driver completes it"),
            _assertion("closure_error_max", .5, "the loop closes", held_out=True),
            _assertion("off_track_ticks_max", 60, "the reference driver mostly stays on track",
                       held_out=True),
        ],
    ),
    _spec(
        "conj-ice", "conjunction",
        "A narrow ice circuit, counterclockwise, two laps, with a 150 degree hairpin in the "
        "bottom right, a 90 degree corner in the top left, a 45 degree kink on the left, two "
        "barriers, and three aggressive rivals.",
        [
            _assertion("surface", "ice", "ice surface"),
            _assertion("direction", "counterclockwise", "runs counterclockwise"),
            _assertion("laps", 2, "two laps"),
            _assertion("track_width_max", 124, "a narrow corridor"),
            _assertion("corner_in_region", {"angle": 150, "region": "bottom-right"},
                       "a 150 degree hairpin in the bottom right", tolerance=5),
            _assertion("corner_in_region", {"angle": 90, "region": "top-left"},
                       "a 90 degree corner in the top left", tolerance=5),
            _assertion("corner_in_region", {"angle": 45, "region": "left"},
                       "a 45 degree kink on the left", tolerance=5),
            _assertion("barrier_count", 2, "two barriers"),
            _assertion("npc_count", 3, "three opponents"),
            _assertion("npc_profiles", ["aggressor", "aggressor", "aggressor"],
                       "all three opponents are aggressors"),
            _assertion("oracle_finishes", True, "the reference driver completes it"),
            _assertion("closure_error_max", .5, "the loop closes", held_out=True),
        ],
    ),
    _spec(
        "out-hardfair", "outcome",
        "A hard but fair asphalt circuit with two barriers that punishes any driver who never "
        "brakes.",
        [
            _assertion("oracle_finishes", True, "a competent driver can finish it"),
            _assertion("naive_off_track_min", 40, "an unbraked driver runs wide repeatedly"),
            _assertion("barrier_count", 2, "two barriers"),
            _assertion("off_track_ticks_max", 40, "but a competent driver keeps it on the road",
                       held_out=True),
        ],
    ),
    _spec(
        "out-overtaking", "outcome",
        "A flowing asphalt circuit with four opponents that produces real overtaking and a "
        "close finish.",
        [
            _assertion("order_changes_min", 3, "at least three position changes in the field"),
            _assertion("field_spread_max", 3.0, "the opponents finish within three seconds"),
            _assertion("npc_count", 4, "four opponents"),
            _assertion("oracle_finishes", True, "the reference driver completes it", held_out=True),
        ],
    ),
    _spec(
        "out-braking", "outcome",
        "A brake-heavy stop-start asphalt circuit with tight corners and no barriers.",
        [
            _assertion("brake_fraction_min", .3, "at least 30% of the race is spent braking"),
            _assertion("barrier_count", 0, "no barriers"),
            _assertion("oracle_finishes", True, "the reference driver completes it"),
            _assertion("min_radius_max", 110, "it genuinely contains tight corners", held_out=True),
        ],
    ),
    _spec(
        "num-laptime", "numeric",
        "A single-lap asphalt circuit that takes a competent driver about 45 seconds, with two "
        "opponents.",
        [
            _assertion("oracle_seconds", 45.0, "a 45 second reference race time", tolerance=5),
            _assertion("laps", 1, "one lap"),
            _assertion("npc_count", 2, "two opponents"),
            _assertion("oracle_finishes", True, "the reference driver completes it", held_out=True),
        ],
    ),
    _spec(
        "num-spread", "numeric",
        "A one-lap asphalt circuit with three opponents that finish within 2 seconds of each "
        "other.",
        [
            _assertion("field_spread_max", 2.0, "the opponents finish within two seconds"),
            _assertion("npc_count", 3, "three opponents"),
            _assertion("laps", 1, "one lap"),
            _assertion("oracle_finishes", True, "the reference driver completes it", held_out=True),
        ],
    ),
)

CLASSES = ("lexical", "conjunction", "outcome", "numeric")

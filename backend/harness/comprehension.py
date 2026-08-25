"""Reading a brief into a contract, before anything is built.

This is the first stage of generation and the only one that interprets English.
Splitting it out is the point: a single agent asked to understand and design at
the same time will quietly resolve every ambiguity in favour of whatever circuit
it was already going to draw, and nothing downstream can tell that happened. Here
the reading is committed to first, as ids, and everything after it is measured
against that commitment.

Two rules keep this honest:

- The model chooses *which* mechanical check expresses a requirement, from a fixed
  vocabulary of evaluators the engine already implements. It never decides whether
  a check passes. Saying what was asked for and saying whether it was delivered
  are different jobs, and the second one belongs to the simulator.
- A detail the brief did not mention is recorded as unspecified rather than
  invented. A generator that is told "you may choose the surface" behaves very
  differently from one that is told "the surface must be ice" because a previous
  stage guessed.
"""

from __future__ import annotations

from typing import Any

from .conversation import conversation_context
from .generation_spec import _EVALUATORS, _PROBE_KINDS
from .prompt_spec import (
    CATEGORIES, PromptSpec, Requirement, RequirementCheck,
)
from .providers import ProviderError, ProviderUsage, active_provider, anthropic_json, configured_model

MAX_REQUIREMENTS = 24


# --- the check vocabulary -----------------------------------------------------
#
# Model-facing descriptions of every evaluator the engine can settle mechanically.
# Kept here rather than on the evaluators themselves because this is prose written
# for a model, not documentation of the measurement. `test_check_vocabulary_is_
# complete` asserts this table and the registry never drift apart, so a new
# evaluator cannot be added without deciding how a brief would ask for it.

CHECK_VOCABULARY: dict[str, str] = {
    # --- world configuration, read straight off the compiled scene ---
    "surface": 'road material. target is one of "asphalt", "clay", "ice". Use clay for '
               'dirt, gravel, sand or rally, ice for snow or frozen. A wet or slippery '
               'paved road is asphalt with low grip, NOT ice.',
    "grip_max": "grip is at most this. target 0.3-1.2. Use ~0.6 for wet/slick/greasy, "
                "~0.45 for very slippery, ~0.4 for treacherous.",
    "grip_min": "grip is at least this. target 0.3-1.2. Use ~1.1 for sticky/high-grip.",
    "grip_target": "grip lands near this. target 0.3-1.2, set tolerance ~0.1.",
    "track_width_max": "corridor is at most this many pixels wide. Range 110-170; 132 is "
                       "standard. Use ~120 for narrow/claustrophobic/street.",
    "track_width_min": "corridor is at least this many pixels wide. Use ~150 for wide.",
    "laps": "exact lap count. target is an integer 1-10.",
    "direction": 'target is "clockwise" or "counterclockwise".',
    "loop_shape": 'exact centerline primitive. target is "circle" only when the brief explicitly asks '
                  'for a circular, round, or no-corner loop; this is a literal constant-radius loop with zero authored corners.',
    "npc_start_mode": 'target is "grid" (default) or "distributed" for traffic spread '
                      "around the lap rather than lined up behind you.",
    "start_line_region": "map region for the shared start/finish line. target is one of the "
                         "named track regions, or auto when the brief does not place it.",
    "player_grid_position": "player's starting grid place. target is integer 1-6; 1 is pole, "
                            "later positions line up behind the shared start line.",
    "elevation_profile": 'vertical road profile. target is one of "flat", "rolling", '
                         '"hilly", or "alpine". Use rolling for gentle undulation, hilly '
                         "for pronounced climbs, and alpine for steep mountain terrain or high/large "
                         "elevation differentials.",
    "elevation_amplitude_min": "peak-to-trough elevation change is at least this many meters. "
                               "Use ~4 for gentle hills, ~8 for pronounced hills, and ~14 for "
                               "steep alpine terrain.",
    "elevation_hill_count": "exact number of visible crests around the lap. target is an "
                            "integer 1-8; use only when the brief names a count.",

    # --- entities ---
    "npc_count": "exact number of opponent cars. target is an integer 0-5.",
    "npc_profiles": 'exact multiset of opponent temperaments, as a sorted list of '
                    '"backmarker" (slow, passive), "cruiser" (steady), "racer" (normal '
                    'competitor), "aggressor" (commits to passes, covers your line), '
                    '"blocker" (slow but defends). e.g. ["aggressor","aggressor","racer"].',
    "barrier_count": "exact number of lane-edge barriers. target is an integer 0-6.",

    # --- geometry ---
    "corner_count_min": "at least this many corners for a cornered layout. target 3-10. Use ~7 for twisty or "
                        "technical, ~9 for very twisty or serpentine, ~4 for flowing. Note that "
                        "corner count and corner size trade against each other: more corners "
                        "means every corner is necessarily smaller.",
    "corner_count_max": "at most this many corners for a cornered layout. target 3-10. Use ~5 for flowing, sweeping, "
                        "or simple, ~4 for an oval.",
    "corner_in_region": 'a corner of a given angle lands in a given screen region. target is '
                        '{"angle": <degrees 12-172>, "region": <one of top-left, top-center, '
                        'top-right, left, center, right, bottom-left, bottom-center, '
                        'bottom-right>}. Set tolerance to 5. Use ONLY when the brief locates '
                        'a specific feature on the screen.',
    "min_radius_max": "the tightest corner is at most this radius in pixels, i.e. the circuit "
                      "DOES contain something genuinely tight. target ~90 for a hairpin, ~110 "
                      "for tight. Easy to satisfy at any corner count.",
    "min_radius_min": "the tightest corner is at least this radius in pixels, i.e. NOTHING on "
                      "the circuit is tight. The check for 'nothing tight', 'no hairpins', "
                      "'sweeping', 'flowing'. HARD LIMIT: the finished circuit is scaled to fit "
                      "a fixed box, so achievable radius falls as corner count rises. The most "
                      "any circuit can reach is about 150 (at 4 corners), 140 (3 corners), 115 "
                      "(5 corners), 95 (7+ corners). Use target 130 for a sweeping circuit, and "
                      "never above 140 — a higher target cannot be built and will always fail. "
                      "Do not combine this with corner_count_min above 5.",
    "longest_straight_min": "the longest straight is at least this many pixels. target ~250 "
                            "for a long straight or slipstreaming, ~180 for a decent one. "
                            "Typical circuits sit at 140-290.",
    "longest_straight_max": "the longest straight is at most this many pixels; for a circuit "
                            "with no real straights. target ~160.",

    # --- outcome properties; these require simulator rollouts, so use sparingly ---
    "oracle_finishes": "a competent reference driver can complete it. target true. Implied by "
                       "every brief, so only state it when the brief stresses difficulty.",
    "naive_finishes": "an unbraked full-throttle driver finishes. target false for a circuit "
                      "that must punish carelessness.",
    "naive_off_track_min": "an unbraked driver runs wide at least this many ticks. target ~40 "
                           "for punishing, unforgiving, or hard-but-fair.",
    "naive_off_track_max": "an unbraked driver runs wide at most this many ticks, i.e. even "
                           "carelessness mostly stays on the road. This is the check for "
                           "'forgiving', 'beginner-friendly', 'hard to crash on'. target ~5.",
    "oracle_seconds": "reference race time near this many seconds. Set tolerance to about 12% "
                      "of the target. Typical single laps land near 35-40s.",
    "oracle_seconds_max": "reference race time is at most this many seconds; for a short or "
                          "quick circuit.",
    "off_track_ticks_max": "the reference driver leaves the road at most this many ticks. "
                           "target 0 for a clean, forgiving layout.",
    "order_changes_min": "at least this many position changes happen in the field. target 2 "
                         "for overtaking, wheel-to-wheel, or changes of position.",
    "field_spread_max": "the opponents finish within this many seconds of each other. target "
                        "3 for a photo finish or close racing.",
    "brake_fraction_min": "at least this share of the race is spent braking, 0-1. target 0.3 "
                          "for brake-heavy or stop-and-go. Typical circuits sit near 0.25.",
    "brake_fraction_max": "at most this share of the race is spent braking, 0-1. target 0.1 "
                          "for flat-out or no-braking.",
    "mean_speed_min": "reference mean speed is at least this, in pixels per tick. target ~4.5 "
                      "for fast or high-speed. Typical circuits sit at 3.8-4.7.",
    "mean_speed_max": "reference mean speed is at most this, in pixels per tick. target ~4.0 "
                      "for slow or technical.",

    # --- appearance; cosmetic, and satisfied by the visual plan rather than the geometry ---
    "road_colour": 'the road surface is roughly this colour. target is "#rrggbb" or a colour '
                   'word like "black" or "neon pink". Set tolerance 0.18 unless the brief is '
                   "insistent about an exact shade, then 0.08.",
    "terrain_colour": "the ground around the circuit is roughly this colour. Same target form.",
    "barrier_colour": "the lane-edge barriers are roughly this colour. Same target form.",
    "player_car_colour": "the player's car is roughly this colour. Same target form.",
    "opponent_car_colour": "the opponent cars are roughly this colour. Same target form.",
    "sky_colour": "the sky is roughly this colour. Same target form. This is how a request "
                  "for a night or sunset look is satisfied.",
    "kerbs_present": "whether the red-and-white edge striping is drawn. target false for 'no "
                     "edge lines', 'no kerbs', 'plain edges'.",
    "scenery_count": "exact number of coloured ground bands crossing the map — how a river, "
                     "sand trap, or painted run-off is represented. target is an integer 0-4.",

    # --- geometric quality; rarely asked for directly ---
    "angle_fidelity_max": "authored-versus-achieved turn angle error, in degrees, at most this.",
    "closure_error_max": "loop closure error at most this many pixels.",
}


def check_vocabulary_text() -> str:
    """The settleable checks, as the comprehension model is shown them."""
    return "\n".join(f"  {kind}: {text}" for kind, text in CHECK_VOCABULARY.items())


_COMPREHENSION_SYSTEM_TEMPLATE = (
    "You read a user's brief for a racing circuit and turn it into an explicit, "
    "checkable contract. You do NOT design the circuit. Another stage builds it from your "
    "contract, and a verifier checks the built world against your contract requirement by "
    "requirement.\n\n"
    "Your only job is to capture what the user actually asked for, completely and without "
    "adding to it.\n\n"
    "RULES\n"
    "1. One requirement per distinct thing the brief asks for. Number them R1, R2, R3 in the "
    "order they appear. Never merge two asks into one requirement: 'a hairpin bottom left and "
    "a chicane on the right' is two requirements, and merging them means a verifier cannot "
    "report which one was missed.\n"
    "2. Quote the user's own words for each requirement in `quote`, verbatim and unedited. It "
    "is shown back to them when something is missed.\n"
    "3. `statement` is a self-contained restatement that makes sense without the brief.\n"
    "4. Attach mechanical checks whenever the vocabulary can settle a requirement. Prefer a "
    "check over leaving it to a judge: a measured number is worth more than an opinion. A "
    "requirement may need several checks — 'three aggressive opponents' is npc_count=3 AND "
    "npc_profiles=[aggressor,aggressor,aggressor].\n"
    "5. Leave `checks` EMPTY when nothing in the vocabulary settles it — atmosphere, theme, "
    "colour, a vibe like 'feels like a rollercoaster'. A human-judged requirement is fine. "
    "Inventing a numeric check that does not mean what the user said is not.\n"
    "6. Translate feel into measurable consequence where you honestly can. 'Beginner-friendly' "
    "means high grip, a wide corridor, and no viciously tight corner. 'Unforgiving' means an "
    "unbraked driver runs wide. Do this only when the mapping is genuinely implied.\n"
    "7. Do NOT invent. If the brief says nothing about opponents, do not require zero opponents "
    "and do not require some — list 'number of opponents' under `unspecified`. Only make it a "
    "requirement if the user constrained it, including negatively ('no traffic' IS npc_count=0).\n"
    "8. `priority` is 'must' by default. Use 'should' only for genuinely softened language: "
    "'ideally', 'if possible', 'maybe', 'would be nice'.\n"
    "8a. A circular, round, or 'no corners' circuit is a literal geometry request. Create a "
    "`loop_shape=circle` requirement and do NOT add a corner-count check: the compiler has a "
    "dedicated constant-radius, zero-corner centerline primitive.\n"
    "9. Put anything the engine has no dial for in `unsupported` and do NOT also make it a "
    "requirement. A requirement the engine cannot possibly satisfy is not a strict standard, "
    "it is a guaranteed failure that hides the real ones. The engine cannot do:\n"
    "   - weather particles, sound, time-of-day lighting, vehicle liveries, pit stops\n"
    "   - tunnels, bridges, or anything overhead\n"
    "   - figure-eight, crossing, or self-intersecting layouts: the compiled circuit is a "
    "single closed loop that provably never overlaps itself\n"
    "   - named real-world tracks reproduced literally (Monaco, Spa); treat those as a "
    "requirement only for the qualities the user implies, like narrowness, and list the "
    "literal reproduction as unsupported\n"
    "   Colour, palette, theme, and coloured ground features like a river or a sand trap ARE "
    "supported through a visual plan, so those are requirements, not unsupported. A visual "
    "requirement is cosmetic: it changes how the circuit looks, never how it drives. If a "
    "brief asks for a river, that is a coloured band of ground (scenery_count) — the car "
    "drives over it normally, and there is no water, no splash, and no grip change unless "
    "the brief separately asks for low grip.\n\n"
    "__DIMENSION_CAPABILITY__\n\n"
    "CHECK VOCABULARY — the only settleable kinds, with the scale each expects\n"
    + check_vocabulary_text()
)


def comprehension_system(dimensions: str = "2d") -> str:
    """Give the reader the capability boundary of the engine it will actually build."""
    if dimensions == "3d":
        capability = (
            "3D CAPABILITY: elevation, hills, banking, climbs, descents, crests, and blind "
            "brows are supported by the 3D surface fitter. Treat them as requirements, not "
            "unsupported requests when the brief asks for them. 'In 3D' selects the runtime and "
            "does NOT by itself request an elevation profile or height. Tunnels, bridges, and any "
            "overhead structure remain unsupported."
        )
    else:
        capability = (
            "2D CAPABILITY: elevation, hills, banking, jumps, crests, and blind brows are "
            "unsupported because the 2D circuit is flat."
        )
    return _COMPREHENSION_SYSTEM_TEMPLATE.replace("__DIMENSION_CAPABILITY__", capability)


COMPREHENSION_SYSTEM = comprehension_system()


_TARGET_SCHEMA: dict[str, Any] = {
    "description": "The value this check compares against. Its type depends on the kind: a "
                   "string for surface/direction, a number for every threshold, a boolean for "
                   "the finish checks, a list of strings for npc_profiles, and an "
                   "{angle, region} object for corner_in_region.",
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "array", "items": {"type": "string"}},
        {
            "type": "object",
            "properties": {
                "angle": {"type": "number"},
                "region": {"type": "string"},
            },
            "required": ["angle", "region"],
            "additionalProperties": False,
        },
    ],
}
"""Polymorphic check targets, spelled out because the structured-output API rejects
an untyped schema node and a bare `additionalProperties: true` object."""


def prompt_spec_schema() -> dict[str, Any]:
    """Structured-output schema for one comprehension call."""
    return {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string", "maxLength": 200,
                "description": "One line: what kind of circuit this brief is asking for.",
            },
            "requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": r"^R\d+$"},
                        "category": {"type": "string", "enum": list(CATEGORIES)},
                        "statement": {"type": "string", "minLength": 3, "maxLength": 200},
                        "quote": {"type": "string", "maxLength": 200},
                        "priority": {"type": "string", "enum": ["must", "should"]},
                        "checks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string", "enum": sorted(CHECK_VOCABULARY)},
                                    "target": _TARGET_SCHEMA,
                                    "tolerance": {"type": "number"},
                                },
                                "required": ["kind", "target"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["id", "category", "statement", "quote", "priority", "checks"],
                    "additionalProperties": False,
                },
            },
            "unspecified": {
                "type": "array",
                "items": {"type": "string", "maxLength": 120},
                "description": "Details the brief left open, which the generator may choose.",
            },
            "unsupported": {
                "type": "array",
                "items": {"type": "string", "maxLength": 120},
                "description": "Asks the engine has no dial for at all.",
            },
        },
        "required": ["summary", "requirements", "unspecified", "unsupported"],
        "additionalProperties": False,
    }


def comprehend(
    prompt: str, provider: str = "auto", conversation: list[dict[str, Any]] | None = None,
    dimensions: str = "2d",
) -> tuple[PromptSpec, ProviderUsage | None]:
    """Read one brief into a checkable contract.

    Falls back to the deterministic parser when no model is available, so the
    harness still runs offline — but that path is a floor, not the product path:
    it can only see phrasings its keyword table already knows.
    """
    import os

    resolved = active_provider() if provider == "auto" else provider
    if resolved == "offline":
        return _offline_spec(prompt), None
    if resolved not in {"anthropic", "openai"}:
        raise ProviderError(f"Unknown comprehension provider: {resolved}")

    context = conversation_context(conversation)
    payload, usage = anthropic_json(
        model=configured_model("ANTHROPIC_COMPREHENSION_MODEL"),
        max_tokens=3_000,
        system=comprehension_system(dimensions),
        prompt=(
            (
                "Recent conversation, for resolving references and retaining stated constraints. "
                "The current build request below is authoritative if anything conflicts:\n"
                f"{context}\n\n"
                if context else ""
            )
            + f"User brief:\n{prompt}\n\n"
            "Return the contract. Capture every concrete ask as its own numbered requirement "
            "with the user's own words quoted, attach mechanical checks wherever the "
            "vocabulary can settle one, and list what the brief left open rather than "
            "deciding it here."
        ),
        json_schema=prompt_spec_schema(),
        cache_system=True,
    )
    return _coerce_spec(prompt, payload), usage


def _coerce_spec(prompt: str, payload: dict) -> PromptSpec:
    """Repair a model-authored contract into the schema without discarding it.

    An unknown check kind or a duplicated id is a flaw in one requirement, not a
    reason to throw away a whole reading of the brief and start over.
    """
    requirements: list[Requirement] = []
    seen: set[str] = set()
    for index, item in enumerate(payload.get("requirements") or []):
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("id") or "").strip().upper()
        if not identifier.startswith("R") or not identifier[1:].isdigit() or identifier in seen:
            identifier = f"R{index + 1}"
        while identifier in seen:
            identifier = f"R{int(identifier[1:]) + 1}"
        seen.add(identifier)
        checks = [
            RequirementCheck(
                kind=check["kind"],
                target=check.get("target"),
                tolerance=float(check.get("tolerance") or 0.0),
            )
            for check in (item.get("checks") or [])
            # A hallucinated kind has no evaluator, so it could never be settled;
            # dropping it demotes the requirement to the judge rather than
            # crashing the whole generation on one bad enum.
            if isinstance(check, dict) and check.get("kind") in _EVALUATORS
        ]
        statement = str(item.get("statement") or "").strip()
        if len(statement) < 3:
            continue
        requirements.append(Requirement(
            id=identifier,
            category=item.get("category") if item.get("category") in CATEGORIES else "constraint",
            statement=statement[:200],
            quote=str(item.get("quote") or "")[:200],
            priority="should" if item.get("priority") == "should" else "must",
            checks=checks[:4],
        ))
    return PromptSpec(
        prompt=prompt,
        requirements=requirements,
        unspecified=[str(item)[:120] for item in (payload.get("unspecified") or [])][:20],
        unsupported=[str(item)[:120] for item in (payload.get("unsupported") or [])][:10],
        summary=str(payload.get("summary") or "")[:200],
    )


def needs_probes(spec: PromptSpec) -> bool:
    """Whether settling this contract requires simulator rollouts."""
    return any(
        check.kind in _PROBE_KINDS
        for requirement in spec.requirements
        for check in requirement.checks
    )


def _offline_spec(prompt: str) -> PromptSpec:
    """The no-API-key floor: the old keyword extractor, wrapped as requirements.

    Retained so tests and CI run without a key. It is deliberately not the
    production path — a keyword table cannot read a brief it has no keyword for,
    which is the failure this module exists to remove.
    """
    from .generation_spec import extract_spec
    from .track_grammar import unsupported_requests

    legacy = extract_spec(prompt)
    requirements = [
        Requirement(
            id=f"R{index + 1}",
            category=_CATEGORY_BY_KIND.get(assertion.kind, "constraint"),
            statement=assertion.describe(),
            quote="",
            checks=[RequirementCheck(
                kind=assertion.kind, target=assertion.target, tolerance=assertion.tolerance,
            )],
        )
        for index, assertion in enumerate(legacy.assertions)
    ]
    return PromptSpec(
        prompt=prompt, requirements=requirements,
        unsupported=unsupported_requests(prompt),
        summary="Offline keyword reading of the brief.",
    )


_CATEGORY_BY_KIND: dict[str, str] = {
    "surface": "dynamics", "grip_max": "dynamics", "grip_min": "dynamics",
    "grip_target": "dynamics", "laps": "objective", "direction": "layout",
    "npc_start_mode": "entity", "start_line_region": "entity", "player_grid_position": "entity",
    "track_width_max": "layout", "track_width_min": "layout",
    "npc_count": "entity", "npc_profiles": "entity", "barrier_count": "entity",
    "loop_shape": "layout", "corner_count_min": "layout", "corner_count_max": "layout",
    "corner_in_region": "layout", "min_radius_max": "layout",
    "longest_straight_min": "layout", "angle_fidelity_max": "layout",
    "closure_error_max": "layout",
}

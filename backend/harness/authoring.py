"""Building a world from a contract, and reporting where each clause landed.

The creator here differs from the old one in exactly two ways, and both exist to
make forgetting a requirement impossible rather than merely unlikely.

It is handed a `PromptSpec` instead of prose. The brief is still shown, for tone
and context, but the thing it must satisfy is an enumerated list of ids, and the
same list is what the verifier will check. There is no longer a private reading
of the brief that only the creator holds.

And it must say where every requirement went. A mapping from `R3` to
`corners[4]` is cheap to produce and expensive to fake: the verifier measures
that corner, so a claim that does not survive contact with the geometry becomes a
named failure attached to the words the user actually wrote. Requiring the
mapping also makes omission visible at authoring time — a requirement with no
entry is one the creator never placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .conversation import conversation_context
from .context_loader import load_context_pack
from .prompt_spec import PromptSpec, RequirementImplementation
from .providers import ProviderError, ProviderUsage, active_provider, anthropic_json, configured_model
from .racing import _coerce_plan_payload
from .models import TrackRegion
from .track_grammar import TrackPlan, track_plan_schema


@dataclass
class AuthoredPlan:
    """One creator proposal, plus its account of where each requirement landed."""

    plan: TrackPlan
    mapping: list[RequirementImplementation] = field(default_factory=list)
    provider: str = "offline"
    model: str = "track-grammar-v1"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    def unmapped(self, spec: PromptSpec) -> list[str]:
        """Requirement ids the creator never claimed to have implemented.

        Reported rather than trusted. A creator that silently drops a requirement
        usually drops it from the mapping too, which makes this the cheapest
        possible detector — no geometry has to be measured to notice it.
        """
        claimed = {item.id for item in self.mapping}
        return [item.id for item in spec.requirements if item.id not in claimed]


_AUTHORING_SYSTEM = load_context_pack("environment-faithful")


_COLOUR = {
    "type": "string",
    "description": "A #rrggbb hex colour, or a plain colour word like 'black' or 'neon pink'. "
                   "Empty string means leave it at the surface default.",
}

_VISUAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "How the circuit looks. Purely cosmetic — nothing here changes handling, lap "
        "times, or collision. Set only the fields the brief actually asks about and "
        "leave the rest as empty strings; an unrequested repaint is still an unrequested "
        "change to a world someone described."
    ),
    "properties": {
        "theme": {"type": "string", "maxLength": 60,
                  "description": "A short name for the look, e.g. 'cyberpunk night'."},
        "road": _COLOUR, "terrain": _COLOUR, "barrier": _COLOUR,
        "player_car": _COLOUR, "opponent_car": _COLOUR,
        "kerbs": {"type": "boolean",
                  "description": "The red-and-white edge striping. Set false if the brief "
                                 "asks for no edge lines or no kerbs."},
        "kerb_light": _COLOUR, "kerb_dark": _COLOUR, "sky": _COLOUR,
        "scenery": {
            "type": "array",
            "description": (
                "Coloured bands of ground crossing the map, passing under the road. This "
                "is how a river, a sand trap, or a painted run-off is represented: terrain "
                "of a different colour in a named place. The car drives over it normally."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "maxLength": 40},
                    "color": _COLOUR,
                    "region": {"type": "string", "enum": [item.value for item in TrackRegion]},
                    "orientation": {"type": "string", "enum": ["horizontal", "vertical"]},
                    "width_pixels": {"type": "number"},
                },
                "required": ["label", "color", "region", "orientation", "width_pixels"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "theme", "road", "terrain", "barrier", "player_car", "opponent_car",
        "kerbs", "kerb_light", "kerb_dark", "sky", "scenery",
    ],
    "additionalProperties": False,
}


def _authoring_schema() -> dict[str, Any]:
    schema = track_plan_schema()
    schema = {**schema, "properties": {**schema["properties"]}}
    schema["properties"]["visual"] = _VISUAL_SCHEMA
    schema["required"] = [*schema["required"], "visual"]
    schema["properties"]["requirement_mapping"] = {
        "type": "array",
        "description": "One entry per requirement id, naming where it was implemented.",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "location": {
                    "type": "string", "maxLength": 120,
                    "description": "Path into this plan: corners[4], npcs[0].profile, grip.",
                },
                "note": {
                    "type": "string", "maxLength": 200,
                    "description": "How it was implemented, or why it could not be.",
                },
            },
            "required": ["id", "location", "note"],
            "additionalProperties": False,
        },
    }
    schema["required"] = [*schema["required"], "requirement_mapping"]
    return schema


def author(
    spec: PromptSpec, provider: str = "auto", feedback: str | None = None,
    repair: str | None = None, precedents: str | None = None,
    conversation: list[dict[str, Any]] | None = None,
) -> AuthoredPlan:
    """Author one plan against a contract.

    `feedback` is a hard compiler rejection — the plan could not become geometry.
    `repair` is a fidelity miss — the plan compiled fine but the world it made
    does not satisfy specific requirement ids. They are different failures and
    they are kept separate: the first says "this is not buildable", the second
    says "this is not what was asked for", and collapsing them into one channel
    was part of how a merely-unfaithful circuit got treated as a success.

    `precedents` is what the same person confirmed on earlier circuits. It is
    evidence about what their words tend to mean, not an instruction, and it is
    empty whenever nothing relevant has been confirmed — which is most of the time.
    """
    resolved = active_provider() if provider == "auto" else provider
    if resolved == "offline":
        from .track_grammar import parse_track_prompt

        return AuthoredPlan(plan=parse_track_prompt(spec.prompt))
    if resolved not in {"anthropic", "openai"}:
        raise ProviderError(f"Unknown authoring provider: {resolved}")

    context = conversation_context(conversation)
    payload, usage = anthropic_json(
        model=configured_model("ANTHROPIC_ENVIRONMENT_MODEL"),
        max_tokens=3_000,
        system=_AUTHORING_SYSTEM,
        prompt=(
            (
                "Recent conversation, for resolving references and retaining stated constraints. "
                "The user brief and requirement ledger below are authoritative:\n"
                f"{context}\n\n"
                if context else ""
            )
            + f"The user asked for:\n{spec.prompt}\n\n"
            + (f"In short: {spec.summary}\n\n" if spec.summary else "")
            + spec.briefing()
            + "\n\nAuthor the track plan that satisfies this contract, and return a "
              "requirement_mapping entry for every id above. If the contract asks for a literal "
              "circle, set loop_shape=circle and corners=[]; do not approximate it with broad "
              "corners. Otherwise give every located corner an explicit angle_degrees and region. "
              "Leave angle_degrees omitted with region=auto on genuinely unspecified linking "
              "corners so the loop closes."
            + (
                "\n\nYour previous plan was REJECTED BY THE COMPILER and never became a "
                f"circuit:\n{feedback}\n"
                "Author a plan that still satisfies the contract but can be built. Opening "
                "tight radii, shortening straights, spreading corners into different regions, "
                "and removing barriers all make a circuit easier to close."
                if feedback else ""
            )
            + (f"\n\n{repair}" if repair else "")
            + (f"\n\n{precedents}" if precedents else "")
        ),
        json_schema=_authoring_schema(),
        cache_system=True,
    )
    try:
        plan = TrackPlan.model_validate(_coerce_plan_payload(payload))
    except Exception as error:
        raise ProviderError(
            f"Racing creator returned an invalid track plan: {str(error)[:360]}"
        ) from error
    mapping = [
        RequirementImplementation(
            id=str(item.get("id") or "").strip().upper(),
            location=str(item.get("location") or "")[:120],
            note=str(item.get("note") or "")[:200],
        )
        for item in (payload.get("requirement_mapping") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    return AuthoredPlan(
        plan=plan, mapping=mapping, provider=usage.provider, model=usage.model,
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        latency_ms=usage.latency_ms,
    )

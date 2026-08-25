"""Load typed, role-bounded Markdown cards into cacheable model prompts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


CONTEXT_ROOT = Path(__file__).with_name("context")
MANIFEST_PATH = CONTEXT_ROOT / "manifest.json"
LESSON_ROOT = CONTEXT_ROOT / "player" / "lessons"

_PACK_AUDIENCE = {
    "environment": "environment",
    "environment-faithful": "environment",
    "player-2d": "player",
    "player-3d": "player",
}
_PACK_DIMENSION = {"player-2d": "2d", "player-3d": "3d"}
_ROLE_ROOTS = {
    "environment": frozenset({"agents", "contracts", "environment"}),
    "environment-faithful": frozenset({"agents", "contracts", "environment"}),
    "player-2d": frozenset({"agents", "contracts", "player"}),
    "player-3d": frozenset({"agents", "contracts", "player"}),
}
_PLAYER_ALLOWED_INPUTS = frozenset({
    "camera_frame", "scalar_speed", "image_features", "recent_controls",
    "active_skill", "timing",
})
_LESSON_ALLOWED_EVIDENCE = frozenset({
    "camera_frame", "scalar_speed", "image_features", "skill_history",
})
_SKILLS = frozenset({
    "follow_lane", "prepare_turn", "take_turn", "take_hairpin",
    "recover_track", "stabilize",
})


class ContextPackError(ValueError):
    """A runtime context card crosses a role boundary or is malformed."""


@dataclass(frozen=True)
class ContextCard:
    relative_path: str
    metadata: dict[str, Any]
    body: str
    raw: str

    @property
    def id(self) -> str:
        return str(self.metadata["id"])


def _json_frontmatter(raw: str, relative_path: str) -> tuple[dict[str, Any], str]:
    """Parse the small JSON-compatible YAML subset used by context cards."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ContextPackError(f"context card lacks frontmatter: {relative_path}")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as error:
        raise ContextPackError(f"context card has unterminated frontmatter: {relative_path}") from error
    metadata: dict[str, Any] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise ContextPackError(f"invalid frontmatter line in {relative_path}: {line!r}")
        try:
            metadata[key.strip()] = json.loads(value.strip())
        except json.JSONDecodeError as error:
            raise ContextPackError(
                f"frontmatter values must be valid JSON in {relative_path}: {line!r}"
            ) from error
    body = "\n".join(lines[end + 1:]).strip()
    return metadata, body


def _read_card(relative_path: str, root: Path = CONTEXT_ROOT) -> ContextCard:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".md":
        raise ContextPackError(f"unsafe context path: {relative_path!r}")
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve()
    if not target.is_relative_to(resolved_root) or not target.is_file():
        raise ContextPackError(f"missing context file: {relative_path!r}")
    raw = target.read_text(encoding="utf-8").strip()
    metadata, body = _json_frontmatter(raw, relative.as_posix())
    required = {"id", "audience", "kind", "load", "dimensions", "requires"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ContextPackError(f"context card {relative_path!r} lacks {', '.join(missing)}")
    if not body:
        raise ContextPackError(f"context card {relative_path!r} has no instructions")
    if not isinstance(metadata["audience"], list) or not metadata["audience"]:
        raise ContextPackError(f"context card {relative_path!r} needs an audience list")
    if not isinstance(metadata["dimensions"], list) or not metadata["dimensions"]:
        raise ContextPackError(f"context card {relative_path!r} needs dimensions")
    if not isinstance(metadata["requires"], list):
        raise ContextPackError(f"context card {relative_path!r} needs a requires list")
    return ContextCard(relative.as_posix(), metadata, body, raw)


def _validate_card_for_pack(card: ContextCard, pack: str) -> None:
    audience = _PACK_AUDIENCE[pack]
    if audience not in card.metadata["audience"]:
        raise ContextPackError(f"{pack!r} cannot load {card.id!r} for another audience")
    dimension = _PACK_DIMENSION.get(pack)
    if dimension and dimension not in card.metadata["dimensions"]:
        raise ContextPackError(f"{pack!r} cannot load {card.id!r} for another dimension")
    if audience == "player":
        unknown = set(card.metadata["requires"]) - _PLAYER_ALLOWED_INPUTS
        if unknown:
            raise ContextPackError(
                f"player context {card.id!r} requires forbidden inputs: {', '.join(sorted(unknown))}"
            )
    if card.metadata["load"] == "failure":
        raise ContextPackError(f"failure lesson {card.id!r} cannot be statically manifested")


def _validate_manifest(payload: Any, root: Path = CONTEXT_ROOT) -> dict[str, tuple[str, ...]]:
    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise ContextPackError("context manifest must use version 2")
    packs = payload.get("packs")
    if not isinstance(packs, dict) or set(packs) != set(_ROLE_ROOTS):
        raise ContextPackError("context manifest must define every known role pack exactly once")

    root = root.resolve()
    validated: dict[str, tuple[str, ...]] = {}
    for pack, entries in packs.items():
        if not isinstance(entries, list) or not entries:
            raise ContextPackError(f"context pack {pack!r} must contain files")
        files: list[str] = []
        ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, str):
                raise ContextPackError(f"context pack {pack!r} contains a non-string path")
            relative = Path(entry)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".md":
                raise ContextPackError(f"unsafe context path in {pack!r}: {entry!r}")
            if not relative.parts or relative.parts[0] not in _ROLE_ROOTS[pack]:
                raise ContextPackError(f"{pack!r} cannot load context file {entry!r}")
            card = _read_card(relative.as_posix(), root)
            _validate_card_for_pack(card, pack)
            if card.id in ids:
                raise ContextPackError(f"duplicate context id in {pack!r}: {card.id!r}")
            ids.add(card.id)
            files.append(relative.as_posix())
        validated[pack] = tuple(files)
    return validated


@lru_cache(maxsize=1)
def context_manifest() -> dict[str, tuple[str, ...]]:
    return _validate_manifest(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


@lru_cache(maxsize=None)
def _pack_cards(name: str) -> tuple[ContextCard, ...]:
    try:
        files = context_manifest()[name]
    except KeyError as error:
        raise ContextPackError(f"unknown context pack: {name!r}") from error
    return tuple(_read_card(relative) for relative in files)


@lru_cache(maxsize=None)
def _pack_body(name: str) -> str:
    return "\n\n".join(
        f"<!-- runtime-context: {card.relative_path} id={card.id} -->\n{card.body}"
        for card in _pack_cards(name)
    )


@lru_cache(maxsize=None)
def context_pack_version(name: str) -> str:
    material = "\n\n".join(
        f"{card.relative_path}\n{card.raw}" for card in _pack_cards(name)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=None)
def load_context_pack(name: str) -> str:
    return f"RACELAB CONTEXT PACK {name}@{context_pack_version(name)}\n\n{_pack_body(name)}"


def context_pack_provenance(name: str) -> dict[str, Any]:
    cards = _pack_cards(name)
    return {
        "name": name,
        "version": context_pack_version(name),
        "files": [card.relative_path for card in cards],
        "cards": [card.id for card in cards],
    }


def _confirmed_lessons(
    dimension: str, skills: Iterable[str] | None = None,
    *, lesson_root: Path | None = None, context_root: Path | None = None,
) -> list[ContextCard]:
    """Return bounded, explicitly confirmed lessons grounded in player-visible evidence."""
    if dimension not in {"2d", "3d"}:
        raise ContextPackError(f"unknown player dimension: {dimension!r}")
    wanted = set(skills or _SKILLS)
    unknown = wanted - _SKILLS
    if unknown:
        raise ContextPackError(f"unknown skill lesson requested: {', '.join(sorted(unknown))}")
    lessons: list[ContextCard] = []
    lesson_root = lesson_root or LESSON_ROOT
    context_root = context_root or CONTEXT_ROOT
    if not lesson_root.exists():
        return lessons
    for target in sorted(lesson_root.glob("*/*.md")):
        relative = target.relative_to(context_root).as_posix()
        card = _read_card(relative, context_root)
        data = card.metadata
        if data.get("kind") != "failure-lesson" or data.get("load") != "failure":
            raise ContextPackError(f"lesson card has wrong kind or load policy: {relative}")
        if data.get("status") != "confirmed":
            continue
        skill = data.get("skill")
        if skill not in wanted or dimension not in data.get("dimensions", []):
            continue
        if "player" not in data.get("audience", []):
            raise ContextPackError(f"lesson {card.id!r} has a non-player audience")
        evidence = data.get("evidence")
        if not isinstance(evidence, list) or set(evidence) - _LESSON_ALLOWED_EVIDENCE:
            raise ContextPackError(f"lesson {card.id!r} cites forbidden evidence")
        if set(data.get("requires", [])) - _LESSON_ALLOWED_EVIDENCE:
            raise ContextPackError(f"lesson {card.id!r} requires forbidden inputs")
        observations = data.get("observations")
        if not isinstance(observations, int) or observations < 2:
            raise ContextPackError(f"lesson {card.id!r} needs at least two observations")
        lessons.append(card)
    lessons.sort(key=lambda card: (
        -int(card.metadata.get("priority", 0)),
        -int(card.metadata.get("observations", 0)),
        card.id,
    ))
    return lessons[:4]


def load_player_context(dimension: str) -> str:
    name = f"player-{dimension}"
    base = load_context_pack(name)
    lessons = _confirmed_lessons(dimension)
    if not lessons:
        return base
    overlay = "\n\n".join(
        f"<!-- confirmed-skill-lesson: {card.relative_path} id={card.id} -->\n{card.body}"
        for card in lessons
    )
    return base + "\n\nCONFIRMED SKILL FAILURE LESSONS\n\n" + overlay


def player_context_provenance(dimension: str) -> dict[str, Any]:
    base = context_pack_provenance(f"player-{dimension}")
    lessons = _confirmed_lessons(dimension)
    return {
        **base,
        "lessons": [{
            "id": card.id,
            "file": card.relative_path,
            "skill": card.metadata["skill"],
            "observations": card.metadata["observations"],
        } for card in lessons],
    }

"""Runtime context is explicit, cacheable, and cannot cross agent roles."""

from __future__ import annotations

import json

import pytest

from harness import providers
from harness.authoring import _AUTHORING_SYSTEM
from harness.context_loader import (
    CONTEXT_ROOT,
    ContextPackError,
    _confirmed_lessons,
    _read_card,
    _validate_manifest,
    context_manifest,
    context_pack_provenance,
    load_context_pack,
    load_player_context,
    player_context_provenance,
)
from harness.providers import ProviderUsage
from harness.racing import RACING_CREATOR_SYSTEM


def test_every_manifest_pack_loads_with_stable_provenance() -> None:
    for name, files in context_manifest().items():
        prompt = load_context_pack(name)
        provenance = context_pack_provenance(name)
        assert prompt.startswith(f"RACELAB CONTEXT PACK {name}@")
        assert provenance["name"] == name
        assert provenance["version"] in prompt.splitlines()[0]
        assert provenance["files"] == list(files)
        assert len(provenance["cards"]) == len(files)
        assert all(f"runtime-context: {path}" in prompt for path in files)


def test_player_context_can_only_load_shared_and_player_files() -> None:
    for name in ("player-2d", "player-3d"):
        assert all(path.startswith(("agents/", "contracts/", "player/"))
                   for path in context_manifest()[name])
        for path in context_manifest()[name]:
            card = _read_card(path)
            assert "player" in card.metadata["audience"]
            assert set(card.metadata["requires"]) <= {
                "camera_frame", "scalar_speed", "image_features", "recent_controls",
                "active_skill", "timing",
            }


def test_manifest_rejects_cross_role_and_traversal_paths() -> None:
    payload = json.loads((CONTEXT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    payload["packs"]["player-2d"].append("environment/capabilities/track-grammar.md")
    with pytest.raises(ContextPackError, match="cannot load"):
        _validate_manifest(payload, CONTEXT_ROOT)

    payload = json.loads((CONTEXT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    payload["packs"]["player-2d"].append("../secret.md")
    with pytest.raises(ContextPackError, match="unsafe context path"):
        _validate_manifest(payload, CONTEXT_ROOT)

    payload = json.loads((CONTEXT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    payload["packs"]["player-2d"].append("agents/environment-agent.md")
    with pytest.raises(ContextPackError, match="another audience"):
        _validate_manifest(payload, CONTEXT_ROOT)


def test_environment_entry_points_use_their_runtime_packs() -> None:
    assert RACING_CREATOR_SYSTEM == load_context_pack("environment")
    assert _AUTHORING_SYSTEM == load_context_pack("environment-faithful")


def test_environment_track_grammar_context_explains_round_loop_authoring() -> None:
    """Named forms are model guidance, not a special case hidden in the compiler."""
    assert "A circular or round loop is a geometry request" in _AUTHORING_SYSTEM
    assert "radius=sweeping" in _AUTHORING_SYSTEM


def test_predictive_player_calls_use_view_specific_cached_context(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_json(**kwargs):
        calls.append(kwargs)
        usage = ProviderUsage(provider="test", model="test")
        if "center_near" in kwargs["json_schema"]["properties"]["predicted"]["properties"]:
            return ({
                "predicted": {"speed": 1.0, "center_near": 0.0, "turn_ahead": 0.0,
                              "road_contact": True},
                "skill": "follow_lane", "target_speed": 1.6, "target_offset": 0.0,
                "turn_direction": 0, "speed_tolerance": 0.5, "lateral_tolerance": 0.5,
                "summary": "follow visible lane",
            }, usage)
        return ({
            "predicted": {"speed": 4.0, "road_offset": 0.0, "bend_ahead": 0.0,
                          "road_contact": True},
            "skill": "follow_lane", "target_speed": 6.0, "target_offset": 0.0,
            "turn_direction": 0, "speed_tolerance": 1.0, "offset_tolerance": 0.5,
            "bend_tolerance": 0.5, "summary": "follow visible lane",
        }, usage)

    monkeypatch.setattr(providers, "anthropic_json", fake_json)
    providers.plan_cone_driving_skill(
        object(), public_state={"speed": 1.0, "center_near": 0.0, "turn_ahead": 0.0,
                                "road_contact": True},
        active_skill={}, recent_controls=[], activation_horizon_ticks=5, control_hz=10,
    )
    providers.plan_predictive_driving_skill(
        object(), public_state={"speed": 4.0, "road_offset": 0.0, "bend_ahead": 0.0,
                                "road_contact": True},
        active_skill={}, previous_controls=[], activation_horizon_ticks=5, control_hz=10,
    )

    assert calls[0]["system"] == load_player_context("2d")
    assert calls[1]["system"] == load_player_context("3d")
    assert all(call["cache_system"] is True for call in calls)


def test_only_confirmed_repeated_camera_grounded_lessons_load(tmp_path) -> None:
    lesson_root = tmp_path / "player" / "lessons"
    confirmed = lesson_root / "take-turn" / "confirmed.md"
    pending = lesson_root / "take-turn" / "pending.md"
    confirmed.parent.mkdir(parents=True)
    confirmed.write_text(
        """---
id: "player.lesson.take-turn.late-entry"
audience: ["player"]
kind: "failure-lesson"
load: "failure"
status: "confirmed"
skill: "take_turn"
dimensions: ["3d"]
requires: ["camera_frame", "scalar_speed", "image_features", "skill_history"]
evidence: ["camera_frame", "scalar_speed", "image_features", "skill_history"]
observations: 3
priority: 50
---
# Late visible entry

Reconsider entry speed when the visible road contracts before turn-in.
""",
        encoding="utf-8",
    )
    pending.write_text(
        confirmed.read_text(encoding="utf-8")
        .replace("late-entry", "one-off")
        .replace('status: "confirmed"', 'status: "pending"'),
        encoding="utf-8",
    )
    lessons = _confirmed_lessons(
        "3d", lesson_root=lesson_root, context_root=tmp_path,
    )
    assert [lesson.id for lesson in lessons] == ["player.lesson.take-turn.late-entry"]
    assert _confirmed_lessons("2d", lesson_root=lesson_root, context_root=tmp_path) == []


def test_confirmed_lesson_rejects_hidden_evidence(tmp_path) -> None:
    lesson_root = tmp_path / "player" / "lessons" / "take-turn"
    lesson_root.mkdir(parents=True)
    (lesson_root / "leak.md").write_text(
        """---
id: "player.lesson.take-turn.leak"
audience: ["player"]
kind: "failure-lesson"
load: "failure"
status: "confirmed"
skill: "take_turn"
dimensions: ["3d"]
requires: ["camera_frame"]
evidence: ["camera_frame", "centerline"]
observations: 2
---
# Leaking lesson

This must never enter the player prompt.
""",
        encoding="utf-8",
    )
    with pytest.raises(ContextPackError, match="forbidden evidence"):
        _confirmed_lessons("3d", lesson_root=lesson_root.parent, context_root=tmp_path)


def test_player_provenance_reports_no_lessons_until_one_is_confirmed() -> None:
    for dimension in ("2d", "3d"):
        provenance = player_context_provenance(dimension)
        assert provenance["lessons"] == []
        assert "CONFIRMED SKILL FAILURE LESSONS" not in load_player_context(dimension)

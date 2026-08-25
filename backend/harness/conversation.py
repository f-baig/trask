"""Bounded, role-labelled conversation context for model-backed stages."""

from __future__ import annotations

from typing import Any


def conversation_context(history: list[dict[str, Any]] | None) -> str:
    """Render recent dialogue as context, never as an implicit replacement brief."""
    turns: list[str] = []
    for turn in history or []:
        role = str(turn.get("role") or "").lower()
        content = str(turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = "User" if role == "user" else "Assistant"
        turns.append(f"{label}: {content[:2_000]}")
    return "\n".join(turns)

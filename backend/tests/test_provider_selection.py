"""Provider-neutral model selection for the environment and player agents."""

from __future__ import annotations

import pytest

from harness import providers, racing
from harness.providers import ProviderUsage
from harness.track_grammar import parse_track_prompt


@pytest.mark.parametrize(
    ("key_name", "provider", "expected_model", "stale_model"),
    [
        ("OPENAI_API_KEY", "openai", "gpt-5.6-luna", "claude-sonnet-5"),
        ("ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-5", "gpt-5.6-luna"),
    ],
)
def test_one_provider_key_selects_matching_models_for_every_role(
    monkeypatch: pytest.MonkeyPatch, key_name: str, provider: str,
    expected_model: str, stale_model: str,
) -> None:
    """A prior provider's model setting cannot strand one of the two agents."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(key_name, "test-key")
    monkeypatch.setenv("RACING_MODEL", stale_model)

    assert providers.active_provider() == provider
    prefix = "gpt-" if provider == "openai" else "claude-"
    assert providers.configured_model("ANTHROPIC_ENVIRONMENT_MODEL") == expected_model
    assert providers.configured_model("ANTHROPIC_PLAYER_MODEL") == expected_model
    assert providers.integration_model().startswith(prefix)
    assert providers._chat_model("main").startswith(prefix)
    assert providers._chat_model("environment").startswith(prefix)


@pytest.mark.parametrize(
    ("key_name", "provider", "expected_model"),
    [
        ("OPENAI_API_KEY", "openai", "gpt-5.6-luna"),
        ("ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-5"),
    ],
)
def test_environment_author_and_player_planner_use_the_same_active_provider(
    monkeypatch: pytest.MonkeyPatch, key_name: str, provider: str, expected_model: str,
) -> None:
    """Exercise both call sites without making a network request."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(key_name, "test-key")
    plan_payload = parse_track_prompt("a simple circuit").model_dump()
    calls: list[str] = []

    def fake_json(**kwargs):
        calls.append(kwargs["model"])
        if "actions" in kwargs["json_schema"].get("properties", {}):
            return {
                "subgoal": "hold the visible road", "summary": "drive forward briefly",
                "confidence": .6,
                "actions": [{"action": "forward", "keys": ["w"], "steps": 1}],
            }, ProviderUsage(provider=provider, model=kwargs["model"])
        return plan_payload, ProviderUsage(provider=provider, model=kwargs["model"])

    monkeypatch.setattr(racing, "anthropic_json", fake_json)
    monkeypatch.setattr(providers, "anthropic_json", fake_json)

    environment = racing.design_racing_environment("a simple circuit")
    player, usage = providers.plan_racing_actions({})

    assert environment.provider == provider
    assert player.actions[0].action == "forward"
    assert usage.provider == provider
    assert calls[0] == expected_model
    prefix = "gpt-" if provider == "openai" else "claude-"
    assert all(model.startswith(prefix) for model in calls)

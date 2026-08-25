"""What a brief asks for, and whether the compiled circuit says so.

The compiler always produces a valid circuit, which is what makes silent infidelity possible:
a request for something outside the grammar does not fail, it produces a circuit that ignores
that part and gets named after it. From the outside that is indistinguishable from the prompt
being discarded at random, and that is exactly how it was reported. These tests pin the three
places that were actually losing intent.
"""

from __future__ import annotations

import pytest

from harness.models import ElevationProfile
from harness.service import HarnessService
from harness.store import HarnessStore
from harness.track3d import parse_elevation_prompt
from harness.track_grammar import unsupported_requests


@pytest.mark.parametrize(("prompt", "profile"), [
    ("an elevated loop circuit", ElevationProfile.HILLY),
    ("a hilly track with banked corners", ElevationProfile.HILLY),
    ("alpine mountain pass with steep climbs", ElevationProfile.ALPINE),
    ("gentle rolling undulations", ElevationProfile.ROLLING),
    ("a track with a big gradient", ElevationProfile.ALPINE),
])
def test_elevation_is_read_from_the_brief(prompt: str, profile: ElevationProfile) -> None:
    """The grammar read surface, corners, laps and opponents but never the vertical profile.

    So "an elevated loop with banked corners" compiled a flat circuit, which is the single most
    visible way the harness looked like it was ignoring its input.
    """
    spec = parse_elevation_prompt(prompt)
    assert spec is not None and spec.profile is profile


@pytest.mark.parametrize("prompt", [
    "a flat technical circuit with two barriers",
    "slippery ice track with a 90 degree bend in the top right",
    "three aggressive npcs on a narrow circuit",
])
def test_a_planar_brief_stays_planar(prompt: str) -> None:
    """Elevation is opt-in: nothing about corners or grip should quietly add hills."""
    assert parse_elevation_prompt(prompt) is None


def test_banking_is_only_applied_when_asked_for() -> None:
    """A cross-slope changes the cornering limit, so it is not a default garnish on hills."""
    assert parse_elevation_prompt("a hilly circuit").banking_degrees < 10
    assert parse_elevation_prompt("a hilly circuit with banked corners").banking_degrees > 10


@pytest.mark.parametrize(("prompt", "expected"), [
    ("a circuit with a pit lane and refuelling", "pit stops"),
    ("a track with jumps and a tunnel", "jumps"),
    ("a race with headlights casting long shadows", "lighting"),
    ("a circuit with a grandstand full of spectators", "spectators"),
])
def test_requests_outside_the_grammar_are_named(prompt: str, expected: str) -> None:
    """Naming a real gap is the whole difference between honest and random."""
    assert any(expected in item for item in unsupported_requests(prompt))


@pytest.mark.parametrize("prompt", [
    "black cars on a red track",
    "a purple circuit with blue barriers and no red and white edge lines",
    "a white player car and green grass",
    "a river running under the back straight",
])
def test_appearance_requests_are_no_longer_called_impossible(prompt: str) -> None:
    """Colour became a real dial, so reporting it as out of grammar became a lie.

    This list used to match every colour word. It fired on any brief mentioning blue and
    told the user a repaint could not be represented — immediately after the verifier had
    measured that it had been. Claiming a limitation the harness does not have is worse
    than claiming a capability it does not have: it turns working features off.
    """
    assert unsupported_requests(prompt) == []


@pytest.mark.parametrize("prompt", [
    "slippery ice track with a 90 degree bend in the top right",
    "three aggressive npcs, narrow corridor, four laps",
    "a hilly circuit with banked corners",
])
def test_what_the_grammar_does_read_is_not_flagged(prompt: str) -> None:
    """Surface, corners, opponents, width, laps and elevation are all real fields."""
    assert unsupported_requests(prompt) == []


def test_a_brief_asking_for_hills_compiles_a_3d_circuit(tmp_path) -> None:
    """No switch involved: the words are enough, which is what a caller expects."""
    service = HarnessService(store=HarnessStore(tmp_path))
    environment = service.create_environment(
        "an elevated hilly circuit with two barriers", seed=17, provider="offline",
    )
    assert environment.scene.elevation is not None
    assert not environment.scene.elevation.is_flat
    assert service.runtime.create(environment.scene).__class__.__name__ == "Racing3DWorld"
    assert environment.playability_certificate and environment.playability_certificate.playable, (
        "an elevated circuit is certified over its gradients, not just flat"
    )


def test_fidelity_is_graded_against_the_request_not_a_rewritten_brief(tmp_path) -> None:
    """The coordinator rewrites a request into a brief, and the brief invents specifics.

    Grading against it reported the coordinator's inventions as though the user had asked for
    them — "4 laps (got 1)" for a request that never mentioned laps. Fidelity reads the user's
    words instead.
    """
    service = HarnessService(store=HarnessStore(tmp_path))
    steps: list[tuple[str, str]] = []
    service.create_environment(
        # The brief the creator sees claims four laps; the user asked for two.
        "A four lap contest on ice.\n\nOriginal request, verbatim: \"ice track, 2 laps\"",
        seed=17, provider="offline", intent_prompt="ice track, 2 laps",
        on_step=lambda stage, detail: steps.append((stage, detail)),
    )
    fidelity = " ".join(detail for stage, detail in steps if stage == "fidelity")
    assert "2 laps" in fidelity
    assert "4 laps" not in fidelity, "the brief's invented lap count is not the user's request"


def test_the_fidelity_report_names_what_landed_and_what_did_not(tmp_path) -> None:
    service = HarnessService(store=HarnessStore(tmp_path))
    environment = service.create_environment(
        "slippery ice circuit, 2 laps, three aggressive npcs", seed=17, provider="offline",
    )
    lines = service.fidelity_report("slippery ice circuit, 2 laps, three aggressive npcs", environment.scene)
    joined = " ".join(lines)
    assert "Honoured:" in joined
    assert "ice" in joined and "2 laps" in joined and "3 opponents" in joined


def test_the_coordinator_carries_the_request_verbatim(tmp_path, monkeypatch) -> None:
    """Every downstream reader keys on exact phrases, so the original text has to survive.

    A rewritten brief is free to drop "ice", "90 degree", or "top right" — the words
    comprehension, the elevation reader and the fidelity verifier all depend on. The
    coordinator used to paraphrase the request and append the original underneath it; now
    it does not paraphrase at all, and the request reaches the compiler untouched. This
    asserts the guarantee rather than the wrapper that used to carry it, and it also pins
    `intent_prompt`, which is what comprehension and fidelity actually read.
    """
    request = "slippery ice track with a 90 degree bend in the top right"
    service = HarnessService(store=HarnessStore(tmp_path))
    seen: list[tuple[str, str | None]] = []
    original = service.create_environment

    def spy(prompt, *args, **kwargs):
        seen.append((prompt, kwargs.get("intent_prompt")))
        return original(prompt, *args, **kwargs)

    monkeypatch.setattr(service, "create_environment", spy)
    service.dispatch_coordinator(request)
    assert seen, "the coordinator has to compile something"
    compiled_prompt, intent = seen[0]
    assert compiled_prompt == request, "the request must reach the compiler unrewritten"
    assert intent == request, "fidelity must be graded against the user's own words"

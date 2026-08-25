import pytest
from pydantic import ValidationError

from harness.interventions import resolve_fork_condition
from harness.models import ForkRequest


@pytest.mark.parametrize(
    ("condition", "perturbation", "guidance"),
    [
        ("Make the grip much lower from here", "low_grip", None),
        ("Add steering delay", "action_delay", None),
        ("Move the barriers", "obstacle_shift", None),
        ("Replay exactly", "none", None),
        ("Brake earlier and hold the inside line", "none", "Brake earlier and hold the inside line"),
        (
            "Lower grip and tell the driver to brake earlier",
            "low_grip",
            "Lower grip and tell the driver to brake earlier",
        ),
        (
            "Keep physics unchanged but brake earlier",
            "none",
            "Keep physics unchanged but brake earlier",
        ),
    ],
)
def test_natural_language_fork_conditions_are_deterministic(
    condition: str, perturbation: str, guidance: str | None,
) -> None:
    resolved = resolve_fork_condition(condition)
    assert resolved.perturbation == perturbation
    assert resolved.guidance == guidance


def test_a_fork_condition_can_select_only_one_engine_change() -> None:
    with pytest.raises(ValueError, match="one engine perturbation"):
        resolve_fork_condition("Add control delay and make the track slippery")


def test_fog_is_rejected_instead_of_becoming_inert_guidance() -> None:
    with pytest.raises(ValueError, match="Fog is not an available condition"):
        resolve_fork_condition("Add fog from here")


def test_natural_and_structured_fork_contracts_cannot_be_mixed() -> None:
    with pytest.raises(ValidationError, match="either a natural-language condition"):
        ForkRequest(fork_step=10, condition="Add low grip", perturbation="low_grip")

import math

import pytest

from harness.models import TrackDrawingCreate, Vec2
from harness.service import HarnessService, default_vision_policy
from harness.store import HarnessStore


def oval_points(count: int = 80) -> list[Vec2]:
    return [
        Vec2(
            x=.5 + .42 * math.cos(index * math.tau / count),
            y=.5 + .34 * math.sin(index * math.tau / count),
        )
        for index in range(count)
    ]


def rectangle_points() -> list[Vec2]:
    """A representative freehand loop with visually square corners."""
    return [
        Vec2(x=x, y=y) for x, y in (
            (.15, .15), (.35, .15), (.65, .15), (.85, .15),
            (.85, .32), (.85, .5), (.85, .68), (.85, .85),
            (.65, .85), (.35, .85), (.15, .85), (.15, .68),
            (.15, .5), (.15, .32),
        )
    ]


def test_saved_drawing_compiles_through_validation_and_replay(tmp_path) -> None:
    service = HarnessService(store=HarnessStore(tmp_path))
    drawing = service.create_drawing(TrackDrawingCreate(name="Test Oval", points=oval_points()))

    environment = service.create_environment_from_drawing(
        drawing, f"use /{drawing.id} with two laps and edge barriers",
    )

    assert service.store.get_drawing(drawing.id) == drawing
    assert environment.origin == f"drawing:{drawing.id}"
    assert environment.scene.name == "Test Oval"
    assert environment.scene.laps == 2
    assert environment.scene.edge_barriers
    assert environment.validation == ["Racing domain contract passed."]
    assert environment.playability_certificate and environment.playability_certificate.playable
    assert default_vision_policy(environment.scene) == "vision-2d-predictive-skills"


def test_freehand_rectangle_is_smoothed_into_a_playable_circuit(tmp_path) -> None:
    service = HarnessService(store=HarnessStore(tmp_path))
    drawing = service.create_drawing(TrackDrawingCreate(
        name="Freehand rectangle", points=rectangle_points(),
    ))

    environment = service.create_environment_from_drawing(drawing, f"use /{drawing.id}")

    assert environment.playability_certificate and environment.playability_certificate.playable
    assert environment.scene.track_report.minimum_radius_pixels > environment.scene.track_width / 2


def test_drawing_compiler_rejects_a_crossing_route(tmp_path) -> None:
    service = HarnessService(store=HarnessStore(tmp_path))
    points = [
        Vec2(x=.1, y=.2), Vec2(x=.25, y=.1), Vec2(x=.5, y=.5), Vec2(x=.75, y=.1),
        Vec2(x=.9, y=.2), Vec2(x=.65, y=.5), Vec2(x=.9, y=.8), Vec2(x=.75, y=.9),
        Vec2(x=.5, y=.5), Vec2(x=.25, y=.9), Vec2(x=.1, y=.8), Vec2(x=.35, y=.5),
    ]
    drawing = service.create_drawing(TrackDrawingCreate(name="Crossing", points=points))

    with pytest.raises(ValueError, match="cannot become a playable circuit"):
        service.create_environment_from_drawing(drawing, f"use /{drawing.id}")


def test_delete_drawing_does_not_delete_compiled_environment(tmp_path) -> None:
    service = HarnessService(store=HarnessStore(tmp_path))
    drawing = service.create_drawing(TrackDrawingCreate(name="Persistent Oval", points=oval_points()))
    environment = service.create_environment_from_drawing(drawing, f"use /{drawing.id}")

    service.delete_drawing(drawing.id)

    assert service.store.get_drawing(drawing.id) is None
    assert service.get_environment(environment.id) is not None


def test_coordinator_use_reference_compiles_the_saved_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = HarnessService(store=HarnessStore(tmp_path))
    drawing = service.create_drawing(TrackDrawingCreate(name="Coordinator Oval", points=oval_points()))

    result = service.dispatch_coordinator(f"use /{drawing.id} with one lap")

    assert result.built
    environment = service.get_environment(result.environment_id)
    assert environment and environment.origin == f"drawing:{drawing.id}"
    assert environment.scene.name == "Coordinator Oval"
    assert service.list_runs(environment.id) == []


def test_coordinator_compiles_a_bare_drawing_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = HarnessService(store=HarnessStore(tmp_path))
    drawing = service.create_drawing(TrackDrawingCreate(name="Bare reference", points=oval_points()))

    result = service.dispatch_coordinator(f"/{drawing.id} with two laps")

    environment = service.get_environment(result.environment_id)
    assert environment and environment.scene.laps == 2
    replies = [message.content for message in service.agent_messages("main") if message.speaker == "assistant"]
    assert replies[-1].startswith(f"Using /{drawing.id} as the circuit centerline.")


def test_failed_drawing_compile_persists_the_geometry_reason(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = HarnessService(store=HarnessStore(tmp_path))
    drawing = service.create_drawing(TrackDrawingCreate(name="Crossing", points=[
        Vec2(x=.1, y=.2), Vec2(x=.25, y=.1), Vec2(x=.5, y=.5), Vec2(x=.75, y=.1),
        Vec2(x=.9, y=.2), Vec2(x=.65, y=.5), Vec2(x=.9, y=.8), Vec2(x=.75, y=.9),
        Vec2(x=.5, y=.5), Vec2(x=.25, y=.9), Vec2(x=.1, y=.8), Vec2(x=.35, y=.5),
    ]))

    with pytest.raises(ValueError, match=rf"Could not compile /{drawing.id}"):
        service.dispatch_coordinator(f"/{drawing.id}")

    replies = [message.content for message in service.agent_messages("main") if message.speaker == "assistant"]
    assert "couldn't compile" in replies[-1].lower()
    assert "Try redrawing" in replies[-1]

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import AgentMessageRequest, CreateEnvironmentRequest, ExperimentRequest, ForkRequest, RunRequest, StudyPanelUpdateRequest, TrackDrawingCreate
from .providers import ProviderError, active_provider, configured_model
from .service import HarnessService


service = HarnessService()

app = FastAPI(title="RaceLab Racing Harness", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def missing(error: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "racing-2d-and-3d"}


@app.get("/domain/search")
def domain_search(query: str) -> dict:
    return service.get_domain_context(query)


@app.get("/agents/{role}/messages")
def agent_messages(role: str, environment_id: str | None = None):
    try:
        return service.agent_messages(role, environment_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/agent-activity")
def agent_activity(limit: int = 40):
    return service.agent_activity(max(1, min(limit, 100)))


@app.post("/agents/{role}/messages")
def send_agent_message(role: str, request: AgentMessageRequest, environment_id: str | None = None):
    try:
        return service.send_agent_message(role, request.message, environment_id)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/coordinator/dispatch")
def coordinator_dispatch(request: AgentMessageRequest):
    try:
        return service.dispatch_coordinator(
            request.message, dimensions=request.dimensions, elevation=request.elevation,
        )
    except (ProviderError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def ndjson(events) -> StreamingResponse:
    """Stream one JSON object per line.

    Newline-delimited JSON rather than server-sent events because these endpoints are POSTs
    with a body, which `EventSource` cannot issue — so the client reads the body with a
    stream reader either way, and NDJSON is the format that needs no framing rules.
    `X-Accel-Buffering` is set because a proxy that buffers the body turns a stream back
    into one late blob, which is the exact failure this replaces.
    """
    def lines():
        for event in events:
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@app.post("/coordinator/stream")
def coordinator_stream(request: AgentMessageRequest):
    """The coordinator flow as progress events: brief tokens, then one step per stage."""
    return ndjson(service.dispatch_coordinator_events(
        request.message, dimensions=request.dimensions, elevation=request.elevation,
    ))


@app.post("/environments/{environment_id}/experiment/stream")
def experiment_stream(environment_id: str, request: AgentMessageRequest):
    """Ask for run conditions on one circuit and launch them, reporting each run as it lands."""
    try:
        return ndjson(service.dispatch_experiment_events(environment_id, request.message))
    except KeyError as error:
        raise missing(error) from error


@app.post("/agents/{role}/stream")
def stream_agent_message(role: str, request: AgentMessageRequest, environment_id: str | None = None):
    """A plain chat reply, streamed."""
    try:
        return ndjson(service.stream_agent_message(role, request.message, environment_id))
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/environments")
def environments() -> list:
    return service.list_environments()


@app.get("/drawings")
def drawings() -> list:
    return service.list_drawings()


@app.post("/drawings")
def create_drawing(request: TrackDrawingCreate):
    try:
        return service.create_drawing(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.delete("/drawings/{drawing_id}")
def delete_drawing(drawing_id: str):
    try:
        return service.delete_drawing(drawing_id)
    except KeyError as error:
        raise missing(error) from error


@app.post("/environments")
def create_environment(request: CreateEnvironmentRequest):
    try:
        return service.create_environment(
            request.prompt, request.seed, request.parent_environment_id, request.origin,
            request.provider, dimensions=request.dimensions, elevation=request.elevation,
        )
    except (ProviderError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/environments/{environment_id}")
def environment(environment_id: str):
    record = service.get_environment(environment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Environment not found")
    return record


@app.delete("/environments/{environment_id}")
def delete_environment(environment_id: str):
    try:
        return service.delete_environment(environment_id)
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/runs")
def runs(environment_id: str | None = None) -> list:
    return service.list_runs(environment_id)


@app.post("/runs")
def create_run(request: RunRequest):
    try:
        return service.run(request)
    except KeyError as error:
        raise missing(error) from error
    except ProviderError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/runs/{run_id}")
def run(run_id: str):
    record = service.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


@app.delete("/runs/{run_id}")
def delete_run(run_id: str):
    try:
        return service.delete_run(run_id)
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


CAMERAS = ("third-person", "first-person", "hood", "third-person-far", "overhead-3d")


@app.get("/environments/{environment_id}/view3d")
def environment_view3d(
    environment_id: str, camera: str = "free", width: int = 900, height: int = 520,
    yaw: float = 45.0, pitch: float = 35.0, distance: float = 900.0, focus: str = "circuit",
):
    """Perspective render of a 3D circuit at the grid.

    Served as an image rather than reimplemented in the browser: a camera is a pure function
    of world state, so the only thing a second renderer could add is a second set of bugs.
    """
    try:
        return Response(
            content=service.render_environment_view3d(
                environment_id, camera, width, height, yaw, pitch, distance, focus,
            ),
            media_type="image/png", headers={"Cache-Control": "no-store"},
        )
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/runs/{run_id}/view3d")
def run_view3d(run_id: str, step: int = 0, camera: str = "third-person", width: int = 900, height: int = 520):
    """Perspective render of one recorded tick of a 3D run."""
    try:
        return Response(
            content=service.render_run_view3d(run_id, step, camera, width, height),
            media_type="image/png", headers={"Cache-Control": "no-store"},
        )
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/runs/{run_id}/replay-bundle")
def replay_bundle(run_id: str):
    try:
        return service.get_replay_bundle(run_id)
    except KeyError as error:
        raise missing(error) from error


@app.post("/runs/{run_id}/open-native-viewer")
def open_native_viewer(run_id: str):
    try:
        return service.launch_native_viewer(run_id)
    except KeyError as error:
        raise missing(error) from error


@app.post("/runs/{run_id}/fork")
def fork_run(run_id: str, request: ForkRequest):
    try:
        return service.fork_run(run_id, request)
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/experiments")
def experiments(environment_id: str | None = None) -> list:
    return service.list_experiments(environment_id)


@app.delete("/experiments/{experiment_key}")
def delete_experiment(experiment_key: str, environment_id: str | None = None):
    try:
        return service.delete_experiment(experiment_key, environment_id)
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/studies/{study_kind}/{study_id}/panels")
def study_panels(study_kind: str, study_id: str):
    try:
        return service.study_panels(study_kind, study_id)
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.put("/studies/{study_kind}/{study_id}/panels")
def update_study_panels(study_kind: str, study_id: str, request: StudyPanelUpdateRequest):
    try:
        return service.update_study_panels(study_kind, study_id, request)
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/experiments")
def create_experiment(request: ExperimentRequest):
    try:
        return service.run_experiment(request)
    except KeyError as error:
        raise missing(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/experiments/plan")
def plan_experiment(request: ExperimentRequest):
    """Return an executor-neutral rollout DAG without launching any workers."""
    try:
        return service.compile_experiment_dag(request)
    except KeyError as error:
        raise missing(error) from error


@app.get("/adapters")
def adapters() -> dict:
    return {
        "players": [
            {"name": "oracle-racing-line", "input_class": "oracle", "transport": "python", "observations": ["racing line", "telemetry", "active checkpoint"], "snapshot": False, "role": "deterministic privileged reference"},
            {"name": "telemetry-direct", "input_class": "telemetry", "transport": "model API", "observations": ["pose and dynamics telemetry", "racing-line lookahead", "track progress", "nearby entities", "active checkpoint"], "snapshot": False, "controls_executed_directly": True},
            {"name": "telemetry-strategy", "input_class": "telemetry", "transport": "one pre-race model call", "observations": ["12 circuit sectors", "curvature", "surface and grip", "barriers and traffic"], "snapshot": False, "controls_executed_directly": False},
            {"name": "telemetry-hierarchical", "input_class": "telemetry", "transport": "model intent plus tick-rate controller", "observations": ["local track telemetry", "speed", "progress", "traffic"], "snapshot": False, "controls_executed_directly": False},
            {"name": "telemetry-reflex", "input_class": "telemetry", "transport": "model tool loop with occasional wakes", "observations": ["normalized local telemetry", "wake cause", "controller trace", "block diagnostics"], "snapshot": False, "controls_executed_directly": False, "role": "writes a tick-rate controller"},
            {"name": "vision-reflex-sim-rehearsal", "input_class": "vision-assisted", "transport": "model tool loop plus forked simulator rehearsal", "observations": ["camera RGB", "physical speed", "image-derived road cues", "simulator rehearsal feedback"], "snapshot": True, "controls_executed_directly": False},
            {"name": "vision-2d-predictive-skills", "input_class": "vision+speed", "transport": "overlapped model calls plus tick-rate visual feedback", "observations": ["forward-cone RGB", "physical speed", "same-image road features", "active skill and prior requested keys"], "snapshot": False, "controls_executed_directly": False, "role": "predicts response-time state and selects a reusable feedback skill"},
            {"name": "vision-2d-direct", "input_class": "vision", "transport": "short-horizon model calls", "observations": ["forward-cone RGB", "optical flow", "prior requested keys"], "snapshot": False, "controls_executed_directly": True},
            {"name": "vision-3d-direct-every-tick", "input_class": "vision+speed", "transport": "one model call per control tick", "observations": ["first-person RGB", "physical speed", "prior requested keys"], "snapshot": False, "controls_executed_directly": True},
            {"name": "vision-3d-direct-short", "input_class": "vision+speed", "transport": "short-horizon model calls", "observations": ["first-person RGB", "physical speed", "prior requested keys"], "snapshot": False, "controls_executed_directly": True},
            {"name": "vision-3d-direct-short-features", "input_class": "vision+speed", "transport": "short-horizon model calls", "observations": ["first-person RGB", "physical speed", "same-image road features", "prior requested keys"], "snapshot": False, "controls_executed_directly": True},
            {"name": "vision-3d-predictive-skills", "input_class": "vision+speed", "transport": "overlapped model calls plus tick-rate visual feedback", "observations": ["first-person RGB", "physical speed", "same-image road features", "active skill and prior requested keys"], "snapshot": False, "controls_executed_directly": False, "role": "predicts response-time state and selects a reusable feedback skill"},
            {"name": "baseline-constant-intent", "input_class": "baseline", "transport": "python", "observations": ["local controller telemetry"], "snapshot": False, "role": "fixed-intent controller baseline"},
            {"name": "baseline-random", "input_class": "baseline", "transport": "python", "observations": [], "snapshot": False, "role": "random-control failure baseline"},
            {"name": "external-telemetry-player", "input_class": "external-telemetry", "enabled": "external-telemetry-player" in service.policies, "transport": "HTTP racelab-policy/v3", "observations": ["complete scene on reset", "overhead PNG", "complete observation telemetry"], "actions": "held WASD plus nitro key state with action repeat", "controls_executed_directly": True, "fallback_action": None},
        ],
        "live_policy_contract": {
            "protocol": "racelab-policy/v3",
            "reset": "POST {EXTERNAL_PLAYER_URL}/reset",
            "act": "POST {EXTERNAL_PLAYER_URL}/act",
            "action_space": "keyboard-wasd-nitro",
            "simultaneous_keys": True,
            "silent_fallback": False,
        },
        "environment_designers": [
            {"name": "racing-grammar", "provider": "local", "token_budget": 0, "domain": "racing-2d-v5"},
            {"name": "model-racing-creator", "provider": "openai-or-anthropic", "output": "typed track plan in the corner grammar", "compiler": "track-grammar-v1", "domain": "racing-2d-v5"},
        ],
        "execution_backends": service.execution_backends(),
        "artifact_contract": {
            "active_store": service.artifact_store.id,
            "control_plane": "SQLite metadata and immutable artifact references only",
            "data_plane": "Workers write replay bundles and large outputs directly to object storage",
            "portable_stores": ["file", "s3-compatible"],
        },
    }


@app.get("/providers")
def providers() -> dict:
    import os

    return {
        "environment_generation": {
            "active": active_provider(),
            "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "active_environment_model": configured_model("ANTHROPIC_ENVIRONMENT_MODEL"),
            "active_player_model": configured_model("ANTHROPIC_PLAYER_MODEL"),
        }
    }


@app.get("/engine")
def engine() -> dict:
    return {
        "active": {"id": service.runtime.id, "display_name": service.runtime.display_name},
        "integration_contract": "RacingBackend.create(scene, perturbation) → racing observation/step/restore interface",
        "viewer_contract": "ReplayBundle v2 / racing-2d",
    }

"""Engine-neutral replay contract shared by native renderers and simulator plugins."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .models import FrameRecord, RunRecord, SceneSpec


class RendererDescriptor(BaseModel):
    id: str
    display_name: str
    transport: str
    supports_live_mode: bool = False
    supports_replay: bool = True


class ReplayMetadata(BaseModel):
    run_id: str
    policy_name: str
    status: str
    seed: int
    total_reward: float
    perturbation: dict[str, str] | None = None


class ReplayBackendManifest(BaseModel):
    """Backend-owned contract; the harness never interprets its state payload."""

    id: str
    display_name: str
    state_schema: str
    action_schema: str
    observation_schema: str


class ReplayTimelineFrame(BaseModel):
    """Renderer-neutral chronological state checkpoint from any simulator."""

    step: int
    time_s: float
    action: str = "idle"
    state: dict[str, Any]
    events: list[str] = Field(default_factory=list)


class ReplayBundle(BaseModel):
    """Portable viewer input.

    A simulator only needs to serialize this bundle (or stream equivalent
    scene/frame messages) for a renderer to display it. It intentionally sits
    outside player-policy observation boundaries: visualizers may use
    evaluator-owned state, but player adapters never receive it.
    """

    schema_version: int = 2
    renderer_hint: str = "racing-topdown-2d"
    renderer: RendererDescriptor = Field(default_factory=lambda: RendererDescriptor(id="native-racing-2d", display_name="Native racing replay viewer", transport="replay-bundle/v2"))
    backend: ReplayBackendManifest = Field(default_factory=lambda: ReplayBackendManifest(id="racing-2d-v5", display_name="Deterministic top-down racing engine", state_schema="racing-2d/state-v5", action_schema="racing-2d/controls-v5", observation_schema="racing-2d/observation-v5"))
    scene: SceneSpec | None = None
    metadata: ReplayMetadata
    frames: list[FrameRecord] = Field(default_factory=list)
    timeline: list[ReplayTimelineFrame] = Field(default_factory=list)

    @classmethod
    def from_run(cls, scene: SceneSpec, run: RunRecord) -> "ReplayBundle":
        return cls.from_frames(
            scene, run.frames,
            metadata=ReplayMetadata(
                run_id=run.id,
                policy_name=run.policy_name,
                status=run.status.value,
                seed=run.seed,
                total_reward=run.total_reward,
                perturbation=run.perturbation,
            ),
        )

    @classmethod
    def from_frames(
        cls, scene: SceneSpec, frames: list[FrameRecord], *, metadata: ReplayMetadata,
    ) -> "ReplayBundle":
        """Build a viewer bundle from frames alone, without a stored `RunRecord`.

        `from_run` is the control-plane path: a run exists in the store and the bundle is
        derived from it. An episode driven outside that path — the reflex runner, for one —
        has the same frames and no run record, and it should not have to fabricate one to be
        watchable. Both go through the same timeline construction, so a reflex replay and a
        `harness run` replay are the same artifact to every renderer.
        """
        # The scene decides the renderer, not the caller. A bundle carrying an elevation
        # profile cannot be drawn correctly by a planar renderer, so it says so rather than
        # letting a viewer guess and silently flatten every hill.
        elevated = scene is not None and scene.elevation is not None and not scene.elevation.is_flat
        return cls(
            renderer_hint="racing-perspective-3d" if elevated else "racing-topdown-2d",
            renderer=RendererDescriptor(
                id="native-racing-3d" if elevated else "native-racing-2d",
                display_name=(
                    "Native racing replay viewer (perspective 3D)" if elevated
                    else "Native racing replay viewer"
                ),
                transport="replay-bundle/v2",
            ),
            backend=ReplayBackendManifest(
                id="racing-3d-v1" if elevated else "racing-2d-v5",
                display_name=(
                    "Deterministic racing engine over an elevated surface" if elevated
                    else "Deterministic top-down racing engine"
                ),
                state_schema="racing-2d/state-v5",
                action_schema="racing-2d/controls-v5",
                observation_schema="racing-2d/observation-v5",
            ),
            scene=scene,
            metadata=metadata,
            frames=frames,
            timeline=[
                ReplayTimelineFrame(
                    step=frame.step, time_s=round(frame.step * 0.1, 3), action=frame.action.value,
                    state={
                        "player": frame.privileged_state.player.model_dump(),
                        "heading": frame.privileged_state.heading,
                        "speed": frame.privileged_state.speed,
                        "nitro": frame.privileged_state.nitro,
                        "nitro_active": frame.privileged_state.nitro_active,
                        "nitro_ready": frame.privileged_state.nitro_ready,
                        "countdown_ticks_remaining": frame.privileged_state.countdown_ticks_remaining,
                        "longitudinal_velocity_mps": frame.privileged_state.longitudinal_velocity_mps,
                        "lateral_velocity_mps": frame.privileged_state.lateral_velocity_mps,
                        "yaw_rate_degrees_per_second": frame.privileged_state.yaw_rate_degrees_per_second,
                        "steering_angle_degrees": frame.privileged_state.steering_angle_degrees,
                        "longitudinal_acceleration_mps2": frame.privileged_state.longitudinal_acceleration_mps2,
                        "lateral_acceleration_mps2": frame.privileged_state.lateral_acceleration_mps2,
                        "slip_angle_degrees": frame.privileged_state.slip_angle_degrees,
                        "aerodynamic_drag_n": frame.privileged_state.aerodynamic_drag_n,
                        "rolling_resistance_n": frame.privileged_state.rolling_resistance_n,
                        "lateral_load_transfer_n": frame.privileged_state.lateral_load_transfer_n,
                        "turning": frame.privileged_state.turning,
                        "barrier_impact": (
                            frame.privileged_state.barrier_impact.model_dump()
                            if frame.privileged_state.barrier_impact else None
                        ),
                        "checkpoint_index": frame.privileged_state.objective_index,
                        "lap": frame.privileged_state.lap,
                    },
                    events=frame.events,
                )
                for frame in frames
            ],
        )

    def write_json(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target


def snapshot_from_frame(frame: FrameRecord) -> dict[str, Any]:
    """Turn one recorded frame back into a world snapshot `RacingWorld.restore` accepts.

    Shared by the fork path and the 3D replay renderer, which need the same thing for
    different reasons: forking resumes a run from a recorded tick, and a perspective
    renderer has to stand the world back up at each tick because a camera is a function of
    world state rather than of a 2D coordinate. `Racing3DWorld.restore` recomputes vertical
    pose from planar position, so a restored 3D frame sits exactly on the road it recorded.
    """
    state = frame.privileged_state
    return {
        "step": frame.step,
        "player": state.player.model_dump(),
        "heading": state.heading,
        "speed": state.speed,
        "nitro": state.nitro,
        "nitro_active": state.nitro_active,
        "turning": state.turning,
        "countdown_ticks_remaining": state.countdown_ticks_remaining,
        "longitudinal_velocity_mps": state.longitudinal_velocity_mps,
        "lateral_velocity_mps": state.lateral_velocity_mps,
        "yaw_rate_radians_per_second": math.radians(state.yaw_rate_degrees_per_second),
        "steering_angle_radians": math.radians(state.steering_angle_degrees),
        "longitudinal_acceleration_mps2": state.longitudinal_acceleration_mps2,
        "lateral_acceleration_mps2": state.lateral_acceleration_mps2,
        "slip_angle_radians": math.radians(state.slip_angle_degrees),
        "aerodynamic_drag_n": state.aerodynamic_drag_n,
        "rolling_resistance_n": state.rolling_resistance_n,
        "lateral_load_transfer_n": state.lateral_load_transfer_n,
        "barrier_impact": state.barrier_impact.model_dump() if state.barrier_impact else None,
        "objective_index": state.objective_index,
        "terminated": False,
        "succeeded": False,
        "reason": None,
        "delayed_action": None,
        "fog": False,
        "held_keys": list(frame.keys),
        "opponents": [
            {
                "entity_id": entity["id"],
                "position": {
                    "x": entity["x"] + entity["width"] / 2,
                    "y": entity["y"] + entity["height"] / 2,
                },
                "target_index": 0,
                "lane_offset": entity.get("lane_offset", 34.0),
                "track_index": entity.get("track_index", 0),
                "base_lane_offset": entity.get("lane_offset", 34.0),
                "target_lane_offset": entity.get("lane_offset", 34.0),
                "heading": entity.get("heading", 0.0),
                "overtake_phase": entity.get("overtake_phase", "cruise"),
                "speed": entity.get("speed", 0.0),
                "nitro": entity.get("nitro", 0.0),
                "nitro_active": entity.get("nitro_active", False),
            }
            for entity in state.entities if entity["kind"] == "npc"
        ],
    }


class ReplayRenderer(Protocol):
    """A desktop, Unity, or third-party renderer can implement this boundary."""

    descriptor: RendererDescriptor

    def replay(self, bundle: ReplayBundle) -> None: ...


class SimulatorReplayBridge(Protocol):
    """Minimal contract a GI simulator integration would expose to this harness."""

    descriptor: RendererDescriptor

    def export_replay(self, run_id: str) -> ReplayBundle: ...

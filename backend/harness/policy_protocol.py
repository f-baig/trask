"""Portable live-policy contract for simulator and third-party player adapters.

The evaluator owns reset/step/termination. A policy owns every requested key.
No racing-line controller or fallback action is part of this boundary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .models import ObservationPacket, SceneSpec


PERSPECTIVE_VIEWPOINTS = frozenset({
    "first-person", "hood", "third-person", "third-person-far", "overhead-3d",
})


class CameraContract(BaseModel):
    """Exactly how a perspective frame was produced.

    A 2D overhead frame needs no camera description because the mapping from
    pixels to world is a fixed scale. A perspective frame does: without knowing
    the eye height, the pitch, and the field of view, a policy cannot turn what
    it sees into a distance. Everything here is a pure function of simulator
    state, so the same tick always yields the same camera.
    """

    mode: Literal["first-person", "hood", "third-person", "third-person-far", "overhead-3d"]
    projection: Literal["pinhole-perspective"] = "pinhole-perspective"
    vertical_fov_degrees: float = Field(gt=0, le=170)
    horizontal_fov_degrees: float = Field(gt=0, le=179)
    eye_height_pixels: float
    """Camera height above the road surface under the car."""
    pitch_degrees: float
    """Camera pitch; negative looks down towards the road."""
    distance_behind_pixels: float = 0
    """Chase distance behind the car; zero for in-car views."""
    follows_ego_heading: bool = True
    near_plane_pixels: float = Field(gt=0)
    pixels_per_meter: float = Field(gt=0)
    """World-unit scale, so an on-screen size can be reasoned about in meters."""


class VisualFrame(BaseModel):
    media_type: Literal["image/png", "image/jpeg", "application/x-rgb"]
    data_base64: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(default=3, ge=1, le=4)
    viewpoint: Literal[
        "overhead", "forward-cone",
        "first-person", "hood", "third-person", "third-person-far", "overhead-3d",
    ] = "overhead"
    orientation: Literal["north-up", "ego-forward-up", "camera-up"] = "north-up"
    ego_anchor: Literal["world-position", "bottom-center", "camera-relative"] = "world-position"
    heading_guide: bool = False
    heading_guide_semantics: Literal["current-ego-heading"] | None = None
    horizontal_fov_degrees: float | None = Field(default=None, gt=0, le=180)
    range_pixels: float | None = Field(default=None, gt=0)
    camera: CameraContract | None = None
    motion_overlay: bool = False
    """Whether per-cell optical-flow arrows are drawn over this frame."""
    motion_overlay_semantics: Literal["grid-averaged-optical-flow"] | None = None
    motion_grid: list[int] | None = None
    """Arrow grid as `[rows, columns]`; one arrow is the mean flow of one cell."""
    motion_arrow_scale: float | None = Field(default=None, gt=0)
    """Drawn arrow pixels per pixel-per-frame of flow, so length is invertible."""
    motion_arrow_max_pixels: float | None = Field(default=None, gt=0)
    """Length at which an arrow saturates; beyond this only direction is exact."""
    motion_base: Literal["grayscale", "color"] | None = None
    """What the arrows are drawn over, since grayscale removes the entity palette."""
    motion_interval_ticks: int | None = Field(default=None, ge=1)
    """Simulator ticks spanned by the flow measurement; one under a synchronous loop."""

    @model_validator(mode="after")
    def validate_view_contract(self) -> "VisualFrame":
        if self.viewpoint == "forward-cone" and (
            self.orientation != "ego-forward-up" or self.ego_anchor != "bottom-center"
        ):
            raise ValueError(
                "forward-cone frames must be ego-forward-up with a bottom-center ego anchor"
            )
        if self.heading_guide and self.heading_guide_semantics != "current-ego-heading":
            raise ValueError("heading-guide frames must declare current-ego-heading semantics")
        if self.motion_overlay:
            # Arrows are uninterpretable without the grid and the length scale, so a
            # frame may not claim a motion overlay while withholding either.
            if self.motion_overlay_semantics != "grid-averaged-optical-flow":
                raise ValueError("motion frames must declare grid-averaged-optical-flow semantics")
            if not self.motion_grid or len(self.motion_grid) != 2 or min(self.motion_grid) < 1:
                raise ValueError("motion frames must declare a [rows, columns] grid")
            if self.motion_arrow_scale is None or self.motion_arrow_max_pixels is None:
                raise ValueError("motion frames must declare the arrow length scale")
            if self.motion_base is None:
                raise ValueError("motion frames must declare a grayscale or color base")
            if self.motion_interval_ticks is None:
                raise ValueError("motion frames must declare the interval they measure")
        elif (
            self.motion_overlay_semantics is not None or self.motion_base is not None
            or self.motion_interval_ticks is not None
        ):
            raise ValueError("only motion frames may declare motion-overlay fields")
        if self.viewpoint in PERSPECTIVE_VIEWPOINTS:
            if self.camera is None or self.camera.mode != self.viewpoint:
                raise ValueError(
                    "perspective frames must carry a camera contract matching their viewpoint"
                )
            if self.orientation != "camera-up" or self.ego_anchor != "camera-relative":
                raise ValueError(
                    "perspective frames must be camera-up with a camera-relative ego anchor"
                )
        elif self.camera is not None:
            raise ValueError("only perspective frames may declare a camera contract")
        return self


class KeyboardState(BaseModel):
    """Held driving keys for one or more simulator ticks, including straight nitro."""

    keys: list[Literal["w", "a", "s", "d", "space"]] = Field(default_factory=list)
    repeat: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def reject_conflicts(self) -> "KeyboardState":
        if "w" in self.keys and "s" in self.keys:
            raise ValueError("w and s cannot be held simultaneously")
        if "a" in self.keys and "d" in self.keys:
            raise ValueError("a and d cannot be held simultaneously")
        self.keys = list(dict.fromkeys(self.keys))
        return self


class PolicyCapabilities(BaseModel):
    protocol: Literal["racelab-policy/v3"] = "racelab-policy/v3"
    observation_modalities: list[Literal["rgb", "telemetry"]] = Field(default_factory=lambda: ["rgb"])
    action_space: Literal["keyboard-wasd-nitro"] = "keyboard-wasd-nitro"
    simultaneous_keys: bool = True
    action_repeat_max: int = Field(default=20, ge=1, le=20)


class PolicyReset(BaseModel):
    protocol: Literal["racelab-policy/v3"] = "racelab-policy/v3"
    episode_id: str
    seed: int
    scene: SceneSpec
    capabilities: PolicyCapabilities = Field(default_factory=PolicyCapabilities)


class PolicyStep(BaseModel):
    protocol: Literal["racelab-policy/v3"] = "racelab-policy/v3"
    episode_id: str
    observation: ObservationPacket
    frame: VisualFrame | None = None
    previous_keys: list[Literal["w", "a", "s", "d", "space"]] = Field(default_factory=list)
    previous_reward: float | None = None
    previous_events: list[str] = Field(default_factory=list)


class PolicyAction(BaseModel):
    protocol: Literal["racelab-policy/v3"] = "racelab-policy/v3"
    episode_id: str
    control: KeyboardState


class PolicyClose(BaseModel):
    protocol: Literal["racelab-policy/v3"] = "racelab-policy/v3"
    episode_id: str
    status: Literal["succeeded", "failed", "timeout", "cancelled"]
    reason: str
    steps: int = Field(ge=0)

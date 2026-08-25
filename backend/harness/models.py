from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Safe to import here: `prompt_spec` is a leaf that depends on nothing in the
# harness, which is what lets the generation contract be stored on a record
# without the scene models and the contract models importing each other.
from .prompt_spec import FidelityReport, PromptSpec
from .visual import VisualPlan


class Vec2(BaseModel):
    x: float
    y: float


class Rect(BaseModel):
    x: float
    y: float
    width: float
    height: float


class EntityKind(StrEnum):
    CHECKPOINT = "checkpoint"
    OBSTACLE = "obstacle"
    NPC = "npc"


class CollisionShape(StrEnum):
    """How an entity's `rect` is interpreted by collision and by renderers.

    One declaration drives both, so an obstacle can never be drawn as one shape
    and collided against as another. New shapes are added here and handled in
    `collision.py`; nothing else needs to branch on them.
    """

    CIRCLE = "circle"
    """Round bollard or tyre stack; the radius is the smaller rect half-extent."""
    BOX = "box"
    """Axis-aligned rectangle."""
    ORIENTED_BOX = "oriented-box"
    """Rectangle rotated by `rotation_degrees`, for barriers that run along a track."""


class EntitySpec(BaseModel):
    id: str
    kind: EntityKind
    rect: Rect
    label: str
    color: str | None = None
    shape: CollisionShape = CollisionShape.BOX
    rotation_degrees: float = Field(default=0.0, ge=-360, le=360)
    """Orientation for `oriented-box`; ignored by the other shapes."""


class ObjectiveKind(StrEnum):
    REACH = "reach"


class ObjectiveSpec(BaseModel):
    kind: ObjectiveKind
    target_id: str
    description: str


class ResearchSceneMetadata(BaseModel):
    """Semantic annotations used by research analyses, never player input."""

    suite: str
    family_id: str
    variant_id: str
    mechanic: str
    intervention: str | None = None


class TrackRegion(StrEnum):
    """Nine addressable screen regions, so briefs can locate track features."""

    AUTO = "auto"
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"


class CornerRadius(StrEnum):
    HAIRPIN = "hairpin"
    TIGHT = "tight"
    MEDIUM = "medium"
    OPEN = "open"
    SWEEPING = "sweeping"


class StraightLength(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class NpcProfile(StrEnum):
    """Named opponent temperaments; each expands to explicit numeric behavior."""

    BACKMARKER = "backmarker"
    CRUISER = "cruiser"
    RACER = "racer"
    AGGRESSOR = "aggressor"
    BLOCKER = "blocker"


class NpcBehaviorSpec(BaseModel):
    """One opponent's serialized driving temperament.

    The engine reads only these numbers, so opponent character is an auditable
    scene parameter rather than a module constant shared by every car.
    """

    entity_id: str
    profile: NpcProfile = NpcProfile.RACER
    pace: float = Field(default=.9, ge=.35, le=1.05)
    """Baseline cruise speed as a fraction of the vehicle maximum."""
    skill: float = Field(default=.7, ge=0, le=1)
    """Cornering commitment: how much of its cruise pace it carries through a turn."""
    intelligence: float = Field(default=.7, ge=0, le=1)
    """How well it drives the line, as opposed to how fast it is willing to go.

    Buys two things: it looks further ahead before deciding a corner has begun,
    and it aims for the geometric inside of that corner instead of holding its
    grid lane. A low value keeps a wandering line, which makes a car beatable
    without making it slow -- a separate research axis from `pace` and `skill`.
    """
    aggression: float = Field(default=.5, ge=0, le=1)
    """Willingness to commit to passes, carry corner speed, and spend nitro early."""
    defends: bool = False
    """Whether the car covers the inside line against a closing player."""
    uses_nitro: bool = True


class CornerReport(BaseModel):
    """Requested versus compiled geometry for one corner."""

    index: int
    direction: Literal["left", "right"]
    requested_angle_degrees: float | None = None
    achieved_angle_degrees: float
    requested_region: TrackRegion = TrackRegion.AUTO
    achieved_region: TrackRegion
    requested_radius: CornerRadius = CornerRadius.MEDIUM
    achieved_radius_pixels: float
    entry_progress_percent: float
    apex: Vec2
    recommended_entry_speed: float = 0
    origin: Literal["requested", "closure-filler"] = "requested"


class TrackReport(BaseModel):
    """Auditable evidence of how faithfully a brief became geometry.

    Environment-creator quality is a research measurement here, so the compiler
    publishes the residual between what was asked for and what was built rather
    than silently absorbing the difference.
    """

    compiler: str = "track-grammar-v1"
    loop_shape: Literal["cornered", "circle", "drawn"] = "cornered"
    """The authored centerline primitive, measured from the compiled geometry."""
    direction: Literal["clockwise", "counterclockwise"]
    corners: list[CornerReport] = Field(default_factory=list)
    length_pixels: float = 0
    longest_straight_pixels: float = 0
    minimum_radius_pixels: float = 0
    sector_count: int = 0
    closure_error_pixels: float = 0
    centerline_spacing_pixels: float = 0
    angle_fidelity_degrees: float = 0
    """Largest absolute requested-to-achieved turn-angle error, in degrees."""
    region_fidelity: float = 1.0
    """Fraction of region-constrained corners that landed in the asked-for region."""
    relaxations: list[str] = Field(default_factory=list)
    """Every deterministic adjustment the compiler had to make, in order."""


class VehicleDynamicsSpec(BaseModel):
    """Serializable physical parameters for one car.

    The engine uses a deterministic transient bicycle model. Keeping these
    values in the scene makes every handling condition auditable and forkable.
    """

    mass_kg: float = Field(default=1_180, ge=500, le=3_000)
    length_m: float = Field(default=4.25, ge=3.0, le=6.0)
    width_m: float = Field(default=1.82, ge=1.3, le=2.6)
    wheelbase_m: float = Field(default=2.58, ge=1.8, le=4.0)
    front_weight_fraction: float = Field(default=.53, ge=.35, le=.65)
    center_of_mass_height_m: float = Field(default=.48, ge=.2, le=1.2)
    yaw_inertia_kg_m2: float = Field(default=1_850, ge=600, le=6_000)
    engine_force_n: float = Field(default=6_500, ge=500, le=20_000)
    engine_power_w: float = Field(default=45_000, ge=10_000, le=500_000)
    brake_force_n: float = Field(default=14_000, ge=1_000, le=40_000)
    max_speed_mps: float = Field(default=14.0, ge=5, le=90)
    max_steering_angle_degrees: float = Field(default=20, ge=8, le=50)
    steering_rate_degrees_per_second: float = Field(default=100, ge=20, le=720)
    front_cornering_stiffness_n_per_rad: float = Field(default=72_000, ge=5_000, le=200_000)
    rear_cornering_stiffness_n_per_rad: float = Field(default=78_000, ge=5_000, le=200_000)
    tire_friction_multiplier: float = Field(default=1.0, ge=.2, le=2.0)
    tire_load_sensitivity: float = Field(default=.12, ge=0, le=.5)
    rolling_resistance_coefficient: float = Field(default=.014, ge=.002, le=.08)
    drag_coefficient: float = Field(default=.32, ge=.1, le=1.5)
    frontal_area_m2: float = Field(default=1.9, ge=1.0, le=4.0)
    lift_coefficient: float = Field(default=-.08, ge=-2.0, le=1.0)
    nitro_force_n: float = Field(default=3_600, ge=0, le=15_000)

    @model_validator(mode="after")
    def validate_vehicle_geometry(self) -> "VehicleDynamicsSpec":
        if self.wheelbase_m >= self.length_m:
            raise ValueError("wheelbase_m must be shorter than length_m")
        return self


class RoadDynamicsSpec(BaseModel):
    friction_coefficient: float = Field(default=1.0, ge=.05, le=2.0)
    lateral_grip_multiplier: float = Field(default=1.0, ge=.1, le=2.0)
    rolling_resistance_multiplier: float = Field(default=1.0, ge=.1, le=5.0)
    off_track_friction_coefficient: float = Field(default=.55, ge=.05, le=1.5)
    off_track_rolling_resistance_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)


class DynamicsSpec(BaseModel):
    model: Literal["transient-bicycle-v1"] = "transient-bicycle-v1"
    vehicle: VehicleDynamicsSpec = Field(default_factory=VehicleDynamicsSpec)
    road: RoadDynamicsSpec = Field(default_factory=RoadDynamicsSpec)
    air_density_kg_m3: float = Field(default=1.225, ge=.0, le=6.0)
    """Ambient air density; zero is a vacuum with no drag or downforce at all."""
    gravity_mps2: float = Field(default=9.81, ge=.5, le=30.0)
    """Surface gravity. The range spans low-gravity bodies through heavy planets so
    grip-versus-weight can be studied, not just Earth-like variation."""
    pixels_per_meter: float = Field(default=8.0, ge=2.0, le=30.0)
    physics_hz: int = Field(default=60, ge=20, le=240)
    control_hz: int = Field(default=10, ge=5, le=60)

    @model_validator(mode="after")
    def validate_fixed_step_ratio(self) -> "DynamicsSpec":
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be an integer multiple of control_hz")
        return self


class ElevationProfile(StrEnum):
    FLAT = "flat"
    ROLLING = "rolling"
    HILLY = "hilly"
    ALPINE = "alpine"


class ElevationSpec(BaseModel):
    """The vertical half of a circuit, used only by the 3D engine.

    A 2D scene leaves this unset and drives a perfectly flat plane. Elevation is
    stored on the scene rather than derived at render time so a 3D replay is
    reproducible and every parameter stays as auditable as the planar ones.
    """

    profile: ElevationProfile = ElevationProfile.ROLLING
    amplitude_m: float = Field(default=5.0, ge=0, le=40)
    """Peak-to-trough height of the elevation change, in meters."""
    hill_count: int = Field(default=3, ge=1, le=8)
    """Number of crests around one lap."""
    banking_degrees: float = Field(default=6.0, ge=0, le=22)
    """Maximum corner cross-slope; zero is a completely flat road surface."""
    crest_sharpness: float = Field(default=.35, ge=0, le=1)
    """Continuous shape control from smooth hills to sharper compound crests."""

    @model_validator(mode="before")
    @classmethod
    def profile_supplies_only_a_shape_default(cls, value: Any) -> Any:
        """Keep profile names semantic instead of making them canned terrain.

        The numeric value is stored on every scene and is what the compiler uses.
        A profile merely chooses a convenient default when an author did not
        provide one, so nearby values can produce nearby (and distinct) surfaces.
        """
        if not isinstance(value, dict) or "crest_sharpness" in value:
            return value
        profile = ElevationProfile(value.get("profile", ElevationProfile.ROLLING))
        defaults = {
            ElevationProfile.FLAT: 0.0,
            ElevationProfile.ROLLING: .18,
            ElevationProfile.HILLY: .48,
            ElevationProfile.ALPINE: .78,
        }
        return {**value, "crest_sharpness": defaults[profile]}

    @property
    def is_flat(self) -> bool:
        return self.profile == ElevationProfile.FLAT or self.amplitude_m <= 0


class SceneSpec(BaseModel):
    id: str
    version: int = 1
    name: str
    prompt: str
    seed: int
    bounds: Rect = Field(default_factory=lambda: Rect(x=0, y=0, width=960, height=640))
    player_spawn: Vec2
    start_line_index: int = Field(default=0, ge=0)
    """Centerline sample carrying the start/finish gate."""
    start_line_region: TrackRegion = TrackRegion.AUTO
    player_grid_position: int = Field(default=1, ge=1, le=6)
    entities: list[EntitySpec]
    objectives: list[ObjectiveSpec]
    domain_pack_version: str = "topdown-v1"
    track_centerline: list[Vec2] = Field(default_factory=list)
    track_width: float = 0
    edge_barriers: bool = False
    """Continuous guardrails following both road edges."""
    laps: int = 1
    surface: Literal["asphalt", "clay", "ice"] = "asphalt"
    grip: float = Field(default=1.0, ge=.3, le=1.2)
    """Continuous grip multiplier layered on top of the surface preset."""
    npc_start_mode: Literal["grid", "distributed"] = "grid"
    npc_behaviors: list[NpcBehaviorSpec] = Field(default_factory=list)
    sector_count: int = Field(default=4, ge=3, le=9)
    """Number of ordered gates per lap, including the finish line."""
    dynamics: DynamicsSpec = Field(default_factory=DynamicsSpec)
    elevation: ElevationSpec | None = None
    """Set only for 3D scenes; `None` means the planar 2D circuit."""
    track_report: TrackReport | None = None
    visual: VisualPlan = Field(default_factory=VisualPlan)
    """Palette and scenery. Read by renderers only; the simulator never looks at it,
    so recolouring a circuit cannot change a lap time or a collision."""
    research: ResearchSceneMetadata | None = None


class PlayabilityCertificate(BaseModel):
    """Deterministic evidence that the compiled scene can be completed."""

    verifier: str = "grid-reachability-v1"
    playable: bool
    checked_seed: int
    objective_trace: list[str] = Field(default_factory=list)
    route_steps: int = 0
    failure: str | None = None


class ActionName(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    IDLE = "idle"
    NITRO = "nitro"


class Action(BaseModel):
    name: ActionName = ActionName.IDLE
    duration_steps: int = Field(default=1, ge=1, le=20)
    keys: list[Literal["w", "a", "s", "d", "space"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_conflicting_keys(self) -> "Action":
        if "w" in self.keys and "s" in self.keys:
            raise ValueError("w and s cannot be held simultaneously")
        if "a" in self.keys and "d" in self.keys:
            raise ValueError("a and d cannot be held simultaneously")
        self.keys = list(dict.fromkeys(self.keys))
        return self


class ObservationPacket(BaseModel):
    step: int
    task: str
    rgb_hint: str
    proprioception: Vec2
    heading: float = 0
    speed: float = 0
    nitro: float = 0
    nitro_active: bool = False
    nitro_ready: bool = False
    countdown_ticks_remaining: int = 0
    dynamics: DynamicsSpec = Field(default_factory=DynamicsSpec)
    longitudinal_speed_mps: float = 0
    lateral_speed_mps: float = 0
    yaw_rate_degrees_per_second: float = 0
    steering_angle_degrees: float = 0
    longitudinal_acceleration_mps2: float = 0
    lateral_acceleration_mps2: float = 0
    slip_angle_degrees: float = 0
    checkpoint_index: int = 0
    local_entities: list[dict[str, Any]] = Field(default_factory=list)


class PrivilegedState(BaseModel):
    step: int
    player: Vec2
    inventory: list[str]
    objective_index: int
    entities: list[dict[str, Any]]
    heading: float = 0
    speed: float = 0
    nitro: float = 0
    nitro_active: bool = False
    nitro_ready: bool = False
    countdown_ticks_remaining: int = 0
    longitudinal_velocity_mps: float = 0
    lateral_velocity_mps: float = 0
    yaw_rate_degrees_per_second: float = 0
    steering_angle_degrees: float = 0
    longitudinal_acceleration_mps2: float = 0
    lateral_acceleration_mps2: float = 0
    slip_angle_degrees: float = 0
    aerodynamic_drag_n: float = 0
    rolling_resistance_n: float = 0
    lateral_load_transfer_n: float = 0
    turning: bool = False
    lap: int = 0
    barrier_impact: Vec2 | None = None
    """One-frame visual marker at the most recent barrier contact."""


class DecisionRecord(BaseModel):
    action: ActionName
    subgoal: str
    confidence: float = Field(ge=0, le=1)
    summary: str
    candidates: list[ActionName] = Field(default_factory=list)
    provider_usage: "ProviderTurnUsage | None" = None


class ProviderTurnUsage(BaseModel):
    """One auditable direct-model turn, attached to the replay decision."""

    turn: int
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_budget: int = 1_500
    remaining_output_budget: int = 1_500
    latency_ms: int = 0


class FrameRecord(BaseModel):
    step: int
    observation: ObservationPacket
    privileged_state: PrivilegedState
    action: ActionName
    keys: list[Literal["w", "a", "s", "d", "space"]] = Field(default_factory=list)
    reward: float
    events: list[str] = Field(default_factory=list)
    decision: DecisionRecord | None = None


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ExecutionState(StrEnum):
    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceRequest(BaseModel):
    """Scheduler-facing request; it is independent of any cloud provider."""

    cpu_cores: float = Field(default=1, gt=0)
    memory_mb: int = Field(default=1_024, ge=128)
    gpu_count: int = Field(default=0, ge=0)
    wall_time_seconds: int = Field(default=900, ge=30)
    queue: str | None = None


class ResourceUsage(BaseModel):
    """Host-process telemetry measured by the worker, not estimated by the UI."""

    cpu_time_ms: int = 0
    wall_time_ms: int = 0
    max_rss_mb: float = 0
    gpu_memory_mb: float | None = None
    host: str | None = None


class ArtifactReference(BaseModel):
    """A content-addressed artifact, held outside the control-plane database."""

    kind: str
    uri: str
    content_sha256: str
    size_bytes: int
    media_type: str = "application/json"


class ExecutionRecord(BaseModel):
    backend: str = "local"
    state: ExecutionState = ExecutionState.PLANNED
    job_id: str | None = None
    submitted_at: str | None = None
    worker_id: str | None = None
    resource_request: ResourceRequest = Field(default_factory=ResourceRequest)
    resource_usage: ResourceUsage | None = None
    scheduler_metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentAddress(BaseModel):
    """Human-readable, fixed-width research address: EXP-001."""

    experiment: int = Field(ge=1, le=999)

    @property
    def prefix(self) -> str:
        return f"EXP-{self.experiment:03d}"


class EnvironmentAddress(ExperimentAddress):
    """An environment family and one concrete variant within an experiment."""

    environment: int = Field(ge=1, le=999)
    variant: int = Field(ge=1, le=999)

    @property
    def prefix(self) -> str:
        return f"EXP-{self.experiment:03d}-ENV-{self.environment:03d}-VAR-{self.variant:03d}"


class PlayerAddress(EnvironmentAddress):
    """The complete 12-digit research address for a player replay."""

    player: int = Field(ge=1, le=999)

    @property
    def prefix(self) -> str:
        return f"EXP-{self.experiment:03d}-ENV-{self.environment:03d}-VAR-{self.variant:03d}-PLAYER-{self.player:03d}"


class ControllerWrite(BaseModel):
    """One controller source authored by a reflex agent at one wake."""

    tick: int = Field(ge=0)
    """Completed control ticks when the model authored this source (countdown excluded)."""
    frame_step: int = Field(ge=0)
    """Equivalent replay state, whose step includes the frozen countdown frames."""
    effective_from_frame_step: int | None = Field(default=None, ge=0)
    effective_until_frame_step: int | None = Field(default=None, ge=0)
    wake: int = Field(ge=1)
    name: str
    label: str | None = None
    source: str
    reads: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    installed: bool = False
    active: bool = False
    """True only when this exact version drove at least one live simulator tick."""
    errors: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    id: str
    environment_id: str
    environment_version: int
    policy_name: str
    seed: int
    status: RunStatus
    started_at: str
    completed_at: str | None = None
    parent_run_id: str | None = None
    fork_step: int | None = None
    perturbation: dict[str, Any] | None = None
    result_reason: str | None = None
    total_reward: float = 0
    token_usage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    player_turns: int = 0
    player_aggression: float | None = Field(default=None, ge=0, le=1)
    """Applied risk/pace dial for a player that supports it; None for controls."""
    output_token_budget: int | None = None
    latency_ms: int = 0
    study_name: str | None = None
    address: PlayerAddress | None = None
    execution: ExecutionRecord = Field(default_factory=ExecutionRecord)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    frames: list[FrameRecord] = Field(default_factory=list)
    controller_writes: list[ControllerWrite] = Field(default_factory=list)
    fork_supported: bool = False
    guidance_supported: bool = False
    realtime_metrics: dict[str, Any] = Field(default_factory=dict)
    """Wall-clock scheduler accounting for genuinely overlapped player runs."""


class EnvironmentRecord(BaseModel):
    id: str
    scene: SceneSpec
    created_at: str
    validation: list[str]
    baseline_solved: bool
    parent_environment_id: str | None = None
    origin: str | None = None
    generator_provider: str = "offline"
    generator_model: str = "deterministic-template"
    generator_rationale: str | None = None
    generator_input_tokens: int = 0
    generator_output_tokens: int = 0
    generator_latency_ms: int = 0
    playability_certificate: PlayabilityCertificate | None = None
    prompt_spec: "PromptSpec | None" = None
    """The contract read from the brief: what was asked for, in the user's words."""
    fidelity: "FidelityReport | None" = None
    """Which of those requirements the built circuit actually carries.

    Stored alongside the playability certificate and never merged with it. One is
    evidence the circuit works; this is evidence it is the circuit that was asked
    for, and a record that has the first without the second is exactly the case
    that used to be reported as an unqualified success.
    """
    study_name: str | None = None
    address: EnvironmentAddress | None = None


class TrackDrawingCreate(BaseModel):
    """A browser sketch in normalized canvas coordinates."""

    name: str = Field(min_length=1, max_length=64)
    points: list[Vec2] = Field(min_length=8, max_length=2_000)

    @model_validator(mode="after")
    def normalized_points(self) -> "TrackDrawingCreate":
        if any(not (0 <= point.x <= 1 and 0 <= point.y <= 1) for point in self.points):
            raise ValueError("Drawing points must stay inside the canvas")
        return self


class TrackDrawing(TrackDrawingCreate):
    id: str
    created_at: str


class CreateEnvironmentRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2_000)
    seed: int | None = None
    parent_environment_id: str | None = None
    origin: str | None = None
    provider: str = "auto"
    dimensions: Literal["2d", "3d"] = "2d"
    """Which engine the compiled scene targets.

    A 3D scene is a 2D scene with an elevation profile, so this only decides whether one
    is fitted and re-certified over its gradients. Everything downstream reads
    `scene.elevation` rather than this field.
    """
    elevation: ElevationSpec | None = None
    """Requested vertical profile for a 3D scene; the compiler fits it to drivable grades."""


class RunRequest(BaseModel):
    environment_id: str
    policy_name: str = "vision-2d-predictive-skills"
    max_steps: int = Field(default=1_200, ge=10, le=15_000)
    policy_decision_budget: int | None = Field(default=None, ge=1, le=4_000)
    player_aggression: float = Field(default=.78, ge=0, le=1)
    """Zero is conservative, one carries maximum pace within visual safety limits."""
    execution_backend: str = "local"
    resources: ResourceRequest = Field(default_factory=ResourceRequest)


class ForkRequest(BaseModel):
    fork_step: int = Field(ge=0)
    condition: str | None = Field(default=None, max_length=600)
    perturbation: Literal[
        "none", "action_delay", "obstacle_shift", "low_grip", "worn_tires",
        "heavy_car", "rear_bias", "high_drag", "high_downforce",
    ] | None = None
    guidance: str | None = Field(default=None, max_length=600)

    @model_validator(mode="after")
    def one_fork_contract(self) -> "ForkRequest":
        if self.condition and self.perturbation is not None:
            raise ValueError("Send either a natural-language condition or a structured perturbation, not both.")
        return self


class ExperimentRequest(BaseModel):
    environment_id: str
    name: str | None = Field(default=None, min_length=3, max_length=96)
    policies: list[str] = Field(default_factory=lambda: ["oracle-racing-line", "baseline-random"])
    perturbations: list[str] = Field(default_factory=lambda: ["normal", "low_grip", "action_delay"])
    seeds: list[int] = Field(default_factory=lambda: [11, 29, 47, 83, 101])
    max_steps: int = Field(default=1_200, ge=100, le=15_000)
    # Kept for clients that predate aggression sweeps.  A supplied sweep takes precedence.
    player_aggression: float = Field(default=.78, ge=0, le=1)
    player_aggressions: list[float] | None = Field(default=None, min_length=1, max_length=5)

    @model_validator(mode="after")
    def valid_aggression_sweep(self) -> "ExperimentRequest":
        if self.player_aggressions and any(not 0 <= value <= 1 for value in self.player_aggressions):
            raise ValueError("Each player aggression value must be between 0 and 1.")
        return self


class ExperimentRecord(BaseModel):
    id: str
    name: str = "Unlabeled comparison"
    address: ExperimentAddress | None = None
    environment_id: str
    created_at: str
    policies: list[str]
    perturbations: list[str]
    seeds: list[int]
    player_aggressions: list[float] = Field(default_factory=lambda: [.78])
    run_ids: list[str]
    summary: dict[str, Any]


class BehaviorFingerprint(BaseModel):
    run_id: str
    policy_name: str
    variant_id: str
    outcome: RunStatus
    route_signature: str
    steps: int
    blocked_moves: int
    mechanic_activations: int
    first_mechanic_step: int | None = None
    first_blocked_step: int | None = None
    recovery_after_block: bool = False
    final_position: Vec2


class CausalInterventionResult(BaseModel):
    intervention: str
    baseline_run_id: str
    counterfactual_run_id: str
    first_action_divergence: int | None = None
    outcome_changed: bool
    observation: str


class ResearchStudy(BaseModel):
    id: str
    created_at: str
    name: str
    address: ExperimentAddress | None = None
    family_id: str
    scenario_ids: list[str]
    run_ids: list[str]
    mechanic_probes: list[dict[str, Any]]
    fingerprints: list[BehaviorFingerprint]
    interventions: list[CausalInterventionResult]


class StudyPanelManifest(BaseModel):
    id: str
    title: str
    description: str
    kind: Literal["metric", "table"]
    version: str = "v1"


class StudyPanelResult(BaseModel):
    id: str
    title: str
    kind: Literal["metric", "table"]
    summary: str
    data: dict[str, Any]
    artifacts: list[ArtifactLink] = Field(default_factory=list)


class StudyPanelConfiguration(BaseModel):
    study_kind: Literal["comparison"]
    study_id: str
    panel_ids: list[str] = Field(default_factory=list)


class StudyPanelDashboard(BaseModel):
    configuration: StudyPanelConfiguration
    catalog: list[StudyPanelManifest]
    panels: list[StudyPanelResult]


class StudyPanelUpdateRequest(BaseModel):
    panel_ids: list[str] = Field(max_length=12)


class ProviderUsage(BaseModel):
    role: Literal["main", "environment", "player"]
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0


class ArtifactLink(BaseModel):
    kind: Literal["environment", "run", "study"]
    id: str
    label: str


class AgentAction(BaseModel):
    """A durable, user-facing step performed behind one assistant turn."""

    id: str
    label: str
    state: Literal["running", "done", "failed"]
    logs: list[str] = Field(default_factory=list)
    artifact: ArtifactLink | None = None


class AgentMessage(BaseModel):
    id: str
    agent_role: Literal["main", "environment"]
    environment_id: str | None = None
    speaker: Literal["user", "assistant"]
    content: str
    created_at: str
    artifacts: list[ArtifactLink] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)


class AgentMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    dimensions: Literal["2d", "3d"] = "2d"
    """Which engine a coordinator dispatch should compile for; ignored by plain chat."""
    elevation: ElevationSpec | None = None


class CoordinatorDispatch(BaseModel):
    """The result of one coordinator turn.

    Every field but the summary is optional, because most turns are conversation. A turn
    that did not build anything is a complete, successful turn — modelling it as a failure
    or as an empty circuit is what made the coordinator compile a racetrack in reply to
    "hey what's up".
    """

    summary: str
    environment_id: str | None = None
    environment_name: str | None = None
    certificate: PlayabilityCertificate | None = None

    @property
    def built(self) -> bool:
        return self.environment_id is not None

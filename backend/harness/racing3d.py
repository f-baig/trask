"""The 3D racing engine: the 2D rules, driven over a surface with height.

`Racing3DWorld` subclasses `RacingWorld` rather than reimplementing it. Every
rule that defines the game — checkpoint order, lap counting, opponent behavior
and overtaking, barrier and car collision, nitro charge-and-burn, the start
countdown, off-track recovery, snapshot and replay — is the same code running in
both engines. The subclass adds exactly three things:

1. it answers the planar engine's `road_attitude` hook with a real grade and
   bank sampled from the elevation profile, which is what makes hills and banked
   corners physical rather than decorative;
2. it tracks the vertical pose (height, pitch, roll) of the player and every
   opponent, so a camera has something to look at and along;
3. it certifies in 3D, because a circuit that is drivable flat is not
   automatically drivable once it has gradients.

Collision and track geometry stay planar. The road is a surface with height, not
a volume: cars cannot leave it vertically, jump, or pass over one another. That
is a deliberate limit of this engine rather than an oversight, and it keeps the
authoritative rules identical to the 2D game they are shared with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .collision import collider_for, outline
from .models import (
    Action, DecisionRecord, ElevationSpec, EntityKind, FrameRecord, PlayabilityCertificate,
    SceneSpec, Vec2,
)
from .racing import (
    CAR_RADIUS, RacingLineController, RacingWorld, validate_racing_scene,
)
from .track3d import (
    TrackSurface3D, compile_track_surface, fit_drivable_elevation, validate_track_surface,
)
from .track_grammar import TrackPlan


# Visual-only chassis attitude. Load transfer already changes grip inside the
# physics; these gains turn that same state into a body that leans and squats,
# and they never feed back into the simulation.
BODY_ROLL_DEGREES_PER_G = 3.4
BODY_PITCH_DEGREES_PER_G = 2.1
# How far around the current sample the surface lookup searches. One control tick
# is at most a few samples of travel, so a small window is both correct and cheap.
SURFACE_SEARCH_WINDOW = 8


@dataclass(frozen=True)
class CarPose3D:
    """A car's full pose in world space: ground position, height, and attitude."""

    x: float
    y: float
    z: float
    heading_degrees: float
    pitch_degrees: float
    roll_degrees: float

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Right-handed forward/left/up axes for this pose.

        Built by rotating world axes rather than multiplying matrices, so the
        result stays orthonormal without a re-normalization pass: yaw the flat
        axes, tip them by pitch about the left axis, then roll about forward.
        """
        yaw = math.radians(self.heading_degrees)
        pitch = math.radians(self.pitch_degrees)
        roll = math.radians(self.roll_degrees)
        forward_flat = (math.cos(yaw), math.sin(yaw), 0.0)
        left_flat = (-math.sin(yaw), math.cos(yaw), 0.0)
        up_world = (0.0, 0.0, 1.0)
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        forward = _combine(forward_flat, cos_pitch, up_world, sin_pitch)
        up_pitched = _combine(up_world, cos_pitch, forward_flat, -sin_pitch)
        cos_roll, sin_roll = math.cos(roll), math.sin(roll)
        left = _combine(left_flat, cos_roll, up_pitched, -sin_roll)
        up = _combine(up_pitched, cos_roll, left_flat, sin_roll)
        return forward, left, up


@dataclass(frozen=True)
class GateMarker3D:
    """One sector gate, with the centreline sample its markers stand on."""

    entity_id: str
    pose: CarPose3D
    track_index: int
    is_finish: bool
    passed: bool


@dataclass
class Racing3DWorld(RacingWorld):
    """The planar racing rules, simulated over a surface with elevation."""

    surface: TrackSurface3D | None = None
    player_z: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    squat_degrees: float = 0.0
    """Load-derived pitch component, separated so a renderer can interpolate it."""
    lean_degrees: float = 0.0
    surface_index: int = 0
    opponent_poses: dict[str, CarPose3D] = field(default_factory=dict)

    @classmethod
    def from_scene(
        cls, scene: SceneSpec, perturbation: str | None = None,
        elevation: ElevationSpec | None = None,
    ) -> "Racing3DWorld":
        world = super().from_scene(scene, perturbation)
        assert isinstance(world, cls)
        surface = compile_track_surface(scene, elevation)
        findings = validate_track_surface(surface)
        if findings:
            raise ValueError("Racing3DWorld rejected an undrivable surface: " + "; ".join(findings))
        world.surface = surface
        world.surface_index = _nearest_index(scene.track_centerline, world.player)
        world._refresh_vertical_state()
        return world

    # ------------------------------------------------------------------ physics

    def road_attitude(self, point: Vec2) -> tuple[float, float]:
        """Grade and bank under a point, answering the planar engine's hook.

        Called once per physics substep, so the sample is found by a bounded
        local search from the last known index instead of a full scan. A global
        nearest-point lookup would also be free to jump between two arms of a
        chicane that pass close together and flip the gradient sign mid-corner.
        """
        if self.surface is None:
            return (0.0, 0.0)
        index = self._local_surface_index(point)
        return self.surface.attitude_at_index(index)

    def _local_surface_index(self, point: Vec2) -> int:
        points = self.scene.track_centerline
        count = len(points)
        candidates = range(
            self.surface_index - SURFACE_SEARCH_WINDOW,
            self.surface_index + SURFACE_SEARCH_WINDOW + 1,
        )
        return min(
            (offset % count for offset in candidates),
            key=lambda index: (
                (points[index].x - point.x) ** 2 + (points[index].y - point.y) ** 2
            ),
        )

    # -------------------------------------------------------------------- state

    def step(
        self, requested_action: Action, decision: DecisionRecord | None = None,
        action_delay: bool = False,
    ) -> FrameRecord:
        frame = super().step(requested_action, decision, action_delay)
        self._refresh_vertical_state()
        return frame

    def restore(self, snapshot: dict) -> None:
        super().restore(snapshot)
        # Vertical pose is a pure function of planar position and the surface, so
        # it is recomputed rather than read back. A replayed height can never
        # disagree with the road the car is standing on.
        if self.surface is not None:
            self.surface_index = _nearest_index(self.scene.track_centerline, self.player)
            self._refresh_vertical_state()

    def snapshot(self) -> dict:
        state = super().snapshot()
        state.update({
            "player_z": self.player_z,
            "pitch_degrees": self.pitch_degrees,
            "roll_degrees": self.roll_degrees,
            "surface_index": self.surface_index,
            "opponent_poses": {
                entity_id: {
                    "z": pose.z, "pitch_degrees": pose.pitch_degrees,
                    "roll_degrees": pose.roll_degrees,
                }
                for entity_id, pose in self.opponent_poses.items()
            },
        })
        return state

    def _refresh_vertical_state(
        self, *, squat_degrees: float | None = None, lean_degrees: float | None = None,
    ) -> None:
        """Recompute every car's height and chassis attitude from the surface.

        Attitude has two parts. The grade and bank under the car vary smoothly
        with position, so they are always recomputed. Squat and lean come from
        acceleration, which is a per-substep quantity that holds one value for a
        whole control tick and then jumps -- visible as jitter at 60 FPS. Passing
        them in lets a renderer interpolate them across the tick instead.
        """
        if self.surface is None:
            return
        points = self.scene.track_centerline
        self.surface_index = self._local_surface_index(self.player)
        grade, bank = self.surface.attitude_at_index(self.surface_index)
        self.player_z = self.surface.surface_height(self.player, points, self.surface_index)
        gravity = self.dynamics.gravity_mps2
        if squat_degrees is None:
            squat_degrees = max(-6.0, min(6.0, (
                self.longitudinal_acceleration_mps2 / gravity * BODY_PITCH_DEGREES_PER_G
            )))
        if lean_degrees is None:
            lean_degrees = max(-8.0, min(8.0, (
                -self.lateral_acceleration_mps2 / gravity * BODY_ROLL_DEGREES_PER_G
            )))
        self.squat_degrees, self.lean_degrees = squat_degrees, lean_degrees
        self.pitch_degrees = math.degrees(grade) + squat_degrees
        self.roll_degrees = math.degrees(bank) + lean_degrees
        poses: dict[str, CarPose3D] = {}
        for opponent in self.opponents:
            index = _nearest_index(points, opponent.position)
            opponent_grade, opponent_bank = self.surface.attitude_at_index(index)
            poses[opponent.entity_id] = CarPose3D(
                x=opponent.position.x, y=opponent.position.y,
                z=self.surface.surface_height(opponent.position, points, index),
                heading_degrees=opponent.heading,
                pitch_degrees=math.degrees(opponent_grade),
                roll_degrees=math.degrees(opponent_bank),
            )
        self.opponent_poses = poses

    # ------------------------------------------------------------------- render

    def player_pose(self) -> CarPose3D:
        return CarPose3D(
            x=self.player.x, y=self.player.y, z=self.player_z,
            heading_degrees=self.heading,
            pitch_degrees=self.pitch_degrees, roll_degrees=self.roll_degrees,
        )

    def opponent_pose(self, entity_id: str) -> CarPose3D | None:
        return self.opponent_poses.get(entity_id)

    def barrier_footprints(self) -> list[tuple[str, CarPose3D, list[tuple[float, float]]]]:
        """Barriers as a ground pose plus the exact 2D outline collision tests.

        The renderer extrudes this outline rather than assuming a box, so any shape
        added to `collision.py` appears correctly in 3D without touching the
        renderer, and a barrier can never be drawn as a different shape than the
        one the car collides with.
        """
        if self.surface is None:
            return []
        points = self.scene.track_centerline
        placed: list[tuple[str, CarPose3D, list[tuple[float, float]]]] = []
        for entity in self.scene.entities:
            if entity.kind != EntityKind.OBSTACLE:
                continue
            collider = collider_for(entity, self.obstacle_shift)
            centre = Vec2(x=collider.centre_x, y=collider.centre_y)
            index = _nearest_index(points, centre)
            grade, bank = self.surface.attitude_at_index(index)
            placed.append((
                entity.id,
                CarPose3D(
                    x=centre.x, y=centre.y,
                    z=self.surface.surface_height(centre, points, index),
                    heading_degrees=_track_bearing(points, index),
                    pitch_degrees=math.degrees(grade), roll_degrees=math.degrees(bank),
                ),
                outline(collider, segments=12),
            ))
        return placed

    def gate_poses(self) -> list[GateMarker3D]:
        """Sector gates as edge-anchored markers, with the sample they sit on.

        The centreline sample is returned because a marker belongs on the road's
        banked *edge*, whose height differs from the centre by `tan(bank)` times
        the half width. Deriving edge positions from the centre pose alone leaves
        markers floating above or sunk into a cambered corner.
        """
        if self.surface is None:
            return []
        points = self.scene.track_centerline
        placed: list[GateMarker3D] = []
        number = 0
        crossed_this_lap = self.objective_index % max(1, self.scene.sector_count)
        for entity in self.scene.entities:
            if entity.kind != EntityKind.CHECKPOINT:
                continue
            centre = Vec2(
                x=entity.rect.x + entity.rect.width / 2,
                y=entity.rect.y + entity.rect.height / 2,
            )
            index = _nearest_index(points, centre)
            grade, bank = self.surface.attitude_at_index(index)
            placed.append(GateMarker3D(
                entity_id=entity.id,
                pose=CarPose3D(
                    x=centre.x, y=centre.y,
                    z=self.surface.surface_height(centre, points, index),
                    heading_degrees=_track_bearing(points, index),
                    pitch_degrees=math.degrees(grade), roll_degrees=math.degrees(bank),
                ),
                track_index=index,
                is_finish=entity.id == "finish-line",
                passed=number < crossed_this_lap,
            ))
            number += 1
        return placed

    @property
    def grade_degrees(self) -> float:
        """Current uphill gradient under the car, for HUD and telemetry."""
        if self.surface is None:
            return 0.0
        return math.degrees(self.surface.grade_at_index(self.surface_index))

    @property
    def bank_degrees(self) -> float:
        if self.surface is None:
            return 0.0
        return math.degrees(self.surface.bank_at_index(self.surface_index))


def compile_racing_3d_scene(
    prompt: str, plan: TrackPlan, elevation: ElevationSpec | None = None,
    seed: int | None = None,
) -> tuple[SceneSpec, PlayabilityCertificate, list[str]]:
    """Compile a plan into a 3D scene the oracle can finish over its gradients.

    The planar circuit is compiled and certified by the shared 2D path first, so
    a 3D scene is always a valid 2D scene with a surface added. The elevation is
    then fitted to the drivable grade for that specific circuit and re-certified,
    because gradients change lap times and cornering and can make an otherwise
    fine circuit uncompletable.
    """
    from .racing import compile_certified_scene

    scene, planar_certificate, notes = compile_certified_scene(prompt, plan, seed)
    return certify_racing_3d_scene(scene, elevation, notes)


def certify_racing_3d_scene(
    scene: SceneSpec, elevation: ElevationSpec | None = None,
    notes: list[str] | None = None,
) -> tuple[SceneSpec, PlayabilityCertificate, list[str]]:
    """Fit and certify elevation on any valid planar scene, including drawings."""
    notes = list(notes or [])
    requested = elevation or ElevationSpec()
    fitted, fit_notes = fit_drivable_elevation(scene, requested)
    notes = [*notes, *fit_notes]
    for candidate in _elevation_ladder(fitted):
        scene_3d = scene.model_copy(update={"elevation": candidate})
        certificate = verify_racing_3d_playability(scene_3d)
        if certificate.playable:
            if candidate is not fitted:
                notes.append(
                    f"Reduced elevation to {candidate.amplitude_m:.1f} m and "
                    f"{candidate.banking_degrees:.0f} degrees of banking so the oracle could "
                    "complete the circuit."
                )
            return scene_3d, certificate, notes
    raise ValueError(
        f"No drivable elevation profile could be certified for {scene.name!r}"
    )


def _elevation_ladder(spec: ElevationSpec):
    """Progressively gentler vertical profiles, cheapest concession first."""
    yield spec
    for scale in (.7, .45, .25):
        yield spec.model_copy(update={
            "amplitude_m": round(spec.amplitude_m * scale, 3),
            "banking_degrees": round(spec.banking_degrees * scale, 2),
        })
    yield spec.model_copy(update={"amplitude_m": 0.0, "banking_degrees": 0.0})


def verify_racing_3d_playability(
    scene: SceneSpec, max_steps: int | None = None,
) -> PlayabilityCertificate:
    """Replay the deterministic oracle through the 3D runtime."""
    validation = validate_racing_scene(scene)
    if validation != ["Racing domain contract passed."]:
        return PlayabilityCertificate(
            verifier="racing-oracle-replay-3d-v1", playable=False, checked_seed=scene.seed,
            failure="; ".join(validation),
        )
    try:
        world = Racing3DWorld.from_scene(scene)
    except ValueError as error:
        return PlayabilityCertificate(
            verifier="racing-oracle-replay-3d-v1", playable=False, checked_seed=scene.seed,
            failure=str(error),
        )
    world.terminate_on_opponent_win = False
    # Gradients cost time, so the 3D budget is deliberately more generous than
    # the planar one rather than failing a circuit for being slow uphill.
    budget = max_steps if max_steps is not None else 2_000 * scene.laps
    controller = RacingLineController()
    controller.reset(scene, scene.seed)
    for _ in range(budget):
        action, decision = controller.act(world.observe())
        world.step(action, decision)
        if world.terminated:
            break
    return PlayabilityCertificate(
        verifier="racing-oracle-replay-3d-v1", playable=world.succeeded,
        checked_seed=scene.seed,
        objective_trace=[
            f"cross:{objective.target_id}"
            for objective in scene.objectives[:world.objective_index]
        ],
        route_steps=world.step_number,
        failure=None if world.succeeded else world.reason or "Oracle exceeded the 3D step budget",
    )


def _nearest_index(points: list[Vec2], point: Vec2) -> int:
    return min(
        range(len(points)),
        key=lambda index: (points[index].x - point.x) ** 2 + (points[index].y - point.y) ** 2,
    )


def _track_bearing(points: list[Vec2], index: int) -> float:
    count = len(points)
    before, after = points[(index - 1) % count], points[(index + 1) % count]
    return math.degrees(math.atan2(after.y - before.y, after.x - before.x)) % 360


def _combine(
    first: tuple[float, float, float], first_scale: float,
    second: tuple[float, float, float], second_scale: float,
) -> tuple[float, float, float]:
    return (
        first[0] * first_scale + second[0] * second_scale,
        first[1] * first_scale + second[1] * second_scale,
        first[2] * first_scale + second[2] * second_scale,
    )

"""The single supported game domain: deterministic top-down circuit racing.

The model chooses within a small circuit grammar. Local code owns geometry,
physics, collision, checkpoint order, and executable verification.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .models import (
    Action, ActionName, CollisionShape, CornerRadius, DecisionRecord, DynamicsSpec,
    EntityKind, EntitySpec,
    FrameRecord, NpcBehaviorSpec, NpcProfile, ObjectiveKind, ObjectiveSpec, ObservationPacket,
    PlayabilityCertificate, PrivilegedState, Rect, SceneSpec, StraightLength, TrackRegion, Vec2,
)
from .collision import (
    Collider, SweepContact, circle_collider, collider_for, edge_barrier_colliders,
)
from .context_loader import load_context_pack
from .providers import ProviderError, active_provider, anthropic_json, configured_model
from .track_grammar import (
    BarrierSpec, CompiledTrack, CornerSpec, NpcSpec, TrackPlan, archetype_plan,
    compile_certified_track, parse_track_prompt, track_plan_schema, validate_track_geometry,
)
from .vehicle_physics import (
    VehiclePhysicsState, apply_dynamics_preset, apply_surface_grip,
    integrate_vehicle_substep, surface_road_dynamics,
)


ENGINE_ID = "racing-2d-v5"
CAR_RADIUS = 11.0
TRACK_WIDTH = 132.0
SCENE_BOUNDS = Rect(x=0, y=0, width=960, height=640)
OFF_TRACK_SPEED_CAP = 3.0
OFF_TRACK_REWARD_PENALTY = 0.02
NITRO_CAPACITY = 100.0
NITRO_RECHARGE_PER_TICK = 1.0
NITRO_DRAIN_PER_TICK = 8.0
NITRO_MAX_SPEED_MULTIPLIER = 1.35
COUNTDOWN_TICKS = 30
NPC_NITRO_CLEARANCE = 90.0
NPC_STRAIGHT_HEADING_DELTA = 8.0
NPC_PASS_LANE_OFFSET = 44.0
NPC_LANE_CHANGE_PER_TICK = 2.5
# Only one car may claim a passing manoeuvre near the player at a time.  This
# keeps a compact grid from sending two cars into the same shoulder of the
# road, which was the source of the visible stop-and-go traffic jams.
NPC_PASS_RESERVATION_STEPS = 18
# A defending opponent may cover a rival's approach lane but never the certified
# centerline: the deterministic racing-line oracle drives lane offset zero, so
# blocking it would make aggressive-traffic scenes fail their own verification.
NPC_MIN_DEFENSIVE_OFFSET = 26.0
# Steering cannot rotate a stationary car, so every traffic hold keeps a floor of
# motion. A car braked to a standstill to avoid something can no longer steer
# around it, and two cars each waiting for the other never move again.
TRAFFIC_CREEP_SPEED = 1.2
# Intelligence shapes the line a car drives rather than how fast it will go.
NPC_CONTACT_CLEARANCE = CAR_RADIUS * 2
"""Centre separation at which two cars are touching."""
NPC_BARRIER_LOOKAHEAD = 6
"""Centerline samples scanned before committing an opponent to a lane."""
NPC_TRAVEL_FRACTIONS = (.75, .5, .25, 0.0)
"""Fallback travel fractions when a full move would put a car into something."""
NPC_APEX_REACH = 30.0
NPC_APEX_TURN_THRESHOLD = 4.0
NPC_APEX_COMMITMENT = .85
NPC_LINE_WANDER = 9.0
BARRIER_RESTITUTION = .22
"""Fraction of inward velocity retained as a small rebound."""
BARRIER_REBOUND_PIXELS = 2.5
BARRIER_COLLISION_PENALTY = .15


class RacingDesignDraft(BaseModel):
    """Legacy archetype selector, retained for the CLI and archived studies.

    New environments author a full `TrackPlan`. A draft is a coarse shorthand
    that expands into one, so the two paths compile through identical geometry.
    """

    title: str = Field(min_length=3, max_length=64)
    rationale: str = Field(min_length=8, max_length=360)
    circuit: str = Field(pattern=r"^(oval|technical|chicane)$")
    surface: str = Field(pattern=r"^(asphalt|clay|ice)$")
    obstacle_count: int = Field(ge=0, le=6)
    edge_barriers: bool = False
    npc_count: int = Field(ge=0, le=3)
    laps: int = Field(default=1, ge=1, le=10)
    npc_start_mode: str = Field(default="grid", pattern=r"^(grid|distributed)$")
    start_region: TrackRegion = TrackRegion.AUTO
    player_grid_position: int = Field(default=1, ge=1, le=6)
    grip: float = Field(default=1.0, ge=.3, le=1.2)
    npc_profile: NpcProfile = NpcProfile.RACER

    def to_plan(self) -> TrackPlan:
        base = archetype_plan(self.circuit, self.surface)
        grid_count = max(
            self.npc_count,
            self.player_grid_position - 1 if self.npc_start_mode == "grid" else 0,
        )
        return base.model_copy(update={
            "title": self.title, "rationale": self.rationale, "laps": self.laps,
            "grip": self.grip, "npc_start_mode": self.npc_start_mode,
            "start_region": self.start_region,
            "player_grid_position": self.player_grid_position,
            "edge_barriers": self.edge_barriers,
            "barriers": [
                BarrierSpec(label=f"track barrier {number}")
                for number in range(1, self.obstacle_count + 1)
            ],
            "npcs": [
                NpcSpec(profile=self.npc_profile, label=f"opponent car {number}")
                for number in range(1, grid_count + 1)
            ],
        })


@dataclass(frozen=True)
class RacingDesignResult:
    plan: TrackPlan
    provider: str
    model: str
    rationale: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @property
    def draft(self) -> TrackPlan:
        """Backwards-compatible alias; the design *is* the plan."""
        return self.plan


RACING_CREATOR_SYSTEM = load_context_pack("environment")


def design_racing_environment(
    prompt: str, provider: str = "auto", feedback: str | None = None,
    guidance: str | None = None,
) -> RacingDesignResult:
    resolved = active_provider() if provider == "auto" else provider
    if resolved == "offline":
        plan = parse_track_prompt(prompt)
        return RacingDesignResult(
            plan=plan, provider="offline", model="track-grammar-v1",
            rationale=plan.rationale,
        )
    if resolved not in {"anthropic", "openai"}:
        raise ProviderError(f"Unknown racing environment provider: {resolved}")
    payload, usage = anthropic_json(
        model=configured_model("ANTHROPIC_ENVIRONMENT_MODEL"),
        max_tokens=2_400,
        system=RACING_CREATOR_SYSTEM,
        prompt=(
            f"Race brief: {prompt}\n\n"
            "Return the track plan. Author 3 to 10 corners, 0 to 6 barriers, and 0 to 5 opponents. "
            "laps is 1 unless the brief names a number; track_width is 110 to 170 pixels (132 is "
            "standard, lower is narrower). Use npc_start_mode=grid unless the brief asks for "
            "spread-out traffic. Give each corner the brief actually describes an explicit "
            "angle_degrees and region, and set angle_degrees to 0 with region=auto on every other "
            "corner so the loop closes and the located corners land where they were asked for."
            + (
                "\n\nYour previous plan for this brief was rejected by the deterministic compiler:\n"
                f"{feedback}\n"
                "Author a different plan that still satisfies the brief. Opening tight radii, "
                "shortening straights, spreading corners into different regions, and removing "
                "barriers all make a circuit easier to close and complete."
                if feedback else ""
            )
            # A caller running a search supplies its own extra block: either a
            # request for a differing proposal, or measured residuals from the
            # scene its last plan compiled into.
            + (f"\n\n{guidance}" if guidance else "")
        ),
        json_schema=track_plan_schema(),
        cache_system=True,
    )
    try:
        plan = TrackPlan.model_validate(_coerce_plan_payload(payload))
    except Exception as error:
        raise ProviderError(
            f"Racing creator returned an invalid track plan: {str(error)[:360]}"
        ) from error
    return RacingDesignResult(
        plan=plan, provider=usage.provider, model=usage.model,
        rationale=plan.rationale, input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens, latency_ms=usage.latency_ms,
    )


def _coerce_plan_payload(payload: dict) -> dict:
    """Clamp a model-authored plan into the grammar without discarding intent.

    Structured output guarantees the shape, not the ranges. Values outside the
    grammar are clamped rather than rejected so one out-of-range number cannot
    waste a generation, and an angle the model left at zero becomes an unset
    angle for the closure solver to fill.
    """
    plan = dict(payload)
    loop_shape = plan.get("loop_shape")
    plan["loop_shape"] = loop_shape if loop_shape in {"cornered", "circle"} else "cornered"
    # Free text is a range too. A creator that explains itself at length was
    # previously discarded outright, which wasted the whole generation over
    # prose rather than over geometry.
    for key, low, high, fallback in (
        ("title", 3, 64, "Compiled circuit"),
        ("rationale", 8, 360, "Compiled from the brief."),
    ):
        text = plan.get(key)
        text = fallback if not isinstance(text, str) or len(text.strip()) < low else text.strip()
        plan[key] = text[:high]
    corners: list[dict] = []
    for item in plan.get("corners") or []:
        if not isinstance(item, dict):
            continue
        corner = dict(item)
        angle = corner.get("angle_degrees")
        if not isinstance(angle, (int, float)) or angle <= 0:
            corner["angle_degrees"] = None
        else:
            corner["angle_degrees"] = min(172.0, max(12.0, float(angle)))
        if corner.get("direction") not in {"left", "right"}:
            corner["direction"] = None
        for key, allowed, fallback in (
            ("radius", {item.value for item in CornerRadius}, "medium"),
            ("region", {item.value for item in TrackRegion}, "auto"),
            ("exit_straight", {item.value for item in StraightLength}, "medium"),
        ):
            if corner.get(key) not in allowed:
                corner[key] = fallback
        corners.append(corner)
    # A circular plan is an explicit zero-corner primitive. Other plans are padded
    # up to the grammar's three-corner minimum rather than rejected.
    corners = corners[:10]
    if plan["loop_shape"] == "circle":
        plan["corners"] = []
    else:
        while len(corners) < 3:
            corners.append({})
        plan["corners"] = corners
    plan["barriers"] = [item for item in (plan.get("barriers") or []) if isinstance(item, dict)][:6]
    plan["npcs"] = [item for item in (plan.get("npcs") or []) if isinstance(item, dict)][:5]
    # Colour words and empty strings are both normal creator output, and `VisualPlan`
    # already coerces them. A malformed block becomes the surface default rather than
    # a rejected plan: nobody should lose a circuit over a swatch.
    visual = plan.get("visual")
    plan["visual"] = visual if isinstance(visual, dict) else {}
    plan["grip"] = min(1.2, max(.3, float(plan.get("grip") or 1.0)))
    plan["track_width"] = min(170.0, max(110.0, float(plan.get("track_width") or 132.0)))
    plan["edge_barriers"] = bool(plan.get("edge_barriers", False))
    plan["laps"] = min(10, max(1, int(plan.get("laps") or 1)))
    plan["start_region"] = (
        plan.get("start_region")
        if plan.get("start_region") in {item.value for item in TrackRegion}
        else TrackRegion.AUTO.value
    )
    plan["player_grid_position"] = min(6, max(1, int(plan.get("player_grid_position") or 1)))
    for npc in plan["npcs"]:
        # `intelligence` belongs here with the others. Leaving it out meant a
        # creator that returned 1.1 for it lost the whole generation to a pydantic
        # rejection, which is exactly the waste this clamping exists to prevent.
        for key, low, high in (
            ("pace", .35, 1.05), ("skill", 0.0, 1.0), ("aggression", 0.0, 1.0),
            ("intelligence", 0.0, 1.0),
        ):
            if isinstance(npc.get(key), (int, float)):
                npc[key] = min(high, max(low, float(npc[key])))
            else:
                npc.pop(key, None)
    return plan


def compile_racing_scene(
    prompt: str, draft: TrackPlan | RacingDesignDraft, seed: int | None = None,
) -> SceneSpec:
    """Compile an authored plan into a complete, deterministic racing scene."""
    plan = draft.to_plan() if isinstance(draft, RacingDesignDraft) else draft
    track, _ = compile_certified_track(plan, SCENE_BOUNDS, CAR_RADIUS)
    return compile_racing_scene_from_track(prompt, plan, track, seed)


def compile_racing_scene_from_track(
    prompt: str, plan: TrackPlan, track: CompiledTrack, seed: int | None = None,
) -> SceneSpec:
    """Build game entities and objectives around already-certified geometry."""
    actual_seed = _stable_seed(prompt) if seed is None else seed
    centerline = track.centerline
    start_index = _start_line_index(track, plan.start_region)
    dynamics = DynamicsSpec(
        road=apply_surface_grip(surface_road_dynamics(plan.surface), plan.grip),
    )

    entities: list[EntitySpec] = []
    sector_indices = _sector_indices_from_start(centerline, track.report.sector_count, start_index)
    for number, point_index in enumerate(sector_indices, start=1):
        finish = point_index == start_index
        point = centerline[point_index]
        entities.append(EntitySpec(
            id="finish-line" if finish else f"sector-{number}",
            kind=EntityKind.CHECKPOINT,
            rect=Rect(x=point.x - 31, y=point.y - 31, width=62, height=62),
            label="finish line" if finish else f"sector {number}",
            color="#ff6b2c" if finish else "#f4d35e",
        ))

    # Barriers sit at the lane edge: real collision objects that narrow one side
    # of the corridor without occluding the certified center racing line.
    barrier_offset = plan.track_width / 2 - 14
    for number, barrier in enumerate(plan.barriers, start=1):
        progress = (
            track.progress_for_region(barrier.region, SCENE_BOUNDS)
            if barrier.region != TrackRegion.AUTO
            else _auto_barrier_progress(number, len(plan.barriers))
        )
        index = _barrier_index(track, progress)
        side = 1.0 if barrier.side == "right" else -1.0 if barrier.side == "left" else (
            -1.0 if number % 2 else 1.0
        )
        point = _offset_track_point(centerline, index, barrier_offset * side)
        barrier_size = {
            "circle": (20.0, 20.0),
            "box": (26.0, 18.0),
            "oriented-box": (54.0, 10.0),
        }[barrier.shape]
        barrier_width, barrier_height = barrier_size
        entities.append(EntitySpec(
            id=f"barrier-{number}", kind=EntityKind.OBSTACLE,
            rect=Rect(
                x=point.x - barrier_width / 2, y=point.y - barrier_height / 2,
                width=barrier_width, height=barrier_height,
            ),
            label=barrier.label or f"track barrier {number}", color="#ff6b2c",
            shape=CollisionShape(barrier.shape),
            # Oriented barriers run along the road rather than across it.
            rotation_degrees=round(_track_heading(centerline, point), 3),
        ))

    behaviors: list[NpcBehaviorSpec] = []
    grid_slots = iter(slot for slot in range(1, len(plan.npcs) + 2) if slot != plan.player_grid_position)
    for number, npc in enumerate(plan.npcs, start=1):
        entity_id = f"opponent-{number}"
        index, lane_offset = _grid_start_slot(
            next(grid_slots), start_index, len(plan.npcs) + 1, plan.npc_start_mode,
            len(centerline), plan.track_width,
        )
        point = _offset_track_point(centerline, index, lane_offset)
        entities.append(EntitySpec(
            id=entity_id, kind=EntityKind.NPC,
            rect=Rect(x=point.x - 9, y=point.y - 15, width=18, height=30),
            label=npc.label or f"opponent car {number}", color="#72a0c1",
        ))
        behaviors.append(npc.resolve(entity_id, grid_index=number - 1))

    player_index, player_lane_offset = _grid_start_slot(
        plan.player_grid_position, start_index, len(plan.npcs) + 1, plan.npc_start_mode,
        len(centerline), plan.track_width,
    )
    player_spawn = _offset_track_point(centerline, player_index, player_lane_offset)

    digest = hashlib.sha256(f"{prompt}\0{actual_seed}".encode()).hexdigest()[:10]
    checkpoints = [entity for entity in entities if entity.kind == EntityKind.CHECKPOINT]
    objectives = [
        ObjectiveSpec(
            kind=ObjectiveKind.REACH,
            target_id=entity.id,
            description=f"Lap {lap}/{plan.laps}: drive through {entity.label}",
        )
        for lap in range(1, plan.laps + 1)
        for entity in checkpoints
    ]
    return SceneSpec(
        id=f"race-{digest}", name=plan.title, prompt=prompt, seed=actual_seed,
        player_spawn=player_spawn, start_line_index=start_index,
        start_line_region=plan.start_region, player_grid_position=plan.player_grid_position,
        entities=entities, objectives=objectives,
        domain_pack_version=ENGINE_ID, track_centerline=centerline,
        track_width=plan.track_width, edge_barriers=plan.edge_barriers,
        laps=plan.laps, surface=plan.surface,
        grip=plan.grip, npc_start_mode=plan.npc_start_mode, npc_behaviors=behaviors,
        sector_count=len(checkpoints), dynamics=dynamics,
        track_report=_annotated_report(track, dynamics),
        # Carried through untouched. The compiler reads nothing from it, which is the
        # property that lets a recolour be verified without re-certifying a lap.
        visual=plan.visual,
    )


def _auto_barrier_progress(number: int, total: int) -> float:
    """Spread unplaced barriers around the lap without clustering them."""
    slots = (.08, .33, .58, .83, .21, .70)
    return slots[(number - 1) % len(slots)] if total <= len(slots) else (number - 1) / total


def _barrier_index(track: CompiledTrack, progress: float) -> int:
    """Place a barrier on the straightest sample near its requested progress.

    A barrier narrows one side of the corridor, so it belongs where the racing
    line is predictable. Dropped mid-corner it would sit exactly where any car
    running the certified line drifts wide, which on a low-grip surface makes the
    circuit unpassable rather than merely harder.
    """
    count = len(track.centerline)
    nominal = int(progress * count) % count
    window = max(2, min(6, count // 12))
    candidates = [
        (nominal + offset) % count
        for offset in range(-window, window + 1)
        if all(
            (nominal + offset - gate) % count > 2 and (gate - nominal - offset) % count > 2
            for gate in track.sector_indices
        )
    ]
    if not candidates:
        return nominal
    return min(
        candidates,
        key=lambda index: (round(track.curvature_at(index), 3), abs(index - nominal)),
    )


def _start_line_index(track: CompiledTrack, region: TrackRegion) -> int:
    """Choose the authoritative start/finish sample in the requested map region."""
    if region == TrackRegion.AUTO:
        return 0  # The grammar already anchors zero at the safest longest straight.
    count = len(track.centerline)
    nominal = round(track.progress_for_region(region, SCENE_BOUNDS) * count) % count
    window = max(2, min(8, count // 18))
    return min(
        ((nominal + offset) % count for offset in range(-window, window + 1)),
        key=lambda index: (round(track.curvature_at(index), 4), abs((index - nominal + count // 2) % count - count // 2)),
    )


def _sector_indices_from_start(
    centerline: list[Vec2], sector_count: int, start_index: int,
) -> tuple[int, ...]:
    """Ordered gates for one lap, measured forward from the authoritative start line."""
    count = len(centerline)

    def curvature(index: int) -> float:
        before = centerline[(index - 2) % count]
        current = centerline[index % count]
        after = centerline[(index + 2) % count]
        return abs(_angle_delta(_bearing(before, current), _bearing(current, after)))

    window = max(1, count // (sector_count * 4))
    gates: list[int] = []
    for sector in range(1, sector_count):
        nominal = (start_index + round(count * sector / sector_count)) % count
        gates.append(min(
            ((nominal + offset) % count for offset in range(-window, window + 1)),
            key=lambda index: (round(curvature(index), 4), abs((index - nominal + count // 2) % count - count // 2)),
        ))
    return (*gates, start_index)


def _grid_start_slot(
    position: int, start_index: int, total: int, start_mode: str, count: int, track_width: float,
) -> tuple[int, float]:
    """Place every competitor behind the same start line, respecting the player's grid slot."""
    if start_mode != "grid":
        return (start_index + int(count * position / (total + 1))) % count, _npc_lane_offset(position)
    if position == 1:
        return (start_index - 2) % count, 0.0
    # The grid is staggered: each competitor gets a longitudinal slot, while
    # alternating sides of the corridor.  A two-abreast row made both cars aim
    # at the next sharply-curved sample together; their straight-line waypoint
    # chords could visibly cross before the collision guard engaged.
    lane = (-1.0 if position % 2 == 0 else 1.0) * min(30.0, track_width * .19)
    return (start_index - position * 2) % count, lane


def _annotated_report(track: CompiledTrack, dynamics: DynamicsSpec):
    """Attach a grip-aware entry speed to every corner in the fidelity report."""
    lateral_grip = (
        dynamics.road.friction_coefficient
        * dynamics.road.lateral_grip_multiplier
        * dynamics.vehicle.tire_friction_multiplier
    )
    conversion = dynamics.pixels_per_meter / dynamics.control_hz
    return track.report.model_copy(update={"corners": [
        corner.model_copy(update={
            "recommended_entry_speed": round(min(
                dynamics.vehicle.max_speed_mps,
                math.sqrt(max(0.0, lateral_grip * dynamics.gravity_mps2
                              * corner.achieved_radius_pixels / dynamics.pixels_per_meter)),
            ) * conversion, 2),
        })
        for corner in track.report.corners
    ]})


def compile_certified_scene(
    prompt: str, plan: TrackPlan, seed: int | None = None,
) -> tuple[SceneSpec, PlayabilityCertificate, list[str]]:
    """Compile a plan into a scene the deterministic oracle can actually finish.

    Geometry repair happens inside the track compiler, but a circuit can be
    valid geometry and still be uncompletable: barriers narrow the corridor
    exactly where a low-grip surface makes the car run wide. Rather than
    discarding the creator's whole plan, walk a short ladder that gives up the
    least important parts of the brief first, and report what was surrendered.
    """
    attempts: list[str] = []
    for candidate, note in _certification_ladder(plan):
        scene = compile_racing_scene(prompt, candidate, seed)
        validation = validate_racing_scene(scene)
        if validation != ["Racing domain contract passed."]:
            attempts.append(f"{note or 'as authored'}: {'; '.join(validation)}")
            continue
        certificate = verify_racing_playability(scene)
        if certificate.playable:
            notes = [note] if note else []
            if notes and scene.track_report is not None:
                scene = scene.model_copy(update={
                    "track_report": scene.track_report.model_copy(update={
                        "relaxations": [*scene.track_report.relaxations, *notes],
                    }),
                })
            return scene, certificate, attempts
        attempts.append(f"{note or 'as authored'}: {certificate.failure}")
    raise ValueError(
        "No variant of the authored circuit could be certified: " + " | ".join(attempts[:4])
    )


def _certification_ladder(plan: TrackPlan):
    """Progressively surrender the least essential parts of a brief."""
    yield plan, ""
    for keep in range(len(plan.barriers) - 1, -1, -1):
        dropped = len(plan.barriers) - keep
        yield plan.model_copy(update={"barriers": plan.barriers[:keep]}), (
            f"Removed {dropped} barrier(s) that made the circuit uncompletable."
        )
    if plan.track_width < 170:
        yield plan.model_copy(update={"track_width": min(170.0, plan.track_width + 20)}), (
            "Widened the corridor because the authored circuit could not be completed."
        )
    if plan.grip < .8:
        yield plan.model_copy(update={"grip": min(1.0, plan.grip + .25)}), (
            "Raised grip because the authored surface could not be driven around this circuit."
        )


def validate_racing_scene(scene: SceneSpec) -> list[str]:
    findings: list[str] = []
    if scene.domain_pack_version != ENGINE_ID:
        findings.append(f"Scene must use {ENGINE_ID}")
    if len(scene.track_centerline) < 8:
        findings.append("Circuit needs at least eight racing-line points")
    if not 96 <= scene.track_width <= 180:
        findings.append("Track width is outside the certified range")
    if scene.surface not in {"asphalt", "clay", "ice"}:
        findings.append("Surface is outside the racing domain catalog")
    if not .3 <= scene.grip <= 1.2:
        findings.append("Grip multiplier is outside the certified range")
    if not 1 <= scene.laps <= 10:
        findings.append("Race length must be between one and ten laps")
    if scene.npc_start_mode not in {"grid", "distributed"}:
        findings.append("NPC start mode must be grid or distributed")
    if not 1 <= scene.player_grid_position <= 6:
        findings.append("Player grid position must be between one and six")
    if not 0 <= scene.start_line_index < max(1, len(scene.track_centerline)):
        findings.append("Start line index must reference the racing line")
    # A `track_report` marks a scene built by the current compiler. Scenes
    # serialized by an earlier one predate uniform sampling and per-opponent
    # behavior, so those two checks would reject replays that are still valid to
    # open. Everything the current compiler produces is held to the full contract.
    compiled_here = scene.track_report is not None
    if compiled_here:
        npc_ids = [entity.id for entity in scene.entities if entity.kind == EntityKind.NPC]
        behavior_ids = [behavior.entity_id for behavior in scene.npc_behaviors]
        if sorted(npc_ids) != sorted(behavior_ids):
            findings.append("Every opponent car must have exactly one serialized behavior")
    if not all(math.isfinite(value) for value in (
        scene.bounds.x, scene.bounds.y, scene.bounds.width, scene.bounds.height,
        scene.player_spawn.x, scene.player_spawn.y, scene.track_width,
    )):
        findings.append("Scene geometry must contain only finite numbers")
    if scene.bounds.width <= 0 or scene.bounds.height <= 0:
        findings.append("Scene bounds must have positive dimensions")
    if any(not math.isfinite(point.x) or not math.isfinite(point.y) for point in scene.track_centerline):
        findings.append("Racing line must contain only finite coordinates")
    if any(
        math.hypot(end.x - start.x, end.y - start.y) <= 1e-6
        for start, end in zip(scene.track_centerline, scene.track_centerline[1:] + scene.track_centerline[:1])
    ):
        findings.append("Racing line cannot contain zero-length segments")
    # The compiler owns corridor geometry; re-check it here so a hand-edited or
    # replayed scene cannot smuggle in an overlapping or unevenly sampled track.
    if compiled_here:
        findings.extend(validate_track_geometry(
            scene.track_centerline, scene.track_width, scene.bounds, CAR_RADIUS,
        ))
    else:
        margin = scene.track_width / 2 + CAR_RADIUS
        if any(
            not (margin <= point.x <= scene.bounds.width - margin)
            or not (margin <= point.y <= scene.bounds.height - margin)
            for point in scene.track_centerline
        ):
            findings.append("Racing line does not leave a safe track margin")
    checkpoints = [entity for entity in scene.entities if entity.kind == EntityKind.CHECKPOINT]
    entity_ids = [entity.id for entity in scene.entities]
    if len(entity_ids) != len(set(entity_ids)):
        findings.append("Entity ids must be unique")
    objective_ids = [objective.target_id for objective in scene.objectives]
    missing_targets = sorted(set(objective_ids) - set(entity_ids))
    if missing_targets:
        findings.append("Objective targets must reference existing entities")
    for entity in scene.entities:
        rect = entity.rect
        values = (rect.x, rect.y, rect.width, rect.height)
        if not all(math.isfinite(value) for value in values):
            findings.append(f"Entity {entity.id} must contain only finite geometry")
            continue
        if rect.width <= 0 or rect.height <= 0:
            findings.append(f"Entity {entity.id} must have positive dimensions")
        if (
            rect.x < scene.bounds.x or rect.y < scene.bounds.y
            or rect.x + rect.width > scene.bounds.x + scene.bounds.width
            or rect.y + rect.height > scene.bounds.y + scene.bounds.height
        ):
            findings.append(f"Entity {entity.id} must remain inside scene bounds")
    expected_objectives = [item.id for _ in range(scene.laps) for item in checkpoints]
    if expected_objectives != objective_ids:
        findings.append("Checkpoint entities and objective order must match")
    if len(checkpoints) != scene.sector_count:
        findings.append("Checkpoint count must match the declared sector count")
    if not 3 <= scene.sector_count <= 9 or (checkpoints and checkpoints[-1].id != "finish-line"):
        findings.append("Circuit requires three to nine ordered gates ending at the finish line")
    if scene.track_centerline and _distance_to_polyline(scene.player_spawn, scene.track_centerline, closed=True) > scene.track_width * .28:
        findings.append("Player grid slot must remain inside the racing corridor")
    for checkpoint in checkpoints:
        if scene.track_centerline and _distance_to_polyline(_rect_center(checkpoint.rect), scene.track_centerline, closed=True) > 1.0:
            findings.append(f"Checkpoint {checkpoint.id} must be centered on the racing line")
    return findings or ["Racing domain contract passed."]


def verify_racing_playability(scene: SceneSpec, max_steps: int | None = None) -> PlayabilityCertificate:
    validation = validate_racing_scene(scene)
    if validation != ["Racing domain contract passed."]:
        return PlayabilityCertificate(
            verifier="racing-oracle-replay-v5", playable=False, checked_seed=scene.seed,
            failure="; ".join(validation),
        )
    world = RacingWorld.from_scene(scene)
    # Certification proves route completion, not race position. Let the
    # reference driver finish even if an NPC crosses first; normal worlds retain
    # competitive first-finisher termination.
    world.terminate_on_opponent_win = False
    step_budget = max_steps if max_steps is not None else 1_400 * scene.laps
    controller = RacingLineController()
    controller.reset(scene, scene.seed)
    for _ in range(step_budget):
        action, decision = controller.act(world.observe())
        world.step(action, decision)
        if world.terminated:
            break
    return PlayabilityCertificate(
        verifier="racing-oracle-replay-v5", playable=world.succeeded,
        checked_seed=scene.seed,
        objective_trace=[f"cross:{objective.target_id}" for objective in scene.objectives[:world.objective_index]],
        route_steps=world.step_number,
        failure=None if world.succeeded else world.reason or "Oracle exceeded the step budget",
    )


OPPONENT_PHASES = frozenset({"cruise", "passing", "merge", "defending"})


@dataclass
class OpponentState:
    entity_id: str
    position: Vec2
    target_index: int
    lane_offset: float
    behavior: NpcBehaviorSpec
    track_index: int = 0
    base_lane_offset: float = 34.0
    target_lane_offset: float = 34.0
    heading: float = 0.0
    overtake_phase: str = "cruise"
    pass_clear_ticks: int = 0
    speed: float = 0.0
    nitro: float = 0.0
    nitro_active: bool = False
    progress_samples: float = 0.0
    """Centerline samples travelled, which is this car's race distance."""
    completed_laps: int = 0
    """Fully completed laps, each earned through the ordered gate sequence."""
    checkpoint_index: int = 0
    """The next sector gate this opponent must cross to progress its current lap."""
    finished_step: int | None = None
    lane_phase: float = 0.0
    """Fixed per-car phase for line wander, derived from the id rather than drawn."""


@dataclass
class RacingWorld:
    scene: SceneSpec
    player: Vec2
    heading: float
    dynamics: DynamicsSpec
    speed: float = 0.0
    nitro: float = 0.0
    nitro_active: bool = False
    turning: bool = False
    held_keys: tuple[str, ...] = ()
    step_number: int = 0
    objective_index: int = 0
    terminated: bool = False
    succeeded: bool = False
    reason: str | None = None
    delayed_keys: tuple[str, ...] | None = None
    fog: bool = False
    opponents: list[OpponentState] = field(default_factory=list)
    finish_order: list[str] = field(default_factory=list)
    """Entity ids in the order they completed the race; "player" is the human/policy."""
    terminate_on_opponent_win: bool = True
    """Competitive play ends at the winner; certification can inspect a full field."""
    obstacle_shift: float = 0.0
    off_track: bool = False
    countdown_ticks_remaining: int = COUNTDOWN_TICKS
    longitudinal_velocity_mps: float = 0.0
    lateral_velocity_mps: float = 0.0
    yaw_rate_radians_per_second: float = 0.0
    steering_angle_radians: float = 0.0
    longitudinal_acceleration_mps2: float = 0.0
    lateral_acceleration_mps2: float = 0.0
    slip_angle_radians: float = 0.0
    aerodynamic_drag_n: float = 0.0
    rolling_resistance_n: float = 0.0
    lateral_load_transfer_n: float = 0.0
    barrier_impact: Vec2 | None = None
    edge_colliders: tuple[tuple[str, Collider], ...] = ()

    @classmethod
    def from_scene(cls, scene: SceneSpec, perturbation: str | None = None) -> "RacingWorld":
        validation = validate_racing_scene(scene)
        if validation != ["Racing domain contract passed."]:
            raise ValueError("RacingWorld rejected invalid scene: " + "; ".join(validation))
        start_index = min(max(0, scene.start_line_index), len(scene.track_centerline) - 1)
        heading = _bearing(
            scene.track_centerline[start_index],
            scene.track_centerline[(start_index + 1) % len(scene.track_centerline)],
        )
        behaviors = {behavior.entity_id: behavior for behavior in scene.npc_behaviors}
        opponents = []
        npc_number = 0
        for entity in scene.entities:
            if entity.kind == EntityKind.NPC:
                npc_number += 1
                center = Vec2(x=entity.rect.x + entity.rect.width / 2, y=entity.rect.y + entity.rect.height / 2)
                track_index = _nearest_point_index(scene.track_centerline, center)
                lane_offset = _npc_lane_offset(npc_number)
                opponents.append(OpponentState(
                    entity_id=entity.id,
                    position=center,
                    target_index=(track_index + 1) % len(scene.track_centerline),
                    lane_offset=lane_offset,
                    behavior=behaviors.get(entity.id) or NpcBehaviorSpec(entity_id=entity.id),
                    track_index=track_index,
                    base_lane_offset=lane_offset,
                    target_lane_offset=lane_offset,
                    heading=_track_heading(scene.track_centerline, center),
                    lane_phase=_stable_phase(entity.id),
                ))
        return cls(
            scene=scene, player=scene.player_spawn.model_copy(), heading=heading,
            dynamics=apply_dynamics_preset(scene.dynamics, perturbation or "normal"),
            fog=perturbation == "fog", opponents=opponents,
            obstacle_shift=18.0 if perturbation == "obstacle_shift" else 0.0,
            countdown_ticks_remaining=scene.dynamics.control_hz * 3,
            edge_colliders=tuple(edge_barrier_colliders(scene)),
        )

    def observe(self) -> ObservationPacket:
        task = self.scene.objectives[min(self.objective_index, len(self.scene.objectives) - 1)].description
        nearby: list[dict[str, object]] = []
        opponent_states = {opponent.entity_id: opponent for opponent in self.opponents}
        player_track_index = _nearest_point_index(self.scene.track_centerline, self.player)
        for entity in self.scene.entities:
            opponent = opponent_states.get(entity.id)
            center = opponent.position if opponent is not None else _rect_center(entity.rect)
            distance = math.hypot(center.x - self.player.x, center.y - self.player.y)
            if distance <= 220:
                item: dict[str, object] = {
                    "id": entity.id, "kind": entity.kind.value,
                    "distance": round(distance, 1),
                    "bearing": round(_angle_delta(self.heading, _bearing(self.player, center)), 1),
                }
                if entity.kind == EntityKind.OBSTACLE:
                    # Which side of the corridor a barrier narrows is the fact a
                    # driver needs; distance and bearing alone do not give it.
                    item["lane_offset"] = round(_signed_lane_offset(
                        self.scene.track_centerline, center,
                        _nearest_point_index(self.scene.track_centerline, center),
                    ), 1)
                if opponent is not None:
                    item.update({
                        "track_steps_ahead": _cyclic_index_delta(
                            player_track_index, opponent.track_index, len(self.scene.track_centerline),
                        ),
                        "lane_offset": opponent.lane_offset,
                        "heading": round(opponent.heading, 2),
                        "overtake_phase": opponent.overtake_phase,
                        # Temperament is part of the public scene definition, so a
                        # driver can anticipate an aggressor instead of only
                        # reacting after it has already committed to a move.
                        "profile": opponent.behavior.profile.value,
                        "aggression": opponent.behavior.aggression,
                        "defends": opponent.behavior.defends,
                        "speed": round(opponent.speed, 2),
                        "nitro": round(opponent.nitro, 1),
                        "nitro_active": opponent.nitro_active,
                        "nitro_ready": opponent.nitro >= NITRO_CAPACITY,
                    })
                nearby.append(item)
        target_objective = self.scene.objectives[min(
            self.objective_index, len(self.scene.objectives) - 1,
        )]
        target = next(
            entity for entity in self.scene.entities
            if entity.id == target_objective.target_id
        )
        target_center = _rect_center(target.rect)
        nearby.append({
            "id": "driving-target", "kind": "racing_line", "distance": round(math.hypot(target_center.x - self.player.x, target_center.y - self.player.y), 1),
            "bearing": round(_angle_delta(self.heading, _bearing(self.player, target_center)), 1),
        })
        return ObservationPacket(
            step=self.step_number, task=task,
            rgb_hint="fogged top-down racing frame" if self.fog else "top-down racing frame",
            proprioception=self.player.model_copy(), heading=round(self.heading, 2),
            speed=round(self.speed, 2), nitro=round(self.nitro, 1),
            nitro_active=self.nitro_active,
            nitro_ready=self.nitro >= NITRO_CAPACITY,
            countdown_ticks_remaining=self.countdown_ticks_remaining,
            dynamics=self.dynamics.model_copy(deep=True),
            longitudinal_speed_mps=round(self.longitudinal_velocity_mps, 3),
            lateral_speed_mps=round(self.lateral_velocity_mps, 3),
            yaw_rate_degrees_per_second=round(math.degrees(self.yaw_rate_radians_per_second), 2),
            steering_angle_degrees=round(math.degrees(self.steering_angle_radians), 2),
            longitudinal_acceleration_mps2=round(self.longitudinal_acceleration_mps2, 3),
            lateral_acceleration_mps2=round(self.lateral_acceleration_mps2, 3),
            slip_angle_degrees=round(math.degrees(self.slip_angle_radians), 2),
            checkpoint_index=self.objective_index,
            local_entities=nearby,
        )

    def render_policy_frame(self):
        """Return a public overhead RGB observation for visuomotor policies."""
        from .vision import render_racing_policy_frame

        return render_racing_policy_frame(self)

    def step(self, requested_action: Action, decision: DecisionRecord | None = None, action_delay: bool = False) -> FrameRecord:
        if self.terminated:
            raise RuntimeError(f"Cannot step a terminated race ({self.reason or 'no reason recorded'}).")
        keys = tuple(requested_action.keys) or _keys_for_action(requested_action.name)
        events: list[str] = []
        if self.countdown_ticks_remaining > 0:
            self.held_keys = ()
            self.nitro_active = False
            self.turning = False
            display_before = math.ceil(self.countdown_ticks_remaining / self.dynamics.control_hz)
            self.countdown_ticks_remaining -= 1
            display_after = math.ceil(self.countdown_ticks_remaining / self.dynamics.control_hz)
            if self.countdown_ticks_remaining == 0:
                events.append("go")
            elif display_after != display_before:
                events.append(f"countdown: {display_after}")
            self.step_number += 1
            return FrameRecord(
                step=self.step_number, observation=self.observe(), privileged_state=self.privileged_state(),
                action=ActionName.IDLE, keys=[], reward=0.0, events=events, decision=None,
            )
        if action_delay:
            previous, self.delayed_keys = self.delayed_keys, keys
            keys = previous or ()
            events.append("control input delayed")
        action = _primary_action(keys)
        reward = -0.001
        previous_player = self.player.model_copy()
        self.barrier_impact = None
        self.held_keys = keys
        self._drive(keys)
        self._move_opponents()
        # Race termination is a world invariant, rather than an incidental side
        # effect of one NPC's movement update. This also closes the replay seam:
        # a restored state that already records an NPC winner cannot run on.
        self._terminate_for_first_finisher()
        if self.terminated:
            events.append(self.reason or "race ended")
            reward -= 1.0
            self.step_number += 1
            return FrameRecord(
                step=self.step_number, observation=self.observe(),
                privileged_state=self.privileged_state(), action=action,
                keys=list(keys), reward=round(reward, 4), events=events,
                decision=decision,
            )
        barrier_collision = self._barrier_collision(previous_player)
        collision = None
        if barrier_collision:
            barrier_id, collider, contact = barrier_collision
            self._bounce_player_from_barrier(collider, contact)
            self.barrier_impact = Vec2(
                x=contact.impact_point[0], y=contact.impact_point[1],
            )
            events.append(f"bounced off {barrier_id}")
            reward -= BARRIER_COLLISION_PENALTY
        else:
            collision = self._opponent_collision(previous_player)
        if collision:
            self.terminated, self.reason = True, f"collision with {collision}"
            events.append(self.reason)
            reward -= 1.0
        else:
            on_track = self._on_track(self.player)
            if not on_track:
                self.speed = min(self.speed, OFF_TRACK_SPEED_CAP)
                reward -= OFF_TRACK_REWARD_PENALTY
                if not self.off_track:
                    events.append("left track: terrain slowdown")
            elif self.off_track:
                events.append("returned to track")
            self.off_track = not on_track
            reward += self._check_checkpoint(
                events, on_track=on_track, previous_player=previous_player,
            )
        self.step_number += 1
        return FrameRecord(
            step=self.step_number, observation=self.observe(), privileged_state=self.privileged_state(),
            action=action, keys=list(keys), reward=round(reward, 4), events=events, decision=decision,
        )

    def privileged_state(self) -> PrivilegedState:
        opponent_states = {item.entity_id: item for item in self.opponents}
        sectors = self.scene.sector_count
        current_lap_progress = (
            sectors if self.succeeded else self.objective_index % sectors
        )
        completed_targets = {
            checkpoint.id
            for checkpoint in [
                entity for entity in self.scene.entities if entity.kind == EntityKind.CHECKPOINT
            ][:current_lap_progress]
        }
        entities = []
        for entity in self.scene.entities:
            rect = entity.rect
            opponent = opponent_states.get(entity.id)
            if opponent is not None:
                point = opponent.position
                x, y = point.x - rect.width / 2, point.y - rect.height / 2
            else:
                x, y = rect.x, rect.y + (self.obstacle_shift if entity.kind == EntityKind.OBSTACLE else 0)
            runtime_entity = {
                "id": entity.id, "kind": entity.kind.value, "x": x, "y": y,
                "width": rect.width, "height": rect.height, "active": True,
                "open": entity.id in completed_targets,
            }
            if opponent is not None:
                runtime_entity.update({
                    "speed": round(opponent.speed, 2), "nitro": round(opponent.nitro, 1),
                    "nitro_active": opponent.nitro_active,
                    "nitro_ready": opponent.nitro >= NITRO_CAPACITY,
                    "heading": round(opponent.heading, 2),
                    "lane_offset": round(opponent.lane_offset, 2),
                    "track_index": opponent.track_index,
                    "overtake_phase": opponent.overtake_phase,
                })
            entities.append(runtime_entity)
        return PrivilegedState(
            step=self.step_number, player=self.player.model_copy(), inventory=[],
            objective_index=self.objective_index, entities=entities,
            heading=round(self.heading, 2), speed=round(self.speed, 2),
            nitro=round(self.nitro, 1), nitro_active=self.nitro_active,
            nitro_ready=self.nitro >= NITRO_CAPACITY,
            countdown_ticks_remaining=self.countdown_ticks_remaining,
            longitudinal_velocity_mps=round(self.longitudinal_velocity_mps, 3),
            lateral_velocity_mps=round(self.lateral_velocity_mps, 3),
            yaw_rate_degrees_per_second=round(math.degrees(self.yaw_rate_radians_per_second), 2),
            steering_angle_degrees=round(math.degrees(self.steering_angle_radians), 2),
            longitudinal_acceleration_mps2=round(self.longitudinal_acceleration_mps2, 3),
            lateral_acceleration_mps2=round(self.lateral_acceleration_mps2, 3),
            slip_angle_degrees=round(math.degrees(self.slip_angle_radians), 2),
            aerodynamic_drag_n=round(self.aerodynamic_drag_n, 2),
            rolling_resistance_n=round(self.rolling_resistance_n, 2),
            lateral_load_transfer_n=round(self.lateral_load_transfer_n, 2),
            turning=self.turning,
            lap=min(self.scene.laps, self.objective_index // sectors),
            barrier_impact=self.barrier_impact.model_copy() if self.barrier_impact else None,
        )

    def snapshot(self) -> dict:
        return {
            "step": self.step_number, "player": self.player.model_dump(), "heading": self.heading,
            "speed": self.speed, "nitro": self.nitro,
            "nitro_active": self.nitro_active, "turning": self.turning,
            "countdown_ticks_remaining": self.countdown_ticks_remaining,
            "longitudinal_velocity_mps": self.longitudinal_velocity_mps,
            "lateral_velocity_mps": self.lateral_velocity_mps,
            "yaw_rate_radians_per_second": self.yaw_rate_radians_per_second,
            "steering_angle_radians": self.steering_angle_radians,
            "longitudinal_acceleration_mps2": self.longitudinal_acceleration_mps2,
            "lateral_acceleration_mps2": self.lateral_acceleration_mps2,
            "slip_angle_radians": self.slip_angle_radians,
            "aerodynamic_drag_n": self.aerodynamic_drag_n,
            "rolling_resistance_n": self.rolling_resistance_n,
            "lateral_load_transfer_n": self.lateral_load_transfer_n,
            "barrier_impact": self.barrier_impact.model_dump() if self.barrier_impact else None,
            "held_keys": list(self.held_keys), "objective_index": self.objective_index,
            "terminated": self.terminated, "succeeded": self.succeeded, "reason": self.reason,
            "off_track": self.off_track,
            "delayed_keys": list(self.delayed_keys) if self.delayed_keys is not None else None,
            "fog": self.fog, "obstacle_shift": self.obstacle_shift,
            "finish_order": list(self.finish_order),
            "terminate_on_opponent_win": self.terminate_on_opponent_win,
            "opponents": [{
                "entity_id": item.entity_id, "position": item.position.model_dump(),
                "target_index": item.target_index, "lane_offset": item.lane_offset,
                "track_index": item.track_index,
                "base_lane_offset": item.base_lane_offset,
                "target_lane_offset": item.target_lane_offset,
                "heading": item.heading,
                "overtake_phase": item.overtake_phase,
                "pass_clear_ticks": item.pass_clear_ticks,
                "speed": item.speed, "nitro": item.nitro, "nitro_active": item.nitro_active,
                "progress_samples": item.progress_samples, "completed_laps": item.completed_laps,
                "checkpoint_index": item.checkpoint_index,
                "finished_step": item.finished_step,
                "lane_phase": item.lane_phase,
            } for item in self.opponents],
        }

    def restore(self, snapshot: dict) -> None:
        # Parse into locals first. Restore is used by replay forks, so failure
        # must leave the existing world completely untouched.
        step_number = int(snapshot["step"])
        player = Vec2.model_validate(snapshot["player"])
        raw_heading = float(snapshot.get("heading", self.heading))
        heading = raw_heading % 360
        speed = float(snapshot.get("speed", 0))
        nitro = float(snapshot.get("nitro", 0.0))
        if step_number < 0:
            raise ValueError("Snapshot step cannot be negative")
        if not all(math.isfinite(value) for value in (player.x, player.y, raw_heading, speed, nitro)):
            raise ValueError("Snapshot contains non-finite racing state")
        if speed < 0:
            raise ValueError("Snapshot speed cannot be negative")
        if not 0 <= nitro <= NITRO_CAPACITY:
            raise ValueError("Snapshot nitro is outside the valid range")
        objective_index = int(snapshot["objective_index"])
        if not 0 <= objective_index <= len(self.scene.objectives):
            raise ValueError("Snapshot objective index is outside the race objective range")
        terminated = bool(snapshot.get("terminated", False))
        succeeded = bool(snapshot.get("succeeded", False))
        if succeeded and not terminated:
            raise ValueError("A successful snapshot must also be terminated")
        reason = snapshot.get("reason")
        off_track = bool(snapshot.get("off_track", not self._on_track(player)))
        if "delayed_keys" in snapshot:
            delayed_keys = tuple(snapshot["delayed_keys"]) if snapshot["delayed_keys"] is not None else None
        else:
            legacy = ActionName(snapshot["delayed_action"]) if snapshot.get("delayed_action") else None
            delayed_keys = _keys_for_action(legacy) if legacy else None
        if delayed_keys is not None:
            delayed_keys = tuple(Action(keys=list(delayed_keys)).keys)
        fog = bool(snapshot.get("fog", False))
        valid_finishers = {"player", *(
            entity.id for entity in self.scene.entities if entity.kind == EntityKind.NPC
        )}
        finish_order = [str(item) for item in snapshot.get("finish_order", [])]
        if len(set(finish_order)) != len(finish_order) or not set(finish_order) <= valid_finishers:
            raise ValueError("Snapshot finish order must uniquely reference racers in this scene")
        terminate_on_opponent_win = bool(snapshot.get(
            "terminate_on_opponent_win", self.terminate_on_opponent_win,
        ))
        nitro_active = bool(snapshot.get("nitro_active", False))
        turning = bool(snapshot.get("turning", False))
        held_keys = tuple(Action(keys=list(snapshot.get("held_keys", []))).keys)
        obstacle_shift = float(snapshot.get("obstacle_shift", 0))
        if not math.isfinite(obstacle_shift):
            raise ValueError("Snapshot obstacle shift must be finite")
        countdown_ticks_remaining = int(snapshot.get("countdown_ticks_remaining", 0))
        if not 0 <= countdown_ticks_remaining <= COUNTDOWN_TICKS:
            raise ValueError("Snapshot countdown is outside the valid range")
        velocity_scale = self.dynamics.control_hz / self.dynamics.pixels_per_meter
        longitudinal_velocity_mps = float(snapshot.get(
            "longitudinal_velocity_mps", speed * velocity_scale,
        ))
        lateral_velocity_mps = float(snapshot.get("lateral_velocity_mps", 0.0))
        yaw_rate_radians_per_second = float(snapshot.get("yaw_rate_radians_per_second", 0.0))
        steering_angle_radians = float(snapshot.get("steering_angle_radians", 0.0))
        longitudinal_acceleration_mps2 = float(snapshot.get("longitudinal_acceleration_mps2", 0.0))
        lateral_acceleration_mps2 = float(snapshot.get("lateral_acceleration_mps2", 0.0))
        slip_angle_radians = float(snapshot.get("slip_angle_radians", 0.0))
        aerodynamic_drag_n = float(snapshot.get("aerodynamic_drag_n", 0.0))
        rolling_resistance_n = float(snapshot.get("rolling_resistance_n", 0.0))
        lateral_load_transfer_n = float(snapshot.get("lateral_load_transfer_n", 0.0))
        barrier_impact = (
            Vec2.model_validate(snapshot["barrier_impact"])
            if snapshot.get("barrier_impact") is not None else None
        )
        dynamic_values = (
            longitudinal_velocity_mps, lateral_velocity_mps, yaw_rate_radians_per_second,
            steering_angle_radians, longitudinal_acceleration_mps2, lateral_acceleration_mps2,
            slip_angle_radians, aerodynamic_drag_n, rolling_resistance_n,
            lateral_load_transfer_n,
        )
        if not all(math.isfinite(value) for value in dynamic_values):
            raise ValueError("Snapshot contains non-finite vehicle dynamics")
        if barrier_impact is not None and not all(math.isfinite(value) for value in (
            barrier_impact.x, barrier_impact.y,
        )):
            raise ValueError("Snapshot contains a non-finite barrier impact")
        if longitudinal_velocity_mps < 0:
            raise ValueError("Snapshot longitudinal velocity cannot be negative")
        opponents = [
            OpponentState(
                entity_id=item.entity_id,
                position=item.position.model_copy(),
                target_index=item.target_index,
                lane_offset=item.lane_offset,
                behavior=item.behavior,
                track_index=item.track_index,
                base_lane_offset=item.base_lane_offset,
                target_lane_offset=item.target_lane_offset,
                heading=item.heading,
                overtake_phase=item.overtake_phase,
                pass_clear_ticks=item.pass_clear_ticks,
                speed=item.speed,
                nitro=item.nitro,
                nitro_active=item.nitro_active,
                progress_samples=item.progress_samples,
                completed_laps=item.completed_laps,
                checkpoint_index=item.checkpoint_index,
                finished_step=item.finished_step,
                lane_phase=item.lane_phase,
            )
            for item in self.opponents
        ]
        if "opponents" in snapshot:
            opponents = []
            seen_opponents: set[str] = set()
            valid_opponents = {
                entity.id for entity in self.scene.entities if entity.kind == EntityKind.NPC
            }
            # Behavior is scene-static, so a restored snapshot always re-reads it
            # from the scene rather than trusting a replayed copy of it.
            scene_behaviors = {
                behavior.entity_id: behavior for behavior in self.scene.npc_behaviors
            }
            for item in snapshot.get("opponents") or []:
                position = Vec2.model_validate(item["position"])
                entity_id = str(item["entity_id"])
                if entity_id not in valid_opponents or entity_id in seen_opponents:
                    raise ValueError("Snapshot opponents must uniquely reference scene NPCs")
                seen_opponents.add(entity_id)
                target_index = int(item.get("target_index", 0))
                if "target_index" not in item:
                    target_index = (_nearest_point_index(self.scene.track_centerline, position) + 1) % len(self.scene.track_centerline)
                lane_offset = float(item.get("lane_offset", 34.0))
                track_index = int(item.get(
                    "track_index", _nearest_point_index(self.scene.track_centerline, position),
                )) % len(self.scene.track_centerline)
                base_lane_offset = float(item.get("base_lane_offset", lane_offset))
                target_lane_offset = float(item.get("target_lane_offset", base_lane_offset))
                opponent_heading = float(item.get(
                    "heading", _track_heading(self.scene.track_centerline, position),
                )) % 360
                overtake_phase = str(item.get("overtake_phase", "cruise"))
                pass_clear_ticks = int(item.get("pass_clear_ticks", 0))
                opponent_speed = float(item.get("speed", 0.0))
                opponent_nitro = float(item.get("nitro", 0.0))
                opponent_nitro_active = bool(item.get("nitro_active", False))
                opponent_progress = float(item.get("progress_samples", 0.0))
                raw_finished = item.get("finished_step")
                opponent_finished_step = None if raw_finished is None else int(raw_finished)
                completed_laps = int(item.get(
                    "completed_laps",
                    self.scene.laps if opponent_finished_step is not None else 0,
                ))
                checkpoint_index = int(item.get("checkpoint_index", 0))
                if opponent_progress < 0 or not math.isfinite(opponent_progress):
                    raise ValueError("Snapshot opponent progress must be finite and non-negative")
                if opponent_finished_step is not None and opponent_finished_step < 0:
                    raise ValueError("Snapshot opponent finish step cannot be negative")
                if not 0 <= completed_laps <= self.scene.laps:
                    raise ValueError("Snapshot opponent completed laps are outside the race range")
                if not 0 <= checkpoint_index < self.scene.sector_count:
                    raise ValueError("Snapshot opponent checkpoint index is outside the sector range")
                if not all(math.isfinite(value) for value in (
                    position.x, position.y, lane_offset, base_lane_offset,
                    target_lane_offset, opponent_heading, opponent_speed, opponent_nitro,
                )):
                    raise ValueError("Snapshot opponent state must be finite")
                if (
                    opponent_speed < 0 or not 0 <= opponent_nitro <= NITRO_CAPACITY
                    or overtake_phase not in OPPONENT_PHASES
                    or pass_clear_ticks < 0
                ):
                    raise ValueError("Snapshot opponent speed or nitro is outside the valid range")
                opponents.append(OpponentState(
                    entity_id=entity_id,
                    position=position,
                    target_index=target_index % len(self.scene.track_centerline),
                    lane_offset=lane_offset,
                    behavior=scene_behaviors.get(entity_id) or NpcBehaviorSpec(entity_id=entity_id),
                    track_index=track_index,
                    base_lane_offset=base_lane_offset,
                    target_lane_offset=target_lane_offset,
                    heading=opponent_heading,
                    overtake_phase=overtake_phase,
                    pass_clear_ticks=pass_clear_ticks,
                    speed=opponent_speed,
                    nitro=opponent_nitro,
                    nitro_active=opponent_nitro_active,
                    progress_samples=opponent_progress,
                    completed_laps=completed_laps,
                    checkpoint_index=checkpoint_index,
                    finished_step=opponent_finished_step,
                    lane_phase=float(item.get("lane_phase", _stable_phase(entity_id))),
                ))

        self.step_number = step_number
        self.player, self.heading, self.speed, self.nitro = player, heading, speed, nitro
        self.nitro_active, self.turning, self.held_keys = nitro_active, turning, held_keys
        self.objective_index = objective_index
        self.terminated, self.succeeded, self.reason = terminated, succeeded, reason
        self.off_track, self.delayed_keys = off_track, delayed_keys
        self.fog, self.obstacle_shift, self.opponents = fog, obstacle_shift, opponents
        self.finish_order = finish_order
        self.terminate_on_opponent_win = terminate_on_opponent_win
        self.countdown_ticks_remaining = countdown_ticks_remaining
        self.barrier_impact = barrier_impact
        (
            self.longitudinal_velocity_mps, self.lateral_velocity_mps,
            self.yaw_rate_radians_per_second, self.steering_angle_radians,
            self.longitudinal_acceleration_mps2, self.lateral_acceleration_mps2,
            self.slip_angle_radians, self.aerodynamic_drag_n,
            self.rolling_resistance_n, self.lateral_load_transfer_n,
        ) = dynamic_values

    @property
    def field_size(self) -> int:
        """Cars in the race: the player plus every opponent."""
        return 1 + len(self.opponents)

    @property
    def player_position(self) -> int | None:
        """Finishing position, or None while the player is still racing."""
        if "player" not in self.finish_order:
            return None
        return self.finish_order.index("player") + 1

    @property
    def live_position(self) -> int:
        """Current running order, counting opponents genuinely ahead on distance.

        The player's own progress is measured in ordered gate crossings, which is
        authoritative, so opponent distance is converted to the same scale rather
        than comparing two different units.
        """
        if (finished := self.player_position) is not None:
            return finished
        samples = len(self.scene.track_centerline)
        sectors = max(1, self.scene.sector_count)
        ahead = sum(
            1 for opponent in self.opponents
            if opponent.progress_samples / max(1, samples) * sectors > self.objective_index
        )
        return ahead + 1

    def road_attitude(self, point: Vec2) -> tuple[float, float]:
        """Uphill grade and cross-slope bank of the road under `point`, in radians.

        The 2D engine drives a perfectly flat plane and always returns zeros, so
        the planar physics is unchanged. `Racing3DWorld` overrides this to sample
        its elevation profile, which is the single seam between the two engines:
        every other rule — checkpoints, laps, opponents, collisions, nitro, the
        countdown — is shared code rather than a parallel implementation.
        """
        return (0.0, 0.0)

    def _drive(self, keys: tuple[str, ...]) -> None:
        on_track = self._on_track(self.player)
        self.turning = "a" in keys or "d" in keys
        nitro_requested = "space" in keys and "w" in keys and not self.turning and on_track
        self.nitro_active = nitro_requested and self.nitro > 0 and (
            self.nitro_active or self.nitro >= NITRO_CAPACITY
        )
        if self.nitro_active:
            self.nitro = max(0.0, self.nitro - NITRO_DRAIN_PER_TICK)
        if not self.nitro_active:
            self.nitro = min(NITRO_CAPACITY, self.nitro + NITRO_RECHARGE_PER_TICK)

        # Tests, forks, and integrations may seed the public scalar speed. Keep
        # that compatibility boundary while the authoritative state remains SI.
        displayed_speed = (
            math.hypot(self.longitudinal_velocity_mps, self.lateral_velocity_mps)
            * self.dynamics.pixels_per_meter / self.dynamics.control_hz
        )
        if abs(self.speed - displayed_speed) > 1e-6:
            self.longitudinal_velocity_mps = (
                self.speed * self.dynamics.control_hz / self.dynamics.pixels_per_meter
            )
            self.lateral_velocity_mps = 0.0

        state = VehiclePhysicsState(
            x=self.player.x, y=self.player.y, heading_radians=math.radians(self.heading),
            longitudinal_velocity_mps=self.longitudinal_velocity_mps,
            lateral_velocity_mps=self.lateral_velocity_mps,
            yaw_rate_radians_per_second=self.yaw_rate_radians_per_second,
            steering_angle_radians=self.steering_angle_radians,
            longitudinal_acceleration_mps2=self.longitudinal_acceleration_mps2,
            lateral_acceleration_mps2=self.lateral_acceleration_mps2,
            slip_angle_radians=self.slip_angle_radians,
            aerodynamic_drag_n=self.aerodynamic_drag_n,
            rolling_resistance_n=self.rolling_resistance_n,
            lateral_load_transfer_n=self.lateral_load_transfer_n,
        )
        steering_input = -1.0 if "a" in keys else 1.0 if "d" in keys else 0.0
        substeps = self.dynamics.physics_hz // self.dynamics.control_hz
        for _ in range(substeps):
            grade_radians, bank_radians = self.road_attitude(Vec2(x=state.x, y=state.y))
            state = integrate_vehicle_substep(
                state, self.dynamics,
                throttle=1.0 if "w" in keys else 0.0,
                brake=1.0 if "s" in keys else 0.0,
                steering=steering_input,
                nitro=self.nitro_active,
                on_track=self._on_track(Vec2(x=state.x, y=state.y)),
                grade_radians=grade_radians,
                bank_radians=bank_radians,
            )
            if not self._on_track(Vec2(x=state.x, y=state.y)):
                off_track_mps = OFF_TRACK_SPEED_CAP * self.dynamics.control_hz / self.dynamics.pixels_per_meter
                magnitude = math.hypot(
                    state.longitudinal_velocity_mps, state.lateral_velocity_mps,
                )
                if magnitude > off_track_mps:
                    scale = off_track_mps / magnitude
                    state.longitudinal_velocity_mps *= scale
                    state.lateral_velocity_mps *= scale

        self.player = Vec2(x=state.x, y=state.y)
        self.heading = math.degrees(state.heading_radians) % 360
        self.longitudinal_velocity_mps = state.longitudinal_velocity_mps
        self.lateral_velocity_mps = state.lateral_velocity_mps
        self.yaw_rate_radians_per_second = state.yaw_rate_radians_per_second
        self.steering_angle_radians = state.steering_angle_radians
        self.longitudinal_acceleration_mps2 = state.longitudinal_acceleration_mps2
        self.lateral_acceleration_mps2 = state.lateral_acceleration_mps2
        self.slip_angle_radians = state.slip_angle_radians
        self.aerodynamic_drag_n = state.aerodynamic_drag_n
        self.rolling_resistance_n = state.rolling_resistance_n
        self.lateral_load_transfer_n = state.lateral_load_transfer_n
        self.speed = (
            math.hypot(self.longitudinal_velocity_mps, self.lateral_velocity_mps)
            * self.dynamics.pixels_per_meter / self.dynamics.control_hz
        )

    def _move_opponents(self) -> None:
        points = self.scene.track_centerline
        player_index = _nearest_point_index(points, self.player)
        player_lane_offset = _signed_lane_offset(points, self.player, player_index)
        vehicle = self.dynamics.vehicle
        road = self.dynamics.road
        speed_conversion = self.dynamics.pixels_per_meter / self.dynamics.control_hz
        traction_acceleration = min(
            vehicle.engine_force_n / vehicle.mass_kg,
            road.friction_coefficient * vehicle.tire_friction_multiplier * self.dynamics.gravity_mps2,
        )
        base_acceleration = traction_acceleration * self.dynamics.pixels_per_meter / (self.dynamics.control_hz ** 2)
        # Opponents are solid to each other and to barriers. The colliders are
        # resolved once per tick because `obstacle_shift` is a perturbation that
        # can move them between ticks but never within one.
        barriers = self._barrier_colliders()
        checkpoints = [
            entity for entity in self.scene.entities if entity.kind == EntityKind.CHECKPOINT
        ]
        primary_passer_id = self._primary_passer(player_index)
        # Scene compilation writes checkpoints in forward lap order: intermediate
        # sectors first, then the shared start/finish gate. Opponents follow that
        # exact sequence, so passing the start line from the grid cannot win a race.
        for opponent in self.opponents:
            behavior = opponent.behavior
            # Progress is advanced only through nearby forward samples. A global
            # nearest-point lookup can alternate between the last and first
            # samples at the start/finish seam (and between close chicane arms).
            previous_index = opponent.track_index
            opponent.track_index = _forward_track_index(
                points, opponent.position, opponent.track_index,
            )
            steps_to_player = _cyclic_index_delta(
                opponent.track_index, player_index, len(points),
            )
            player_distance = math.hypot(
                self.player.x - opponent.position.x,
                self.player.y - opponent.position.y,
            )
            straight = self._npc_on_straight(
                opponent.track_index, anticipation=2 + round(8 * behavior.intelligence),
            )
            # Pace is the car's baseline capability. Aggression is the independent
            # willingness to use it: a patient driver leaves a little margin while
            # an attacker reaches for the full authored pace.
            attack_pace = .94 + .12 * behavior.aggression
            cruise_speed = (
                vehicle.max_speed_mps * speed_conversion * behavior.pace * attack_pace
            )
            # Skill supplies cornering competence; aggression supplies commitment.
            # Grip remains the ceiling, so low-friction perturbations still punish
            # every temperament and cannot be cancelled by turning this dial up.
            corner_commitment = .70 + .32 * behavior.skill + .12 * behavior.aggression
            corner_speed = cruise_speed * min(.98, max(
                .42, min(.82, road.friction_coefficient * road.lateral_grip_multiplier),
            ) * corner_commitment)
            target_speed = cruise_speed if straight else corner_speed

            # Aggression sets how early a car commits, how small a gap it will
            # accept, and how long it insists on holding the passing lane.
            trigger_distance = 130.0 + 80.0 * behavior.aggression
            trigger_steps = 8 + round(12 * behavior.aggression)
            pace_threshold = .55 + .33 * behavior.aggression
            clear_distance = 98.0 - 40.0 * behavior.aggression
            clear_ticks = max(2, round(9 - 6 * behavior.aggression))
            player_is_blocking = (
                # Let the field begin choosing clean lanes shortly after the
                # green flag.  A five-second lockout meant a player could close
                # the grid before an NPC had enough lateral travel to pass.
                self.step_number > self.dynamics.control_hz
                and 0 < steps_to_player <= trigger_steps
                and player_distance < trigger_distance
                and self.speed < target_speed * pace_threshold
                # A passing lane is a shared piece of road, not a binary state
                # every nearby opponent may enter on the same tick.  Give the
                # nearest trailing car first refusal; following cars keep their
                # normal line until that manoeuvre has cleared the player.
                and opponent.entity_id == primary_passer_id
            )
            player_is_attacking = (
                behavior.defends
                and self.step_number > self.dynamics.control_hz
                and -trigger_steps <= steps_to_player < -1
                and player_distance < trigger_distance
                and self.speed > opponent.speed * .92
            )
            if opponent.overtake_phase == "cruise" and player_is_blocking:
                opponent.overtake_phase = "passing"
                opponent.target_lane_offset = self._npc_passing_lane(
                    opponent, player_lane_offset,
                )
                opponent.pass_clear_ticks = 0
            elif opponent.overtake_phase == "cruise" and player_is_attacking:
                opponent.overtake_phase = "defending"
                opponent.pass_clear_ticks = 0
            elif opponent.overtake_phase == "passing":
                # The passing decision is latched across the cyclic seam. Merge
                # only after the NPC is conclusively ahead for several ticks.
                if steps_to_player <= -4 and player_distance >= clear_distance:
                    opponent.pass_clear_ticks += 1
                else:
                    opponent.pass_clear_ticks = 0
                if opponent.pass_clear_ticks >= clear_ticks:
                    opponent.overtake_phase = "merge"
                    opponent.target_lane_offset = opponent.base_lane_offset
                    opponent.pass_clear_ticks = 0
            elif opponent.overtake_phase == "defending":
                opponent.target_lane_offset = self._npc_defensive_lane(
                    opponent, player_lane_offset,
                )
                if not player_is_attacking or player_distance < 46:
                    # Cover the approach, never the overlap: once the rival is
                    # alongside, the defender releases the lane it was holding.
                    opponent.overtake_phase = "merge"
                    opponent.target_lane_offset = opponent.base_lane_offset
            elif opponent.overtake_phase == "merge":
                opponent.target_lane_offset = opponent.base_lane_offset
                if abs(opponent.lane_offset - opponent.base_lane_offset) <= .25:
                    opponent.lane_offset = opponent.base_lane_offset
                    opponent.overtake_phase = "cruise"
            if opponent.overtake_phase == "cruise":
                # Only the uncontested phase picks its own line; passing,
                # defending, and merging are driven by the player's position.
                opponent.target_lane_offset = self._npc_cruise_lane(
                    opponent, player_lane_offset, steps_to_player, player_distance,
                )

            # Every phase above chooses a lane for racing reasons alone. A barrier
            # narrows one edge of the corridor, so the chosen lane is then checked
            # against the geometry it is about to drive through and moved to the
            # nearest lane that is actually free.
            opponent.target_lane_offset = self._npc_barrier_safe_lane(
                barriers, opponent.track_index, opponent.target_lane_offset,
            )
            opponent.lane_offset = _move_toward(
                opponent.lane_offset,
                opponent.target_lane_offset,
                NPC_LANE_CHANGE_PER_TICK,
            )
            target = _offset_track_point(points, opponent.target_index % len(points), opponent.lane_offset)
            dx, dy = target.x - opponent.position.x, target.y - opponent.position.y
            distance = math.hypot(dx, dy)
            # Advance through close spline samples without spending a stationary
            # tick at every waypoint. Opponents should flow around the lap, not
            # behave like stop-and-go obstacles on the racing line.
            advances = 0
            while distance < 18 and advances < 4:
                opponent.target_index = (opponent.target_index + 1) % len(points)
                target = _offset_track_point(points, opponent.target_index, opponent.lane_offset)
                dx, dy = target.x - opponent.position.x, target.y - opponent.position.y
                distance = math.hypot(dx, dy)
                advances += 1
            if distance <= 1e-9:
                continue
            clear = all(
                other.entity_id == opponent.entity_id
                or math.hypot(other.position.x - opponent.position.x, other.position.y - opponent.position.y) >= NPC_NITRO_CLEARANCE
                or abs(other.lane_offset - opponent.lane_offset) >= CAR_RADIUS * 2
                for other in self.opponents
            ) and math.hypot(self.player.x - opponent.position.x, self.player.y - opponent.position.y) >= NPC_NITRO_CLEARANCE
            nitro_attack_charge = NITRO_CAPACITY * (.95 - .35 * behavior.aggression)
            opponent.nitro_active = (
                behavior.uses_nitro and straight and clear and opponent.nitro > 0
                and (opponent.nitro_active or opponent.nitro >= nitro_attack_charge)
            )
            if opponent.nitro_active:
                target_speed *= NITRO_MAX_SPEED_MULTIPLIER

            # A car that has not opened lateral clearance follows at a controlled
            # gap; once the lane change is established it carries normal racing
            # speed past the slower player. Contact is caused by proximity, not by
            # whatever condition started the maneuver, so this guard stays armed
            # for its whole duration. Gating it on `player_is_blocking` released
            # it the moment the cars drew level, which is when they can touch.
            # Only the approaching car yields; slowing one that is already ahead
            # would pull it back into the player instead.
            approaching = -1 <= steps_to_player <= 12
            closing_speed = max(0.0, opponent.speed - self.speed)
            lateral_gap = abs(opponent.lane_offset - player_lane_offset)
            # Two different tolerances, because they buy different things.
            #
            # Re-picking the passing lane is free, so it happens as soon as a
            # conflict is even plausible: a lane change is rate-limited and cannot
            # be completed in a few ticks. Widening it with closing speed costs
            # nothing.
            #
            # Throttling back is expensive: it is what stops the car racing. Using
            # the same widened figure for both meant that behind a slow player the
            # tolerance exceeded the widest passing lane a car can reach, so it
            # could never be satisfied and every opponent simply queued up in line
            # at 82% of the player's pace instead of overtaking.
            reconsider_gap = _npc_safe_gap(behavior) + min(28.0, closing_speed * 2.4)
            hold_gap = _npc_safe_gap(behavior)
            if approaching and lateral_gap < reconsider_gap:
                if opponent.overtake_phase in {"passing", "defending"}:
                    opponent.target_lane_offset = (
                        # With longitudinal room to spare, cross to the clear side.
                        self._npc_passing_lane(opponent, player_lane_offset)
                        if abs(steps_to_player) > 3
                        # Level with the rival, the opposite side is on the far
                        # side of it: crossing there drives straight through it.
                        else self._npc_evasive_lane(opponent, player_lane_offset)
                    )
            if approaching and lateral_gap < hold_gap:
                # Bleeding a closing speed takes distance. A fixed following gap
                # is unreachable for a fast car behind a slow one, so the gap that
                # arms the speed hold is the distance that closing speed needs.
                required_gap = (
                    closing_speed ** 2 / (2 * max(1e-6, base_acceleration * 1.6))
                    + CAR_RADIUS * 3
                )
                if player_distance < max(96.0, required_gap):
                    target_speed = min(
                        target_speed, max(TRAFFIC_CREEP_SPEED, self.speed * .82),
                    )
                    opponent.nitro_active = False
            # The same reasoning applied to traffic rather than to the player. The
            # guard above stops a car driving into the player; without this one two
            # opponents in the same lane close on each other until the move clamp
            # is the only thing left, which reads as a car braking into a wall
            # rather than following at a gap.
            ahead = self._npc_car_ahead(opponent)
            if ahead is not None:
                gap, leader_speed = ahead
                closing = max(0.0, opponent.speed - leader_speed)
                required_gap = (
                    closing ** 2 / (2 * max(1e-6, base_acceleration * 1.6))
                    + NPC_CONTACT_CLEARANCE * 1.5
                )
                if gap < max(NPC_CONTACT_CLEARANCE * 2.5, required_gap):
                    target_speed = min(
                        target_speed, max(TRAFFIC_CREEP_SPEED, leader_speed * .9),
                    )
                    opponent.nitro_active = False
            acceleration = base_acceleration
            if opponent.nitro_active:
                acceleration += vehicle.nitro_force_n / vehicle.mass_kg * self.dynamics.pixels_per_meter / (self.dynamics.control_hz ** 2)
            if opponent.nitro_active:
                opponent.speed = min(target_speed, opponent.speed + acceleration)
                opponent.nitro = max(0.0, opponent.nitro - NITRO_DRAIN_PER_TICK)
            else:
                if opponent.speed > target_speed:
                    opponent.speed = max(target_speed, opponent.speed - acceleration * 1.6)
                else:
                    opponent.speed = min(target_speed, opponent.speed + acceleration)
                opponent.nitro = min(NITRO_CAPACITY, opponent.nitro + NITRO_RECHARGE_PER_TICK)
            travel = min(opponent.speed, distance)
            previous_position = opponent.position
            # Last line of defence, and the one that makes "opponents never
            # overlap a barrier or each other" an invariant rather than an
            # expectation. Avoidance is behavioral and can be defeated by a lane
            # the geometry does not allow; this clamp cannot be.
            opponent.position = self._npc_clear_position(
                opponent,
                previous_position,
                Vec2(
                    x=previous_position.x + dx / distance * travel,
                    y=previous_position.y + dy / distance * travel,
                ),
                barriers,
            )
            if opponent.position is previous_position:
                # Held by traffic or geometry: carry no speed into the next tick,
                # otherwise the car accelerates against something solid.
                opponent.speed = min(opponent.speed, TRAFFIC_CREEP_SPEED)
            if travel > .01:
                desired_heading = _bearing(previous_position, opponent.position)
                heading_step = max(-22.5, min(22.5, _angle_delta(opponent.heading, desired_heading)))
                opponent.heading = (opponent.heading + heading_step) % 360
            opponent.track_index = _forward_track_index(
                points, opponent.position, opponent.track_index,
            )
            # Progress drives the live position display. Completion is intentionally
            # stricter: the car must take the painted gate in the forward direction
            # once per lap, exactly as the player does. A nearest-centerline sample
            # can advance early near the seam, so it cannot be the finish detector.
            advanced = (opponent.track_index - previous_index) % len(points)
            if advanced <= len(points) // 2:
                opponent.progress_samples += advanced
            expected_checkpoint = checkpoints[opponent.checkpoint_index]
            if _crossed_checkpoint_gate(
                self.scene, previous_position, opponent.position, expected_checkpoint, CAR_RADIUS,
            ):
                opponent.checkpoint_index += 1
                if opponent.checkpoint_index == len(checkpoints):
                    opponent.completed_laps += 1
                    opponent.checkpoint_index = 0
            if opponent.finished_step is None and opponent.completed_laps >= self.scene.laps:
                opponent.finished_step = self.step_number
                if opponent.entity_id not in self.finish_order:
                    self.finish_order.append(opponent.entity_id)

    def _terminate_for_first_finisher(self) -> None:
        """End a competitive race as soon as its recorded winner is an NPC."""
        if not self.terminate_on_opponent_win or self.terminated or not self.finish_order:
            return
        winner = self.finish_order[0]
        if winner == "player":
            return
        self.terminated = True
        self.succeeded = False
        self.reason = f"{winner} finished first"

    def _npc_cruise_lane(
        self, opponent: OpponentState, player_lane_offset: float,
        steps_to_player: int, player_distance: float,
    ) -> float:
        """The line a car aims for while not actively passing or defending.

        Intelligence moves the target from the car's grid lane toward the
        geometric inside of the corner, and steadies it. A less capable driver
        keeps a wandering line instead, which makes it beatable without making it
        slow -- a different axis from pace and cornering commitment.

        The optimal line is only taken when it is free. A racing line runs through
        exactly where the player also wants to be, so when the player is close
        enough to contest it the target is pulled back to a safe gap: an opponent
        drives well, but not into somebody.
        """
        behavior = opponent.behavior
        points = self.scene.track_centerline
        count = len(points)
        index = opponent.track_index
        turn = _angle_delta(
            _bearing(points[(index - 2) % count], points[index]),
            _bearing(points[index], points[(index + 3) % count]),
        )
        limit = max(4.0, self.scene.track_width / 2 - CAR_RADIUS - 4)
        target = opponent.base_lane_offset
        if abs(turn) >= NPC_APEX_TURN_THRESHOLD:
            # A positive lane offset is the driver's right, and the inside of a
            # right-hand corner is to the right.
            apex = math.copysign(min(limit, NPC_APEX_REACH), turn)
            target += (apex - target) * behavior.intelligence * NPC_APEX_COMMITMENT
        # Deterministic wander: a function of track position and car identity, so
        # it is repeatable across machines rather than sampled.
        wander = (1 - behavior.intelligence) * NPC_LINE_WANDER * math.sin(
            index * .55 + opponent.lane_phase,
        )
        target += wander
        if abs(steps_to_player) <= 6 and player_distance < 130:
            gap = _npc_safe_gap(behavior)
            if abs(target - player_lane_offset) < gap:
                # Concede to whichever side of the player this car is already on,
                # so conceding never means crossing through them.
                separation = opponent.lane_offset - player_lane_offset
                side = math.copysign(1.0, separation if abs(separation) > 1e-6 else 1.0)
                target = player_lane_offset + side * gap
        return max(-limit, min(limit, target))

    def _npc_evasive_lane(
        self, opponent: OpponentState, player_lane_offset: float,
    ) -> float:
        """Widen away from a rival that is alongside, without crossing its lane.

        Switching to the opposite side is only safe with longitudinal room. When
        the cars are level, the far side is reached by driving through the rival,
        so the only safe direction is further out on the side already held.
        """
        separation = opponent.lane_offset - player_lane_offset
        away = (
            math.copysign(1.0, separation) if abs(separation) > 1e-6
            else (1.0 if opponent.lane_offset >= 0 else -1.0)
        )
        limit = self.scene.track_width / 2 - CAR_RADIUS
        desired = opponent.lane_offset + away * _npc_safe_gap(opponent.behavior)
        return max(-limit, min(limit, desired))

    def _npc_defensive_lane(
        self, opponent: OpponentState, player_lane_offset: float,
    ) -> float:
        """Cover a closing rival's approach lane without blocking the center.

        The deterministic racing-line oracle certifies every scene by driving
        lane offset zero, so a defender that parked on the centerline would make
        aggressive-traffic circuits fail their own playability verification. A
        defending car therefore shades toward the rival's lane but never inside
        `NPC_MIN_DEFENSIVE_OFFSET`, which always leaves one side passable.
        """
        aggression = opponent.behavior.aggression
        reach = NPC_MIN_DEFENSIVE_OFFSET + (NPC_PASS_LANE_OFFSET - NPC_MIN_DEFENSIVE_OFFSET) * (
            1.0 - aggression
        )
        side = 1.0 if player_lane_offset >= 0 else -1.0
        if abs(player_lane_offset) < 1e-6:
            side = 1.0 if opponent.base_lane_offset >= 0 else -1.0
        return side * max(NPC_MIN_DEFENSIVE_OFFSET, min(NPC_PASS_LANE_OFFSET, reach))

    def _npc_passing_lane(
        self, opponent: OpponentState, player_lane_offset: float,
    ) -> float:
        """Choose a clear, unreserved side without cutting across a blocked car."""
        current_side = NPC_PASS_LANE_OFFSET if opponent.lane_offset >= 0 else -NPC_PASS_LANE_OFFSET
        opposite_side = -current_side
        safe_clearance = _npc_safe_gap(opponent.behavior)
        candidates = (current_side, opposite_side)
        for candidate in candidates:
            if (
                abs(candidate - player_lane_offset) >= safe_clearance
                and not self._npc_lane_reserved(opponent, candidate)
            ):
                return candidate
        # This is only reachable if the player is almost sideways across the
        # road or the two pass lanes are reserved; choose maximum clearance and
        # let the follow-gap logic hold speed until a lane opens.
        return max(
            candidates,
            key=lambda candidate: (
                not self._npc_lane_reserved(opponent, candidate),
                abs(candidate - player_lane_offset),
            ),
        )

    def _primary_passer(self, player_index: int) -> str | None:
        """Return the sole nearby car allowed to begin an overtake this tick.

        Existing manoeuvres retain the reservation until they have moved clear.
        That makes the choice stable while a compact field filters past a slow
        player, instead of reshuffling it every control tick.
        """
        points = self.scene.track_centerline
        contenders = [
            opponent for opponent in self.opponents
            if opponent.overtake_phase in {"passing", "merge"}
            and math.hypot(
                opponent.position.x - self.player.x,
                opponent.position.y - self.player.y,
            ) < 220.0
        ]
        if not contenders:
            contenders = [
                opponent for opponent in self.opponents
                if opponent.overtake_phase == "cruise"
                and 0 < _cyclic_index_delta(
                    opponent.track_index, player_index, len(points),
                ) <= NPC_PASS_RESERVATION_STEPS
                and math.hypot(
                    opponent.position.x - self.player.x,
                    opponent.position.y - self.player.y,
                ) < 220.0
            ]
        if not contenders:
            return None
        return min(
            contenders,
            key=lambda opponent: (
                _cyclic_index_delta(opponent.track_index, player_index, len(points))
                if _cyclic_index_delta(opponent.track_index, player_index, len(points)) > 0
                else len(points),
                round(math.hypot(
                    opponent.position.x - self.player.x,
                    opponent.position.y - self.player.y,
                ), 4),
                opponent.entity_id,
            ),
        ).entity_id

    def _npc_lane_reserved(self, opponent: OpponentState, candidate: float) -> bool:
        """Whether another local pass is already committed to this lane."""
        points = self.scene.track_centerline
        for other in self.opponents:
            if other.entity_id == opponent.entity_id or other.overtake_phase not in {"passing", "merge"}:
                continue
            forward = _cyclic_index_delta(opponent.track_index, other.track_index, len(points))
            reverse = _cyclic_index_delta(other.track_index, opponent.track_index, len(points))
            if min(forward, reverse) > NPC_PASS_RESERVATION_STEPS:
                continue
            if abs(other.target_lane_offset - candidate) < NPC_CONTACT_CLEARANCE:
                return True
        return False

    def _npc_on_straight(self, index: int, anticipation: int = 4) -> bool:
        """Whether the road is straight, looking `anticipation` samples further on.

        A car that looks further ahead recognises a corner before it is in it, so
        it slows in time and carries more speed through. This is what intelligence
        buys on the longitudinal axis; `skill` sets how fast it dares go.
        """
        points = self.scene.track_centerline
        entry = _bearing(points[(index - 2) % len(points)], points[(index + 2) % len(points)])
        exit_heading = _bearing(
            points[(index + 2) % len(points)],
            points[(index + 2 + max(1, anticipation)) % len(points)],
        )
        return abs(_angle_delta(entry, exit_heading)) <= NPC_STRAIGHT_HEADING_DELTA

    def _on_track(self, point: Vec2) -> bool:
        return _distance_to_polyline(point, self.scene.track_centerline, closed=True) <= self.scene.track_width / 2 - CAR_RADIUS

    def _barrier_colliders(self) -> list:
        """Every barrier's exact collision shape at this tick's shift."""
        discrete = [
            collider_for(entity, self.obstacle_shift)
            for entity in self.scene.entities
            if entity.kind == EntityKind.OBSTACLE
        ]
        return [*discrete, *(collider for _, collider in self.edge_colliders)]

    def _barrier_entries(self) -> list[tuple[str, Collider]]:
        discrete = [
            (entity.id, collider_for(entity, self.obstacle_shift))
            for entity in self.scene.entities
            if entity.kind == EntityKind.OBSTACLE
        ]
        return [*discrete, *self.edge_colliders]

    def _npc_barrier_safe_lane(
        self, barriers: list, track_index: int, desired_lane: float,
    ) -> float:
        """Nearest lane to `desired_lane` that stays clear of barrier geometry.

        The certified centerline is never occluded by a barrier, so lane zero is
        always a legal answer and this search cannot come back empty.
        """
        if not barriers or self._npc_lane_clear(barriers, track_index, desired_lane):
            return desired_lane
        safe_half_width = max(4.0, self.scene.track_width / 2 - CAR_RADIUS - 4)
        step = CAR_RADIUS / 2
        candidates: list[float] = [0.0]
        offset = step
        while offset <= safe_half_width:
            candidates.extend((desired_lane - offset, desired_lane + offset))
            offset += step
        legal = [
            candidate for candidate in candidates
            if -safe_half_width <= candidate <= safe_half_width
        ]
        for candidate in sorted(legal, key=lambda value: (
            round(abs(value - desired_lane), 4), round(abs(value), 4),
        )):
            if self._npc_lane_clear(barriers, track_index, candidate):
                return candidate
        return 0.0

    def _npc_lane_clear(self, barriers: list, track_index: int, lane_offset: float) -> bool:
        """Whether a lane stays free of barriers over the samples just ahead."""
        points = self.scene.track_centerline
        for step in range(NPC_BARRIER_LOOKAHEAD):
            point = _offset_track_point(points, (track_index + step) % len(points), lane_offset)
            if any(
                collider.hits_circle(point.x, point.y, CAR_RADIUS)
                for collider in barriers
            ):
                return False
        return True

    def _npc_car_ahead(self, opponent: OpponentState) -> tuple[float, float] | None:
        """Gap and speed of the closest opponent ahead in the same lane band."""
        points = self.scene.track_centerline
        closest: tuple[float, float] | None = None
        for other in self.opponents:
            if other.entity_id == opponent.entity_id:
                continue
            steps = _cyclic_index_delta(opponent.track_index, other.track_index, len(points))
            if not 0 <= steps <= 14:
                continue
            if abs(other.lane_offset - opponent.lane_offset) >= NPC_CONTACT_CLEARANCE:
                continue
            gap = math.hypot(
                other.position.x - opponent.position.x,
                other.position.y - opponent.position.y,
            )
            if closest is None or gap < closest[0]:
                closest = (gap, other.speed)
        return closest

    def _npc_clear_position(
        self, opponent: OpponentState, start: Vec2, proposed: Vec2, barriers: list,
    ) -> Vec2:
        """The furthest point along this move that touches nothing solid.

        Cars already in contact at the start of the tick are excluded from their
        own pair test: with no escape hatch two overlapping cars would each refuse
        to move and stay welded together for the rest of the race. Excluding them
        lets the pair separate, which is the only outcome that resolves it.

        The player is also treated as solid for an opponent's planned motion.
        Player contact remains authoritative when the player drives into an NPC,
        but an NPC should not manufacture a collision by continuing to steer into
        the player after its lane-change decision is no longer safe.
        """
        others = [
            other for other in self.opponents
            if other.entity_id != opponent.entity_id
            and math.hypot(
                other.position.x - start.x, other.position.y - start.y,
            ) >= NPC_CONTACT_CLEARANCE
        ]
        player_is_separate = math.hypot(
            self.player.x - start.x, self.player.y - start.y,
        ) >= NPC_CONTACT_CLEARANCE
        path_start, path_end = (start.x, start.y), (proposed.x, proposed.y)
        barrier_contacts = [
            (collider, contact)
            for collider in barriers
            if (contact := collider.sweep_contact(path_start, path_end, CAR_RADIUS)) is not None
        ]
        if barrier_contacts:
            collider, contact = min(barrier_contacts, key=lambda item: item[1].fraction)
            normal_x, normal_y = contact.normal
            x, y = contact.safe_point
            for _ in range(24):
                if not collider.hits_circle(x, y, CAR_RADIUS):
                    break
                x += normal_x * max(.5, CAR_RADIUS * .2)
                y += normal_y * max(.5, CAR_RADIUS * .2)
            rebound = Vec2(
                x=x + normal_x * BARRIER_REBOUND_PIXELS,
                y=y + normal_y * BARRIER_REBOUND_PIXELS,
            )
            opponent.speed *= BARRIER_RESTITUTION
            opponent.nitro_active = False
            if not any(
                circle_collider(other.position.x, other.position.y, CAR_RADIUS).hits_circle(
                    rebound.x, rebound.y, CAR_RADIUS,
                )
                for other in others
            ) and (
                not player_is_separate
                or not circle_collider(self.player.x, self.player.y, CAR_RADIUS).hits_circle(
                    rebound.x, rebound.y, CAR_RADIUS,
                )
            ):
                return rebound
            return start

        def clear(target: Vec2) -> bool:
            path_start, path_end = (start.x, start.y), (target.x, target.y)
            if any(
                collider.hits_swept_circle(path_start, path_end, CAR_RADIUS)
                for collider in barriers
            ):
                return False
            clear_of_opponents = not any(
                circle_collider(
                    other.position.x, other.position.y, CAR_RADIUS,
                ).hits_swept_circle(path_start, path_end, CAR_RADIUS)
                for other in others
            )
            clear_of_player = (
                not player_is_separate
                or not circle_collider(
                    self.player.x, self.player.y, CAR_RADIUS,
                ).hits_swept_circle(path_start, path_end, CAR_RADIUS)
            )
            return clear_of_opponents and clear_of_player

        if clear(proposed):
            return proposed
        for fraction in NPC_TRAVEL_FRACTIONS:
            if fraction <= 0:
                break
            candidate = Vec2(
                x=start.x + (proposed.x - start.x) * fraction,
                y=start.y + (proposed.y - start.y) * fraction,
            )
            if clear(candidate):
                return candidate
        return start

    def _collision(self, previous_player: Vec2 | None = None) -> str | None:
        """First obstacle or car touched anywhere along this tick's travel.

        The test is swept rather than evaluated only at the tick's end position.
        Six physics substeps run per control tick and top speed is a tunable
        parameter, so a fast enough car would otherwise step straight over a
        barrier between two tests and register no contact at all.
        """
        barrier = self._barrier_collision(previous_player)
        if barrier:
            return barrier[0]
        return self._opponent_collision(previous_player)

    def _barrier_collision(
        self, previous_player: Vec2 | None = None,
    ) -> tuple[str, Collider, SweepContact] | None:
        start = (previous_player.x, previous_player.y) if previous_player else (self.player.x, self.player.y)
        end = (self.player.x, self.player.y)
        for barrier_id, collider in self._barrier_entries():
            contact = collider.sweep_contact(start, end, CAR_RADIUS)
            if contact:
                return barrier_id, collider, contact
        return None

    def _opponent_collision(self, previous_player: Vec2 | None = None) -> str | None:
        start = (previous_player.x, previous_player.y) if previous_player else (self.player.x, self.player.y)
        end = (self.player.x, self.player.y)
        for opponent in self.opponents:
            if circle_collider(
                opponent.position.x, opponent.position.y, CAR_RADIUS,
            ).hits_swept_circle(start, end, CAR_RADIUS):
                return opponent.entity_id
        return None

    def _bounce_player_from_barrier(
        self, collider: Collider, contact: SweepContact,
    ) -> None:
        """Place the car outside the barrier and damp/reflect its planar velocity."""
        normal_x, normal_y = contact.normal
        x, y = contact.safe_point
        # A restored or hand-authored state can begin inside a collider. Walk it
        # out before adding the small visible rebound shared by both renderers.
        separation_step = max(.5, CAR_RADIUS * .2)
        for _ in range(24):
            if not collider.hits_circle(x, y, CAR_RADIUS):
                break
            x += normal_x * separation_step
            y += normal_y * separation_step
        self.player = Vec2(
            x=x + normal_x * BARRIER_REBOUND_PIXELS,
            y=y + normal_y * BARRIER_REBOUND_PIXELS,
        )

        heading = math.radians(self.heading)
        forward = (math.cos(heading), math.sin(heading))
        left = (-forward[1], forward[0])
        velocity_x = (
            forward[0] * self.longitudinal_velocity_mps
            + left[0] * self.lateral_velocity_mps
        )
        velocity_y = (
            forward[1] * self.longitudinal_velocity_mps
            + left[1] * self.lateral_velocity_mps
        )
        inward = velocity_x * normal_x + velocity_y * normal_y
        if inward < 0:
            velocity_x -= (1.0 + BARRIER_RESTITUTION) * inward * normal_x
            velocity_y -= (1.0 + BARRIER_RESTITUTION) * inward * normal_y
        self.longitudinal_velocity_mps = max(
            0.0, velocity_x * forward[0] + velocity_y * forward[1],
        )
        self.lateral_velocity_mps = velocity_x * left[0] + velocity_y * left[1]
        self.yaw_rate_radians_per_second *= -.25
        self.longitudinal_acceleration_mps2 = 0.0
        self.lateral_acceleration_mps2 = 0.0
        self.nitro_active = False
        self.speed = (
            math.hypot(self.longitudinal_velocity_mps, self.lateral_velocity_mps)
            * self.dynamics.pixels_per_meter / self.dynamics.control_hz
        )

    def _check_checkpoint(
        self, events: list[str], *, on_track: bool, previous_player: Vec2,
    ) -> float:
        if self.objective_index >= len(self.scene.objectives):
            return 0.0
        objective = self.scene.objectives[self.objective_index]
        target = next(entity for entity in self.scene.entities if entity.id == objective.target_id)
        if not on_track or self.speed <= 0 or not _crossed_checkpoint_gate(
            self.scene, previous_player, self.player, target, CAR_RADIUS,
        ):
            return 0.0
        target_index = _nearest_point_index(self.scene.track_centerline, _rect_center(target.rect))
        track_heading = _bearing(
            self.scene.track_centerline[(target_index - 1) % len(self.scene.track_centerline)],
            self.scene.track_centerline[(target_index + 1) % len(self.scene.track_centerline)],
        )
        if abs(_angle_delta(self.heading, track_heading)) >= 90:
            return 0.0
        self.objective_index += 1
        events.append(f"crossed {target.id}")
        if self.objective_index >= len(self.scene.objectives):
            self.terminated, self.succeeded = True, True
            self.finish_order.append("player")
            position = self.player_position or 1
            self.reason = (
                f"{self.scene.laps}-lap race completed in P{position} of {self.field_size}"
            )
            events.append(f"race completed P{position}/{self.field_size}")
            return 1.0
        if target.id == "finish-line":
            events.append(f"lap {self.objective_index // self.scene.sector_count} completed")
        return 0.2


class RacingBackend:
    id = ENGINE_ID
    display_name = "Deterministic top-down racing engine"

    def create(self, scene: SceneSpec, perturbation: str | None = None) -> RacingWorld:
        """Build the world this scene describes, planar or elevated.

        The scene decides the dimension, not the caller. `Racing3DWorld` subclasses
        `RacingWorld` and shares every rule, so dispatching here is what makes the whole
        harness 3D-capable at once: runs, policies, the reflex driver, forks, experiments,
        and probes all obtain their world through this one method, and none of them needs
        to know which engine answered.
        """
        if scene.elevation is not None and not scene.elevation.is_flat:
            from .racing3d import Racing3DWorld

            return Racing3DWorld.from_scene(scene, perturbation)
        return RacingWorld.from_scene(scene, perturbation)


@dataclass
class RacingLineController:
    """Deterministic oracle/baseline that follows the public racing line."""

    name: str = "oracle-racing-line"
    scene: SceneSpec | None = None
    target_index: int = 1

    def reset(self, scene: SceneSpec, seed: int) -> None:
        self.scene, self.target_index = scene, 1

    def act(self, observation: ObservationPacket) -> tuple[Action, DecisionRecord]:
        assert self.scene is not None
        points = self.scene.track_centerline
        nearest = _nearest_point_index(points, observation.proprioception)
        lookahead = max(4, min(8, 4 + round(observation.speed / 3)))
        self.target_index = (nearest + lookahead) % len(points)
        avoid_offset, yielding = self._traffic_avoidance(observation, nearest)
        target = _offset_track_point(points, self.target_index, avoid_offset)
        desired = _bearing(observation.proprioception, target)
        delta = _angle_delta(observation.heading, desired)
        entry_heading = _bearing(points[(nearest + 1) % len(points)], points[(nearest + 4) % len(points)])
        exit_heading = _bearing(points[(nearest + 4) % len(points)], points[(nearest + 9) % len(points)])
        upcoming_curvature = abs(_angle_delta(entry_heading, exit_heading))
        grip = (
            observation.dynamics.road.friction_coefficient
            * observation.dynamics.road.lateral_grip_multiplier
            * observation.dynamics.vehicle.tire_friction_multiplier
        )
        corner_speed = max(1.8, min(4.0, 4.0 * math.sqrt(grip)))
        longitudinal_key = (
            "s" if yielding or (upcoming_curvature > 7 and observation.speed > corner_speed) else "w"
        )
        if delta > 6:
            action = ActionName.RIGHT
            keys = [longitudinal_key, "d"]
        elif delta < -6:
            action = ActionName.LEFT
            keys = [longitudinal_key, "a"]
        elif longitudinal_key == "s":
            action = ActionName.BACKWARD
            keys = ["s"]
        else:
            action = ActionName.FORWARD
            keys = ["w"]
        return Action(name=action, keys=keys), DecisionRecord(
            action=action, subgoal=observation.task, confidence=.98,
            summary=(
                f"Following the certified racing line; target bearing {delta:.1f}°, "
                f"upcoming curvature {upcoming_curvature:.1f}°, corner target {corner_speed:.1f}"
                + (
                    f", yielding to traffic at lane {avoid_offset:+.0f}." if yielding
                    else f", lane {avoid_offset:+.0f}." if avoid_offset else "."
                )
            ), candidates=[action],
        )

    def _traffic_avoidance(
        self, observation: ObservationPacket, nearest: int,
    ) -> tuple[float, bool]:
        """Offset the aimed line, and lift off only for traffic still in the way.

        Certification is only meaningful if the certifying driver can handle the
        traffic and barriers the brief asked for. Without this the oracle drove a
        fixed centerline, so any low-grip circuit where it drifted into a legal
        side lane failed verification and the whole environment was discarded.

        A barrier is static and can simply be driven around, so it moves the aimed
        line but never causes braking. Only a car that is close and still sharing
        this lane does, because that gap can close on its own.
        """
        assert self.scene is not None
        ahead = [
            entity for entity in observation.local_entities
            if entity.get("kind") in {EntityKind.NPC.value, EntityKind.OBSTACLE.value}
            and float(entity.get("distance", 999)) <= 105
            and abs(float(entity.get("bearing", 180))) <= 55
        ]
        # A car level with this one is outside the forward cone entirely, so it was
        # invisible here. Drifting laterally then squeezed it against the track
        # edge until they touched, with neither car able to give way.
        alongside = [
            entity for entity in observation.local_entities
            if entity.get("kind") == EntityKind.NPC.value
            and float(entity.get("distance", 999)) <= 74
            and abs(float(entity.get("bearing", 180))) > 55
        ]
        if not ahead and not alongside:
            return 0.0, False
        # Steering considers both; only a car ahead can justify lifting off.
        threat = min(ahead + alongside, key=lambda entity: float(entity["distance"]))
        braking_threat = min(ahead, key=lambda entity: float(entity["distance"])) if ahead else None
        threat_lane = float(threat.get("lane_offset", 0.0))
        own_lane = _signed_lane_offset(
            self.scene.track_centerline, observation.proprioception, nearest,
        )
        safe_half_width = max(0.0, self.scene.track_width / 2 - CAR_RADIUS - 6)
        side = math.copysign(1.0, threat_lane) if abs(threat_lane) > 1 else 1.0
        offset = -side * min(safe_half_width, CAR_RADIUS * 3.4)
        # Lift off only for a car this one is actually closing on. Braking for any
        # car merely present ahead means a driver being lapped by faster traffic
        # yields continuously and never completes the race at all.
        if braking_threat is None:
            return offset, False
        distance = float(braking_threat["distance"])
        closing = observation.speed > float(braking_threat.get("speed", 0.0)) * 1.05
        yielding = (
            braking_threat.get("kind") == EntityKind.NPC.value
            and abs(float(braking_threat.get("lane_offset", 0.0)) - own_lane) < CAR_RADIUS * 2.8
            and (distance < CAR_RADIUS * 3 or (closing and distance < 72))
            # Steering cannot rotate a stopped car, so braking to a standstill in
            # order to avoid something is self-defeating: the car then has no way
            # to steer around it. Below a creep speed the driver keeps rolling and
            # uses the offset line instead. Without this a yielding driver and a
            # yielding opponent deadlock at zero and neither ever moves again.
            and observation.speed > TRAFFIC_CREEP_SPEED
        )
        return offset, yielding


@dataclass
class RacingIntentController:
    """Execute model-selected sector intents with deterministic motor control."""

    scene: SceneSpec
    intents: list[dict[str, float | int]]
    strategy_summary: str

    def act(self, observation: ObservationPacket) -> tuple[Action, DecisionRecord]:
        """Execute the model's sector intent, correcting it only where unsafe.

        Traffic and barriers adjust the requested speed and lane by a bounded
        amount. They never replace the plan with the oracle's: doing so would
        make this policy measure the deterministic racing line instead of the
        model whenever a scene contained a single opponent.
        """
        points = self.scene.track_centerline
        nearest = _nearest_point_index(points, observation.proprioception)
        sector = min(11, nearest * 12 // len(points))
        intent = self.intents[sector]
        requested_speed = float(intent["target_speed"])
        requested_lane = float(intent["lane_offset"])
        safe_half_width = max(4.0, self.scene.track_width / 2 - CAR_RADIUS - 6)
        target_speed = requested_speed
        lane_offset = max(-safe_half_width, min(safe_half_width, requested_lane))
        corrections: list[str] = []

        hazards = [
            entity for entity in observation.local_entities
            if entity.get("kind") in {EntityKind.NPC.value, EntityKind.OBSTACLE.value}
            and float(entity.get("distance", 999)) <= 170
        ]
        for hazard in hazards:
            distance = float(hazard.get("distance", 999))
            ahead = abs(float(hazard.get("bearing", 180))) <= 50
            if not ahead:
                continue
            if distance < 130:
                target_speed = min(target_speed, max(2.0, requested_speed * .62))
                corrections.append(f"slowed for {hazard.get('id')} at {distance:.0f}px")
            # Take the other half of the road rather than snapping to center.
            hazard_lane = float(hazard.get("lane_offset", 0.0))
            if distance < 150 and abs(lane_offset - hazard_lane) < CAR_RADIUS * 3:
                side = math.copysign(1.0, hazard_lane) if abs(hazard_lane) > 1 else 1.0
                lane_offset = -side * min(safe_half_width, CAR_RADIUS * 2.4)
                corrections.append(f"shifted clear of {hazard.get('id')}")

        # A lane is only available if nothing is already occupying it. Reacting
        # only to hazards ahead lets a requested lane change steer straight into
        # a car running alongside, which no downstream guard can undo. The lane is
        # solved against every neighbour at once: correcting for them one at a time
        # oscillates, and the last correction wins by pushing into the first car.
        occupied = [
            float(entity.get("lane_offset", 0.0))
            for entity in observation.local_entities
            if entity.get("kind") == EntityKind.NPC.value
            and float(entity.get("distance", 999)) <= 74
        ]
        if occupied:
            needed = CAR_RADIUS * 2.8
            candidates = [lane_offset, 0.0, *(
                lane + side * needed for lane in occupied for side in (-1.0, 1.0)
            )]
            feasible = [
                candidate for candidate in candidates
                if -safe_half_width <= candidate <= safe_half_width
                and all(abs(candidate - lane) >= needed for lane in occupied)
            ]
            if feasible:
                chosen = min(feasible, key=lambda candidate: (
                    round(abs(candidate - lane_offset), 4), round(abs(candidate), 4),
                ))
            else:
                # Fully boxed in: take the roomiest gap and drop to a pace that
                # keeps the remaining clearance survivable.
                steps = 24
                chosen = max(
                    (-safe_half_width + index * 2 * safe_half_width / steps for index in range(steps + 1)),
                    key=lambda candidate: (
                        round(min(abs(candidate - lane) for lane in occupied), 3),
                        -round(abs(candidate - lane_offset), 3),
                    ),
                )
                target_speed = min(target_speed, 2.5)
                corrections.append("boxed in by traffic; holding a minimum-risk pace")
            if abs(chosen - lane_offset) > 1e-6:
                corrections.append(f"held clear of {len(occupied)} car(s) alongside")
            lane_offset = chosen

        lookahead = max(3, min(7, 3 + round(observation.speed / 2.5)))
        target_index = (nearest + lookahead) % len(points)
        target = _offset_track_point(points, target_index, lane_offset)
        desired = _bearing(observation.proprioception, target)
        delta = _angle_delta(observation.heading, desired)
        braking = observation.speed > target_speed + .7
        longitudinal_key = "s" if braking else "w" if observation.speed < target_speed - .3 else None
        if delta > 7:
            action, keys = ActionName.RIGHT, [*filter(None, [longitudinal_key]), "d"]
        elif delta < -7:
            action, keys = ActionName.LEFT, [*filter(None, [longitudinal_key]), "a"]
        elif braking:
            action, keys = ActionName.BACKWARD, ["s"]
        elif longitudinal_key == "w":
            action, keys = ActionName.FORWARD, ["w"]
        else:
            action, keys = ActionName.IDLE, []
        return Action(name=action, keys=keys), DecisionRecord(
            action=action, subgoal=f"execute sector {sector + 1} strategy", confidence=.94,
            summary=(
                f"Sector intent: requested speed {requested_speed:.1f}, executed {target_speed:.1f}, "
                f"requested lane {requested_lane:+.1f}, executed {lane_offset:+.1f}; "
                f"heading error {delta:+.1f}. "
                + (f"Corrections: {'; '.join(corrections[:3])}. " if corrections else "No safety correction. ")
                + self.strategy_summary[:120]
            ),
            candidates=[action],
        )


def racing_local_state(
    scene: SceneSpec, observation: ObservationPacket, track_index_hint: int | None = None,
) -> dict:
    """Local proprioception: where the car sits on the road right here.

    This is the whole of the track information a forward-cone policy receives — the
    view deliberately withholds global route knowledge, and these continuity-tracked
    local values are what it gets instead. It is factored out so a low-level
    controller can be built on exactly the fields the prompt receives and nothing
    more, which makes the privilege boundary checkable rather than asserted.
    """
    points = scene.track_centerline
    nearest = _nearest_track_index_with_hint(points, observation.proprioception, track_index_hint)
    current = points[nearest]
    before = points[(nearest - 1) % len(points)]
    after = points[(nearest + 1) % len(points)]
    tangent_x, tangent_y = after.x - before.x, after.y - before.y
    tangent_length = max(1.0, math.hypot(tangent_x, tangent_y))
    signed_lane_offset = (
        (observation.proprioception.x - current.x) * (-tangent_y / tangent_length)
        + (observation.proprioception.y - current.y) * (tangent_x / tangent_length)
    )
    track_heading = _bearing(before, after)
    return {
        "centerline_index": nearest,
        "progress_percent": round(nearest / len(points) * 100, 1),
        "signed_lane_offset": round(signed_lane_offset, 1),
        "centerline_heading": round(track_heading, 1),
        "centerline_heading_error": round(_angle_delta(observation.heading, track_heading), 1),
        "safe_lane_half_width": round(scene.track_width / 2 - CAR_RADIUS, 1),
        "surface": scene.surface,
        "on_track": _distance_to_polyline(
            observation.proprioception, points, closed=True,
        ) <= scene.track_width / 2 - CAR_RADIUS,
    }


def racing_public_context(
    scene: SceneSpec,
    observation: ObservationPacket,
    recent_trace: list[dict[str, object]] | None = None,
    track_index_hint: int | None = None,
    previous_chunk: dict | None = None,
    control_budget: dict | None = None,
) -> dict:
    """Compact, chat-free driving context for a model-backed player."""
    track_state = racing_local_state(scene, observation, track_index_hint)
    nearest = int(track_state["centerline_index"])
    lookahead = [scene.track_centerline[(nearest + offset) % len(scene.track_centerline)] for offset in range(1, 5)]
    annotated_lookahead = []
    for offset, point in enumerate(lookahead, start=1):
        absolute_heading = _bearing(observation.proprioception, point)
        annotated_lookahead.append({
            "x": round(point.x, 1), "y": round(point.y, 1),
            "distance": round(math.hypot(point.x - observation.proprioception.x, point.y - observation.proprioception.y), 1),
            "absolute_heading": round(absolute_heading, 1),
            "heading_error": round(_angle_delta(observation.heading, absolute_heading), 1),
            "priority": offset,
        })
    return {
        "tool_surface": "racing-line-v2",
        "telemetry": {
            "x": round(observation.proprioception.x, 1),
            "y": round(observation.proprioception.y, 1),
            "heading_degrees": round(observation.heading, 1),
            "speed": round(observation.speed, 1),
            "nitro": round(observation.nitro, 1),
            "nitro_active": observation.nitro_active,
            "nitro_ready": observation.nitro_ready,
            "countdown_ticks_remaining": observation.countdown_ticks_remaining,
            "longitudinal_speed_mps": observation.longitudinal_speed_mps,
            "lateral_speed_mps": observation.lateral_speed_mps,
            "yaw_rate_degrees_per_second": observation.yaw_rate_degrees_per_second,
            "steering_angle_degrees": observation.steering_angle_degrees,
            "longitudinal_acceleration_mps2": observation.longitudinal_acceleration_mps2,
            "lateral_acceleration_mps2": observation.lateral_acceleration_mps2,
            "slip_angle_degrees": observation.slip_angle_degrees,
        },
        "track_state": track_state,
        "active_checkpoint": observation.task,
        "racing_line_lookahead": annotated_lookahead,
        "track_map": racing_track_map(scene),
        "upcoming_corner": _upcoming_corner(scene, nearest),
        "steering_rule": "heading_error > 5 means right; heading_error < -5 means left; otherwise accelerate or coast",
        "physics": racing_physics_context(scene, observation),
        "nearby": observation.local_entities,
        "recent_trajectory": recent_trace or [],
        # Absent on the first decision of an episode, when there is no prior chunk.
        **({"previous_chunk": previous_chunk} if previous_chunk else {}),
        # Absent under the synchronous loop, where the next decision is one tick away.
        **({"control_budget": control_budget} if control_budget else {}),
        "controls": {
            "forward": "throttle", "backward": "brake", "left": "steer left",
            "right": "steer right", "idle": "coast",
            "nitro": "straight-line throttle boost; can start only at 100% charge, then burns while held; recharges while inactive",
        },
    }


def racing_track_map(scene: SceneSpec) -> list[dict]:
    """The circuit's corner sequence as public route knowledge.

    A driver that can see the whole overhead frame can already see the shape of
    the track. Naming the corners makes that shape usable: each entry says where
    a corner starts as a lap percentage, which way it turns, how far it turns,
    and the grip-limited speed it can be taken at.
    """
    if scene.track_report is None:
        return []
    return [
        {
            "corner": corner.index + 1,
            "entry_progress_percent": corner.entry_progress_percent,
            "direction": corner.direction,
            "turn_degrees": corner.achieved_angle_degrees,
            "radius_pixels": corner.achieved_radius_pixels,
            "recommended_entry_speed": corner.recommended_entry_speed,
            "screen_region": corner.achieved_region.value,
        }
        for corner in scene.track_report.corners
    ]


def _upcoming_corner(scene: SceneSpec, track_index: int) -> dict | None:
    """The next corner ahead, with the distance and speed delta to prepare for."""
    corners = racing_track_map(scene)
    if not corners:
        return None
    count = len(scene.track_centerline)
    progress = track_index / count * 100
    spacing = (
        scene.track_report.centerline_spacing_pixels
        if scene.track_report and scene.track_report.centerline_spacing_pixels
        else 22.0
    )
    ahead = min(
        corners, key=lambda corner: (corner["entry_progress_percent"] - progress) % 100,
    )
    gap_percent = (ahead["entry_progress_percent"] - progress) % 100
    return {
        **ahead,
        "distance_pixels": round(gap_percent / 100 * count * spacing, 1),
    }


def racing_physics_context(scene: SceneSpec, observation: ObservationPacket) -> dict:
    """Publish exact parameters and deterministic one-control-tick predictions."""
    dynamics = observation.dynamics
    vehicle = dynamics.vehicle
    road = dynamics.road
    on_track = _distance_to_polyline(
        observation.proprioception, scene.track_centerline, closed=True,
    ) <= scene.track_width / 2 - CAR_RADIUS
    nitro_available = on_track and observation.nitro > 0 and (
        observation.nitro_active or observation.nitro_ready
    )

    def initial_state() -> VehiclePhysicsState:
        return VehiclePhysicsState(
            x=0.0, y=0.0, heading_radians=0.0,
            longitudinal_velocity_mps=max(0.0, observation.longitudinal_speed_mps),
            lateral_velocity_mps=observation.lateral_speed_mps,
            yaw_rate_radians_per_second=math.radians(observation.yaw_rate_degrees_per_second),
            steering_angle_radians=math.radians(observation.steering_angle_degrees),
            longitudinal_acceleration_mps2=observation.longitudinal_acceleration_mps2,
            lateral_acceleration_mps2=observation.lateral_acceleration_mps2,
            slip_angle_radians=math.radians(observation.slip_angle_degrees),
        )

    def predict(action: str) -> dict[str, object]:
        state = initial_state()
        throttle = 1.0 if action in {"forward", "nitro"} else 0.0
        brake = 1.0 if action == "backward" else 0.0
        steering = -1.0 if action == "left" else 1.0 if action == "right" else 0.0
        use_nitro = action == "nitro" and nitro_available
        for _ in range(dynamics.physics_hz // dynamics.control_hz):
            state = integrate_vehicle_substep(
                state, dynamics, throttle=throttle, brake=brake,
                steering=steering, nitro=use_nitro, on_track=on_track,
            )
        speed = math.hypot(
            state.longitudinal_velocity_mps, state.lateral_velocity_mps,
        ) * dynamics.pixels_per_meter / dynamics.control_hz
        heading_delta = math.degrees(state.heading_radians)
        if heading_delta > 180:
            heading_delta -= 360
        return {
            "action": action,
            "nitro_applied": use_nitro,
            "heading_delta_degrees": round(heading_delta, 2),
            "next_speed": round(speed, 2),
            "travel_distance_this_tick": round(math.hypot(state.x, state.y), 2),
            "forward_displacement": round(state.x, 2),
            "lateral_displacement": round(abs(state.y), 2),
            "lateral_direction": "left" if state.y < 0 else "right" if state.y > 0 else None,
            "next_yaw_rate_degrees_per_second": round(math.degrees(state.yaw_rate_radians_per_second), 2),
            "next_slip_angle_degrees": round(math.degrees(state.slip_angle_radians), 2),
        }

    brake_state = initial_state()
    braking_ticks = 0
    while brake_state.longitudinal_velocity_mps > .05 and braking_ticks < 200:
        for _ in range(dynamics.physics_hz // dynamics.control_hz):
            brake_state = integrate_vehicle_substep(
                brake_state, dynamics, throttle=0, brake=1, steering=0,
                nitro=False, on_track=on_track,
            )
        braking_ticks += 1
    return {
        "model": dynamics.model,
        "units": {"distance": "pixels and meters", "time": "seconds and control ticks", "heading": "degrees"},
        "surface": scene.surface,
        "currently_on_track": on_track,
        "integration": {
            "physics_hz": dynamics.physics_hz,
            "control_hz": dynamics.control_hz,
            "substeps_per_control": dynamics.physics_hz // dynamics.control_hz,
            "method": "fixed-step semi-implicit transient bicycle",
        },
        "update_order": "slew steering, balance longitudinal forces, resolve grip/yaw/slip, integrate pose, then test track and collisions",
        "primitive_controls_are_mutually_exclusive": True,
        "keyboard_supports_simultaneous_throttle_and_steering": True,
        "lateral_momentum_is_explicit": True,
        "decision_priority": [
            "avoid imminent collision", "remain on drivable road",
            "align with road", "increase speed",
        ],
        "steering_is_not_lateral_dodge": True,
        "off_track_behavior": {
            "terminal": False,
            "speed_cap": OFF_TRACK_SPEED_CAP,
            "reward_penalty_per_tick": OFF_TRACK_REWARD_PENALTY,
            "recovery": "steer back onto the pale drivable road; normal surface limits resume immediately",
        },
        "car_radius": CAR_RADIUS,
        "track_center_safe_half_width": round(scene.track_width / 2 - CAR_RADIUS, 1),
        "limits": {
            "max_speed_mps": vehicle.max_speed_mps,
            "nitro_max_speed_mps": round(vehicle.max_speed_mps * NITRO_MAX_SPEED_MULTIPLIER, 2),
            "max_steering_angle_degrees": vehicle.max_steering_angle_degrees,
            "steering_rate_degrees_per_second": vehicle.steering_rate_degrees_per_second,
            "nitro_capacity": NITRO_CAPACITY,
            "nitro_recharge_per_tick": NITRO_RECHARGE_PER_TICK,
            "nitro_drain_per_tick": NITRO_DRAIN_PER_TICK,
            "nitro_force_n": vehicle.nitro_force_n,
            "nitro_requires_straight_throttle": True,
            "nitro_activation_charge": NITRO_CAPACITY,
            "nitro_must_fully_recharge_after_interruption": True,
        },
        "vehicle": vehicle.model_dump(),
        "road": road.model_dump(),
        "aerodynamics": {
            "air_density_kg_m3": dynamics.air_density_kg_m3,
            "drag_coefficient": vehicle.drag_coefficient,
            "frontal_area_m2": vehicle.frontal_area_m2,
            "lift_coefficient": vehicle.lift_coefficient,
        },
        "braking_from_current_speed": {
            "ticks_to_stop": braking_ticks,
            "distance_until_stopped_pixels": round(math.hypot(brake_state.x, brake_state.y), 2),
        },
        "next_tick_outcomes": {
            action: predict(action)
            for action in ("forward", "nitro", "backward", "left", "right", "idle")
        },
    }


def racing_strategy_context(scene: SceneSpec) -> dict:
    """Compress a public circuit into twelve tactical sectors for one pre-race call."""
    points = scene.track_centerline
    sector_size = len(points) // 12
    entity_sectors: dict[int, dict[str, int]] = {index: {"barriers": 0, "npcs": 0} for index in range(12)}
    for entity in scene.entities:
        if entity.kind not in {EntityKind.OBSTACLE, EntityKind.NPC}:
            continue
        center = _rect_center(entity.rect)
        sector = min(11, _nearest_point_index(points, center) * 12 // len(points))
        entity_sectors[sector]["barriers" if entity.kind == EntityKind.OBSTACLE else "npcs"] += 1
    sectors = []
    for sector in range(12):
        start = sector * sector_size
        middle = (start + sector_size // 2) % len(points)
        end = (start + sector_size) % len(points)
        entry_heading = _bearing(points[start], points[middle])
        exit_heading = _bearing(points[middle], points[end])
        sectors.append({
            "sector": sector,
            "curvature_degrees": round(_angle_delta(entry_heading, exit_heading), 1),
            **entity_sectors[sector],
        })
    return {
        "tool_surface": "racing-strategy-v1",
        "surface": scene.surface,
        "grip": scene.grip,
        "track_width": scene.track_width,
        "laps": scene.laps,
        "sector_gates": scene.sector_count,
        "corners": racing_track_map(scene),
        "opponents": [
            {
                "id": behavior.entity_id, "profile": behavior.profile.value,
                "pace": behavior.pace, "aggression": behavior.aggression,
                "defends": behavior.defends,
            }
            for behavior in scene.npc_behaviors
        ],
        "sectors": sectors,
    }


def _npc_lane_offset(number: int) -> float:
    """Deterministic alternating lane for distributed starts and grid slots."""
    return 34.0 if number % 2 else -34.0


def _stable_phase(entity_id: str) -> float:
    """A fixed phase per car, stable across machines and processes."""
    digest = hashlib.sha256(entity_id.encode()).hexdigest()[:6]
    return int(digest, 16) / 0xFFFFFF * math.tau


def _npc_safe_gap(behavior: NpcBehaviorSpec) -> float:
    """Lateral clearance an opponent insists on before committing to a move."""
    return CAR_RADIUS * (3.2 - 1.35 * behavior.aggression)


def _offset_track_point(points: list[Vec2], index: int, offset: float) -> Vec2:
    current = points[index % len(points)]
    before = points[(index - 1) % len(points)]
    after = points[(index + 1) % len(points)]
    dx, dy = after.x - before.x, after.y - before.y
    length = max(1.0, math.hypot(dx, dy))
    return Vec2(x=current.x - dy / length * offset, y=current.y + dx / length * offset)


def _stable_seed(prompt: str) -> int:
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)


def _rect_center(rect: Rect) -> Vec2:
    return Vec2(x=rect.x + rect.width / 2, y=rect.y + rect.height / 2)


def _bearing(origin: Vec2, target: Vec2) -> float:
    return math.degrees(math.atan2(target.y - origin.y, target.x - origin.x)) % 360


def _track_heading(points: list[Vec2], position: Vec2) -> float:
    index = _nearest_point_index(points, position)
    return _bearing(points[(index - 1) % len(points)], points[(index + 1) % len(points)])


def _angle_delta(current: float, target: float) -> float:
    return (target - current + 180) % 360 - 180


def _nearest_point_index(points: list[Vec2], point: Vec2) -> int:
    return min(range(len(points)), key=lambda index: math.hypot(points[index].x - point.x, points[index].y - point.y))


def _nearest_track_index_with_hint(points: list[Vec2], point: Vec2, hint: int | None) -> int:
    """Track-local nearest point that cannot teleport across close chicane arms."""
    if hint is None:
        return _nearest_point_index(points, point)
    candidates = {(hint + offset) % len(points) for offset in range(-4, 13)}
    return min(candidates, key=lambda index: math.hypot(points[index].x - point.x, points[index].y - point.y))


def _forward_track_index(points: list[Vec2], point: Vec2, current: int) -> int:
    """Return the closest reachable forward sample without seam regressions."""
    candidates = [(current + offset) % len(points) for offset in range(13)]
    return min(
        candidates,
        key=lambda index: (points[index].x - point.x) ** 2 + (points[index].y - point.y) ** 2,
    )


def _signed_lane_offset(points: list[Vec2], point: Vec2, index: int) -> float:
    center = points[index % len(points)]
    before = points[(index - 1) % len(points)]
    after = points[(index + 1) % len(points)]
    dx, dy = after.x - before.x, after.y - before.y
    length = max(1.0, math.hypot(dx, dy))
    return (point.x - center.x) * (-dy / length) + (point.y - center.y) * (dx / length)


def _move_toward(value: float, target: float, maximum_delta: float) -> float:
    if value < target:
        return min(target, value + maximum_delta)
    return max(target, value - maximum_delta)


def _cyclic_index_delta(origin: int, target: int, size: int) -> int:
    """Shortest signed index delta; positive values are ahead on the lap."""
    return (target - origin + size // 2) % size - size // 2


def _keys_for_action(action: ActionName) -> tuple[str, ...]:
    return {
        ActionName.FORWARD: ("w",),
        ActionName.BACKWARD: ("s",),
        ActionName.LEFT: ("a",),
        ActionName.RIGHT: ("d",),
        ActionName.IDLE: (),
        ActionName.NITRO: ("w", "space"),
    }[action]


def _primary_action(keys: tuple[str, ...]) -> ActionName:
    if "space" in keys and "w" in keys and "a" not in keys and "d" not in keys:
        return ActionName.NITRO
    if "a" in keys:
        return ActionName.LEFT
    if "d" in keys:
        return ActionName.RIGHT
    if "w" in keys:
        return ActionName.FORWARD
    if "s" in keys:
        return ActionName.BACKWARD
    return ActionName.IDLE


def _distance_to_polyline(point: Vec2, points: list[Vec2], closed: bool) -> float:
    pairs = list(zip(points, points[1:]))
    if closed:
        pairs.append((points[-1], points[0]))
    return min(_distance_to_segment(point, start, end) for start, end in pairs)


def _distance_to_segment(point: Vec2, start: Vec2, end: Vec2) -> float:
    dx, dy = end.x - start.x, end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(point.x - start.x, point.y - start.y)
    t = max(0.0, min(1.0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared))
    projection = Vec2(x=start.x + t * dx, y=start.y + t * dy)
    return math.hypot(point.x - projection.x, point.y - projection.y)


def _checkpoint_coordinates(
    scene: SceneSpec, point: Vec2, checkpoint: EntitySpec,
) -> tuple[float, float]:
    """Return longitudinal/lateral position in an oriented checkpoint frame."""
    center = _rect_center(checkpoint.rect)
    index = _nearest_point_index(scene.track_centerline, center)
    before = scene.track_centerline[(index - 1) % len(scene.track_centerline)]
    after = scene.track_centerline[(index + 1) % len(scene.track_centerline)]
    dx, dy = after.x - before.x, after.y - before.y
    length = max(1e-9, math.hypot(dx, dy))
    forward_x, forward_y = dx / length, dy / length
    side_x, side_y = -forward_y, forward_x
    relative_x, relative_y = point.x - center.x, point.y - center.y
    return (
        relative_x * forward_x + relative_y * forward_y,
        relative_x * side_x + relative_y * side_y,
    )


def _crossed_checkpoint_gate(
    scene: SceneSpec, previous: Vec2, current: Vec2,
    checkpoint: EntitySpec, radius: float,
) -> bool:
    """Detect forward entry into the oriented 16px gate drawn by renderers."""
    previous_longitudinal, previous_lateral = _checkpoint_coordinates(scene, previous, checkpoint)
    current_longitudinal, current_lateral = _checkpoint_coordinates(scene, current, checkpoint)
    leading_edge = -8 - radius
    if not previous_longitudinal < leading_edge <= current_longitudinal:
        return False
    progress = (leading_edge - previous_longitudinal) / max(
        1e-9, current_longitudinal - previous_longitudinal,
    )
    crossing_lateral = previous_lateral + (current_lateral - previous_lateral) * progress
    return abs(crossing_lateral) <= scene.track_width / 2 + radius

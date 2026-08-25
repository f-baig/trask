"""Local fixed-tick keyboard player for the 3D racing runtime.

This drives `Racing3DWorld.step` — the same authoritative function policies use —
at the scene's control rate, and interpolates between authoritative states for a
60 FPS picture. Interpolation is display-only: it never feeds back into physics,
collision, or scoring, exactly as in the 2D player.

The start lights, terminal result screen, and interpolation helper are imported
from the 2D player rather than reimplemented, and the window keeps the same
geometry so the two versions of the game look and behave like siblings.

There is deliberately no plan-view inset. A corner-of-the-screen map answers the
question the 3D cameras exist to pose — what can the driver see from here — so
having one made the perspective views decorative rather than load-bearing.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .models import Action
from .play import (
    VIEW_SIZE, WINDOW_SIZE, _draw_start_lights, _draw_terminal_overlay, _interpolated_world,
)
from .racing import NITRO_CAPACITY
from .racing3d import Racing3DWorld
from .view3d import DEFAULT_ROAD_DETAIL, ViewMode, render_view_surface


CAMERA_ORDER = (
    ViewMode.THIRD_PERSON,
    ViewMode.FIRST_PERSON,
    ViewMode.HOOD,
    ViewMode.THIRD_PERSON_FAR,
    ViewMode.OVERHEAD_3D,
)
CAMERA_LABELS = {
    ViewMode.FIRST_PERSON: "COCKPIT",
    ViewMode.HOOD: "BUMPER",
    ViewMode.THIRD_PERSON: "CHASE",
    ViewMode.THIRD_PERSON_FAR: "CHASE FAR",
    ViewMode.OVERHEAD_3D: "OVERHEAD",
}


def play_scene_3d(
    scene, view: ViewMode = ViewMode.THIRD_PERSON,
    road_detail: int = DEFAULT_ROAD_DETAIL,
) -> dict[str, object]:
    """Run a human-controlled 3D episode on the shared racing runtime."""
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("Manual play requires the pygame-ce native dependency") from error

    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(f"RaceLab 3D — {scene.name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 17)
    small = pygame.font.SysFont("Menlo", 13)
    result_title = pygame.font.SysFont("Menlo", 54, bold=True)
    result_outcome = pygame.font.SysFont("Menlo", 28, bold=True)

    world = Racing3DWorld.from_scene(scene)
    camera_index = CAMERA_ORDER.index(view) if view in CAMERA_ORDER else 0
    previous_player = world.player.model_copy()
    previous_heading = world.heading
    previous_opponents = _opponent_snapshot(world)
    previous_attitude = _attitude_snapshot(world)
    accumulator = 0.0
    running = True

    while running:
        elapsed = min(clock.tick(60) / 1000, 0.25)
        accumulator = 0.0 if world.terminated else accumulator + elapsed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    world = Racing3DWorld.from_scene(scene)
                    previous_player = world.player.model_copy()
                    previous_heading = world.heading
                    previous_opponents = _opponent_snapshot(world)
                    previous_attitude = _attitude_snapshot(world)
                    accumulator = 0.0
                elif event.key == pygame.K_c:
                    camera_index = (camera_index + 1) % len(CAMERA_ORDER)
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    camera_index = event.key - pygame.K_1

        keys = _held_keys(pygame)
        while accumulator >= 1 / world.dynamics.control_hz:
            if not world.terminated:
                previous_player = world.player.model_copy()
                previous_heading = world.heading
                previous_opponents = _opponent_snapshot(world)
                previous_attitude = _attitude_snapshot(world)
                world.step(Action(keys=keys))
                if world.terminated:
                    # Collapse interpolation onto the authoritative terminal pose
                    # so the wreck does not rock between two states forever.
                    previous_player = world.player.model_copy()
                    previous_heading = world.heading
                    previous_opponents = _opponent_snapshot(world)
                    previous_attitude = _attitude_snapshot(world)
                    accumulator = 0.0
                    break
            accumulator -= 1 / world.dynamics.control_hz

        mode = CAMERA_ORDER[camera_index % len(CAMERA_ORDER)]
        render_world = _interpolated_world_3d(
            world, previous_player, previous_heading, previous_opponents, previous_attitude,
            min(1.0, accumulator * world.dynamics.control_hz),
        )
        window.blit(
            render_view_surface(render_world, mode, *VIEW_SIZE, road_detail), (0, 0),
        )
        _draw_camera_badge(pygame, window, small, mode)
        _draw_start_lights(pygame, window, font, world)
        _draw_hud(pygame, window, font, small, world, scene)
        if world.terminated:
            _draw_terminal_overlay(
                pygame, window, result_title, result_outcome, font, small, world,
            )
        pygame.display.flip()

    result = {
        "succeeded": world.succeeded,
        "won": world.succeeded and (world.player_position or 1) == 1,
        "position": world.player_position,
        "field_size": world.field_size,
        "finish_order": list(world.finish_order),
        "reason": world.reason or "player quit",
        "steps": world.step_number,
        "checkpoint_index": world.objective_index,
        "laps_completed": min(scene.laps, world.objective_index // scene.sector_count),
        "view": CAMERA_ORDER[camera_index % len(CAMERA_ORDER)].value,
        "road_detail": road_detail,
        "relief_pixels": round(world.surface.relief_pixels, 2) if world.surface else 0.0,
    }
    pygame.quit()
    return result


def _held_keys(pygame) -> list[str]:
    """Read the physical keyboard into a valid transport action."""
    pressed = pygame.key.get_pressed()
    keys: list[str] = []
    if pressed[pygame.K_w] or pressed[pygame.K_UP]:
        keys.append("w")
    if pressed[pygame.K_s] or pressed[pygame.K_DOWN]:
        keys.append("s")
    if pressed[pygame.K_a] or pressed[pygame.K_LEFT]:
        keys.append("a")
    if pressed[pygame.K_d] or pressed[pygame.K_RIGHT]:
        keys.append("d")
    if pressed[pygame.K_SPACE]:
        keys.append("space")
    # Opposing pairs cancel for a physical keyboard rather than producing an
    # action the transport would reject.
    if "w" in keys and "s" in keys:
        keys = [key for key in keys if key not in {"w", "s"}]
    if "a" in keys and "d" in keys:
        keys = [key for key in keys if key not in {"a", "d"}]
    return keys


def _opponent_snapshot(world: Racing3DWorld) -> dict[str, tuple]:
    return {
        item.entity_id: (item.position.model_copy(), item.heading)
        for item in world.opponents
    }


def _attitude_snapshot(world: Racing3DWorld) -> tuple[float, float]:
    """The load-derived part of chassis attitude, which needs interpolating."""
    return (world.squat_degrees, world.lean_degrees)


def _interpolated_world_3d(
    world: Racing3DWorld, previous_player, previous_heading, previous_opponents,
    previous_attitude: tuple[float, float], alpha: float,
) -> Racing3DWorld:
    """Interpolate the planar pose and body attitude, then resolve road height.

    Grade and bank are recomputed from the interpolated position, so a car is
    never drawn floating above or sunk into a crest. Squat and lean are held for a
    whole control tick by the physics and then jump, so they are interpolated
    across the tick instead; without that the camera visibly stutters six times a
    second in the in-car views.
    """
    rendered = _interpolated_world(
        world, previous_player, previous_heading, previous_opponents, alpha,
    )
    rendered.opponents = [
        replace(opponent, position=opponent.position.model_copy())
        for opponent in rendered.opponents
    ]
    previous_squat, previous_lean = previous_attitude
    rendered._refresh_vertical_state(
        squat_degrees=previous_squat + (world.squat_degrees - previous_squat) * alpha,
        lean_degrees=previous_lean + (world.lean_degrees - previous_lean) * alpha,
    )
    return rendered


def _draw_camera_badge(pygame, window, small, mode: ViewMode) -> None:
    label = f"{CAMERA_LABELS[mode]}  ·  C CYCLES"
    rendered = small.render(label, True, (232, 236, 226))
    badge = pygame.Surface((rendered.get_width() + 18, 24), pygame.SRCALPHA)
    badge.fill((8, 12, 14, 190))
    badge.blit(rendered, (9, 5))
    window.blit(badge, (18, VIEW_SIZE[1] - 42))


def _draw_hud(pygame, window, font, small, world: Racing3DWorld, scene) -> None:
    """Draw the same telemetry strip as the 2D player, plus the gradient."""
    pygame.draw.rect(window, (14, 17, 20), (0, VIEW_SIZE[1], WINDOW_SIZE[0], 60))
    status = (
        "FINISHED — R restarts"
        if world.succeeded else (
            f"CRASHED — {world.reason} — R restarts"
            if (world.reason or "").lower().startswith("collision")
            else f"RACE LOST — {world.reason} — R restarts"
        )
        if world.terminated else
        f"P{world.live_position}/{world.field_size}  "
        f"LAP {min(scene.laps, world.objective_index // scene.sector_count + 1)}/{scene.laps}  "
        f"SECTOR {world.objective_index % scene.sector_count + 1}/{scene.sector_count}"
    )
    terrain = "OFF TRACK" if world.off_track else scene.surface.upper()
    drive_mode = (
        "NITRO" if world.nitro_active else
        f"TURN {abs(world.lateral_acceleration_mps2) / world.dynamics.gravity_mps2:.2f}G · "
        f"SLIP {abs(math.degrees(world.slip_angle_radians)):.1f}°" if world.turning else
        f"STRAIGHT · DRAG {world.aerodynamic_drag_n:.0f}N"
    )
    grade = world.grade_degrees
    slope = (
        f"CLIMB {grade:.0f}°" if grade > 1.2
        else f"DESCENT {abs(grade):.0f}°" if grade < -1.2
        else "LEVEL"
    )
    hud = (
        f"{status}    SPEED {world.longitudinal_velocity_mps * 3.6:03.0f} KM/H    "
        f"{drive_mode}    {terrain}    {slope}"
    )
    window.blit(font.render(hud, True, (245, 240, 226)), (18, VIEW_SIZE[1] + 10))

    meter = pygame.Rect(748, VIEW_SIZE[1] + 36, 190, 10)
    pygame.draw.rect(window, (48, 55, 60), meter, border_radius=4)
    fill = meter.copy()
    fill.width = round(meter.width * world.nitro / 100)
    pygame.draw.rect(window, (66, 205, 232), fill, border_radius=4)
    nitro_label = (
        "BURN" if world.nitro_active
        else "READY" if world.nitro >= NITRO_CAPACITY else "CHARGING"
    )
    window.blit(small.render(f"NITRO {nitro_label}", True, (180, 224, 235)), (646, VIEW_SIZE[1] + 34))
    help_text = "WASD · Space nitro · C camera · R restart · Q quit"
    window.blit(small.render(help_text, True, (166, 174, 178)), (18, VIEW_SIZE[1] + 38))

"""Local fixed-tick keyboard player for exercising the real racing runtime."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

from .models import Action
from .racing import NITRO_CAPACITY, RacingWorld
from .vision import render_racing_overhead_surface


WINDOW_SIZE = (960, 700)
VIEW_SIZE = (960, 640)


def play_scene(scene) -> dict[str, object]:
    """Run a human-controlled episode using the same step function as policies."""
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("Manual play requires the pygame-ce native dependency") from error

    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(f"RaceLab — {scene.name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 17)
    small = pygame.font.SysFont("Menlo", 13)
    result_title = pygame.font.SysFont("Menlo", 54, bold=True)
    result_outcome = pygame.font.SysFont("Menlo", 28, bold=True)
    world = RacingWorld.from_scene(scene)
    previous_player = world.player.model_copy()
    previous_heading = world.heading
    previous_opponents = {
        item.entity_id: (item.position.model_copy(), item.heading)
        for item in world.opponents
    }
    accumulator = 0.0
    running = True

    while running:
        elapsed = min(clock.tick(60) / 1000, 0.25)
        if world.terminated:
            accumulator = 0.0
        else:
            accumulator += elapsed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    world = RacingWorld.from_scene(scene)
                    previous_player = world.player.model_copy()
                    previous_heading = world.heading
                    previous_opponents = {
                        item.entity_id: (item.position.model_copy(), item.heading)
                        for item in world.opponents
                    }
                    accumulator = 0.0

        pressed = pygame.key.get_pressed()
        keys = []
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
        # Opposing pairs cancel for a physical keyboard instead of producing
        # an invalid transport action.
        if "w" in keys and "s" in keys:
            keys = [key for key in keys if key not in {"w", "s"}]
        if "a" in keys and "d" in keys:
            keys = [key for key in keys if key not in {"a", "d"}]

        while accumulator >= 1 / world.dynamics.control_hz:
            if not world.terminated:
                previous_player = world.player.model_copy()
                previous_heading = world.heading
                previous_opponents = {
                    item.entity_id: (item.position.model_copy(), item.heading)
                    for item in world.opponents
                }
                world.step(Action(keys=keys))
                if world.terminated:
                    # Collapse interpolation onto the authoritative terminal
                    # pose. Otherwise the render alpha restarts every frame and
                    # visibly rocks between the pre-impact and terminal states.
                    previous_player = world.player.model_copy()
                    previous_heading = world.heading
                    previous_opponents = {
                        item.entity_id: (item.position.model_copy(), item.heading)
                        for item in world.opponents
                    }
                    accumulator = 0.0
                    break
            accumulator -= 1 / world.dynamics.control_hz

        render_world = _interpolated_world(
            world, previous_player, previous_heading, previous_opponents,
            min(1.0, accumulator * world.dynamics.control_hz),
        )
        window.blit(render_racing_overhead_surface(
            render_world, *VIEW_SIZE, include_checkpoints=True,
        ), (0, 0))
        _draw_start_lights(pygame, window, font, world)
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
            f"TURN {abs(world.lateral_acceleration_mps2) / world.dynamics.gravity_mps2:.2f}G · SLIP {abs(math.degrees(world.slip_angle_radians)):.1f}°" if world.turning else
            f"STRAIGHT · DRAG {world.aerodynamic_drag_n:.0f}N"
        )
        hud = f"{status}    SPEED {world.longitudinal_velocity_mps * 3.6:03.0f} KM/H    {drive_mode}    {terrain}"
        window.blit(font.render(hud, True, (245, 240, 226)), (18, 650))
        meter = pygame.Rect(748, 676, 190, 10)
        pygame.draw.rect(window, (48, 55, 60), meter, border_radius=4)
        fill = meter.copy()
        fill.width = round(meter.width * world.nitro / 100)
        pygame.draw.rect(window, (66, 205, 232), fill, border_radius=4)
        nitro_label = "BURN" if world.nitro_active else "READY" if world.nitro >= NITRO_CAPACITY else "CHARGING"
        window.blit(small.render(f"NITRO {nitro_label}", True, (180, 224, 235)), (646, 674))
        help_text = "WASD / arrows drive · Space nitro at 100% (straight only) · R restart · Q/Esc quit"
        window.blit(small.render(help_text, True, (166, 174, 178)), (18, 678))
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
    }
    pygame.quit()
    return result


def _interpolated_world(world, previous_player, previous_heading, previous_opponents, alpha: float):
    """Create a display-only state one control tick behind for smooth 60 FPS motion."""
    rendered = copy.copy(world)
    if world.terminated:
        rendered.player = world.player.model_copy()
        rendered.opponents = [
            replace(opponent, position=opponent.position.model_copy())
            for opponent in world.opponents
        ]
        return rendered
    if world.barrier_impact is not None:
        # Display the within-tick path as approach -> contact -> rebound. A
        # straight interpolation from the previous point to the final repelled
        # point would visually cut through the solid barrier physics avoided.
        if alpha < .65:
            phase = alpha / .65
            origin, target = previous_player, world.barrier_impact
        else:
            phase = (alpha - .65) / .35
            origin, target = world.barrier_impact, world.player
        rendered.player = origin.model_copy(update={
            "x": origin.x + (target.x - origin.x) * phase,
            "y": origin.y + (target.y - origin.y) * phase,
        })
    else:
        rendered.player = previous_player.model_copy(update={
            "x": previous_player.x + (world.player.x - previous_player.x) * alpha,
            "y": previous_player.y + (world.player.y - previous_player.y) * alpha,
        })
    heading_delta = (world.heading - previous_heading + 180) % 360 - 180
    rendered.heading = (previous_heading + heading_delta * alpha) % 360
    rendered.opponents = []
    for opponent in world.opponents:
        prior_position, prior_heading = previous_opponents.get(
            opponent.entity_id, (opponent.position, opponent.heading),
        )
        opponent_heading_delta = (opponent.heading - prior_heading + 180) % 360 - 180
        rendered.opponents.append(replace(
            opponent,
            position=prior_position.model_copy(update={
                "x": prior_position.x + (opponent.position.x - prior_position.x) * alpha,
                "y": prior_position.y + (opponent.position.y - prior_position.y) * alpha,
            }),
            heading=(prior_heading + opponent_heading_delta * alpha) % 360,
        ))
    return rendered


def _terminal_message(world: RacingWorld) -> tuple[str, str, str]:
    """Return stable display copy for current and future terminal outcomes.

    Finishing the configured laps is not the same as winning. Opponents race the
    same distance and finish independently, so the outcome is the player's actual
    position in the field rather than the mere fact that the flag was reached.
    """
    if world.succeeded:
        position, field = world.player_position or 1, world.field_size
        if field == 1:
            return "RACE ENDED", "COMPLETED", world.reason or "Race completed"
        ahead = position - 1
        outcome = "YOU WON" if position == 1 else f"P{position} OF {field}"
        detail = (
            f"Finished P{position} of {field}"
            + ("" if position == 1 else f" — {ahead} car{'s' if ahead > 1 else ''} ahead")
        )
        return "RACE ENDED", outcome, detail
    reason = world.reason or "race ended"
    if reason.lower().startswith("collision"):
        return "CRASH", "CAR DISABLED", reason
    return "RACE ENDED", "YOU LOST", reason


def _draw_terminal_overlay(pygame, window, title_font, outcome_font, font, small, world: RacingWorld) -> None:
    title, outcome, detail = _terminal_message(world)
    won = world.succeeded and (world.player_position or 1) == 1
    accent = (66, 211, 126) if won else (245, 172, 48) if world.succeeded else (255, 86, 61)
    veil = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
    veil.fill((5, 8, 10, 218))
    window.blit(veil, (0, 0))

    panel = pygame.Rect(190, 176, 580, 330)
    pygame.draw.rect(window, (14, 19, 22), panel, border_radius=18)
    pygame.draw.rect(window, accent, panel, 3, border_radius=18)
    pygame.draw.rect(window, accent, (panel.x, panel.y, 8, panel.height), border_radius=4)

    title_surface = title_font.render(title, True, (246, 242, 229))
    window.blit(title_surface, title_surface.get_rect(center=(panel.centerx, panel.y + 74)))
    outcome_surface = outcome_font.render(outcome, True, accent)
    window.blit(outcome_surface, outcome_surface.get_rect(center=(panel.centerx, panel.y + 137)))
    detail_surface = font.render(detail.upper(), True, (176, 185, 187))
    window.blit(detail_surface, detail_surface.get_rect(center=(panel.centerx, panel.y + 190)))

    pygame.draw.line(
        window, (61, 70, 73),
        (panel.x + 72, panel.y + 224), (panel.right - 72, panel.y + 224), 1,
    )
    restart_surface = outcome_font.render("PRESS R TO RESTART", True, (246, 242, 229))
    window.blit(restart_surface, restart_surface.get_rect(center=(panel.centerx, panel.y + 266)))
    quit_surface = small.render("Q / ESC TO QUIT", True, (126, 137, 140))
    window.blit(quit_surface, quit_surface.get_rect(center=(panel.centerx, panel.y + 304)))


def _draw_start_lights(pygame, window, font, world: RacingWorld) -> None:
    """Render a compact motorsport start gantry over authoritative countdown state."""
    if world.countdown_ticks_remaining <= 0 and world.step_number > world.dynamics.control_hz * 3 + 5:
        return
    active_number = math.ceil(world.countdown_ticks_remaining / world.dynamics.control_hz)
    panel = pygame.Surface((270, 92), pygame.SRCALPHA)
    pygame.draw.rect(panel, (10, 14, 16, 226), panel.get_rect(), border_radius=14)
    pygame.draw.rect(panel, (236, 240, 231, 150), panel.get_rect(), 2, border_radius=14)
    colors = ((226, 58, 47), (245, 172, 48), (57, 211, 112))
    for index, color in enumerate(colors, start=1):
        center = (62 + (index - 1) * 73, 43)
        pygame.draw.circle(panel, (35, 42, 44), center, 24)
        lit = active_number == 4 - index if world.countdown_ticks_remaining > 0 else index == 3
        pygame.draw.circle(panel, color if lit else tuple(component // 4 for component in color), center, 18)
    label = str(active_number) if active_number > 0 else "GO"
    rendered = font.render(label, True, (250, 247, 230))
    panel.blit(rendered, rendered.get_rect(center=(135, 78)))
    window.blit(panel, (345, 36))

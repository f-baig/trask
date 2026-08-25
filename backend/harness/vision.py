"""Policy-facing rasterization for the 2D racing engine."""

from __future__ import annotations

import base64
import io
import math
import os

from .collision import EDGE_BARRIER_THICKNESS, collider_for, outline, track_edge_points
from .models import EntityKind, Vec2
from .policy_protocol import VisualFrame


SURFACE_PALETTES = {
    # High-separation sensor colors: outside terrain, drivable road, boundary.
    "asphalt": ((52, 120, 65), (42, 45, 52), (248, 241, 225)),
    "clay": ((77, 101, 55), (171, 91, 57), (255, 232, 202)),
    "ice": ((70, 106, 114), (185, 220, 226), (248, 255, 255)),
}


def render_racing_policy_frame(world, width: int = 480, height: int = 320) -> VisualFrame:
    """Render only visible game state—no reward, oracle line, or evaluator labels."""
    if os.environ.get("RACING_POLICY_VIEW", "overhead") == "forward-cone":
        return render_racing_forward_cone(world, width, height)
    surface = render_racing_overhead_surface(world, width, height)
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("RGB policy observations require the native pygame dependency") from error
    buffer = io.BytesIO()
    pygame.image.save(surface, buffer, "policy-frame.png")
    return VisualFrame(
        media_type="image/png", data_base64=base64.b64encode(buffer.getvalue()).decode(),
        width=width, height=height, channels=3, viewpoint="overhead",
        orientation="north-up", ego_anchor="world-position",
    )


def render_racing_overhead_surface(
    world, width: int = 480, height: int = 320, *, include_checkpoints: bool = False,
):
    """Draw authoritative live state to a Pygame surface for policies or humans."""
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("RGB policy observations require the native pygame dependency") from error

    scene = world.scene
    surface = pygame.Surface((width, height))
    ground, road, edge = SURFACE_PALETTES[scene.surface]
    surface.fill(ground)
    sx, sy = width / scene.bounds.width, height / scene.bounds.height

    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * sx), round(y * sy)

    track = [point(item.x, item.y) for item in scene.track_centerline]
    scale = (sx + sy) / 2
    road_width = max(8, round(scene.track_width * scale))
    _draw_round_track(pygame, surface, track, road_width + 10, edge)
    _draw_round_track(pygame, surface, track, road_width, road)
    _draw_edge_barriers(pygame, surface, scene, point, scale)

    live_opponents = {opponent.entity_id: opponent for opponent in world.opponents}
    checkpoint_number = 0
    for entity in scene.entities:
        if entity.kind == EntityKind.CHECKPOINT:
            if include_checkpoints:
                center_world = (
                    entity.rect.x + entity.rect.width / 2,
                    entity.rect.y + entity.rect.height / 2,
                )
                center = point(*center_world)
                heading = math.radians(_track_heading(
                    scene.track_centerline, Vec2(x=center_world[0], y=center_world[1]),
                ))
                side = (-math.sin(heading), math.cos(heading))
                half_width = road_width / 2
                start = (round(center[0] - side[0] * half_width), round(center[1] - side[1] * half_width))
                end = (round(center[0] + side[0] * half_width), round(center[1] + side[1] * half_width))
                passed = checkpoint_number < world.objective_index
                color = (112, 120, 122) if passed else (255, 107, 44) if entity.id == "finish-line" else (244, 211, 94)
                pygame.draw.line(surface, color, start, end, max(3, round(16 * scale)))
                checkpoint_number += 1
            continue
        if entity.kind == EntityKind.NPC:
            opponent = live_opponents[entity.id]
            center = opponent.position
            _draw_car(
                pygame, surface, point(center.x, center.y),
                opponent.heading, (52, 166, 230), scale,
                nitro=opponent.nitro_active,
            )
        elif entity.kind == EntityKind.OBSTACLE:
            # Draw the shape collision actually tests, so what is visible is what
            # the car can hit rather than a stand-in circle over a rectangle.
            collider = collider_for(entity, world.obstacle_shift)
            hull = [point(x, y) for x, y in outline(collider)]
            pygame.draw.polygon(surface, (237, 79, 55), hull)
            pygame.draw.polygon(surface, (28, 32, 34), hull, max(1, round(2 * scale)))

    _draw_car(
        pygame, surface, point(world.player.x, world.player.y), world.heading,
        (247, 242, 230), scale, accent=(255, 90, 54), nitro=world.nitro_active,
    )
    if world.barrier_impact is not None:
        impact = point(world.barrier_impact.x, world.barrier_impact.y)
        radius = max(6, round(18 * scale))
        pygame.draw.circle(surface, (255, 216, 84), impact, radius, max(2, round(3 * scale)))
        for angle_degrees in (0, 72, 144, 216, 288):
            angle = math.radians(angle_degrees)
            inner = (
                round(impact[0] + math.cos(angle) * radius * .65),
                round(impact[1] + math.sin(angle) * radius * .65),
            )
            outer = (
                round(impact[0] + math.cos(angle) * radius * 1.4),
                round(impact[1] + math.sin(angle) * radius * 1.4),
            )
            pygame.draw.line(surface, (255, 241, 168), inner, outer, max(2, round(3 * scale)))
    return surface


def render_racing_forward_cone(
    world, width: int = 480, height: int = 320,
    horizontal_fov_degrees: float = 120.0, range_pixels: float = 330.0,
) -> VisualFrame:
    """Render a car-centric top-down cone; behind/out-of-FOV state is black."""
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("RGB policy observations require the native pygame dependency") from error

    scene = world.scene
    ground, road, edge = SURFACE_PALETTES[scene.surface]
    rendered = pygame.Surface((width, height))
    rendered.fill(ground)
    player_screen = (width // 2, height - 28)
    scale = (height - 38) / range_pixels
    heading = math.radians(world.heading)
    forward = (math.cos(heading), math.sin(heading))
    right = (-forward[1], forward[0])

    def point(x: float, y: float) -> tuple[int, int]:
        dx, dy = x - world.player.x, y - world.player.y
        forward_distance = dx * forward[0] + dy * forward[1]
        lateral_distance = dx * right[0] + dy * right[1]
        return (
            round(player_screen[0] + lateral_distance * scale),
            round(player_screen[1] - forward_distance * scale),
        )

    track = [point(item.x, item.y) for item in scene.track_centerline]
    road_width = max(8, round(scene.track_width * scale))
    _draw_round_track(pygame, rendered, track, road_width + 10, edge)
    _draw_round_track(pygame, rendered, track, road_width, road)
    _draw_edge_barriers(pygame, rendered, scene, point, scale)

    live_opponents = {opponent.entity_id: opponent for opponent in world.opponents}
    for entity in scene.entities:
        if entity.kind == EntityKind.CHECKPOINT:
            continue
        if entity.kind == EntityKind.NPC:
            opponent = live_opponents[entity.id]
            center = opponent.position
            relative_heading = (opponent.heading - world.heading - 90) % 360
            _draw_car(
                pygame, rendered, point(center.x, center.y), relative_heading,
                (52, 166, 230), scale, nitro=opponent.nitro_active,
            )
        elif entity.kind == EntityKind.OBSTACLE:
            center = point(
                entity.rect.x + entity.rect.width / 2,
                entity.rect.y + entity.rect.height / 2 + world.obstacle_shift,
            )
            pygame.draw.circle(rendered, (237, 79, 55), center, max(4, round(10 * scale)))
            pygame.draw.circle(rendered, (28, 32, 34), center, max(2, round(5 * scale)))

    # The mask is the authoritative sensor boundary. Rendering the scene first
    # keeps track clipping smooth; multiplying by the mask removes every pixel
    # behind the car or outside the requested forward field of view.
    half_angle = math.radians(horizontal_fov_degrees / 2)
    cone_half_width = range_pixels * math.tan(half_angle) * scale
    cone_top_y = player_screen[1] - range_pixels * scale
    mask = pygame.Surface((width, height))
    mask.fill((0, 0, 0))
    pygame.draw.polygon(mask, (255, 255, 255), [
        player_screen,
        (round(player_screen[0] - cone_half_width), round(cone_top_y)),
        (round(player_screen[0] + cone_half_width), round(cone_top_y)),
    ])
    rendered.blit(mask, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    heading_guide = os.environ.get("RACING_HEADING_GUIDE", "0").lower() in {"1", "true", "yes", "on"}
    if heading_guide:
        _draw_heading_guide(pygame, rendered, player_screen)
    _draw_car(
        pygame, rendered, player_screen, -90, (247, 242, 230), scale,
        accent=(255, 90, 54), nitro=world.nitro_active,
    )

    buffer = io.BytesIO()
    pygame.image.save(rendered, buffer, "policy-forward-cone.png")
    return VisualFrame(
        media_type="image/png", data_base64=base64.b64encode(buffer.getvalue()).decode(),
        width=width, height=height, channels=3, viewpoint="forward-cone",
        orientation="ego-forward-up", ego_anchor="bottom-center",
        heading_guide=heading_guide,
        heading_guide_semantics="current-ego-heading" if heading_guide else None,
        horizontal_fov_degrees=horizontal_fov_degrees, range_pixels=range_pixels,
    )


def _draw_edge_barriers(pygame, surface, scene, point, scale: float) -> None:
    """Draw the continuous walls at the exact offsets used by collision."""
    if not scene.edge_barriers or len(scene.track_centerline) < 3:
        return
    offset = scene.track_width / 2 + EDGE_BARRIER_THICKNESS / 2
    thickness = max(3, round(EDGE_BARRIER_THICKNESS * scale))
    left, right = track_edge_points(scene, offset)
    for edge_points in (left, right):
        screen_points = [point(x, y) for x, y in edge_points]
        pygame.draw.lines(
            surface, (20, 24, 27), True, screen_points, thickness + max(2, round(4 * scale)),
        )
        pygame.draw.lines(surface, (237, 79, 55), True, screen_points, thickness)
        # Repeated pale caps make this unambiguously a physical guardrail rather
        # than another painted road-edge line, even in a small policy frame.
        stride = max(1, len(screen_points) // 28)
        for post in screen_points[::stride]:
            pygame.draw.circle(surface, (246, 239, 222), post, max(1, thickness // 3))


def _draw_heading_guide(pygame, surface, player_screen: tuple[int, int]) -> None:
    """Draw a compact trajectory ray without revealing the desired racing line."""
    color = (255, 222, 64)
    x, bottom = player_screen[0], player_screen[1] - 15
    top = max(18, bottom - 118)
    dash_length, gap = 12, 7
    y = bottom
    while y > top:
        pygame.draw.line(surface, color, (x, y), (x, max(top, y - dash_length)), 3)
        y -= dash_length + gap
    pygame.draw.line(surface, color, (x, top), (x - 8, top + 11), 3)
    pygame.draw.line(surface, color, (x, top), (x + 8, top + 11), 3)


def _draw_round_track(pygame, surface, points: list[tuple[int, int]], width: int, color: tuple[int, int, int]) -> None:
    """Rasterize the closed centerline as a geometric union with round joins."""
    radius = max(1, width // 2)
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        pygame.draw.line(surface, color, start, end, width)
        pygame.draw.circle(surface, color, start, radius)


def _draw_car(
    pygame, surface, center: tuple[int, int], heading: float,
    color: tuple[int, int, int], scale: float,
    accent: tuple[int, int, int] = (245, 205, 65), *, nitro: bool = False,
) -> None:
    """Draw a readable top-down sports car, sharing the engine's heading axis."""
    half_length, half_width = max(8, round(19 * scale)), max(4, round(9 * scale))
    radians = math.radians(heading)
    forward = (math.cos(radians), math.sin(radians))
    side = (-forward[1], forward[0])

    def at(longitudinal: float, lateral: float) -> tuple[int, int]:
        return (
            round(center[0] + forward[0] * longitudinal + side[0] * lateral),
            round(center[1] + forward[1] * longitudinal + side[1] * lateral),
        )

    if nitro:
        flame = [at(-half_length - 1, 0), at(-half_length - 10 * scale, -4 * scale), at(-half_length - 7 * scale, 0), at(-half_length - 10 * scale, 4 * scale)]
        pygame.draw.polygon(surface, (73, 219, 255), flame)
        pygame.draw.line(surface, (245, 252, 255), at(-half_length, 0), at(-half_length - 6 * scale, 0), max(1, round(2 * scale)))

    wheel_radius = max(2, round(3.2 * scale))
    for longitudinal in (-half_length * .58, half_length * .58):
        for lateral in (-half_width * 1.05, half_width * 1.05):
            pygame.draw.circle(surface, (14, 17, 19), at(longitudinal, lateral), wheel_radius)

    chassis = [
        at(half_length, 0), at(half_length * .72, -half_width * .88),
        at(-half_length * .72, -half_width), at(-half_length, -half_width * .68),
        at(-half_length, half_width * .68), at(-half_length * .72, half_width),
        at(half_length * .72, half_width * .88),
    ]
    pygame.draw.polygon(surface, (15, 20, 23), chassis)
    body = [at(value * half_length, side_value * half_width) for value, side_value in (
        (.88, 0), (.62, -.75), (-.66, -.86), (-.88, -.55),
        (-.88, .55), (-.66, .86), (.62, .75),
    )]
    pygame.draw.polygon(surface, color, body)
    pygame.draw.polygon(surface, accent, body, max(1, round(1.6 * scale)))

    cabin = [
        at(half_length * .35, -half_width * .52), at(half_length * .48, 0),
        at(half_length * .35, half_width * .52), at(-half_length * .35, half_width * .56),
        at(-half_length * .5, 0), at(-half_length * .35, -half_width * .56),
    ]
    pygame.draw.polygon(surface, (28, 52, 62), cabin)
    pygame.draw.line(surface, (139, 186, 199), at(half_length * .37, -half_width * .48), at(half_length * .37, half_width * .48), max(1, round(scale)))
    pygame.draw.line(surface, accent, at(-half_length * .7, -half_width * .62), at(-half_length * .7, half_width * .62), max(1, round(2 * scale)))
    for lateral in (-half_width * .48, half_width * .48):
        pygame.draw.circle(surface, (255, 244, 184), at(half_length * .75, lateral), max(1, round(1.5 * scale)))


def _track_heading(points, position) -> float:
    nearest = min(range(len(points)), key=lambda index: math.hypot(points[index].x - position.x, points[index].y - position.y))
    before, after = points[(nearest - 1) % len(points)], points[(nearest + 1) % len(points)]
    return math.degrees(math.atan2(after.y - before.y, after.x - before.x)) % 360

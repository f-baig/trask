"""Native Pygame replay viewer for planar and elevated racing replays.

One viewer, two renderers, chosen by the bundle rather than by a flag. A planar bundle is
drawn top-down from scene geometry alone; an elevated one cannot be, because a camera in a
perspective view is a function of world state, not of a 2D coordinate. So a 3D bundle is
replayed by standing the world back up at each tick with `restore` and asking `view3d` for
the frame — the same renderer the 3D player uses, so a replay and live play look identical.

`Racing3DWorld.restore` recomputes vertical pose from planar position, which is what makes
that sound: a replayed car sits exactly on the road it recorded, not at a stored height that
could disagree with the surface.
"""

from __future__ import annotations

import argparse

from .collision import EDGE_BARRIER_THICKNESS, collider_for, outline, track_edge_points
from .rendering import RendererDescriptor, ReplayBundle, snapshot_from_frame


DESCRIPTOR = RendererDescriptor(
    id="native-racing-2d",
    display_name="Native racing replay viewer",
    transport="replay-bundle/v2",
    supports_live_mode=False,
)

CAMERA_ORDER = ("third-person", "first-person", "hood", "third-person-far", "overhead-3d")
CAMERA_LABELS = {
    "first-person": "COCKPIT", "hood": "BUMPER", "third-person": "CHASE",
    "third-person-far": "CHASE FAR", "overhead-3d": "OVERHEAD",
}


class NativeReplayViewer:
    """Small desktop viewer; it renders replay bundles, never web state."""

    descriptor = DESCRIPTOR

    def replay(self, bundle: ReplayBundle) -> None:
        import pygame

        pygame.init()
        window = pygame.display.set_mode((1280, 820))
        pygame.display.set_caption(f"RaceLab — {bundle.metadata.policy_name} replay")
        clock = pygame.time.Clock()
        fonts = {
            "title": pygame.font.SysFont("Avenir Next", 24),
            "body": pygame.font.SysFont("Avenir Next", 15),
            "mono": pygame.font.SysFont("Menlo", 12),
            "small": pygame.font.SysFont("Menlo", 10),
        }
        step = 0
        playing = bool(bundle.frames)
        elapsed = 0.0
        show_grid = True
        running = True
        world = self._elevated_world(bundle)
        camera = 0
        while running:
            delta = clock.tick(60) / 1000
            elapsed += delta
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_p):
                        playing = not playing
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        step = min(len(bundle.frames) - 1, step + 1)
                        playing = False
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        step = max(0, step - 1)
                        playing = False
                    elif event.key == pygame.K_HOME:
                        step, playing = 0, False
                    elif event.key == pygame.K_END:
                        step, playing = max(0, len(bundle.frames) - 1), False
                    elif event.key == pygame.K_g:
                        show_grid = not show_grid
                    elif event.key == pygame.K_c and world is not None:
                        camera = (camera + 1) % len(CAMERA_ORDER)
                    elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
            if playing and bundle.frames and elapsed > 0.065:
                step = min(len(bundle.frames) - 1, step + 1)
                elapsed = 0
                if step == len(bundle.frames) - 1:
                    playing = False
            if world is None:
                self._draw(pygame, window, fonts, bundle, step, playing, show_grid)
            else:
                self._draw_3d(pygame, window, fonts, bundle, step, playing, world, camera)
            pygame.display.flip()
        pygame.quit()

    @staticmethod
    def _elevated_world(bundle: ReplayBundle):
        """Stand up a 3D world for an elevated bundle, or return None for a planar one."""
        scene = bundle.scene
        if scene is None or scene.elevation is None or scene.elevation.is_flat:
            return None
        from .racing3d import Racing3DWorld

        return Racing3DWorld.from_scene(scene)

    def _draw_3d(self, pygame, surface, fonts, bundle: ReplayBundle, step: int, playing: bool, world, camera: int) -> None:
        """Restore the recorded tick and render it through the 3D cameras."""
        from .view3d import DEFAULT_ROAD_DETAIL, ViewMode, render_view_surface

        ink, carbon, rule, paper, silver, blush = (
            (13, 13, 15), (21, 21, 24), (49, 49, 56), (244, 242, 239), (167, 167, 175), (241, 215, 222)
        )
        surface.fill(ink)
        pygame.draw.line(surface, rule, (0, 52), (1280, 52), 1)
        scene = bundle.scene
        assert scene is not None and scene.elevation is not None
        self._text(surface, fonts["small"], "RACELAB  /  NATIVE REPLAY  ·  PERSPECTIVE 3D", silver, (25, 20))
        self._text(surface, fonts["small"], bundle.backend.id.upper(), blush, (1100, 20))
        self._text(surface, fonts["title"], scene.name, paper, (25, 77))
        self._text(
            surface, fonts["mono"],
            f"{bundle.metadata.policy_name.upper()}  ·  {bundle.metadata.status.upper()}  ·  SEED {bundle.metadata.seed}",
            silver, (27, 110),
        )

        frame = bundle.frames[min(step, max(0, len(bundle.frames) - 1))] if bundle.frames else None
        viewport = pygame.Rect(26, 148, 920, 613)
        if frame is not None:
            world.restore(snapshot_from_frame(frame))
        mode = ViewMode(CAMERA_ORDER[camera])
        view = render_view_surface(world, mode, viewport.width, viewport.height, DEFAULT_ROAD_DETAIL)
        surface.blit(view, viewport.topleft)
        pygame.draw.rect(surface, rule, viewport, 1)
        self._text(surface, fonts["small"], CAMERA_LABELS[mode.value], paper, (viewport.x + 14, viewport.y + 12))

        if frame is not None and frame.privileged_state.countdown_ticks_remaining > 0:
            import math

            countdown = math.ceil(frame.privileged_state.countdown_ticks_remaining / 10)
            overlay = pygame.Surface((190, 72), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (10, 14, 16, 225), overlay.get_rect(), border_radius=12)
            label = fonts["title"].render(str(countdown), True, paper)
            overlay.blit(label, label.get_rect(center=(95, 36)))
            surface.blit(overlay, (viewport.centerx - 95, viewport.y + 28))

        panel = pygame.Rect(982, 148, 272, 613)
        pygame.draw.rect(surface, carbon, panel)
        pygame.draw.rect(surface, rule, panel, 1)
        self._text(surface, fonts["small"], "REPLAY INSPECTOR", silver, (1000, 169))
        self._rule(pygame, surface, (1000, 195), 234, rule)
        self._text(surface, fonts["small"], "PLAYING" if playing else "PAUSED", blush if playing else silver, (1000, 214))
        self._text(surface, fonts["title"], f"{step + 1:03d} / {max(1, len(bundle.frames)):03d}", paper, (1000, 236))
        if frame is not None:
            self._text(surface, fonts["small"], f"LAP {min(scene.laps, frame.privileged_state.lap + 1)}/{scene.laps}", silver, (1002, 268))
        self._rule(pygame, surface, (1000, 302), 234, rule)
        self._label_value(surface, fonts, "ACTION", frame.action if frame else "idle", (1000, 324), paper, silver)
        # Grade and bank are the whole reason this bundle needs a 3D renderer, so the
        # inspector reports them rather than leaving them implicit in the picture.
        attitude = "flat"
        if frame is not None:
            grade, bank = world.road_attitude(frame.privileged_state.player)
            import math

            attitude = f"{math.degrees(grade):+.1f}° GRADE · {math.degrees(bank):+.1f}° BANK"
        self._label_value(surface, fonts, "ROAD ATTITUDE", attitude, (1000, 372), paper, silver)
        speed_value = "0.0 KM/H"
        if frame is not None:
            drive = "NITRO" if frame.privileged_state.nitro_active else "READY" if frame.privileged_state.nitro_ready else "CHARGING"
            speed_value = f"{frame.privileged_state.longitudinal_velocity_mps * 3.6:.0f} KM/H · {drive}"
        self._label_value(surface, fonts, "SPEED / DRIVE MODE", speed_value, (1000, 420), paper, silver)
        self._text(surface, fonts["small"], "CURRENT INTENT", silver, (1000, 478))
        self._wrapped_text(
            surface, fonts["body"],
            frame.decision.subgoal if frame and frame.decision else "No decision telemetry",
            paper, pygame.Rect(1000, 498, 230, 52),
        )
        self._rule(pygame, surface, (1000, 566), 234, rule)
        self._text(surface, fonts["small"], "ELEVATION", silver, (1000, 588))
        self._wrapped_text(
            surface, fonts["body"],
            f"{scene.elevation.profile.value} · {scene.elevation.amplitude_m:.1f} m over "
            f"{scene.elevation.hill_count} crest(s) · banking to {scene.elevation.banking_degrees:.0f}°",
            silver, pygame.Rect(1000, 608, 230, 70),
        )
        self._rule(pygame, surface, (1000, 668), 234, rule)
        self._text(surface, fonts["small"], "SPACE PLAY · ← → SCRUB · C CAMERA", silver, (1000, 694))
        self._text(surface, fonts["small"], "ESC CLOSE", silver, (1000, 714))

        timeline = pygame.Rect(26, 785, 920, 4)
        pygame.draw.rect(surface, rule, timeline)
        fill = int(timeline.width * (step / max(1, len(bundle.frames) - 1)))
        pygame.draw.rect(surface, blush, pygame.Rect(timeline.x, timeline.y, fill, timeline.height))
        pygame.draw.circle(surface, paper, (timeline.x + fill, timeline.centery), 5)
        self._text(surface, fonts["small"], "DESKTOP CHANNEL · REPLAY-BUNDLE/V2 · RACING-3D", silver, (26, 798))

    def _draw(self, pygame, surface, fonts, bundle: ReplayBundle, step: int, playing: bool, show_grid: bool) -> None:
        if bundle.scene is None:
            raise ValueError("The 2D viewer requires a top-down scene; select the renderer from the replay manifest instead.")
        ink, carbon, rule, paper, silver, blush, rose = (
            (13, 13, 15), (21, 21, 24), (49, 49, 56), (244, 242, 239), (167, 167, 175), (241, 215, 222), (219, 164, 179)
        )
        surface.fill(ink)
        pygame.draw.line(surface, rule, (0, 52), (1280, 52), 1)
        self._text(surface, fonts["small"], "RACELAB  /  NATIVE REPLAY", silver, (25, 20))
        self._text(surface, fonts["small"], "RACING-2D-V4", blush, (1100, 20))
        self._text(surface, fonts["title"], bundle.scene.name, paper, (25, 77))
        self._text(surface, fonts["mono"], f"{bundle.metadata.policy_name.upper()}  ·  {bundle.metadata.status.upper()}  ·  SEED {bundle.metadata.seed}", silver, (27, 110))

        viewport = pygame.Rect(26, 148, 920, 613)
        surface_palette = {
            "asphalt": ((63, 103, 70), (32, 39, 43), (52, 58, 64)),
            "clay": ((101, 115, 67), (96, 67, 47), (155, 101, 68)),
            "ice": ((184, 211, 216), (111, 147, 156), (169, 197, 204)),
        }
        ground_color, edge_color, road_color = surface_palette[bundle.scene.surface]
        pygame.draw.rect(surface, ground_color, viewport)
        pygame.draw.rect(surface, rule, viewport, 1)
        scale_x = viewport.width / bundle.scene.bounds.width
        scale_y = viewport.height / bundle.scene.bounds.height
        racing_points = [(viewport.x + int(point.x * scale_x), viewport.y + int(point.y * scale_y)) for point in bundle.scene.track_centerline]
        track_width = max(4, int(bundle.scene.track_width * (scale_x + scale_y) / 2))
        if racing_points:
            pygame.draw.lines(surface, (18, 24, 27), True, racing_points, track_width + 24)
            pygame.draw.lines(surface, edge_color, True, racing_points, track_width + 16)
            pygame.draw.lines(surface, (239, 235, 219), True, racing_points, track_width + 10)
            pygame.draw.lines(surface, road_color, True, racing_points, track_width)
            pygame.draw.aalines(surface, (205, 213, 210), True, racing_points)
        if bundle.scene.edge_barriers:
            barrier_offset = bundle.scene.track_width / 2 + EDGE_BARRIER_THICKNESS / 2
            left_edge, right_edge = track_edge_points(bundle.scene, barrier_offset)
            thickness = max(4, round(EDGE_BARRIER_THICKNESS * (scale_x + scale_y) / 2))
            for edge_points in (left_edge, right_edge):
                screen_edge = [
                    (viewport.x + round(x * scale_x), viewport.y + round(y * scale_y))
                    for x, y in edge_points
                ]
                pygame.draw.lines(surface, (17, 21, 24), True, screen_edge, thickness + 4)
                pygame.draw.lines(surface, (237, 79, 55), True, screen_edge, thickness)
                stride = max(1, len(screen_edge) // 30)
                for post in screen_edge[::stride]:
                    pygame.draw.circle(surface, (246, 239, 222), post, max(1, thickness // 3))

        frame = bundle.frames[min(step, max(0, len(bundle.frames) - 1))] if bundle.frames else None
        entities = frame.privileged_state.entities if frame else [
            {"id": entity.id, "kind": entity.kind, "x": entity.rect.x, "y": entity.rect.y, "width": entity.rect.width, "height": entity.rect.height, "active": True, "open": False}
            for entity in bundle.scene.entities
        ]
        palette = {"checkpoint": (244, 211, 94), "obstacle": (255, 107, 44), "npc": (112, 160, 183)}
        scene_entities = {entity.id: entity for entity in bundle.scene.entities}
        for entity in entities:
            if not entity["active"]:
                continue
            rect = pygame.Rect(
                viewport.x + int(entity["x"] * scale_x), viewport.y + int(entity["y"] * scale_y),
                int(entity["width"] * scale_x), int(entity["height"] * scale_y),
            )
            spec = scene_entities.get(entity["id"])
            color = self._hex_color(spec.color) if spec and spec.color else palette.get(entity["kind"], silver)
            if entity["kind"] == "npc":
                center = rect.center
                heading = float(entity.get("heading", self._nearest_track_heading(center, racing_points)))
                self._draw_race_car(
                    pygame, surface, center, heading, color, (237, 239, 230), "2", False,
                    nitro=bool(entity.get("nitro_active", False)),
                )
                continue
            if entity["kind"] == "obstacle":
                if spec is not None:
                    collider = collider_for(spec, float(entity["y"]) - spec.rect.y)
                    hull = [
                        (viewport.x + round(x * scale_x), viewport.y + round(y * scale_y))
                        for x, y in outline(collider)
                    ]
                    pygame.draw.polygon(surface, color, hull)
                    pygame.draw.polygon(surface, (22, 27, 29), hull, 2)
                continue
            alpha = 80 if entity.get("open") else 255
            tile = pygame.Surface(rect.size, pygame.SRCALPHA)
            tile.fill((*color, alpha))
            surface.blit(tile, rect)
            if entity["kind"] == "checkpoint":
                pygame.draw.line(surface, paper, (rect.left, rect.centery), (rect.right, rect.centery), 3)

        player = frame.privileged_state.player if frame else bundle.scene.player_spawn
        point = (viewport.x + int(player.x * scale_x), viewport.y + int(player.y * scale_y))
        heading = frame.privileged_state.heading if frame else 0
        self._draw_race_car(
            pygame, surface, point, heading, paper, (255, 90, 54), "1", True,
            nitro=bool(frame and frame.privileged_state.nitro_active),
        )
        if frame and frame.privileged_state.barrier_impact:
            impact = frame.privileged_state.barrier_impact
            centre = (
                viewport.x + round(impact.x * scale_x),
                viewport.y + round(impact.y * scale_y),
            )
            pygame.draw.circle(surface, (255, 216, 84), centre, 16, 3)
            for dx, dy in ((24, 0), (-24, 0), (0, 24), (0, -24)):
                pygame.draw.line(
                    surface, (255, 241, 168),
                    (centre[0] + dx // 2, centre[1] + dy // 2),
                    (centre[0] + dx, centre[1] + dy), 3,
                )
        if frame and frame.privileged_state.countdown_ticks_remaining > 0:
            import math
            countdown = math.ceil(frame.privileged_state.countdown_ticks_remaining / 10)
            overlay = pygame.Surface((190, 72), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (10, 14, 16, 225), overlay.get_rect(), border_radius=12)
            label = fonts["title"].render(str(countdown), True, (244, 242, 239))
            overlay.blit(label, label.get_rect(center=(95, 36)))
            surface.blit(overlay, (viewport.centerx - 95, viewport.y + 28))

        panel = pygame.Rect(982, 148, 272, 613)
        pygame.draw.rect(surface, carbon, panel)
        pygame.draw.rect(surface, rule, panel, 1)
        self._text(surface, fonts["small"], "REPLAY INSPECTOR", silver, (1000, 169))
        self._rule(pygame, surface, (1000, 195), 234, rule)
        state = "PLAYING" if playing else "PAUSED"
        self._text(surface, fonts["small"], state, blush if playing else silver, (1000, 214))
        self._text(surface, fonts["title"], f"{step + 1:03d} / {max(1, len(bundle.frames)):03d}", paper, (1000, 236))
        lap_label = f"LAP {min(bundle.scene.laps, frame.privileged_state.lap + 1)}/{bundle.scene.laps}" if frame else "FRAME"
        self._text(surface, fonts["small"], lap_label, silver, (1002, 268))
        self._rule(pygame, surface, (1000, 302), 234, rule)
        self._label_value(surface, fonts, "ACTION", frame.action if frame else "idle", (1000, 324), paper, silver)
        speed_value = "0.0 PX/T"
        if frame:
            mode = "NITRO" if frame.privileged_state.nitro_active else "READY" if frame.privileged_state.nitro_ready else "CHARGING"
            speed_value = (
                f"{frame.privileged_state.longitudinal_velocity_mps * 3.6:.0f} KM/H · {mode} · "
                f"{abs(frame.privileged_state.slip_angle_degrees):.1f}° SLIP"
            )
        self._label_value(surface, fonts, "SPEED / DRIVE MODE", speed_value, (1000, 372), paper, silver)
        intent = frame.decision.subgoal if frame and frame.decision else "No decision telemetry"
        self._text(surface, fonts["small"], "CURRENT INTENT", silver, (1000, 430))
        self._wrapped_text(surface, fonts["body"], intent, paper, pygame.Rect(1000, 450, 230, 52))
        self._rule(pygame, surface, (1000, 526), 234, rule)
        observation = frame.observation.task if frame else bundle.scene.prompt
        self._text(surface, fonts["small"], "PLAYER-VISIBLE TASK", silver, (1000, 548))
        self._wrapped_text(surface, fonts["body"], observation, silver, pygame.Rect(1000, 568, 230, 70))
        self._rule(pygame, surface, (1000, 668), 234, rule)
        self._text(surface, fonts["small"], "SPACE PLAY  ·  ← → SCRUB  ·  G GRID", silver, (1000, 694))
        self._text(surface, fonts["small"], "ESC CLOSE", silver, (1000, 714))

        timeline = pygame.Rect(26, 785, 920, 4)
        pygame.draw.rect(surface, rule, timeline)
        fill = int(timeline.width * (step / max(1, len(bundle.frames) - 1)))
        pygame.draw.rect(surface, blush, pygame.Rect(timeline.x, timeline.y, fill, timeline.height))
        pygame.draw.circle(surface, paper, (timeline.x + fill, timeline.centery), 5)
        self._text(surface, fonts["small"], "DESKTOP CHANNEL · REPLAY-BUNDLE/V2 · RACING", silver, (26, 798))

    @staticmethod
    def _text(surface, font, value: str, color, position: tuple[int, int]) -> None:
        surface.blit(font.render(value, True, color), position)

    @staticmethod
    def _hex_color(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        if len(value) != 6:
            return (167, 167, 175)
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))

    @staticmethod
    def _rule(pygame, surface, position: tuple[int, int], length: int, color) -> None:
        pygame.draw.line(surface, color, position, (position[0] + length, position[1]), 1)

    @staticmethod
    def _nearest_track_heading(point: tuple[int, int], track: list[tuple[int, int]]) -> float:
        import math
        nearest = min(range(len(track)), key=lambda index: (track[index][0] - point[0]) ** 2 + (track[index][1] - point[1]) ** 2)
        before, after = track[(nearest - 1) % len(track)], track[(nearest + 1) % len(track)]
        return math.degrees(math.atan2(after[1] - before[1], after[0] - before[0]))

    @staticmethod
    def _draw_race_car(pygame, surface, point, heading: float, body, accent, number: str, player: bool, nitro: bool = False) -> None:
        import math

        radians = math.radians(heading)
        forward = (math.cos(radians), math.sin(radians))
        side = (-forward[1], forward[0])

        def transformed(longitudinal: float, lateral: float) -> tuple[int, int]:
            return (
                int(point[0] + forward[0] * longitudinal + side[0] * lateral),
                int(point[1] + forward[1] * longitudinal + side[1] * lateral),
            )

        if player:
            pygame.draw.circle(surface, (255, 222, 105), point, 24, 1)
        if nitro:
            pygame.draw.polygon(surface, (56, 215, 255), [
                transformed(-20, -4), transformed(-39, 0), transformed(-20, 4),
            ])
            pygame.draw.line(surface, (245, 253, 255), transformed(-21, 0), transformed(-33, 0), 3)
        for longitudinal in (-13, 11):
            for lateral in (-10, 10):
                pygame.draw.circle(surface, (13, 17, 19), transformed(longitudinal, lateral), 4)
        chassis = [transformed(21, 0), transformed(12, -9), transformed(-18, -8), transformed(-21, 0), transformed(-18, 8), transformed(12, 9)]
        pygame.draw.polygon(surface, (15, 20, 22), chassis)
        inner = [transformed(18, 0), transformed(10, -7), transformed(-17, -6), transformed(-19, 0), transformed(-17, 6), transformed(10, 7)]
        pygame.draw.polygon(surface, body, inner)
        cockpit = [transformed(11, -5), transformed(-5, -5), transformed(-5, 5), transformed(11, 5)]
        pygame.draw.polygon(surface, (28, 49, 58), cockpit)
        pygame.draw.line(surface, accent, transformed(19, -7), transformed(19, 7), 3)
        pygame.draw.circle(surface, (245, 240, 222), transformed(-10, 0), 5)
        label = pygame.font.SysFont("Menlo", 8, bold=True).render(number, True, (20, 26, 29))
        surface.blit(label, label.get_rect(center=transformed(-10, 0)))
        if player:
            pygame.draw.polygon(surface, (255, 222, 105), [transformed(29, 0), transformed(35, -4), transformed(35, 4)])

    def _label_value(self, surface, fonts, label: str, value: str, position: tuple[int, int], paper, silver) -> None:
        self._text(surface, fonts["small"], label, silver, position)
        self._text(surface, fonts["body"], value, paper, (position[0], position[1] + 16))

    @staticmethod
    def _wrapped_text(surface, font, value: str, color, rect) -> None:
        words, line, y = value.split(), "", rect.y
        for word in words:
            trial = f"{line} {word}".strip()
            if font.size(trial)[0] > rect.width and line:
                surface.blit(font.render(line, True, color), (rect.x, y))
                y += font.get_linesize()
                line = word
            else:
                line = trial
        if line:
            surface.blit(font.render(line, True, color), (rect.x, y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a native RaceLab replay")
    parser.add_argument("--bundle", required=True, help="ReplayBundle JSON exported by the harness")
    args = parser.parse_args()
    bundle = ReplayBundle.model_validate_json(open(args.bundle, encoding="utf-8").read())
    NativeReplayViewer().replay(bundle)


if __name__ == "__main__":
    main()

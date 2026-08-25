"""The 3D view component: cameras, a software rasterizer, and policy frames.

This is the boundary a future visuomotor policy will see. Two properties make it
usable for that rather than only for a human window:

- **Cameras are pure functions of simulator state.** `camera_for(world, mode)`
  reads the world and returns a pose. There is no smoothing, no velocity spring,
  and no frame-to-frame memory, so the same tick renders the same image and a
  replayed rollout is reproducible frame for frame.
- **Every frame declares how it was made.** `render_policy_view` returns a
  `VisualFrame` carrying a `CameraContract`: eye height, pitch, field of view,
  chase distance, near plane, and the world scale in pixels per meter. Without
  that a policy can see a shape but cannot convert it into a distance.

Rendering is a small painter's-algorithm rasterizer over `pygame.draw.polygon`.
It is deliberately not a GPU pipeline: it has no external dependency beyond the
pygame already used for 2D, it runs headless for batch frame generation, and it
is deterministic. Polygons are clipped against the near plane properly, which is
what keeps an in-car camera from smearing geometry across the screen when the
road passes behind the eye.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from enum import StrEnum

from .collision import EDGE_BARRIER_THICKNESS
from .models import EntityKind, Vec2
from .policy_protocol import CameraContract, VisualFrame
from .racing3d import CarPose3D, Racing3DWorld
from .vision import SURFACE_PALETTES
from .visual import rgb, shade as _tint


NEAR_PLANE = 5.0
DEFAULT_VERTICAL_FOV = 58.0
# Beyond this the road is a few pixels tall and costs more to transform than it
# adds, so segments are culled in world space before any projection happens.
DRAW_DISTANCE = 1_500.0
SKIRT_WIDTH_MULTIPLE = 1.15
SKIRT_DROP = 34.0
LIGHT_DIRECTION = (-.38, -.52, .76)
AMBIENT_LIGHT = .58
DIFFUSE_LIGHT = .52
SCREEN_LIMIT = 30_000

SKY_TOP = (58, 104, 168)
SKY_HORIZON = (176, 206, 232)
CURB_LIGHT = (232, 236, 238)
CURB_DARK = (196, 66, 52)
GATE_PASSED = (112, 120, 122)
GATE_SECTOR = (244, 211, 94)
GATE_FINISH = (255, 107, 44)
BARRIER_BODY = (237, 79, 55)
BARRIER_TOP = (28, 32, 34)
PLAYER_BODY = (247, 242, 230)
PLAYER_ACCENT = (255, 90, 54)
OPPONENT_BODY = (52, 166, 230)
CABIN_GLASS = (28, 52, 62)
TIRE = (18, 20, 23)
NITRO_FLAME = (73, 219, 255)
IMPACT_SPARK = (255, 216, 84)


class ViewMode(StrEnum):
    FIRST_PERSON = "first-person"
    HOOD = "hood"
    THIRD_PERSON = "third-person"
    THIRD_PERSON_FAR = "third-person-far"
    OVERHEAD_3D = "overhead-3d"


# Eye height, chase distance, look-ahead, and pitch for each mode. Heights and
# distances are in pixels; the scene's pixels_per_meter makes them physical.
_MODE_RIG: dict[ViewMode, tuple[float, float, float, float]] = {
    #                        height, behind, look-ahead, pitch
    ViewMode.FIRST_PERSON: (17.0, -2.0, 300.0, 0.0),
    ViewMode.HOOD: (6.0, 12.0, 300.0, 0.0),
    ViewMode.THIRD_PERSON: (34.0, -105.0, 210.0, -6.0),
    ViewMode.THIRD_PERSON_FAR: (72.0, -186.0, 250.0, -13.0),
    ViewMode.OVERHEAD_3D: (430.0, -30.0, 0.0, -88.0),
}
# Modes whose eye is inside the car, so the ego bodywork is not drawn.
_IN_CAR_MODES = frozenset({ViewMode.FIRST_PERSON, ViewMode.HOOD})
# An in-car camera that inherits body roll whole tilts the horizon enough to be
# disorienting. Damping keeps the sensation of leaning without losing the road.
IN_CAR_ROLL_DAMPING = .4
# A car stands ON the road, so its faces and the road quad beneath it are the same
# distance from the camera and their centroid sort order is arbitrary. Biasing cars
# slightly nearer resolves that tie in the direction physics guarantees, which is
# what stops a car flickering through the surface on elevation changes. It is small
# enough that genuinely nearer scenery still occludes the car correctly.
CAR_DEPTH_BIAS = 16.0
# Render-only subdivision of each road segment. Physics is unaffected; higher
# values just evaluate the smooth elevation profile at more points, so crests look
# rounder at a linear cost in polygons.
DEFAULT_ROAD_DETAIL = 2


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is, what it looks at, and how wide it sees."""

    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    vertical_fov_degrees: float
    mode: ViewMode
    eye_height_pixels: float
    distance_behind_pixels: float

    def basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Orthonormal right/up/forward axes, safe when looking straight down.

        World axes are x east, y south, and z up, which is right-handed. Screen
        right is therefore `up x forward`, not `forward x up`: facing east with
        z up, the driver's right hand points south, and `forward x up` returns
        north. Getting that backwards mirrors the entire image, which reads as
        inverted steering rather than as a rendering fault.
        """
        forward = _normalize(_subtract(self.target, self.eye))
        world_up = (0.0, 0.0, 1.0)
        if abs(_dot(forward, world_up)) > .999:
            # Looking along the world up axis collapses the usual cross product,
            # so the frame is anchored to the direction the camera was aimed from.
            planar = _subtract(self.eye, self.target)
            reference = (
                _normalize((planar[0], planar[1], 0.0))
                if abs(planar[0]) + abs(planar[1]) > 1e-6 else (1.0, 0.0, 0.0)
            )
            right = _normalize(_cross(reference, forward))
        else:
            right = _normalize(_cross(world_up, forward))
        up = _normalize(_cross(forward, right))
        return right, up, forward

    @property
    def pitch_degrees(self) -> float:
        forward = _normalize(_subtract(self.target, self.eye))
        return math.degrees(math.asin(max(-1.0, min(1.0, forward[2]))))


def camera_for(
    world: Racing3DWorld, mode: ViewMode, vertical_fov_degrees: float = DEFAULT_VERTICAL_FOV,
) -> CameraPose:
    """Build a camera from world state alone, with no inter-frame memory."""
    pose = world.player_pose()
    height, behind, look_ahead, pitch = _MODE_RIG[mode]
    rig_pose = (
        CarPose3D(
            x=pose.x, y=pose.y, z=pose.z, heading_degrees=pose.heading_degrees,
            pitch_degrees=pose.pitch_degrees,
            roll_degrees=pose.roll_degrees * IN_CAR_ROLL_DAMPING,
        )
        if mode in _IN_CAR_MODES else pose
    )
    forward, left, up = rig_pose.basis()
    yaw = math.radians(pose.heading_degrees)
    forward_flat = (math.cos(yaw), math.sin(yaw), 0.0)
    world_up = (0.0, 0.0, 1.0)

    if mode in (ViewMode.FIRST_PERSON, ViewMode.HOOD):
        # In-car views ride the chassis, so pitch and roll are felt rather than
        # stabilized: cresting a hill genuinely drops the road out of frame.
        eye = _add(_add(pose.position, _scale(forward, behind)), _scale(up, height))
        target = _add(eye, _scale(_pitched(forward, up, pitch), look_ahead))
    elif mode == ViewMode.OVERHEAD_3D:
        eye = _add(_add(pose.position, _scale(forward_flat, behind)), _scale(world_up, height))
        target = (pose.x, pose.y, pose.z)
    else:
        # Chase cameras use the flat heading so body roll does not tilt the
        # horizon, which is disorienting and hides the road edge.
        eye = _add(_add(pose.position, _scale(forward_flat, behind)), _scale(world_up, height))
        target = _add(
            _add(pose.position, _scale(forward_flat, look_ahead)),
            _scale(world_up, height * .35),
        )
    del left
    return CameraPose(
        eye=eye, target=target, vertical_fov_degrees=vertical_fov_degrees, mode=mode,
        eye_height_pixels=eye[2] - pose.z,
        # An in-car eye sits a few pixels back for the driving position, which is
        # a seat offset rather than a chase distance; reporting it as one would
        # tell a policy this is an external camera.
        distance_behind_pixels=0.0 if mode in _IN_CAR_MODES else max(0.0, -behind),
    )


FREE_PITCH_RANGE = (3.0, 86.0)
"""Pitch limits for the free camera, in degrees below horizontal.

Not the full hemisphere: at zero the eye sits in the road surface and the picture is a
sliver of tarmac, and past 86 the orbit basis degenerates as forward approaches world up.
"""


def orbit_camera(
    world: Racing3DWorld, *, yaw_degrees: float, pitch_degrees: float,
    distance_pixels: float, focus: str = "circuit",
    vertical_fov_degrees: float = DEFAULT_VERTICAL_FOV,
) -> CameraPose:
    """A free-floating camera orbiting the circuit or the car.

    The preset rigs in `_MODE_RIG` answer "what can the driver see from here", which is the
    question a policy asks. Inspecting an environment is the opposite question — what shape is
    this circuit, where do the hills sit — and no car-relative camera answers it, because it
    can only ever look where the car is pointed.

    Orbit parameters rather than a free-flying position because two angles and a radius always
    frame the subject: there is no way to end up lost in the terrain with the circuit off
    screen, which is most of what makes a fly-through camera annoying to use.
    """
    surface = world.surface
    centerline = world.scene.track_centerline
    if focus == "car":
        pose = world.player_pose()
        centre = (pose.x, pose.y, pose.z)
    else:
        centre = (
            sum(point.x for point in centerline) / len(centerline),
            sum(point.y for point in centerline) / len(centerline),
            (
                sum(surface.height_at_index(index) for index in range(len(centerline)))
                / len(centerline)
            ) if surface is not None else 0.0,
        )
    pitch = max(FREE_PITCH_RANGE[0], min(FREE_PITCH_RANGE[1], pitch_degrees))
    distance = max(80.0, min(4_000.0, distance_pixels))
    yaw = math.radians(yaw_degrees)
    horizontal = math.cos(math.radians(pitch)) * distance
    eye = (
        centre[0] - math.cos(yaw) * horizontal,
        centre[1] - math.sin(yaw) * horizontal,
        centre[2] + math.sin(math.radians(pitch)) * distance,
    )
    return CameraPose(
        eye=eye, target=centre, vertical_fov_degrees=vertical_fov_degrees,
        # Reported as an overhead camera because that is the closest thing in the contract and
        # it is what decides that the ego car is drawn. A free camera is a viewer's tool; it is
        # deliberately not offered to a policy, so it needs no viewpoint of its own.
        mode=ViewMode.OVERHEAD_3D,
        eye_height_pixels=eye[2] - centre[2],
        distance_behind_pixels=distance,
    )


@dataclass(frozen=True)
class _Projector:
    """Precomputed world-to-screen transform for one frame."""

    eye: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    forward: tuple[float, float, float]
    focal: float
    centre_x: float
    centre_y: float

    def to_view(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        offset = (point[0] - self.eye[0], point[1] - self.eye[1], point[2] - self.eye[2])
        return (_dot(offset, self.right), _dot(offset, self.up), _dot(offset, self.forward))

    def project(self, view: tuple[float, float, float]) -> tuple[int, int]:
        depth = max(NEAR_PLANE * .5, view[2])
        return (
            _clamp_screen(self.centre_x + self.focal * view[0] / depth),
            _clamp_screen(self.centre_y - self.focal * view[1] / depth),
        )

    def horizon_y(self) -> float:
        """Screen row where the ground plane recedes to infinity."""
        horizontal = (self.forward[0], self.forward[1], 0.0)
        if abs(horizontal[0]) + abs(horizontal[1]) < 1e-9:
            return -1e6 if self.forward[2] < 0 else 1e6
        direction = _normalize(horizontal)
        depth = _dot(direction, self.forward)
        if depth <= 1e-6:
            return 1e6
        return self.centre_y - self.focal * _dot(direction, self.up) / depth


def render_view_surface(
    world: Racing3DWorld, mode: ViewMode, width: int, height: int,
    road_detail: int = DEFAULT_ROAD_DETAIL,
):
    """Rasterize one frame of authoritative world state to a pygame Surface.

    `road_detail` subdivides each road segment for rendering only. It changes no
    simulation state, so it is a pure quality-versus-cost dial.
    """
    return render_pose_surface(world, camera_for(world, mode), width, height, road_detail)


def render_pose_surface(
    world: Racing3DWorld, camera: CameraPose, width: int, height: int,
    road_detail: int = DEFAULT_ROAD_DETAIL, draw_distance: float | None = None,
):
    """Rasterize from an explicit camera, so a viewer can supply one of its own.

    Split out from `render_view_surface` because the preset rigs and the free camera differ
    only in how the pose is built; sharing the rasterizer means the free camera cannot drift
    into rendering the world differently from the way the driver sees it.
    """
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("3D rendering requires the pygame-ce native dependency") from error

    mode = camera.mode
    palette = _scene_palette(world.scene)
    right, up, forward = camera.basis()
    focal = (height / 2) / math.tan(math.radians(camera.vertical_fov_degrees) / 2)
    projector = _Projector(
        eye=camera.eye, right=right, up=up, forward=forward,
        focal=focal, centre_x=width / 2, centre_y=height / 2,
    )
    surface = pygame.Surface((width, height))
    ground, road, edge = palette["terrain"], palette["road"], palette["edge"]
    _draw_background(pygame, surface, projector, width, height, ground, palette["sky"])

    visible = _visible_segments(world, projector, draw_distance or DRAW_DISTANCE)
    faces = [
        *_terrain_faces(world, visible, ground),
        *_scenery_faces(world, palette),
        *_road_faces(world, visible, road, edge, road_detail, palette),
        *_edge_barrier_faces(world, visible, palette["barrier"]),
        *_gate_faces(world),
        *_barrier_faces(world, palette["barrier"]),
        *_barrier_impact_faces(world),
        *_car_faces(world, include_player=mode not in _IN_CAR_MODES, palette=palette),
    ]
    _draw_faces(pygame, surface, projector, faces, width, height)
    return surface


def ensure_headless_video() -> None:
    """Initialise SDL with no window, for rendering inside a server process.

    `render_view_surface` only ever draws into an off-screen `Surface`, but SDL still wants a
    video backend initialised before it will create one. A web request cannot open a window,
    so the driver is forced to `dummy` — and only when nothing has selected one already, so
    this never steals the display from `play3d` running in the same interpreter.
    """
    import os

    import pygame

    if pygame.display.get_init():
        return
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    if not pygame.font.get_init():
        pygame.font.init()


def encode_view_png(
    world: Racing3DWorld, mode: ViewMode, width: int, height: int,
    road_detail: int = DEFAULT_ROAD_DETAIL,
) -> bytes:
    """One perspective frame as PNG bytes, for a viewer that is not a pygame window.

    The browser gets the same renderer the desktop player uses rather than an approximation
    of it, because a camera here is a pure function of world state: the only thing a second
    implementation could add is a second set of bugs.
    """
    import io

    import pygame

    ensure_headless_video()
    surface = render_view_surface(world, mode, width, height, road_detail)
    buffer = io.BytesIO()
    pygame.image.save(surface, buffer, "view3d.png")
    return buffer.getvalue()


def render_policy_view(
    world: Racing3DWorld, mode: ViewMode = ViewMode.THIRD_PERSON,
    width: int = 480, height: int = 320, road_detail: int = DEFAULT_ROAD_DETAIL,
) -> VisualFrame:
    """Render a policy-facing frame with no reward, oracle, or evaluator state."""
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("3D rendering requires the pygame-ce native dependency") from error

    surface = render_view_surface(world, mode, width, height, road_detail)
    camera = camera_for(world, mode)
    vertical_fov = camera.vertical_fov_degrees
    horizontal_fov = math.degrees(
        2 * math.atan(math.tan(math.radians(vertical_fov) / 2) * width / height)
    )
    buffer = io.BytesIO()
    pygame.image.save(surface, buffer, "policy-view-3d.png")
    return VisualFrame(
        media_type="image/png", data_base64=base64.b64encode(buffer.getvalue()).decode(),
        width=width, height=height, channels=3,
        viewpoint=mode.value, orientation="camera-up", ego_anchor="camera-relative",
        horizontal_fov_degrees=min(179.0, horizontal_fov),
        camera=CameraContract(
            mode=mode.value,
            vertical_fov_degrees=vertical_fov,
            horizontal_fov_degrees=min(179.0, horizontal_fov),
            eye_height_pixels=round(camera.eye_height_pixels, 2),
            pitch_degrees=round(camera.pitch_degrees, 2),
            distance_behind_pixels=round(camera.distance_behind_pixels, 2),
            follows_ego_heading=True,
            near_plane_pixels=NEAR_PLANE,
            pixels_per_meter=world.dynamics.pixels_per_meter,
        ),
    )


def _scene_palette(scene) -> dict:
    """Resolve the scene's own colours, falling back to the surface's sensor palette.

    A scene with no visual plan renders byte-identically to how it did before visual
    plans existed, which is what lets this be additive rather than a redesign.
    """
    terrain, road, edge = SURFACE_PALETTES[scene.surface]
    visual = getattr(scene, "visual", None)
    if visual is None:
        return {
            "terrain": terrain, "road": road, "edge": edge, "sky": SKY_TOP,
            "barrier": BARRIER_BODY, "player": PLAYER_BODY, "opponent": OPPONENT_BODY,
            "kerbs": True, "kerb_light": CURB_LIGHT, "kerb_dark": CURB_DARK, "scenery": [],
        }
    resolved = visual.resolved(scene.surface)
    return {
        "terrain": rgb(resolved["terrain"]) if visual.terrain else terrain,
        "road": rgb(resolved["road"]) if visual.road else road,
        # The lane edge is derived from the road rather than named separately: one
        # colour per material keeps a recoloured circuit looking like a material.
        "edge": _tint(rgb(resolved["road"]), 2.4) if visual.road else edge,
        "sky": rgb(resolved["sky"]) if visual.sky else SKY_TOP,
        "barrier": rgb(resolved["barrier"]),
        "player": rgb(resolved["player_car"]),
        "opponent": rgb(resolved["opponent_car"]),
        "kerbs": resolved["kerbs"],
        "kerb_light": rgb(resolved["kerb_light"]),
        "kerb_dark": rgb(resolved["kerb_dark"]),
        "scenery": resolved["scenery"],
    }


def _scenery_faces(world: Racing3DWorld, palette: dict):
    """Flat coloured ground bands — a river, a sand trap — laid on the terrain.

    Drawn slightly above the terrain plane and biased nearer so the painter's sort puts
    them over the ground but still under the road, which is what "under the track" means
    when the road is a surface rather than a volume.
    """
    bounds = world.scene.bounds
    if not palette["scenery"]:
        return
    span = len(world.scene.track_centerline)
    ground_height = (
        sum(world.surface.height_at_index(index) for index in range(span)) / max(1, span)
    ) - 1.0
    for band in palette["scenery"]:
        centre = _region_centre(band["region"], bounds)
        half = band["width_pixels"] / 2
        horizontal = band["orientation"] != "vertical"
        if horizontal:
            corners = [
                (bounds.x, centre[1] - half), (bounds.x + bounds.width, centre[1] - half),
                (bounds.x + bounds.width, centre[1] + half), (bounds.x, centre[1] + half),
            ]
        else:
            corners = [
                (centre[0] - half, bounds.y), (centre[0] + half, bounds.y),
                (centre[0] + half, bounds.y + bounds.height), (centre[0] - half, bounds.y + bounds.height),
            ]
        # The 3D ground is a skirt either side of the road plus a flat background fill,
        # so there is no height to sample away from the circuit. The band is laid at the
        # circuit's mean height, which is exact on a flat scene and close enough on a
        # rolling one for what is a piece of scenery.
        # Negative bias pushes the band away in the painter's sort. A band spans the whole
        # map, so its centroid can sit nearer than the road segment crossing it and paint
        # over the circuit — the opposite of the "under the track" it is meant to be.
        yield ([(x, y, ground_height) for x, y in corners], rgb(band["color"]), False, -400.0)


def _region_centre(region: str, bounds) -> tuple[float, float]:
    """The nine-cell grid the corner grammar addresses, in scene coordinates."""
    columns = {"left": .5, "right": 2.5, "center": 1.5}
    rows = {"top": .5, "bottom": 2.5, "center": 1.5}
    parts = region.split("-")
    if len(parts) == 2:
        column, row = columns.get(parts[1], 1.5), rows.get(parts[0], 1.5)
    else:
        column, row = columns.get(parts[0], 1.5), rows.get(parts[0], 1.5)
    return bounds.x + bounds.width / 3 * column, bounds.y + bounds.height / 3 * row


def _draw_background(pygame, surface, projector: _Projector, width: int, height: int, ground, sky_top=None) -> None:
    """Sky gradient above the horizon, hazed terrain below it."""
    top = sky_top or SKY_TOP
    # Derived rather than named: a recoloured sky still fades toward the skyline, so a
    # night palette does not turn the upper half into one flat rectangle.
    sky_low = SKY_HORIZON if sky_top in (None, SKY_TOP) else _tint(top, 1.9)
    horizon = projector.horizon_y()
    surface.fill(ground)
    sky_bottom = max(0, min(height, int(round(horizon))))
    for row in range(sky_bottom):
        blend = row / max(1, sky_bottom)
        pygame.draw.line(surface, _mix(top, sky_low, blend), (0, row), (width, row))
    # Fade the far ground into the horizon so culled distant geometry does not
    # end in a hard edge.
    haze_depth = min(height - sky_bottom, max(1, height // 6))
    for row in range(haze_depth):
        blend = 1 - row / haze_depth
        pygame.draw.line(
            surface, _mix(ground, SKY_HORIZON, blend * .75),
            (0, sky_bottom + row), (width, sky_bottom + row),
        )


def _draw_faces(
    pygame, surface, projector: _Projector, faces, width: int, height: int,
) -> None:
    """Clip, shade, sort, and fill every polygon."""
    drawable: list[tuple[float, list[tuple[int, int]], tuple[int, int, int]]] = []
    for face in faces:
        points, colour, shade = face[0], face[1], face[2]
        bias = face[3] if len(face) > 3 else 0.0
        view_points = [projector.to_view(point) for point in points]
        clipped = _clip_near(view_points)
        if len(clipped) < 3:
            continue
        depth = sum(point[2] for point in clipped) / len(clipped) - bias
        if not math.isfinite(depth):
            continue
        # A face grazing the near plane projects far outside the viewport. Bounding
        # it here keeps every filled polygon on-screen-sized instead of asking the
        # blitter to rasterize something tens of thousands of pixels wide.
        screen = _clip_viewport([projector.project(point) for point in clipped], width, height)
        if len(screen) < 3:
            continue
        drawable.append((depth, screen, _shade(colour, points) if shade else colour))
    # Painter's algorithm on face centroids: far polygons first. The road is a
    # ribbon of non-intersecting quads and cars sit on it, so centroids order the
    # scene correctly. Sorting on the nearest vertex instead is tempting -- it
    # reads more like a depth test -- but the road quad a car stands on has an
    # edge right under the camera, so it would sort in front of the car and paint
    # over it.
    drawable.sort(key=lambda item: -item[0])
    for _, screen, colour in drawable:
        pygame.draw.polygon(surface, colour, screen)


def _clip_near(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Sutherland-Hodgman clip of one polygon against the near plane.

    Without this, a vertex behind the eye projects to the wrong side of the
    screen and the polygon is drawn inside out, which in an in-car view smears
    the road across the whole frame every time a segment passes the camera.
    """
    clipped: list[tuple[float, float, float]] = []
    count = len(points)
    for index in range(count):
        current = points[index]
        following = points[(index + 1) % count]
        current_inside = current[2] >= NEAR_PLANE
        following_inside = following[2] >= NEAR_PLANE
        if current_inside:
            clipped.append(current)
        if current_inside != following_inside:
            span = following[2] - current[2]
            if abs(span) < 1e-9:
                continue
            fraction = (NEAR_PLANE - current[2]) / span
            clipped.append((
                current[0] + (following[0] - current[0]) * fraction,
                current[1] + (following[1] - current[1]) * fraction,
                NEAR_PLANE,
            ))
    return clipped


def _clip_viewport(
    points: list[tuple[int, int]], width: int, height: int,
) -> list[tuple[int, int]]:
    """Clip a convex screen polygon to the viewport, one edge at a time.

    A face grazing the near plane projects far outside the frame. Bounding it
    here means every filled polygon is at most screen-sized, instead of asking
    the blitter to rasterize something tens of thousands of pixels wide.
    """
    margin = 2.0
    polygon: list[tuple[float, float]] = [(float(x), float(y)) for x, y in points]
    edges = (
        (0, True, -margin), (0, False, width + margin),
        (1, True, -margin), (1, False, height + margin),
    )
    for axis, keep_above, limit in edges:
        if len(polygon) < 3:
            return []
        clipped: list[tuple[float, float]] = []
        count = len(polygon)
        for index in range(count):
            current = polygon[index]
            following = polygon[(index + 1) % count]
            current_inside = (
                current[axis] >= limit if keep_above else current[axis] <= limit
            )
            following_inside = (
                following[axis] >= limit if keep_above else following[axis] <= limit
            )
            if current_inside:
                clipped.append(current)
            if current_inside != following_inside:
                span = following[axis] - current[axis]
                if abs(span) < 1e-9:
                    continue
                fraction = (limit - current[axis]) / span
                clipped.append((
                    current[0] + (following[0] - current[0]) * fraction,
                    current[1] + (following[1] - current[1]) * fraction,
                ))
        polygon = clipped
    return [(int(round(x)), int(round(y))) for x, y in polygon]


def _shade(colour: tuple[int, int, int], points) -> tuple[int, int, int]:
    """Flat-shade a face from its own geometric normal."""
    if len(points) < 3:
        return colour
    normal = _cross(_subtract(points[1], points[0]), _subtract(points[2], points[0]))
    length = math.sqrt(_dot(normal, normal))
    if length <= 1e-9:
        return colour
    normal = (normal[0] / length, normal[1] / length, normal[2] / length)
    intensity = AMBIENT_LIGHT + DIFFUSE_LIGHT * abs(_dot(normal, LIGHT_DIRECTION))
    return (
        min(255, int(colour[0] * intensity)),
        min(255, int(colour[1] * intensity)),
        min(255, int(colour[2] * intensity)),
    )


def _visible_segments(
    world: Racing3DWorld, projector: _Projector, draw_distance: float = DRAW_DISTANCE,
) -> list[int]:
    """Indices whose road segment can plausibly appear, culled in world space.

    The distance limit is a parameter because a driving camera and an inspection camera want
    different ones: a chase view never needs the far side of the circuit, and a camera orbiting
    the whole circuit needs nothing else.
    """
    points = world.scene.track_centerline
    count = len(points)
    visible: list[int] = []
    for index in range(count):
        current = points[index]
        offset = (
            current.x - projector.eye[0],
            current.y - projector.eye[1],
            world.surface.height_at_index(index) - projector.eye[2],
        )
        distance = math.sqrt(_dot(offset, offset))
        if distance > draw_distance:
            continue
        # Keep segments slightly behind the eye so the quad the camera sits on is
        # still drawn and clipped rather than popping out of frame.
        if _dot(offset, projector.forward) < -120.0:
            continue
        visible.append(index)
    return visible


def _road_faces(world: Racing3DWorld, visible: list[int], road, edge, detail: int = 1, kerbs: dict | None = None):
    """Road surface, curb strips, and centre dashes as world-space quads."""
    scene = world.scene
    surface = world.surface
    points = scene.track_centerline
    count = len(points)
    half_width = scene.track_width / 2
    faces = []
    steps = max(1, detail)
    for index in visible:
        # Curbs alternate per segment like real kerbing, so speed and corner entry
        # read at a glance; subdividing must not turn that into a fine stripe.
        kerb = kerbs or {}
        if not kerb.get("kerbs", True):
            # "No red-and-white edge lines" is a real request, so the strip takes the
            # road's own colour rather than being skipped: the kerb quads are also what
            # give the road a defined edge against the terrain.
            curb_colour = road
        else:
            curb_colour = (
                kerb.get("kerb_light", CURB_LIGHT) if index % 2 == 0
                else kerb.get("kerb_dark", CURB_DARK)
            )
        for step in range(steps):
            start = index + step / steps
            finish = index + (step + 1) / steps
            left_a, right_a = _edge_points(world, start, half_width)
            left_b, right_b = _edge_points(world, finish, half_width)
            faces.append(([left_a, right_a, right_b, left_b], road, True))
            outer_left_a, outer_right_a = _edge_points(world, start, half_width + 9)
            outer_left_b, outer_right_b = _edge_points(world, finish, half_width + 9)
            faces.append(([left_a, outer_left_a, outer_left_b, left_b], curb_colour, True))
            faces.append(([right_a, outer_right_a, outer_right_b, right_b], curb_colour, True))
        if index % 4 < 2:
            faces.append((
                [
                    _lane_point(world, index, -3.0), _lane_point(world, index, 3.0),
                    _lane_point(world, index + 1, 3.0), _lane_point(world, index + 1, -3.0),
                ],
                edge, False,
            ))
    del count
    return faces


def _terrain_faces(world: Racing3DWorld, visible: list[int], ground):
    """A skirt of ground either side of the road, following its elevation."""
    scene = world.scene
    points = scene.track_centerline
    count = len(points)
    half_width = scene.track_width / 2
    outer = half_width + half_width * SKIRT_WIDTH_MULTIPLE
    shadowed = _mix(ground, (0, 0, 0), .18)
    faces = []
    for index in visible:
        following = (index + 1) % count
        left_a, right_a = _edge_points(world, index, half_width + 9)
        left_b, right_b = _edge_points(world, following, half_width + 9)
        far_left_a = _drop(_edge_points(world, index, outer)[0], SKIRT_DROP)
        far_left_b = _drop(_edge_points(world, following, outer)[0], SKIRT_DROP)
        far_right_a = _drop(_edge_points(world, index, outer)[1], SKIRT_DROP)
        far_right_b = _drop(_edge_points(world, following, outer)[1], SKIRT_DROP)
        faces.append(([left_a, far_left_a, far_left_b, left_b], shadowed, True))
        faces.append(([right_a, far_right_a, far_right_b, right_b], shadowed, True))
    return faces


def _edge_points(
    world: Racing3DWorld, position: float, offset: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Left and right road-edge points at a sample position, raised by banking.

    `position` may be fractional. Height and bank come from the smooth elevation
    profile rather than a straight line between samples, so subdividing a segment
    genuinely rounds a crest instead of splitting one flat plane into two.
    """
    points = world.scene.track_centerline
    count = len(points)
    index = math.floor(position)
    fraction = position - index
    current = points[index % count]
    following = points[(index + 1) % count]
    centre_x = current.x + (following.x - current.x) * fraction
    centre_y = current.y + (following.y - current.y) * fraction
    before = points[(index - 1) % count]
    after = points[(index + 2) % count]
    tangent_x = following.x - before.x + (after.x - current.x - (following.x - before.x)) * fraction
    tangent_y = following.y - before.y + (after.y - current.y - (following.y - before.y)) * fraction
    length = max(1e-6, math.hypot(tangent_x, tangent_y))
    normal_x, normal_y = -tangent_y / length, tangent_x / length
    centre_z = world.surface.height_at_position(position)
    rise = math.tan(world.surface.bank_at_position(position)) * offset
    return (
        (centre_x + normal_x * offset, centre_y + normal_y * offset, centre_z + rise),
        (centre_x - normal_x * offset, centre_y - normal_y * offset, centre_z - rise),
    )


def _lane_point(world: Racing3DWorld, index: float, offset: float) -> tuple[float, float, float]:
    left, right = _edge_points(world, index, abs(offset))
    return left if offset >= 0 else right


def _gate_faces(world: Racing3DWorld):
    """Sector gates as a pair of edge markers, with nothing spanning the road.

    A stripe painted across the corridor is the obvious equivalent of the 2D
    line, but a single flat quad cannot follow a road that is both climbing and
    cambered, so on an elevated circuit it cuts through the surface or hovers
    over it. Markers on the two edges need no such surface fit: each stands on
    its own edge, at that edge's own banked height.
    """
    faces = []
    half_width = world.scene.track_width / 2
    for marker in world.gate_poses():
        if marker.passed:
            # A gate already crossed carries no information for the driver, and
            # its markers would sit right beside the camera.
            continue
        colour = GATE_FINISH if marker.is_finish else GATE_SECTOR
        forward, left, up = marker.pose.basis()
        left_foot, right_foot = _edge_points(world, marker.track_index, half_width + 7)
        height = 38.0 if marker.is_finish else 30.0
        for foot in (left_foot, right_foot):
            faces.extend(_box_faces(
                foot, forward, left, up,
                half_length=3.5, half_width=3.5, height=height,
                colour=colour, cap_colour=_mix(colour, (255, 255, 255), .35),
            ))
    return faces


def _barrier_faces(world: Racing3DWorld, barrier_colour=BARRIER_BODY):
    """Barriers extruded from their declared 2D collision outline.

    Height is the only thing the renderer adds: the footprint comes straight from
    the collider, so a round bollard, an axis-aligned block, and a wall laid along
    the road all look like what the physics will test against.
    """
    faces = []
    height = 20.0
    for entity_id, pose, footprint in world.barrier_footprints():
        if len(footprint) < 3:
            continue
        base_z = pose.z
        top = [(x, y, base_z + height) for x, y in footprint]
        faces.append((top, BARRIER_TOP, True))
        for index, corner in enumerate(footprint):
            following = footprint[(index + 1) % len(footprint)]
            faces.append((
                [
                    (corner[0], corner[1], base_z),
                    (following[0], following[1], base_z),
                    (following[0], following[1], base_z + height),
                    (corner[0], corner[1], base_z + height),
                ],
                barrier_colour, True,
            ))
        del entity_id
    return faces


def _edge_barrier_faces(
    world: Racing3DWorld, visible: list[int], barrier_colour=BARRIER_BODY,
):
    """Continuous guardrail walls following both banked road boundaries.

    The inner face begins at exactly `track_width / 2`, and the outer face ends
    one shared collision thickness farther out. Thus the visible footprint and
    planar collider agree even while their base heights follow elevation and
    camber in 3D.
    """
    if not world.scene.edge_barriers:
        return []
    inner_offset = world.scene.track_width / 2
    outer_offset = inner_offset + EDGE_BARRIER_THICKNESS
    height = 20.0
    faces = []
    for index in visible:
        inner_a = _edge_points(world, index, inner_offset)
        inner_b = _edge_points(world, index + 1, inner_offset)
        outer_a = _edge_points(world, index, outer_offset)
        outer_b = _edge_points(world, index + 1, outer_offset)
        for side in (0, 1):
            ia, ib = inner_a[side], inner_b[side]
            oa, ob = outer_a[side], outer_b[side]
            ia_top = (ia[0], ia[1], ia[2] + height)
            ib_top = (ib[0], ib[1], ib[2] + height)
            oa_top = (oa[0], oa[1], oa[2] + height)
            ob_top = (ob[0], ob[1], ob[2] + height)
            faces.append(([ia, ib, ib_top, ia_top], barrier_colour, True))
            faces.append(([oa, oa_top, ob_top, ob], _mix(barrier_colour, (0, 0, 0), .18), True))
            faces.append(([ia_top, ib_top, ob_top, oa_top], BARRIER_TOP, True))
    return faces


def _barrier_impact_faces(world: Racing3DWorld):
    """A one-frame 3D spark at the authoritative planar contact point."""
    if world.barrier_impact is None or world.surface is None:
        return []
    point = world.barrier_impact
    track = world.scene.track_centerline
    index = min(
        range(len(track)),
        key=lambda item: (track[item].x - point.x) ** 2 + (track[item].y - point.y) ** 2,
    )
    base_z = world.surface.surface_height(Vec2(x=point.x, y=point.y), track, index)
    faces = []
    for angle_degrees in (0, 72, 144, 216, 288):
        angle = math.radians(angle_degrees)
        direction = (math.cos(angle), math.sin(angle))
        perpendicular = (-direction[1], direction[0])
        faces.append(([
            (
                point.x + perpendicular[0] * 1.8,
                point.y + perpendicular[1] * 1.8,
                base_z + 5.0,
            ),
            (
                point.x + direction[0] * 15.0,
                point.y + direction[1] * 15.0,
                base_z + 12.0,
            ),
            (
                point.x - perpendicular[0] * 1.8,
                point.y - perpendicular[1] * 1.8,
                base_z + 22.0,
            ),
        ], IMPACT_SPARK, True))
    return faces


def _car_faces(world: Racing3DWorld, *, include_player: bool, palette: dict | None = None):
    """Every car as a low-poly body, cabin, and wheels sized from its dynamics.

    An in-car camera sits at driver head height, which is physically inside the
    cabin, so the ego car's own bodywork is skipped for those modes rather than
    filling the frame with the inside of its roof.
    """
    colours = palette or {}
    opponent_body = colours.get("opponent", OPPONENT_BODY)
    player_body = colours.get("player", PLAYER_BODY)
    vehicle = world.dynamics.vehicle
    scale = world.dynamics.pixels_per_meter
    length = vehicle.length_m * scale
    width = vehicle.width_m * scale
    faces = []
    for entity in world.scene.entities:
        if entity.kind != EntityKind.NPC:
            continue
        pose = world.opponent_pose(entity.id)
        if pose is None:
            continue
        opponent = next(
            (item for item in world.opponents if item.entity_id == entity.id), None,
        )
        faces.extend(_single_car_faces(
            pose, length, width, opponent_body, _tint(opponent_body, 1.5),
            nitro=bool(opponent and opponent.nitro_active),
        ))
    if include_player:
        faces.extend(_single_car_faces(
            world.player_pose(), length, width, player_body,
            PLAYER_ACCENT if player_body == PLAYER_BODY else _tint(player_body, .55),
            nitro=world.nitro_active,
        ))
    return faces


def _single_car_faces(
    pose: CarPose3D, length: float, width: float,
    body_colour: tuple[int, int, int], accent: tuple[int, int, int], *, nitro: bool,
):
    """Every face carries `CAR_DEPTH_BIAS` so the road cannot paint over the car."""
    forward, left, up = pose.basis()
    # Sit the body on its suspension rather than on the road, so wheels read as
    # separate parts instead of sinking into the surface.
    ride_height = width * .18
    base = _add(pose.position, _scale(up, ride_height))
    body_height = width * .52
    faces = _box_faces(
        base, forward, left, up,
        half_length=length / 2, half_width=width / 2, height=body_height,
        colour=body_colour, cap_colour=accent,
    )
    cabin_base = _add(base, _scale(up, body_height))
    faces.extend(_box_faces(
        _add(cabin_base, _scale(forward, -length * .06)),
        forward, left, up,
        half_length=length * .26, half_width=width * .38, height=width * .27,
        colour=CABIN_GLASS, cap_colour=_mix(body_colour, (0, 0, 0), .18),
    ))
    wheel_radius = width * .17
    for longitudinal in (length * .31, -length * .31):
        for lateral in (width * .5, -width * .5):
            hub = _add(
                _add(pose.position, _scale(forward, longitudinal)),
                _add(_scale(left, lateral), _scale(up, wheel_radius)),
            )
            faces.extend(_box_faces(
                _add(hub, _scale(up, -wheel_radius)), forward, left, up,
                half_length=wheel_radius, half_width=width * .07,
                height=wheel_radius * 2, colour=TIRE,
            ))
    if nitro:
        tail = _add(base, _scale(forward, -length / 2))
        faces.append((
            [
                _add(tail, _scale(left, width * .22)),
                _add(tail, _scale(left, -width * .22)),
                _add(_add(tail, _scale(forward, -length * .42)), _scale(up, body_height * .3)),
            ],
            NITRO_FLAME, False,
        ))
    return [(points, colour, shade, CAR_DEPTH_BIAS) for points, colour, shade in faces]


def _box_faces(
    base: tuple[float, float, float],
    forward: tuple[float, float, float],
    left: tuple[float, float, float],
    up: tuple[float, float, float],
    *, half_length: float, half_width: float, height: float,
    colour: tuple[int, int, int], cap_colour: tuple[int, int, int] | None = None,
):
    """A ground-anchored box in a car-local frame, as five visible quads."""
    def corner(longitudinal: float, lateral: float, vertical: float):
        return _add(
            _add(base, _scale(forward, longitudinal)),
            _add(_scale(left, lateral), _scale(up, vertical)),
        )

    front_left_low = corner(half_length, half_width, 0.0)
    front_right_low = corner(half_length, -half_width, 0.0)
    rear_right_low = corner(-half_length, -half_width, 0.0)
    rear_left_low = corner(-half_length, half_width, 0.0)
    front_left_high = corner(half_length, half_width, height)
    front_right_high = corner(half_length, -half_width, height)
    rear_right_high = corner(-half_length, -half_width, height)
    rear_left_high = corner(-half_length, half_width, height)
    side = _mix(colour, (0, 0, 0), .12)
    return [
        ([front_left_high, front_right_high, rear_right_high, rear_left_high],
         cap_colour or colour, True),
        ([front_left_low, front_right_low, front_right_high, front_left_high], colour, True),
        ([rear_left_low, rear_right_low, rear_right_high, rear_left_high], side, True),
        ([front_left_low, rear_left_low, rear_left_high, front_left_high], side, True),
        ([front_right_low, rear_right_low, rear_right_high, front_right_high], side, True),
    ]


def _pitched(
    forward: tuple[float, float, float], up: tuple[float, float, float], degrees: float,
) -> tuple[float, float, float]:
    radians = math.radians(degrees)
    return _normalize(_add(_scale(forward, math.cos(radians)), _scale(up, math.sin(radians))))


def _drop(point: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    return (point[0], point[1], point[2] - amount)


def _add(first, second):
    return (first[0] + second[0], first[1] + second[1], first[2] + second[2])


def _subtract(first, second):
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _scale(vector, factor: float):
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(first, second) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(first, second):
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _normalize(vector):
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _mix(first, second, blend: float):
    return (
        int(first[0] + (second[0] - first[0]) * blend),
        int(first[1] + (second[1] - first[1]) * blend),
        int(first[2] + (second[2] - first[2]) * blend),
    )


def _clamp_screen(value: float) -> int:
    if not math.isfinite(value):
        return SCREEN_LIMIT
    return int(max(-SCREEN_LIMIT, min(SCREEN_LIMIT, value)))

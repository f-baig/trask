"""Exact, swept collision between the car and scene obstacles.

Two properties matter here, and the engine got both wrong before this module
existed.

**Shapes are explicit.** An obstacle used to be an axis-aligned rectangle in the
physics while both renderers drew it as a circle, so what you saw was not quite
what you hit. A collider now declares its shape, every renderer draws that shape,
and the same declaration is what the physics tests against. Adding a new shape
means adding one branch here rather than hunting through renderers.

**Tests are swept, not sampled.** Collision used to be evaluated once per control
tick, after six substeps of integration had already moved the car. Vehicle top
speed is a tunable parameter, so a fast enough car simply passed through a
barrier between two tests. Every test now covers the whole path travelled during
the tick, which makes tunnelling impossible at any speed the dynamics allow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import CollisionShape, EntitySpec, SceneSpec


# The path travelled in one control tick is sampled no more coarsely than this
# fraction of the moving circle's radius, so a barrier cannot slip between two
# samples however fast the car is going.
SWEEP_RESOLUTION = .5
CONTACT_BISECTION_STEPS = 14
EDGE_BARRIER_THICKNESS = 6.0


@dataclass(frozen=True)
class SweepContact:
    """First contact along a swept circle path.

    `safe_point` is the last non-overlapping position and `impact_point` is the
    first overlapping one to bisection precision. The normal points from the
    collider toward the car, which is exactly the direction a rebound uses.
    """

    safe_point: tuple[float, float]
    impact_point: tuple[float, float]
    normal: tuple[float, float]
    fraction: float


@dataclass(frozen=True)
class Collider:
    """One obstacle's collision geometry, resolved from its scene entity."""

    shape: CollisionShape
    centre_x: float
    centre_y: float
    half_width: float
    half_height: float
    rotation_degrees: float = 0.0

    @property
    def bounding_radius(self) -> float:
        """Radius of the smallest circle containing this collider."""
        if self.shape == CollisionShape.CIRCLE:
            return self.radius
        return math.hypot(self.half_width, self.half_height)

    @property
    def radius(self) -> float:
        """Radius used when this collider is a circle."""
        return min(self.half_width, self.half_height)

    def hits_circle(self, x: float, y: float, radius: float) -> bool:
        """Whether a circle at `(x, y)` overlaps this collider."""
        if self.shape == CollisionShape.CIRCLE:
            return math.hypot(x - self.centre_x, y - self.centre_y) < self.radius + radius
        local_x, local_y = x - self.centre_x, y - self.centre_y
        if self.shape == CollisionShape.ORIENTED_BOX and self.rotation_degrees:
            # Rotate the circle into the box's frame; a box test is then trivial.
            angle = math.radians(-self.rotation_degrees)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            local_x, local_y = (
                local_x * cos_a - local_y * sin_a,
                local_x * sin_a + local_y * cos_a,
            )
        nearest_x = max(-self.half_width, min(self.half_width, local_x))
        nearest_y = max(-self.half_height, min(self.half_height, local_y))
        return math.hypot(local_x - nearest_x, local_y - nearest_y) < radius

    def hits_swept_circle(
        self, start: tuple[float, float], end: tuple[float, float], radius: float,
    ) -> bool:
        """Whether a circle sweeping from `start` to `end` touches this collider.

        The segment is sampled finely enough that no gap can exceed the moving
        circle's own size, and it is rejected early with a cheap bounding test so
        the common case of a distant barrier costs almost nothing.
        """
        return self.sweep_contact(start, end, radius) is not None

    def sweep_contact(
        self, start: tuple[float, float], end: tuple[float, float], radius: float,
    ) -> SweepContact | None:
        """Resolve the first swept hit, including a stable outward normal."""
        span = (end[0] - start[0], end[1] - start[1])
        travel = math.hypot(*span)
        fallback = _normalized((-span[0], -span[1]), (1.0, 0.0))
        if self.hits_circle(start[0], start[1], radius):
            return SweepContact(
                safe_point=start, impact_point=start,
                normal=self.contact_normal(start[0], start[1], fallback), fraction=0.0,
            )
        if travel <= 1e-9:
            return None
        # Cheap reject: distance from the collider centre to the swept segment.
        if _distance_point_to_segment(
            (self.centre_x, self.centre_y), start, end,
        ) > self.bounding_radius + radius:
            return None
        steps = max(1, math.ceil(travel / max(1e-6, radius * SWEEP_RESOLUTION)))
        for index in range(1, steps + 1):
            fraction = index / steps
            point = (start[0] + span[0] * fraction, start[1] + span[1] * fraction)
            if not self.hits_circle(*point, radius):
                continue
            low, high = (index - 1) / steps, fraction
            for _ in range(CONTACT_BISECTION_STEPS):
                middle = (low + high) / 2
                probe = (start[0] + span[0] * middle, start[1] + span[1] * middle)
                if self.hits_circle(*probe, radius):
                    high = middle
                else:
                    low = middle
            safe = (start[0] + span[0] * low, start[1] + span[1] * low)
            impact = (start[0] + span[0] * high, start[1] + span[1] * high)
            return SweepContact(
                safe_point=safe, impact_point=impact,
                normal=self.contact_normal(*impact, fallback), fraction=high,
            )
        return None

    def contact_normal(
        self, x: float, y: float, fallback: tuple[float, float] = (1.0, 0.0),
    ) -> tuple[float, float]:
        """Outward surface normal nearest a circle centre.

        Deep overlaps can put the centre inside a box. In that case the nearest
        face supplies the escape direction instead of returning a zero vector.
        """
        if self.shape == CollisionShape.CIRCLE:
            return _normalized((x - self.centre_x, y - self.centre_y), fallback)
        local_x, local_y = x - self.centre_x, y - self.centre_y
        angle = 0.0
        if self.shape == CollisionShape.ORIENTED_BOX and self.rotation_degrees:
            angle = math.radians(-self.rotation_degrees)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            local_x, local_y = (
                local_x * cos_a - local_y * sin_a,
                local_x * sin_a + local_y * cos_a,
            )
        nearest_x = max(-self.half_width, min(self.half_width, local_x))
        nearest_y = max(-self.half_height, min(self.half_height, local_y))
        delta = (local_x - nearest_x, local_y - nearest_y)
        if math.hypot(*delta) <= 1e-9:
            x_clearance = self.half_width - abs(local_x)
            y_clearance = self.half_height - abs(local_y)
            if x_clearance <= y_clearance:
                delta = (1.0 if local_x >= 0 else -1.0, 0.0)
            else:
                delta = (0.0, 1.0 if local_y >= 0 else -1.0)
        normal = _normalized(delta, fallback)
        if angle:
            # Undo the world-to-local rotation used above.
            cos_a, sin_a = math.cos(-angle), math.sin(-angle)
            normal = (
                normal[0] * cos_a - normal[1] * sin_a,
                normal[0] * sin_a + normal[1] * cos_a,
            )
        return _normalized(normal, fallback)


def collider_for(entity: EntitySpec, y_shift: float = 0.0) -> Collider:
    """Build the collider an obstacle entity declares.

    `y_shift` carries the obstacle-shift perturbation, so a forked experiment
    collides against the moved obstacle rather than its authored position.
    """
    rect = entity.rect
    return Collider(
        shape=entity.shape,
        centre_x=rect.x + rect.width / 2,
        centre_y=rect.y + rect.height / 2 + y_shift,
        half_width=rect.width / 2,
        half_height=rect.height / 2,
        rotation_degrees=entity.rotation_degrees,
    )


def circle_collider(centre_x: float, centre_y: float, radius: float) -> Collider:
    """A round collider, used for cars and for round obstacles."""
    return Collider(
        shape=CollisionShape.CIRCLE, centre_x=centre_x, centre_y=centre_y,
        half_width=radius, half_height=radius,
    )


def track_edge_points(
    scene: SceneSpec, offset: float | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Left and right points following the sampled road boundary."""
    points = scene.track_centerline
    distance = scene.track_width / 2 if offset is None else offset
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for index, current in enumerate(points):
        before = points[(index - 1) % len(points)]
        after = points[(index + 1) % len(points)]
        tangent_x, tangent_y = after.x - before.x, after.y - before.y
        length = max(1e-9, math.hypot(tangent_x, tangent_y))
        normal_x, normal_y = -tangent_y / length, tangent_x / length
        left.append((current.x + normal_x * distance, current.y + normal_y * distance))
        right.append((current.x - normal_x * distance, current.y - normal_y * distance))
    return left, right


def edge_barrier_colliders(scene: SceneSpec) -> list[tuple[str, Collider]]:
    """Continuous guardrails as overlapping oriented boxes on both road edges.

    The wall centre sits half its thickness outside the authored road boundary,
    so the car's legal centre corridor remains exactly `track_width / 2 - radius`.
    Adjacent boxes overlap by one wall thickness to leave no gaps through bends.
    """
    if not scene.edge_barriers or len(scene.track_centerline) < 3:
        return []
    offset = scene.track_width / 2 + EDGE_BARRIER_THICKNESS / 2
    left, right = track_edge_points(scene, offset)
    colliders: list[tuple[str, Collider]] = []
    for side_name, edge in (("left", left), ("right", right)):
        for index, start in enumerate(edge):
            end = edge[(index + 1) % len(edge)]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            colliders.append((
                f"edge-{side_name}-{index}",
                Collider(
                    shape=CollisionShape.ORIENTED_BOX,
                    centre_x=(start[0] + end[0]) / 2,
                    centre_y=(start[1] + end[1]) / 2,
                    half_width=length / 2 + EDGE_BARRIER_THICKNESS,
                    half_height=EDGE_BARRIER_THICKNESS / 2,
                    rotation_degrees=math.degrees(math.atan2(dy, dx)),
                ),
            ))
    return colliders


def outline(collider: Collider, segments: int = 16) -> list[tuple[float, float]]:
    """The collider's boundary as a polygon, so renderers draw what physics tests."""
    if collider.shape == CollisionShape.CIRCLE:
        return [
            (
                collider.centre_x + math.cos(math.tau * index / segments) * collider.radius,
                collider.centre_y + math.sin(math.tau * index / segments) * collider.radius,
            )
            for index in range(segments)
        ]
    angle = math.radians(
        collider.rotation_degrees if collider.shape == CollisionShape.ORIENTED_BOX else 0.0
    )
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = (
        (-collider.half_width, -collider.half_height),
        (collider.half_width, -collider.half_height),
        (collider.half_width, collider.half_height),
        (-collider.half_width, collider.half_height),
    )
    return [
        (
            collider.centre_x + local_x * cos_a - local_y * sin_a,
            collider.centre_y + local_x * sin_a + local_y * cos_a,
        )
        for local_x, local_y in corners
    ]


def _distance_point_to_segment(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float],
) -> float:
    span_x, span_y = end[0] - start[0], end[1] - start[1]
    length_squared = span_x * span_x + span_y * span_y
    if length_squared <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = max(0.0, min(1.0, (
        (point[0] - start[0]) * span_x + (point[1] - start[1]) * span_y
    ) / length_squared))
    return math.hypot(
        point[0] - (start[0] + span_x * fraction),
        point[1] - (start[1] + span_y * fraction),
    )


def _normalized(
    vector: tuple[float, float], fallback: tuple[float, float],
) -> tuple[float, float]:
    magnitude = math.hypot(*vector)
    if magnitude <= 1e-12:
        fallback_magnitude = math.hypot(*fallback)
        return (
            (fallback[0] / fallback_magnitude, fallback[1] / fallback_magnitude)
            if fallback_magnitude > 1e-12 else (1.0, 0.0)
        )
    return vector[0] / magnitude, vector[1] / magnitude

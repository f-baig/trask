"""Pixels-only local cues for the forward-cone reflex variant."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..motion import dense_optical_flow, _grayscale, _surface_from_frame


@dataclass
class ConeVisionSense:
    """Stateful screenshot reader. Units are image pixels, never world units."""

    previous_gray: object | None = field(default=None, repr=False)
    tick: int = 0
    frames_since_flow: int = 0
    previous_center: float | None = None
    previous_turn: float | None = None
    previous_confidence: float | None = None
    previous_edge_gap: float | None = None
    road_colour: object | None = field(default=None, repr=False)

    def update(self, frame) -> dict[str, float | bool]:
        numpy = _numpy()
        gray = _grayscale(_surface_from_frame(frame))
        height, width = gray.shape
        anchor_x, anchor_y = width // 2, height - 38
        # The frozen comparison fixtures are asphalt.  Their road is visibly dark
        # while grass, edge paint, cars, and the black cone mask are not.  This is a
        # screenshot threshold (not a palette/world lookup), and unlike sampling the
        # ego neighbourhood it cannot relabel off-road grass as drivable road.
        road = self._road_mask(frame, gray)
        profile = []
        for depth, y in (("near", .78), ("near_mid", .66), ("mid", .54), ("far_mid", .42), ("far", .30)):
            runs = _runs(road[min(height - 1, max(0, round(height * y)))])
            if runs:
                left, right = min(runs, key=lambda run: abs(((run[0] + run[1]) / 2) - anchor_x))
                if right - left >= 8:
                    profile.append({"depth": depth, "lateral": round(((left + right) / 2 - anchor_x) / max(10.0, (right - left) / 2), 4), "width": round((right - left) / width, 4), "detected": True})
                    continue
            profile.append({"depth": depth, "lateral": None, "width": None, "detected": False})
        centers: list[tuple[float, float, float, float]] = []
        for y in (round(height * 0.74), round(height * 0.57), round(height * 0.40)):
            runs = _runs(road[min(height - 1, max(0, y))])
            if runs:
                left, right = min(runs, key=lambda run: abs(((run[0] + run[1]) / 2) - anchor_x))
                if right - left >= 8:
                    centers.append(((left + right) / 2, right - left, left, right))
        if centers:
            near_x, near_width, near_left, near_right = centers[0]
            lane = (near_x - anchor_x) / max(10.0, near_width / 2)
            turn = (centers[-1][0] - anchor_x) / max(10.0, near_width / 2)
            visible = True
        else:
            near_x = near_width = near_left = near_right = 0.0
            lane = turn = 0.0
            visible = False
        # Dense flow is the expensive part of the visual sensor.  Four-frame
        # intervals preserve a per-tick magnitude while keeping controller
        # rehearsals practical; all four inputs are still cone screenshots.
        self.frames_since_flow += 1
        flow_speed = flow_rotation = 0.0
        if self.previous_gray is not None and self.previous_gray.shape == gray.shape and self.frames_since_flow >= 4:
            flow = dense_optical_flow(self.previous_gray, gray)
            magnitudes = numpy.sqrt((flow[:, :, 0] ** 2) + (flow[:, :, 1] ** 2))
            observed = road & (numpy.indices(road.shape)[0] < anchor_y - 8)
            if int(observed.sum()) > 40:
                flow_speed = float(numpy.median(magnitudes[observed])) / self.frames_since_flow
                flow_rotation = float(numpy.median(flow[:, :, 0][observed])) / self.frames_since_flow
            self.previous_gray = gray
            self.frames_since_flow = 0
        elif self.previous_gray is None:
            self.previous_gray = gray
        self.tick += 1
        offsets = [((center - anchor_x) / max(10.0, span / 2)) for center, span, _, _ in centers]
        centers_by_depth = tuple(
            max(-2.0, min(2.0, value))
            for value in (offsets + [0.0, 0.0, 0.0])[:3]
        )
        # These are geometric measurements of the *rendered road*, not a lookup into the
        # track.  Separating the bend across depth from the current lane offset lets an
        # authored controller brake before a tight turn rather than react at its edge.
        turn_ahead = centers_by_depth[2] - centers_by_depth[0] if len(centers) >= 2 else 0.0
        turn_severity = (
            abs(centers_by_depth[2] - centers_by_depth[0])
            + 0.5 * abs(centers_by_depth[1] - centers_by_depth[0])
            if len(centers) >= 2 else 0.0
        )
        depth_rows = []
        for y in range(anchor_y - 12, max(round(height * 0.24), 1), -4):
            runs = _runs(road[y])
            if runs:
                nearest = min(runs, key=lambda run: abs(((run[0] + run[1]) / 2) - anchor_x))
                if nearest[1] - nearest[0] >= 8:
                    depth_rows.append(y)
        lookahead_depth = (
            (anchor_y - min(depth_rows)) / max(1.0, anchor_y - round(height * 0.24))
            if depth_rows else 0.0
        )
        confidence = len(centers) / 3.0
        left_gap = float((anchor_x - near_left) / width) if visible else 0.0
        right_gap = float((near_right - anchor_x) / width) if visible else 0.0
        edge_gap = min(left_gap, right_gap)
        # Ego contact is intentionally stricter than seeing road somewhere ahead:
        # sample a small patch immediately in front of the camera anchor.
        ego_patch = road[max(0, anchor_y - 10):anchor_y - 2, anchor_x - 7:anchor_x + 8]
        ego_contact = bool(ego_patch.size and float(ego_patch.mean()) > 0.45)
        # When contact is lost, find the closest detected road pixels in the lower
        # cone. Its signed image direction is a visual recovery hint, not a route map.
        lower = road[round(height * .45):anchor_y]
        ys, xs = numpy.nonzero(lower)
        recovery = 0.0 if not len(xs) else float(numpy.median(xs) - anchor_x) / max(1.0, width / 2)
        center_rate = 0.0 if self.previous_center is None else centers_by_depth[0] - self.previous_center
        turn_delta = 0.0 if self.previous_turn is None else turn - self.previous_turn
        confidence_trend = 0.0 if self.previous_confidence is None else confidence - self.previous_confidence
        edge_closing = 0.0 if self.previous_edge_gap is None else edge_gap - self.previous_edge_gap
        self.previous_center, self.previous_turn = centers_by_depth[0], turn
        self.previous_confidence, self.previous_edge_gap = confidence, edge_gap
        return {
            "vision_lane": float(max(-2.0, min(2.0, lane))), "vision_turn": float(max(-2.0, min(2.0, turn))),
            "vision_flow": float(max(0.0, min(30.0, flow_speed))), "vision_flow_rotation": float(max(-30.0, min(30.0, flow_rotation))),
            "vision_road_visible": visible, "vision_road_lost": not ego_contact, "vision_ego_road_contact": ego_contact, "vision_recovery_direction": float(max(-1.0, min(1.0, recovery))),
            "vision_center_rate": float(center_rate), "vision_turn_delta": float(turn_delta), "vision_edge_closing_rate": float(edge_closing), "vision_confidence_trend": float(confidence_trend), "vision_center_near": float(centers_by_depth[0]),
            "vision_center_mid": float(centers_by_depth[1]), "vision_center_far": float(centers_by_depth[2]),
            "vision_turn_ahead": float(max(-2.0, min(2.0, turn_ahead))),
            "vision_turn_severity": float(max(0.0, min(2.0, turn_severity))),
            "vision_lookahead_depth": float(max(0.0, min(1.0, lookahead_depth))),
            "vision_road_width": float(max(0.0, min(1.0, near_width / width))),
            "vision_left_gap": float(max(0.0, min(1.0, (anchor_x - near_left) / width))) if visible else 0.0,
            "vision_right_gap": float(max(0.0, min(1.0, (near_right - anchor_x) / width))) if visible else 0.0,
            "vision_confidence": confidence, "on_track": ego_contact, "tick": self.tick,
            # This nested profile is intentionally inspection-only. It is a compact road
            # geometry readout from pixels, not a hidden route or planner output.
            "vision_profile": profile,
        }

    def _road_mask(self, frame, gray):
        """Road segmentation for the flat forward-cone renderer.

        Calibration is strictly from pixels sampled immediately ahead of the ego car.
        A new episode always begins on the road, so that local patch gives us the road's
        appearance without consulting the generated scene, surface preset, or palette.
        Keeping the estimate across frames also means a brief trip onto grass does not
        relabel grass as road.  This supports asphalt, clay, and ice rather than baking
        the old comparison fixtures' dark-asphalt threshold into the player.

        The perspective renderer shares this operation.  It deliberately remains an image
        operation so neither renderer can acquire a hidden world-state shortcut.
        """
        numpy = _numpy()
        rgb = _rgb(frame)
        if self.road_colour is None:
            height, width = gray.shape
            # This window sits just ahead of the ego car, avoids the vehicle sprite at
            # the bottom of the cone, and is road-covered at a legal spawn point.
            patch = rgb[
                round(height * .66):round(height * .80),
                round(width * .40):round(width * .60),
            ]
            if patch.size:
                self.road_colour = numpy.median(patch.reshape(-1, 3), axis=0)
            else:  # Defensive only; a normal RGB frame is never empty.
                self.road_colour = numpy.asarray((.18, .18, .18), dtype=numpy.float32)
        reference = numpy.asarray(self.road_colour, dtype=numpy.float32)
        luminance = rgb.mean(axis=2)
        colour_sum = numpy.maximum(rgb.sum(axis=2, keepdims=True), .08)
        chroma = rgb / colour_sum
        reference_chroma = reference / max(.08, float(reference.sum()))
        chroma_error = numpy.sqrt(((chroma - reference_chroma) ** 2).sum(axis=2))
        reference_luminance = float(reference.mean())
        # Road shading may darken substantially in 3D, so compare hue more tightly
        # than brightness.  The upper cap excludes bright kerbs and lane markings;
        # the lower cap excludes the black cone mask while retaining shaded asphalt.
        return (
            (luminance > max(.025, reference_luminance * .10))
            & (luminance < min(1.0, reference_luminance * 3.0 + .12))
            & (chroma_error < .14)
        )


def _runs(row) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(row):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index)); start = None
    if start is not None:
        result.append((start, len(row)))
    return result


def _numpy():
    import numpy
    return numpy


def _rgb(frame):
    """Return normalized image RGB in ordinary row-major order.

    Pygame is the raster interface, not an observation escape hatch: this reads the
    same supplied screenshot as the model and performs no scene or world lookup.
    """
    numpy = _numpy()
    import pygame

    surface = _surface_from_frame(frame)
    return numpy.asarray(
        pygame.surfarray.array3d(surface), dtype=numpy.float32,
    ).transpose(1, 0, 2) / 255.0

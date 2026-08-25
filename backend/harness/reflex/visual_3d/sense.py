"""Perspective road measurements from first-person 3D screenshots only."""

from __future__ import annotations

from dataclasses import dataclass

from ..vision_sense import ConeVisionSense, _runs
from ...motion import _grayscale, _surface_from_frame


@dataclass
class PerspectiveVisionSense(ConeVisionSense):
    """Adapt the shared image geometry reader to the 3D renderer's road appearance.

    The road detector intentionally uses only RGB pixels.  It does not access the scene,
    elevation surface, car pose, or physics observation.
    """

    previous_horizon: float | None = None

    def update(self, frame) -> dict[str, float | bool]:
        """Measure one coherent perspective road corridor from RGB pixels.

        The original adapter inherited the 2D sensor's independent fixed-row
        scanlines. Lane paint split the asphalt into several runs, so each row
        could select a different fragment and still report confidence 1.0. Here
        we close small paint gaps, follow one overlapping corridor from the ego
        edge toward the horizon, and score the quality of that whole fit.
        """
        numpy = _numpy()
        gray = _grayscale(_surface_from_frame(frame))
        height, width = gray.shape
        anchor_x, anchor_y = width // 2, height - 10
        road = _close_horizontal_gaps(
            self._road_mask(frame, gray), max_gap=max(14, width // 12),
        )
        corridor = _trace_corridor(road, anchor_x=anchor_x, anchor_y=anchor_y)
        self.tick += 1

        if len(corridor) < 10:
            recovery = _recovery_direction(road, anchor_x, anchor_y)
            horizon = 1.0
            shift = 0.0 if self.previous_horizon is None else horizon - self.previous_horizon
            self.previous_horizon = horizon
            self.previous_center = self.previous_turn = 0.0
            self.previous_confidence = 0.0
            return {
                "vision_track_offset": 0.0, "vision_track_heading": 0.0,
                "vision_bend_ahead": 0.0, "vision_bend_severity": 0.0,
                "vision_visible_depth": 0.0, "vision_left_gap": 0.0,
                "vision_right_gap": 0.0, "vision_road_contact": False,
                "vision_recovery_direction": recovery,
                "vision_road_horizon": horizon,
                "vision_horizon_shift": _clamp(shift, -1.0, 1.0),
                "vision_crest_risk": 1.0, "vision_confidence": 0.0,
                "vision_profile": [], "on_track": False, "tick": self.tick,
            }

        bottom_y = max(row[0] for row in corridor)
        top_y = min(row[0] for row in corridor)
        span = max(1, bottom_y - top_y)
        samples = [
            _sample_corridor(corridor, bottom_y - span * fraction)
            for fraction in (.05, .38, .78)
        ]
        near, middle, far = samples
        near_half_width = max(8.0, near[2] / 2)
        offset = _clamp((near[1] - anchor_x) / near_half_width, -2.0, 2.0)
        heading = _clamp((middle[1] - near[1]) / near_half_width, -2.0, 2.0)
        # Offset is ego-near placement, heading is its first image-depth
        # derivative, and bend is the second derivative. Using far-minus-near
        # for bend counted a straight but angled corridor as curvature and made
        # the controller add the same corner three times.
        bend = _clamp(
            (far[1] - 2.0 * middle[1] + near[1]) / near_half_width,
            -2.0, 2.0,
        )
        severity = _clamp(abs(bend) + 0.35 * abs(heading), 0.0, 2.0)
        visible_depth = _clamp(
            (anchor_y - top_y) / max(1.0, anchor_y - height * .18), 0.0, 1.0,
        )
        near_left, near_right = near[3], near[4]
        left_gap = _clamp((anchor_x - near_left) / width, 0.0, 1.0)
        right_gap = _clamp((near_right - anchor_x) / width, 0.0, 1.0)
        # Contact comes from the fitted corridor envelope rather than raw dark
        # pixels, so a white centre stripe cannot make the road disappear.
        lower_rows = [row for row in corridor if row[0] >= height * .70]
        contact_votes = sum(row[3] - 5 <= anchor_x <= row[4] + 5 for row in lower_rows)
        contact = bool(lower_rows and contact_votes / len(lower_rows) >= .55)
        recovery = _clamp((near[1] - anchor_x) / max(1.0, width / 2), -1.0, 1.0)

        coverage = len(corridor) / max(1, bottom_y - top_y + 1)
        span_score = _clamp(span / max(1.0, height * .42), 0.0, 1.0)
        centres = numpy.asarray([row[1] for row in corridor], dtype=float)
        widths = numpy.asarray([row[2] for row in corridor], dtype=float)
        ys = numpy.asarray([row[0] for row in corridor], dtype=float)
        order = numpy.argsort(ys)
        centres, widths, ys = centres[order], widths[order], ys[order]
        if len(corridor) >= 20:
            normalized_y = (ys - ys.mean()) / max(1.0, float(numpy.ptp(ys)))
            fitted = numpy.polyval(numpy.polyfit(normalized_y, centres, 2), normalized_y)
            fit_error = float(numpy.sqrt(numpy.mean((centres - fitted) ** 2))) / near_half_width
        else:
            fit_error = 1.0
        jumps = numpy.abs(numpy.diff(centres)) / numpy.maximum(8.0, widths[:-1] / 2)
        continuity = math_exp(-3.0 * float(numpy.median(jumps))) if len(jumps) else 0.0
        fit_quality = math_exp(-2.5 * fit_error)
        confidence = _clamp(
            .28 * coverage + .28 * span_score + .22 * continuity + .22 * fit_quality,
            0.0, 1.0,
        )
        if max(abs(offset), abs(heading), abs(bend)) >= 1.98:
            confidence *= .55
        if not contact:
            confidence *= .7

        horizon = top_y / max(1, height - 1)
        horizon_shift = 0.0 if self.previous_horizon is None else horizon - self.previous_horizon
        self.previous_horizon = horizon
        center_rate = 0.0 if self.previous_center is None else offset - self.previous_center
        turn_delta = 0.0 if self.previous_turn is None else bend - self.previous_turn
        self.previous_center, self.previous_turn = offset, bend
        confidence_trend = 0.0 if self.previous_confidence is None else confidence - self.previous_confidence
        self.previous_confidence = confidence
        crest_risk = _clamp((horizon - .24) / .42, 0.0, 1.0)
        profile = []
        for label, fraction in zip(
            ("near", "near_mid", "mid", "far_mid", "far"),
            (.10, .28, .46, .64, .82),
        ):
            row = _sample_corridor(corridor, bottom_y - span * fraction)
            profile.append({
                "depth": label,
                "lateral": round(_clamp((row[1] - anchor_x) / max(8.0, row[2] / 2), -2.0, 2.0), 4),
                "width": round(row[2] / width, 4), "detected": True,
            })
        return {
            "vision_track_offset": float(offset),
            "vision_track_heading": float(heading),
            "vision_bend_ahead": float(bend),
            "vision_bend_severity": float(severity),
            "vision_visible_depth": float(visible_depth),
            "vision_left_gap": float(left_gap), "vision_right_gap": float(right_gap),
            "vision_road_contact": contact,
            "vision_recovery_direction": float(recovery),
            "vision_road_horizon": float(horizon),
            "vision_horizon_shift": float(_clamp(horizon_shift, -1.0, 1.0)),
            "vision_crest_risk": float(crest_risk),
            "vision_confidence": float(confidence),
            "vision_center_rate": float(center_rate),
            "vision_turn_delta": float(turn_delta),
            "vision_profile": profile, "on_track": contact, "tick": self.tick,
        }

    def _road_mask(self, frame, gray):
        # Reuse the screenshot-derived calibration shared with the 2D cone.  The
        # previous dark-neutral-asphalt rule was a frozen-fixture assumption and
        # made valid clay or ice circuits invisible to the default player.
        return super()._road_mask(frame, gray)


    def _horizon_cues(self, frame) -> tuple[float, float]:
        """Camera-only crest/dip cues from the highest detected road row."""
        gray = _grayscale(_surface_from_frame(frame))
        road = self._road_mask(frame, gray)
        height, width = gray.shape
        anchor = width // 2
        rows: list[int] = []
        for y in range(round(height * .22), round(height * .82), 3):
            runs = _runs(road[y])
            if runs:
                left, right = min(runs, key=lambda run: abs(((run[0] + run[1]) / 2) - anchor))
                if right - left >= 8:
                    rows.append(y)
        if not rows:
            return 1.0, 1.0
        horizon = min(rows) / max(1, height - 1)
        near_visible = any(y >= round(height * .70) for y in rows)
        crest_risk = _clamp((horizon - .24) / .42, 0.0, 1.0) if near_visible else 1.0
        return horizon, crest_risk


def _numpy():
    import numpy
    return numpy


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def math_exp(value: float) -> float:
    import math
    return math.exp(value)


def _close_horizontal_gaps(mask, max_gap: int):
    """Fill bounded non-road gaps such as white lane paint, row by row."""
    closed = mask.copy()
    height, _width = closed.shape
    for y in range(height):
        row = closed[y]
        false_runs = _runs(~row)
        for left, right in false_runs:
            if left > 0 and right < len(row) and right - left <= max_gap:
                row[left:right] = True
    return closed


def _trace_corridor(mask, *, anchor_x: int, anchor_y: int) -> list[tuple[int, float, float, float, float]]:
    """Follow one overlapping asphalt component from the ego edge upward."""
    height, width = mask.shape
    previous: tuple[float, float] | None = None
    rows: list[tuple[int, float, float, float, float]] = []
    misses = 0
    for y in range(min(height - 1, anchor_y), max(1, round(height * .16)), -1):
        candidates = [(left, right) for left, right in _runs(mask[y]) if right - left >= 7]
        if previous is None:
            if not candidates:
                continue
            containing = [run for run in candidates if run[0] - 4 <= anchor_x <= run[1] + 4]
            pool = containing or candidates
            left, right = max(
                pool,
                key=lambda run: (run[1] - run[0]) - .7 * abs((run[0] + run[1]) / 2 - anchor_x),
            )
        else:
            prior_left, prior_right = previous
            prior_width = max(8.0, prior_right - prior_left)
            margin = max(10.0, prior_width * .32)
            connected = []
            for left, right in candidates:
                overlap = max(0.0, min(right, prior_right + margin) - max(left, prior_left - margin))
                if overlap > 0:
                    center_gap = abs((left + right - prior_left - prior_right) / 2)
                    score = overlap + .16 * (right - left) - .20 * center_gap
                    connected.append((score, left, right))
            if not connected:
                misses += 1
                if misses > 5:
                    break
                continue
            _score, left, right = max(connected)
        misses = 0
        center, span = (left + right) / 2, float(right - left)
        rows.append((y, center, span, float(left), float(right)))
        previous = (float(left), float(right))
    return rows


def _sample_corridor(
    rows: list[tuple[int, float, float, float, float]], target_y: float,
) -> tuple[int, float, float, float, float]:
    closest = sorted(rows, key=lambda row: abs(row[0] - target_y))[:5]
    closest.sort(key=lambda row: row[0])
    middle = closest[len(closest) // 2]
    # Median geometry rejects a one-row shadow/curb excursion.
    numpy = _numpy()
    return (
        middle[0], float(numpy.median([row[1] for row in closest])),
        float(numpy.median([row[2] for row in closest])),
        float(numpy.median([row[3] for row in closest])),
        float(numpy.median([row[4] for row in closest])),
    )


def _recovery_direction(mask, anchor_x: int, anchor_y: int) -> float:
    numpy = _numpy()
    lower = mask[round(mask.shape[0] * .42):anchor_y]
    _ys, xs = numpy.nonzero(lower)
    if not len(xs):
        return 0.0
    return _clamp(float(numpy.median(xs) - anchor_x) / max(1.0, mask.shape[1] / 2), -1.0, 1.0)

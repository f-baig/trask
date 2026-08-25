"""Audit the 3D RGB road sensor against evaluator-only truth on an oracle lap."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from harness.collision import track_edge_points
from harness.policies import RacingLineController
from harness.racing import (
    CAR_RADIUS, RacingBackend, _angle_delta, _bearing,
    _distance_to_polyline, _nearest_point_index, _signed_lane_offset,
)
from harness.reflex.visual_3d import PerspectiveVisionSense
from harness.store import HarnessStore
from harness.view3d import ViewMode, render_policy_view


DEFAULT_STORE = Path(".harness-data/direct_3d_visual/compact-fixture-20260819T020117Z")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def truth_at(world) -> dict:
    points = world.scene.track_centerline
    index = _nearest_point_index(points, world.player)
    count = len(points)
    lane = _signed_lane_offset(points, world.player, index)
    half_width = max(1.0, world.scene.track_width / 2)
    near_heading = _bearing(points[(index + 1) % count], points[(index + 4) % count])
    far_heading = _bearing(points[(index + 5) % count], points[(index + 10) % count])
    bend_degrees = _angle_delta(near_heading, far_heading)
    track_heading = _bearing(points[(index - 1) % count], points[(index + 2) % count])
    grade, bank = world.surface.attitude_at_index(index)
    distance = _distance_to_polyline(world.player, points, closed=True)
    return {
        "track_index": index,
        # Positive means the road centre is image-right of the car.
        "true_offset": clamp(lane / half_width, -2.0, 2.0),
        "true_heading": clamp(_angle_delta(world.heading, track_heading) / 45.0, -2.0, 2.0),
        "true_bend": clamp(bend_degrees / 45.0, -2.0, 2.0),
        "true_bend_degrees": bend_degrees,
        "true_contact": distance <= half_width - CAR_RADIUS,
        "distance_from_center": distance,
        "grade_degrees": math.degrees(grade),
        "bank_degrees": math.degrees(bank),
        "chassis_pitch_degrees": world.pitch_degrees,
        "chassis_roll_degrees": world.roll_degrees,
    }


def correlation(rows: list[dict], left: str, right: str) -> float | None:
    import numpy

    a = numpy.asarray([row[left] for row in rows], dtype=float)
    b = numpy.asarray([row[right] for row in rows], dtype=float)
    if len(a) < 2 or float(a.std()) < 1e-9 or float(b.std()) < 1e-9:
        return None
    return round(float(numpy.corrcoef(a, b)[0, 1]), 4)


def mean(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / max(1, len(rows))


def sign_accuracy(rows: list[dict], truth: str, sensed: str, threshold: float) -> float | None:
    selected = [row for row in rows if abs(float(row[truth])) >= threshold]
    if not selected:
        return None
    correct = sum(math.copysign(1, row[truth]) == math.copysign(1, row[sensed]) for row in selected)
    return round(correct / len(selected), 4)


def save_worst_frames(samples: list[tuple[float, int, object]], output: Path) -> list[str]:
    paths = []
    for rank, (_score, tick, frame) in enumerate(sorted(samples, reverse=True)[:3], start=1):
        path = output / f"worst-{rank}-tick-{tick}.png"
        path.write_bytes(base64.b64decode(frame.data_base64))
        paths.append(str(path))
    return paths


def plot(scene, rows: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    figure.patch.set_facecolor("#101619")
    for axis in axes.flat:
        axis.set_facecolor("#182124")
        axis.tick_params(colors="#aebabc", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#526164")

    # Where the sensor fails around the circuit.
    axis = axes[0, 0]
    left, right = track_edge_points(scene)
    for edge in (left, right):
        closed = [*edge, edge[0]]
        axis.plot([p[0] for p in closed], [p[1] for p in closed], color="#d9ded8", linewidth=1.2)
    scatter = axis.scatter(
        [row["x"] for row in rows], [row["y"] for row in rows],
        c=[row["combined_error"] for row in rows], cmap="magma", s=13, vmin=0,
    )
    colorbar = figure.colorbar(scatter, ax=axis, fraction=.05, pad=.02)
    colorbar.set_label("combined RGB error", color="#aebabc")
    colorbar.ax.tick_params(colors="#aebabc", labelsize=8)
    axis.set_title("Error location on oracle lap", color="#f4f0e5")
    axis.set_aspect("equal"); axis.invert_yaxis()

    ticks = [row["tick"] for row in rows]
    for axis, truth, sensed, title in (
        (axes[0, 1], "true_offset", "visual_offset", "Road offset"),
        (axes[0, 2], "true_bend", "visual_bend", "Bend ahead"),
    ):
        axis.plot(ticks, [row[truth] for row in rows], color="#77dd9b", linewidth=1.8, label="evaluator truth")
        axis.plot(ticks, [row[sensed] for row in rows], color="#82cfff", linewidth=1.3, alpha=.9, label="RGB sensor")
        axis.axhline(0, color="#748184", linewidth=.7)
        axis.set_title(title, color="#f4f0e5")
        axis.set_xlabel("oracle tick", color="#aebabc")
        axis.set_ylabel("normalized image direction", color="#aebabc")
        axis.legend(facecolor="#101619", edgecolor="#526164", labelcolor="#f4f0e5", fontsize=8)

    axis = axes[1, 0]
    points = axis.scatter(
        [abs(row["bank_degrees"]) for row in rows],
        [row["offset_abs_error"] for row in rows],
        c=[abs(row["grade_degrees"]) for row in rows], cmap="viridis", s=16, alpha=.8,
    )
    bar = figure.colorbar(points, ax=axis, fraction=.05, pad=.02)
    bar.set_label("|grade| (degrees)", color="#aebabc")
    bar.ax.tick_params(colors="#aebabc", labelsize=8)
    axis.set_title("Offset error vs banking", color="#f4f0e5")
    axis.set_xlabel("|bank| (degrees)", color="#aebabc")
    axis.set_ylabel("absolute offset error", color="#aebabc")

    axis = axes[1, 1]
    axis.scatter(
        [row["camera_pitch_degrees"] for row in rows],
        [row["bend_abs_error"] for row in rows],
        c=[abs(row["bank_degrees"]) for row in rows], cmap="plasma", s=16, alpha=.8,
    )
    axis.set_title("Bend error vs camera pitch", color="#f4f0e5")
    axis.set_xlabel("camera pitch (degrees)", color="#aebabc")
    axis.set_ylabel("absolute bend error", color="#aebabc")

    axis = axes[1, 2]
    axis.plot(ticks, [row["visual_confidence"] for row in rows], color="#ffbf59", label="confidence")
    axis.plot(ticks, [row["visual_depth"] for row in rows], color="#bd93f9", label="visible depth")
    axis.step(ticks, [float(row["visual_contact"]) for row in rows], where="mid", color="#82cfff", alpha=.75, label="RGB contact")
    axis.step(ticks, [float(row["true_contact"]) for row in rows], where="mid", color="#77dd9b", alpha=.75, label="true contact")
    axis.set_ylim(-.08, 1.08)
    axis.set_title("Visibility and contact", color="#f4f0e5")
    axis.set_xlabel("oracle tick", color="#aebabc")
    axis.set_ylabel("camera-derived value", color="#aebabc")
    axis.legend(facecolor="#101619", edgecolor="#526164", labelcolor="#f4f0e5", fontsize=8)

    figure.suptitle("3D visual-contract observability audit", color="#f4f0e5", fontsize=18)
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or Path(".harness-data/3d_visual_audit") /
              datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = HarnessStore(args.fixture_store).list_environments()[0]
    scene = environment.scene
    world = RacingBackend().create(scene)
    world.terminate_on_opponent_win = False
    oracle = RacingLineController()
    oracle.reset(scene, scene.seed)
    sensor = PerspectiveVisionSense()
    rows: list[dict] = []
    frame_samples: list[tuple[float, int, object]] = []

    for _ in range(args.max_steps):
        frame = render_policy_view(world, mode=ViewMode.FIRST_PERSON)
        sense = sensor.update(frame)
        truth = truth_at(world)
        if world.countdown_ticks_remaining == 0:
            camera_pitch = frame.camera.pitch_degrees if frame.camera else 0.0
            row = {
                "tick": len(rows), "x": world.player.x, "y": world.player.y,
                **truth,
                "camera_pitch_degrees": camera_pitch,
                "visual_offset": float(sense["vision_track_offset"]),
                "visual_heading": float(sense["vision_track_heading"]),
                "visual_bend": float(sense["vision_bend_ahead"]),
                "visual_contact": bool(sense["vision_road_contact"]),
                "visual_confidence": float(sense["vision_confidence"]),
                "visual_depth": float(sense["vision_visible_depth"]),
                "visual_crest_risk": float(sense["vision_crest_risk"]),
                "abs_bank": abs(truth["bank_degrees"]),
                "abs_grade": abs(truth["grade_degrees"]),
            }
            row["offset_abs_error"] = abs(row["visual_offset"] - row["true_offset"])
            row["heading_abs_error"] = abs(row["visual_heading"] - row["true_heading"])
            row["bend_abs_error"] = abs(row["visual_bend"] - row["true_bend"])
            row["combined_error"] = (
                row["offset_abs_error"] + row["bend_abs_error"]
                + (0.75 if row["visual_contact"] != row["true_contact"] else 0.0)
            )
            rows.append(row)
            frame_samples.append((row["combined_error"], row["tick"], frame))
        action, decision = oracle.act(world.observe())
        world.step(action, decision)
        if world.terminated:
            break

    if not rows:
        raise SystemExit("Oracle produced no active driving frames")
    csv_path = output / "sensor-vs-truth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plot_path = output / "visual-contract-audit.png"
    plot(scene, rows, plot_path)
    worst_frames = save_worst_frames(frame_samples, output)
    true_contact_ticks = sum(bool(row["true_contact"]) for row in rows)
    false_contact_losses = sum(row["true_contact"] and not row["visual_contact"] for row in rows)
    summary = {
        "created_at": datetime.now(UTC).isoformat(), "environment_id": environment.id,
        "track": scene.name, "seed": scene.seed, "oracle_completed": world.succeeded,
        "oracle_reason": world.reason, "oracle_active_ticks": len(rows),
        "agent_contract_audited": "first-person RGB sensor; evaluator truth never enters player input",
        "metrics": {
            "offset_mae": round(mean(rows, "offset_abs_error"), 4),
            "heading_mae": round(mean(rows, "heading_abs_error"), 4),
            "bend_mae": round(mean(rows, "bend_abs_error"), 4),
            "offset_correlation": correlation(rows, "true_offset", "visual_offset"),
            "heading_correlation": correlation(rows, "true_heading", "visual_heading"),
            "bend_correlation": correlation(rows, "true_bend", "visual_bend"),
            "offset_sign_accuracy": sign_accuracy(rows, "true_offset", "visual_offset", .08),
            "bend_sign_accuracy": sign_accuracy(rows, "true_bend", "visual_bend", .12),
            "road_contact_recall": round(
                (true_contact_ticks - false_contact_losses) / max(1, true_contact_ticks), 4,
            ),
            "false_contact_loss_ticks": false_contact_losses,
            "mean_visual_confidence": round(mean(rows, "visual_confidence"), 4),
            "offset_error_correlation_abs_bank": correlation(rows, "offset_abs_error", "abs_bank"),
            "bend_error_correlation_abs_grade": correlation(rows, "bend_abs_error", "abs_grade"),
            "bend_error_correlation_camera_pitch": correlation(rows, "bend_abs_error", "camera_pitch_degrees"),
        },
        "plot": str(plot_path), "data": str(csv_path), "worst_frames": worst_frames,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Run matched 2D-cone and flat-3D player trials with speed-only telemetry."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from harness.models import ElevationProfile, ElevationSpec, TrackRegion
from harness.racing import RacingWorld, compile_racing_scene
from harness.racing3d import Racing3DWorld
from harness.reflex.episode import run_reflex_episode
from harness.track_grammar import CornerSpec, StraightLength, TrackPlan


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def rectangle_scene(seed: int):
    regions = (
        TrackRegion.TOP_RIGHT, TrackRegion.BOTTOM_RIGHT,
        TrackRegion.BOTTOM_LEFT, TrackRegion.TOP_LEFT,
    )
    plan = TrackPlan(
        title="Four-corner rectangle",
        rationale="A controlled flat benchmark with four matched right-angle corners.",
        direction="clockwise", surface="asphalt", grip=1.0, track_width=152,
        laps=1, barriers=[], npcs=[],
        corners=[
            CornerSpec(
                direction="right", angle_degrees=90, region=region,
                exit_straight=StraightLength.LONG, label=f"corner-{index}",
            )
            for index, region in enumerate(regions, start=1)
        ],
    )
    scene = compile_racing_scene("Matched four-corner rectangle", plan, seed=seed)
    return scene.model_copy(update={
        "elevation": ElevationSpec(
            profile=ElevationProfile.FLAT, amplitude_m=0, banking_degrees=0,
        ),
    })


def trace(report) -> list[dict]:
    return [
        {
            "step": frame.step,
            "x": round(frame.privileged_state.player.x, 4),
            "y": round(frame.privileged_state.player.y, 4),
            "speed": round(frame.privileged_state.speed, 4),
        }
        for frame in report.frames
        if frame.privileged_state.countdown_ticks_remaining == 0
    ]


def chart(scene, rows: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    center_x = [point.x for point in scene.track_centerline] + [scene.track_centerline[0].x]
    center_y = [point.y for point in scene.track_centerline] + [scene.track_centerline[0].y]
    for axis, row in zip(axes, rows):
        axis.plot(center_x, center_y, color="#b8c1cc", linewidth=10, alpha=.5, label="track center")
        axis.plot([p["x"] for p in row["trace"]], [p["y"] for p in row["trace"]], color="#e45756", linewidth=2, label="agent path")
        axis.scatter([row["trace"][0]["x"]], [row["trace"][0]["y"]], color="#2ca02c", s=35, label="start")
        axis.set(title=f"{row['dimension']} — {row['status']}", aspect="equal", xlabel="world x", ylabel="world y")
        axis.invert_yaxis()
        axis.legend(fontsize=8)
    figure.suptitle("Speed-only player paths on the matched four-corner rectangle")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--max-wakes", type=int, default=6)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    output = (args.output_dir or Path(".harness-data") / "speed_awareness" /
              datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = rectangle_scene(args.seed)
    rows = []
    for dimension, world, view in (
        ("2D cone", RacingWorld.from_scene(scene), "2d"),
        ("3D first-person flat", Racing3DWorld.from_scene(scene), "3d"),
    ):
        print(f"[{dimension}] {args.player_model} (reasoning effort: none)", flush=True)
        report = run_reflex_episode(
            world, model=args.player_model, max_steps=args.max_steps,
            max_wakes=args.max_wakes, vision_only=True, visual_mode=view,
            verbose=True,
        )
        row = {
            "dimension": dimension, "view": view, "status": "completed" if report.succeeded else "not_completed",
            "reason": report.reason, "steps": report.ticks, "model_calls": report.usage.calls,
            "input_tokens": report.usage.input_tokens, "output_tokens": report.usage.output_tokens,
            "total_tokens": report.usage.input_tokens + report.usage.output_tokens,
            "trace": trace(report),
        }
        rows.append(row)
        print({key: value for key, value in row.items() if key != "trace"}, flush=True)

    chart_path = output / "top_down_paths.png"
    chart(scene, rows, chart_path)
    payload = {
        "created_at": datetime.now(UTC).isoformat(), "model": args.player_model,
        "reasoning_effort": "none", "seed": args.seed,
        "track": "four 90-degree corners; asphalt; flat; no barriers; no opponents",
        "agent_contract": "view pixels plus physical speed; no other engine information",
        "runs": rows, "chart": str(chart_path),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output / 'summary.json'}", flush=True)
    print(f"Chart: {chart_path}", flush=True)


if __name__ == "__main__":
    main()

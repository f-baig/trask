"""Render write-up figures from a completed 3D environment-generation study."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from render_generation_ab_writeup import cost_chart, paired_conjunction, quality_chart  # noqa: E402
from run_generation_ab import aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/environment-generation-comparison-3d/writeup"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.study.read_text(encoding="utf-8"))
    summary.update({
        "created_at": datetime.now(UTC).isoformat(),
        "source_summary": str(args.study.resolve()),
        "aggregate": aggregate(summary["rows"]),
        "statistics": paired_conjunction(summary["rows"], "harness", "oneshot"),
    })
    quality = args.output_dir / "environment_generation_quality_3d.png"
    cost = args.output_dir / "environment_generation_cost_3d.png"
    quality_chart(summary, quality, dimension_label="3D")
    cost_chart(summary, cost, dimension_label="3D")
    summary["quality_chart"] = str(quality.resolve())
    summary["cost_chart"] = str(cost.resolve())
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(output.resolve()), "quality": str(quality.resolve()),
        "cost": str(cost.resolve()), "headline": summary["aggregate"]["by_arm"],
        "statistics": summary["statistics"],
    }, indent=2))


if __name__ == "__main__":
    main()

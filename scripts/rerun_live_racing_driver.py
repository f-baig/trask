#!/usr/bin/env python3
"""Re-evaluate the Claude driver on environments from a prior live benchmark."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from harness.models import RunRequest
from harness.service import HarnessService
from harness.store import HarnessStore

from run_live_racing_benchmark import aggregate_runs, load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--label", default="control-v2")
    parser.add_argument("--model", help="Override the player model recorded in the source benchmark.")
    parser.add_argument("--indices", type=int, nargs="+", help="Run only these zero-based environment rows.")
    parser.add_argument("--max-steps", type=int, default=1_200, help="Episode step budget for each rerun.")
    args = parser.parse_args()
    load_dotenv(Path(".env"))
    baseline = json.loads(args.summary.read_text(encoding="utf-8"))
    model = args.model or baseline["player_model"]
    os.environ["ANTHROPIC_PLAYER_MODEL"] = model
    output_dir = Path(baseline["output_dir"])
    service = HarnessService(store=HarnessStore(output_dir))
    rows = []
    environments = baseline["environments"]
    if args.indices is not None:
        environments = [environments[index] for index in args.indices]
    for item in environments:
        print(f"[telemetry-direct v2] {item['scenario']} seed={item['seed']}", flush=True)
        try:
            run = service.run(RunRequest(environment_id=item["environment_id"], policy_name="telemetry-direct", max_steps=args.max_steps))
            row = {
                "scenario": item["scenario"], "seed": item["seed"], "environment_id": item["environment_id"],
                "run_id": run.id, "policy": "telemetry-direct", "status": run.status.value,
                "reason": run.result_reason, "steps": len(run.frames), "reward": run.total_reward,
                "input_tokens": run.input_tokens, "output_tokens": run.output_tokens,
                "uncached_input_tokens": run.uncached_input_tokens,
                "cache_creation_input_tokens": run.cache_creation_input_tokens,
                "cache_read_input_tokens": run.cache_read_input_tokens,
                "player_turns": run.player_turns, "model_latency_ms": max(0, run.latency_ms - len(run.frames) * 4),
                "replay_uri": run.artifacts[0].uri if run.artifacts else None,
            }
            rows.append(row)
            print(f"  {run.status.value}: {run.result_reason} ({len(run.frames)} steps)", flush=True)
        except Exception as error:
            rows.append({"scenario": item["scenario"], "seed": item["seed"], "environment_id": item["environment_id"], "policy": "telemetry-direct", "error": str(error)})
            print(f"  error: {error}", flush=True)
    result = {
        "created_at": datetime.now(UTC).isoformat(), "label": args.label, "model": model,
        "baseline_summary": str(args.summary.resolve()), "runs": rows,
        "aggregate": aggregate_runs(rows).get("telemetry-direct", {}),
    }
    target = output_dir / f"{args.label}-summary.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    aggregate = result["aggregate"]
    reasons = Counter(row.get("reason") for row in rows if row.get("status") != "succeeded" and row.get("reason"))
    report = [
        f"# RaceLab Claude driver rerun · {args.label}", "",
        f"Exact environments from [{args.summary.name}]({args.summary.resolve()}) using `{model}`.", "",
        f"- Success: **{aggregate.get('successes', 0)} / {aggregate.get('completed', 0)} ({aggregate.get('success_rate', 0):.1%})**",
        f"- Mean steps: **{aggregate.get('mean_steps', 0)}**",
        f"- Model turns: **{aggregate.get('model_turns', 0)}**",
        f"- Tokens: **{aggregate.get('input_tokens', 0)} input / {aggregate.get('output_tokens', 0)} output**", "",
        f"- Input billing split: **{aggregate.get('uncached_input_tokens', 0)} uncached / {aggregate.get('cache_creation_input_tokens', 0)} cache writes / {aggregate.get('cache_read_input_tokens', 0)} cache reads**", "",
        "| Scenario | Seed | Status | Steps | Reason |", "|---|---:|---|---:|---|",
    ]
    report.extend(f"| {row['scenario']} | {row['seed']} | {row.get('status', 'error')} | {row.get('steps', 0)} | {row.get('reason', row.get('error', ''))} |" for row in rows)
    report.extend(["", "## Failure reasons", ""])
    report.extend(f"- `{reason}`: {count}" for reason, count in reasons.items())
    report_path = output_dir / f"{args.label.upper().replace('-', '_')}_REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Rerun complete: {report_path}", flush=True)


if __name__ == "__main__":
    main()

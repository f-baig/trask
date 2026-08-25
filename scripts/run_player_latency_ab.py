#!/usr/bin/env python3
"""Benchmark direct Claude control against one-call sector planning on matched races."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from harness.models import RunRequest
from harness.service import HarnessService
from harness.store import HarnessStore

from run_live_racing_benchmark import load_dotenv


VARIANTS = {
    "without": "telemetry-direct",
    "with": "telemetry-strategy",
}


def percentile(values: list[int | float], quantile: float) -> float | None:
    """Return a linearly interpolated sample percentile (same convention as NumPy)."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[int | float]) -> dict:
    return {
        "n": len(values),
        "p50": round(percentile(values, 0.50), 1) if values else None,
        "p95": round(percentile(values, 0.95), 1) if values else None,
        "p99": round(percentile(values, 0.99), 1) if values else None,
    }


def call_latencies(run) -> list[int]:
    return [
        frame.decision.provider_usage.latency_ms
        for frame in run.frames
        if frame.decision is not None and frame.decision.provider_usage is not None
    ]


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for label in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == label and row.get("status")]
        by_outcome = {}
        for outcome in ("success", "failure"):
            selected = [row for row in variant_rows if row["outcome"] == outcome]
            by_outcome[outcome] = {
                "runs": len(selected),
                "calls": sum(row["calls"] for row in selected),
                "call_latency_ms": distribution([latency for row in selected for latency in row["call_latencies_ms"]]),
                "run_model_latency_ms": distribution([row["model_latency_ms"] for row in selected]),
                "run_wall_latency_ms": distribution([row["wall_latency_ms"] for row in selected]),
                "calls_per_run": distribution([row["calls"] for row in selected]),
            }
        summary[label] = {
            "policy": VARIANTS[label],
            "completed": len(variant_rows),
            "successes": sum(row["outcome"] == "success" for row in variant_rows),
            "success_rate": round(sum(row["outcome"] == "success" for row in variant_rows) / len(variant_rows), 4) if variant_rows else None,
            "outcomes": by_outcome,
            "failure_reasons": dict(Counter(row["reason"] for row in variant_rows if row["outcome"] == "failure")),
        }
    return summary


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:,.1f}"


def report(payload: dict) -> str:
    lines = [
        "# RaceLab player latency A/B",
        "",
        f"Measured at `{payload['created_at']}` with `{payload['model']}` on the same {len(payload['environments'])} generated circuits.",
        "",
        "- **Without optimization:** direct Claude receding-horizon control, up to 40 calls per race.",
        "- **With optimization:** one Claude 12-sector strategy call, then deterministic intent control.",
        f"- Conditions: {', '.join(f'`{item}`' for item in payload['perturbations'])}. Percentiles use linear interpolation.",
        "",
        "## Outcome and individual model-call latency",
        "",
        "| Variant | Outcome | Runs | Calls | p50 ms | p95 ms | p99 ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in VARIANTS:
        item = payload["summary"][label]
        for outcome in ("success", "failure"):
            group = item["outcomes"][outcome]
            latency = group["call_latency_ms"]
            lines.append(
                f"| {label} | {outcome} | {group['runs']} | {group['calls']} | "
                f"{fmt(latency['p50'])} | {fmt(latency['p95'])} | {fmt(latency['p99'])} |"
            )
    lines.extend([
        "",
        "## Cumulative model latency per race",
        "",
        "| Variant | Outcome | Runs | p50 ms | p95 ms | p99 ms |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for label in VARIANTS:
        for outcome in ("success", "failure"):
            group = payload["summary"][label]["outcomes"][outcome]
            latency = group["run_model_latency_ms"]
            lines.append(
                f"| {label} | {outcome} | {group['runs']} | {fmt(latency['p50'])} | "
                f"{fmt(latency['p95'])} | {fmt(latency['p99'])} |"
            )
    lines.extend([
        "",
        "## End-to-end wall latency per race",
        "",
        "| Variant | Outcome | Runs | p50 ms | p95 ms | p99 ms |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for label in VARIANTS:
        for outcome in ("success", "failure"):
            group = payload["summary"][label]["outcomes"][outcome]
            latency = group["run_wall_latency_ms"]
            lines.append(
                f"| {label} | {outcome} | {group['runs']} | {fmt(latency['p50'])} | "
                f"{fmt(latency['p95'])} | {fmt(latency['p99'])} |"
            )
    lines.extend([
        "",
        "## Run matrix",
        "",
        "| Circuit | Seed | Condition | Variant | Status | Calls | Model ms | Wall ms | Reason |",
        "|---|---:|---|---|---|---:|---:|---:|---|",
    ])
    for row in payload["runs"]:
        lines.append(
            f"| {row['scenario']} | {row['seed']} | {row['perturbation']} | {row['variant']} | "
            f"{row.get('status', 'error')} | {row.get('calls', 0)} | {row.get('model_latency_ms', 0)} | "
            f"{row.get('wall_latency_ms', 0)} | {row.get('reason', row.get('error', ''))} |"
        )
    lines.extend(["", f"Raw measurements: `{payload['output_dir']}/summary.json`.", ""])
    return "\n".join(lines)


def checkpoint(path: Path, payload: dict) -> None:
    payload["summary"] = summarize(payload["runs"])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (path.parent / "REPORT.md").write_text(report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--perturbations", nargs="+", choices=["normal", "action_delay"], default=["normal", "action_delay"])
    parser.add_argument("--limit", type=int, help="Use only the first N source environments for a pilot.")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required for the live A/B benchmark")
    source_payload = json.loads(args.source_summary.read_text(encoding="utf-8"))
    model = args.model or source_payload["player_model"]
    os.environ["ANTHROPIC_PLAYER_MODEL"] = model
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path(".harness-data") / "latency-benchmarks" / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    environments = source_payload["environments"][: args.limit]
    source_store = HarnessStore(Path(source_payload["output_dir"]))
    target_store = HarnessStore(output_dir)
    for item in environments:
        environment = source_store.get_environment(item["environment_id"])
        if environment is None:
            raise SystemExit(f"Source environment not found: {item['environment_id']}")
        target_store.save_environment(environment)
    service = HarnessService(store=target_store)

    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "output_dir": str(output_dir),
            "source_summary": str(args.source_summary.resolve()),
            "model": model,
            "perturbations": args.perturbations,
            "environments": environments,
            "runs": [],
            "summary": {},
        }
    completed_keys = {
        (row["environment_id"], row["perturbation"], row["variant"])
        for row in payload["runs"]
        if row.get("status")
    }
    total = len(environments) * len(args.perturbations) * len(VARIANTS)
    print(f"RaceLab latency A/B: {total} matched runs in {output_dir}", flush=True)
    for environment in environments:
        for perturbation in args.perturbations:
            for label, policy in VARIANTS.items():
                key = (environment["environment_id"], perturbation, label)
                if key in completed_keys:
                    continue
                print(f"[{label}] {environment['scenario']} seed={environment['seed']} condition={perturbation}", flush=True)
                try:
                    run = service.run(RunRequest(
                        environment_id=environment["environment_id"],
                        policy_name=policy,
                        max_steps=1_200,
                    ), perturbation=None if perturbation == "normal" else perturbation)
                    latencies = call_latencies(run)
                    wall = run.execution.resource_usage.wall_time_ms if run.execution.resource_usage else run.latency_ms
                    row = {
                        "scenario": environment["scenario"],
                        "seed": environment["seed"],
                        "environment_id": environment["environment_id"],
                        "perturbation": perturbation,
                        "variant": label,
                        "policy": policy,
                        "run_id": run.id,
                        "status": run.status.value,
                        "outcome": "success" if run.status.value == "succeeded" else "failure",
                        "reason": run.result_reason,
                        "steps": len(run.frames),
                        "calls": len(latencies),
                        "call_latencies_ms": latencies,
                        "model_latency_ms": sum(latencies),
                        "wall_latency_ms": wall,
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                        "uncached_input_tokens": run.uncached_input_tokens,
                        "cache_creation_input_tokens": run.cache_creation_input_tokens,
                        "cache_read_input_tokens": run.cache_read_input_tokens,
                    }
                    print(f"  {row['status']}: calls={row['calls']} model={row['model_latency_ms']}ms wall={wall}ms", flush=True)
                except Exception as error:
                    row = {
                        "scenario": environment["scenario"], "seed": environment["seed"],
                        "environment_id": environment["environment_id"], "perturbation": perturbation,
                        "variant": label, "policy": policy, "error": str(error), "outcome": "error",
                    }
                    print(f"  error: {error}", flush=True)
                # A resumed provider/transport failure should be replaced by
                # its retry, not retained as a duplicate matrix observation.
                payload["runs"] = [
                    existing for existing in payload["runs"]
                    if (
                        existing["environment_id"], existing["perturbation"], existing["variant"]
                    ) != key or existing.get("status")
                ]
                payload["runs"].append(row)
                completed_keys.add(key)
                checkpoint(summary_path, payload)
    checkpoint(summary_path, payload)
    print(f"Benchmark complete: {output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the model-backed RaceLab creator and driver across a compact seed matrix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from harness.models import RunRequest
from harness.service import HarnessService
from harness.store import HarnessStore


# Each brief exercises a different axis of the corner grammar: bare geometry, a
# located corner, opponent temperament, continuous grip, and a chicane pair.
SCENARIOS = (
    ("oval-asphalt", "Build a fast oval asphalt circuit with no barriers and one opponent."),
    ("technical-clay", "Build a technical clay circuit with two barriers and two opponents."),
    ("chicane-ice", "Build a chicane ice circuit with four barriers and three opponents."),
    ("located-corner", "A curvy circuit with a 90 degree bend in the top right and a long back straight."),
    ("aggressive-traffic", "A flowing asphalt circuit with three aggressive opponents and no barriers."),
    ("slippery-hairpin", "A slippery low-grip circuit with a hairpin in the bottom left and two cautious rivals."),
)
POLICIES = ("oracle-racing-line", "baseline-random", "telemetry-direct")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def aggregate_runs(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    results = {}
    for policy, items in grouped.items():
        completed = [item for item in items if item.get("status")]
        successes = [item for item in completed if item["status"] == "succeeded"]
        results[policy] = {
            "attempted": len(items),
            "completed": len(completed),
            "successes": len(successes),
            "success_rate": round(len(successes) / len(completed), 3) if completed else 0,
            "mean_steps": round(statistics.mean(item["steps"] for item in completed), 1) if completed else 0,
            "input_tokens": sum(item.get("input_tokens", 0) for item in completed),
            "output_tokens": sum(item.get("output_tokens", 0) for item in completed),
            "uncached_input_tokens": sum(item.get("uncached_input_tokens", 0) for item in completed),
            "cache_creation_input_tokens": sum(item.get("cache_creation_input_tokens", 0) for item in completed),
            "cache_read_input_tokens": sum(item.get("cache_read_input_tokens", 0) for item in completed),
            "model_turns": sum(item.get("player_turns", 0) for item in completed),
            "model_latency_ms": sum(item.get("model_latency_ms", 0) for item in completed),
            "failure_reasons": dict(Counter(item.get("reason", "unknown") for item in completed if item["status"] != "succeeded")),
        }
    return results


def markdown_report(summary: dict) -> str:
    lines = [
        "# RaceLab live model benchmark",
        "",
        f"Run at `{summary['created_at']}` using `{summary['environment_model']}` for circuit creation and `{summary['player_model']}` for driving.",
        "",
        "## Outcome",
        "",
        f"- Environment proposals accepted: **{summary['environment_generation']['accepted']} / {summary['environment_generation']['attempted']}**",
        f"- Generator tokens: **{summary['environment_generation']['input_tokens']} input / {summary['environment_generation']['output_tokens']} output**",
        "",
        "| Policy | Success | Mean steps | Model turns | Tokens (in/out) |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        item = summary["runs_by_policy"].get(policy, {})
        lines.append(
            f"| {policy} | {item.get('successes', 0)}/{item.get('completed', 0)} ({item.get('success_rate', 0):.1%}) "
            f"| {item.get('mean_steps', 0)} | {item.get('model_turns', 0)} | {item.get('input_tokens', 0)}/{item.get('output_tokens', 0)} |"
        )
    claude_usage = summary["runs_by_policy"].get("telemetry-direct", {})
    lines.extend([
        "",
        "Claude input billing split: "
        f"{claude_usage.get('uncached_input_tokens', 0)} uncached / "
        f"{claude_usage.get('cache_creation_input_tokens', 0)} cache writes / "
        f"{claude_usage.get('cache_read_input_tokens', 0)} cache reads.",
    ])
    lines.extend(["", "## Circuit results", "", "| Scenario | Seed | Generated circuit | Surface | Barriers | NPCs | Oracle | Random | Claude |", "|---|---:|---|---|---:|---:|---|---|---|"])
    run_lookup = {(row["environment_id"], row["policy"]): row for row in summary["runs"]}
    for environment in summary["environments"]:
        cells = []
        for policy in POLICIES:
            result = run_lookup.get((environment["environment_id"], policy), {})
            cells.append(result.get("status", f"error: {result.get('error', 'not run')[:32]}"))
        lines.append(
            f"| {environment['scenario']} | {environment['seed']} | {environment['circuit']} | {environment['surface']} | "
            f"{environment['barriers']} | {environment['npcs']} | {' | '.join(cells)} |"
        )
    failures = summary["runs_by_policy"].get("telemetry-direct", {}).get("failure_reasons", {})
    lines.extend(["", "## Claude-driver failure reasons", ""])
    if failures:
        lines.extend(f"- `{reason}`: {count}" for reason, count in failures.items())
    else:
        lines.append("- None")
    lines.extend(["", f"Raw results and replay artifacts are stored under `{summary['output_dir']}`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 43, 89])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required for the live benchmark")
    os.environ["ANTHROPIC_ENVIRONMENT_MODEL"] = args.model
    os.environ["ANTHROPIC_PLAYER_MODEL"] = args.model
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path(".harness-data") / "live-benchmarks" / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    service = HarnessService(store=HarnessStore(output_dir))

    environment_rows: list[dict] = []
    run_rows: list[dict] = []
    generation_errors: list[dict] = []
    attempted = len(SCENARIOS) * len(args.seeds)
    print(f"RaceLab live benchmark: {attempted} environments, {attempted * len(POLICIES)} runs", flush=True)

    for scenario, prompt in SCENARIOS:
        for seed in args.seeds:
            print(f"[creator] {scenario} seed={seed}", flush=True)
            try:
                environment = service.create_environment(prompt, seed=seed, provider="anthropic", origin="live multi-seed benchmark")
            except Exception as error:  # retain raw-provider failures without aborting the matrix
                generation_errors.append({"scenario": scenario, "seed": seed, "error": str(error)})
                print(f"  rejected: {error}", flush=True)
                continue
            counts = Counter(entity.kind.value for entity in environment.scene.entities)
            environment_rows.append({
                "scenario": scenario,
                "seed": seed,
                "environment_id": environment.id,
                "circuit": environment.scene.name,
                "surface": environment.scene.surface,
                "barriers": counts["obstacle"],
                "npcs": counts["npc"],
                "generator_model": environment.generator_model,
                "generator_input_tokens": environment.generator_input_tokens,
                "generator_output_tokens": environment.generator_output_tokens,
                "generator_latency_ms": environment.generator_latency_ms,
                "certificate_steps": environment.playability_certificate.route_steps if environment.playability_certificate else None,
            })
            for policy in POLICIES:
                print(f"  [driver] {policy}", flush=True)
                try:
                    run = service.run(RunRequest(environment_id=environment.id, policy_name=policy, max_steps=1_200))
                    provider_latency = max(0, run.latency_ms - len(run.frames) * 4)
                    run_rows.append({
                        "scenario": scenario,
                        "seed": seed,
                        "environment_id": environment.id,
                        "run_id": run.id,
                        "policy": policy,
                        "status": run.status.value,
                        "reason": run.result_reason,
                        "steps": len(run.frames),
                        "reward": run.total_reward,
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                        "uncached_input_tokens": run.uncached_input_tokens,
                        "cache_creation_input_tokens": run.cache_creation_input_tokens,
                        "cache_read_input_tokens": run.cache_read_input_tokens,
                        "player_turns": run.player_turns,
                        "model_latency_ms": provider_latency,
                        "replay_uri": run.artifacts[0].uri if run.artifacts else None,
                    })
                    print(f"    {run.status.value}: {run.result_reason} ({len(run.frames)} steps)", flush=True)
                except Exception as error:
                    run_rows.append({"scenario": scenario, "seed": seed, "environment_id": environment.id, "policy": policy, "error": str(error)})
                    print(f"    error: {error}", flush=True)

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "environment_model": args.model,
        "player_model": args.model,
        "seeds": args.seeds,
        "scenarios": [scenario for scenario, _ in SCENARIOS],
        "environment_generation": {
            "attempted": attempted,
            "accepted": len(environment_rows),
            "rejected": len(generation_errors),
            "input_tokens": sum(row["generator_input_tokens"] for row in environment_rows),
            "output_tokens": sum(row["generator_output_tokens"] for row in environment_rows),
            "latency_ms": sum(row["generator_latency_ms"] for row in environment_rows),
            "errors": generation_errors,
        },
        "environments": environment_rows,
        "runs": run_rows,
        "runs_by_policy": aggregate_runs(run_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    print(f"Benchmark complete: {output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()

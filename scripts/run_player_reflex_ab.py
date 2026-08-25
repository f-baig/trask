"""Run two shared GPT-generated tracks with direct and reflex GPT players.

The matrix is intentionally small and inspectable: two environment seeds, each
driven once by `telemetry-direct` and once by `telemetry-reflex`.  The output is
four rows, a Markdown report, durable replays, and a Matplotlib PNG that makes
completion, active simulator steps, calls, and token use visible together.

Example:
    PYTHONPATH=backend:scripts .venv/bin/python scripts/run_player_reflex_ab.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from harness.models import RunRequest, RunStatus
from harness.providers import ProviderError
from harness.service import HarnessService
from harness.store import HarnessStore


TRACK_BRIEFS = (
    "A smooth, playable one-lap asphalt circuit with flowing 60 to 90 degree corners, no barriers, and no opponents.",
    "A technical but playable one-lap asphalt circuit with two 90 degree corners, no barriers, and no opponents.",
)
ARMS = {"direct": "telemetry-direct", "reflex": "telemetry-reflex"}


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load missing local values without printing credentials or overwriting shell config."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def active_steps(run) -> int:
    """Count controlled race ticks, excluding the frozen countdown frames."""
    return sum(
        frame.privileged_state.countdown_ticks_remaining == 0
        for frame in run.frames
    )


def run_row(service: HarnessService, environment, *, label: str, policy: str, max_steps: int, decision_budget: int) -> dict:
    arm = next(name for name, value in ARMS.items() if value == policy)
    try:
        run = service.run(RunRequest(
            environment_id=environment.id,
            policy_name=policy,
            max_steps=max_steps,
            policy_decision_budget=decision_budget,
        ), study_name="GPT direct vs reflex player A/B")
    except ProviderError as error:
        # An API timeout is a real experimental outcome.  Keep it in the matrix
        # and continue the other arm/track rather than discarding their evidence.
        return {
            "track": label, "seed": environment.scene.seed, "environment_id": environment.id,
            "run_id": None, "arm": arm, "policy": policy, "completed": False,
            "status": "provider_error", "reason": str(error), "active_steps": 0,
            "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "latency_ms": 0, "replay": None,
        }
    return {
        "track": label,
        "seed": environment.scene.seed,
        "environment_id": environment.id,
        "run_id": run.id,
        "arm": arm,
        "policy": policy,
        "completed": run.status == RunStatus.SUCCEEDED,
        "status": run.status.value,
        "reason": run.result_reason,
        "active_steps": active_steps(run),
        "model_calls": run.player_turns or 0,
        "input_tokens": run.input_tokens or 0,
        "output_tokens": run.output_tokens or 0,
        "total_tokens": (run.input_tokens or 0) + (run.output_tokens or 0),
        "latency_ms": run.latency_ms,
        "replay": run.artifacts[0].uri if run.artifacts else None,
    }


def write_chart(rows: list[dict], path: Path, *, player_model: str, environment_model: str) -> None:
    """Write a compact, headless-safe figure for the four requested runs."""
    # Desktop sandboxes often make the normal user cache read-only. Keep the
    # Matplotlib cache beside the benchmark artifact instead of emitting noisy
    # warnings or relying on an unwritable home directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['track']}\n{row['arm']}" for row in rows]
    colors = ["#4477AA" if row["arm"] == "direct" else "#44AA99" for row in rows]
    hatches = ["" if row["completed"] else "//" for row in rows]
    completion = [1 if row["completed"] else 0 for row in rows]
    steps = [row["active_steps"] for row in rows]
    calls = [row["model_calls"] for row in rows]
    tokens = [row["total_tokens"] for row in rows]
    x = list(range(len(rows)))

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    bars = axes[0].bar(x, steps, color=colors, edgecolor="#1f2937")
    for bar, hatch, row in zip(bars, hatches, rows):
        bar.set_hatch(hatch)
        axes[0].annotate(
            "completed" if row["completed"] else "not completed",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4),
            textcoords="offset points", ha="center", va="bottom", fontsize=8, rotation=90,
        )
    axes[0].set_title("Active steps to outcome")
    axes[0].set_ylabel("simulator ticks (countdown excluded)")
    axes[0].set_xticks(x, labels, fontsize=8)

    axes[1].bar(x, calls, color=colors, edgecolor="#1f2937")
    axes[1].set_title("Model calls")
    axes[1].set_ylabel("provider calls")
    axes[1].set_xticks(x, labels, fontsize=8)

    axes[2].bar(x, tokens, color=colors, edgecolor="#1f2937")
    axes[2].set_title("Total model tokens")
    axes[2].set_ylabel("input + output tokens")
    axes[2].set_xticks(x, labels, fontsize=8)
    axes[2].ticklabel_format(axis="y", style="plain")

    figure.suptitle(
        f"GPT player A/B — player: {player_model}; environment: {environment_model}\n"
        "Blue = direct controls; green = reflex controller; hatched = did not complete",
        fontsize=11,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_report(payload: dict) -> str:
    lines = [
        "# GPT direct vs reflex player A/B", "",
        f"Player model: `{payload['player_model']}`  ",
        f"Environment model: `{payload['environment_model']}`  ",
        "Each track is generated once, then both arms drive that exact certified scene.", "",
        "| track | arm | completion | active steps | model calls | input tokens | output tokens | reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["runs"]:
        outcome = "completed" if row["completed"] else "not completed"
        lines.append(
            f"| {row['track']} | {row['arm']} | {outcome} | {row['active_steps']} | "
            f"{row['model_calls']} | {row['input_tokens']:,} | {row['output_tokens']:,} | {row['reason']} |"
        )
    lines += ["", f"Chart: `{payload['chart_file']}`", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-model", default="gpt-5-nano")
    parser.add_argument("--environment-model", default="gpt-5-mini")
    parser.add_argument("--seeds", type=int, nargs=2, default=[17, 43])
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--decision-budget", type=int, default=80)
    parser.add_argument(
        "--request-timeout", type=int, default=90,
        help="maximum seconds for one OpenAI request before it is retried once",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required: this benchmark will not substitute another provider.")
    # These settings are deliberately process-local and cover the creator's
    # contract reader plus any fidelity checks, not merely the final plan call.
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    os.environ["RACING_ENVIRONMENT_MODEL"] = args.environment_model
    os.environ["RACING_COMPREHENSION_MODEL"] = args.environment_model
    os.environ["RACING_FIDELITY_MODEL"] = args.environment_model
    os.environ["OPENAI_REQUEST_TIMEOUT_SECONDS"] = str(args.request_timeout)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or Path(".harness-data") / "player_reflex_ab" / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    service = HarnessService(store=HarnessStore(output_dir))
    rows: list[dict] = []
    environments: list[dict] = []

    for index, (prompt, seed) in enumerate(zip(TRACK_BRIEFS, args.seeds), start=1):
        label = f"track-{index}"
        print(f"[environment] {label} seed={seed}", flush=True)
        # `anthropic` is the repository's historical provider selector. The
        # model id routes the request through its OpenAI transport.
        environment = service.create_environment(
            prompt, seed=seed, provider="anthropic", origin="GPT player A/B fixture",
        )
        environments.append({"label": label, "seed": seed, "id": environment.id, "prompt": prompt})
        for arm, policy in ARMS.items():
            print(f"  [{arm}] {args.player_model}", flush=True)
            row = run_row(
                service, environment, label=label, policy=policy,
                max_steps=args.max_steps, decision_budget=args.decision_budget,
            )
            rows.append(row)
            print(
                f"    {'completed' if row['completed'] else 'not completed'}: "
                f"{row['active_steps']} steps, {row['model_calls']} calls, {row['total_tokens']:,} tokens",
                flush=True,
            )

    chart_path = output_dir / "player_reflex_ab.png"
    write_chart(rows, chart_path, player_model=args.player_model, environment_model=args.environment_model)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "player_model": args.player_model,
        "environment_model": args.environment_model,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "decision_budget": args.decision_budget,
        "environments": environments,
        "runs": rows,
        "chart_file": str(chart_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(markdown_report(payload), encoding="utf-8")
    print(f"Wrote {output_dir / 'REPORT.md'}", flush=True)
    print(f"Chart: {chart_path}", flush=True)


if __name__ == "__main__":
    main()

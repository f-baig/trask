"""Run the screenshot-only, one-model-call-per-tick 3D diagnostic arm."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from harness.models import RunRequest, RunStatus
from harness.service import HarnessService
from harness.store import HarnessStore


def _load_dotenv() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--decision-budget", type=int, default=80)
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--policy", default="vision-3d-direct-every-tick")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    _load_dotenv()
    os.environ["ANTHROPIC_PLAYER_MODEL"] = args.player_model
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or Path(".harness-data/direct_3d_visual") / f"tick-run-{stamp}").resolve()
    source, target = HarnessStore(args.fixture_store), HarnessStore(output)
    service = HarnessService(store=target)
    environment = source.list_environments()[0]
    target.save_environment(environment)
    run = service.run(RunRequest(
        environment_id=environment.id, policy_name=args.policy,
        max_steps=args.max_steps, policy_decision_budget=args.decision_budget,
    ), study_name="Direct per-tick 3D screenshot-only diagnostic")
    payload = {
        "run_id": run.id, "status": run.status.value, "reason": run.result_reason,
        "steps": sum(frame.privileged_state.countdown_ticks_remaining == 0 for frame in run.frames),
        "model_calls": run.player_turns or 0, "input_tokens": run.input_tokens or 0,
        "output_tokens": run.output_tokens or 0, "player_model": args.player_model, "policy": args.policy,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Run the vision-only reflex arm on already-generated shared racing fixtures.

The fixture store is copied logically (scenes only) into a fresh result store, so no
environment-generation model calls occur and the original direct/reflex baseline remains
untouched.
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


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-store", type=Path, default=Path(".harness-data/player_reflex_ab/20260818T163834Z"))
    parser.add_argument("--player-model", default="gpt-5.6-luna")
    parser.add_argument("--policy", default="vision-reflex-sim-rehearsal")
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--decision-budget", type=int, default=80)
    parser.add_argument("--tracks", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    os.environ["RACING_PLAYER_MODEL"] = args.player_model
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or Path(".harness-data") / "reflex_vision" / stamp).resolve()
    source = HarnessStore(args.fixture_store)
    target = HarnessStore(output)
    service = HarnessService(store=target)
    rows = []
    fixtures = source.list_environments()
    if args.seeds:
        fixtures = [item for item in fixtures if item.scene.seed in args.seeds]
    for index, environment in enumerate(fixtures[:args.tracks], start=1):
        target.save_environment(environment)
        print(f"[track-{index}] cone screenshots + optical flow, {args.player_model}", flush=True)
        try:
            run = service.run(RunRequest(
                environment_id=environment.id, policy_name=args.policy,
                max_steps=args.max_steps, policy_decision_budget=args.decision_budget,
            ), study_name="Vision-only reflex on frozen fixtures")
            rows.append({"track": index, "run_id": run.id, "completed": run.status == RunStatus.SUCCEEDED,
                         "status": run.status.value, "reason": run.result_reason,
                         "steps": sum(frame.privileged_state.countdown_ticks_remaining == 0 for frame in run.frames),
                         "model_calls": run.player_turns or 0, "input_tokens": run.input_tokens or 0,
                         "output_tokens": run.output_tokens or 0})
        except ProviderError as error:
            rows.append({"track": index, "completed": False, "status": "provider_error", "reason": str(error), "steps": 0, "model_calls": 0, "input_tokens": 0, "output_tokens": 0})
        print(rows[-1], flush=True)
    payload = {"created_at": datetime.now(UTC).isoformat(), "player_model": args.player_model,
               "policy": args.policy,
               "perception": "forward-cone screenshots; optical flow derived only from adjacent screenshots",
               "fixture_store": str(args.fixture_store.resolve()), "runs": rows}
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

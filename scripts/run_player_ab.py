"""Same circuit, two drivers: the reflex harness against a bare per-tick model driver.

Run with `make player-ab`. Every run lands in the store, so each one is inspectable
tick by tick in the Runs tab.

The two arms differ in one thing — what the model is allowed to do with a turn.

- `telemetry-direct` is the naive driver. Each turn it sees public telemetry and returns
  a short queue of control ticks, which are executed directly. No tools, no memory
  between turns beyond the queue, and no way to find out what a control would do
  except by doing it.
- `telemetry-reflex` is the harnessed driver. It writes a small controller, tries
  it against a fork of the current world — deterministic, no model calls, free — reads
  the measured result, retunes, then installs it and says what should wake it. Tick-rate
  control is generated code; the model is consulted a handful of times per race.

Both drive the same compiled scenes at the same seed with the same step budget, and
both are scored on what the simulator recorded rather than on anything either claimed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from harness.cli import _load_local_env  # noqa: E402
from harness.models import RunRequest, RunStatus  # noqa: E402
from harness.service import HarnessService  # noqa: E402

ARMS = {"harness": "telemetry-reflex", "naive": "telemetry-direct"}


def drive(environment_id: str, label: str, arm: str, max_steps: int) -> dict:
    service = HarnessService()
    policy = ARMS[arm]
    try:
        run = service.run(
            RunRequest(environment_id=environment_id, policy_name=policy, max_steps=max_steps),
            study_name="Harness vs naive driver",
        )
    except Exception as error:  # noqa: BLE001 - one arm failing must not lose the other
        return {"label": label, "arm": arm, "error": str(error)[:160]}
    frames = run.frames
    off_track = sum(1 for frame in frames if "off-track" in frame.events)
    return {
        "label": label, "arm": arm, "run_id": run.id,
        "finished": run.status == RunStatus.SUCCEEDED,
        "reason": run.result_reason or run.status.value,
        "ticks": len(frames),
        "reward": round(run.total_reward, 2),
        "off_track": off_track,
        "calls": run.player_turns or 0,
        "output_tokens": run.output_tokens or 0,
        "latency_ms": run.latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=1_400)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", help="Substring of the circuit label, e.g. P1")
    args = parser.parse_args()
    _load_local_env()

    service = HarnessService()
    circuits = sorted(service.list_environments(), key=lambda item: item.scene.name)
    if args.only:
        circuits = [item for item in circuits if args.only in item.scene.name]
    if not circuits:
        print("no circuits to drive")
        return 1

    jobs = [
        (item.id, item.scene.name[:7], arm)
        for item in circuits for arm in ARMS
    ]
    print(f"driving {len(circuits)} circuit(s) with {len(ARMS)} arms = {len(jobs)} runs\n")

    results: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(drive, environment_id, label, arm, args.max_steps): (label, arm)
            for environment_id, label, arm in jobs
        }
        for future in futures.as_completed(pending):
            label, arm = pending[future]
            result = future.result()
            results.append(result)
            if "error" in result:
                print(f"  {label} {arm:<8} FAILED  {result['error'][:90]}")
            else:
                print(f"  {label} {arm:<8} {'finished' if result['finished'] else 'DNF':<9}"
                      f"{result['ticks']:>5} ticks  {result['calls']:>3} calls  "
                      f"off-track {result['off_track']:>3}")

    ok = [item for item in results if "error" not in item]
    if not ok:
        print("\nno runs completed")
        return 1

    print("\n" + "=" * 84)
    print(f"{'circuit':<9}{'arm':<10}{'result':<10}{'ticks':>7}{'off':>6}{'calls':>7}{'out tok':>10}")
    for item in sorted(ok, key=lambda row: (row["label"], row["arm"])):
        print(f"{item['label']:<9}{item['arm']:<10}"
              f"{'finished' if item['finished'] else 'DNF':<10}{item['ticks']:>7}"
              f"{item['off_track']:>6}{item['calls']:>7}{item['output_tokens']:>10}")
    print("=" * 84)
    for arm in ARMS:
        rows = [item for item in ok if item["arm"] == arm]
        if not rows:
            continue
        finished = [item for item in rows if item["finished"]]
        print(f"{arm:<8} finished {len(finished)}/{len(rows)}   "
              f"mean ticks {sum(i['ticks'] for i in rows) / len(rows):.0f}   "
              f"mean off-track {sum(i['off_track'] for i in rows) / len(rows):.1f}   "
              f"model calls {sum(i['calls'] for i in rows)}   "
              f"output tokens {sum(i['output_tokens'] for i in rows):,}")
    print("\nEvery run is in the Runs tab, scrubbable tick by tick.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

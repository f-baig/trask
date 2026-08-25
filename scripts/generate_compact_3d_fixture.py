"""Generate one compact 3D visual-control fixture for costly per-tick experiments."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

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
    _load_dotenv()
    for setting in ("RACING_ENVIRONMENT_MODEL", "RACING_COMPREHENSION_MODEL", "RACING_INTEGRATION_MODEL"):
        os.environ[setting] = "gpt-5.6"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(".harness-data/direct_3d_visual") / f"compact-fixture-{stamp}"
    prompt = (
        "A very compact single-lap, wide asphalt 3D circuit for a short visual-control test. "
        "Keep the lap as short as safely possible while preserving generous radius corners: one "
        "gentle right bend and one gentle left bend, no hairpins, no barriers, no opponents. Add "
        "only subtle rolling elevation and light banking."
    )
    service = HarnessService(store=HarnessStore(output))
    record = service.create_environment(
        prompt, seed=211, provider="anthropic", dimensions="3d", origin="compact direct 3D visual fixture",
        study_name="Per-tick 3D visual control", on_step=lambda stage, message: print(f"[{stage}] {message}", flush=True),
    )
    payload = {
        "environment_id": record.id, "store": str(output.resolve()), "prompt": prompt,
        "generator_model": record.generator_model,
        "certificate_ticks": record.playability_certificate.route_steps if record.playability_certificate else None,
        "fidelity": record.fidelity.summary() if record.fidelity else None,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

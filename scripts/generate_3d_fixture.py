"""Create a harness-authored elevated circuit fixture for visual-player testing."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from harness.service import HarnessService
from harness.store import HarnessStore


def load_dotenv() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    load_dotenv()
    for setting in ("RACING_ENVIRONMENT_MODEL", "RACING_COMPREHENSION_MODEL", "RACING_INTEGRATION_MODEL"):
        os.environ[setting] = "gpt-5.6"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(".harness-data/reflex_vision") / f"three-d-fixture-{stamp}"
    prompt = (
        "A single-lap fair, wide asphalt circuit with rolling hills, moderate banking, and "
        "flowing corners. Include one clear 90 degree bend in the top right, but no hairpins, "
        "opponents, or barriers."
    )
    service = HarnessService(store=HarnessStore(output))
    record = service.create_environment(
        prompt, seed=113, provider="anthropic", dimensions="3d", origin="3D visual-player fixture",
        study_name="Harnessed 3D visual driving", on_step=lambda stage, message: print(f"[{stage}] {message}", flush=True),
    )
    payload = {
        "environment_id": record.id, "store": str(output.resolve()), "prompt": prompt,
        "generator_model": record.generator_model, "input_tokens": record.generator_input_tokens,
        "output_tokens": record.generator_output_tokens, "certificate_ticks": record.playability_certificate.route_steps if record.playability_certificate else None,
        "fidelity": record.fidelity.summary() if record.fidelity else None,
        "elevation": record.scene.elevation.model_dump() if record.scene.elevation else None,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Create one harness-authored technical track fixture for vision-controller testing."""

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
    # This fixture must not fall back to the exhausted Anthropic configuration in .env.
    for setting in ("RACING_ENVIRONMENT_MODEL", "RACING_COMPREHENSION_MODEL", "RACING_INTEGRATION_MODEL"):
        os.environ[setting] = "gpt-5.6"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(".harness-data/reflex_vision") / f"hairpin-fixture-{stamp}"
    prompt = (
        "A single-lap wide, fair asphalt circuit with a precise 90 degree bend in the top right "
        "and a steep 175 degree hairpin in the bottom left. Keep the remaining linking corners "
        "flowing enough for recovery. No opponents and no barriers."
    )
    service = HarnessService(store=HarnessStore(output))
    record = service.create_environment(
        prompt, seed=71, provider="anthropic", origin="technical-vision fixture",
        study_name="Harnessed technical track generation",
        on_step=lambda stage, message: print(f"[{stage}] {message}", flush=True),
    )
    summary = {
        "environment_id": record.id, "store": str(output.resolve()), "prompt": prompt,
        "generator_provider": record.generator_provider, "generator_model": record.generator_model,
        "model_calls": record.generator_input_tokens is not None,
        "input_tokens": record.generator_input_tokens, "output_tokens": record.generator_output_tokens,
        "latency_ms": record.generator_latency_ms,
        "certificate_ticks": record.playability_certificate.route_steps if record.playability_certificate else None,
        "fidelity": record.fidelity.summary() if record.fidelity else None,
        "track_report": record.scene.track_report.model_dump() if record.scene.track_report else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

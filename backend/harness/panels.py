"""Versioned, read-only analysis panels for persisted study artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from .models import ArtifactLink, RunRecord, StudyPanelManifest, StudyPanelResult


@dataclass(frozen=True)
class PanelContext:
    study_name: str
    runs: list[RunRecord]


PanelFn = Callable[[PanelContext], StudyPanelResult]


def _run_link(run: RunRecord) -> ArtifactLink:
    return ArtifactLink(kind="run", id=run.id, label=f"{run.address.prefix if run.address else 'PLAYER'} · {run.policy_name}")


def success_rate(context: PanelContext) -> StudyPanelResult:
    succeeded = [run for run in context.runs if run.status.value == "succeeded"]
    total = len(context.runs)
    rate = round(100 * len(succeeded) / total) if total else 0
    return StudyPanelResult(
        id="success-rate", title="Success rate", kind="metric",
        summary=f"{len(succeeded)} of {total} replays completed a certified lap.",
        data={"value": f"{rate}%", "label": "completed", "secondary": f"{len(succeeded)} / {total} runs"},
        artifacts=[_run_link(run) for run in context.runs],
    )


def blocked_moves(context: PanelContext) -> StudyPanelResult:
    values = [(run, sum(event.startswith(("collision with", "bounced off", "left track")) for frame in run.frames for event in frame.events)) for run in context.runs]
    total = sum(value for _, value in values)
    worst_run, worst_value = max(values, key=lambda item: item[1], default=(None, 0))
    return StudyPanelResult(
        id="blocked-moves", title="Track incidents", kind="metric",
        summary="Counts collision events recorded by the simulator, not model self-report.",
        data={"value": str(total), "label": "collisions / off-track entries", "secondary": f"highest: {worst_value} · {worst_run.policy_name if worst_run else '—'}"},
        artifacts=[_run_link(run) for run, value in values if value > 0],
    )


def terminal_states(context: PanelContext) -> StudyPanelResult:
    rows: list[dict[str, str | int | float]] = []
    for run in context.runs:
        final = run.frames[-1].privileged_state.player if run.frames else None
        rows.append({
            "run_id": run.id, "policy": run.policy_name, "outcome": run.status.value, "steps": len(run.frames),
            "position": f"({final.x:.0f}, {final.y:.0f})" if final else "—",
        })
    return StudyPanelResult(
        id="terminal-states", title="Terminal state distribution", kind="table",
        summary="Final simulator positions and outcomes. Open a source replay to inspect the visible terminal frame.",
        data={"columns": ["policy", "outcome", "steps", "position"], "rows": rows},
        artifacts=[_run_link(run) for run in context.runs],
    )


def policy_outcomes(context: PanelContext) -> StudyPanelResult:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for run in context.runs:
        grouped[run.policy_name].append(run)
    rows = []
    for policy, runs in sorted(grouped.items()):
        outcomes = Counter(run.status.value for run in runs)
        rows.append({"policy": policy, "completed": outcomes["succeeded"], "other": len(runs) - outcomes["succeeded"], "runs": len(runs)})
    return StudyPanelResult(
        id="policy-outcomes", title="Outcome by policy", kind="table",
        summary="A compact comparison across the player adapters in this study.",
        data={"columns": ["policy", "completed", "other", "runs"], "rows": rows},
        artifacts=[_run_link(run) for run in context.runs],
    )


REGISTRY: dict[str, tuple[StudyPanelManifest, PanelFn]] = {
    "success-rate": (StudyPanelManifest(id="success-rate", title="Success rate", description="Completion from actual replay status.", kind="metric"), success_rate),
    "blocked-moves": (StudyPanelManifest(id="blocked-moves", title="Track incidents", description="Collision and recoverable off-track entries from racing replay events.", kind="metric"), blocked_moves),
    "terminal-states": (StudyPanelManifest(id="terminal-states", title="Terminal states", description="Visible final positions and outcomes.", kind="table"), terminal_states),
    "policy-outcomes": (StudyPanelManifest(id="policy-outcomes", title="Outcome by policy", description="Compact policy-level completion comparison.", kind="table"), policy_outcomes),
}

DEFAULT_PANEL_IDS = ["success-rate", "blocked-moves", "terminal-states"]


def catalog() -> list[StudyPanelManifest]:
    return [entry[0] for entry in REGISTRY.values()]


def validate_panel_ids(panel_ids: list[str]) -> list[str]:
    unknown = [panel_id for panel_id in panel_ids if panel_id not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown panel id: {', '.join(unknown)}")
    return list(dict.fromkeys(panel_ids))


def evaluate(panel_ids: list[str], context: PanelContext) -> list[StudyPanelResult]:
    return [REGISTRY[panel_id][1](context) for panel_id in panel_ids]

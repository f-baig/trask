"""Control-plane DAG and executor contracts for local and cluster rollouts."""

from __future__ import annotations

import os
import shlex
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from .models import ExecutionRecord, ExecutionState, ResourceRequest, ResourceUsage


@dataclass(frozen=True)
class RolloutNode:
    id: str
    run_id: str
    dependencies: tuple[str, ...] = ()
    resources: ResourceRequest = field(default_factory=ResourceRequest)


@dataclass(frozen=True)
class RolloutDag:
    id: str
    nodes: tuple[RolloutNode, ...]

    def topological_nodes(self) -> tuple[RolloutNode, ...]:
        pending = {node.id: node for node in self.nodes}
        complete: set[str] = set()
        ordered: list[RolloutNode] = []
        while pending:
            ready = [node for node in pending.values() if set(node.dependencies) <= complete]
            if not ready:
                raise ValueError("Rollout DAG has a dependency cycle or missing dependency.")
            for node in sorted(ready, key=lambda item: item.id):
                ordered.append(node)
                complete.add(node.id)
                pending.pop(node.id)
        return tuple(ordered)


class RolloutExecutor(Protocol):
    id: str

    def submit(self, *, run_id: str, resources: ResourceRequest) -> ExecutionRecord: ...


class LocalExecutor:
    id = "local"

    def submit(self, *, run_id: str, resources: ResourceRequest) -> ExecutionRecord:
        return ExecutionRecord(backend=self.id, state=ExecutionState.RUNNING, job_id=f"local-{uuid.uuid4().hex[:10]}", submitted_at=_timestamp(), worker_id=socket.gethostname(), resource_request=resources)


class SlurmExecutor:
    """Generic Slurm submitter. A worker command calls the harness worker API/CLI."""

    id = "slurm"

    def __init__(self, worker_command: str = "harness-worker run") -> None:
        self.worker_command = worker_command

    def sbatch_script(self, *, run_id: str, resources: ResourceRequest) -> str:
        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name=trask-{run_id}",
            f"#SBATCH --cpus-per-task={max(1, round(resources.cpu_cores))}",
            f"#SBATCH --mem={resources.memory_mb}M",
            f"#SBATCH --time={max(1, resources.wall_time_seconds // 60)}",
        ]
        if resources.gpu_count:
            directives.append(f"#SBATCH --gpus={resources.gpu_count}")
        if resources.queue:
            directives.append(f"#SBATCH --partition={resources.queue}")
        directives.extend(["set -euo pipefail", f"{self.worker_command} --run-id {shlex.quote(run_id)}"])
        return "\n".join(directives) + "\n"

    def submit(self, *, run_id: str, resources: ResourceRequest) -> ExecutionRecord:
        # This creates a durable handoff record. Actual sbatch submission belongs
        # to deployment configuration, not the web-control process.
        job_id = f"slurm-pending-{uuid.uuid4().hex[:10]}"
        return ExecutionRecord(backend=self.id, state=ExecutionState.QUEUED, job_id=job_id, submitted_at=_timestamp(), resource_request=resources, scheduler_metadata={"submit_command": "sbatch", "script": self.sbatch_script(run_id=run_id, resources=resources)})


def measure_usage(start_monotonic: float) -> ResourceUsage:
    import resource

    stats = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS and KiB on Linux.
    rss = stats.ru_maxrss / (1024 * 1024) if os.uname().sysname == "Darwin" else stats.ru_maxrss / 1024
    return ResourceUsage(cpu_time_ms=round((stats.ru_utime + stats.ru_stime) * 1_000), wall_time_ms=round((time.monotonic() - start_monotonic) * 1_000), max_rss_mb=round(rss, 2), host=socket.gethostname())


def _timestamp() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat()

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .models import AgentMessage, EnvironmentRecord, ExperimentRecord, ResearchStudy, RunRecord, StudyPanelConfiguration, TrackDrawing


T = TypeVar("T", bound=BaseModel)


def now() -> str:
    return datetime.now(UTC).isoformat()


class HarnessStore:
    def __init__(self, data_dir: Path | str = ".harness-data/racing") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "harness.sqlite3"
        self._create_schema()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS environments (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    environment_id TEXT NOT NULL,
                    parent_run_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    environment_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id TEXT PRIMARY KEY,
                    agent_role TEXT NOT NULL,
                    environment_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_studies (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS study_panels (
                    study_kind TEXT NOT NULL,
                    study_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (study_kind, study_id)
                );
                CREATE TABLE IF NOT EXISTS precedents (
                    id TEXT PRIMARY KEY,
                    check_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                -- Precedents are read by check kind and nothing else, so the index is the
                -- whole retrieval strategy: a generation asks only about the kinds its own
                -- contract contains, and never scans the table.
                CREATE INDEX IF NOT EXISTS precedents_by_kind
                    ON precedents(check_kind, created_at DESC);
                CREATE TABLE IF NOT EXISTS drawings (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    def save_environment(self, record: EnvironmentRecord) -> None:
        self._upsert("environments", record.id, record.created_at, record.model_dump_json())

    def save_drawing(self, record: TrackDrawing) -> None:
        self._upsert("drawings", record.id, record.created_at, record.model_dump_json())

    def get_drawing(self, record_id: str) -> TrackDrawing | None:
        return self._get("drawings", record_id, TrackDrawing)

    def list_drawings(self) -> list[TrackDrawing]:
        return self._list("drawings", TrackDrawing)

    def delete_drawing(self, record_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM drawings WHERE id = ?", (record_id,))
            return cursor.rowcount > 0

    def save_run(self, record: RunRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO runs(id, environment_id, parent_run_id, created_at, payload) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (record.id, record.environment_id, record.parent_run_id, record.started_at, record.model_dump_json()),
            )

    def save_experiment(self, record: ExperimentRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO experiments(id, environment_id, created_at, payload) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (record.id, record.environment_id, record.created_at, record.model_dump_json()),
            )

    def save_precedent(self, record: "Precedent") -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO precedents(id, check_kind, created_at, payload) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (record.id, record.check_kind, record.created_at, record.model_dump_json()),
            )

    def precedents_for(self, check_kind: str, limit: int = 8) -> list["Precedent"]:
        """Confirmed precedents for one check kind, newest first.

        The `limit` is a read bound, not a display bound — the caller ranks by how close
        each target is to the one being asked about, which needs more than one row to
        choose from but nothing like the whole history.
        """
        from .precedents import Precedent

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM precedents WHERE check_kind = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (check_kind, limit),
            )
            return [Precedent.model_validate_json(row["payload"]) for row in rows]

    def count_precedents(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) AS n FROM precedents").fetchone()["n"])

    def save_agent_message(self, record: AgentMessage) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO agent_messages(id, agent_role, environment_id, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (record.id, record.agent_role, record.environment_id, record.created_at, record.model_dump_json()),
            )

    def save_research_study(self, record: ResearchStudy) -> None:
        self._upsert("research_studies", record.id, record.created_at, record.model_dump_json())

    def save_study_panels(self, record: StudyPanelConfiguration) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO study_panels(study_kind, study_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(study_kind, study_id) DO UPDATE SET payload = excluded.payload",
                (record.study_kind, record.study_id, record.model_dump_json()),
            )

    def get_study_panels(self, study_kind: str, study_id: str) -> StudyPanelConfiguration | None:
        with self._connection() as connection:
            row = connection.execute("SELECT payload FROM study_panels WHERE study_kind = ? AND study_id = ?", (study_kind, study_id)).fetchone()
        return StudyPanelConfiguration.model_validate_json(row["payload"]) if row else None

    def list_agent_messages(self, role: str, environment_id: str | None) -> list[AgentMessage]:
        query = "SELECT payload FROM agent_messages WHERE agent_role = ?"
        values: list[str] = [role]
        if environment_id is None:
            query += " AND environment_id IS NULL"
        else:
            query += " AND environment_id = ?"
            values.append(environment_id)
        query += " ORDER BY created_at ASC"
        with self._connection() as connection:
            return [AgentMessage.model_validate_json(row["payload"]) for row in connection.execute(query, values)]

    def list_agent_activity(self, limit: int = 40) -> list[AgentMessage]:
        """Recent assistant turns across every circuit, newest first.

        This is the shared activity ledger used outside the chat tabs. Reading the
        serialized payload keeps it backward-compatible with databases created
        before messages carried actions.
        """
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM agent_messages ORDER BY created_at DESC LIMIT ?",
                (max(limit * 3, limit),),
            )
        messages = [AgentMessage.model_validate_json(row["payload"]) for row in rows]
        return [item for item in messages if item.speaker == "assistant" and (item.actions or item.artifacts)][:limit]

    def get_environment(self, record_id: str) -> EnvironmentRecord | None:
        return self._get("environments", record_id, EnvironmentRecord)

    def get_run(self, record_id: str) -> RunRecord | None:
        return self._get("runs", record_id, RunRecord)

    def get_experiment(self, record_id: str) -> ExperimentRecord | None:
        return self._get("experiments", record_id, ExperimentRecord)

    def list_environments(self) -> list[EnvironmentRecord]:
        return self._list("environments", EnvironmentRecord)

    def list_runs(self, environment_id: str | None = None) -> list[RunRecord]:
        query = "SELECT payload FROM runs"
        values: tuple[str, ...] = ()
        if environment_id:
            query += " WHERE environment_id = ?"
            values = (environment_id,)
        query += " ORDER BY created_at DESC"
        with self._connection() as connection:
            return [RunRecord.model_validate_json(row["payload"]) for row in connection.execute(query, values)]

    def list_experiments(self, environment_id: str | None = None) -> list[ExperimentRecord]:
        query = "SELECT payload FROM experiments"
        values: tuple[str, ...] = ()
        if environment_id:
            query += " WHERE environment_id = ?"
            values = (environment_id,)
        query += " ORDER BY created_at DESC"
        with self._connection() as connection:
            return [ExperimentRecord.model_validate_json(row["payload"]) for row in connection.execute(query, values)]

    def list_research_studies(self) -> list[ResearchStudy]:
        return self._list("research_studies", ResearchStudy)

    def run_tree_ids(self, record_id: str) -> list[str]:
        """Return one run and every fork descended from it, parent first."""
        with self._connection() as connection:
            rows = connection.execute(
                "WITH RECURSIVE run_tree(id, depth) AS ("
                "SELECT id, 0 FROM runs WHERE id = ? "
                "UNION ALL "
                "SELECT runs.id, run_tree.depth + 1 FROM runs "
                "JOIN run_tree ON runs.parent_run_id = run_tree.id"
                ") SELECT id FROM run_tree ORDER BY depth, id",
                (record_id,),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def delete_runs(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        with self._connection() as connection:
            for offset in range(0, len(record_ids), 500):
                batch = record_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                connection.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", batch)

    def delete_experiment(self, record_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM experiments WHERE id = ?", (record_id,))
            connection.execute(
                "DELETE FROM study_panels WHERE study_kind = 'comparison' AND study_id = ?",
                (record_id,),
            )

    def delete_environments(self, record_ids: list[str]) -> list[str]:
        """Delete circuit records and their circuit-scoped conversations.

        Runs are removed by the service first so their fork and artifact cleanup can use
        the still-present circuit metadata. This method owns the final metadata sweep.
        """
        if not record_ids:
            return []
        deleted_experiments: list[str] = []
        with self._connection() as connection:
            for offset in range(0, len(record_ids), 500):
                batch = record_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                deleted_experiments.extend(
                    str(row["id"]) for row in connection.execute(
                        f"SELECT id FROM experiments WHERE environment_id IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
                connection.execute(
                    f"DELETE FROM agent_messages WHERE environment_id IN ({placeholders})",
                    batch,
                )
                connection.execute(
                    f"DELETE FROM experiments WHERE environment_id IN ({placeholders})",
                    batch,
                )
                connection.execute(
                    f"DELETE FROM environments WHERE id IN ({placeholders})",
                    batch,
                )
            for experiment_id in deleted_experiments:
                connection.execute(
                    "DELETE FROM study_panels WHERE study_kind = 'comparison' AND study_id = ?",
                    (experiment_id,),
                )
        return deleted_experiments

    def delete_research_study(self, record_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM research_studies WHERE id = ?", (record_id,))
            connection.execute("DELETE FROM study_panels WHERE study_id = ?", (record_id,))

    def _upsert(self, table: str, record_id: str, created_at: str, payload: str) -> None:
        with self._connection() as connection:
            connection.execute(
                f"INSERT INTO {table}(id, created_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (record_id, created_at, payload),
            )

    def _get(self, table: str, record_id: str, model: type[T]) -> T | None:
        with self._connection() as connection:
            row = connection.execute(f"SELECT payload FROM {table} WHERE id = ?", (record_id,)).fetchone()
        return model.model_validate_json(row["payload"]) if row else None

    def _list(self, table: str, model: type[T]) -> list[T]:
        with self._connection() as connection:
            rows = connection.execute(f"SELECT payload FROM {table} ORDER BY created_at DESC").fetchall()
        return [model.model_validate_json(row["payload"]) for row in rows]

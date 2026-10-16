"""SQLite evidence and execution history with atomic, repeatable writes."""

from __future__ import annotations

import json
import fcntl
from contextlib import contextmanager
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from .domain import Document, Evidence, Mode, Observation, RunRecord, RunResult, SourceAttempt, TraceEvent, Usage, stable_id, utc_now

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, mode TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id), mode TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, semantic_key TEXT NOT NULL, mode TEXT NOT NULL, revision_of TEXT REFERENCES observations(id), body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS observations_semantic ON observations(semantic_key);
CREATE TABLE IF NOT EXISTS observation_evidence (observation_id TEXT NOT NULL REFERENCES observations(id), evidence_id TEXT NOT NULL REFERENCES evidence(id), PRIMARY KEY(observation_id,evidence_id));
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, mode TEXT NOT NULL, started_at TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traces (seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id), body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS source_attempts (seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id), body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, mode TEXT NOT NULL, week_start TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(id), created_at TEXT NOT NULL, body TEXT NOT NULL, UNIQUE(mode, week_start));
"""


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[redacted]" if any(part in key.lower() for part in ("api_key", "apikey", "authorization", "password", "secret", "access_token")) else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class RunBusy(RuntimeError):
    pass


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._lock_handle = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def ingest(self, document: Document, evidence: list[Evidence], observations: list[Observation]) -> dict[str, int]:
        counts = {"documents": 0, "evidence": 0, "observations": 0, "revisions": 0}
        with self.connection:
            cursor = self.connection.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?)", (document.id, document.mode, document.model_dump_json()))
            counts["documents"] += cursor.rowcount
            for item in evidence:
                if item.document_id != document.id or item.mode != document.mode:
                    raise ValueError("evidence must belong to the imported document and mode")
                if item.source_url != document.url or item.published_at != document.published_at:
                    raise ValueError("evidence provenance must match its document")
                item = item.model_copy(update={"first_ingested_at": item.first_ingested_at or utc_now()})
                cursor = self.connection.execute("INSERT OR IGNORE INTO evidence VALUES (?,?,?,?)", (item.id, item.document_id, item.mode, item.model_dump_json()))
                counts["evidence"] += cursor.rowcount
            for item in observations:
                if item.mode != document.mode:
                    raise ValueError("observation mode differs from document mode")
                for evidence_id in item.evidence_ids:
                    linked = self.connection.execute("SELECT mode,body FROM evidence WHERE id=?", (evidence_id,)).fetchone()
                    if linked is None or linked["mode"] != item.mode:
                        raise ValueError("observation must cite stored evidence in its mode")
                    if Evidence.model_validate_json(linked["body"]).validation_status != "validated":
                        raise ValueError("normalized observations require validated monitoring evidence")
                existing = self.connection.execute("SELECT id FROM observations WHERE id=?", (item.id,)).fetchone()
                if existing is None:
                    previous = self.connection.execute("SELECT id FROM observations WHERE semantic_key=? ORDER BY rowid DESC LIMIT 1", (item.semantic_key,)).fetchone()
                    revision_of = previous["id"] if previous else None
                    if item.revision_of and item.revision_of != revision_of:
                        raise ValueError("revision must reference the prior stored value of this measurement")
                    item = item.model_copy(update={"revision_of": revision_of})
                    self.connection.execute("INSERT INTO observations VALUES (?,?,?,?,?)", (item.id, item.semantic_key, item.mode, item.revision_of, item.model_dump_json()))
                    counts["observations"] += 1
                    counts["revisions"] += int(revision_of is not None)
                self.connection.executemany("INSERT OR IGNORE INTO observation_evidence VALUES (?,?)", [(item.id, eid) for eid in item.evidence_ids])
        return counts

    def list_documents(self, mode: Mode) -> list[Document]:
        return [Document.model_validate_json(row["body"]) for row in self.connection.execute("SELECT body FROM documents WHERE mode=? ORDER BY rowid", (mode,))]

    def get_document(self, document_id: str, mode: Mode | None = None) -> Document | None:
        row = self.connection.execute("SELECT body,mode FROM documents WHERE id=?", (document_id,)).fetchone()
        return Document.model_validate_json(row["body"]) if row and (mode is None or row["mode"] == mode) else None

    def list_evidence(self, mode: Mode) -> list[Evidence]:
        return [Evidence.model_validate_json(row["body"]) for row in self.connection.execute("SELECT body FROM evidence WHERE mode=? ORDER BY rowid", (mode,))]

    def get_evidence(self, evidence_id: str, mode: Mode | None = None) -> Evidence | None:
        row = self.connection.execute("SELECT body,mode FROM evidence WHERE id=?", (evidence_id,)).fetchone()
        return Evidence.model_validate_json(row["body"]) if row and (mode is None or row["mode"] == mode) else None

    def get_observation(self, observation_id: str, mode: Mode | None = None) -> Observation | None:
        row = self.connection.execute("SELECT body,mode FROM observations WHERE id=?", (observation_id,)).fetchone()
        if row is None or (mode is not None and row["mode"] != mode):
            return None
        item = Observation.model_validate_json(row["body"])
        ids = [linked["evidence_id"] for linked in self.connection.execute("SELECT evidence_id FROM observation_evidence WHERE observation_id=? ORDER BY evidence_id", (item.id,))]
        return item.model_copy(update={"evidence_ids": ids})

    def list_observations(self, mode: Mode) -> list[Observation]:
        # Distinct same-period values remain visible conflicts; import order cannot establish authority.
        return [self.get_observation(row["id"]) for row in self.connection.execute("SELECT id FROM observations WHERE mode=? ORDER BY rowid", (mode,))]

    def set_evidence_validation(self, evidence_id: str, status: str, reason: str) -> Evidence:
        item = self.get_evidence(evidence_id)
        if item is None:
            raise ValueError("unknown evidence ID")
        updated = Evidence.model_validate({**item.model_dump(), "validation_status": status, "validation_reason": reason})
        with self.connection:
            self.connection.execute("UPDATE evidence SET body=? WHERE id=?", (updated.model_dump_json(), evidence_id))
        return updated

    def counts(self, mode: Mode) -> dict[str, int]:
        return {table: self.connection.execute(f"SELECT COUNT(*) FROM {table} WHERE mode=?", (mode,)).fetchone()[0] for table in ("documents", "evidence", "observations", "runs", "reports")}

    def save_run(self, run: RunRecord) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO runs VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,body=excluded.body", (run.id, run.task.mode, run.started_at.isoformat(), run.status, run.model_dump_json()))

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self.connection.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row["body"]) if row else None

    def list_runs(self, mode: Mode, limit: int = 100) -> list[RunRecord]:
        return [RunRecord.model_validate_json(row["body"]) for row in self.connection.execute("SELECT body FROM runs WHERE mode=? ORDER BY started_at DESC LIMIT ?", (mode, limit))]

    def add_trace(self, run_id: str, event: TraceEvent) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO traces(run_id,body) VALUES (?,?)", (run_id, json.dumps(redact(event.model_dump(mode="json")))))

    def list_traces(self, run_id: str) -> list[TraceEvent]:
        return [TraceEvent.model_validate_json(row["body"]) for row in self.connection.execute("SELECT body FROM traces WHERE run_id=? ORDER BY seq", (run_id,))]

    def add_source_attempt(self, run_id: str, attempt: SourceAttempt) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO source_attempts(run_id,body) VALUES (?,?)", (run_id, attempt.model_dump_json()))

    def list_source_attempts(self, run_id: str) -> list[SourceAttempt]:
        return [SourceAttempt.model_validate_json(row["body"]) for row in self.connection.execute("SELECT body FROM source_attempts WHERE run_id=? ORDER BY seq", (run_id,))]

    def save_report(self, run_id: str, week_start: date, result: RunResult) -> str:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("report requires a stored run")
        if result.status not in ("completed", "partial"):
            raise ValueError("only completed or partial results become reports")
        report_id = stable_id("report", run.task.mode, week_start)
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO reports VALUES (?,?,?,?,?,?)", (report_id, run.task.mode, week_start.isoformat(), run_id, utc_now().isoformat(), result.model_dump_json()))
        return report_id

    def list_reports(self, mode: Mode) -> list[dict[str, Any]]:
        return [{"id": row["id"], "mode": row["mode"], "week_start": row["week_start"], "run_id": row["run_id"], "created_at": row["created_at"], "result": RunResult.model_validate_json(row["body"])} for row in self.connection.execute("SELECT * FROM reports WHERE mode=? ORDER BY week_start DESC", (mode,))]

    @property
    def execution_locked(self) -> bool:
        return self._lock_handle is not None

    @contextmanager
    def execution_lock(self):
        if self._lock_handle is not None:
            yield
            return
        handle = self.path.with_name(self.path.name + ".run.lock").open("a+")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunBusy("Another run currently holds this database's execution lock") from None
            self._lock_handle = handle
            yield
        finally:
            if self._lock_handle is handle:
                self._lock_handle = None
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def reconcile_abandoned_runs(self) -> list[str]:
        if self._lock_handle is None:
            raise RuntimeError("Recovery requires the database execution lock")
        recovered = []
        rows = self.connection.execute("SELECT body FROM runs WHERE status='running'").fetchall()
        for row in rows:
            run = RunRecord.model_validate_json(row["body"])
            usage = run.usage
            traces = self.list_traces(run.id)
            if traces and traces[-1].outcome == "reserved":
                bound = traces[-1].data
                usage = Usage(input_tokens=usage.input_tokens + bound.get("input_token_bound", 0), output_tokens=usage.output_tokens + bound.get("output_token_bound", 0), spend_usd=usage.spend_usd + bound.get("spend_bound_usd", 0), cost_known=False)
            reason = "Interrupted process: recovered abandoned run after acquiring the execution lock"
            result = RunResult(status="failed", answer="The previous process stopped before completion; retained evidence and source attempts remain inspectable.", reason=reason, usage=usage)
            terminal = run.model_copy(update={"finished_at": utc_now(), "status": "failed", "result": result, "usage": usage})
            self.save_run(terminal)
            self.add_trace(run.id, TraceEvent(kind="recovery", name="run", outcome="failed", data={"reason": reason, "usage": usage.model_dump()}))
            recovered.append(run.id)
        return recovered

    def last_success(self, mode: Mode, entry_skill: str) -> RunRecord | None:
        for row in self.connection.execute("SELECT body FROM runs WHERE mode=? AND status IN ('completed','partial') ORDER BY started_at DESC", (mode,)):
            run = RunRecord.model_validate_json(row["body"])
            if run.task.entry_skill == entry_skill:
                return run
        return None

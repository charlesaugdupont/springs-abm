"""
Persistent job store for simulation runs, backing webapp/jobs.py. Replaces
the earlier in-memory dict: a process restart (redeploy, crash, a host that
sleeps and wakes) no longer silently drops every in-flight/finished run.

Still a single-file SQLite database, not a client/server database - this
deployment runs a single process (see webapp/executor.py's docstring on
why exactly one simulation ever executes at a time), so there is no
multi-writer concurrency to design for beyond guarding against FastAPI's
own multiple request-handling threads, which the existing threading.Lock
pattern from jobs.py already covers.

Note this does not, by itself, survive every hosting environment's notion
of "restart" - e.g. a container platform with an ephemeral filesystem
loses this file on a cold restart same as it would lose an in-memory dict.
It is a strict improvement (survives everything short of that, enables a
recent-runs list) and a one-line path change away from real durability
(point WEBAPP_DB_PATH at a mounted persistent volume) if that's ever needed.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any

from webapp.settings import WEBAPP_DB_PATH

_LOCK = Lock()
_CONN: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    config_form TEXT NOT NULL,
    summary TEXT,
    result TEXT,
    error TEXT,
    progress_day INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


def get_connection() -> sqlite3.Connection:
    """Lazily opens (and schema-initializes) the single module-level connection.
    check_same_thread=False because webapp/executor.py's worker thread and
    FastAPI's request-handling threads both need to use this connection -
    safe because every actual access is already serialized by _LOCK below."""
    global _CONN
    if _CONN is None:
        db_path = Path(WEBAPP_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(str(db_path), check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        with _LOCK:
            _CONN.executescript(_SCHEMA)
            _CONN.commit()
    return _CONN


def reset_for_tests(db_path: str) -> None:
    """Test-only helper: points the module-level connection at a fresh path
    (e.g. an in-memory or tmp_path database) so tests don't share state with
    real runs or each other. Not called from application code.

    Rebinds this module's own WEBAPP_DB_PATH name (not webapp.settings') -
    get_connection() reads the name bound here by the top-of-file `from
    webapp.settings import WEBAPP_DB_PATH`, which is a one-time value copy,
    not a live reference back into the settings module."""
    global _CONN, WEBAPP_DB_PATH
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
        _CONN = None
    WEBAPP_DB_PATH = db_path


def insert_run(job_id: str, status: str, created_at: float, config_form: dict[str, Any]) -> None:
    conn = get_connection()
    with _LOCK:
        conn.execute(
            "INSERT INTO runs (job_id, status, created_at, config_form) VALUES (?, ?, ?, ?)",
            (job_id, status, created_at, json.dumps(config_form)),
        )
        conn.commit()


def update_status(job_id: str, status: str, *, progress_total: int | None = None) -> None:
    conn = get_connection()
    with _LOCK:
        if progress_total is not None:
            conn.execute(
                "UPDATE runs SET status = ?, progress_total = ? WHERE job_id = ?",
                (status, progress_total, job_id),
            )
        else:
            conn.execute("UPDATE runs SET status = ? WHERE job_id = ?", (status, job_id))
        conn.commit()


def update_progress(job_id: str, day: int) -> None:
    conn = get_connection()
    with _LOCK:
        conn.execute("UPDATE runs SET progress_day = ? WHERE job_id = ?", (day, job_id))
        conn.commit()


def set_done(job_id: str, status: str, summary: dict[str, Any], result: dict[str, Any]) -> None:
    conn = get_connection()
    with _LOCK:
        conn.execute(
            "UPDATE runs SET status = ?, summary = ?, result = ? WHERE job_id = ?",
            (status, json.dumps(summary), json.dumps(result), job_id),
        )
        conn.commit()


def set_error(job_id: str, status: str, error: str) -> None:
    conn = get_connection()
    with _LOCK:
        conn.execute("UPDATE runs SET status = ?, error = ? WHERE job_id = ?", (status, error, job_id))
        conn.commit()


def get_run(job_id: str) -> sqlite3.Row | None:
    conn = get_connection()
    with _LOCK:
        return conn.execute("SELECT * FROM runs WHERE job_id = ?", (job_id,)).fetchone()


def list_recent(limit: int) -> list[sqlite3.Row]:
    conn = get_connection()
    with _LOCK:
        return conn.execute(
            "SELECT job_id, status, created_at, config_form, summary, error "
            "FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def count_active() -> int:
    conn = get_connection()
    with _LOCK:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM runs WHERE status IN ('queued', 'running')"
        ).fetchone()
        return row["n"]


def evict_old(max_retained: int, max_retention_seconds: float) -> None:
    """Drops old finished (done/error) runs. Mirrors jobs.py's old in-memory
    eviction policy: never evicts queued/running rows, evicts anything
    finished past the retention window, then trims oldest-first overflow."""
    conn = get_connection()
    now = time.time()
    with _LOCK:
        conn.execute(
            "DELETE FROM runs WHERE status IN ('done', 'error') AND ? - created_at > ?",
            (now, max_retention_seconds),
        )
        conn.execute(
            """
            DELETE FROM runs WHERE job_id IN (
                SELECT job_id FROM runs
                WHERE status IN ('done', 'error')
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (max_retained,),
        )
        conn.commit()

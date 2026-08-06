"""Tests for webapp/db.py - the SQLite-backed job store that replaced the
old in-memory dict. The core claim being tested is genuine persistence:
a row written via one connection must be readable after that connection is
closed and a fresh one opened against the same file, proving durability
isn't just an artifact of the same Python object staying alive."""
from __future__ import annotations

import sqlite3
import time

from webapp import db


def test_insert_and_get_run():
    db.insert_run("job-1", "queued", 100.0, {"seed": 1})
    row = db.get_run("job-1")
    assert row["job_id"] == "job-1"
    assert row["status"] == "queued"
    assert row["progress_day"] == 0


def test_row_survives_a_fresh_connection_to_the_same_file(tmp_path):
    """The actual persistence claim: write via one connection, close it,
    open a brand new connection to the same path, and confirm the row is
    still there - not just still-alive-in-memory."""
    db_path = str(tmp_path / "persist.db")
    db.reset_for_tests(db_path)
    db.insert_run("job-persist", "queued", 100.0, {"seed": 42})

    # Simulate a process restart: close the connection this module is
    # holding, then open a completely independent one to the same file.
    db.get_connection().close()
    fresh_conn = sqlite3.connect(db_path)
    fresh_conn.row_factory = sqlite3.Row
    row = fresh_conn.execute("SELECT * FROM runs WHERE job_id = ?", ("job-persist",)).fetchone()
    fresh_conn.close()

    assert row is not None
    assert row["job_id"] == "job-persist"
    assert row["status"] == "queued"


def test_update_status_and_progress():
    db.insert_run("job-2", "queued", 100.0, {})
    db.update_status("job-2", "running", progress_total=30)
    db.update_progress("job-2", 5)
    row = db.get_run("job-2")
    assert row["status"] == "running"
    assert row["progress_total"] == 30
    assert row["progress_day"] == 5


def test_set_done_and_set_error():
    db.insert_run("job-3", "queued", 100.0, {})
    db.set_done("job-3", "done", {"n_u5": 10}, {"n_u5": 10, "days": [0, 1]})
    row = db.get_run("job-3")
    assert row["status"] == "done"

    db.insert_run("job-4", "queued", 100.0, {})
    db.set_error("job-4", "error", "boom")
    row = db.get_run("job-4")
    assert row["status"] == "error"
    assert row["error"] == "boom"


def test_count_active_only_counts_queued_and_running():
    db.insert_run("job-a", "queued", 100.0, {})
    db.insert_run("job-b", "running", 101.0, {})
    db.insert_run("job-c", "done", 102.0, {})
    db.insert_run("job-d", "error", 103.0, {})
    assert db.count_active() == 2


def test_list_recent_orders_newest_first():
    db.insert_run("older", "done", 100.0, {})
    db.insert_run("newer", "done", 200.0, {})
    rows = db.list_recent(limit=10)
    assert [r["job_id"] for r in rows] == ["newer", "older"]


def test_list_recent_respects_limit():
    for i in range(5):
        db.insert_run(f"job-{i}", "done", float(i), {})
    rows = db.list_recent(limit=2)
    assert len(rows) == 2


def test_evict_old_never_evicts_queued_or_running():
    db.insert_run("stuck-queued", "queued", 0.0, {})  # ancient, but active
    db.evict_old(max_retained=0, max_retention_seconds=0)
    assert db.get_run("stuck-queued") is not None


def test_evict_old_drops_stale_finished_runs():
    db.insert_run("old-done", "done", 0.0, {})
    db.evict_old(max_retained=50, max_retention_seconds=1)
    assert db.get_run("old-done") is None


def test_evict_old_trims_overflow_oldest_first():
    now = time.time()
    for i in range(5):
        db.insert_run(f"job-{i}", "done", now + i, {})
    db.evict_old(max_retained=2, max_retention_seconds=10_000)
    remaining = {r["job_id"] for r in db.list_recent(limit=10)}
    assert remaining == {"job-3", "job-4"}

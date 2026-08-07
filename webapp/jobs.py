"""
Job store for simulation runs triggered from the scenario builder. Backed
by SQLite (webapp/db.py) rather than an in-memory dict - see that module's
docstring for why. Every function here keeps its original signature from
the in-memory-dict version (public API used by webapp/executor.py and
webapp/simulation_runner.py, both of which need zero changes as a result),
plus one new function (list_recent) added for the recent-runs feature.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from webapp import db


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class SimResultBundle:
    """Plain-data results of one simulation run - the only thing that
    crosses back from the worker thread into the main process/state store.
    No live SVEIRModel, no torch tensors.
    """
    config_snapshot: dict[str, Any]
    pathogen_names: list[str]

    # Daily time series, one list entry per simulated day.
    days: list[int]
    u5_prevalence: dict[str, list[float]]
    all_ages_prevalence: dict[str, list[float]]
    cumulative_u5_illness_days: dict[str, list[float]]
    mean_household_wealth: list[float]
    cumulative_care_seeking_events: list[float]

    # Spatial scrubber: one 25x25 (list-of-lists) grid per day, cumulative
    # infection footprint across all configured pathogens.
    spatial_grid_size: int
    spatial_daily_grids: list[list[list[float]]]

    # End-of-run summary metrics (from experiments/metrics.py, reused as-is).
    summary_metrics: dict[str, Any]

    proportion_infected_at_least_once: float
    n_u5: int
    runtime_seconds: float

    # Per-day new Campylobacter infections split by route (keys: zoonotic,
    # fecal_oral, food_borne); empty dict when campy is disabled. Has a default
    # so runs serialized before this field was added still deserialize via
    # SimResultBundle(**json.loads(...)) in _row_to_record.
    campy_daily_infections_by_route: dict[str, list[float]] = field(default_factory=dict)


# Fields of SimResultBundle cheap enough to duplicate into the `runs` table's
# denormalized `summary` column, so GET /api/runs (the list view) never has
# to load a full run's (potentially multi-MB) time-series/spatial payload
# just to show a "recent runs" row.
_SUMMARY_FIELDS = (
    "pathogen_names", "summary_metrics", "proportion_infected_at_least_once",
    "n_u5", "runtime_seconds",
)


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: float
    config_form: dict[str, Any]
    result: SimResultBundle | None = None
    error: str | None = None
    progress_day: int = 0
    progress_total: int = 0


MAX_RETAINED_JOBS = 50
MAX_RETENTION_SECONDS = 2 * 60 * 60  # 2 hours


def _row_to_record(row) -> JobRecord:
    result = None
    if row["result"] is not None:
        result = SimResultBundle(**json.loads(row["result"]))
    return JobRecord(
        job_id=row["job_id"],
        status=JobStatus(row["status"]),
        created_at=row["created_at"],
        config_form=json.loads(row["config_form"]),
        result=result,
        error=row["error"],
        progress_day=row["progress_day"],
        progress_total=row["progress_total"],
    )


def new_job(config_form: dict[str, Any]) -> JobRecord:
    job_id = uuid.uuid4().hex
    created_at = time.time()
    db.evict_old(MAX_RETAINED_JOBS, MAX_RETENTION_SECONDS)
    db.insert_run(job_id, JobStatus.QUEUED.value, created_at, config_form)
    return JobRecord(job_id=job_id, status=JobStatus.QUEUED, created_at=created_at, config_form=config_form)


def get_job(job_id: str) -> JobRecord | None:
    row = db.get_run(job_id)
    return _row_to_record(row) if row is not None else None


def set_running(job_id: str, total_days: int = 0) -> None:
    db.update_status(job_id, JobStatus.RUNNING.value, progress_total=total_days)


def set_progress(job_id: str, day: int) -> None:
    """Called from inside the simulation's day loop (webapp/simulation_runner.py) to report
    real progress - safe to call from the worker thread, db.py's lock is a threading.Lock
    (not asyncio), and this module already assumes cross-thread access."""
    db.update_progress(job_id, day)


def set_done(job_id: str, result: SimResultBundle) -> None:
    full = asdict(result)
    summary = {k: full[k] for k in _SUMMARY_FIELDS}
    db.set_done(job_id, JobStatus.DONE.value, summary, full)


def set_error(job_id: str, error: str) -> None:
    db.set_error(job_id, JobStatus.ERROR.value, error)


def count_active() -> int:
    """Number of jobs currently queued or running - used to cap accepted jobs."""
    return db.count_active()


def list_recent(limit: int = 20) -> list[dict[str, Any]]:
    """Lightweight recent-runs listing - reads only the denormalized `summary`
    column, not the full per-day result blob (see _SUMMARY_FIELDS)."""
    rows = db.list_recent(limit)
    out = []
    for row in rows:
        out.append({
            "job_id": row["job_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "config_form": json.loads(row["config_form"]),
            "summary": json.loads(row["summary"]) if row["summary"] is not None else None,
            "error": row["error"],
        })
    return out

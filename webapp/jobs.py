"""
In-memory job store for simulation runs triggered from the scenario
builder. Deliberately not a database - see the plan's "Result storage"
note: light/occasional consortium use, no run-history feature in v1, and
this module is a process-local singleton by design (the deployed process
runs with exactly one worker - see the Dockerfile).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


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


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: float
    config_form: dict[str, Any]
    result: SimResultBundle | None = None
    error: str | None = None


_JOBS: dict[str, JobRecord] = {}
_LOCK = Lock()

MAX_RETAINED_JOBS = 50
MAX_RETENTION_SECONDS = 2 * 60 * 60  # 2 hours


def new_job(config_form: dict[str, Any]) -> JobRecord:
    job_id = uuid.uuid4().hex
    record = JobRecord(
        job_id=job_id, status=JobStatus.QUEUED, created_at=time.time(), config_form=config_form,
    )
    with _LOCK:
        _evict_locked()
        _JOBS[job_id] = record
    return record


def get_job(job_id: str) -> JobRecord | None:
    with _LOCK:
        return _JOBS.get(job_id)


def set_running(job_id: str) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].status = JobStatus.RUNNING


def set_done(job_id: str, result: SimResultBundle) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].status = JobStatus.DONE
            _JOBS[job_id].result = result


def set_error(job_id: str, error: str) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].status = JobStatus.ERROR
            _JOBS[job_id].error = error


def count_active() -> int:
    """Number of jobs currently queued or running - used to cap accepted jobs."""
    with _LOCK:
        return sum(1 for j in _JOBS.values() if j.status in (JobStatus.QUEUED, JobStatus.RUNNING))


def _evict_locked() -> None:
    """Drops old DONE/ERROR jobs. Caller must hold _LOCK. Never evicts QUEUED/RUNNING jobs."""
    now = time.time()
    finished = [
        j for j in _JOBS.values()
        if j.status in (JobStatus.DONE, JobStatus.ERROR)
    ]
    stale = [j for j in finished if now - j.created_at > MAX_RETENTION_SECONDS]
    for j in stale:
        del _JOBS[j.job_id]

    finished = sorted(
        (j for j in _JOBS.values() if j.status in (JobStatus.DONE, JobStatus.ERROR)),
        key=lambda j: j.created_at,
    )
    overflow = len(finished) - MAX_RETAINED_JOBS
    for j in finished[:max(0, overflow)]:
        del _JOBS[j.job_id]

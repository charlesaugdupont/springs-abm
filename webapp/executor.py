"""
Single-slot execution queue for simulation runs - see the plan's §2 for why
this is deliberately NOT a multi-process pool: reproducibility (avoiding a
race on abm/'s process-global RNG state - set_global_seed/torch.manual_seed,
see abm/utils/rng.py) and fitting a free-tier ~512MB RAM budget both favor
running exactly one simulation at a time rather than paying for parallelism
this app doesn't need yet.

A single-worker ThreadPoolExecutor already gives this for free: work
submitted to it runs strictly one task at a time, in submission (FIFO)
order - with exactly one job ever executing, there's no concurrent access
to the global RNG, so no separate lock/queue data structure is needed on
top of it. If genuine parallelism is ever wanted on a beefier host, this is
the one place to change (max_workers=1 -> a ProcessPoolExecutor).
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from config import SVEIRConfig
from webapp import jobs
from webapp.simulation_runner import run_simulation_for_ui

MAX_ACCEPTED_JOBS = 5  # queued + running; beyond this, submit_job() raises QueueFullError

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim-worker")


class QueueFullError(Exception):
    """Raised when MAX_ACCEPTED_JOBS is already reached - callers should return HTTP 429."""


async def submit_job(config: SVEIRConfig, config_form: dict[str, Any]) -> jobs.JobRecord:
    if jobs.count_active() >= MAX_ACCEPTED_JOBS:
        raise QueueFullError(
            f"Already {MAX_ACCEPTED_JOBS} simulations queued or running - please try again shortly."
        )

    record = jobs.new_job(config_form)
    loop = asyncio.get_running_loop()
    # Fire-and-forget: the route returns immediately so the browser can start polling the
    # status page. Exceptions are caught inside _run_and_record, not left on this Future.
    loop.run_in_executor(_EXECUTOR, _run_and_record, config, record.job_id)
    return record


def warm_up() -> None:
    """Submits a throwaway job at app startup so the worker thread pays the torch/abm import
    cost once, before the first real user request arrives."""
    _EXECUTOR.submit(_import_only)


def _import_only() -> None:
    import abm.model.initialize_model  # noqa: F401


def _run_and_record(config: SVEIRConfig, job_id: str) -> None:
    """Runs inside the single worker thread. Any exception is caught here (not left to
    propagate into an unawaited Future) and recorded on the job so the UI can show it."""
    jobs.set_running(job_id, total_days=config.step_target)
    try:
        result = run_simulation_for_ui(config, job_id)
        jobs.set_done(job_id, result)
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure must reach the UI, not vanish silently
        jobs.set_error(job_id, str(e))

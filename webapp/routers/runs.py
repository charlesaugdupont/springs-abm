"""Run status/results as JSON. GET /api/runs/{id} is the poll target the
frontend's React Query hook re-fetches on an interval while a job is
queued/running (replacing the old HTMX partial-swap poll) - it always
returns the same shape regardless of status, so a fast job that finishes
before the first poll still renders correctly on the very first response.

Critically, this now returns `config_form` on every response - the actual
fix for "submitted parameters vanish after a run starts" (see the plan's
Context section): the value was already captured server-side before this
phase, just never sent back to any client.

The former _spatial_axes() helper (mapping the spatial grid's row/col
indices onto the basemap's geographic extent) is deliberately not carried
over here - it's pure display-coordinate arithmetic with no server-side
state dependency, so it's been reimplemented client-side instead, in
frontend/src/lib/spatialAxes.ts."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from webapp.auth import require_login
from webapp.jobs import get_job, list_recent

router = APIRouter()


def _run_json(record) -> dict:
    return {
        "job_id": record.job_id,
        "status": record.status.value,
        "created_at": record.created_at,
        "config_form": record.config_form,
        "error": record.error,
        "progress_day": record.progress_day,
        "progress_total": record.progress_total,
        "result": asdict(record.result) if record.result is not None else None,
    }


@router.get("/runs", dependencies=[Depends(require_login)])
def list_runs(limit: int = 20):
    return {"runs": list_recent(limit=limit)}


@router.get("/runs/{job_id}", dependencies=[Depends(require_login)])
def run_detail(job_id: str):
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found (it may have expired).")
    return _run_json(record)


@router.get("/runs/{job_id}/data.json", dependencies=[Depends(require_login)])
def run_data(job_id: str):
    record = get_job(job_id)
    if record is None or record.result is None:
        raise HTTPException(status_code=404, detail="Result not available for this run.")
    return JSONResponse(asdict(record.result))

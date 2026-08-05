"""Results page: shows a job's status while running, and its curves/spatial
scrubber/summary metrics once done. GET /runs/{id}/status is the HTMX poll
target - it returns the same _results.html fragment once done, so a fast
job that finishes before the browser's first poll still renders correctly
on the very first response."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from webapp.auth import require_login
from webapp.jobs import JobStatus, get_job
from webapp.templating import templates

router = APIRouter()

_BASEMAP_META = json.loads(Path("webapp/static/img/akuse_basemap.json").read_text())


def _spatial_axes() -> dict:
    """Maps the downsampled spatial grid's row/col indices onto the basemap's geographic
    extent, so the heatmap trace lines up with the background image (see
    webapp/simulation_runner.py's _spatial_grid: row=y-bin index (0=south), col=x-bin index
    (0=west), matching the basemap PNG's own [minx,maxx] x [miny,maxy] orientation)."""
    from webapp.simulation_runner import SPATIAL_GRID_SIZE

    minx, maxx = _BASEMAP_META["minx"], _BASEMAP_META["maxx"]
    miny, maxy = _BASEMAP_META["miny"], _BASEMAP_META["maxy"]
    xs = [minx + (k + 0.5) / SPATIAL_GRID_SIZE * (maxx - minx) for k in range(SPATIAL_GRID_SIZE)]
    ys = [miny + (k + 0.5) / SPATIAL_GRID_SIZE * (maxy - miny) for k in range(SPATIAL_GRID_SIZE)]
    return {"x": xs, "y": ys}


def _context(job_id: str) -> dict:
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found (it may have expired).")
    return {
        "authenticated": True,
        "job_id": job_id,
        "status": record.status,
        "error": record.error,
        "result": record.result,
        "basemap": _BASEMAP_META,
        "spatial_axes": _spatial_axes(),
    }


@router.get("/runs/{job_id}", response_class=HTMLResponse, dependencies=[Depends(require_login)])
def run_page(request: Request, job_id: str):
    ctx = {"request": request, **_context(job_id)}
    template = "runs/_results.html" if ctx["status"] == JobStatus.DONE else "runs/_status.html"
    return templates.TemplateResponse(request, "runs/page.html", {**ctx, "inner_template": template})


@router.get("/runs/{job_id}/status", response_class=HTMLResponse, dependencies=[Depends(require_login)])
def run_status(request: Request, job_id: str):
    ctx = _context(job_id)
    template = "runs/_results.html" if ctx["status"] == JobStatus.DONE else "runs/_status.html"
    return templates.TemplateResponse(request, template, ctx)


@router.get("/runs/{job_id}/data.json", dependencies=[Depends(require_login)])
def run_data(job_id: str):
    record = get_job(job_id)
    if record is None or record.result is None:
        raise HTTPException(status_code=404, detail="Result not available for this run.")
    return JSONResponse(asdict(record.result))

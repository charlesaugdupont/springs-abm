"""POST /api/scenario/run: validates a submitted scenario (via
ScenarioFormInput, FastAPI validates automatically since the body is typed
as that Pydantic model - the client/server dual-bounds pattern documented
in webapp/scenario_form.py is unchanged), builds a SVEIRConfig, and submits
a job. The only validation FastAPI doesn't do for us is the cross-field
"at least one pathogen enabled" rule, raised as a plain ValueError from
build_sveir_config."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from webapp.auth import require_login
from webapp.executor import QueueFullError, submit_job
from webapp.scenario_form import ScenarioFormInput, build_sveir_config

router = APIRouter()


@router.post("/scenario/run", dependencies=[Depends(require_login)])
async def scenario_run(form: ScenarioFormInput):
    try:
        config = build_sveir_config(form)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        record = await submit_job(config, form.model_dump())
    except QueueFullError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e

    return JSONResponse({"job_id": record.job_id, "status": record.status.value}, status_code=202)

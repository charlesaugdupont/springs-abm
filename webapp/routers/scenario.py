"""Scenario-builder page: GET renders the parameter_registry.py-driven form,
POST validates the submission, builds a SVEIRConfig, and submits a job."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from webapp.auth import require_login
from webapp.executor import QueueFullError, submit_job
from webapp.parameter_registry import CATEGORY_ORDER, by_category
from webapp.scenario_form import FORM_FIELD_TO_PATH, ScenarioFormInput, build_sveir_config
from webapp.templating import templates

router = APIRouter()

PATH_TO_FORM_FIELD = {path: name for name, path in FORM_FIELD_TO_PATH.items()}


def _category_view() -> list[dict]:
    grouped = by_category()
    view = []
    for cat in CATEGORY_ORDER:
        metas = grouped.get(cat, [])
        if not metas:
            continue
        editable = [{"meta": m, "form_name": PATH_TO_FORM_FIELD[m.path]} for m in metas if m.editable]
        readonly = [m for m in metas if not m.editable]
        view.append({"name": cat, "editable": editable, "readonly": readonly})
    return view


def _render(request: Request, *, form_values: dict | None = None, errors: str | None = None, status_code: int = 200):
    # A plain dict, not the Pydantic model itself - keeps the template's default-value lookups
    # to simple `defaults[form_name]` subscripting rather than method calls/attribute checks.
    defaults = form_values if form_values is not None else ScenarioFormInput().model_dump()
    return templates.TemplateResponse(
        request, "scenario_builder.html",
        {"authenticated": True, "categories": _category_view(), "defaults": defaults, "errors": errors},
        status_code=status_code,
    )


@router.get("/scenario", response_class=HTMLResponse, dependencies=[Depends(require_login)])
def scenario_builder(request: Request):
    return _render(request)


@router.post("/scenario/run", dependencies=[Depends(require_login)])
async def scenario_run(request: Request):
    form_data = await request.form()
    # Unchecked HTML checkboxes submit no field at all - normalize before validation.
    raw = dict(form_data)
    raw["rota_enabled"] = "rota_enabled" in form_data
    raw["campy_enabled"] = "campy_enabled" in form_data

    try:
        form = ScenarioFormInput.model_validate(raw)
        config = build_sveir_config(form)
    except (ValidationError, ValueError) as e:
        return _render(request, errors=str(e), status_code=422)

    try:
        record = await submit_job(config, form.model_dump())
    except QueueFullError as e:
        return _render(request, form_values=form.model_dump(), errors=str(e), status_code=429)

    return RedirectResponse(f"/runs/{record.job_id}", status_code=303)

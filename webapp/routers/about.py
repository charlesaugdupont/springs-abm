"""About-the-Model page: hand-written narrative sections plus a
parameter-trust table rendered directly from parameter_registry.py, so it
never drifts from the scenario builder's own info boxes (see that
module's docstring)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from webapp.auth import require_login
from webapp.parameter_registry import EVIDENCE_TIER_LABELS, EVIDENCE_TIER_ORDER, by_evidence_tier
from webapp.templating import templates

router = APIRouter()


@router.get("/about", response_class=HTMLResponse, dependencies=[Depends(require_login)])
def about(request: Request):
    grouped = by_evidence_tier()
    tiers = [
        {"key": t, "label": EVIDENCE_TIER_LABELS[t], "params": grouped[t]}
        for t in EVIDENCE_TIER_ORDER
    ]
    return templates.TemplateResponse(request, "about.html", {"authenticated": True, "tiers": tiers})

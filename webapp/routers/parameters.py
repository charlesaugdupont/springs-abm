"""GET /api/parameters: the single JSON view of webapp/parameter_registry.py
that both the frontend's scenario form and its About/trust-table page
render from. Replaces the old scenario_builder.html/about.html server-side
rendering - the registry itself is unchanged and remains the only source
of truth (see that module's docstring); this is a thin read-only view over
it, grouped both by category (for the form) and by evidence tier (for the
trust table) in one response so a single fetch serves both."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from webapp.auth import require_login
from webapp.parameter_registry import (
    CATEGORY_ORDER, EVIDENCE_TIER_LABELS, EVIDENCE_TIER_ORDER, ParamMeta,
    by_category, by_evidence_tier,
)
from webapp.scenario_form import FORM_FIELD_TO_PATH, RANGE_PAIR_TO_PATH, ScenarioFormInput

router = APIRouter()

_PATH_TO_FORM_FIELD: dict[str, str] = {
    **{path: name for name, path in FORM_FIELD_TO_PATH.items()},
    **{path: name for name, path in RANGE_PAIR_TO_PATH.items()},
}

_DEFAULTS = ScenarioFormInput()


def _param_json(meta: ParamMeta) -> dict:
    out = asdict(meta)
    if not meta.editable:
        return out

    form_name = _PATH_TO_FORM_FIELD[meta.path]
    out["form_name"] = form_name
    if meta.ui_widget == "range-pair":
        out["default"] = [
            getattr(_DEFAULTS, f"{form_name}_min"),
            getattr(_DEFAULTS, f"{form_name}_max"),
        ]
    else:
        out["default"] = getattr(_DEFAULTS, form_name)
    return out


@router.get("/parameters", dependencies=[Depends(require_login)])
def get_parameters():
    by_cat = []
    for cat, metas in by_category().items():
        if not metas:
            continue
        by_cat.append({
            "category": cat,
            "editable": [_param_json(m) for m in metas if m.editable],
            "readonly": [_param_json(m) for m in metas if not m.editable],
        })

    by_tier = []
    grouped_tiers = by_evidence_tier()
    for tier in EVIDENCE_TIER_ORDER:
        by_tier.append({
            "tier": tier,
            "label": EVIDENCE_TIER_LABELS[tier],
            "params": [_param_json(m) for m in grouped_tiers[tier]],
        })

    return {
        "category_order": CATEGORY_ORDER,
        "by_category": by_cat,
        "by_evidence_tier": by_tier,
    }

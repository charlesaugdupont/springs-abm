# experiments/gsa/param_space.py
"""
GSA parameter space, built FROM the parameter registry (single source of truth).

The web UI's parameter_registry (webapp/parameter_registry.py) already carries,
for every model parameter: whether it's user-editable, its category, and its UI
slider bounds (ui_min/ui_max). The GSA varies exactly the set of user-editable
model parameters over exactly those UI ranges - which is the whole point: "which
of the controls a user actually sees move the outputs, over the range they can
drag them?" Pulling the list and bounds from the registry (rather than
hardcoding) means the GSA can never drift out of sync with the UI.

Excluded: seed / number_agents / step_target are experiment controls, not model
inputs, and internal params (category="internal") are never user-facing.
"""
from __future__ import annotations

import numpy as np

from webapp.parameter_registry import REGISTRY, ParamMeta

# Experiment controls, not scientific model inputs.
GSA_EXCLUDE = {"seed", "number_agents", "step_target"}


def gsa_params(exclude: set[str] | None = None) -> list[ParamMeta]:
    """The registry entries the GSA varies: editable, non-internal, bounded."""
    exclude = GSA_EXCLUDE if exclude is None else exclude
    params = []
    for m in REGISTRY:
        if not m.editable or m.category == "internal":
            continue
        if m.ui_min is None or m.ui_max is None:
            continue
        if m.path in exclude:
            continue
        params.append(m)
    return params


def build_problem(params: list[ParamMeta]) -> dict:
    """SALib 'problem' dict for a list of ParamMeta (bounds = UI slider range)."""
    return {
        "num_vars": len(params),
        "names": [m.path for m in params],
        "bounds": [[float(m.ui_min), float(m.ui_max)] for m in params],
    }


def problem_from_selected(paths: list[str]) -> dict:
    """Stage-2 subset problem, preserving the given path order."""
    by_path = {m.path: m for m in gsa_params()}
    missing = [p for p in paths if p not in by_path]
    if missing:
        raise ValueError(f"Selected paths not in GSA param space: {missing}")
    return build_problem([by_path[p] for p in paths])


def _is_integer_path(path: str) -> bool:
    for m in REGISTRY:
        if m.path == path:
            return bool(getattr(m, "is_integer", False))
    return False


def combos_from_sample(X: np.ndarray, names: list[str]) -> list[dict]:
    """Turn a SALib sample matrix into orchestrator combo dicts.

    Integer-typed params (e.g. exposure_period, in days) are rounded to the
    nearest int so the config validation accepts them; everything else stays a
    float. A gsa.design_id tag is added downstream (in run_gsa), not here, so
    this stays reusable for problem construction.
    """
    int_flags = [_is_integer_path(n) for n in names]
    combos = []
    for row in X:
        combo = {}
        for name, val, is_int in zip(names, row, int_flags):
            combo[name] = int(round(float(val))) if is_int else float(val)
        combos.append(combo)
    return combos

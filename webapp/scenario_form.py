"""
The validated, server-authoritative parameter surface for scenario
submissions. Per parameter_registry.py's module docstring, the client's
HTML slider min/max/step attributes are a UX nicety only - the bounds
built into ScenarioFormInput below are what actually enforce limits,
re-checked on every request regardless of what the client submits.

ScenarioFormInput's editable numeric fields are generated FROM
parameter_registry.py's editable entries (bounds) and experiments/
orchestrator.py's get_param (live default values), rather than hand-typed
a second time - eliminating a whole class of "the form's bounds silently
drifted from the registry's documented bounds" bugs. build_sveir_config()
then applies submitted values back onto a fresh SVEIRConfig via
orchestrator.py's set_param - the same dot-path setter every sweep
experiment already uses - so the registry's path convention does real,
single-purpose work end-to-end: registry path -> form field -> set_param
call, never a raw pass-through of client-submitted JSON onto the config.
"""
from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, create_model

from config import CampylobacterConfig, RotavirusConfig, SVEIRCONFIG, SVEIRConfig
from experiments.orchestrator import get_param, set_param
from webapp.parameter_registry import REGISTRY_BY_PATH, editable_fields
from webapp.settings import DEFAULT_GRID_ID, MAX_AGENTS, MAX_STEPS, MIN_AGENTS, MIN_STEPS

_STRUCTURAL_PATHS = ("seed", "number_agents", "step_target")

# Fail loudly at import time if the registry's displayed slider bounds and the server's
# actually-enforced bounds ever drift apart again (see settings.py's comment on MAX_AGENTS).
assert (REGISTRY_BY_PATH["number_agents"].ui_min, REGISTRY_BY_PATH["number_agents"].ui_max) == (MIN_AGENTS, MAX_AGENTS)
assert (REGISTRY_BY_PATH["step_target"].ui_min, REGISTRY_BY_PATH["step_target"].ui_max) == (MIN_STEPS, MAX_STEPS)


def _form_field_name(path: str) -> str:
    """'pathogens[rota].vaccination_rate' -> 'rota_vaccination_rate'
    'steering_parameters.cost_of_care' -> 'cost_of_care'"""
    if path.startswith("pathogens["):
        bracket_end = path.index("]")
        pname = path[len("pathogens["):bracket_end]
        attr = path[bracket_end + 2:]
        return f"{pname}_{attr}"
    if path.startswith("steering_parameters."):
        return path[len("steering_parameters."):]
    return path.replace(".", "_")


_field_definitions: dict[str, Any] = {
    "seed": (int, Field(default=SVEIRCONFIG.seed, ge=0, le=2_147_483_647)),
    "number_agents": (int, Field(default=SVEIRCONFIG.number_agents, ge=MIN_AGENTS, le=MAX_AGENTS)),
    "step_target": (int, Field(default=SVEIRCONFIG.step_target, ge=MIN_STEPS, le=MAX_STEPS)),
    "rota_enabled": (bool, Field(default=True)),
    "campy_enabled": (bool, Field(default=True)),
}
# seed/number_agents/step_target use an identity mapping (top-level SVEIRConfig fields,
# no pathogens[]/steering_parameters. prefix to strip) - registered here too, not just in
# _field_definitions above, so callers resolving path -> form field (e.g. the scenario
# builder template, via routers/scenario.py's PATH_TO_FORM_FIELD) find them like any other
# registry-driven field rather than needing special-cased lookups.
FORM_FIELD_TO_PATH: dict[str, str] = {p: p for p in _STRUCTURAL_PATHS}

for _meta in editable_fields():
    if _meta.path in _STRUCTURAL_PATHS:
        continue  # handled explicitly above (hosting caps, not registry ui_max)
    _form_name = _form_field_name(_meta.path)
    _default = float(get_param(SVEIRCONFIG, _meta.path))
    _field_definitions[_form_name] = (float, Field(default=_default, ge=_meta.ui_min, le=_meta.ui_max))
    FORM_FIELD_TO_PATH[_form_name] = _meta.path

ScenarioFormInput = create_model(
    "ScenarioFormInput",
    __config__=ConfigDict(extra="forbid"),
    **_field_definitions,
)


def build_sveir_config(form: "ScenarioFormInput") -> SVEIRConfig:
    if not form.rota_enabled and not form.campy_enabled:
        raise ValueError("At least one pathogen must be enabled.")

    cfg = SVEIRConfig()
    cfg.seed = form.seed
    cfg.number_agents = form.number_agents
    cfg.step_target = form.step_target
    cfg.spatial_creation_args.grid_id = DEFAULT_GRID_ID  # always hardcoded - never from the client

    pathogens = []
    if form.rota_enabled:
        pathogens.append(RotavirusConfig())
    if form.campy_enabled:
        pathogens.append(CampylobacterConfig())
    cfg.pathogens = pathogens

    active = {name for name, on in (("rota", form.rota_enabled), ("campy", form.campy_enabled)) if on}
    for form_name, path in FORM_FIELD_TO_PATH.items():
        if path.startswith("pathogens["):
            pname = path[len("pathogens["):path.index("]")]
            if pname not in active:
                continue  # no object to set this on - the pathogen isn't in cfg.pathogens
        set_param(cfg, path, getattr(form, form_name))

    return cfg

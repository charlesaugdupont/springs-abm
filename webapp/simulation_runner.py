"""
Purpose-built simulation-running function for the web UI. Deliberately NOT
a reuse of main.py::run_single_simulation() - the UI needs richer per-day
data (a day-by-day spatial animation, all-ages prevalence, household
wealth/care-seeking trajectories) that the CLI's function doesn't produce,
and main.py is left untouched per the plan. This module does not import
from main.py.

Drives the model with the same manual day-by-day stepping pattern already
established in experiments/orchestrator.py::_run_one (see that function's
step_callback branch) instead of calling model.run(), so a per-day capture
hook can run between steps.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from config import SVEIRConfig
from abm.constants import AgentPropertyKeys, Compartment
from abm.environment.grid_constants import GRID_SIZE
from abm.model.initialize_model import SVEIRModel
from abm.utils.rng import set_global_seed
from experiments.metrics import (
    campy_route_fractions, care_seeking_metrics, epidemic_metrics, wellbeing_metrics,
)
from webapp import jobs
from webapp.jobs import SimResultBundle

# Downsampled resolution for the results-page spatial scrubber - see the
# plan's Finding #3: a full GRID_SIZE x GRID_SIZE array per simulated day
# would be too large a JSON payload for a day-by-day animation.
SPATIAL_GRID_SIZE = 25
_BLOCK = GRID_SIZE // SPATIAL_GRID_SIZE
assert _BLOCK * SPATIAL_GRID_SIZE == GRID_SIZE, (
    f"GRID_SIZE ({GRID_SIZE}) must be evenly divisible by SPATIAL_GRID_SIZE ({SPATIAL_GRID_SIZE})"
)


def _household_mean_wealth(household_id: np.ndarray, wealth: np.ndarray) -> float:
    """Same dedup approach as experiments/metrics.py::wellbeing_metrics - wealth is pooled
    per household and duplicated across every member, so a naive per-agent mean would
    over-weight large households."""
    _, hh_first_idx = np.unique(household_id, return_index=True)
    return float(wealth[hh_first_idx].mean())


def _spatial_grid(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> list[list[float]]:
    """Cumulative infection-density grid, downsampled from GRID_SIZE to SPATIAL_GRID_SIZE via
    block-summing (matches run_viz.py's full-resolution binning approach, just coarser for a
    small, animatable per-day payload)."""
    xi = np.clip(x.astype(int), 0, GRID_SIZE - 1)
    yi = np.clip(y.astype(int), 0, GRID_SIZE - 1)
    heat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    np.add.at(heat, (yi, xi), weights)
    downsampled = heat.reshape(SPATIAL_GRID_SIZE, _BLOCK, SPATIAL_GRID_SIZE, _BLOCK).sum(axis=(1, 3))
    return downsampled.tolist()


def _as_numpy(t) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _downsample(layer: np.ndarray, agg: str) -> list[list[float]]:
    """Downsample a full-res (GRID_SIZE, GRID_SIZE) [y, x] layer to SPATIAL_GRID_SIZE with the
    same block-binning as _spatial_grid. agg='sum' for point/count masks, 'mean' for continuous
    [0, 1] density/fraction layers - result is [y_bin][x_bin], aligned with spatial_daily_grids."""
    blocks = layer.reshape(SPATIAL_GRID_SIZE, _BLOCK, SPATIAL_GRID_SIZE, _BLOCK)
    reduced = blocks.sum(axis=(1, 3)) if agg == "sum" else blocks.mean(axis=(1, 3))
    return reduced.astype(float).tolist()


def _static_layers(model: SVEIRModel, config: SVEIRConfig, pathogen_names: list[str]) -> dict[str, list[list[float]]]:
    """Static/reference spatial layers for the results-map overlays (households, animal density,
    schools/worship/water points, water bodies), all downsampled to the same SPATIAL_GRID_SIZE
    [y][x] frame as the infection grids so they line up with the basemap. Best-effort: each block
    is guarded so an optional overlay never breaks a simulation run."""
    layers: dict[str, list[list[float]]] = {}
    g = model.graph

    # Household locations: dedup by household and bin their home cells. The day loop ends on the
    # night phase (everyone home), so the final X/Y are home cells.
    try:
        household_id = g.ndata[AgentPropertyKeys.HOUSEHOLD_ID].cpu().numpy()
        x = g.ndata[AgentPropertyKeys.X].cpu().numpy()
        y = g.ndata[AgentPropertyKeys.Y].cpu().numpy()
        _, first_idx = np.unique(household_id, return_index=True)
        hx = np.clip(x[first_idx].astype(int), 0, GRID_SIZE - 1)
        hy = np.clip(y[first_idx].astype(int), 0, GRID_SIZE - 1)
        hh = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        np.add.at(hh, (hy, hx), 1.0)
        layers["household_density"] = _downsample(hh, "sum")
    except Exception:  # pragma: no cover - optional overlay, never fatal
        pass

    # Animal density (weighted poultry + ruminant) - only built when campy is enabled.
    if "campy" in pathogen_names:
        try:
            poultry = _as_numpy(model.grid_environment.get_dynamic_layer("poultry_density"))
            ruminant = _as_numpy(model.grid_environment.get_dynamic_layer("ruminant_density"))
            campy = next(p for p in model.pathogens if p.name == "campy")
            w_p = float(getattr(campy.config, "poultry_weight", 1.0))
            w_r = float(getattr(campy.config, "ruminant_weight", 0.45))
            combined = np.clip(poultry * w_p + ruminant * w_r, 0.0, 1.0)
            layers["animal_density"] = _downsample(combined, "mean")
        except Exception:  # pragma: no cover
            pass

    # Static POI / water-body masks baked into the grid file for this grid_id.
    try:
        grid_id = config.spatial_creation_args.grid_id
        data = np.load(f"grids/{grid_id}/grid.npz", allow_pickle=True)
        grid = data["grid"]
        property_map = data["property_map"].item()
        name_to_idx = {v: k for k, v in property_map.items()}
        for name, agg in (("school", "sum"), ("place_of_worship", "sum"),
                          ("water", "sum"), ("natural_water", "mean")):
            if name in name_to_idx:
                layers[name] = _downsample(grid[:, :, name_to_idx[name]].astype(float), agg)
    except Exception:  # pragma: no cover
        pass

    return layers


def _capture_daily_snapshot(model: SVEIRModel, daily: dict[str, list]) -> None:
    """Called once per simulated day, right after model.step(). Reads model.graph.ndata
    directly - cheap vectorized ops, capturing state that isn't tracked historically anywhere
    else in the model (see the plan's Finding #4)."""
    g = model.graph

    for p in model.pathogens:
        status = g.ndata[AgentPropertyKeys.status(p.name)].cpu().numpy()
        daily[f"all_ages_prevalence_{p.name}"].append(float((status == Compartment.INFECTIOUS).mean()))

        # Campylobacter attributes each new infection to exactly one of its three
        # routes. Its per-day counters are reset at the start of each day and
        # incremented during that day's transmission/progression, so right here
        # (immediately after model.step()) they hold THIS day's new infections by
        # route. Captured only when campy is enabled.
        if p.name == "campy" and hasattr(p, "cases_zoonotic"):
            daily["campy_cases_zoonotic"].append(float(p.cases_zoonotic))
            daily["campy_cases_fecal_oral"].append(float(p.cases_fecal_oral))
            daily["campy_cases_food_borne"].append(float(p.cases_food_borne))

    x = g.ndata[AgentPropertyKeys.X].cpu().numpy()
    y = g.ndata[AgentPropertyKeys.Y].cpu().numpy()
    household_id = g.ndata[AgentPropertyKeys.HOUSEHOLD_ID].cpu().numpy()
    wealth = g.ndata[AgentPropertyKeys.WEALTH].cpu().numpy()
    care_seeking_count = g.ndata[AgentPropertyKeys.CARE_SEEKING_COUNT].cpu().numpy()

    daily["mean_household_wealth"].append(_household_mean_wealth(household_id, wealth))
    # CARE_SEEKING_COUNT is a per-agent counter that only ever increments, so summing it each
    # day directly gives a running total - no need to diff against the previous day.
    daily["cumulative_care_seeking_events"].append(float(care_seeking_count.sum()))

    infection_weights = np.zeros(len(x), dtype=float)
    for p in model.pathogens:
        infection_weights += g.ndata[AgentPropertyKeys.num_infections(p.name)].cpu().numpy()
    daily["spatial_grid"].append(_spatial_grid(x, y, infection_weights))


def run_simulation_for_ui(config: SVEIRConfig, job_id: str) -> SimResultBundle:
    """Runs one simulation for the web UI. This function blocks for the full run duration -
    callers must invoke it from a worker thread (see webapp/executor.py), never directly from
    an async route handler."""
    t0 = time.time()
    set_global_seed(config.seed)

    model = SVEIRModel(model_identifier=f"webapp_{job_id}", root_path="outputs/webapp")
    model.set_model_parameters(**config.model_dump())
    model.initialize_model(verbose=False)

    # Manual day-by-day driver, replicating model.run()'s own setup exactly (the same
    # established pattern as experiments/orchestrator.py::_run_one's step_callback branch) so
    # u5_prevalence_history is populated identically to model.run(), while also letting us
    # capture additional per-day snapshots between steps.
    model.infection_incidence.clear()
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    child_mask = model.graph.ndata[AgentPropertyKeys.IS_CHILD]

    daily: dict[str, list] = defaultdict(list)
    for day in range(config.step_target):
        model.step(child_mask=child_mask)
        _capture_daily_snapshot(model, daily)
        jobs.set_progress(job_id, day + 1)

    pathogen_names = [p.name for p in model.pathogens]
    n_u5 = int(child_mask.sum().item())
    days = list(range(config.step_target))

    # Cumulative under-5 illness-days, derived post-hoc from u5_prevalence_history rather than
    # captured fresh - this reconstructs experiments/metrics.py::epidemic_metrics's
    # cumulative_u5_days formula (prev.sum() * n_u5) as a running partial sum, so the final
    # day's value is guaranteed identical to that already-established metric (Finding #4).
    cumulative_u5_illness_days = {
        name: (np.cumsum(np.array(series)) * n_u5).tolist()
        for name, series in model.u5_prevalence_history.items()
    }

    summary_metrics = {
        **epidemic_metrics(model),
        **care_seeking_metrics(model),
        **wellbeing_metrics(model),
        **campy_route_fractions(model),
    }

    # Per-day new-infection counts split by Campylobacter's three routes (empty
    # when campy is disabled). Feeds the results page's 100%-stacked route plot.
    campy_daily_infections_by_route: dict[str, list[float]] = {}
    if "campy" in pathogen_names:
        campy_daily_infections_by_route = {
            "zoonotic": daily["campy_cases_zoonotic"],
            "fecal_oral": daily["campy_cases_fecal_oral"],
            "food_borne": daily["campy_cases_food_borne"],
        }

    return SimResultBundle(
        config_snapshot=config.model_dump(),
        pathogen_names=pathogen_names,
        days=days,
        u5_prevalence=dict(model.u5_prevalence_history),
        all_ages_prevalence={name: daily[f"all_ages_prevalence_{name}"] for name in pathogen_names},
        cumulative_u5_illness_days=cumulative_u5_illness_days,
        mean_household_wealth=daily["mean_household_wealth"],
        cumulative_care_seeking_events=daily["cumulative_care_seeking_events"],
        spatial_grid_size=SPATIAL_GRID_SIZE,
        spatial_daily_grids=daily["spatial_grid"],
        static_layers=_static_layers(model, config, pathogen_names),
        campy_daily_infections_by_route=campy_daily_infections_by_route,
        summary_metrics=summary_metrics,
        proportion_infected_at_least_once=model.get_proportion_infected_at_least_once(),
        n_u5=n_u5,
        runtime_seconds=time.time() - t0,
    )

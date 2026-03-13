import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

from config import SVEIRCONFIG
from abm.model.initialize_model import SVEIRModel
from abm.constants import AgentPropertyKeys, Compartment
from abm.environment.grid_generator import (
    create_and_save_realistic_grid,
    get_grid_id,
    AKUSE_BOUNDARY_COORDS,
    GRID_SIZE,
    OSM_POI_TAGS,
    PROCEDURAL_POI_COUNTS,
)
from abm.utils.rng import set_global_seed, GRID_GENERATION_SEED

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def ensure_grid_exists() -> str:
    """
    Ensure that the realistic OSM-based grid exists on disk.
    Returns the grid_id that should be used in the config.
    """
    grid_id = get_grid_id(
        AKUSE_BOUNDARY_COORDS,
        GRID_SIZE,
        OSM_POI_TAGS,
        PROCEDURAL_POI_COUNTS,
    )
    grid_dir = os.path.join("grids", grid_id)
    grid_file = os.path.join(grid_dir, "grid.npz")

    if not os.path.exists(grid_file):
        set_global_seed(GRID_GENERATION_SEED)
        create_and_save_realistic_grid()
    return grid_id


def build_config_from_ui():
    """
    Take the base SVEIRCONFIG and override values based on UI controls.
    Returns config.
    """
    cfg = SVEIRCONFIG.model_copy(deep=True)
    ste = cfg.steering_parameters

    # --- Basic run settings ---
    st.sidebar.markdown("# Run Settings")
    cfg.number_agents = st.sidebar.slider(
        "Number of agents",
        min_value=500,
        max_value=10000,
        value=3000,
        step=500,
    )
    cfg.step_target = st.sidebar.slider(
        "Number of steps (days)",
        min_value=20,
        max_value=365,
        value=150,
        step=10,
    )
    cfg.seed = int(st.sidebar.number_input("Random seed", min_value=0, value=cfg.seed, step=1))
    cfg.device = "cpu"   # safer & simpler for Streamlit
    cfg.spatial = True   # use the spatial grid

    # We disable the built-in Zarr data collection in the model
    ste.data_collection_period = 0
    ste.ndata = None

    # --- Demography & global infection modifiers ---
    st.sidebar.markdown("# Demography & Global Modifiers")
    cfg.average_household_size = st.sidebar.slider(
        "Average household size",
        min_value=2.0,
        max_value=6.0,
        value=float(cfg.average_household_size),
        step=0.1,
    )
    cfg.child_probability = st.sidebar.slider(
        "Probability that a non-first household member is a child",
        min_value=0.0,
        max_value=0.4,
        value=float(cfg.child_probability),
        step=0.01,
    )
    ste.infection_reduction_factor_per_health_unit = st.sidebar.slider(
        "Infection reduction per unit of health (higher = stronger protection)",
        min_value=0.0,
        max_value=2.0,
        value=float(ste.infection_reduction_factor_per_health_unit),
        step=0.05,
    )
    ste.prior_infection_immunity_factor = st.sidebar.slider(
        "Immunity factor per prior infection",
        min_value=0.0,
        max_value=3.0,
        value=float(ste.prior_infection_immunity_factor),
        step=0.1,
    )

    # --- Rotavirus parameters ---
    st.sidebar.markdown("# Rotavirus")
    rota_init_inf = st.sidebar.slider(
        "Rota initial exposed proportion",
        min_value=0.0,
        max_value=0.2,
        value=float(cfg.pathogens[0].initial_exposed_proportion),
        step=0.005,
    )
    rota_inf_mean = st.sidebar.slider(
        "Rota infection_prob_mean",
        min_value=0.0,
        max_value=0.5,
        value=float(cfg.pathogens[0].infection_prob_mean),
        step=0.005,
    )
    rota_inf_std = st.sidebar.slider(
        "Rota infection_prob_std",
        min_value=0.0,
        max_value=0.1,
        value=float(cfg.pathogens[0].infection_prob_std),
        step=0.005,
    )
    rota_rec = st.sidebar.slider(
        "Rota recovery_rate",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.pathogens[0].recovery_rate),
        step=0.05,
    )
    rota_exp = st.sidebar.slider(
        "Rota exposure_period (days)",
        min_value=1,
        max_value=10,
        value=int(cfg.pathogens[0].exposure_period),
        step=1,
    )
    rota_vacc_rate = st.sidebar.slider(
        "Rota vaccination_rate (per day)",
        min_value=0.0,
        max_value=0.1,
        value=float(cfg.pathogens[0].vaccination_rate),
        step=0.005,
    )
    rota_vacc_eff = st.sidebar.slider(
        "Rota vaccine_efficacy",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.pathogens[0].vaccine_efficacy),
        step=0.05,
    )

    # --- Campylobacter parameters ---
    st.sidebar.markdown("# Campylobacter")
    campy_init_inf = st.sidebar.slider(
        "Campy initial exposed proportion",
        min_value=0.0,
        max_value=0.2,
        value=float(cfg.pathogens[1].initial_exposed_proportion),
        step=0.005,
    )
    campy_rec = st.sidebar.slider(
        "Campy recovery_rate",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.pathogens[1].recovery_rate),
        step=0.05,
    )
    campy_exp = st.sidebar.slider(
        "Campy exposure_period (days)",
        min_value=1,
        max_value=10,
        value=int(cfg.pathogens[1].exposure_period),
        step=1,
    )
    campy_interaction = st.sidebar.slider(
        "Human–animal interaction rate",
        min_value=0.0,
        max_value=10.0,
        value=float(cfg.pathogens[1].human_animal_interaction_rate),
        step=0.5,
    )

    # --- Water & environment parameters ---
    st.sidebar.markdown("# Water & Environment")
    ste.human_to_water_infection_prob = st.sidebar.slider(
        "Human → water infection prob",
        min_value=0.0,
        max_value=0.5,
        value=float(ste.human_to_water_infection_prob),
        step=0.005,
    )
    ste.water_to_human_infection_prob = st.sidebar.slider(
        "Water → human infection prob",
        min_value=0.0,
        max_value=0.5,
        value=float(ste.water_to_human_infection_prob),
        step=0.005,
    )
    ste.water_recovery_prob = st.sidebar.slider(
        "Daily water recovery prob",
        min_value=0.0,
        max_value=1.0,
        value=float(ste.water_recovery_prob),
        step=0.05,
    )
    ste.shock_daily_prob = st.sidebar.slider(
        "Daily water shock prob",
        min_value=0.0,
        max_value=0.5,
        value=float(ste.shock_daily_prob),
        step=0.01,
    )

    # --- Care seeking parameters ---
    st.sidebar.markdown("# Care-Seeking")
    ste.cost_of_care = st.sidebar.slider(
        "Cost of care (fraction of max wealth)",
        min_value=0.0,
        max_value=0.5,
        value=float(ste.cost_of_care),
        step=0.01,
    )
    ste.treatment_success_prob = st.sidebar.slider(
        "Treatment success probability",
        min_value=0.0,
        max_value=1.0,
        value=float(ste.treatment_success_prob),
        step=0.05,
    )
    ste.natural_worsening_prob = st.sidebar.slider(
        "Natural worsening probability (if no care)",
        min_value=0.0,
        max_value=1.0,
        value=float(ste.natural_worsening_prob),
        step=0.05,
    )

    # --- Economics parameters ---
    st.sidebar.markdown("# Economic Feedback")
    ste.health_based_income = st.sidebar.checkbox(
        "Health-based income",
        value=bool(ste.health_based_income),
    )
    ste.daily_income_rate = st.sidebar.slider(
        "Daily income rate (for perfectly healthy adult)",
        min_value=0.0,
        max_value=0.2,
        value=float(ste.daily_income_rate),
        step=0.005,
    )
    ste.daily_cost_of_living = st.sidebar.slider(
        "Daily cost of living",
        min_value=0.0,
        max_value=0.1,
        value=float(ste.daily_cost_of_living),
        step=0.005,
    )

    # Write back pathogen-specific params
    for p in cfg.pathogens:
        if p.name == "rota":
            p.initial_exposed_proportion = rota_init_inf
            p.infection_prob_mean = rota_inf_mean
            p.infection_prob_std = rota_inf_std
            p.recovery_rate = rota_rec
            p.exposure_period = rota_exp
            p.vaccination_rate = rota_vacc_rate
            p.vaccine_efficacy = rota_vacc_eff
        elif p.name == "campy":
            p.initial_exposed_proportion = campy_init_inf
            p.recovery_rate = campy_rec
            p.exposure_period = campy_exp
            p.human_animal_interaction_rate = campy_interaction

    # Attach grid id
    grid_id = ensure_grid_exists()
    cfg.spatial_creation_args.grid_id = grid_id

    return cfg


def run_simulation(config):
    """
    Run a single SVEIRModel simulation with the given config and return:
    - model
    - child_incidence / child_prevalence time series (children only)
    """
    set_global_seed(config.seed)
    root_path = "streamlit_outputs"
    Path(root_path).mkdir(parents=True, exist_ok=True)

    model = SVEIRModel(model_identifier="ui_run", root_path=root_path)
    model.set_model_parameters(**config.model_dump())
    model.initialize_model(verbose=False)

    # --- child-only incidence & prevalence ---
    g = model.graph
    is_child_t = g.ndata[AgentPropertyKeys.IS_CHILD].bool()

    # baseline child infection counts per pathogen
    prev_child_infections = {}
    for p in model.pathogens:
        key = AgentPropertyKeys.num_infections(p.name)
        prev_child_infections[p.name] = g.ndata[key][is_child_t].sum().item()

    child_incidence = []
    child_prevalence = []

    for _ in range(config.step_target):
        model.step()

        # Child-only prevalence: sum of INFECTIOUS children over pathogens
        child_prev_this_step = 0
        for p in model.pathogens:
            status_key = AgentPropertyKeys.status(p.name)
            status = model.graph.ndata[status_key]
            child_prev_this_step += (status[is_child_t] == Compartment.INFECTIOUS).sum().item()
        child_prevalence.append(child_prev_this_step)

        # Child-only incidence: increase in num_infections among children
        child_inc_this_step = 0
        for p in model.pathogens:
            key = AgentPropertyKeys.num_infections(p.name)
            current = model.graph.ndata[key][is_child_t].sum().item()
            delta = current - prev_child_infections[p.name]
            child_inc_this_step += max(0, int(delta))
            prev_child_infections[p.name] = current
        child_incidence.append(child_inc_this_step)

    child_incidence = np.array(child_incidence)
    child_prevalence = np.array(child_prevalence)

    return model, child_incidence, child_prevalence


def child_metrics(model: SVEIRModel) -> dict:
    """
    Compute summary metrics for children under 5 (age < 60 months).
    Returns a dict of metrics and arrays useful for plotting.
    """
    g = model.graph

    is_child = g.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age_months = g.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    under5 = is_child & (age_months < 60.0)

    n_children_total = is_child.sum()
    n_u5 = under5.sum()

    metrics = {
        "n_children_total": n_children_total,
        "n_u5": n_u5,
    }

    for name in ["rota", "campy"]:
        key = AgentPropertyKeys.num_infections(name)
        if key in g.ndata:
            num_inf = g.ndata[key].cpu().numpy()
            ever_infected_u5 = (num_inf[under5] > 0).sum()
            metrics[f"{name}_ever_infected_u5"] = int(ever_infected_u5)
            metrics[f"{name}_num_infections_u5"] = num_inf[under5]
        else:
            metrics[f"{name}_ever_infected_u5"] = 0
            metrics[f"{name}_num_infections_u5"] = np.array([])

    severity = g.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY].cpu().numpy()
    duration = g.ndata[AgentPropertyKeys.ILLNESS_DURATION].cpu().numpy()

    metrics["severity_u5"] = severity[under5]
    metrics["duration_u5"] = duration[under5]

    return metrics


def plot_epidemic_curves(incidence: np.ndarray, prevalence: np.ndarray):
    df = pd.DataFrame(
        {
            "day": np.arange(len(prevalence)),
            "incidence": incidence,
            "prevalence": prevalence,
        }
    ).set_index("day")
    st.subheader("Epidemic Curves (Children <= 5)")
    st.line_chart(df)


def _plot_discrete_hist(ax, data, title, xlabel, ylabel):
    """Plot a discrete histogram for integer-valued data using bars."""
    if data.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    data_int = data.astype(int)
    values, counts = np.unique(data_int, return_counts=True)

    ax.bar(values, counts, width=0.8, align="center", edgecolor="black")
    ax.set_xticks(values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="SPRINGS-ABM interactive UI", layout="wide")

st.title("SPRINGS-ABM – Interactive Simulation")
st.markdown(
    """
This interface wraps the existing SPRINGS-ABM model.

**Workflow:**

1. Adjust parameters in the sidebar (demography, pathogens, water, care-seeking, economics).
2. Press **▶ Run simulation** to run a single full simulation.
3. Inspect epidemic curves
"""
)

# Build config from UI
config = build_config_from_ui()

col_run, col_info = st.columns([1, 3])
with col_run:
    run_button = st.button("▶ Run simulation", type="primary")
with col_info:
    st.write(
        f"Current configuration: {config.number_agents} agents, "
        f"{config.step_target} steps, grid ID `{config.spatial_creation_args.grid_id}`."
    )

if run_button:
    with st.spinner("Running simulation... this may take a moment."):
        model, incidence, prevalence = run_simulation(config)

    st.success("Simulation finished.")

    # Epidemic curves
    plot_epidemic_curves(incidence, prevalence)

    st.subheader("Child Infection Metrics")

    g = model.graph
    is_child_t = g.ndata[AgentPropertyKeys.IS_CHILD].bool()
    n_children = int(is_child_t.sum().item())

    # total infections among children (all pathogens)
    child_total_infections = 0
    ever_infected_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    for p in model.pathogens:
        key = AgentPropertyKeys.num_infections(p.name)
        num_inf = g.ndata[key]
        child_total_infections += int(num_inf[is_child_t].sum().item())
        ever_infected_mask |= (num_inf > 0)

    if n_children > 0:
        prop_children_infected = (ever_infected_mask & is_child_t).sum().item() / n_children
    else:
        prop_children_infected = 0.0

    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Total infections among children (all pathogens)", f"{child_total_infections}")
    mcol2.metric("Proportion of children infected at least once", f"{prop_children_infected:.2%}")

    # Child metrics & histograms (final snapshot)
    metrics = child_metrics(model)

    st.subheader("Children under 5 – summary metrics")
    rota_ever = metrics.get("rota_ever_infected_u5", 0)
    campy_ever = metrics.get("campy_ever_infected_u5", 0)
    if metrics["n_u5"] > 0:
        rota_prop = rota_ever / metrics["n_u5"]
        campy_prop = campy_ever / metrics["n_u5"]
    else:
        rota_prop = campy_prop = 0.0

    st.markdown(
        f"- Rotavirus: {rota_ever} / {metrics['n_u5']} children under 5 ever infected ({rota_prop:.1%}).\n"
        f"- Campylobacter: {campy_ever} / {metrics['n_u5']} children under 5 ever infected ({campy_prop:.1%})."
    )

    # Episodes per child-year
    sim_years = config.step_target / 365.0 if config.step_target > 0 else 1.0
    rota_inf_counts = metrics.get("rota_num_infections_u5", np.array([]))
    campy_inf_counts = metrics.get("campy_num_infections_u5", np.array([]))

    if rota_inf_counts.size > 0 and metrics["n_u5"] > 0:
        rota_mean_inf = rota_inf_counts.mean()
        rota_ep_yr = rota_mean_inf / sim_years
    else:
        rota_mean_inf = rota_ep_yr = 0.0

    if campy_inf_counts.size > 0 and metrics["n_u5"] > 0:
        campy_mean_inf = campy_inf_counts.mean()
        campy_ep_yr = campy_mean_inf / sim_years
    else:
        campy_mean_inf = campy_ep_yr = 0.0

    st.markdown(
        f"- **Rotavirus**: mean {rota_mean_inf:.2f} infections per child under 5 "
        f"over {config.step_target} days (~{rota_ep_yr:.2f} episodes/child-year).\n"
        f"- **Campylobacter**: mean {campy_mean_inf:.2f} infections per child under 5 "
        f"over {config.step_target} days (~{campy_ep_yr:.2f} episodes/child-year)."
    )

else:
    st.info("Adjust parameters in the sidebar and press **▶ Run simulation** to start.")
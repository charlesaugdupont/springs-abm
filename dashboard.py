import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
        # First-time creation can take a bit (OSM download)
        create_and_save_realistic_grid()
    return grid_id


def build_config_from_ui():
    """
    Take the base SVEIRCONFIG and override values based on UI controls.
    Returns (config, collect_history_flag).
    """
    cfg = SVEIRCONFIG.model_copy(deep=True)
    ste = cfg.steering_parameters

    # --- Basic run settings ---
    st.sidebar.markdown("### Run settings")
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
    cfg.seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))
    cfg.device = "cpu"   # safer & simpler for Streamlit
    cfg.spatial = True   # use the spatial grid

    # We disable the built-in Zarr data collection in the model
    ste.data_collection_period = 0
    ste.ndata = None

    # --- Demography & global infection modifiers ---
    st.sidebar.markdown("### Demography & global modifiers")
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
    st.sidebar.markdown("### Rotavirus parameters")
    rota_init_inf = st.sidebar.slider(
        "Rota initial infected proportion",
        min_value=0.0,
        max_value=0.2,
        value=float(cfg.pathogens[0].initial_infected_proportion),
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
    st.sidebar.markdown("### Campylobacter parameters")
    campy_init_inf = st.sidebar.slider(
        "Campy initial infected proportion",
        min_value=0.0,
        max_value=0.2,
        value=float(cfg.pathogens[1].initial_infected_proportion),
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
    st.sidebar.markdown("### Water & environment")
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
    st.sidebar.markdown("### Care-seeking behavior")
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
    st.sidebar.markdown("### Economic feedback")
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

    # --- Child illness history collection (in-memory, not Zarr) ---
    st.sidebar.markdown("### Data collection (history)")
    collect_history = st.sidebar.checkbox(
        "Collect full child illness history in memory (slower)",
        value=False,
        help="Stores symptom severity and illness duration each day in memory; "
             "used to summarize total sick days and peak severity for children under 5."
    )

    # Write back pathogen-specific params
    for p in cfg.pathogens:
        if p.name == "rota":
            p.initial_infected_proportion = rota_init_inf
            p.infection_prob_mean = rota_inf_mean
            p.infection_prob_std = rota_inf_std
            p.recovery_rate = rota_rec
            p.exposure_period = rota_exp
            p.vaccination_rate = rota_vacc_rate
            p.vaccine_efficacy = rota_vacc_eff
        elif p.name == "campy":
            p.initial_infected_proportion = campy_init_inf
            p.recovery_rate = campy_rec
            p.exposure_period = campy_exp
            p.human_animal_interaction_rate = campy_interaction

    # Attach grid id
    grid_id = ensure_grid_exists()
    cfg.spatial_creation_args.grid_id = grid_id

    return cfg, collect_history


def run_simulation(config: SVEIRCONFIG, collect_history: bool):
    """
    Run a single SVEIRModel simulation with the given config and return:
    - model
    - incidence / prevalence time series
    - history (or None): dict with per-step severity and duration if collect_history=True
    """
    root_path = "streamlit_outputs"
    Path(root_path).mkdir(parents=True, exist_ok=True)

    model = SVEIRModel(model_identifier="ui_run", root_path=root_path)
    model.set_model_parameters(**config.model_dump())
    model.initialize_model(verbose=False)

    history = None
    if collect_history:
        history = {
            "symptom_severity": [],   # list of arrays (n_agents,) per step
            "illness_duration": [],
            "is_child": model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy(),
            "age": model.graph.ndata[AgentPropertyKeys.AGE].cpu().numpy(),
        }

    # Manual loop so we can capture per-step states
    for _ in range(config.step_target):
        model.step()
        if collect_history:
            sev = model.graph.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY].cpu().numpy()
            dur = model.graph.ndata[AgentPropertyKeys.ILLNESS_DURATION].cpu().numpy()
            history["symptom_severity"].append(sev)
            history["illness_duration"].append(dur)

    incidence = np.array(model.infection_incidence)
    prevalence = np.array(model.prevalence_history)

    if collect_history:
        # Convert lists of (n_agents,) into arrays shape (n_agents, n_time)
        history["symptom_severity"] = np.stack(history["symptom_severity"], axis=1)
        history["illness_duration"] = np.stack(history["illness_duration"], axis=1)

    return model, incidence, prevalence, history


def child_metrics(model: SVEIRModel) -> dict:
    """
    Compute summary metrics for children under 5 (age < 60 months).
    Returns a dict of metrics and arrays useful for plotting.
    """
    g = model.graph

    is_child = g.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age_months = g.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    under5 = is_child & (age_months < 60.0)

    status_rota = g.ndata[AgentPropertyKeys.status("rota")].cpu().numpy()
    status_campy = g.ndata[AgentPropertyKeys.status("campy")].cpu().numpy()

    ever_infectious_rota = np.any(status_rota[is_child] == Compartment.INFECTIOUS)
    ever_infectious_campy = np.any(status_campy[is_child] == Compartment.INFECTIOUS)

    st.markdown(
        f"**DEBUG:** any children infectious at final step? "
        f"Rota={ever_infectious_rota}, Campy={ever_infectious_campy}"
    )

    # Final snapshot of severity/duration
    sev_final = g.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY].cpu().numpy()
    dur_final = g.ndata[AgentPropertyKeys.ILLNESS_DURATION].cpu().numpy()
    st.markdown(
        f"**DEBUG:** final snapshot - max severity={sev_final.max():.4f}, "
        f"max illness_duration={dur_final.max()}"
    )


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
    st.subheader("Epidemic curves")
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


def plot_child_histograms(metrics: dict):
    """
    Plot histograms for infections / severity / duration among children under 5.
    Uses discrete bar plots for count variables.
    """
    n_u5 = metrics["n_u5"]
    if n_u5 == 0:
        st.info("No children under 5 in this simulation (check demographic parameters).")
        return

    rota_inf = metrics["rota_num_infections_u5"]
    campy_inf = metrics["campy_num_infections_u5"]
    severity = metrics["severity_u5"]
    duration = metrics["duration_u5"]

    severity_nonzero = severity[severity > 0] if severity.size > 0 else np.array([])
    duration_nonzero = duration[duration > 0] if duration.size > 0 else np.array([])

    st.subheader("Children under 5: infection & illness distributions (final day snapshot)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # 1. Rotavirus infections per child (discrete)
    _plot_discrete_hist(
        axes[0],
        rota_inf,
        "Rotavirus infections per child (u5)",
        "Number of infections over simulation",
        "Number of children",
    )

    # 2. Campylobacter infections per child (discrete)
    _plot_discrete_hist(
        axes[1],
        campy_inf,
        "Campylobacter infections per child (u5)",
        "Number of infections over simulation",
        "Number of children",
    )

    # 3. Final-day symptom severity
    ax2 = axes[2]
    if severity_nonzero.size > 0:
        ax2.hist(severity_nonzero, bins=20, range=(0, 1), edgecolor="black")
        ax2.set_title("Final-day symptom severity (u5, currently sick)")
        ax2.set_xlabel("Severity (0–1)")
        ax2.set_ylabel("Number of children")
    else:
        ax2.text(0.5, 0.5, "No sick children at final step", ha="center", va="center")
        ax2.set_axis_off()

    # 4. Remaining illness duration in days (discrete)
    _plot_discrete_hist(
        axes[3],
        duration_nonzero,
        "Remaining illness duration (days, final day)",
        "Days remaining",
        "Number of children",
    )

    plt.tight_layout()
    st.pyplot(fig)


def compute_child_illness_history_from_memory(history: dict):
    """
    Summarize illness over all timesteps for children under 5.
    - total_sick_days: days with ILLNESS_DURATION > 0
    - max_severity: maximum SYMPTOM_SEVERITY over time
    """
    if history is None:
        return None

    sev = history["symptom_severity"]      # (n_agents, n_time)
    dur = history["illness_duration"]      # (n_agents, n_time)
    is_child = history["is_child"]
    age = history["age"]

    under5 = is_child.astype(bool) & (age < 60.0)
    if not under5.any():
        return None

    sev_u5 = sev[under5, :]
    dur_u5 = dur[under5, :]

    # Sick day = any day with illness_duration > 0
    total_sick_days = (dur_u5 > 0).sum(axis=1).astype(int)

    # Max severity across the entire horizon
    max_severity = sev_u5.max(axis=1)

    return {
        "total_sick_days": total_sick_days,
        "max_severity": max_severity,
    }


def plot_child_illness_history(history_summary: dict):
    """
    Plot aggregates of illness history across all timesteps for children under 5.
    """
    if history_summary is None:
        st.info(
            "No child illness history found. Enable "
            "'Collect full child illness history in memory' before running."
        )
        return

    total_sick_days = history_summary["total_sick_days"]
    max_severity = history_summary["max_severity"]

    st.subheader("Child illness history (all timesteps, children under 5)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 1. Total sick days per child (discrete)
    _plot_discrete_hist(
        axes[0],
        total_sick_days,
        "Total sick days per child (u5)",
        "Days sick during simulation",
        "Number of children",
    )

    # 2. Max severity
    max_severity_nonzero = max_severity[max_severity > 0]
    if max_severity_nonzero.size > 0:
        axes[1].hist(max_severity_nonzero, bins=20, range=(0, 1), edgecolor="black")
        axes[1].set_title("Maximum symptom severity per child (u5)")
        axes[1].set_xlabel("Severity (0–1)")
        axes[1].set_ylabel("Number of children")
    else:
        axes[1].text(0.5, 0.5, "No symptomatic episodes recorded", ha="center", va="center")
        axes[1].set_axis_off()

    plt.tight_layout()
    st.pyplot(fig)


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
2. Optionally enable *Data collection* to record full child illness history in memory (slower).
3. Press **▶ Run simulation** to run a single full simulation.
4. Inspect epidemic curves and child-level outcome summaries.
"""
)

# Build config from UI
config, collect_history = build_config_from_ui()

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
        model, incidence, prevalence, history = run_simulation(config, collect_history)

    st.success("Simulation finished.")

    # Epidemic curves
    plot_epidemic_curves(incidence, prevalence)

    # Basic overall metrics
    st.subheader("Overall infection metrics")
    total_infections = model.get_total_infections()
    prop_infected = model.get_proportion_infected_at_least_once()

    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Total infections (all pathogens)", f"{total_infections}")
    mcol2.metric("Proportion of agents infected at least once", f"{prop_infected:.2%}")

    # Child metrics & histograms (final snapshot)
    metrics = child_metrics(model)

    st.subheader("Children under 5 – summary metrics")
    ccol1, ccol2 = st.columns(2)
    ccol1.metric("Total children", f"{metrics['n_children_total']}")
    ccol2.metric("Children under 5", f"{metrics['n_u5']}")

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

    plot_child_histograms(metrics)

    # Illness history across all timesteps (if collected)
    if collect_history:
        history_summary = compute_child_illness_history_from_memory(history)
        plot_child_illness_history(history_summary)

        if history_summary is not None:

            max_sev_hist = history["symptom_severity"].max()
            max_dur_hist = history["illness_duration"].max()
            st.markdown(
                f"**DEBUG:** max severity in history = {max_sev_hist:.4f}, "
                f"max illness_duration in history = {max_dur_hist}"
            )

            total_sick_days = history_summary["total_sick_days"]
            max_severity = history_summary["max_severity"]
            n_children_hist = len(total_sick_days)

            mean_sick_days = total_sick_days.mean() if n_children_hist > 0 else 0.0
            sick_days_per_year = mean_sick_days / sim_years if sim_years > 0 else 0.0
            mean_max_sev = max_severity.mean() if n_children_hist > 0 else 0.0

            st.markdown(
                f"- **Mean total sick days per child (u5)**: {mean_sick_days:.1f} "
                f"over {config.step_target} days (~{sick_days_per_year:.1f} days/child-year).\n"
                f"- **Mean maximum severity per child (u5)**: {mean_max_sev:.2f} (0–1 scale)."
            )
else:
    st.info("Adjust parameters in the sidebar and press **▶ Run simulation** to start.")
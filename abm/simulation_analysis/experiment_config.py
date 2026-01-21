# abm/simulation_analysis/experiment_config.py
import os
import json
import hashlib
from config import SVEIRConfig

# --- Define Top-Level Directory Settings ---
OUTPUT_DIR = "outputs"
POLICY_SETS_DIR = "policy_sets"
RESULTS_SUBDIR = "simulation_results"
SIM_RUNS_SUBDIR = "simulation_configs"

# --- Functions for Policy Set Versioning ---

def get_policy_set_id(base_config: SVEIRConfig) -> str:
    """
    Creates a unique, deterministic hash from all parameters that influence
    the policy generation process.
    """
    exp = base_config.experiment_params
    steer = base_config.steering_parameters
    params = {
        "cost_subsidy_factors": [round(f, 5) for f in exp.cost_subsidy_factors],
        "efficacy_multipliers": [round(f, 5) for f in exp.efficacy_multipliers],
        "infection_risk_levels": [round(f, 5) for f in exp.infection_risk_levels],
        "num_agent_personas": base_config.num_agent_personas,
        "persona_generation_seed": base_config.seed,
        "alpha_range": base_config.alpha_range, "gamma_range": base_config.gamma_range,
        "omega_range": base_config.omega_range, "eta_range": base_config.eta_range,
        "max_state_value": steer.max_state_value,
        "beta": steer.beta, "theta": steer.theta,
        "P_H_increase": steer.P_H_increase, "P_H_decrease": steer.P_H_decrease,
        "wealth_update_A": steer.wealth_update_A,
        "infection_health_shock": steer.infection_health_shock,
        "infection_reduction_factor_per_health_unit": steer.infection_reduction_factor_per_health_unit,
    }
    params_string = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_string.encode('utf-8')).hexdigest()[:12]

def get_policy_set_path(policy_set_id: str) -> str:
    """Gets the path to the directory for a specific, versioned policy set."""
    return os.path.join(POLICY_SETS_DIR, policy_set_id)

def get_policy_path(policy_set_id: str, efficacy: float, subsidy: float) -> str:
    """Generates the full, standardized path for a single policy library file."""
    set_path = get_policy_set_path(policy_set_id)
    filename = f"policy_eff_{efficacy:.2f}_cost_{subsidy:.2f}.npz"
    return os.path.join(set_path, filename)

# --- Functions for Experiment Results Organization ---

def get_experiment_base_path(experiment_name: str) -> str:
    """Gets the base path for a given unique experiment run."""
    return os.path.join(OUTPUT_DIR, experiment_name)

def get_sim_runs_path(experiment_name: str) -> str:
    """Generates the path for storing individual simulation outputs (e.g., YAML files)."""
    sim_runs_dir = os.path.join(get_experiment_base_path(experiment_name), SIM_RUNS_SUBDIR)
    os.makedirs(sim_runs_dir, exist_ok=True)
    return sim_runs_dir

def get_results_path(experiment_name: str, number_agents: int, repetitions: int) -> str:
    """Generates the path for the simulation summary results grid (.npy file)."""
    results_dir = os.path.join(get_experiment_base_path(experiment_name), RESULTS_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"summary_grid_prop_infected_agents_{number_agents}_reps_{repetitions}.npy"
    return os.path.join(results_dir, filename)

def get_full_results_path(experiment_name: str, config: SVEIRConfig) -> str:
    """Generates the path for the full, detailed simulation results (.pkl file)."""
    results_dir = os.path.join(get_experiment_base_path(experiment_name), RESULTS_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"full_results_agents_{config.number_agents}_reps_{config.experiment_params.repetitions}.pkl"
    return os.path.join(results_dir, filename)
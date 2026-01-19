# simulation_analysis/experiment_config.py

import numpy as np
import os
import json
import hashlib
from config import SVEIRConfig

# --- Define the Parameter Space for the Interventions ---
COST_SUBSIDY_FACTORS = np.linspace(1.0, 0.4, 5)
EFFICACY_MULTIPLIERS = np.linspace(1.0, 2.0, 5)
INFECTION_RISK_LEVELS = np.array([0.001, 0.002, 0.003, 0.004, 0.005])

# --- Define Top-Level Directory Settings ---
OUTPUT_DIR = "outputs"
POLICY_SETS_DIR = "policy_sets"

# Subdirectories that will be created inside each unique run folder.
RESULTS_SUBDIR = "simulation_results"
SIM_RUNS_SUBDIR = "simulation_configs"

# ==============================================================================
# --- HELPER FUNCTIONS FOR REPRODUCIBLE PATH MANAGEMENT ---
# ==============================================================================

# --- Functions for Policy Set Versioning ---

def get_policy_set_id(
    cost_factors: np.ndarray,
    efficacy_multipliers: np.ndarray,
    risk_levels: np.ndarray,
    base_config: SVEIRConfig
) -> str:
    """
    Creates a unique, deterministic hash from ALL parameters that influence
    the policy generation process.
    """
    params = {
        # Sweep Space Definition
        "cost_subsidy_factors": np.round(cost_factors, 5).tolist(),
        "efficacy_multipliers": np.round(efficacy_multipliers, 5).tolist(),
        "infection_risk_levels": np.round(risk_levels, 5).tolist(),
        # Agent Persona Generation Parameters
        "num_agent_personas": base_config.num_agent_personas,
        "persona_generation_seed": base_config.seed,
        "alpha_range": base_config.alpha_range,
        "gamma_range": base_config.gamma_range,
        "omega_range": base_config.omega_range,
        "eta_range": base_config.eta_range,
        # Core Value Iteration & CPT Parameters
        "max_state_value": base_config.steering_parameters.max_state_value,
        "beta": base_config.steering_parameters.beta,
        "theta": base_config.steering_parameters.theta,
        "omega_range_proxy": base_config.omega_range, # Using range as proxy
        "eta_range_proxy": base_config.eta_range, # Using range as proxy
        "P_H_increase": base_config.steering_parameters.P_H_increase,
        "P_H_decrease": base_config.steering_parameters.P_H_decrease,
        # Health & Wealth Dynamics Parameters
        "wealth_update_A": base_config.steering_parameters.wealth_update_A,
        "infection_health_shock": base_config.steering_parameters.infection_health_shock,
        "infection_reduction_factor_per_health_unit": base_config.steering_parameters.infection_reduction_factor_per_health_unit,
    }
    params_string = json.dumps(params, sort_keys=True, indent=None)
    return hashlib.sha256(params_string.encode('utf-8')).hexdigest()[:12]


def get_policy_set_path(policy_set_id: str) -> str:
    """
    Gets the path to the directory for a specific, versioned policy set.
    Example: 'policy_sets/a1b2c3d4e5f6'
    """
    return os.path.join(POLICY_SETS_DIR, policy_set_id)


def get_policy_path(policy_set_id: str, efficacy: float, subsidy: float) -> str:
    """
    Generates the full, standardized path for a single policy library file
    within a versioned policy set.
    """
    set_path = get_policy_set_path(policy_set_id)
    filename = f"policy_eff_{efficacy:.2f}_cost_{subsidy:.2f}.npz"
    return os.path.join(set_path, filename)


# --- Functions for Experiment Results Organization ---

def get_experiment_base_path(experiment_name: str) -> str:
    """
    Gets the base path for a given unique experiment run.
    Example: 'outputs/run_20250814_140000_policies_a1b2c3d4e5f6'
    """
    return os.path.join(OUTPUT_DIR, experiment_name)


def get_sim_runs_path(experiment_name: str) -> str:
    """
    Generates the path for storing individual simulation outputs (like YAML files)
    for a specific experiment run.
    """
    base_path = get_experiment_base_path(experiment_name)
    sim_runs_dir = os.path.join(base_path, SIM_RUNS_SUBDIR)
    # Ensure the directory exists before returning the path
    os.makedirs(sim_runs_dir, exist_ok=True)
    return sim_runs_dir


def get_results_path(experiment_name: str, number_agents: int, repetitions: int) -> str:
    """
    Generates the standardized path for the simulation summary results grid (.npy file).
    """
    base_path = get_experiment_base_path(experiment_name)
    results_dir = os.path.join(base_path, RESULTS_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"summary_grid_prop_infected_agents_{number_agents}reps{repetitions}.npy"
    return os.path.join(results_dir, filename)


def get_full_results_path(experiment_name: str, number_agents: int, repetitions: int) -> str:
    """
    Generates the standardized path for the full, detailed simulation results (.pkl file).
    """
    base_path = get_experiment_base_path(experiment_name)
    results_dir = os.path.join(base_path, RESULTS_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"full_results_agents_{number_agents}_reps_{repetitions}.pkl"
    return os.path.join(results_dir, filename)
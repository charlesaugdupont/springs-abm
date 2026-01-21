# abm/simulation_analysis/experiment_config.py
import os
from config import SVEIRConfig

# --- Define Top-Level Directory Settings ---
OUTPUT_DIR = "outputs"
RESULTS_SUBDIR = "simulation_results"
SIM_RUNS_SUBDIR = "simulation_configs"

# --- Functions for Experiment Results Organization ---

def get_experiment_base_path(experiment_name: str) -> str:
    """Gets the base path for a given unique experiment run."""
    return os.path.join(OUTPUT_DIR, experiment_name)

def get_sim_runs_path(experiment_name: str) -> str:
    """Generates the path for storing individual simulation outputs (e.g., YAML files)."""
    sim_runs_dir = os.path.join(get_experiment_base_path(experiment_name), SIM_RUNS_SUBDIR)
    os.makedirs(sim_runs_dir, exist_ok=True)
    return sim_runs_dir

def get_full_results_path(experiment_name: str, config: SVEIRConfig) -> str:
    """Generates the path for the full, detailed simulation results (.pkl file)."""
    results_dir = os.path.join(get_experiment_base_path(experiment_name), RESULTS_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"full_results_agents_{config.number_agents}.pkl"
    return os.path.join(results_dir, filename)
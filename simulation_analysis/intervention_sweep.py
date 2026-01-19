# simulation_analysis/intervention_sweep.py

import numpy as np
import os
import multiprocessing
from tqdm import tqdm
import time
import pickle
import traceback

from .experiment_config import (
    COST_SUBSIDY_FACTORS,
    EFFICACY_MULTIPLIERS,
    get_policy_path,
    get_results_path,
    get_full_results_path,
    get_sim_runs_path
)
from dgl_ptm.config import SVEIRConfig
from dgl_ptm.model.initialize_model import SVEIRModel

def run_single_simulation(params: dict) -> dict:
    """
    Worker function: runs a single simulation instance and returns the result.
    """
    run_name = params['run_name']
    sim_runs_path = params['sim_runs_path']
    config_dict = params["config_dict"]

    try:
        model = SVEIRModel(model_identifier=run_name, root_path=sim_runs_path)
        config_for_model = config_dict.copy()
        config_for_model = config_dict.copy()
        config_for_model.pop('model_identifier', None)
        model.set_model_parameters(**config_for_model)
        model.initialize_model(verbose=False)
        model.run()
        
        time_series_data = model.get_time_series_data()
        final_states = model.get_final_agent_states()
        incidence_curve = time_series_data['incidence']
        prevalence_curve = time_series_data['prevalence']
        infection_counts = model.get_final_infection_counts()
        personas = model.get_agent_personas()
        initial_wealth = model.get_initial_wealth()
        initial_health = model.get_initial_health()

        return {
            'efficacy': config_dict['steering_parameters']['efficacy_multiplier'],
            'subsidy': config_dict['steering_parameters']['cost_subsidy_factor'],
            'total_infections': model.get_total_infections(),
            'peak_incidence': max(incidence_curve) if incidence_curve else 0,
            'incidence_curve': incidence_curve,
            'prevalence_curve': prevalence_curve,
            'proportion_infected': model.get_proportion_infected_at_least_once(),
            'final_health': final_states['health'],
            'final_wealth': final_states['wealth'],
            'final_num_infections': infection_counts,
            'personas': personas,
            'initial_wealth': initial_wealth,
            'initial_health': initial_health
        }

    except Exception:
        print(f"\n--- ERROR IN WORKER: {run_name} ---")
        traceback.print_exc()
        print(f"--- END ERROR ---")
        return {
            'efficacy': config_dict.get('steering_parameters', {}).get('efficacy_multiplier', -1),
            'subsidy': config_dict.get('steering_parameters', {}).get('cost_subsidy_factor', -1),
            'total_infections': -1,
            'peak_incidence': -1,
            'incidence_curve': [],
            'prevalence_curve': [],
            'proportion_infected': -1.0,
            'final_health': [],
            'final_wealth': [],
            'final_num_infections': np.array([]),
            'personas': np.array([]),
            'initial_wealth': np.array([]),
            'initial_health': np.array([])
        }


def worker_unpacker(args):
    """Helper to unpack arguments for the pool."""
    return run_single_simulation(args)


def run_simulation_sweep(number_agents: int, repetitions: int, num_cores: int, steps: int, experiment_name: str, grid_id: str, policy_set_id: str):
    """
    Runs the full simulation sweep in parallel and saves both detailed and summary results.
    """
    print(f"Starting parallel simulation sweep for experiment: {experiment_name}")
    sim_runs_path = get_sim_runs_path(experiment_name)
    print(f"Individual run outputs will be saved in: {sim_runs_path}")

    tasks = []
    base_seed = SVEIRConfig().seed

    for i, efficacy in enumerate(EFFICACY_MULTIPLIERS):
        for j, subsidy in enumerate(COST_SUBSIDY_FACTORS):
            policy_path = get_policy_path(policy_set_id, efficacy, subsidy)
            if not os.path.exists(policy_path):
                print(f"WARNING: Policy file not found at '{policy_path}'. Skipping combination.")
                continue

            for r in range(repetitions):
                unique_seed = base_seed + (i * len(COST_SUBSIDY_FACTORS) * repetitions) + (j * repetitions) + r
                run_name = f"sim_eff_{efficacy:.2f}_cost_{subsidy:.2f}_rep_{r+1}"

                # Create the complete configuration dictionary for this specific run
                config_for_run = {
                    "model_identifier": run_name,
                    "number_agents": number_agents,
                    "step_target": steps,
                    "seed": unique_seed,
                    "policy_library_path": policy_path,
                    "spatial_creation_args": {"grid_id": grid_id},
                    "steering_parameters": {
                        "efficacy_multiplier": efficacy,
                        "cost_subsidy_factor": subsidy
                    }
                }

                tasks.append({
                    'run_name': run_name,
                    'config_dict': config_for_run,
                    'sim_runs_path': sim_runs_path
                })
    
    if not tasks:
        print("No valid policy files found. Cannot run simulations.")
        return

    print(f"Total simulation runs to perform: {len(tasks)}\n")
    time.sleep(1)

    raw_results = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        with tqdm(total=len(tasks), desc="Running Simulations") as pbar:
            for result in pool.imap_unordered(worker_unpacker, tasks):
                if result:
                    raw_results.append(result)
                pbar.update(1)
    
    full_results_path = get_full_results_path(experiment_name, number_agents, repetitions)
    with open(full_results_path, 'wb') as f:
        pickle.dump(raw_results, f)
    print(f"\nResults saved to: {full_results_path}")

    results_agg = {}
    for res in raw_results:
        key = (res['efficacy'], res['subsidy'])
        if key not in results_agg:
            results_agg[key] = []
        results_agg[key].append(res['proportion_infected'])

    summary_grid = np.zeros((len(EFFICACY_MULTIPLIERS), len(COST_SUBSIDY_FACTORS)))
    for i, efficacy in enumerate(EFFICACY_MULTIPLIERS):
        for j, subsidy in enumerate(COST_SUBSIDY_FACTORS):
            key = (efficacy, subsidy)
            proportions_list = results_agg.get(key, [-1.0])
            valid_proportions = [p for p in proportions_list if p >= 0]
            summary_grid[i, j] = np.mean(valid_proportions) if valid_proportions else -1
            
    summary_grid_path = get_results_path(experiment_name, number_agents, repetitions)
    np.save(summary_grid_path, summary_grid)
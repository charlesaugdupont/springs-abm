# abm/simulation_analysis/intervention_sweep.py
import numpy as np
import os
import multiprocessing
from tqdm import tqdm
import time
import pickle
import traceback
from typing import Dict

from .experiment_config import get_policy_path, get_results_path, get_full_results_path, get_sim_runs_path
from config import SVEIRConfig
from abm.model.initialize_model import SVEIRModel

def run_single_simulation(params: Dict) -> Dict:
    """Worker function: runs one simulation instance and returns key results."""
    run_name = params['run_name']
    sim_runs_path = params['sim_runs_path']
    config_dict = params["config_dict"]

    try:
        model = SVEIRModel(model_identifier=run_name, root_path=sim_runs_path)
        # Create a full config object for the model to use
        model_config = SVEIRConfig.from_dict(config_dict)
        model.set_model_parameters(**model_config.model_dump())
        model.initialize_model(verbose=False)
        model.run()

        time_series = model.get_time_series_data()
        final_states = model.get_final_agent_states()
        return {
            'efficacy': model.config.steering_parameters.efficacy_multiplier,
            'subsidy': model.config.steering_parameters.cost_subsidy_factor,
            'proportion_infected': model.get_proportion_infected_at_least_once(),
            'prevalence_curve': time_series['prevalence'],
            'final_health': final_states['health'],
            'final_wealth': final_states['wealth'],
        }
    except Exception:
        print(f"\n--- ERROR IN WORKER: {run_name} ---")
        traceback.print_exc()
        return {'efficacy': -1, 'subsidy': -1, 'proportion_infected': -1.0}

def worker_unpacker(args):
    """Helper to unpack arguments for the multiprocessing pool."""
    return run_single_simulation(args)

def run_simulation_sweep(config: SVEIRConfig, experiment_name: str, policy_set_id: str):
    """Runs the full simulation sweep in parallel and saves results."""
    print(f"Starting parallel simulation sweep for experiment: {experiment_name}")
    sim_runs_path = get_sim_runs_path(experiment_name)
    print(f"Individual run configs will be saved in: {sim_runs_path}")

    tasks = []
    exp_params = config.experiment_params
    base_seed = config.seed

    for i, efficacy in enumerate(exp_params.efficacy_multipliers):
        for j, subsidy in enumerate(exp_params.cost_subsidy_factors):
            policy_path = get_policy_path(policy_set_id, efficacy, subsidy)
            if not os.path.exists(policy_path):
                print(f"WARNING: Policy file not found at '{policy_path}'. Skipping combination.")
                continue

            for r in range(exp_params.repetitions):
                unique_seed = base_seed + (i * len(exp_params.cost_subsidy_factors) * exp_params.repetitions) + (j * exp_params.repetitions) + r
                run_name = f"sim_eff_{efficacy:.2f}_cost_{subsidy:.2f}_rep_{r+1}"

                # Create a temporary config object for this specific run
                run_config = config.model_copy(deep=True)
                run_config.seed = unique_seed
                run_config.policy_library_path = policy_path
                run_config.steering_parameters.efficacy_multiplier = efficacy
                run_config.steering_parameters.cost_subsidy_factor = subsidy

                tasks.append({
                    'run_name': run_name,
                    'config_dict': run_config.model_dump(by_alias=True),
                    'sim_runs_path': sim_runs_path
                })

    if not tasks:
        print("No valid simulation tasks could be created. Aborting.")
        return

    print(f"Total simulation runs to perform: {len(tasks)}\n")
    time.sleep(1)

    raw_results = []
    with multiprocessing.Pool(processes=exp_params.num_cores) as pool:
        with tqdm(total=len(tasks), desc="Running Simulations") as pbar:
            for result in pool.imap_unordered(worker_unpacker, tasks):
                if result:
                    raw_results.append(result)
                pbar.update(1)

    # Save detailed results
    full_results_path = get_full_results_path(experiment_name, config)
    with open(full_results_path, 'wb') as f:
        pickle.dump(raw_results, f)
    print(f"\nFull results saved to: {full_results_path}")

    # Aggregate and save summary grid
    results_agg = {}
    for res in raw_results:
        key = (res['efficacy'], res['subsidy'])
        if key not in results_agg: results_agg[key] = []
        results_agg[key].append(res['proportion_infected'])

    summary_grid = np.zeros((len(exp_params.efficacy_multipliers), len(exp_params.cost_subsidy_factors)))
    for i, efficacy in enumerate(exp_params.efficacy_multipliers):
        for j, subsidy in enumerate(exp_params.cost_subsidy_factors):
            key = (efficacy, subsidy)
            proportions = [p for p in results_agg.get(key, [-1.0]) if p >= 0]
            summary_grid[i, j] = np.mean(proportions) if proportions else -1.0

    summary_grid_path = get_results_path(experiment_name, config.number_agents, exp_params.repetitions)
    np.save(summary_grid_path, summary_grid)
    print(f"Summary grid saved to: {summary_grid_path}")
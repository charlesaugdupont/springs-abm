# abm/simulation_analysis/intervention_sweep.py
import numpy as np
import os
import multiprocessing
from tqdm import tqdm
import time
import pickle
import traceback
from typing import Dict

from .experiment_config import get_summary_results_path, get_full_results_path, get_sim_runs_path
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
            'run_name': run_name,
            'proportion_infected': model.get_proportion_infected_at_least_once(),
            'prevalence_curve': time_series['prevalence'],
            'final_health': final_states['health'],
            'final_wealth': final_states['wealth'],
        }
    except Exception:
        print(f"\n--- ERROR IN WORKER: {run_name} ---")
        traceback.print_exc()
        return {'run_name': run_name, 'proportion_infected': -1.0}

def worker_unpacker(args):
    """Helper to unpack arguments for the multiprocessing pool."""
    return run_single_simulation(args)

def run_simulation_sweep(config: SVEIRConfig, experiment_name: str):
    """Runs the full simulation sweep in parallel and saves results."""
    print(f"Starting parallel simulation sweep for experiment: {experiment_name}")
    sim_runs_path = get_sim_runs_path(experiment_name)
    print(f"Individual run configs will be saved in: {sim_runs_path}")

    tasks = []
    exp_params = config.experiment_params
    base_seed = config.seed

    for r in range(exp_params.repetitions):
        unique_seed = base_seed + r
        run_name = f"sim_rep_{r+1}"

        # Create a temporary config object for this specific run
        run_config = config.model_copy(deep=True)
        run_config.seed = unique_seed

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

    # Save detailed results (this remains the same)
    full_results_path = get_full_results_path(experiment_name, config)
    with open(full_results_path, 'wb') as f:
        pickle.dump(raw_results, f)
    print(f"\nFull results saved to: {full_results_path}")

    summary_outcomes = {
        res['run_name']: res['proportion_infected']
        for res in raw_results if 'run_name' in res
    }
    summary_path = get_summary_results_path(experiment_name, config)
    with open(summary_path, 'wb') as f:
        pickle.dump(summary_outcomes, f)
    print(f"Summary of outcomes saved to: {summary_path}")
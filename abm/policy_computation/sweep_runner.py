# abm/policy_computation/sweep_runner.py
import os
import multiprocessing
from tqdm import tqdm
import numpy as np

from abm.simulation_analysis.experiment_config import get_policy_path, get_policy_set_path
from config import SVEIRConfig
from .generator import create_and_save_policy_library

def process_one_policy_set(args):
    """Worker function for multiprocessing pool."""
    config_for_job, pbar_position, lock = args
    try:
        create_and_save_policy_library(
            config=config_for_job,
            infection_risk_levels=np.array(config_for_job.experiment_params.infection_risk_levels),
            pbar_position=pbar_position,
            lock=lock
        )
        return "Success"
    except Exception as e:
        print(f"\n--- ERROR in worker for policy: {config_for_job.policy_library_path} ---\n{e}")
        return "Failed"

def compute_all_policies_for_sweep(base_config: SVEIRConfig, policy_set_id: str):
    """Orchestrates the parallel pre-computation of all policy libraries for an experiment."""
    print("Starting parallel batch pre-computation of all policy libraries...")
    exp_params = base_config.experiment_params
    num_cores = exp_params.num_cores
    print(f"Using up to {num_cores} CPU cores.")

    policy_set_path = get_policy_set_path(policy_set_id)
    print("-" * 60)
    print(f"Generating policies for Policy Set ID: {policy_set_id}")
    print(f"Output directory will be: {policy_set_path}")
    print("-" * 60)
    os.makedirs(policy_set_path, exist_ok=True)

    tasks = []
    for efficacy in exp_params.efficacy_multipliers:
        for subsidy in exp_params.cost_subsidy_factors:
            policy_path = get_policy_path(policy_set_id, efficacy, subsidy)
            if not os.path.exists(policy_path):
                job_config = base_config.model_copy(deep=True)
                job_config.policy_library_path = policy_path
                job_config.steering_parameters.efficacy_multiplier = efficacy
                job_config.steering_parameters.cost_subsidy_factor = subsidy
                tasks.append(job_config)

    if not tasks:
        print(f"All policy libraries for set '{policy_set_id}' already exist. Nothing to do.")
        return

    print(f"Found {len(tasks)} new policy sets to generate for set '{policy_set_id}'.")
    print("Efficacy Multipliers:", exp_params.efficacy_multipliers)
    print("Cost Subsidy Factors:", exp_params.cost_subsidy_factors)
    print(f"Agent Personas: {base_config.num_agent_personas}\n")

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    main_pbar = tqdm(total=len(tasks), desc="Overall Progress", position=0)

    with multiprocessing.Pool(processes=num_cores) as pool:
        # Assign a consistent position in the terminal for each worker's progress bar
        tasks_with_args = [(task_config, (i % num_cores) + 1, lock) for i, task_config in enumerate(tasks)]
        for result in pool.imap_unordered(process_one_policy_set, tasks_with_args):
            with lock:
                main_pbar.update(1)

    main_pbar.close()
    print("\nAll policy libraries have been computed.")
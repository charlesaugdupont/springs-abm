# policy_computation/sweep_runner.py

import os
import multiprocessing
from tqdm import tqdm

from abm.simulation_analysis.experiment_config import (
    COST_SUBSIDY_FACTORS,
    EFFICACY_MULTIPLIERS,
    INFECTION_RISK_LEVELS,
    get_policy_path,
    get_policy_set_id,
    get_policy_set_path
)
from config import SVEIRConfig
from .generator import create_and_save_policy_library

# --- Worker Function (Updated with correct error logging) ---
def process_one_policy_set(config_for_job: SVEIRConfig, pbar_position: int, lock: multiprocessing.Lock):
    """
    The worker function, with corrected error logging.
    """
    try:
        create_and_save_policy_library(
            config=config_for_job,
            infection_risk_levels=INFECTION_RISK_LEVELS,
            pbar_position=pbar_position,
            lock=lock
        )
        return "Success"
    except Exception as e:
        print(f"\n--- ERROR in worker for policy: {config_for_job.policy_library_path} ---")
        return "Failed"

# --- Unpacker Helper (Unchanged) ---
def worker_unpacker(args):
    """Helper to unpack arguments for pool.imap_unordered."""
    config_for_job, pbar_position, lock = args
    return process_one_policy_set(config_for_job, pbar_position, lock)

# --- Main Orchestrator (Unchanged) ---
def compute_all_policies_for_sweep():
    print("Starting PARALLEL batch pre-computation of all policy libraries...")
    
    NUM_CORES = 6
    print(f"Using up to {NUM_CORES} CPU cores.")

    base_config = SVEIRConfig()

    policy_set_id = get_policy_set_id(
        cost_factors=COST_SUBSIDY_FACTORS,
        efficacy_multipliers=EFFICACY_MULTIPLIERS,
        risk_levels=INFECTION_RISK_LEVELS,
        base_config=base_config
    )
    policy_set_path = get_policy_set_path(policy_set_id)

    print("-" * 60)
    print(f"Generating policies for Policy Set ID: {policy_set_id}")
    print(f"Output directory will be: {policy_set_path}")
    print("-" * 60)

    os.makedirs(policy_set_path, exist_ok=True)
    
    tasks = []
    for efficacy in EFFICACY_MULTIPLIERS:
        for subsidy in COST_SUBSIDY_FACTORS:
            policy_path = get_policy_path(policy_set_id, efficacy, subsidy)
            if not os.path.exists(policy_path):
                job_config = base_config.model_copy(deep=True)
                job_config.policy_library_path = policy_path 
                job_config.steering_parameters.efficacy_multiplier = efficacy
                job_config.steering_parameters.cost_subsidy_factor = subsidy
                tasks.append(job_config)

    if not tasks:
        print(f"All policy libraries for set '{policy_set_id}' have already been computed. Nothing to do.")
        return

    print(f"Found {len(tasks)} policy sets to generate for set '{policy_set_id}'.")
    print("Efficacy Multipliers:", EFFICACY_MULTIPLIERS)
    print("Cost Subsidy Factors:", COST_SUBSIDY_FACTORS)
    print("Infection Risk Levels:", INFECTION_RISK_LEVELS)
    print(f"Number of Agent Personas: {base_config.num_agent_personas}\n")

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    main_pbar = tqdm(total=len(tasks), desc="Overall Progress", position=0)

    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        tasks_with_args = [(task_config, (i % NUM_CORES) + 1, lock) for i, task_config in enumerate(tasks)]
        
        for result in pool.imap_unordered(worker_unpacker, tasks_with_args):
            with lock:
                if result == "Success":
                    main_pbar.set_postfix_str("Last task finished successfully.")
                else:
                    main_pbar.set_postfix_str("Last task failed. Check logs.")
                main_pbar.update(1)

    main_pbar.close()
    print("All policy libraries have been computed.")
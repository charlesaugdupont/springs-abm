# policy_computation/generator.py

import numpy as np
from scipy.stats import qmc
import torch
from tqdm import tqdm
import multiprocessing

from dgl_ptm.config import SVEIRConfig
from .engine import value_iteration

def create_and_save_policy_library(
    config: SVEIRConfig,
    infection_risk_levels: np.ndarray,
    pbar_position: int,
    lock: multiprocessing.Lock
):
    """
    Generates and saves a policy library, with a granular progress bar
    tracking each individual policy computation (persona * risk_level).
    """
    # 1. Generate agent personas (no change here)
    sampler = qmc.LatinHypercube(d=4, seed=config.seed)
    samples = sampler.random(n=config.num_agent_personas)
    param_ranges = [config.alpha_range, config.gamma_range, config.omega_range, config.eta_range]
    scaled_samples = qmc.scale(samples, [r[0] for r in param_ranges], [r[1] for r in param_ranges])
    agent_personas = torch.from_numpy(scaled_samples).float()

    # 2. Pre-compute policies
    policy_library = {}
    
    desc = f"W{pbar_position-1} (E:{config.steering_parameters.efficacy_multiplier:.1f}, S:{config.steering_parameters.cost_subsidy_factor:.1f})"
    
    # --- KEY CHANGE 1: Set the total to the correct number of computations ---
    total_policies_for_worker = config.num_agent_personas * len(infection_risk_levels)

    with lock:
        persona_pbar = tqdm(
            total=total_policies_for_worker, # Set the total to 80 (16 * 5)
            desc=desc,
            position=pbar_position,
            leave=False,
            unit=" policy" # Add a unit to the progress bar for clarity
        )

    for persona_id in range(config.num_agent_personas):
        persona_policies = []
        alpha, gamma, omega, eta = agent_personas[persona_id]

        for risk_level in infection_risk_levels:
            # Create a fresh copy of the steering parameters for each risk level
            params_for_vi = config.steering_parameters.model_dump()
            params_for_vi['global_infection_prob'] = risk_level
            
            policy = value_iteration(
                max_state_value=100,
                alpha=alpha.item(),
                gamma=gamma.item(),
                theta=config.steering_parameters.theta,
                omega=omega.item(),
                eta=eta.item(),
                beta=config.steering_parameters.beta,
                params=params_for_vi
            )
            persona_policies.append(policy)
            
            # --- KEY CHANGE 2: Update the bar inside the inner loop ---
            # This increments the bar after each of the 80 individual computations.
            with lock:
                persona_pbar.update(1)
        
        policy_library[persona_id] = np.stack(persona_policies)

    # We still close the bar at the end, although it should already be at 100%
    with lock:
        persona_pbar.close()

    # 3. Save the results (no change here)
    np.savez_compressed(
        config.policy_library_path,
        agent_personas=agent_personas.numpy(),
        infection_risk_levels=infection_risk_levels,
        **{f"policies_{pid}": policy_library[pid] for pid in policy_library}
    )
# abm/policy_computation/generator.py
import numpy as np
from scipy.stats import qmc
import torch
from tqdm import tqdm
import multiprocessing

from config import SVEIRConfig
from .engine import value_iteration

def create_and_save_policy_library(
    config: SVEIRConfig,
    infection_risk_levels: np.ndarray,
    pbar_position: int,
    lock: multiprocessing.Lock
):
    """Generates and saves a complete policy library for a given configuration."""
    # 1. Generate agent behavioral personas
    sampler = qmc.LatinHypercube(d=4, seed=config.seed)
    samples = sampler.random(n=config.num_agent_personas)
    ranges = [config.alpha_range, config.gamma_range, config.omega_range, config.eta_range]
    scaled_samples = qmc.scale(samples, [r[0] for r in ranges], [r[1] for r in ranges])
    agent_personas = torch.from_numpy(scaled_samples).float()

    # 2. Pre-compute policies for each persona and risk level
    policy_library = {}
    total_iterations = config.num_agent_personas * len(infection_risk_levels)
    desc = f"Worker {pbar_position-1} (E:{config.steering_parameters.efficacy_multiplier:.2f}, S:{config.steering_parameters.cost_subsidy_factor:.2f})"

    with lock:
        pbar = tqdm(total=total_iterations, desc=desc, position=pbar_position, leave=False, unit="policy")

    for persona_id in range(config.num_agent_personas):
        persona_policies = []
        alpha, gamma, omega, eta = agent_personas[persona_id]

        for risk_level in infection_risk_levels:
            params_for_vi = config.steering_parameters.model_dump()
            params_for_vi['global_infection_prob'] = risk_level

            policy = value_iteration(
                max_state_value=config.steering_parameters.max_state_value,
                alpha=alpha.item(), gamma=gamma.item(),
                theta=config.steering_parameters.theta,
                omega=omega.item(), eta=eta.item(),
                beta=config.steering_parameters.beta,
                params=params_for_vi
            )
            persona_policies.append(policy)
            with lock:
                pbar.update(1)

        policy_library[persona_id] = np.stack(persona_policies)

    with lock:
        pbar.close()

    # 3. Save the generated library
    np.savez_compressed(
        config.policy_library_path,
        agent_personas=agent_personas.numpy(),
        infection_risk_levels=infection_risk_levels,
        **{f"policies_{pid}": policy_library[pid] for pid in policy_library}
    )
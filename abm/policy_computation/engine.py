# abm/policy_computation/engine.py
import torch
import numpy as np
from typing import Dict

from abm.agent.health_cpt_utils import (
    probability_weighting,
    cpt_value_function,
    compute_new_wealth,
    utility,
    compute_health_delta,
    compute_health_decline,
    compute_health_cost
)

CONVERGENCE_TOLERANCE = 1e-3

def _calculate_expected_value(
    wealth: int, health: int, value_function: np.ndarray, params: Dict,
    action_health_change: int, action_health_prob: float, action_cost: float
) -> float:
    """Generic helper to calculate the expected value of an action."""
    if wealth <= action_cost:
        return -np.inf

    w_after_cost = wealth - action_cost
    ref_utility = utility(wealth, health, params['alpha'])

    # Outcome 1: Health changes as expected by the action
    h1 = min(max(1, health + action_health_change), params['max_state_value'])
    w1 = min(int(compute_new_wealth(w_after_cost, params['wealth_update_A'], utility(w_after_cost, h1, params['alpha']))), params['max_state_value'])
    cpt_delta1 = cpt_value_function(utility(w1, h1, params['alpha']) - ref_utility, params)
    future_val1 = value_function[w1 - 1, h1 - 1]

    # Outcome 2: Health remains steady (action fails)
    h2 = health
    w2 = min(int(compute_new_wealth(w_after_cost, params['wealth_update_A'], utility(w_after_cost, h2, params['alpha']))), params['max_state_value'])
    cpt_delta2 = cpt_value_function(utility(w2, h2, params['alpha']) - ref_utility, params)
    future_val2 = value_function[h2 - 1, w2 - 1]

    cpt_prob1 = probability_weighting(action_health_prob, params['gamma'])
    cpt_prob2 = probability_weighting(1 - action_health_prob, params['gamma'])

    immediate_cpt_value = cpt_prob1 * cpt_delta1 + cpt_prob2 * cpt_delta2
    expected_future_value = cpt_prob1 * future_val1 + cpt_prob2 * future_val2

    # Factor in the external risk of infection
    health_susceptibility = np.exp(-params["infection_reduction_factor_per_health_unit"] * (health - 1.0))
    prob_infection = np.clip(params['global_infection_prob'] * health_susceptibility, 0, 1)

    h_infected = max(1, health - params["infection_health_shock"])
    val_if_infected = utility(wealth, h_infected, params['alpha']) + params['beta'] * value_function[wealth - 1, h_infected - 1]

    final_value = (1 - prob_infection) * (immediate_cpt_value + params['beta'] * expected_future_value) + \
                  (prob_infection) * cpt_value_function(val_if_infected - ref_utility, params)
    return final_value

def value_iteration(max_state_value, alpha, gamma, theta, omega, eta, beta, params) -> np.ndarray:
    """Computes the optimal health investment policy for an agent using value iteration."""
    value_function = np.zeros((max_state_value, max_state_value))
    policy = np.zeros((max_state_value, max_state_value), dtype=int)

    local_params = params.copy()
    local_params.update({'alpha': alpha, 'gamma': gamma, 'beta': beta, 'max_state_value': max_state_value,
                         'theta': theta, 'omega': omega, 'eta': eta})

    # Pre-calculate health/cost dynamics for all states
    health_states = np.arange(1, max_state_value + 1)
    health_gain = compute_health_delta(torch.from_numpy(health_states), local_params).numpy().astype(int)
    health_loss = compute_health_decline(torch.from_numpy(health_states)).numpy().astype(int)
    invest_cost = compute_health_cost(torch.from_numpy(health_states), local_params).numpy()

    norm = np.inf
    while norm > CONVERGENCE_TOLERANCE:
        old_value_function = value_function.copy()
        for w_idx in range(max_state_value):
            for h_idx in range(max_state_value):
                w, h = w_idx + 1, h_idx + 1
                invest_val = _calculate_expected_value(
                    w, h, old_value_function, local_params,
                    action_health_change=health_gain[h_idx],
                    action_health_prob=params['P_H_increase'],
                    action_cost=invest_cost[h_idx]
                )
                save_val = _calculate_expected_value(
                    w, h, old_value_function, local_params,
                    action_health_change=-health_loss[h_idx],
                    action_health_prob=params['P_H_decrease'],
                    action_cost=0
                )
                if invest_val > save_val:
                    value_function[w_idx, h_idx] = invest_val
                    policy[w_idx, h_idx] = 1 # Invest
                else:
                    value_function[w_idx, h_idx] = save_val
                    policy[w_idx, h_idx] = 0 # Save
        norm = np.linalg.norm(value_function - old_value_function)
    return policy
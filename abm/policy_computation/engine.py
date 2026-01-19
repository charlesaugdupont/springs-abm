# policy_computation/engine.py

import numpy as np

from abm.agent.health_cpt_utils import (
    probability_weighting,
    cpt_value_function,
    compute_new_wealth,
    utility,
    compute_health_delta,
    compute_health_decline,
    compute_health_cost
)

# --- Constants for Value Iteration ---
CONVERGENCE_TOLERANCE = 1e-3

def _calculate_expected_value(wealth, health, value_function, params, action_health_change, action_health_prob, action_cost):
    """
    Generic helper to calculate the expected value of an action, including external risk.
    """
    if wealth <= action_cost:
        return -np.inf

    wealth_after_cost = wealth - action_cost
    reference_utility = utility(wealth, health, params['alpha'])

    # Outcome 1: Health changes as expected
    health_1 = min(max(1, health + action_health_change), params['max_state_value'])
    wealth_1 = min(int(compute_new_wealth(wealth_after_cost, params['wealth_update_A'], utility(wealth_after_cost, health_1, params['alpha']))), params['max_state_value'])
    cpt_delta_1 = cpt_value_function(utility(wealth_1, health_1, params['alpha']) - reference_utility, params)
    future_val_1 = value_function[wealth_1 - 1, health_1 - 1]

    # Outcome 2: Health remains steady
    health_2 = health
    wealth_2 = min(int(compute_new_wealth(wealth_after_cost, params['wealth_update_A'], utility(wealth_after_cost, health_2, params['alpha']))), params['max_state_value'])
    cpt_delta_2 = cpt_value_function(utility(wealth_2, health_2, params['alpha']) - reference_utility, params)
    future_val_2 = value_function[wealth_2 - 1, health_2 - 1]
    
    cpt_prob_1 = probability_weighting(action_health_prob, params['gamma'])
    cpt_prob_2 = probability_weighting(1 - action_health_prob, params['gamma'])
    
    immediate_cpt_value = cpt_prob_1 * cpt_delta_1 + cpt_prob_2 * cpt_delta_2
    expected_future_value = cpt_prob_1 * future_val_1 + cpt_prob_2 * future_val_2
    
    health_susceptibility = np.exp(-params["infection_reduction_factor_per_health_unit"] * (health - 1.0))
    prob_infection = np.clip(params['global_infection_prob'] * health_susceptibility, 0, 1)

    # calculate the new health state if infected
    health_if_infected = max(1, health - params["infection_health_shock"])    

    value_if_infected = utility(wealth, health_if_infected, params['alpha']) + params['beta'] * value_function[wealth - 1, health_if_infected - 1]
    
    final_value = (1 - prob_infection) * (immediate_cpt_value + params['beta'] * expected_future_value) + \
                  (prob_infection) * cpt_value_function(value_if_infected - reference_utility, params)

    return final_value


def value_iteration(max_state_value, alpha, gamma, theta, omega, eta, beta, params):
    """
    Computes the optimal health investment policy for an agent using value iteration.
    """
    value_function = np.zeros((max_state_value, max_state_value))
    policy = np.zeros((max_state_value, max_state_value), dtype=int)
    
    local_params = params.copy()
    local_params.update({'alpha': alpha, 'gamma': gamma, 'beta': beta, 'max_state_value': max_state_value,
                         'theta': theta, 'omega': omega, 'eta': eta})

    P_H_increase = params['P_H_increase']
    P_H_decrease = params['P_H_decrease']

    norm = np.inf
    while norm > CONVERGENCE_TOLERANCE:
        old_value_function = value_function.copy()
        
        health_delta_array = compute_health_delta(np.arange(1, max_state_value + 1), local_params).numpy().astype(int)
        health_decline_array = compute_health_decline(np.arange(1, max_state_value + 1)).numpy().astype(int)
        invest_cost_array = compute_health_cost(np.arange(1, max_state_value + 1), local_params).numpy()

        for w_idx in range(max_state_value):
            for h_idx in range(max_state_value):
                wealth, health = w_idx + 1, h_idx + 1

                invest_value = _calculate_expected_value(
                    wealth, health, old_value_function, local_params,
                    action_health_change=health_delta_array[h_idx],
                    action_health_prob=P_H_increase,
                    action_cost=invest_cost_array[h_idx]
                )
                
                save_value = _calculate_expected_value(
                    wealth, health, old_value_function, local_params,
                    action_health_change=-health_decline_array[h_idx],
                    action_health_prob=P_H_decrease,
                    action_cost=0
                )

                if invest_value > save_value:
                    value_function[w_idx, h_idx] = invest_value
                    policy[w_idx, h_idx] = 1
                else:
                    value_function[w_idx, h_idx] = save_value
                    policy[w_idx, h_idx] = 0
        
        norm = np.linalg.norm(value_function - old_value_function)

    return policy
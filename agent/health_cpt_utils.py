# agent/health_cpt_utils.py

import torch

# --- CPT & Utility Functions ---

def probability_weighting(p, gamma):
    """
    Applies a probability weighting function from Cumulative Prospect Theory (CPT).
    """
    return (p**gamma) / ((p**gamma + (1 - p)**gamma)**(1 / gamma))

def cpt_value_function(x, params):
    """
    Applies the value function from Cumulative Prospect Theory (CPT).
    Note: This function expects a scalar input, as used in the value iteration engine.
    """
    if x >= 0:
        return x**params["theta"]
    return -params["omega"] * (-x)**params["eta"]

def compute_new_wealth(w, wealth_update_scale, utility_val):
    """Calculates the agent's new wealth based on a utility-driven adjustment."""
    delta = utility_val - w
    return w + wealth_update_scale * delta

def utility(w, h, alpha, rate=1.0):
    """Calculates the Cobb-Douglas utility from wealth and health."""
    # Add a small epsilon to prevent log(0) or power of zero issues if w or h is 0
    return (w + 1e-6)**alpha * (h + 1e-6)**(rate - alpha)

# --- Functions for Health/Cost Dynamics ---

def _calculate_base_health_change(h):
    """
    Calculates the potential health decline from not investing (natural decay).
    This now uses torch functions to be compatible with both torch and numpy inputs.
    """
    k = torch.log(torch.tensor(10.0)) / 150
    return 10 * torch.exp(-k * h) + 1

def compute_health_delta(h, params):
    """
    Calculates the POSITIVE change in health from a successful investment.
    This now uses torch functions to be compatible with both torch and numpy inputs.
    """
    base_delta = _calculate_base_health_change(h)
    return base_delta * params.get('efficacy_multiplier', 1.0)

def compute_health_decline(h):
    """
    Calculates the potential health decline from not investing (natural decay).
    This now uses torch functions to be compatible with both torch and numpy inputs.
    """
    return _calculate_base_health_change(h)

def compute_health_cost(h, params):
    """Calculates the cost of investing to improve health."""
    # Cost is based on the potential gain, but subsidized.
    base_cost = -compute_health_delta(h, params) + 11
    return base_cost * params.get('cost_subsidy_factor', 1.0)
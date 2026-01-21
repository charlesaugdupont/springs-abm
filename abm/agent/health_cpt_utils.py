# abm/agent/health_cpt_utils.py
from typing import Dict
import torch

# --- CPT & Utility Functions ---

def probability_weighting(p: float, gamma: float) -> float:
    """Applies the Prelec probability weighting function from CPT."""
    return (p**gamma) / ((p**gamma + (1 - p)**gamma)**(1 / gamma))

def cpt_value_function(x: float, params: Dict[str, float]) -> float:
    """Applies the CPT value function, sensitive to gains and losses."""
    if x >= 0:
        return x**params["theta"]
    return -params["omega"] * (-x)**params["eta"]

def compute_new_wealth(w: torch.Tensor, wealth_update_scale: float, utility_val: torch.Tensor) -> torch.Tensor:
    """Calculates the agent's new wealth based on a utility-driven adjustment."""
    delta = utility_val - w
    return w + wealth_update_scale * delta

def utility(w: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    """Calculates the Cobb-Douglas utility from wealth and health."""
    # Add a small epsilon to prevent log(0) or power of zero issues
    return (w + 1e-9)**alpha * (h + 1e-9)**(rate - alpha)

# --- Functions for Health/Cost Dynamics ---

def _calculate_base_health_change(h: torch.Tensor) -> torch.Tensor:
    """Calculates the potential health change, which decays as health increases."""
    k = torch.log(torch.tensor(10.0)) / 150.0
    return 10 * torch.exp(-k * h) + 1

def compute_health_delta(h: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    """Calculates the POSITIVE change in health from a successful investment."""
    base_delta = _calculate_base_health_change(h)
    return base_delta * params.get('efficacy_multiplier', 1.0)

def compute_health_decline(h: torch.Tensor) -> torch.Tensor:
    """Calculates the potential health decline from not investing (natural decay)."""
    return _calculate_base_health_change(h)

def compute_health_cost(h: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    """Calculates the cost of investing to improve health."""
    base_cost = -compute_health_delta(h, params) + 11
    return base_cost * params.get('cost_subsidy_factor', 1.0)
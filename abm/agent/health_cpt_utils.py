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
    return -params["lambda"] * (-x)**params["eta"]

def utility(w: torch.Tensor, h: torch.Tensor, alpha: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    """Calculates the Cobb-Douglas utility from wealth and health."""
    # Add a small epsilon to prevent log(0) or power of zero issues
    return (w + 1e-9)**alpha * (h + 1e-9)**(rate - alpha)
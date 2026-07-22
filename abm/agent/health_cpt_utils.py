# abm/agent/health_cpt_utils.py
import torch


# ---------------------------------------------------------------------------
# CPT building blocks
# ---------------------------------------------------------------------------

def probability_weighting(p: float, gamma: float) -> float:
    """
    Prelec (1998) probability weighting function.

    Maps an objective probability p ∈ [0, 1] to a decision weight.
    gamma < 1 produces the typical inverse-S shape (overweighting of small
    probabilities, underweighting of large ones).
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    denom = (p ** gamma + (1.0 - p) ** gamma) ** (1.0 / gamma)
    return (p ** gamma) / denom


def cpt_value_function(x: float, theta: float, eta: float) -> float:
    """
    CPT value function (Tversky & Kahneman 1992).

    theta: gain sensitivity exponent. eta: loss sensitivity exponent. Read
    from steering_parameters.cpt_theta/cpt_eta by the caller, so they're
    reachable via the same dot-path sweep/override mechanism as every other
    steering parameter. The loss-aversion coefficient lambda is applied by
    the caller so that it can vary per agent.

    Returns an *unscaled* value; multiply losses by -lambda outside.
    """
    if x >= 0:
        return x ** theta
    return -((-x) ** eta)


def utility(
    w: torch.Tensor,
    h: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Cobb-Douglas utility from wealth (w) and health (h).

        U(w, h) = w^alpha * h^(1 - alpha)

    alpha ∈ (0, 1) controls the relative weight placed on wealth vs. health.
    A small epsilon prevents log(0) / 0^alpha issues.
    """
    return (w + 1e-9) ** alpha * (h + 1e-9) ** (1.0 - alpha)

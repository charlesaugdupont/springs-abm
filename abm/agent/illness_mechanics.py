# abm/agent/illness_mechanics.py
import torch

# --- Constants for Severity Calculation ---
PATHOGEN_BASE_SEVERITY = {"rota": 0.4, "campy": 0.3}

# Age-effect model: severity multiplier starts at (1 + AGE_MAX_MULTIPLIER) for
# newborns and decays exponentially toward 1.0 as the child ages.
AGE_MAX_MULTIPLIER = 1.5
AGE_DECAY_RATE: float = 0.08

# Wealth resulience: fraction by which maximum wealth can reduce severity
WEALTH_RESILIENCE_FACTOR = 0.5

# Immunity reductions: applied multiplicatively to base severity
IMMUNITY_FACTOR_VACCINE = 0.4
IMMUNITY_FACTOR_PER_INFECTION = 0.20


def _get_age_effect(age_in_months: torch.Tensor, is_child: torch.Tensor) -> torch.Tensor:
    """
    Returns a per-agent severity multiplier based on age.
    Adults receive a multiplier of 1.0. Children receive a higher multiplier
    that decays exponentially from (1 + AGE_MAX_MULTIPLIER) at birth toward 1.0.
    """
    multiplier = torch.ones_like(age_in_months)

    if not torch.any(is_child):
        return multiplier

    child_ages = age_in_months[is_child]
    pathogen_multiplier = 1.0 + AGE_MAX_MULTIPLIER * torch.exp(-AGE_DECAY_RATE * child_ages)
    multiplier[is_child] = pathogen_multiplier

    return multiplier


def _get_resilience_effect(wealth: torch.Tensor) -> torch.Tensor:
    """Returns a per-agent severity multiplier based on wealth (lower wealth → higher severity)."""
    return 1.0 - (wealth * WEALTH_RESILIENCE_FACTOR)


def _get_immunity_effect(vaccine_status: torch.Tensor | None, num_infections: torch.Tensor) -> torch.Tensor:
    """Returns a per-agent severity multiplier that accounts for vaccine and prior-infection immunity."""
    immunity_reduction = torch.zeros_like(num_infections, dtype=torch.float)
    if vaccine_status is not None:
        immunity_reduction += torch.where(vaccine_status, IMMUNITY_FACTOR_VACCINE, 0.0)
    immunity_reduction += num_infections.float() * IMMUNITY_FACTOR_PER_INFECTION
    return torch.clamp(1.0 - immunity_reduction, min=0.1)


def calculate_illness_severity(
    pathogen_name: str,
    is_child: torch.Tensor,
    age: torch.Tensor,
    wealth: torch.Tensor,
    vaccine_status: torch.Tensor | None,
    num_infections: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates a normalised (0-1) illness severity score for a subset of agents
    by combining age, wealth-resilience, and immunity effects multiplicatively.

    All input tensors must be pre-filtered to the target agent subset before
    calling this function.
    """
    device = is_child.device
    num_agents = len(is_child)
    severity = torch.zeros(num_agents, device=device)

    if not torch.any(is_child):
        return severity

    base = PATHOGEN_BASE_SEVERITY.get(pathogen_name, 0.3)

    age_effect = _get_age_effect(age, is_child)
    resilience_effect = _get_resilience_effect(wealth)
    immunity_effect = _get_immunity_effect(vaccine_status, num_infections)

    final_severity = base * age_effect * resilience_effect * immunity_effect
    return torch.clamp(final_severity, 0.0, 1.0)


def calculate_illness_duration(severity: torch.Tensor) -> torch.Tensor:
    """
    Calculates the initial illness duration in timesteps (days) based on severity.
    """
    # Maps illness severity to an initial duration in days.
    # Severity = 0 -> 2 days; severity = 1 -> 12 days
    base_duration = 2
    duration_from_severity = torch.floor(severity * 10)
    return (base_duration + duration_from_severity).int()
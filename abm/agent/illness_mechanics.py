# abm/agent/illness_mechanics.py
import torch

# --- Constants for Severity Calculation ---
# These would be tuned based on literature/data
PATHOGEN_BASE_SEVERITY = {"rota": 0.4, "campy": 0.3}
AGE_SEVERITY_MULTIPLIER = 1.5  # Younger children are more affected
WEALTH_RESILIENCE_FACTOR = 0.5 # Higher wealth improves resilience
IMMUNITY_FACTOR_VACCINE = 0.3  # Reduction for vaccine
IMMUNITY_FACTOR_PER_INFECTION = 0.15 # Reduction per prior infection

def _get_age_effect(pathogen_name: str, age_in_months: torch.Tensor, is_child: torch.Tensor) -> torch.Tensor:
    """
    Calculates a severity multiplier based on age, with pathogen-specific logic.
    Assumes `age_in_months` is a tensor for all agents.
    """
    device = age_in_months.device
    multiplier = torch.ones_like(age_in_months)

    # Only apply age effects to children
    if not torch.any(is_child):
        return multiplier

    child_ages = age_in_months[is_child]

    if pathogen_name == "rota":
        # Log-normal distribution to model peak risk between 6-24 months.
        # mu and sigma are chosen to center the peak around 12-15 months.
        mu = torch.tensor(2.6, device=device)   # Corresponds to ~13.5 months
        sigma = torch.tensor(0.4, device=device)
        
        # Calculate the log-normal PDF value
        pdf_vals = (1 / (child_ages * sigma * torch.sqrt(torch.tensor(2 * torch.pi, device=device)))) * \
                   torch.exp(-((torch.log(child_ages + 1e-9) - mu)**2) / (2 * sigma**2))
        
        # Scale the PDF to create a reasonable multiplier (e.g., peak at ~2.5x)
        pathogen_multiplier = 1.0 + (pdf_vals * 15.0)
        multiplier[is_child] = pathogen_multiplier

    elif pathogen_name == "campy":
        # Exponential decay model: highest risk for the youngest, decreasing over time.
        # Multiplier starts high (e.g., 2.5x at age 0) and decays.
        max_multiplier = 1.5
        decay_rate = 0.15
        pathogen_multiplier = 1.0 + max_multiplier * torch.exp(-decay_rate * child_ages)
        multiplier[is_child] = pathogen_multiplier

    else:
        multiplier[is_child] = AGE_SEVERITY_MULTIPLIER

    return multiplier


def _get_resilience_effect(wealth: torch.Tensor) -> torch.Tensor:
    """Calculates severity multiplier based on wealth."""
    return 1.0 - (wealth * WEALTH_RESILIENCE_FACTOR)

def _get_immunity_effect(vaccine_status: torch.Tensor, num_infections: torch.Tensor) -> torch.Tensor:
    """Calculates severity reduction from immunity."""
    immunity_reduction = torch.zeros_like(num_infections, dtype=torch.float)
    if vaccine_status is not None:
        immunity_reduction += torch.where(vaccine_status, IMMUNITY_FACTOR_VACCINE, 0.0)
    immunity_reduction += (num_infections * IMMUNITY_FACTOR_PER_INFECTION)
    return torch.clamp(1.0 - immunity_reduction, min=0.1)

def calculate_illness_severity(
    pathogen_name: str,
    is_child: torch.Tensor,
    age: torch.Tensor,
    wealth: torch.Tensor,
    vaccine_status: torch.Tensor,
    num_infections: torch.Tensor
) -> torch.Tensor:
    """
    Calculates a normalized (0-1) illness severity score for agents by combining modular effects.
    """
    num_agents = len(is_child)
    device = is_child.device
    severity = torch.zeros(num_agents, device=device)

    if not torch.any(is_child):
        return severity

    # 1. Base Severity from Pathogen
    base = PATHOGEN_BASE_SEVERITY.get(pathogen_name, 0.3)

    # 2. Combine modular effect multipliers
    age_effect = _get_age_effect(pathogen_name, age, is_child)
    resilience_effect = _get_resilience_effect(wealth)
    immunity_effect = _get_immunity_effect(vaccine_status, num_infections)

    # 3. Combine Factors (Multiplicatively)
    final_severity = base * age_effect * resilience_effect * immunity_effect

    return torch.clamp(final_severity, 0.0, 1.0)


def calculate_illness_duration(severity: torch.Tensor) -> torch.Tensor:
    """
    Calculates the initial illness duration in timesteps (days) based on severity.
    """
    # A simple linear mapping: min 2 days, max ~10 days
    # severity=0 -> 2 days, severity=1 -> 12 days
    base_duration = 2
    duration_from_severity = torch.floor(severity * 10)
    return base_duration + duration_from_severity
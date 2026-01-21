# abm/agent/illness_mechanics.py
import torch

# --- Constants for Severity Calculation ---
# These would be tuned based on literature/data
PATHOGEN_BASE_SEVERITY = {"rota": 0.4, "campy": 0.3}
AGE_SEVERITY_MULTIPLIER = 1.5  # Younger children are more affected
WEALTH_RESILIENCE_FACTOR = 0.5 # Higher wealth improves resilience
IMMUNITY_FACTOR_VACCINE = 0.3  # Reduction for vaccine
IMMUNITY_FACTOR_PER_INFECTION = 0.15 # Reduction per prior infection

def calculate_illness_severity(
    pathogen_name: str,
    is_child: torch.Tensor,
    wealth: torch.Tensor,
    vaccine_status: torch.Tensor, # Note: We may need to add this property
    num_infections: torch.Tensor
) -> torch.Tensor:
    """
    Calculates a normalized (0-1) illness severity score for agents.
    """
    num_agents = len(is_child)
    device = is_child.device
    severity = torch.zeros(num_agents, device=device)

    # Only children can get symptoms in this model version
    if not torch.any(is_child):
        return severity

    # 1. Base Severity from Pathogen
    base = PATHOGEN_BASE_SEVERITY.get(pathogen_name, 0.3)

    # 2. Age Factor (simplification: only children are vulnerable)
    # A more complex model could use an actual age property.
    age_effect = torch.where(is_child, torch.tensor(AGE_SEVERITY_MULTIPLIER, device=device), 1.0)

    # 3. Nutrition/Resilience Factor (from wealth)
    # Agents with higher wealth are more resilient (lower severity multiplier)
    resilience_effect = 1.0 - (wealth * WEALTH_RESILIENCE_FACTOR)

    # 4. Immunity Factor (from vaccine and prior infections)
    immunity_reduction = torch.zeros_like(wealth)
    # Note: Assumes vaccine_status tensor exists. If not, this part is ignored.
    if vaccine_status is not None:
         immunity_reduction += torch.where(vaccine_status, IMMUNITY_FACTOR_VACCINE, 0.0)
    immunity_reduction += (num_infections * IMMUNITY_FACTOR_PER_INFECTION)
    immunity_effect = torch.clamp(1.0 - immunity_reduction, min=0.1) # Cannot have zero or negative effect

    # 5. Combine Factors (Multiplicatively)
    # This non-linear combination allows factors to compound
    final_severity = base * age_effect * resilience_effect * immunity_effect

    # Clamp to ensure the result is always between 0 and 1
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
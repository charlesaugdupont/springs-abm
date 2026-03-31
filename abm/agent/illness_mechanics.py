# abm/agent/illness_mechanics.py
import torch
from config import IllnessMechanicsConfig

def _get_age_effect(
    age_in_months: torch.Tensor,
    is_child: torch.Tensor,
    cfg: IllnessMechanicsConfig,
) -> torch.Tensor:
    """
    Returns a per-agent severity multiplier based on age.

    Adults receive a multiplier of 1.0. Children receive a higher multiplier
    that decays exponentially from (1 + age_max_multiplier) at birth toward
    1.0 as the child ages.
    """
    multiplier = torch.ones_like(age_in_months)

    if not torch.any(is_child):
        return multiplier

    child_ages = age_in_months[is_child]
    pathogen_multiplier = 1.0 + cfg.age_max_multiplier * torch.exp(-cfg.age_decay_rate * child_ages)
    multiplier[is_child] = pathogen_multiplier
    return multiplier


def _get_immunity_effect(
    vaccine_status: torch.Tensor | None,
    num_infections: torch.Tensor,
    cfg: IllnessMechanicsConfig,
) -> torch.Tensor:
    """
    Returns a per-agent severity multiplier that accounts for vaccine and
    prior-infection immunity. The multiplier is clamped to [0.1, 1.0] so
    that even fully immune agents retain a small residual severity.
    """
    immunity_reduction = torch.zeros_like(num_infections, dtype=torch.float)
    if vaccine_status is not None:
        immunity_reduction += torch.where(
            vaccine_status,
            torch.tensor(cfg.immunity_factor_vaccine, dtype=torch.float, device=num_infections.device),
            torch.zeros(1, device=num_infections.device),
        )
    immunity_reduction += num_infections.float() * cfg.immunity_factor_per_infection
    return torch.clamp(1.0 - immunity_reduction, min=0.1)


def calculate_illness_severity(
    pathogen_name: str,
    is_child: torch.Tensor,
    age: torch.Tensor,
    vaccine_status: torch.Tensor | None,
    num_infections: torch.Tensor,
    cfg: IllnessMechanicsConfig,
) -> torch.Tensor:
    """
    Calculates a normalised (0-1) illness severity score for a subset of
    agents by combining age and immunity effects multiplicatively.

    Wealth is intentionally excluded: its effect on outcomes is captured
    through care-seeking access and economic dynamics, not through a direct
    biological severity reduction.

    All input tensors must be pre-filtered to the target agent subset before
    calling this function.
    """
    device = is_child.device
    num_agents = len(is_child)
    severity = torch.zeros(num_agents, device=device)

    if not torch.any(is_child):
        return severity

    base_severity_map = {
        "rota": cfg.base_severity_rota,
        "campy": cfg.base_severity_campy,
    }
    base = base_severity_map.get(pathogen_name, 0.3)

    age_effect = _get_age_effect(age, is_child, cfg)
    immunity_effect = _get_immunity_effect(vaccine_status, num_infections, cfg)

    final_severity = base * age_effect * immunity_effect
    return torch.clamp(final_severity, 0.0, 1.0)


def calculate_illness_duration(
    severity: torch.Tensor,
    cfg: IllnessMechanicsConfig,
) -> torch.Tensor:
    """
    Calculates the initial illness duration in days based on severity.

    The expected duration scales linearly from ``duration_min_days`` (at
    severity = 0) to ``duration_max_days`` (at severity = 1). Gaussian
    noise with std ``duration_noise_std`` is added to capture the
    substantial individual variation in illness length that is independent
    of severity. The result is clamped to a minimum of 1 day.
    """
    mean_duration = (
        cfg.duration_min_days
        + severity * (cfg.duration_max_days - cfg.duration_min_days)
    )
    noise = torch.randn_like(mean_duration) * cfg.duration_noise_std
    duration = torch.round(mean_duration + noise).int()
    return torch.clamp(duration, min=1)

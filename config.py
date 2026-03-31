# config.py

"""Configuration parameters for the SVEIR model."""
from typing import List, Union
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

# --- CPT Constants (fixed, not per-agent) ---
CPT_THETA: float = 0.88 # gain sensitivity exponent in the value function
CPT_ETA: float = 0.88 # loss sensitivity exponent in the value function

# --- Pathogen Configuration ---

class PathogenConfig(BaseModel):
    """Base class for pathogen-specific parameters."""
    name: str
    initial_exposed_proportion: float = 0.03
    recovery_rate: float
    exposure_period: int

class RotavirusConfig(PathogenConfig):
    """Parameters specific to Rotavirus."""
    name: str = "rota"
    infection_prob_mean: float = 0.20
    infection_prob_std: float = 0.02
    recovery_rate: float = 0.2
    exposure_period: int = 2
    vaccination_rate: float = 0.005
    vaccine_efficacy: float = 0.55

class CampylobacterConfig(PathogenConfig):
    """Parameters specific to Campylobacter."""
    name: str = "campy"
    # Beta-Poisson Dose Response Constants (zoonotic route)
    beta_poisson_alpha: float = 0.038
    beta_poisson_beta: float = 0.022
    # Disease Dynamics
    recovery_rate: float = 0.15 # ~1 week duration
    exposure_period: int = 3
    # Environmental
    human_animal_interaction_rate: float = 2.0
    # Fecal-oral (household) route
    fecal_oral_prob: float = 0.03 # per-contact probability within household

# --- Illness Mechanics Constants ---

class IllnessMechanicsConfig(BaseModel):
    """Parameters controlling how illness severity and duration are calculated."""
    # Base severity per pathogen (0-1 scale)
    base_severity_rota: float = 0.4
    base_severity_campy: float = 0.3

    # Age effect: severity multiplier decays from (1 + age_max_multiplier) at
    # birth toward 1.0 as the child ages.
    age_max_multiplier: float = 1.5
    age_decay_rate: float = 0.08

    # Immunity reductions applied multiplicatively to base severity
    immunity_factor_vaccine: float = 0.4
    immunity_factor_per_infection: float = 0.20

    # Duration model: duration ~ Normal(mean, std) where mean is linearly
    # scaled by severity. Values are in days.
    duration_min_days: float = 2.0 # expected duration at severity = 0
    duration_max_days: float = 12.0 # expected duration at severity = 1
    duration_noise_std: float = 1.5 # stochastic spread around the mean

# --- General Model Configuration ---

class GridCreationParams(BaseModel):
    """Arguments for creating the spatial grid environment."""
    method: str = "realistic_import"
    grid_id: str | None = None
    x: int | None = 75
    y: int | None = 75
    properties: dict | None = None
    model_config = ConfigDict(validate_default=True)

class SteeringParamsSVEIR(BaseModel):
    """Steering parameters used within each step of the SVEIR model."""
    # Shared / Global Parameters
    prior_infection_immunity_factor: float = 0.5

    # Water Parameters (Shared Reservoir)
    human_to_water_infection_prob: float = 0.0001
    water_to_human_infection_prob: float = 0.05
    water_recovery_prob: float = 0.2
    shock_daily_prob: float = 1/30

    # Spatial / social parameters
    social_interaction_radius: float = 5.0

    # --- Care-Seeking Parameters ---
    moderate_severity_threshold: float = 0.2
    severe_severity_threshold: float = 0.7
    cost_of_care: float = 0.05 # Cost as a proportion of max wealth
    treatment_success_prob: float = 0.80
    duration_reduction_on_success: int = 5 # Days illness is shortened on successful treatment
    natural_worsening_prob: float = 0.35 # Prob illness worsens if untreated
    parent_stress_health_impact: float = 0.10
    untreated_severity_penalty: float = 0.20
    severity_health_impact_factor: float = 0.02 # Daily health reduction per unit of severity
    daily_health_recovery_rate: float = 0.005 # Base daily recovery when not sick

    # Income and wealth dynamics
    daily_income_rate: float = 0.02
    daily_cost_of_living: float = 0.01
    health_based_income: bool = True

class SVEIRConfig(BaseModel):
    """Main configuration class for the SVEIR model."""
    model_identifier: str = "sveir_model"
    description: str = "Configuration for the SVEIR agent-based model."
    device: str = "cpu"
    seed: int = 23
    number_agents: PositiveInt = 5000
    spatial: bool = True
    spatial_creation_args: GridCreationParams = GridCreationParams()
    step_target: PositiveInt = 150

    # Pathogen Configuration
    pathogens: List[Union[RotavirusConfig, CampylobacterConfig]] = [RotavirusConfig(), CampylobacterConfig()]

    # Demographic Parameters
    average_household_size: float = 3.2
    child_probability: float = 0.145

    # Parameters for Agent Personas
    num_agent_personas: int = 32
    alpha_range: list[float] = [0.1, 0.9]
    gamma_range: list[float] = [0.4, 0.9]
    lambda_range: list[float] = [1.0, 3.0]

    # Illness mechanics
    illness_mechanics: IllnessMechanicsConfig = IllnessMechanicsConfig()

    steering_parameters: SteeringParamsSVEIR = SteeringParamsSVEIR()

    model_config = ConfigDict(
        validate_default=True,
        protected_namespaces=(),
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )

    @field_validator('pathogens', mode='before')
    def set_pathogen_types(cls, v):
        if not v:
            return v
        pathogen_map = {'rota': RotavirusConfig, 'campy': CampylobacterConfig}
        return [pathogen_map[p['name']](**p) if isinstance(p, dict) else p for p in v]

    @classmethod
    def from_dict(cls, cfg):
        if not isinstance(cfg, dict):
            raise TypeError("Input must be a dictionary.")
        return cls(**cfg)

    def to_yaml(self, config_file):
        if Path(config_file).exists():
            print(f"Overwriting config file {config_file}.")
        cfg = self.model_dump(by_alias=True, warnings=False)
        def _convert_tensors(nested_dict):
            for key, value in nested_dict.items():
                if isinstance(value, torch.Tensor):
                    nested_dict[key] = value.tolist()
                elif isinstance(value, dict):
                    _convert_tensors(value)
            return nested_dict
        cfg = _convert_tensors(cfg)
        with open(config_file, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    @classmethod
    def from_yaml(cls, config_file):
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file {config_file} not found.")
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)

SVEIRCONFIG = SVEIRConfig()

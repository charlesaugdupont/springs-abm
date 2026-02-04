# config.py

"""Configuration parameters for the SVEIR model."""
from typing import List, Union
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

# --- Pathogen Configuration ---

class PathogenConfig(BaseModel):
    """Base class for pathogen-specific parameters."""
    name: str
    initial_infected_proportion: float = 0.03
    recovery_rate: float
    exposure_period: int

class RotavirusConfig(PathogenConfig):
    """Parameters specific to Rotavirus."""
    name: str = "rota"
    infection_prob_mean: float = 0.20
    infection_prob_std: float = 0.002
    recovery_rate: float = 0.2
    exposure_period: int = 2
    vaccination_rate: float = 0.01
    vaccine_efficacy: float = 0.9

class CampylobacterConfig(PathogenConfig):
    """Parameters specific to Campylobacter."""
    name: str = "campy"
    # Beta-Poisson Dose Response Constants
    beta_poisson_alpha: float = 0.038
    beta_poisson_beta: float = 0.022
    # Disease Dynamics
    recovery_rate: float = 0.15  # ~1 week duration
    exposure_period: int = 3
    # Environmental
    human_animal_interaction_rate: float = 2.0

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
    npath: str = "./agent_data.zarr"
    ndata: list | None = Field(default_factory=lambda: ["all_except", ["a_table"]])
    mode: str = "w"

    # Shared / Global Parameters
    infection_reduction_factor_per_health_unit: float = 0.5
    theta: float = 0.88
    eta: float = 0.88
    infection_health_shock: float = 0.2

    # Water Parameters (Shared Reservoir)
    human_to_water_infection_prob: float = 0.0001
    water_to_human_infection_prob: float = 0.05
    water_recovery_prob: float = 0.2
    shock_daily_prob: float = 1/30
    shock_infection_prob: float = 1.0

    # Other simulation parameters
    truncation_weight: float = 1.0e-10
    proximity_decay_rate: float = 0.5
    data_collection_period: int = 0
    data_collection_list: list[int] | None = None
    max_state_value: float = 1.0
    prior_infection_immunity_factor: float = 1.5
    social_interaction_radius: float = 5.0

    # --- Care-Seeking Parameters ---
    moderate_severity_threshold: float = 0.2
    severe_severity_threshold: float = 0.7
    cost_of_care: float = 0.05  # Cost as a proportion of max wealth
    treatment_success_prob: float = 0.80 # Probability care-seeking is effective
    duration_reduction_on_success: int = 5 # Days illness is shortened
    natural_worsening_prob: float = 0.35 # Prob illness worsens if untreated
    parent_stress_health_impact: float = 0.10 # Health drop for parent if child worsens
    severity_health_impact_factor: float = 0.02 # Daily health reduction per unit of severity

    # Income and wealth dynamics
    daily_income_rate: float = 0.02 # Wealth gained per day for a perfectly healthy adult
    daily_cost_of_living: float = 0.01 # daily wealth expense
    health_based_income: bool = True # Flag to enable the feedback loop

class SVEIRConfig(BaseModel):
    """Main configuration class for the SVEIR model."""
    model_identifier: str = Field("sveir_model", alias='_model_identifier')
    description: str = "Configuration for the SVEIR agent-based model."
    device: str = "cpu"
    seed: int = 42
    number_agents: PositiveInt = 300
    spatial: bool = True
    spatial_creation_args: GridCreationParams = GridCreationParams()    
    step_target: PositiveInt = 150

    # Pathogen Configuration
    pathogens: List[Union[RotavirusConfig, CampylobacterConfig]] = [RotavirusConfig(), CampylobacterConfig()]

    # Demographic Parameters
    average_household_size: int = 4
    child_probability: float = 0.2

    # Parameters for Agent Personas
    num_agent_personas: int = 32
    alpha_range: list[float] = [0.1, 0.9]
    gamma_range: list[float] = [0.4, 0.9]
    lambda_range: list[float] = [1.0, 3.0]

    steering_parameters: SteeringParamsSVEIR = SteeringParamsSVEIR()

    model_config = ConfigDict(
        validate_default=True, protected_namespaces=(), populate_by_name=True,
        validate_assignment=True, extra="forbid",
    )

    @field_validator('pathogens', mode='before')
    def set_pathogen_types(cls, v):
        if not v:
            return v
        pathogen_map = {
            'rota': RotavirusConfig,
            'campy': CampylobacterConfig
        }
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
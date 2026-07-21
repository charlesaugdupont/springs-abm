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
    initial_exposed_proportion: float = 0.01
    recovery_rate: float
    exposure_period: int

class RotavirusConfig(PathogenConfig):
    """Parameters specific to Rotavirus."""
    name: str = "rota"
    # Calibrated against sensitivity.py / experiments/calibration/targets.py
    # empirical target ranges via experiments/calibration/run_calibration.py
    # (LHS search, 4/5 targets met - see experiments/outputs/calibration/).
    infection_prob_mean: float = 0.0015
    infection_prob_std: float = 0.0002
    recovery_rate: float = 0.2968
    exposure_period: int = 2
    vaccination_rate: float = 0.001
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
    # Calibrated (see rota's calibration note above; same search/output).
    human_animal_interaction_rate: float = 0.0114
    # Fecal-oral (household) route
    fecal_oral_prob: float = 0.0154 # per-contact probability within household
    # Background risk due to food consumption
    food_borne_prob: float = 0.0024

    # Household animal ownership (zoonotic route)
    # Ownership probabilities calibrated from a compiled Ghana DHS/MICS-style
    # survey (notebooks/ghana_data.ipynb): inverse-distance-weighted estimate
    # over rural clusters nationwide, since no survey cluster falls inside
    # Akuse itself - effectively a rural-southern-Ghana average. Pig
    # ownership (~1%, within noise of the estimation method) was dropped.
    poultry_ownership_prob: float = 0.536
    ruminant_ownership_prob: float = 0.330

    # Relative contribution of each species to zoonotic risk. Poultry
    # dominates C. jejuni source-attribution studies and the household-
    # exposure literature; ruminants contribute meaningfully but less so via
    # this backyard-proximity pathway. Deliberately uncertain - sweep via
    # experiments rather than treat this as fixed.
    poultry_weight: float = 1.0
    ruminant_weight: float = 0.45

    # Gaussian roam radius (in grid cells) used to diffuse each owning
    # household's animals into the surrounding area - backyard poultry stay
    # close to the yard; grazing/tethered ruminants plausibly range further.
    poultry_roam_sigma: float = 1.0
    ruminant_roam_sigma: float = 2.0

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
    severity_reduction_per_infection: float = 0.20

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
    properties: dict | None = None
    model_config = ConfigDict(validate_default=True)

class SteeringParamsSVEIR(BaseModel):
    """Steering parameters used within each step of the SVEIR model."""
    # Shared / Global Parameters
    prior_infection_immunity_factor: float = 0.15

    # Water Parameters (Shared Reservoir)
    human_to_water_infection_prob: float = 0.0001
    # Calibrated (see RotavirusConfig's calibration note; same search/output).
    water_to_human_infection_prob: float = 0.0006
    water_recovery_prob: float = 0.2
    shock_daily_prob: float = 1/30

    # Spatial / social parameters
    social_interaction_radius: float = 5.0

    # --- Care-Seeking Parameters ---
    cost_of_care: float = 0.025 # Cost as a proportion of max wealth
    treatment_success_prob: float = 0.80
    natural_worsening_prob: float = 0.35 # Prob illness worsens if untreated
    parent_stress_health_impact: float = 0.30
    untreated_severity_penalty: float = 0.20
    severity_health_impact_factor: float = 0.05 # Daily health reduction per unit of severity
    daily_health_recovery_rate: float = 0.001 # Base daily recovery when not sick
    child_health_weight: float = 0.5 # weight placed on child health in the parent's utility function.

    # Income and wealth dynamics
    daily_income_rate: float = 0.03
    daily_cost_of_living: float = 0.025
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

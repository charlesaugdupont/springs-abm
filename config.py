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
    name: str = Field(..., description="Pathogen identifier ('rota' or 'campy').")
    initial_exposed_proportion: float = Field(
        0.01, description="Fraction of agents seeded as exposed at simulation start."
    )
    recovery_rate: float = Field(..., description="Daily probability of recovering from infection.")
    exposure_period: int = Field(
        ..., description="Latent period (days) between exposure and becoming infectious."
    )

class RotavirusConfig(PathogenConfig):
    """Parameters specific to Rotavirus."""
    name: str = Field("rota", description="Pathogen identifier.")
    # Calibrated against sensitivity.py / experiments/calibration/targets.py
    # empirical target ranges via experiments/calibration/run_calibration.py
    # (LHS search round 5, recovery_rate bound restricted to the ~3-7 day
    # literature range for rotavirus illness duration - 3/5 targets met,
    # both rota episodes/peak in-range misses are ~8% over their upper
    # bound; see experiments/outputs/calibration/).
    infection_prob_mean: float = Field(
        0.0011,
        description="Mean per-exposure infection probability. LHS-calibrated to hit literature "
                    "target ranges for episodes/child-year and peak prevalence.",
    )
    infection_prob_std: float = Field(
        0.0002, description="Spread of the per-agent infection-probability draw around the mean."
    )
    recovery_rate: float = Field(
        0.3005,
        description="Daily recovery probability (~3.3 day expected illness duration). Constrained "
                    "to the ~3-7 day rotavirus illness-duration literature range; excluded from "
                    "calibration search for this reason.",
    )
    exposure_period: int = Field(
        2, description="Latent period (days) before an exposed agent becomes infectious."
    )
    vaccination_rate: float = Field(
        0.001, description="Daily probability an unvaccinated agent receives the rotavirus vaccine."
    )
    vaccine_efficacy: float = Field(
        0.55, description="Reduction in severity/susceptibility conferred by vaccination."
    )

class CampylobacterConfig(PathogenConfig):
    """Parameters specific to Campylobacter."""
    name: str = Field("campy", description="Pathogen identifier.")
    # Beta-Poisson Dose Response Constants (zoonotic route)
    beta_poisson_alpha: float = Field(
        0.038, description="Beta-Poisson dose-response shape parameter for the zoonotic route."
    )
    beta_poisson_beta: float = Field(
        0.022, description="Beta-Poisson dose-response scale parameter for the zoonotic route."
    )
    # Disease Dynamics
    # Calibrated (see rota's calibration note above; same search/output).
    recovery_rate: float = Field(
        0.1466,
        description="Daily recovery probability (~6.8 day expected illness duration). Literature-"
                    "grounded illness-duration constraint; excluded from calibration search.",
    )
    exposure_period: int = Field(
        3, description="Latent period (days) before an exposed agent becomes infectious."
    )
    # Environmental
    human_animal_interaction_rate: float = Field(
        0.0062, description="Daily probability of zoonotic-route exposure for an animal-owning household."
    )
    # Fecal-oral (household) route
    fecal_oral_prob: float = Field(
        0.0105, description="Per-contact fecal-oral transmission probability within an infected household."
    )
    # Background risk due to food consumption
    food_borne_prob: float = Field(
        0.0014, description="Daily background infection probability via food-borne exposure."
    )

    # Household animal ownership (zoonotic route)
    # Ownership probabilities calibrated from a compiled Ghana DHS/MICS-style
    # survey (notebooks/ghana_data.ipynb): inverse-distance-weighted estimate
    # over rural clusters nationwide, since no survey cluster falls inside
    # Akuse itself - effectively a rural-southern-Ghana average. Pig
    # ownership (~1%, within noise of the estimation method) was dropped.
    poultry_ownership_prob: float = Field(
        0.536,
        description="Probability a household owns poultry. Sourced from a compiled Ghana DHS/MICS-"
                    "style rural survey (no cluster falls inside Akuse itself; a rural-southern-"
                    "Ghana average).",
    )
    ruminant_ownership_prob: float = Field(
        0.330,
        description="Probability a household owns ruminants (goats/sheep/cattle). Same DHS/MICS-"
                    "style survey source as poultry ownership.",
    )

    # Relative contribution of each species to zoonotic risk. Poultry
    # dominates C. jejuni source-attribution studies and the household-
    # exposure literature; ruminants contribute meaningfully but less so via
    # this backyard-proximity pathway. Deliberately uncertain - sweep via
    # experiments rather than treat this as fixed.
    poultry_weight: float = Field(
        1.0,
        description="Relative zoonotic-risk weight for poultry. Literature-motivated (poultry "
                    "dominates C. jejuni source-attribution studies) but not quantitatively fit - "
                    "a deliberately uncertain assumption.",
    )
    ruminant_weight: float = Field(
        0.45,
        description="Relative zoonotic-risk weight for ruminants, versus poultry's 1.0. Same "
                    "qualitative literature motivation, not quantitatively fit.",
    )

    # Gaussian roam radius (in grid cells) used to diffuse each owning
    # household's animals into the surrounding area - backyard poultry stay
    # close to the yard; grazing/tethered ruminants plausibly range further.
    poultry_roam_sigma: float = Field(
        1.0, description="Gaussian roam radius (grid cells) diffusing poultry risk around an owning household."
    )
    ruminant_roam_sigma: float = Field(
        2.0, description="Gaussian roam radius (grid cells) diffusing ruminant risk - wider than "
                          "poultry, reflecting grazing/tethered range."
    )

# --- Illness Mechanics Constants ---

class IllnessMechanicsConfig(BaseModel):
    """Parameters controlling how illness severity and duration are calculated."""
    # Base severity per pathogen (0-1 scale)
    base_severity_rota: float = Field(0.4, description="Base rotavirus illness severity (0-1 scale) before age/immunity adjustments.")
    base_severity_campy: float = Field(0.3, description="Base campylobacter illness severity (0-1 scale) before age/immunity adjustments.")

    # Age effect: severity multiplier decays from (1 + age_max_multiplier) at
    # birth toward 1.0 as the child ages.
    age_max_multiplier: float = Field(
        1.5, description="Extra severity multiplier at birth (decays toward 1.0 with age)."
    )
    age_decay_rate: float = Field(
        0.08, description="Rate at which the age-related severity multiplier decays toward 1.0."
    )

    # Immunity reductions applied multiplicatively to base severity
    immunity_factor_vaccine: float = Field(
        0.4, description="Severity reduction factor conferred by vaccination."
    )
    severity_reduction_per_infection: float = Field(
        0.20, description="Severity reduction per prior infection (partial, never-complete immunity)."
    )

    # Duration model: duration ~ Normal(mean, std) where mean is linearly
    # scaled by severity. Values are in days.
    duration_min_days: float = Field(2.0, description="Expected illness duration (days) at severity = 0.")
    duration_max_days: float = Field(12.0, description="Expected illness duration (days) at severity = 1.")
    duration_noise_std: float = Field(1.5, description="Stochastic spread (days) around the expected illness duration.")

# --- General Model Configuration ---

class GridCreationParams(BaseModel):
    """Arguments for creating the spatial grid environment."""
    method: str = Field("realistic_import", description="Grid-creation method (internal - not user-configurable).")
    grid_id: str | None = Field(None, description="ID of the cached spatial grid to load (internal - not user-configurable).")
    properties: dict | None = Field(None, description="Extra grid-creation properties (internal - not user-configurable).")
    model_config = ConfigDict(validate_default=True)

class SteeringParamsSVEIR(BaseModel):
    """Steering parameters used within each step of the SVEIR model."""
    # Shared / Global Parameters
    prior_infection_immunity_factor: float = Field(
        0.15, description="Additional immunity factor applied per prior infection, shared across pathogens."
    )

    # Water Parameters (Shared Reservoir)
    human_to_water_infection_prob: float = Field(
        0.0001, description="Daily probability an infectious agent contaminates its local water source."
    )
    # Calibrated (see RotavirusConfig's calibration note; same search/output).
    water_to_human_infection_prob: float = Field(
        0.0097,
        description="Daily probability a susceptible agent is infected via contaminated water. "
                    "LHS-calibrated tuning knob, not independently measured for Akuse.",
    )
    water_recovery_prob: float = Field(
        0.2, description="Daily probability a contaminated water source naturally reverts to clean."
    )
    shock_daily_prob: float = Field(
        1/30,
        description="Daily probability of a water-contamination shock event. Stated in the paper "
                    "as an assumption pending calibration against real water-quality/infrastructure "
                    "failure data - explored via sensitivity, not a validated real-world value.",
    )

    # Spatial / social parameters
    social_interaction_radius: float = Field(
        5.0, description="Radius (grid cells) within which agents can socially interact/transmit."
    )

    # --- Care-Seeking Parameters ---
    # cost_of_care/daily_income_rate calibrated together against the Ghana DHS
    # childhood-diarrhea care-seeking rate (69.2%, 95% CI 65.6-72.8% from the
    # n=621 survey) via experiments/care_seeking/run_care_seeking_calibration.py.
    # The prior defaults (0.025/0.03) gave episode_care_seeking_rate=22.4% and
    # could_not_afford_rate=85% - never actually checked against DHS before
    # this calibration. This is the lowest-could_not_afford_rate point among
    # several (income, cost) combos that equally hit the DHS band (a single
    # 1D target underdetermines this 2D space - see
    # experiments/outputs/care_seeking_calibration/care_seeking_calibration_ranked.csv
    # for the full frontier); episode_care_seeking_rate=71.1%,
    # could_not_afford_rate=27.4%.
    cost_of_care: float = Field(
        0.0702,
        description="Cost of seeking medical care, as a fraction of max household wealth. "
                    "Calibrated jointly with daily_income_rate against the Ghana DHS childhood-"
                    "diarrhea care-seeking rate (69.2%, 95% CI 65.6-72.8%, n=621). Changing this "
                    "without daily_income_rate may pull the model away from that validated figure.",
    )
    treatment_success_prob: float = Field(
        0.80, description="Probability that sought medical care successfully cures the illness."
    )
    natural_worsening_prob: float = Field(
        0.35, description="Probability an untreated illness worsens rather than staying the same."
    )
    parent_stress_health_impact: float = Field(
        0.30, description="Health toll on a parent from the stress of a sick child."
    )
    untreated_severity_penalty: float = Field(
        0.20, description="Extra severity applied when an illness worsens untreated."
    )
    cpt_theta: float = Field(
        0.88,
        description="Cumulative Prospect Theory gain-sensitivity exponent. Set to Tversky & "
                    "Kahneman's (1992) own estimated value, not fit to this population.",
    )
    cpt_eta: float = Field(
        0.88,
        description="Cumulative Prospect Theory loss-sensitivity exponent. Same Tversky & Kahneman "
                    "(1992) literature value as cpt_theta.",
    )
    severity_health_impact_factor: float = Field(
        0.05, description="Daily health reduction per unit of illness severity."
    )
    # Base daily recovery when not sick. Must stay roughly in scale with
    # daily_income_rate/daily_cost_of_living: with health_based_income=True,
    # adult income is scaled by the adult's own health, and a rate an order
    # of magnitude below those (as 0.001 was) left adults permanently below
    # the income/cost breakeven, collapsing household wealth to 0 regardless
    # of disease burden. Raised 10x to actually recover within weeks.
    daily_health_recovery_rate: float = Field(
        0.01,
        description="Base daily health recovery when not sick. Must stay in scale with "
                    "daily_income_rate/daily_cost_of_living - too low collapses household wealth "
                    "to 0 regardless of disease burden (a real bug found and fixed; see project "
                    "history). Not a free-standing tuning knob.",
    )
    child_health_weight: float = Field(
        0.5, description="Weight placed on child health (versus parent's own state) in the parent's care-seeking utility."
    )

    # Income and wealth dynamics
    # daily_income_rate calibrated jointly with cost_of_care above (see that
    # field's comment) against the Ghana DHS care-seeking rate.
    daily_income_rate: float = Field(
        0.0462,
        description="Daily household income, as a fraction of max wealth. Calibrated jointly with "
                    "cost_of_care against the Ghana DHS care-seeking rate - see that field's note.",
    )
    daily_cost_of_living: float = Field(
        0.025, description="Daily household cost of living, as a fraction of max wealth."
    )
    health_based_income: bool = Field(
        True, description="Whether adult income scales with the adult's own health (internal model toggle)."
    )
    # Equivalence-scale-style discount: a child costs this fraction of an
    # adult's cost-of-living share (0.3 ~ OECD-modified equivalence scale).
    # Without this, cost-of-living charged a full adult share per head
    # regardless of age, while only adults earn income - so any household
    # with a typical adult:child ratio (median ~0.67 for parent-headed
    # households, see calibration notes) was structurally unable to break
    # even at ANY health level, since breakeven required
    # adult_fraction >= daily_cost_of_living/daily_income_rate = 0.833.
    child_cost_weight: float = Field(
        0.3,
        description="A child's cost-of-living share, as a fraction of an adult's (~0.3, an OECD-"
                    "modified-equivalence-scale-style discount). Without this, typical parent-"
                    "headed households were structurally unable to break even - a real bug found "
                    "and fixed; see project history.",
    )

class SVEIRConfig(BaseModel):
    """Main configuration class for the SVEIR model."""
    model_identifier: str = Field("sveir_model", description="Internal run identifier (not user-configurable).")
    description: str = Field("Configuration for the SVEIR agent-based model.", description="Internal config description (not user-configurable).")
    device: str = Field("cpu", description="Compute device (internal - not user-configurable; this deployment is CPU-only).")
    seed: int = Field(
        23,
        description="Random seed. Reshuffles household placement, family composition, and "
                    "behavioral personas for this run. The underlying map of Akuse - roads, water "
                    "sources, schools - stays the same real-world geography every time; only the "
                    "simulated population living on it changes.",
    )
    number_agents: PositiveInt = Field(5000, description="Total number of agents (population size) to simulate.")
    spatial: bool = Field(True, description="Whether to use the spatial grid environment (internal - always True in the webapp).")
    spatial_creation_args: GridCreationParams = Field(default_factory=GridCreationParams, description="Spatial grid configuration (internal - not user-configurable).")
    step_target: PositiveInt = Field(150, description="Number of simulated days to run.")

    # Pathogen Configuration
    pathogens: List[Union[RotavirusConfig, CampylobacterConfig]] = Field(
        default_factory=lambda: [RotavirusConfig(), CampylobacterConfig()],
        description="Which pathogens are active in this simulation.",
    )

    # Demographic Parameters
    average_household_size: float = Field(
        3.2, description="Mean household size (Poisson-distributed). Empirically sourced from Ghana census/DHS-MICS-style survey data."
    )
    child_probability: float = Field(
        0.145, description="Probability a non-first household member is a child. Empirically sourced, same survey data as household size."
    )

    # Parameters for Agent Personas
    num_agent_personas: int = Field(32, description="Number of distinct behavioral (CPT) personas sampled via Latin Hypercube (internal - not user-configurable).")
    alpha_range: list[float] = Field([0.1, 0.9], description="Sampling range for the CPT wealth/health utility weight (alpha). Plausible, exploratory range - not fit to measured risk preferences.")
    gamma_range: list[float] = Field([0.4, 0.9], description="Sampling range for the CPT probability-distortion parameter (gamma). Plausible, exploratory range - not fit to measured risk preferences.")
    lambda_range: list[float] = Field([1.0, 3.0], description="Sampling range for the CPT loss-aversion parameter (lambda). Plausible, exploratory range - not fit to measured risk preferences.")

    # Illness mechanics
    illness_mechanics: IllnessMechanicsConfig = Field(default_factory=IllnessMechanicsConfig, description="Illness severity/duration mechanics parameters.")

    steering_parameters: SteeringParamsSVEIR = Field(default_factory=SteeringParamsSVEIR, description="Care-seeking, economic, and environmental steering parameters.")

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

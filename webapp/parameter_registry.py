"""
The single source of truth for how SVEIRConfig parameters are presented in
the web UI: which category they belong to, whether they're safe to edit or
shown read-only, and - for every parameter, editable or not - a plain-
language rationale explaining what it is and whether its value is grounded
in literature, calibrated against data, a documented assumption, or purely
structural/internal.

This registry drives two consumers that must never disagree with each
other: the scenario-builder form (grouped by `category`) and the About
page's "how much to trust each number" table (grouped by `evidence_tier`).
Keeping both in one place, keyed by the same dot-path convention
experiments/orchestrator.py already uses (get_param/set_param), is the
mechanism that keeps the About page from drifting out of sync with the
model - see webapp/tests/test_parameter_registry.py for the completeness
check that enforces it.

Editability: every scientific/behavioral parameter is editable, regardless
of evidence_tier - the tier badge tells you how much to trust the value,
it is not a permission gate. The only fields kept out of the UI entirely
are genuine internal plumbing (category="internal": device, grid_id,
model_identifier, ...) that have no other valid value in this deployment.
Range-valued parameters (alpha/gamma/lambda_range) are editable via a
min+max pair (ui_widget="range-pair") rather than a single slider.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Literal

from pydantic import BaseModel

from config import (
    SVEIRConfig, RotavirusConfig, CampylobacterConfig,
    GridCreationParams, IllnessMechanicsConfig, SteeringParamsSVEIR,
)

EvidenceTier = Literal["literature", "calibrated", "assumption", "structural"]


@dataclass(frozen=True)
class ParamMeta:
    path: str                       # dot-path, e.g. "steering_parameters.cost_of_care" or "pathogens[rota].recovery_rate"
    label: str                      # short human-readable name
    category: str                   # UI grouping; "internal" is excluded from every rendered view
    evidence_tier: EvidenceTier
    rationale: str                  # what it is + why it's trusted (or not) - shown in the info box
    editable: bool = False
    unit: str | None = None
    ui_min: float | None = None
    ui_max: float | None = None
    ui_widget: str = "slider"       # "slider" | "number+randomize-button" | "range-pair"
    ui_step: float | None = None    # explicit step for whole-number fields (e.g. agent/day counts);
                                     # None means "continuous" - the template renders step="any" so
                                     # the browser never rejects a dragged value as "off-grid" (a real
                                     # bug: a computed fractional step like 0.0035 combined with
                                     # floating-point drift made some sliders un-submittable)
    is_integer: bool = False        # whole-number config field (e.g. exposure_period, in days) -
                                     # the generated form field is typed int, not float, so a
                                     # fractional slider value is rejected with a clear validation
                                     # error rather than silently corrupting the config


CATEGORY_ORDER = [
    "Population & Demographics",
    "Rotavirus",
    "Campylobacter",
    "Illness Mechanics",
    "Care-Seeking & Behavioral Economics",
    "Household Economics",
    "Environment, Water & Shocks",
]

EVIDENCE_TIER_ORDER: list[EvidenceTier] = ["literature", "calibrated", "assumption", "structural"]

EVIDENCE_TIER_LABELS: dict[EvidenceTier, str] = {
    "literature": "Literature-grounded",
    "calibrated": "Calibrated against data",
    "assumption": "Documented assumption",
    "structural": "Structural / internal",
}


def _p(*metas: ParamMeta) -> list[ParamMeta]:
    return list(metas)


REGISTRY: list[ParamMeta] = [
    # ------------------------------------------------------------------
    # Population & Demographics
    # ------------------------------------------------------------------
    ParamMeta(
        path="seed", label="Random seed", category="Population & Demographics",
        evidence_tier="structural", editable=True, ui_widget="number+randomize-button",
        ui_min=0, ui_max=2_147_483_647,
        rationale="Reshuffles household placement, family composition, and behavioral personas "
                  "for this run. The underlying map of Akuse - roads, water sources, schools - "
                  "stays the same real-world geography every time; only the simulated population "
                  "living on it changes.",
    ),
    ParamMeta(
        path="number_agents", label="Population size", category="Population & Demographics",
        evidence_tier="structural", editable=True, ui_widget="slider", unit="agents",
        ui_min=500, ui_max=10_000, ui_step=100,
        rationale="Total number of simulated agents. Larger populations give smoother, less noisy "
                  "outcomes but take longer to run.",
    ),
    ParamMeta(
        path="step_target", label="Simulation length", category="Population & Demographics",
        evidence_tier="structural", editable=True, ui_widget="slider", unit="days",
        ui_min=30, ui_max=500, ui_step=10,
        rationale="Number of simulated days. The model shows one large initial epidemic wave "
                  "(peaking around day 30) before settling into a lower quasi-equilibrium by "
                  "roughly day 150-200 - short runs may only capture the initial wave.",
    ),
    ParamMeta(
        path="average_household_size", label="Average household size", category="Population & Demographics",
        evidence_tier="literature", editable=True, ui_min=1.5, ui_max=6.0, unit="people",
        rationale="Mean household size, used as the Poisson-distribution parameter for "
                  "generating households. Empirically sourced from Ghana census/DHS-MICS-style "
                  "survey data, not an assumption.",
    ),
    ParamMeta(
        path="child_probability", label="Child probability", category="Population & Demographics",
        evidence_tier="literature", editable=True, ui_min=0.0, ui_max=0.5,
        rationale="Probability a non-first household member is a child. Empirically sourced from "
                  "the same Ghana census/DHS-MICS-style survey data as household size.",
    ),
    ParamMeta(
        # Deliberately kept out of the UI (category="internal") per explicit user request -
        # these still exist and are used exactly as before, just not exposed as a control.
        # editable=False too, not just hidden: no path from the UI to change these anymore.
        path="alpha_range", label="Wealth/health utility weight range (α)", category="internal",
        evidence_tier="assumption",
        rationale="Sampling range for each agent's individual weighting of wealth vs. health in "
                  "their care-seeking decisions. A plausible, exploratory range representing "
                  "behavioral diversity - not fit to measured risk preferences in this population.",
    ),
    ParamMeta(
        path="gamma_range", label="Probability-distortion range (γ)", category="internal",
        evidence_tier="assumption",
        rationale="Sampling range for each agent's Cumulative Prospect Theory probability-"
                  "distortion parameter. Same status as the α range above: plausible and "
                  "exploratory, not independently fit.",
    ),
    ParamMeta(
        path="lambda_range", label="Loss-aversion range (λ)", category="internal",
        evidence_tier="assumption",
        rationale="Sampling range for each agent's loss-aversion parameter - how much more a "
                  "potential loss weighs versus an equivalent gain. Same status as the other "
                  "persona ranges: plausible and exploratory.",
    ),
    ParamMeta(
        path="num_agent_personas", label="Number of behavioral personas", category="internal",
        evidence_tier="structural",
        rationale="How many distinct behavioral archetypes are sampled via Latin Hypercube "
                  "sampling. Internal model-construction detail, not a scenario lever.",
    ),
    ParamMeta(
        path="model_identifier", label="Model identifier", category="internal", evidence_tier="structural",
        rationale="Internal run-naming detail, not a scientific parameter.",
    ),
    ParamMeta(
        path="description", label="Config description", category="internal", evidence_tier="structural",
        rationale="Internal config label, not a scientific parameter.",
    ),
    ParamMeta(
        path="device", label="Compute device", category="internal", evidence_tier="structural",
        rationale="This deployment always runs on CPU.",
    ),
    ParamMeta(
        path="spatial", label="Use spatial environment", category="internal", evidence_tier="structural",
        rationale="Always enabled in the web UI - the model always runs on the real Akuse spatial grid.",
    ),
    ParamMeta(
        path="spatial_creation_args.method", label="Grid creation method", category="internal", evidence_tier="structural",
        rationale="Internal grid-construction detail.",
    ),
    ParamMeta(
        path="spatial_creation_args.grid_id", label="Grid ID", category="internal", evidence_tier="structural",
        rationale="Identifies which pre-built spatial grid to load. Always the one cached grid of "
                  "real, OSM-derived Akuse geography - not user-configurable, and there is "
                  "currently no live randomness in grid generation to vary even if it were.",
    ),
    ParamMeta(
        path="spatial_creation_args.properties", label="Grid properties", category="internal", evidence_tier="structural",
        rationale="Internal grid-construction detail.",
    ),

    # ------------------------------------------------------------------
    # Rotavirus
    # ------------------------------------------------------------------
    ParamMeta(
        path="pathogens[rota].name", label="Pathogen ID", category="internal", evidence_tier="structural",
        rationale="Internal identifier.",
    ),
    ParamMeta(
        path="pathogens[rota].initial_exposed_proportion", label="Initial exposed fraction",
        category="Rotavirus", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.05,
        rationale="Fraction of agents seeded as exposed to rotavirus at the very start of the "
                  "simulation - the initial spark for the outbreak. A modeling convenience, not "
                  "an independently measured quantity.",
    ),
    ParamMeta(
        path="pathogens[rota].recovery_rate", label="Recovery rate", category="Rotavirus",
        evidence_tier="literature", editable=True, ui_min=0.14, ui_max=0.33,
        unit="probability/day",
        rationale="Daily recovery probability (default 0.3005, a ~3.3 day expected duration). "
                  "Constrained to the ~3-7 day rotavirus illness-duration literature range; "
                  "deliberately excluded from calibration search for this reason, so its default "
                  "value reflects clinical literature, not a model fit.",
    ),
    ParamMeta(
        path="pathogens[rota].exposure_period", label="Latent period", category="Rotavirus",
        evidence_tier="literature", editable=True, is_integer=True, ui_min=1, ui_max=7, ui_step=1, unit="days",
        rationale="Days between exposure and becoming infectious. Anchored to cited rotavirus "
                  "incubation-period literature in the paper's Methods.",
    ),
    ParamMeta(
        path="pathogens[rota].infection_prob_mean", label="Infection probability (mean)",
        category="Rotavirus", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.025,
        rationale="Mean per-exposure infection probability. LHS-calibrated to hit literature "
                  "target ranges for episodes/child-year and peak prevalence - not independently "
                  "measured for Akuse. Changing this moves the model away from that calibrated fit.",
    ),
    ParamMeta(
        path="pathogens[rota].infection_prob_std", label="Infection probability (spread)",
        category="Rotavirus", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.001,
        rationale="Spread of the per-agent infection-probability draw around the mean. Part of "
                  "the same LHS-calibrated fit as the mean infection probability.",
    ),
    ParamMeta(
        path="pathogens[rota].vaccination_rate", label="Vaccination rate", category="Rotavirus",
        evidence_tier="assumption", editable=True, ui_min=0.0, ui_max=0.05,
        rationale="Daily probability an unvaccinated agent receives the rotavirus vaccine - the "
                  "main intervention lever for exploring vaccination-campaign scenarios.",
    ),
    ParamMeta(
        path="pathogens[rota].vaccine_efficacy", label="Vaccine efficacy", category="Rotavirus",
        evidence_tier="assumption", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Reduction in severity/susceptibility conferred by vaccination. Rate and "
                  "efficacy interact in a non-obvious way in this model: efficacy gates whether a "
                  "faster rollout matters at all.",
    ),
    ParamMeta(
        path="illness_mechanics.base_severity_rota", label="Base rotavirus severity",
        category="Rotavirus", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Base illness severity (0-1 scale) before age/immunity adjustments. Anchored to "
                  "cited literature in the paper's Methods.",
    ),

    # ------------------------------------------------------------------
    # Campylobacter
    # ------------------------------------------------------------------
    ParamMeta(
        path="pathogens[campy].name", label="Pathogen ID", category="internal", evidence_tier="structural",
        rationale="Internal identifier.",
    ),
    ParamMeta(
        path="pathogens[campy].initial_exposed_proportion", label="Initial exposed fraction",
        category="Campylobacter", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.05,
        rationale="Fraction of agents seeded as exposed to campylobacter at the very start of the "
                  "simulation. A modeling convenience, not an independently measured quantity.",
    ),
    ParamMeta(
        path="pathogens[campy].recovery_rate", label="Recovery rate", category="Campylobacter",
        evidence_tier="literature", editable=True, ui_min=0.14, ui_max=0.18,
        unit="probability/day",
        rationale="Daily recovery probability (default 0.1466, a ~6.8 day expected duration), "
                  "literature-grounded illness-duration constraint; excluded from calibration "
                  "search for the same reason as rotavirus's.",
    ),
    ParamMeta(
        path="pathogens[campy].exposure_period", label="Latent period", category="Campylobacter",
        evidence_tier="literature", editable=True, is_integer=True, ui_min=1, ui_max=10, ui_step=1, unit="days",
        rationale="Days between exposure and becoming infectious. Anchored to cited campylobacter "
                  "incubation-period literature in the paper's Methods.",
    ),
    ParamMeta(
        path="pathogens[campy].beta_poisson_alpha", label="Dose-response shape (α)",
        category="Campylobacter", evidence_tier="literature", editable=True, ui_min=0.01, ui_max=0.1,
        rationale="Beta-Poisson dose-response shape parameter for the zoonotic transmission route "
                  "- a standard microbial dose-response model form from the QMRA literature.",
    ),
    ParamMeta(
        path="pathogens[campy].beta_poisson_beta", label="Dose-response scale (β)",
        category="Campylobacter", evidence_tier="literature", editable=True, ui_min=0.005, ui_max=0.06,
        rationale="Beta-Poisson dose-response scale parameter, same literature basis as the shape parameter.",
    ),
    ParamMeta(
        path="pathogens[campy].human_animal_interaction_rate", label="Animal-contact exposure rate",
        category="Campylobacter", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.02,
        rationale="Daily probability of zoonotic-route exposure for an animal-owning household. "
                  "LHS-calibrated to hit literature target ranges - not measured directly for Akuse.",
    ),
    ParamMeta(
        path="pathogens[campy].fecal_oral_prob", label="Fecal-oral transmission probability",
        category="Campylobacter", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.03,
        rationale="Per-contact fecal-oral transmission probability within an infected household. "
                  "LHS-calibrated, same fit as the other campylobacter transmission parameters.",
    ),
    ParamMeta(
        path="pathogens[campy].food_borne_prob", label="Food-borne background risk",
        category="Campylobacter", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.005,
        rationale="Daily background infection probability via food-borne exposure, independent of "
                  "household animal contact. LHS-calibrated, same fit as the other campylobacter "
                  "transmission parameters.",
    ),
    ParamMeta(
        path="pathogens[campy].poultry_ownership_prob", label="Poultry ownership rate",
        category="Campylobacter", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Probability a household owns poultry. Sourced from a compiled Ghana DHS/MICS-"
                  "style rural survey (no cluster falls inside Akuse itself; a rural-southern-"
                  "Ghana average) - genuinely empirical, not an assumption.",
    ),
    ParamMeta(
        path="pathogens[campy].ruminant_ownership_prob", label="Ruminant ownership rate",
        category="Campylobacter", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Probability a household owns ruminants (goats/sheep/cattle). Same DHS/MICS-"
                  "style survey source as poultry ownership.",
    ),
    ParamMeta(
        path="pathogens[campy].poultry_weight", label="Poultry risk weight", category="Campylobacter",
        evidence_tier="assumption", editable=True, ui_min=0.0, ui_max=2.0,
        rationale="Relative zoonotic-risk weight for poultry. Qualitatively literature-motivated "
                  "(poultry dominates C. jejuni source-attribution studies) but not quantitatively "
                  "fit - deliberately uncertain, a good candidate for sensitivity exploration.",
    ),
    ParamMeta(
        path="pathogens[campy].ruminant_weight", label="Ruminant risk weight", category="Campylobacter",
        evidence_tier="assumption", editable=True, ui_min=0.0, ui_max=2.0,
        rationale="Relative zoonotic-risk weight for ruminants, versus poultry's baseline of 1.0. "
                  "Same qualitative literature motivation as the poultry weight, not quantitatively fit.",
    ),
    ParamMeta(
        path="pathogens[campy].poultry_roam_sigma", label="Poultry roam radius", category="Campylobacter",
        evidence_tier="assumption", editable=True, ui_min=0.1, ui_max=5.0, unit="grid cells",
        rationale="Gaussian radius diffusing poultry-related risk around an owning household - "
                  "backyard poultry stay close to the yard. A qualitative, not quantitatively fit, assumption.",
    ),
    ParamMeta(
        path="pathogens[campy].ruminant_roam_sigma", label="Ruminant roam radius", category="Campylobacter",
        evidence_tier="assumption", editable=True, ui_min=0.1, ui_max=5.0, unit="grid cells",
        rationale="Gaussian radius diffusing ruminant-related risk - wider than poultry, "
                  "reflecting grazing/tethered range. A qualitative, not quantitatively fit, assumption.",
    ),
    ParamMeta(
        path="illness_mechanics.base_severity_campy", label="Base campylobacter severity",
        category="Campylobacter", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Base illness severity (0-1 scale) before age/immunity adjustments. Anchored to "
                  "cited literature in the paper's Methods.",
    ),

    # ------------------------------------------------------------------
    # Illness Mechanics
    # ------------------------------------------------------------------
    ParamMeta(
        path="illness_mechanics.age_max_multiplier", label="Age severity multiplier (max)",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=3.0,
        rationale="Extra severity multiplier at birth, decaying toward 1.0 as a child ages. "
                  "Anchored to cited literature on age-related severity in the paper's Methods.",
    ),
    ParamMeta(
        path="illness_mechanics.age_decay_rate", label="Age severity decay rate",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=0.3,
        rationale="Rate at which the age-related severity multiplier decays toward 1.0. Same "
                  "literature basis as the age severity multiplier.",
    ),
    ParamMeta(
        path="illness_mechanics.immunity_factor_vaccine", label="Vaccine immunity factor",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Severity reduction factor conferred by vaccination. Anchored to cited literature "
                  "in the paper's Methods.",
    ),
    ParamMeta(
        path="illness_mechanics.severity_reduction_per_infection", label="Immunity gain per infection",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Severity reduction per prior infection. This immunity is deliberately floored, "
                  "never reaching zero - the mechanistic reason the model sustains transmission "
                  "indefinitely in a closed population with no births.",
    ),
    ParamMeta(
        path="illness_mechanics.duration_min_days", label="Illness duration (minimum)",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=0.5, ui_max=5.0, unit="days",
        rationale="Expected illness duration at severity = 0. Anchored to cited illness-duration "
                  "literature, consistent with each pathogen's recovery_rate.",
    ),
    ParamMeta(
        path="illness_mechanics.duration_max_days", label="Illness duration (maximum)",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=5.0, ui_max=25.0, unit="days",
        rationale="Expected illness duration at severity = 1. Same literature basis as the minimum duration.",
    ),
    ParamMeta(
        path="illness_mechanics.duration_noise_std", label="Illness duration (spread)",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=4.0, unit="days",
        rationale="Stochastic spread around the expected illness duration. Same literature basis "
                  "as the min/max duration.",
    ),

    # ------------------------------------------------------------------
    # Care-Seeking & Behavioral Economics
    # ------------------------------------------------------------------
    ParamMeta(
        path="steering_parameters.cost_of_care", label="Cost of seeking care",
        category="Care-Seeking & Behavioral Economics", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.3, unit="fraction of wealth",
        rationale="Cost of seeking medical care, as a fraction of max household wealth. Calibrated "
                  "jointly with the household income rate against the Ghana DHS childhood-diarrhea "
                  "care-seeking rate (69.2%, 95% CI 65.6-72.8%, n=621). Changing this without also "
                  "adjusting income may pull the model away from that validated figure.",
    ),
    ParamMeta(
        path="steering_parameters.treatment_success_prob", label="Treatment success probability",
        category="Care-Seeking & Behavioral Economics", evidence_tier="assumption", editable=True,
        ui_min=0.3, ui_max=1.0,
        rationale="Probability that sought medical care successfully cures the illness. A plausible "
                  "modeling default, not independently measured for Akuse.",
    ),
    ParamMeta(
        path="steering_parameters.natural_worsening_prob", label="Untreated worsening probability",
        category="Care-Seeking & Behavioral Economics", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=1.0,
        rationale="Probability an untreated illness worsens rather than staying the same - part of "
                  "what makes seeking care a meaningful trade-off. A plausible modeling default, "
                  "not independently measured.",
    ),
    ParamMeta(
        path="steering_parameters.parent_stress_health_impact", label="Parental stress impact",
        category="Care-Seeking & Behavioral Economics", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=1.0,
        rationale="Health toll on a parent from the stress of having a sick child. A plausible "
                  "modeling default representing a real caregiving-burden effect, not independently measured.",
    ),
    ParamMeta(
        path="steering_parameters.untreated_severity_penalty", label="Untreated worsening penalty",
        category="Care-Seeking & Behavioral Economics", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=1.0,
        rationale="Extra severity applied when an untreated illness worsens. A plausible modeling "
                  "default, not independently measured.",
    ),
    ParamMeta(
        path="steering_parameters.cpt_theta", label="CPT gain-sensitivity exponent (θ)",
        category="Care-Seeking & Behavioral Economics", evidence_tier="literature", editable=True,
        ui_min=0.3, ui_max=1.0,
        rationale="Cumulative Prospect Theory gain-sensitivity exponent, shared across every "
                  "agent. Set to Tversky & Kahneman's (1992) own estimated value, not fit to this "
                  "population.",
    ),
    ParamMeta(
        path="steering_parameters.cpt_eta", label="CPT loss-sensitivity exponent (η)",
        category="Care-Seeking & Behavioral Economics", evidence_tier="literature", editable=True,
        ui_min=0.3, ui_max=1.0,
        rationale="Cumulative Prospect Theory loss-sensitivity exponent, shared across every "
                  "agent. Same Tversky & Kahneman (1992) literature value as the gain-sensitivity exponent.",
    ),
    ParamMeta(
        path="steering_parameters.severity_health_impact_factor", label="Severity-to-health scaling",
        category="Care-Seeking & Behavioral Economics", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.2,
        rationale="Daily health reduction per unit of illness severity - the core conversion from "
                  "'how sick' to 'how much health lost per day.' A plausible modeling default.",
    ),
    ParamMeta(
        path="steering_parameters.daily_health_recovery_rate", label="Baseline health recovery rate",
        category="Care-Seeking & Behavioral Economics", evidence_tier="structural", editable=True,
        ui_min=0.005, ui_max=0.05,
        rationale="Base daily health recovery when not sick. Worth caution: must stay roughly in "
                  "scale with the household income and cost-of-living rates below - a documented "
                  "past bug found this rate an order of magnitude too low, which left adults "
                  "permanently below the income/cost breakeven and collapsed household wealth to "
                  "zero regardless of disease burden. ui_min is set well above that bug's exact "
                  "value (0.001) so the slider can't silently reintroduce it.",
    ),
    ParamMeta(
        path="steering_parameters.child_health_weight", label="Weight on child health",
        category="Care-Seeking & Behavioral Economics", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=1.0,
        rationale="Weight placed on the child's health (versus the parent's own state) in the "
                  "parent's care-seeking utility calculation. A plausible modeling default.",
    ),

    # ------------------------------------------------------------------
    # Household Economics
    # ------------------------------------------------------------------
    ParamMeta(
        path="steering_parameters.daily_income_rate", label="Daily household income rate",
        category="Household Economics", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.15, unit="fraction of wealth",
        rationale="Daily household income, as a fraction of max wealth. Calibrated jointly with "
                  "the cost of care against the Ghana DHS care-seeking rate - see that field's note. "
                  "Changing this alone may pull the model away from the validated DHS-matched fit.",
    ),
    ParamMeta(
        path="steering_parameters.daily_cost_of_living", label="Daily cost of living",
        category="Household Economics", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.1, unit="fraction of wealth",
        rationale="Daily household cost of living, as a fraction of max wealth. A plausible "
                  "modeling default, not independently calibrated (unlike income and cost of care).",
    ),
    ParamMeta(
        path="steering_parameters.child_cost_weight", label="Child cost-of-living weight",
        category="Household Economics", evidence_tier="structural", editable=True,
        ui_min=0.0, ui_max=0.85,
        rationale="A child's cost-of-living share, as a fraction of an adult's (~0.3, an OECD-"
                  "modified-equivalence-scale-style discount). Worth caution: a documented past bug "
                  "found that without this discount, typical parent-headed households were "
                  "structurally unable to break even at any health level. ui_max is set below 1.0 "
                  "(the bug's exact 'no discount' value) so the slider can't silently reintroduce it.",
    ),
    ParamMeta(
        path="steering_parameters.health_based_income", label="Health-based income scaling",
        category="internal", evidence_tier="structural",
        rationale="Whether adult income scales with the adult's own health. A structural model "
                  "toggle, not a numeric tuning parameter.",
    ),

    # ------------------------------------------------------------------
    # Environment, Water & Shocks
    # ------------------------------------------------------------------
    ParamMeta(
        path="steering_parameters.shock_daily_prob", label="Water-contamination shock frequency",
        category="Environment, Water & Shocks", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.2, unit="probability/day",
        rationale="Daily probability of a water-contamination shock event. Stated in the paper as "
                  "an assumption pending calibration against real water-quality/infrastructure-"
                  "failure data - explored via sensitivity analysis, not a validated real-world value.",
    ),
    ParamMeta(
        path="steering_parameters.water_recovery_prob", label="Water recovery probability",
        category="Environment, Water & Shocks", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=1.0, unit="probability/day",
        rationale="Daily probability a contaminated water source naturally reverts to clean. "
                  "Stated in the paper as an assumption pending real-world calibration, alongside "
                  "the shock frequency.",
    ),
    ParamMeta(
        path="steering_parameters.human_to_water_infection_prob", label="Human-to-water contamination rate",
        category="Environment, Water & Shocks", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.001,
        rationale="Daily probability an infectious agent contaminates its local water source. A "
                  "plausible modeling default completing the water-transmission feedback loop, not "
                  "independently calibrated.",
    ),
    ParamMeta(
        path="steering_parameters.water_to_human_infection_prob", label="Water-to-human infection rate",
        category="Environment, Water & Shocks", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.03,
        rationale="Daily probability a susceptible agent is infected via contaminated water. "
                  "LHS-calibrated tuning knob (alongside rotavirus's other transmission "
                  "parameters), not independently measured for Akuse.",
    ),
    ParamMeta(
        path="steering_parameters.social_interaction_radius", label="Social interaction radius",
        category="Environment, Water & Shocks", evidence_tier="assumption", editable=True,
        ui_min=1.0, ui_max=15.0, unit="grid cells",
        rationale="Radius within which agents can socially interact and transmit human-to-human. "
                  "A plausible modeling default, not independently measured.",
    ),
    ParamMeta(
        path="steering_parameters.prior_infection_immunity_factor", label="Cross-episode immunity factor",
        category="Environment, Water & Shocks", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.5,
        rationale="Additional immunity factor applied per prior infection, shared across "
                  "pathogens. Related to why the model sustains endemic transmission indefinitely "
                  "in a closed population - immunity here is always partial, never complete.",
    ),
]

REGISTRY_BY_PATH: dict[str, ParamMeta] = {m.path: m for m in REGISTRY}


def iter_config_paths() -> list[str]:
    """Recursively enumerates every real field path on SVEIRConfig, using the
    same dot-path / pathogens[name].attr convention as the registry above
    and experiments/orchestrator.py's get_param/set_param. Used by the
    completeness-check test to make sure every field has a registry entry.
    """
    paths: list[str] = []

    def walk(model_cls: type[BaseModel], prefix: str = "") -> None:
        for name, model_field in model_cls.model_fields.items():
            path = f"{prefix}{name}"
            if name == "pathogens":
                for label, pathogen_cls in (("rota", RotavirusConfig), ("campy", CampylobacterConfig)):
                    for pname in pathogen_cls.model_fields:
                        paths.append(f"pathogens[{label}].{pname}")
                continue
            annotation = model_field.annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                walk(annotation, prefix=f"{path}.")
            else:
                paths.append(path)

    walk(SVEIRConfig)
    return paths


def by_category() -> dict[str, list[ParamMeta]]:
    """Registry entries grouped by category, in CATEGORY_ORDER, excluding 'internal'."""
    grouped: dict[str, list[ParamMeta]] = {c: [] for c in CATEGORY_ORDER}
    for meta in REGISTRY:
        if meta.category in grouped:
            grouped[meta.category].append(meta)
    return grouped


def by_evidence_tier() -> dict[str, list[ParamMeta]]:
    """Registry entries grouped by evidence tier, in EVIDENCE_TIER_ORDER, excluding internal-category fields."""
    grouped: dict[str, list[ParamMeta]] = {t: [] for t in EVIDENCE_TIER_ORDER}
    for meta in REGISTRY:
        if meta.category != "internal":
            grouped[meta.evidence_tier].append(meta)
    return grouped


def editable_fields() -> list[ParamMeta]:
    return [m for m in REGISTRY if m.editable]

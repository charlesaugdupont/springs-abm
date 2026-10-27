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
    PATHOGEN_CONFIG_TYPES,
    SVEIRConfig, RotavirusConfig, CampylobacterConfig, CryptosporidiumConfig,
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
    "Cryptosporidium",
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
        path="demography.ageing", label="Ageing and under-5 turnover", category="internal",
        evidence_tier="structural",
        rationale="Children age daily; one reaching five years leaves the modelled population and "
                  "its slot is reused for a newborn. Without this the population is closed and "
                  "incidence declines indefinitely as immunity accumulates.",
    ),
    ParamMeta(
        path="demography.child_max_age_months", label="Upper age of modelled children",
        category="internal", evidence_tier="structural",
        rationale="The modelled children are the under-5s (60 months).",
    ),
    ParamMeta(
        path="demography.school_entry_age_months", label="School entry age",
        category="Population & Demographics", evidence_tier="literature", editable=True,
        ui_min=0.0, ui_max=60.0, unit="months",
        rationale="Youngest age at which a child attends nursery/kindergarten. Ghana has two years "
                  "of compulsory kindergarten from age 4 and common nursery attendance from age 3. "
                  "Younger children stay home rather than spending ~30% of days at school.",
    ),
    ParamMeta(
        path="demography.school_attendance_prob", label="School attendance (age-eligible)",
        category="Population & Demographics", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=1.0,
        rationale="Probability an age-eligible child attends. 0.7 is a placeholder pending the "
                  "Ghana MICS early-childhood-education attendance indicator (36-59 months).",
    ),
    ParamMeta(
        path="demography.initial_prior_infections", label="Initial infection history",
        category="internal", evidence_tier="literature",
        rationale="'maled': children start with prior infection counts drawn around MAL-ED's "
                  "cumulative age-specific incidence, so a 4-year-old is not naive at t=0. Shortens "
                  "the burn-in; not a target.",
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
        ui_min=0.0, ui_max=0.06,
        rationale="Mean per-exposure person-to-person infection probability. History-matched "
                  "jointly with the other transmission parameters against MAL-ED incidence, "
                  "prior-infection rate ratios and route fractions (experiments/calibration/"
                  "history_matching.py) - not measured for Akuse. Trades off against the water "
                  "dose: change one and the other must move to keep the fit.",
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
                  "History-matched jointly with the other transmission parameters (MAL-ED incidence "
                  "and a 50-80% zoonotic share) - not measured directly for Akuse.",
    ),
    ParamMeta(
        path="pathogens[campy].fecal_oral_prob", label="Fecal-oral transmission probability",
        category="Campylobacter", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.03,
        rationale="Per-contact fecal-oral transmission probability within an infected household. "
                  "History-matched with the other transmission parameters; weakly identified, as "
                  "this route carries little campylobacter transmission.",
    ),
    ParamMeta(
        path="pathogens[campy].food_borne_prob", label="Food-borne background risk",
        category="Campylobacter", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.005,
        rationale="Daily background infection probability via food-borne exposure, independent of "
                  "household animal contact. History-matched with the other transmission "
                  "parameters.",
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
    # Shared per-pathogen illness coupling (rotavirus / campylobacter)
    # ------------------------------------------------------------------
    # These fields live on the PathogenConfig base class, so every pathogen
    # has them. symptomatic_fraction is now a MAL-ED literature value for all
    # three pathogens (abm/evidence/maled.py) and is editable like crypto's. The
    # age profile, duration_scale and immunity override stay internal: the
    # profile is a list (no slider fits it), and duration_scale/immunity_decay
    # are exact no-ops for these two pathogens at their defaults.
    ParamMeta(
        path="pathogens[rota].symptomatic_fraction", label="Symptomatic fraction", category="Rotavirus",
        evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Fraction of infections in children 0-24 months that cause diarrhoea attributable "
                  "to rotavirus: 472 of 734 infections (0.64) in the MAL-ED birth cohort (Amin et "
                  "al. 2026, unvaccinated sites). Subclinical infections still shed and transmit.",
    ),
    ParamMeta(
        path="pathogens[rota].symptomatic_fraction_age_profile", label="Symptomatic fraction age profile",
        category="internal", evidence_tier="literature",
        rationale="MAL-ED 6-month-band values (0.58, 0.72, 0.67, 0.50 to 24 months) as multipliers "
                  "on the symptomatic fraction; held flat beyond 24 months.",
    ),
    ParamMeta(
        path="pathogens[rota].treatment_clears_infection_prob", label="Treatment clears infection",
        category="internal", evidence_tier="literature",
        rationale="0: there is no antiviral for rotavirus, and ORS/zinc do not shorten shedding, so "
                  "care-seeking ends symptoms but not the infection.",
    ),
    ParamMeta(
        path="pathogens[rota].duration_scale", label="Illness-duration scale", category="internal",
        evidence_tier="structural",
        rationale="Per-pathogen multiplier on the shared illness-duration window. 1.0: MAL-ED found "
                  "rotavirus diarrhoea durations (median 2-3 days) fit the shared window directly.",
    ),
    ParamMeta(
        path="pathogens[rota].immunity_decay_per_infection", label="Immunity decay override",
        category="internal", evidence_tier="literature",
        rationale="Per-prior-infection decay of susceptibility, exp(-0.45 N): 74% protection after "
                  "three infections, as measured by Velazquez et al. 1996 (Mexico) and MAL-ED "
                  "(Mohan et al. 2017). Internal for the same reason as Cryptosporidium's override.",
    ),
    ParamMeta(
        path="pathogens[campy].symptomatic_fraction", label="Symptomatic fraction", category="Campylobacter",
        evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Fraction of infections in children 0-24 months that cause diarrhoea attributable "
                  "to Campylobacter: 502 of 2938 infections (0.17) in the MAL-ED birth cohort (Amin "
                  "et al. 2026). Most childhood Campylobacter infection is subclinical.",
    ),
    ParamMeta(
        path="pathogens[campy].symptomatic_fraction_age_profile", label="Symptomatic fraction age profile",
        category="internal", evidence_tier="literature",
        rationale="MAL-ED 6-month-band values (0.22, 0.20, 0.15, 0.11 to 24 months) as multipliers "
                  "on the symptomatic fraction; held flat beyond 24 months.",
    ),
    ParamMeta(
        path="pathogens[campy].treatment_clears_infection_prob", label="Treatment clears infection",
        category="internal", evidence_tier="assumption",
        rationale="1.0 keeps the original model's behaviour: every successful visit is assumed to "
                  "include an effective antibiotic that ends the infection. An upper bound pending "
                  "local prescribing data.",
    ),
    ParamMeta(
        path="pathogens[campy].duration_scale", label="Illness-duration scale", category="internal",
        evidence_tier="structural",
        rationale="Per-pathogen multiplier on the shared illness-duration window. 1.0: MAL-ED found "
                  "campylobacter diarrhoea durations (median 2-3 days) fit the shared window.",
    ),
    ParamMeta(
        path="pathogens[campy].immunity_decay_per_infection", label="Immunity decay override",
        category="internal", evidence_tier="structural",
        rationale="Optional per-pathogen override of the shared acquired-immunity decay. Unset for "
                  "campylobacter, which uses the shared steering parameter.",
    ),

    ParamMeta(
        path="pathogens[rota].immunity_floor", label="Immunity plateau / adult susceptibility",
        category="Rotavirus", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Lowest relative susceptibility repeated infection can produce (0.26 = 74% protection, the most measured after three infections; Velazquez et al. 1996, Mohan et al. 2017), and the susceptibility of every adult.",
    ),
    ParamMeta(
        path="pathogens[rota].infant_susceptibility_at_birth", label="Newborn relative susceptibility",
        category="Rotavirus", evidence_tier="calibrated", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Infection risk of a newborn relative to an older child - maternal antibody, breastfeeding and limited mobility. 0.44: MAL-ED's 0-3 month rotavirus incidence is 0.55x its 6-9 month peak.",
    ),
    ParamMeta(
        path="pathogens[rota].infant_susceptibility_ramp_months", label="Infant protection ramp",
        category="Rotavirus", evidence_tier="calibrated", editable=True, ui_min=0.0, ui_max=24.0,
        unit="months",
        rationale="Age by which an infant's susceptibility has risen linearly to that of an older "
                  "child. First-pass value read off the MAL-ED age-incidence curve.",
    ),
    ParamMeta(
        path="pathogens[campy].immunity_floor", label="Immunity plateau / adult susceptibility",
        category="Campylobacter", evidence_tier="assumption", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Lowest relative susceptibility repeated infection can produce, and the susceptibility of every adult. An assumption (0.30), set comparable to the other two pathogens: no direct estimate exists.",
    ),
    ParamMeta(
        path="pathogens[campy].infant_susceptibility_at_birth", label="Newborn relative susceptibility",
        category="Campylobacter", evidence_tier="calibrated", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Infection risk of a newborn relative to an older child. 0.18: MAL-ED's 0-3 month Campylobacter incidence is 0.30x its 9-12 month peak.",
    ),
    ParamMeta(
        path="pathogens[campy].infant_susceptibility_ramp_months", label="Infant protection ramp",
        category="Campylobacter", evidence_tier="calibrated", editable=True, ui_min=0.0, ui_max=24.0,
        unit="months",
        rationale="Age by which an infant's susceptibility has risen linearly to that of an older "
                  "child. First-pass value read off the MAL-ED age-incidence curve.",
    ),
    # ------------------------------------------------------------------
    # Cryptosporidium
    # ------------------------------------------------------------------
    ParamMeta(
        path="pathogens[crypto].name", label="Pathogen ID", category="internal",
        evidence_tier="structural",
        rationale="Internal pathogen identifier, not a scientific parameter.",
    ),
    ParamMeta(
        path="pathogens[crypto].initial_exposed_proportion", label="Initial exposed fraction",
        category="Cryptosporidium", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.2,
        rationale="Fraction of agents seeded as exposed to Cryptosporidium at the very start of "
                  "the run. A seeding choice, not an epidemiological estimate - the model runs to "
                  "its own quasi-equilibrium regardless.",
    ),
    ParamMeta(
        path="pathogens[crypto].exposure_period", label="Latent period", category="Cryptosporidium",
        evidence_tier="literature", editable=True, ui_min=1, ui_max=14, unit="days",
        ui_step=1, is_integer=True,
        rationale="Days between infection and becoming infectious. Cryptosporidiosis has an "
                  "incubation period averaging about 7 days (range 2-10), notably longer than "
                  "rotavirus or campylobacter - which delays and flattens its epidemic response.",
    ),
    ParamMeta(
        path="pathogens[crypto].recovery_rate", label="Recovery rate", category="Cryptosporidium",
        evidence_tier="literature", editable=True, ui_min=0.02, ui_max=0.2,
        rationale="Daily probability of clearing infection. The default 0.0476 is a ~21-day mean "
                  "shedding period: Korpe et al. 2022 measured 16-19 day means across infants, "
                  "older children and adults, and an Ethiopian case series a 31-day median from "
                  "diarrhoea onset. Shedding lasts far longer than for the other two pathogens, "
                  "and much of it is symptom-free.",
    ),
    ParamMeta(
        path="pathogens[crypto].symptomatic_fraction", label="Symptomatic fraction",
        category="Cryptosporidium", evidence_tier="literature", editable=True,
        ui_min=0.0, ui_max=1.0,
        rationale="Fraction of childhood infections that cause diarrhoea attributable to "
                  "Cryptosporidium: 245 of 1364 infections (0.18) in children 0-24 months in the "
                  "MAL-ED birth cohort (Amin et al. 2026). The large subclinical majority still "
                  "shed and still infect others - the 'silent reservoir' that sustains "
                  "transmission. Lowering this reduces measured disease without reducing "
                  "transmission.",
    ),
    ParamMeta(
        path="pathogens[crypto].symptomatic_fraction_age_profile", label="Symptomatic fraction age profile",
        category="internal", evidence_tier="literature",
        rationale="MAL-ED 6-month-band values (0.22, 0.20, 0.17, 0.17 to 24 months) as multipliers "
                  "on the symptomatic fraction; held flat beyond 24 months.",
    ),
    ParamMeta(
        path="pathogens[crypto].treatment_clears_infection_prob", label="Treatment clears infection",
        category="internal", evidence_tier="literature",
        rationale="0: standard care (ORS + zinc) does not shorten oocyst shedding, and nitazoxanide "
                  "is not part of IMCI diarrhoea management. Care-seeking ends symptoms, not the "
                  "infection.",
    ),
    ParamMeta(
        path="pathogens[crypto].duration_scale", label="Illness-duration scale",
        category="Cryptosporidium", evidence_tier="literature", editable=True,
        ui_min=0.5, ui_max=3.0,
        rationale="Stretches the shared illness-duration window for this pathogen. 1.0: MAL-ED found "
                  "Cryptosporidium-attributable diarrhoea in children under two lasted no longer "
                  "than rotavirus or campylobacter diarrhoea (median 2-3 days).",
    ),
    ParamMeta(
        path="pathogens[crypto].immunity_decay_per_infection", label="Immunity decay override",
        category="internal", evidence_tier="literature",
        rationale="Per-prior-infection immunity decay, overriding the shared value. Set to 0.29 "
                  "from Kattula et al. 2016, who found only 69% protection against infection even "
                  "after four prior infections. Kept out of the UI because it is one of two "
                  "controls on the same quantity and exposing both would let them silently "
                  "contradict each other.",
    ),
    ParamMeta(
        path="pathogens[crypto].immunity_floor", label="Immunity plateau / adult susceptibility",
        category="Cryptosporidium", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Lowest relative susceptibility repeated infection can produce (0.31 = 69% protection after four infections; Kattula et al. 2016), and the susceptibility of every adult.",
    ),
    ParamMeta(
        path="pathogens[crypto].infant_susceptibility_at_birth", label="Newborn relative susceptibility",
        category="Cryptosporidium", evidence_tier="calibrated", editable=True, ui_min=0.0, ui_max=1.0,
        rationale="Infection risk of a newborn relative to an older child. 0: MAL-ED's 0-3 month Cryptosporidium incidence is only 0.08x its 15-18 month peak - the strongest early-life protection of the three.",
    ),
    ParamMeta(
        path="pathogens[crypto].infant_susceptibility_ramp_months", label="Infant protection ramp",
        category="Cryptosporidium", evidence_tier="calibrated", editable=True, ui_min=0.0, ui_max=24.0,
        unit="months",
        rationale="Age by which an infant's susceptibility has risen linearly to that of an older "
                  "child. First-pass value read off the MAL-ED age-incidence curve.",
    ),
    ParamMeta(
        path="pathogens[crypto].household_transmission_prob", label="Household transmission probability",
        category="Cryptosporidium", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.08,
        rationale="Daily chance a susceptible household member is infected when someone in their "
                  "household is infectious. A main route for this pathogen. Anchored on measured "
                  "household secondary attack rates (Korpe et al. 2019: 35.8% urban, 7.8% rural "
                  "over 8 weeks) and history-matched against MAL-ED incidence and a 30-60% "
                  "household share of infections.",
    ),
    ParamMeta(
        path="pathogens[crypto].community_infection_prob", label="Community transmission probability",
        category="Cryptosporidium", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.005,
        rationale="Per-contact chance of infection from each co-located infectious person outside "
                  "the household - schools, worship, water points, social visits. Carries the "
                  "neighbouring-children transmission signal found in rural sub-Saharan Africa "
                  "(Krumkamp et al. 2021). History-matched with the other transmission parameters.",
    ),
    ParamMeta(
        path="pathogens[crypto].dose_response_r", label="Dose-response rate (r)",
        category="Cryptosporidium", evidence_tier="literature", editable=True,
        ui_min=0.005, ui_max=0.07,
        rationale="Exponential dose-response parameter: a larger r means fewer oocysts are needed "
                  "to infect. The range spans human-challenge infectious doses, which vary widely "
                  "by isolate (ID50 from 9 to 1042 oocysts; 10-83 for the C. hominis strain most "
                  "relevant to Ghana). The default 0.018 corresponds to an ID50 of about 39.",
    ),
    ParamMeta(
        path="pathogens[crypto].oocyst_decay_rate", label="Oocyst die-off rate",
        category="Cryptosporidium", evidence_tier="literature", editable=True,
        ui_min=0.02, ui_max=0.5,
        rationale="Daily die-off of oocysts in a water source. The default 0.15 means a 10-fold "
                  "decline takes about 15 days, from measurements at tropical water temperatures "
                  "(King et al. 2005; Li et al. 2009). Oocysts persist far longer than rotavirus "
                  "in water, which is why contaminated sources stay risky for weeks. The often-"
                  "quoted 'survives for months' figure applies to cold water, not to Akuse.",
    ),
    ParamMeta(
        path="pathogens[crypto].oocyst_shedding_rate", label="Oocyst shedding rate",
        category="internal", evidence_tier="structural",
        rationale="Load added to a water source per infectious person present per day. Fixed at "
                  "1.0 by definition: it sets the reservoir's unit (person-days of shedding) so "
                  "that the dose scale below carries all the real scaling. Changing it would just "
                  "rescale that other parameter.",
    ),
    ParamMeta(
        path="pathogens[crypto].water_dose_scale", label="Waterborne dose scale",
        category="Cryptosporidium", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=5.0,
        rationale="Converts the contamination of a household's drinking-water source into the "
                  "dose its members receive each day. Absorbs oocysts shed per person, the "
                  "fraction carried by runoff, dilution, how much water is drunk and any treatment "
                  "- none of which are measured for Akuse - so it is the calibrated knob for the "
                  "waterborne route. History-matched, but WEAKLY identified: values from ~0.02 to "
                  "~2.7 fit the targets equally, giving very different waterborne shares (~10% at "
                  "the default). Vary it in any water-shock scenario.",
    ),
    ParamMeta(
        path="pathogens[crypto].shock_oocyst_load", label="Shock oocyst load",
        category="internal", evidence_tier="assumption",

        rationale="LEGACY water model only (WaterConfig.model='legacy'). How heavily every water source is loaded when a contamination shock (flood, "
                  "infrastructure failure) occurs - the equivalent for this pathogen of 'all "
                  "sources become contaminated'. A documented assumption to be explored by "
                  "sensitivity analysis, like the shock frequency itself, not a measured value.",
    ),
    ParamMeta(
        path="pathogens[crypto].zoonotic_interaction_rate", label="Animal-contact exposure rate",
        category="Cryptosporidium", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.1,
        rationale="Scales local ruminant density into infection risk. Expected to stay SMALL: "
                  "contact-network studies in rural sub-Saharan Africa, including Ghana, find no "
                  "elevated risk from animal contacts and conclude human-to-human spread "
                  "dominates. The control exists so livestock-focused scenarios can be explored, "
                  "not because it drives the epidemiology here. History-matched against a 2-15% "
                  "zoonotic share.",
    ),
    ParamMeta(
        path="pathogens[crypto].ruminant_weight", label="Ruminant risk weight",
        category="Cryptosporidium", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=2.0,
        rationale="Relative weight on ruminants within this pathogen's zoonotic route. Poultry "
                  "are excluded entirely - birds do not carry the Cryptosporidium species that "
                  "infect people - so this is effectively a units choice folded into the exposure "
                  "rate above.",
    ),
    ParamMeta(
        path="illness_mechanics.base_severity_crypto", label="Base cryptosporidiosis severity",
        category="Cryptosporidium", evidence_tier="literature", editable=True,
        ui_min=0.0, ui_max=1.0,
        rationale="Base illness severity (0-1 scale) before age and immunity adjustments. Set "
                  "between campylobacter (0.3) and rotavirus (0.4): Cryptosporidium diarrhoea in "
                  "young children is more often dehydrating and more often fatal than the average "
                  "cause (GEMS), but rotavirus remains the archetypal severe dehydrating "
                  "infection, and much of this pathogen's burden comes from how LONG episodes "
                  "last rather than how intense they are.",
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
        rationale="Expected illness duration at severity = 0 (and the minimum). Set with the "
                  "maximum and spread so that median durations match MAL-ED's 2-3 days (IQR within "
                  "1-7) for all three pathogens in children under two.",
    ),
    ParamMeta(
        path="illness_mechanics.duration_max_days", label="Illness duration (maximum)",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=3.0, ui_max=25.0, unit="days",
        rationale="Expected illness duration at severity = 1 (and the maximum). Same MAL-ED basis "
                  "as the minimum duration.",
    ),
    ParamMeta(
        path="illness_mechanics.duration_noise_std", label="Illness duration (spread)",
        category="Illness Mechanics", evidence_tier="literature", editable=True, ui_min=0.0, ui_max=4.0, unit="days",
        rationale="Stochastic spread around the expected illness duration. Same MAL-ED basis as "
                  "the min/max duration.",
    ),
    ParamMeta(
        path="illness_mechanics.duration_cap_days", label="Illness duration (ceiling)",
        category="internal", evidence_tier="literature",
        rationale="Hard ceiling on one episode's duration, separate from the expected maximum so "
                  "high-severity episodes do not pile up at it. 14 days is the WHO boundary "
                  "between acute and persistent diarrhoea.",
    ),

    # ------------------------------------------------------------------
    # Care-Seeking & Behavioral Economics
    # ------------------------------------------------------------------
    ParamMeta(
        path="steering_parameters.cost_of_care", label="Cost of seeking care",
        category="Care-Seeking & Behavioral Economics", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=0.3, unit="fraction of wealth",
        rationale="Cost of seeking medical care, as a fraction of max household wealth. History-"
                  "matched against the Ghana DHS childhood-diarrhoea care-seeking rate (69.2%, 95% "
                  "CI 65.6-72.8%, n=621). The care-seeking rate responds strongly to this and "
                  "hardly to income.",
    ),
    ParamMeta(
        path="steering_parameters.treatment_success_prob", label="Treatment success probability",
        category="Care-Seeking & Behavioral Economics", evidence_tier="assumption", editable=True,
        ui_min=0.3, ui_max=1.0,
        rationale="Probability that a care-seeking visit delivers effective treatment (correct "
                  "rehydration and zinc, taken). What effective treatment does is set by the "
                  "treatment parameters below. A plausible modeling default, not independently "
                  "measured for Akuse.",
    ),
    ParamMeta(
        path="steering_parameters.treatment_effect_model", label="Treatment effect model", category="internal",
        evidence_tier="structural",
        rationale="'supportive' (default): treatment reduces the episode's harm and shortens it modestly (ORS, zinc, antibiotics - see the treatment_* parameters). 'cure': treatment ends every active episode immediately (original model).",
    ),
    ParamMeta(
        path="steering_parameters.treatment_harm_reduction", label="Treatment: harm averted (ORS)",
        category="Care-Seeking & Behavioral Economics", evidence_tier="literature", editable=True,
        ui_min=0.0, ui_max=1.0,
        rationale="Fraction of an episode's severity (its daily health toll, and how sick the child appears) removed by effective treatment. 0.93: ORS may prevent 93% of diarrhoea deaths (Munos, Fischer Walker & Black 2010, Int J Epidemiol 39:i75-87) - dehydration, not diarrhoea itself, is what ORS treats.",
    ),
    ParamMeta(
        path="steering_parameters.treatment_duration_reduction_days", label="Treatment: days shortened (zinc)",
        category="Care-Seeking & Behavioral Economics", evidence_tier="literature", editable=True,
        ui_min=0.0, ui_max=3.0, unit="days",
        rationale="Days by which effective treatment shortens an episode in children aged at least treatment_duration_min_age_months (applied by stochastic rounding). Zinc: mean difference -11.46 h (95% CI -19.72 to -3.19) in children over 6 months; no effect under 6 months (Lazzerini & Wanzira 2016, Cochrane CD005436).",
    ),
    ParamMeta(
        path="steering_parameters.treatment_duration_min_age_months", label="Treatment: minimum age for zinc effect",
        category="internal", evidence_tier="literature",
        rationale="Youngest age (months) at which treatment shortens episodes (zinc: no effect under 6 months).",
    ),
    ParamMeta(
        path="steering_parameters.antibiotic_duration_reduction_days", label="Treatment: days shortened (antibiotic)",
        category="internal", evidence_tier="literature",
        rationale="Additional shortening of an episode of a pathogen whose infection the treatment clears (treatment_clears_infection_prob; campylobacter). Antibiotics reduce the duration of Campylobacter intestinal symptoms by 1.32 days (95% CI 0.64-1.99; Ternhag et al. 2007, Clin Infect Dis 44:696-700).",
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
        category="Household Economics", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.15, unit="fraction of wealth",
        rationale="Daily household income, as a fraction of max wealth. Kept from the first-"
                  "generation calibration: the current care-seeking target does not identify it "
                  "(one decision per episode, so affordability rarely binds), so the household "
                  "wealth outputs it drives are not calibrated.",
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
        category="internal", evidence_tier="assumption",

        rationale="LEGACY water model only (WaterConfig.model='legacy'). Daily probability a contaminated water source naturally reverts to clean. "
                  "Stated in the paper as an assumption pending real-world calibration, alongside "
                  "the shock frequency.",
    ),
    ParamMeta(
        path="steering_parameters.human_to_water_infection_prob", label="Human-to-water contamination rate",
        category="internal", evidence_tier="assumption",

        rationale="LEGACY water model only (WaterConfig.model='legacy'). Daily probability an infectious agent contaminates its local water source. A "
                  "plausible modeling default completing the water-transmission feedback loop, not "
                  "independently calibrated.",
    ),
    ParamMeta(
        path="steering_parameters.water_to_human_infection_prob", label="Water-to-human infection rate",
        category="internal", evidence_tier="calibrated",

        rationale="LEGACY water model only (WaterConfig.model='legacy'). Daily probability a susceptible agent is infected via contaminated water. "
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
        path="water.model", label="Water model", category="internal",
        evidence_tier="structural",
        rationale="'catchment' (default) or 'legacy' (original point-source model).",
    ),
    ParamMeta(
        path="water.share_piped", label="Piped drinking water (share)", category="Environment, Water & Shocks",
        evidence_tier="literature", editable=True, ui_min=0, ui_max=1,
        rationale="Share of households whose primary drinking water is piped (DHS/MICS within 30 km of Akuse).",
    ),
    ParamMeta(
        path="water.share_surface", label="Surface drinking water (share)", category="Environment, Water & Shocks",
        evidence_tier="literature", editable=True, ui_min=0, ui_max=1,
        rationale="Share drinking surface water (canal, river) - same survey basis.",
    ),
    ParamMeta(
        path="water.share_ground", label="Groundwater drinking water (share)", category="Environment, Water & Shocks",
        evidence_tier="literature", editable=True, ui_min=0, ui_max=1,
        rationale="Share drinking groundwater (well, borehole) - same survey basis.",
    ),
    ParamMeta(
        path="water.share_packaged", label="Sachet/packaged drinking water (share)", category="Environment, Water & Shocks",
        evidence_tier="literature", editable=True, ui_min=0, ui_max=1,
        rationale="Share drinking packaged (sachet/bottled) or other improved non-local water - same survey basis.",
    ),
    ParamMeta(
        path="water.source_choice_length_m", label="Source-choice distance scale", category="internal",
        evidence_tier="assumption",
        rationale="Households nearer a surface (or ground) source are more likely to use it: the required number of users of each type is drawn with weight exp(-distance / this). An assumption; the survey gives shares, not their spatial pattern.",
    ),
    ParamMeta(
        path="water.share_improved_onsite", label="Improved onsite sanitation (share)", category="Environment, Water & Shocks",
        evidence_tier="literature", editable=True, ui_min=0, ui_max=1,
        rationale="Share with improved non-sewered sanitation (pit with slab, VIP, septic) - same survey basis.",
    ),
    ParamMeta(
        path="water.share_sewer", label="Sewer sanitation (share)", category="Environment, Water & Shocks",
        evidence_tier="literature", editable=True, ui_min=0, ui_max=1,
        rationale="Share with a sewer connection - same survey basis.",
    ),
    ParamMeta(
        path="water.share_unimproved", label="Unimproved sanitation (share)", category="Environment, Water & Shocks",
        evidence_tier="literature", editable=True, ui_min=0, ui_max=1,
        rationale="Share with unimproved sanitation - same survey basis.",
    ),
    ParamMeta(
        path="water.share_open_defecation", label="Open defecation (share)", category="Environment, Water & Shocks",
        evidence_tier="literature", editable=True, ui_min=0, ui_max=1,
        rationale="Share practising open defecation - same survey basis.",
    ),
    ParamMeta(
        path="water.surface_release_open_defecation", label="Surface release: open defecation", category="internal",
        evidence_tier="literature",
        rationale="Relative surface-water release, open defecation (reference).",
    ),
    ParamMeta(
        path="water.surface_release_unimproved", label="Surface release: unimproved", category="internal",
        evidence_tier="assumption",
        rationale="Relative surface-water release, unimproved sanitation (assumption).",
    ),
    ParamMeta(
        path="water.surface_release_improved_onsite", label="Surface release: improved onsite", category="internal",
        evidence_tier="literature",
        rationale="Relative surface-water release, improved onsite sanitation (GloWPa: non-source).",
    ),
    ParamMeta(
        path="water.surface_release_sewer", label="Surface release: sewer", category="internal",
        evidence_tier="assumption",
        rationale="Relative surface-water release, sewer (assumption: ~99% contained/treated).",
    ),
    ParamMeta(
        path="water.ground_release_open_defecation", label="Groundwater release: open defecation", category="internal",
        evidence_tier="assumption",
        rationale="Relative groundwater release, open defecation (assumption).",
    ),
    ParamMeta(
        path="water.ground_release_unimproved", label="Groundwater release: unimproved", category="internal",
        evidence_tier="assumption",
        rationale="Relative groundwater release, unimproved pit (assumption).",
    ),
    ParamMeta(
        path="water.ground_release_improved_onsite", label="Groundwater release: improved onsite", category="internal",
        evidence_tier="assumption",
        rationale="Relative groundwater release, improved pit/septic (assumption).",
    ),
    ParamMeta(
        path="water.ground_release_sewer", label="Groundwater release: sewer", category="internal",
        evidence_tier="assumption",
        rationale="Relative groundwater release, sewer (assumption).",
    ),
    ParamMeta(
        path="water.child_stool_unsafe_frac", label="Unsafe child-stool disposal", category="Environment, Water & Shocks",
        evidence_tier="assumption", editable=True, ui_min=0, ui_max=1,
        rationale="Fraction of an infectious child's stools disposed of unsafely (released as if by open defecation) whatever the household's sanitation. ASSUMPTION pending the DHS 'safe disposal of child's stools' indicator; children are the main shedders, so this matters more than household sanitation.",
    ),
    ParamMeta(
        path="water.runoff_length_m", label="Runoff distance scale", category="Environment, Water & Shocks",
        evidence_tier="assumption", editable=True, ui_min=10, ui_max=1000, unit="m",
        rationale="Distance attenuation of faecal runoff to surface water: the fraction of a household's released load reaching water falls as exp(-d / this), d = distance to the nearest water cell. An assumption to be explored in sensitivity analysis.",
    ),
    ParamMeta(
        path="water.groundwater_length_m", label="Groundwater seepage distance scale", category="internal",
        evidence_tier="assumption",
        rationale="Distance attenuation of pit/faecal contamination to a well, exp(-d / this). Of the order of the 15-30 m latrine-well setback distances in WHO guidance.",
    ),
    ParamMeta(
        path="water.borehole_protection", label="Borehole protection", category="internal",
        evidence_tier="assumption",
        rationale="Load reaching a (deep, sealed) borehole relative to a shallow well. Assumption.",
    ),
    ParamMeta(
        path="water.river_dilution", label="River dilution", category="internal",
        evidence_tier="assumption",
        rationale="Concentration in a Volta river reach relative to an irrigation-canal reach receiving the same load - the river's discharge is orders of magnitude larger. Assumption.",
    ),
    ParamMeta(
        path="water.reach_block_cells", label="Reach size", category="internal",
        evidence_tier="structural",
        rationale="Surface water is split into reaches by tiling the grid into blocks of this many cells (8 cells ~ 450 m); each block's canal cells and river cells form one reach each.",
    ),
    ParamMeta(
        path="water.river_min_width_cells", label="River width threshold", category="internal",
        evidence_tier="structural",
        rationale="Natural-water cells that survive a morphological opening with a square of this many cells are classed as river; the rest (thin features) as canal.",
    ),
    ParamMeta(
        path="water.shock_magnitude", label="Shock magnitude", category="Environment, Water & Shocks",
        evidence_tier="assumption", editable=True, ui_min=0, ui_max=50,
        rationale="A shock adds to every source this many days' worth of the current mean daily surface-water loading (times the source type's vulnerability). An assumption, like shock_daily_prob itself.",
    ),
    ParamMeta(
        path="water.shock_vulnerability_ground", label="Shock vulnerability: groundwater", category="internal",
        evidence_tier="assumption",
        rationale="Shock pulse reaching wells/boreholes relative to surface water (assumption).",
    ),
    ParamMeta(
        path="water.shock_vulnerability_piped", label="Shock vulnerability: piped", category="internal",
        evidence_tier="assumption",
        rationale="Shock pulse reaching the piped network relative to surface water (assumption: pipe breaks, intermittent-supply intrusion).",
    ),
    ParamMeta(
        path="water.piped_flush_rate", label="Piped network flushing", category="internal",
        evidence_tier="assumption",
        rationale="Extra daily removal of contamination from the piped network (flushing and residual chlorine), on top of the pathogen's own die-off. Assumption.",
    ),
    ParamMeta(
        path="pathogens[rota].water_dose_scale", label="Waterborne dose scale", category="Rotavirus",
        evidence_tier="calibrated", editable=True, ui_min=0.0, ui_max=2.0,
        rationale="Converts the contamination of a household's drinking-water source into a daily "
                  "infection probability (catchment water model). History-matched jointly with the "
                  "other transmission parameters; ~7% of rotavirus infections are waterborne at "
                  "the default. Calibrated for 4000 agents - does not transfer to other sizes.",
    ),
    ParamMeta(
        path="pathogens[rota].water_decay_rate", label="Die-off rate in water", category="Rotavirus",
        evidence_tier="literature", editable=True, ui_min=0.02, ui_max=1.0, unit="per day",
        rationale="Daily first-order die-off of rotavirus in water (0.25: T90 ~ 9 days). Enteric "
                  "viruses persist for weeks in surface water; tropical temperatures put Akuse at "
                  "the faster end.",
    ),
    ParamMeta(
        path="pathogens[rota].importation_rate", label="Imported infection rate", category="Rotavirus",
        evidence_tier="calibrated", editable=True, ui_min=0.0, ui_max=0.003, unit="per person per day",
        rationale="Daily per-person probability of a rotavirus infection acquired outside Akuse. "
                  "Without it a town of a few thousand people cannot sustain rotavirus at MAL-ED-level "
                  "incidence: it fades out, or runs as outbreaks that cluster infections unrealistically. "
                  "History-matched with the share of imported infections held below 15% (~10% at the "
                  "default). This share bounds what any intervention inside Akuse can achieve.",
    ),
    ParamMeta(
        path="pathogens[crypto].importation_rate", label="Imported infection rate", category="internal",
        evidence_tier="assumption",
        rationale="Daily per-person probability of infection acquired outside Akuse. As for rotavirus.",
    ),
    ParamMeta(
        path="pathogens[rota].asymptomatic_relative_infectiousness", label="Infectiousness of symptomless infections", category="internal", evidence_tier="assumption",
        rationale="Infectiousness of a symptomless infection (every adult one, children's subclinical ones) relative to one that caused illness, on every transmission route. Symptomless infections shed far less; calibrated within 0.1-1.",
    ),
    ParamMeta(
        path="pathogens[rota].symptomatic_reduction_per_prior_infection", label="Fall in illness probability per prior infection", category="internal", evidence_tier="assumption",
        rationale="Acquired immunity protects against illness more than against infection (Velazquez et al. 1996): P(illness | infection) is multiplied by exp(-value x prior infections). 0 = no dependence on history.",
    ),
    ParamMeta(
        path="pathogens[rota].detectable_shedding_days_symptomatic", label="Detectable shedding, symptomatic (observation only)", category="internal", evidence_tier="structural",
        rationale="Used only to reproduce what MAL-ED's monthly stool surveillance would have detected; no effect on the simulation.",
    ),
    ParamMeta(
        path="pathogens[rota].detectable_shedding_days_asymptomatic", label="Detectable shedding, symptomless (observation only)", category="internal", evidence_tier="structural",
        rationale="Used only to reproduce what MAL-ED's monthly stool surveillance would have detected; no effect on the simulation.",
    ),
    ParamMeta(
        path="pathogens[campy].asymptomatic_relative_infectiousness", label="Infectiousness of symptomless infections", category="internal", evidence_tier="assumption",
        rationale="Infectiousness of a symptomless infection (every adult one, children's subclinical ones) relative to one that caused illness, on every transmission route. Symptomless infections shed far less; calibrated within 0.1-1.",
    ),
    ParamMeta(
        path="pathogens[campy].symptomatic_reduction_per_prior_infection", label="Fall in illness probability per prior infection", category="internal", evidence_tier="assumption",
        rationale="Acquired immunity protects against illness more than against infection (Velazquez et al. 1996): P(illness | infection) is multiplied by exp(-value x prior infections). 0 = no dependence on history.",
    ),
    ParamMeta(
        path="pathogens[campy].detectable_shedding_days_symptomatic", label="Detectable shedding, symptomatic (observation only)", category="internal", evidence_tier="structural",
        rationale="Used only to reproduce what MAL-ED's monthly stool surveillance would have detected; no effect on the simulation.",
    ),
    ParamMeta(
        path="pathogens[campy].detectable_shedding_days_asymptomatic", label="Detectable shedding, symptomless (observation only)", category="internal", evidence_tier="structural",
        rationale="Used only to reproduce what MAL-ED's monthly stool surveillance would have detected; no effect on the simulation.",
    ),
    ParamMeta(
        path="pathogens[crypto].asymptomatic_relative_infectiousness", label="Infectiousness of symptomless infections", category="internal", evidence_tier="assumption",
        rationale="Infectiousness of a symptomless infection (every adult one, children's subclinical ones) relative to one that caused illness, on every transmission route. Symptomless infections shed far less; calibrated within 0.1-1.",
    ),
    ParamMeta(
        path="pathogens[crypto].symptomatic_reduction_per_prior_infection", label="Fall in illness probability per prior infection", category="internal", evidence_tier="assumption",
        rationale="Acquired immunity protects against illness more than against infection (Velazquez et al. 1996): P(illness | infection) is multiplied by exp(-value x prior infections). 0 = no dependence on history.",
    ),
    ParamMeta(
        path="pathogens[crypto].detectable_shedding_days_symptomatic", label="Detectable shedding, symptomatic (observation only)", category="internal", evidence_tier="structural",
        rationale="Used only to reproduce what MAL-ED's monthly stool surveillance would have detected; no effect on the simulation.",
    ),
    ParamMeta(
        path="pathogens[crypto].detectable_shedding_days_asymptomatic", label="Detectable shedding, symptomless (observation only)", category="internal", evidence_tier="structural",
        rationale="Used only to reproduce what MAL-ED's monthly stool surveillance would have detected; no effect on the simulation.",
    ),
    ParamMeta(
        path="pathogens[rota].adult_relative_susceptibility", label="Adult susceptibility", category="internal", evidence_tier="assumption",
        rationale="Relative susceptibility of every agent aged 5+. Unset, adults sit at the children's immunity plateau (campaign v3); carriage surveys find adults carry less than young children.",
    ),
    ParamMeta(
        path="pathogens[rota].adult_infectious_duration_scale", label="Adult infectious-period multiplier", category="internal", evidence_tier="assumption",
        rationale="Multiplier on the mean infectious period of an adult's infection (1 = as children).",
    ),
    ParamMeta(
        path="pathogens[campy].adult_relative_susceptibility", label="Adult susceptibility", category="internal", evidence_tier="assumption",
        rationale="Relative susceptibility of every agent aged 5+. Unset, adults sit at the children's immunity plateau (campaign v3); carriage surveys find adults carry less than young children.",
    ),
    ParamMeta(
        path="pathogens[campy].adult_infectious_duration_scale", label="Adult infectious-period multiplier", category="internal", evidence_tier="assumption",
        rationale="Multiplier on the mean infectious period of an adult's infection (1 = as children).",
    ),
    ParamMeta(
        path="steering_parameters.max_contacts_per_visit", label="Close contacts per place visited", category="internal", evidence_tier="literature",
        rationale="Contacts an agent makes with other households at one location per day phase; in a larger crowd exposure scales with the share infectious. Contact surveys: ~7 contacts a day, half in the household (le Polain de Waroux et al. 2018, Uganda). Unset = everyone in the cell is a contact.",
    ),
    ParamMeta(
        path="pathogens[crypto].adult_relative_susceptibility", label="Adult susceptibility", category="internal", evidence_tier="assumption",
        rationale="Relative susceptibility of every agent aged 5+. Unset, adults sit at the children's immunity plateau (campaign v3); carriage surveys find adults carry less than young children.",
    ),
    ParamMeta(
        path="pathogens[crypto].adult_infectious_duration_scale", label="Adult infectious-period multiplier", category="internal", evidence_tier="assumption",
        rationale="Multiplier on the mean infectious period of an adult's infection (1 = as children).",
    ),
    ParamMeta(
        path="pathogens[campy].poultry_roam_sigma_m", label="Poultry roam radius (m)", category="internal", evidence_tier="structural",
        rationale="Metre version of the cell-based roam radius, converted with the grid's cell size so a finer grid keeps the same physical radius.",
    ),
    ParamMeta(
        path="pathogens[campy].ruminant_roam_sigma_m", label="Ruminant roam radius (m)", category="internal", evidence_tier="structural",
        rationale="Metre version of the cell-based roam radius, converted with the grid's cell size so a finer grid keeps the same physical radius.",
    ),
    ParamMeta(
        path="steering_parameters.social_interaction_radius_m", label="Social-visit radius (m)", category="internal", evidence_tier="structural",
        rationale="Metre version of social_interaction_radius, converted with the grid's cell size so a finer grid keeps the same physical radius.",
    ),
    ParamMeta(
        path="water.reach_block_m", label="Surface-water reach size (m)", category="internal", evidence_tier="structural",
        rationale="Metre version of reach_block_cells, converted with the grid's cell size.",
    ),
    ParamMeta(
        path="water.river_min_width_m", label="Minimum river width (m)", category="internal", evidence_tier="structural",
        rationale="Metre version of river_min_width_cells, converted with the grid's cell size.",
    ),
    ParamMeta(
        path="illness_mechanics.onset_before_transmission", label="Illness onset decided before transmission", category="internal", evidence_tier="structural",
        rationale="Decide a newly infectious child's illness before the day's transmission, so its infectiousness reflects whether it is ill.",
    ),
    ParamMeta(
        path="spatial_creation_args.household_placement", label="Household placement", category="internal", evidence_tier="structural",
        rationale="'uniform' (one household per random residence cell, original) or 'population' (cells in proportion to GHS-POP population).",
    ),
    ParamMeta(
        path="spatial_creation_args.population_epoch", label="Population layer epoch", category="internal", evidence_tier="structural",
        rationale="GHS-POP epoch (2020, 2025 or 2030) of the population layer used for household placement.",
    ),
    ParamMeta(
        path="observation.surveillance_start_months", label="First surveillance month", category="internal", evidence_tier="structural",
        rationale="MAL-ED surveillance design for the observation layer (experiments/observation.py); analysis only, no effect on the simulation.",
    ),
    ParamMeta(
        path="observation.surveillance_end_months", label="Last surveillance month", category="internal", evidence_tier="structural",
        rationale="MAL-ED surveillance design for the observation layer (experiments/observation.py); analysis only, no effect on the simulation.",
    ),
    ParamMeta(
        path="observation.diarrhoeal_stool_tested_prob", label="Diarrhoeal stools tested", category="internal", evidence_tier="structural",
        rationale="MAL-ED surveillance design for the observation layer (experiments/observation.py); analysis only, no effect on the simulation.",
    ),
    ParamMeta(
        path="observation.shedding_duration_shape", label="Shedding duration shape", category="internal", evidence_tier="structural",
        rationale="MAL-ED surveillance design for the observation layer (experiments/observation.py); analysis only, no effect on the simulation.",
    ),
    ParamMeta(
        path="observation.consecutive_gap_months", label="Consecutive-detection gap", category="internal", evidence_tier="structural",
        rationale="MAL-ED surveillance design for the observation layer (experiments/observation.py); analysis only, no effect on the simulation.",
    ),
    ParamMeta(
        path="observation.incubation_shedding_days", label="Incubation shedding window", category="internal", evidence_tier="structural",
        rationale="MAL-ED surveillance design for the observation layer (experiments/observation.py); analysis only, no effect on the simulation.",
    ),
    ParamMeta(
        path="observation.rng_seed_offset", label="Observation RNG seed offset", category="internal", evidence_tier="structural",
        rationale="MAL-ED surveillance design for the observation layer (experiments/observation.py); analysis only, no effect on the simulation.",
    ),
    ParamMeta(
        path="pathogens[campy].importation_rate", label="Imported infection rate", category="internal",
        evidence_tier="structural",
        rationale="0: campylobacter's food-borne background route already plays this role.",
    ),
    ParamMeta(
        path="steering_parameters.household_exposure_sdlog", label="Household environmental-exposure heterogeneity",
        category="Environment, Water & Shocks", evidence_tier="calibrated", editable=True,
        ui_min=0.0, ui_max=1.5,
        rationale="Spread (log-scale SD) of a household-level multiplier on infection risk via "
                  "water, animals and food, shared by all members and pathogens. Identified by "
                  "MAL-ED's finding that children already infected with Campylobacter go on to be "
                  "infected 1.5-3.7x as often as naive children - the signature of some households "
                  "being far more exposed than others. 0 = every household equally exposed.",
    ),
    ParamMeta(
        path="steering_parameters.prior_infection_immunity_factor", label="Cross-episode immunity factor",
        category="Environment, Water & Shocks", evidence_tier="assumption", editable=True,
        ui_min=0.0, ui_max=0.5,
        rationale="Per-prior-infection decay of susceptibility, exp(-kappa N), for any pathogen "
                  "without its own value. Only campylobacter uses it now: rotavirus (0.45) and "
                  "Cryptosporidium (0.29) have cohort-based overrides. No LMIC estimate exists for "
                  "campylobacter, so 0.15 remains an assumption.",
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
                # Driven by config.PATHOGEN_CONFIG_TYPES rather than a hardcoded
                # tuple, so adding a pathogen automatically makes this
                # completeness check demand registry entries for its fields
                # instead of silently ignoring them.
                for label, pathogen_cls in PATHOGEN_CONFIG_TYPES.items():
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

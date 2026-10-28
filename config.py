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
    name: str = Field(..., description="Pathogen identifier ('rota', 'campy' or 'crypto').")
    initial_exposed_proportion: float = Field(
        0.01, description="Fraction of agents seeded as exposed at simulation start."
    )
    recovery_rate: float = Field(..., description="Daily probability of recovering from infection.")
    exposure_period: int = Field(
        ..., description="Latent period (days) between exposure and becoming infectious."
    )

    # --- Infection -> symptoms coupling -----------------------------------
    # Fraction of infections in children that produce a symptomatic illness
    # episode at all. 1.0 (the default, and the value for rotavirus and
    # campylobacter) reproduces the original behaviour exactly: every
    # infectious child gets an episode. Cryptosporidium sets this below 1
    # because the majority of childhood infections in endemic settings are
    # subclinical - yet still shed and still transmit, which is precisely the
    # "silent reservoir" that drives its epidemiology.
    symptomatic_fraction: float = Field(
        1.0,
        description="Fraction of infections in children that produce a symptomatic illness "
                    "episode. Infections that do not still progress S/R -> E -> I -> R and "
                    "still transmit; they simply never acquire a severity or duration.",
        ge=0.0, le=1.0,
    )
    # Multiplies the expected illness duration from IllnessMechanicsConfig
    # (duration_min_days .. duration_max_days) and its clamp, per pathogen.
    # 1.0 for rotavirus/campylobacter, whose durations were set against that
    # shared 2-12 day window; >1 for cryptosporidium, whose episodes commonly
    # run past the shared 12-day ceiling.
    duration_scale: float = Field(
        1.0,
        description="Per-pathogen multiplier on expected illness-episode duration (and on the "
                    "duration clamp). 1.0 leaves the shared IllnessMechanicsConfig window as-is.",
        gt=0.0,
    )
    # Age profile of symptomatic_fraction. MAL-ED (abm/evidence/maled.py,
    # Supplementary Table 5) shows P(attributable diarrhoea | infection)
    # changes materially over the first two years - campylobacter halves from
    # 0.22 to 0.11 - so one number misstates it for most ages. The profile is a
    # set of RELATIVE multipliers on symptomatic_fraction, not absolute
    # probabilities, so that symptomatic_fraction stays the single scalar a
    # sweep, calibration or GSA varies (scaling every age band together) and
    # there is no second field that could silently override it. Each entry is
    # (upper_age_months, multiplier): a child of age a uses the first band with
    # a <= upper_age_months; children older than the last upper edge use the
    # last band. P(symptomatic) = clamp(symptomatic_fraction * multiplier, 0, 1).
    symptomatic_fraction_age_profile: list[tuple[float, float]] | None = Field(
        None,
        description="Optional age profile [(upper_age_months, multiplier), ...] (strictly "
                    "increasing upper ages) applied multiplicatively to symptomatic_fraction; "
                    "the last band also applies to all older children. None = no age dependence.",
    )
    # Optional per-pathogen override of the shared acquired-immunity decay
    # (steering_parameters.prior_infection_immunity_factor). None = use the
    # shared value, which is what rotavirus and campylobacter do.
    immunity_decay_per_infection: float | None = Field(
        None,
        description="Per-prior-infection decay rate kappa in the susceptibility term "
                    "exp(-kappa * N_infections). None means use the shared "
                    "steering_parameters.prior_infection_immunity_factor.",
    )
    # --- Age-specific susceptibility (see Pathogen._relative_susceptibility) ---
    # Relative susceptibility of a child is max(immunity_floor, exp(-kappa*N))
    # times an infant ramp; an adult's is immunity_floor. The floor is the
    # plateau that repeated infection converges to, and adults are assumed to
    # have reached it - which also makes adult susceptibility stationary
    # rather than drifting downward forever in a population whose adults
    # never age out. None reproduces the original model: no floor, and adults
    # treated exactly like children (exp(-kappa*N) from N=0 at t=0).
    immunity_floor: float | None = Field(
        None,
        description="Plateau of relative susceptibility after repeated infection (and the "
                    "susceptibility of every adult). None = no plateau; adults treated like "
                    "children (original behaviour).",
        ge=0.0, le=1.0,
    )
    # --- Adults (every agent aged 5+; see Pathogen._relative_susceptibility) ---
    # Holding adults at the children's plateau made them far more often
    # infected than young children once the population was data-defined (v3
    # validation: rotavirus 6.6 true infections per person-year at 5+, adult
    # carriage ~10x young children's, 98% of the infectious pressure). Carriage
    # surveys find adults carry less than young children (rotavirus antigen in
    # symptomless people: adults ~3% vs 4.2% at 1-5 years, Endara et al. 2007,
    # Ecuador; Cryptosporidium by qPCR in households of children without the
    # infection: 4% of household members, median age 15, vs 8% of neighbouring
    # under-5s, Krumkamp et al. 2021, Clin Infect Dis 72:1358, Ghana among four
    # countries). Adult immunity, and adults' smaller faecal-oral exposure, are
    # not separately identifiable, so one relative susceptibility stands for
    # both. Campaign v4 calibrates it for rotavirus and Cryptosporidium
    # (RotavirusConfig, CryptosporidiumConfig); Campylobacter keeps the default.
    adult_relative_susceptibility: float | None = Field(
        None,
        description="Relative susceptibility of every adult (agent aged 5+), on every route. None = "
                    "immunity_floor (adults at the children's plateau, as in campaign v3).",
        ge=0.0, le=1.0,
    )
    adult_infectious_duration_scale: float = Field(
        1.0,
        description="Multiplier on the mean infectious period of an adult's infection: an infectious "
                    "adult recovers each day with probability min(1, recovery_rate / scale). 1.0 = as "
                    "children.",
        gt=0.0,
    )
    # Susceptibility x exposure multiplier for young infants, rising linearly
    # from infant_susceptibility_at_birth at age 0 to 1 at
    # infant_susceptibility_ramp_months. Represents everything that keeps
    # infection rates low in the first months - maternal antibody, exclusive
    # breastfeeding, limited mobility and mouthing - without separating them.
    # Defaults (1.0, 0) = no ramp.
    infant_susceptibility_at_birth: float = Field(
        1.0, description="Relative susceptibility of a newborn (multiplier at age 0).", ge=0.0, le=1.0,
    )
    infant_susceptibility_ramp_months: float = Field(
        0.0, description="Age (months) by which the infant multiplier has risen to 1. 0 = no ramp.",
        ge=0.0,
    )
    # Infection acquired OUTSIDE the modelled population (travel, markets,
    # visitors): a daily per-person probability, scaled by relative
    # susceptibility, attributed to the "imported" route. A closed town of a
    # few thousand people cannot sustain a pathogen at the low endemic
    # incidence MAL-ED reports - it fades out stochastically and, with no
    # outside contact, never returns (the legacy model was reseeded instead by
    # its water shocks, which set every source to contaminated regardless of
    # whether anyone was shedding). Kept small, so imported infections are a
    # spark rather than a route; report the imported fraction alongside any
    # result. 0 = closed population (original behaviour).
    importation_rate: float = Field(
        0.0,
        description="Daily per-person probability of an infection acquired outside the modelled "
                    "population (route 'imported'). Prevents stochastic fade-out in a small "
                    "closed population. 0 = closed population.",
        ge=0.0,
    )
    # What a SUCCESSFUL care-seeking visit does to this pathogen's infection,
    # as opposed to its symptoms (whose treatment is set by
    # steering_parameters.treatment_effect_model: "supportive" shortens and
    # softens the episode, "cure" ends it - see CareSeekingSystem). Standard
    # care for childhood watery diarrhoea (IMCI: ORS + zinc) rehydrates but
    # does not shorten pathogen shedding, so for most pathogens this should be
    # 0. 1.0 reproduces the original model, where a cured child also stopped
    # shedding every pathogen at once; under "supportive" a clearance also
    # shortens the episode by antibiotic_duration_reduction_days.
    treatment_clears_infection_prob: float = Field(
        1.0,
        description="Probability that a successful care-seeking visit also ends this "
                    "pathogen's infection (INFECTIOUS -> RECOVERED), i.e. that the treatment "
                    "given is pathogen-specific and effective. 0 = supportive care only "
                    "(shedding continues).",
        ge=0.0, le=1.0,
    )

    # Acquired immunity protects against DISEASE far more than against
    # infection: after one rotavirus infection, protection was 77% against
    # diarrhoea but 38% against any infection (Velazquez et al. 1996, N Engl J
    # Med 335:1022-8), i.e. a reinfection is ~0.37x as likely to cause
    # diarrhoea as a first infection. P(episode | infection) is multiplied by
    # exp(-this * number of prior infections), on top of the age profile.
    # 0 = no dependence on history (original model).
    symptomatic_reduction_per_prior_infection: float = Field(
        0.0,
        description="Per-prior-infection log-reduction of P(illness episode | infection): the "
                    "probability is multiplied by exp(-value x prior infections). 0 = none.",
        ge=0.0,
    )

    # --- Symptomless infections ---
    # Infectiousness: every transmission route counts an infectious agent with
    # weight 1 if its current infection produced an illness episode, and this
    # weight otherwise (all adult infections, and children's subclinical ones).
    # Symptomless infections shed far less virus - rotavirus: median Cq 31 vs
    # 17 at presentation in Vellore children (repository.ias.ac.in/98006) -
    # and household transmission rises with the index child's severity
    # (Bennett et al. 2020, Lancet Infect Dis). Estimates of the resulting
    # relative infectiousness range from ~1 per day of shedding (White et al.
    # 2008, Oxfordshire day care) to 1/5-1/10 for later rotavirus infections
    # (a model fitted to Kazakh hospital data, Emerg Infect Dis 2014 technical
    # appendix), so it is calibrated within [0.1, 1]. 1.0 = original model.
    asymptomatic_relative_infectiousness: float = Field(
        1.0,
        description="Infectiousness of a symptomless infection (every adult infection, children's "
                    "subclinical ones) relative to one that produced an illness episode, on every "
                    "transmission route. 1.0 = no difference (original model).",
        ge=0.0, le=1.0,
    )
    # Detectability, used ONLY by the MAL-ED observation layer
    # (experiments/observation.py) - no effect on the simulation. Days from
    # the onset of infectiousness during which a stool would test positive at
    # MAL-ED's cut-off (Ct <= 30; rotavirus <= 32.6), gamma-distributed with
    # these means (ObservationConfig.shedding_duration_shape). This is not the
    # infectious period: qPCR detects shedding for weeks, but MAL-ED's
    # cut-off is high enough that a symptomless rotavirus infection starting
    # at Cq ~31 may never be counted.
    detectable_shedding_days_symptomatic: float = Field(
        14.0,
        description="Observation layer only: mean days a symptomatic infection's stool tests "
                    "positive at MAL-ED's cut-off.",
        gt=0.0,
    )
    detectable_shedding_days_asymptomatic: float = Field(
        7.0,
        description="Observation layer only: mean days a symptomless infection's stool tests "
                    "positive at MAL-ED's cut-off.",
        gt=0.0,
    )

    @field_validator("symptomatic_fraction_age_profile")
    @classmethod
    def _check_age_profile(cls, v):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("symptomatic_fraction_age_profile must be None or non-empty")
        uppers = [float(a) for a, _ in v]
        if uppers[0] <= 0 or any(b <= a for a, b in zip(uppers, uppers[1:])):
            raise ValueError("symptomatic_fraction_age_profile upper ages must be positive and "
                             "strictly increasing")
        if any(float(m) < 0.0 for _, m in v):
            raise ValueError("symptomatic_fraction_age_profile multipliers must be >= 0")
        return [(float(a), float(m)) for a, m in v]


def _maled_age_profile(band_fractions: tuple[float, ...], overall: float) -> list[tuple[float, float]]:
    """MAL-ED 6-month-band P(diarrhoea | infection) as multipliers on `overall`.

    band_fractions are Supplementary Table 5 (algorithm attribution,
    consecutive-detections infections) for ages (0,6], (6,12], (12,18],
    (18,24] months; overall is attributable episodes / infections from
    Supplementary Table 1. Values are transcribed in abm/evidence/maled.py, and
    model_tests/test_disease_scale.py checks these configs against that module.
    Children older than 24 months take the 18-24 month band: MAL-ED followed
    children only to age two, so this is an extrapolation, held flat rather
    than extrapolating the declining trend.
    """
    return [(float(u), round(f / overall, 4)) for u, f in zip((6, 12, 18, 24), band_fractions)]

class RotavirusConfig(PathogenConfig):
    """Parameters specific to Rotavirus."""
    name: str = Field("rota", description="Pathogen identifier.")
    # CALIBRATION (campaign v4, 2026-10-27; docs/calibration_v4_adults.md).
    # The parameters marked "History-matched, campaign v4" below -
    # transmission, illness and infant protection for each pathogen, adults'
    # susceptibility to rotavirus and Cryptosporidium, the infectiousness of
    # symptomless infections, household exposure heterogeneity and
    # Campylobacter's and Cryptosporidium's detectable-shedding durations -
    # were set together by history matching (experiments/calibration/
    # history_matching.py, campaign "v4": 3 waves of 300 design points x 2
    # seeds, 29 parameters) against 31 targets: MAL-ED as its surveillance
    # would have seen the model (experiments/observation.py - detected
    # infections and attributable diarrhoea 0-24 months, the age shape of
    # detected incidence, rates by detected infection history, repeat
    # detections, age at first detection), true rotavirus infection from the
    # Vellore cohort, the route-share bands, and adult : young-child carriage
    # of rotavirus and Cryptosporidium. v4 changed the structure as well as
    # the values: close contacts with other households are capped
    # (steering_parameters.max_contacts_per_visit = 3.5), adults have their
    # own susceptibility, and Cryptosporidium adults shed for a shorter time
    # (adult_infectious_duration_scale = 0.65). The adult-transmission pilots
    # showed v3's adult epidemic (6.8 rotavirus infections per person-year at
    # 5+) to be crowd exposure at the 8 mapped places of worship.
    #
    # THESE VALUES ARE FOR THE AKUSE SETTLEMENT GRID: grid 6855dd54618c (the
    # GHS-POP settlement domain at 25 m, scripts/make_settlement_grid.py),
    # 11,476 agents = its 2025 GHS-POP population, households placed from the
    # population layer onto OSM buildings (scripts/add_population_layer.py),
    # distances in metres, 1460 days with a 730-day burn-in, rotavirus
    # vaccination off. Transmission does not transfer to another grid or
    # population size. The web app still runs the original grid, on the
    # previous calibration (webapp/scenario_form.py WEBAPP_DEFAULTS =
    # these defaults + experiments/legacy.py::V2_OVERRIDES, which also
    # reproduces campaign v2 and its 4000-agent runs exactly).
    #
    # How the default was chosen (history_matching.py --campaign v4 --select
    # --route-ranges): illness protection from prior infections at literature
    # values (V3_LITERATURE_FIXED); for rotavirus and Cryptosporidium,
    # illness-causing infections detectable at least as long as symptomless
    # ones (v3_physical_constraint); detected incidence and the share of
    # detected infections causing diarrhoea inside the range across MAL-ED's
    # high-burden sites; imported rotavirus below 15%; adult : child carriage
    # in its plausible range (V4_CARRIAGE_RANGES); and at most 10% of
    # rotavirus infections waterborne (V4_ROUTE_RANGES) - without that rule
    # the best candidate fitted every target but had 34% of rotavirus
    # waterborne, against the evidence that rotavirus spreads mainly by direct
    # contact. Rotavirus's two detection windows were then set by re-scoring
    # the verification runs (see detectable_shedding_days_* below).
    #
    # Verified on 8 fresh seeds (verify_v4_routes_det5 e2, re-scored): 30 of
    # the 31 targets within 2.5 SD; ONE MISS, accepted by the user:
    # Cryptosporidium's detected incidence at 0-6 vs 6-12 months (0.63 vs
    # MAL-ED 0.25, 3.3 SD; every v4 candidate was weakest there). Every
    # incidence, episode rate and illness share inside MAL-ED's site range;
    # carriage ratios 0.56 (rotavirus) and 0.44 (Cryptosporidium); no
    # rotavirus fade-out; rotavirus 91% person-to-person, 2.2% waterborne,
    # 6.6% imported. Weakest other fits (2.1-2.5 SD): Campylobacter's
    # second-year fall among never-detected children and its prior-infection
    # rate ratio, Cryptosporidium's second-year rise, rotavirus's
    # never-detected decline and its true infection by 6 months (0.32 vs
    # Vellore 0.56).
    #
    # NO ADMISSIBLE RANGES are quoted for v4. Under the emulators almost no
    # parameter set meets every rule (the route-constrained admissible set is
    # one sampled point), so the default is a single best fit, not a point in
    # a well-sampled region; several values sit at or near a search bound
    # (Cryptosporidium community transmission and infant ramp). The parameters
    # are strongly correlated: for sensitivity analysis, vary them JOINTLY
    # around the default, not one at a time. The v3 values are in
    # experiments/legacy.py::V3_OVERRIDES (regression_check --v3), v2's in
    # V2_OVERRIDES.
    infection_prob_mean: float = Field(
        0.343,
        description="Mean per-exposure person-to-person infection probability. History-matched, "
                    "campaign v4 (see the CALIBRATION note). Per close contact under the contact cap "
                    "(steering_parameters.max_contacts_per_visit), so not comparable with v3's 0.0405 "
                    "(everyone in a cell a contact), v2's 0.0261 (original grid) or the first "
                    "generation's 0.01263, which targeted the seeded initial epidemic.",
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

    # --- Infection -> disease (MAL-ED; see abm/evidence/maled.py) ---
    # Before this, every rotavirus infection in a child produced an episode.
    # MAL-ED (Amin et al. 2026, BMC Infect Dis 26:106; unvaccinated sites)
    # found 472 attributable diarrhoeal episodes from 734 infections in
    # children 0-24 months (algorithm attribution, consecutive-detection
    # infections) - 0.643 - varying with age as 0.58, 0.72, 0.67, 0.50 across
    # the four 6-month bands. Monthly surveillance stools under-detect short
    # asymptomatic infections, so if anything the true fraction is lower.
    symptomatic_fraction: float = Field(
        0.476,
        description="P(diarrhoea | FIRST infection) at the age profile's reference; later "
                    "infections are multiplied by exp(-symptomatic_reduction_per_prior_infection x "
                    "prior infections). History-matched, campaign v4 (v3: 0.467). MAL-ED's 472 "
                    "attributable episodes / 734 DETECTED infections (0.643) is over-weighted towards "
                    "illness, since diarrhoeal stools catch symptomatic infections that monthly stools "
                    "miss; the observation layer gives 0.73 at the default (MAL-ED site range "
                    "0.24-0.74).",
        ge=0.0, le=1.0,
    )
    symptomatic_fraction_age_profile: list[tuple[float, float]] | None = Field(
        default_factory=lambda: _maled_age_profile((0.58, 0.72, 0.67, 0.50), 0.643),
        description="MAL-ED Supplementary Table 5 by 6-month band, as multipliers on "
                    "symptomatic_fraction; held at the 18-24 month value beyond 24 months.",
    )
    # Importation was a fixed 2e-5 until history-matching wave 3 showed every
    # otherwise-acceptable point sitting at rotavirus's persistence threshold:
    # it faded out in 1-3 of 8 seeds, and the stochastic outbreaks in between
    # clustered infections in time, inflating the prior-infection rate ratio
    # to 2-12 (MAL-ED 1.008). Freed in waves 4-5, it removed both.
    importation_rate: float = Field(
        3.24e-4,
        description="Daily per-person probability of a rotavirus infection acquired outside "
                    "Akuse (route 'imported'), scaled by relative susceptibility. History-matched, "
                    "campaign v4. 6.6% of infections are imported at the default, with no fade-out "
                    "in 8 seeds (v3: 0.6%; v2's 4000-agent town needed ~10%). The share is capped at "
                    "15% when choosing the default - MAL-ED cannot tell a self-sustaining town from a "
                    "sink for outside infection - and it bounds what any intervention inside Akuse "
                    "can achieve; vary it in a GSA.",
        ge=0.0,
    )

    # --- Catchment water model (WaterConfig.model == "catchment") ---
    water_dose_scale: float = Field(
        0.00258,
        description="Converts a drinking-water source's concentration (infectious-person-days of "
                    "shedding reaching it, dilution-weighted) into P(infection) = 1 - exp(-scale * C) "
                    "per day of drinking. The calibrated knob of the catchment water route. Unused "
                    "by the legacy water model (which uses "
                    "steering_parameters.water_to_human_infection_prob). History-matched, campaign v4, "
                    "with at most 10% of rotavirus infections waterborne required of the default "
                    "(history_matching.V4_ROUTE_RANGES): 2.2% at the default (v3: 0.2%). Weakly "
                    "identified by the targets. Grid- and population-size dependent.",
        ge=0.0,
    )
    water_decay_rate: float = Field(
        0.25,
        description="Daily first-order die-off of rotavirus in water (T90 ~ 9 days). Enteric "
                    "viruses persist for weeks in surface water at 20-25 C (reviewed in Rzezutka & "
                    "Cook 2004, FEMS Microbiol Rev 28:441-53); tropical temperatures put Akuse at "
                    "the faster end. Order-of-magnitude value; comparable to the ~5-day memory of "
                    "the legacy model.",
        gt=0.0,
    )

    # --- Acquired immunity and infant protection ---
    # Velazquez et al. 1996 (N Engl J Med 335:1022-8; Mexico City birth
    # cohort): protection against reinfection of 38%, 62% and 74% after one,
    # two and three infections; Mohan et al. 2017 (J Infect Dis 216:305-16;
    # MAL-ED, unvaccinated sites): 74% after three. exp(-0.45 N) gives 0.64,
    # 0.41, 0.26 for N = 1, 2, 3 - i.e. 36%, 59%, 74%. The previous shared
    # value (0.15) implied only 36% protection after three infections.
    immunity_decay_per_infection: float | None = Field(
        0.45,
        description="Per-prior-infection decay kappa in exp(-kappa N). 0.45 reproduces 74% "
                    "protection after three infections (Velazquez et al. 1996; Mohan et al. 2017, "
                    "MAL-ED).",
    )
    immunity_floor: float | None = Field(
        0.26,
        description="Plateau of relative susceptibility, reached at three infections: 74% "
                    "protection is the most either cohort measured. Adults have their own "
                    "susceptibility (adult_relative_susceptibility; until campaign v4 they took this "
                    "value).",
        ge=0.0, le=1.0,
    )
    adult_relative_susceptibility: float | None = Field(
        0.216,
        description="Relative susceptibility of every adult (agent aged 5+), on every route. "
                    "History-matched, campaign v4, against an adult : young-child (12-59 months) "
                    "carriage ratio of 0.7 (Endara et al. 2007; Lopman et al. 2013; plausible range "
                    "0.4-1.1): 0.56 at the default. Below the children's plateau (0.26): it stands "
                    "for adults' smaller faecal-oral exposure as well as their immunity. v3: None "
                    "(= immunity_floor).",
        ge=0.0, le=1.0,   # redeclared: a subclass field replaces the base field outright
    )
    # MAL-ED Supplementary Table 2: infection rate at 0-3 months is 0.55x the
    # 6-9 month peak (2.58 vs 4.67 per 100 child-months). A linear ramp
    # reaching 1 at the peak (7.5 months) matches that ratio at 1.5 months
    # when it starts at 0.44. First-pass values read off the curve's shape,
    # to be refined by calibration against the full age curve.
    infant_susceptibility_at_birth: float = Field(
        0.725, description="Newborn relative susceptibility. History-matched, campaign v4, against "
                           "MAL-ED's detected age curve and Vellore's share infected by 6 months "
                           "(v3: 0.320; first-pass value 0.44).",
        ge=0.0, le=1.0,
    )
    infant_susceptibility_ramp_months: float = Field(
        9.27, description="Age (months) of full susceptibility. History-matched, campaign v4 (v3: 7.96; "
                          "first-pass value 7.5, MAL-ED's 6-9 month peak).", ge=0.0,
    )
    # Disease-specific immunity (see PathogenConfig) - set from the
    # literature, not calibrated (history_matching.V3_LITERATURE_FIXED).
    symptomatic_reduction_per_prior_infection: float = Field(
        0.50,
        description="P(diarrhoea | infection) x exp(-0.50) per prior infection. Velazquez et al. "
                    "1996: diarrhoea vs any infection after 1, 2, 3 infections gives relative "
                    "P(diarrhoea | infection) 0.37, 0.42, 0.24 - least-squares fit 0.50. MAL-ED's "
                    "calibrated hazard ratios (Rogawski McQuade et al. 2020, J Infect Dis "
                    "222:1858) give 0.33, a lower bound since its monthly stools missed most prior "
                    "infections.",
        ge=0.0,
    )
    asymptomatic_relative_infectiousness: float = Field(
        0.397,
        description="Infectiousness of a symptomless infection relative to one causing illness, "
                    "on every route. History-matched, campaign v4 - one value shared by the three "
                    "pathogens (v3: 0.201). Within the published 0.1-1 range (see PathogenConfig).",
        ge=0.0, le=1.0,
    )
    # No antiviral exists; ORS/zinc do not shorten shedding.
    treatment_clears_infection_prob: float = Field(
        0.0,
        description="Rotavirus has no specific treatment: ORS and zinc rehydrate but do not "
                    "shorten viral shedding, so a successful care-seeking visit ends the "
                    "episode's symptoms but not the infection.",
        ge=0.0, le=1.0,
    )
    # Observation layer only (see PathogenConfig). Symptomless infections
    # start near MAL-ED's cut-off (median Cq 31 vs 17 at presentation,
    # Vellore), so most are never caught by a monthly stool; MAL-ED's 2% excess
    # of crude over consecutive-detection infections pins both short. In
    # campaign v4 both were set by re-scoring the verified runs (the layer
    # runs after the simulation, so they change nothing else): the selection
    # had put the symptomatic window at 15.5 days, where a monthly stool often
    # catches an infection already detected in its diarrhoeal stool and
    # crude : consecutive infections reach 1.24 (MAL-ED 1.02). 8 seeds at
    # 5 / 3 days: 1.06, and 73% of detected infections attributable to
    # diarrhoea - the shortest symptomless window inside MAL-ED's site range
    # (0.24-0.74; 2 days gave 0.78). See docs/calibration_v4_adults.md.
    detectable_shedding_days_symptomatic: float = Field(
        5.0,
        description="Observation layer only: mean days a symptomatic infection's stool tests positive "
                    "at MAL-ED's cut-off. Set by re-scoring, campaign v4, against MAL-ED's crude : "
                    "consecutive infection ratio (v3: 5.84).",
        gt=0.0,
    )
    detectable_shedding_days_asymptomatic: float = Field(
        3.0,
        description="Observation layer only: mean days a symptomless infection's stool tests positive "
                    "at MAL-ED's cut-off. Set by re-scoring, campaign v4: the shortest window putting "
                    "P(attributable | detected) inside MAL-ED's site range (v3: 3.75).",
        gt=0.0,
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
    # Environmental. The three route parameters are history-matched (see
    # RotavirusConfig's CALIBRATION note). At the v4 default 53% of
    # infections are zoonotic, 46% food-borne and 0.2% faecal-oral (v3: 83%,
    # 17%, <1%): the faecal-oral route carries almost nothing anywhere in the
    # admissible region (1.4% at its top, 0.0096, in a v2 candidate) - a
    # mechanism to revisit.
    human_animal_interaction_rate: float = Field(
        0.00327,
        description="Daily probability of zoonotic-route exposure for an animal-owning household. "
                    "History-matched, campaign v4, against a 0.5-0.8 zoonotic share (0.53 at the "
                    "default; v3: 0.00117).",
    )
    # Fecal-oral (household) route
    fecal_oral_prob: float = Field(
        0.000744,
        description="Per-contact fecal-oral transmission probability within an infected household. "
                    "History-matched, campaign v4 (v3: 0.00174) but not identified: the route carries "
                    "<2% of infections anywhere in the search range.",
    )
    # Background risk due to food consumption
    food_borne_prob: float = Field(
        0.00513,
        description="Daily background infection probability via food-borne exposure. "
                    "History-matched, campaign v4 (v3: 0.000427).",
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
    poultry_roam_sigma_m: float | None = Field(
        55.0, description="Poultry roam radius in metres; overrides poultry_roam_sigma when set. 55 m "
                          "= 1 cell of the original grid (campaign v2's value).", gt=0.0,
    )
    ruminant_roam_sigma_m: float | None = Field(
        110.0, description="Ruminant roam radius in metres; overrides ruminant_roam_sigma when set. "
                           "110 m = 2 cells of the original grid.", gt=0.0,
    )

    # --- Infection -> disease (MAL-ED; see abm/evidence/maled.py) ---
    # Before this, every campylobacter infection in a child produced an
    # episode. MAL-ED: 502 attributable episodes / 2938 infections (0-24
    # months, all eight sites) = 0.171, falling with age from 0.22 to 0.11 -
    # the bulk of childhood Campylobacter infection is subclinical.
    symptomatic_fraction: float = Field(
        0.194,
        description="P(diarrhoea | infection) at the age profile's reference (no change with "
                    "prior infections: symptomatic_reduction_per_prior_infection 0). "
                    "History-matched, campaign v4 (v3: 0.344). Through the observation layer it "
                    "gives 25% of DETECTED infections attributable to diarrhoea (MAL-ED 502/2938 = "
                    "17%; site range 2-32%).",
        ge=0.0, le=1.0,
    )
    symptomatic_fraction_age_profile: list[tuple[float, float]] | None = Field(
        default_factory=lambda: _maled_age_profile((0.22, 0.20, 0.15, 0.11), 0.171),
        description="MAL-ED Supplementary Table 5 by 6-month band, as multipliers on "
                    "symptomatic_fraction; held at the 18-24 month value beyond 24 months.",
    )
    # --- Acquired immunity and infant protection ---
    # Campylobacter keeps the shared kappa (0.15): no LMIC cohort estimate of
    # protection by infection order was found, and MAL-ED's prior-infection
    # comparison (Supplementary Table 3) is confounded by heterogeneous
    # exposure (previously infected children had HIGHER subsequent rates).
    immunity_floor: float | None = Field(
        0.30,
        description="Plateau of relative susceptibility and adults' susceptibility. ASSUMPTION: "
                    "no direct estimate; set comparable to rotavirus (0.26) and Cryptosporidium "
                    "(0.31). Calibration/GSA should treat it as uncertain.",
        ge=0.0, le=1.0,
    )
    # MAL-ED Supplementary Table 2: 0-3 month rate 0.30x the 9-12 month peak
    # (3.80 vs 12.69 per 100 child-months); a linear ramp to 1 at 10 months
    # matches that at 1.5 months when it starts at 0.18. First-pass.
    infant_susceptibility_at_birth: float = Field(
        0.0642, description="Newborn relative susceptibility. History-matched, campaign v4 (v3: 0.402; "
                            "first-pass value 0.18).",
        ge=0.0, le=1.0,
    )
    infant_susceptibility_ramp_months: float = Field(
        4.18, description="Age (months) of full susceptibility. History-matched, campaign v4 (v3: 5.06; "
                          "first-pass value 10, MAL-ED's 9-12 month peak).", ge=0.0,
    )
    symptomatic_reduction_per_prior_infection: float = Field(
        0.0,
        description="No change in P(diarrhoea | infection) with prior infections: Rogawski McQuade "
                    "et al. 2020 (MAL-ED) found no evidence of protection after Campylobacter "
                    "infection. Literature value (history_matching.V3_LITERATURE_FIXED).",
        ge=0.0,
    )
    asymptomatic_relative_infectiousness: float = Field(
        0.397,
        description="Infectiousness of a symptomless infection relative to one causing illness. "
                    "History-matched, campaign v4; shared by the three pathogens (see RotavirusConfig).",
        ge=0.0, le=1.0,
    )
    # Kept at the original model's 1.0: antibiotics (macrolides) do shorten
    # Campylobacter excretion, so treatment clearing infection is plausible
    # here as it is not for rotavirus or Cryptosporidium. But 1.0 assumes EVERY
    # successful visit involves an effective antibiotic, which is an UPPER
    # BOUND (WHO IMCI reserves antibiotics for bloody diarrhoea, and
    # fluoroquinolone resistance is common). Replace with P(effective
    # antibiotic | visit) once local prescribing data are available.
    treatment_clears_infection_prob: float = Field(
        1.0,
        description="Probability a successful care-seeking visit also ends the Campylobacter "
                    "infection. 1.0 (the original model's behaviour) assumes an effective "
                    "antibiotic at every successful visit - an upper bound pending local "
                    "prescribing data.",
        ge=0.0, le=1.0,
    )
    # Observation layer only (see PathogenConfig). MAL-ED's pattern - many
    # detected infections, few causing diarrhoea, a 30% excess of crude over
    # consecutive-detection infections - needs long detectable carriage, the
    # norm for Campylobacter in endemic settings; the rule that illness-causing
    # infections stay detectable longer is not applied to it (see
    # history_matching.v3_physical_constraint), though the v4 default meets it.
    detectable_shedding_days_symptomatic: float = Field(
        40.2,
        description="Observation layer only: mean days a symptomatic infection's stool tests positive "
                    "at MAL-ED's cut-off. History-matched, campaign v4 (v3: 14.1).",
        gt=0.0,
    )
    detectable_shedding_days_asymptomatic: float = Field(
        14.3,
        description="Observation layer only: mean days a symptomless infection's stool tests positive "
                    "at MAL-ED's cut-off. History-matched, campaign v4 (v3: 39.5).",
        gt=0.0,
    )

class CryptosporidiumConfig(PathogenConfig):
    """Parameters specific to Cryptosporidium spp.

    Four routes operate (see abm/pathogens/cryptosporidium.py for the
    mechanics). The route hierarchy the parameterisation targets is
    household person-to-person > community person-to-person > waterborne >
    zoonotic, which is what the evidence for rural sub-Saharan Africa -
    including Ghana - supports:

      * Krumkamp et al. 2021, Clin Infect Dis 72(8):1358-66. Contact-network
        tracing around 1363 children <5y with diarrhoea in Gabon, GHANA,
        Madagascar and Tanzania; 184 (13%) Cryptosporidium-positive. Versus
        negative index cases, positive ones had elevated risk of positive
        household members (RR 3.6, 95% CI 1.7-7.5) and positive neighbouring
        children (RR 2.9, 95% CI 1.6-5.1) but NOT of positive animals
        (RR 1.2, 95% CI 0.8-1.9). Identical gp60 subtypes in 36% of networks
        around positive cases vs 1% around negative ones. Neighbouring
        children had the highest prevalence of any contact group (23%).
      * Eibach et al. 2015, PLoS Negl Trop Dis 9(3):e0003551. 2232 children
        <14y, Agogo, Ashanti region, GHANA; 5.2% positive. Dominant gp60
        subtypes IIcA5G3 (29.6%), IbA13G3 (19.3%), IaA21R3 (13.6%) - all
        anthroponotically transmitted. Infections peaked at the start of the
        rainy season.
      * Sow et al. 2016, PLoS Negl Trop Dis 10(5):e0004729 (GEMS).
        C. hominis 77.8%, C. parvum 9.9%; 92% (34/37) of C. parvum isolates
        were the anthroponotic IIc family.
      * Bouzid et al. 2018, PLoS Negl Trop Dis 12(6):e0006553 (LMIC
        meta-analysis). Animal contact OR 1.98 (1.11-3.54), household
        diarrhoea OR 1.98 (1.13-3.49), open defecation OR 1.82 (1.19-2.80),
        and poor drinking-water quality OR 1.06 (0.77-1.47), NOT significant.
    """
    name: str = Field("crypto", description="Pathogen identifier.")

    # --- Natural history (literature-grounded; excluded from calibration) ---
    exposure_period: int = Field(
        7,
        description="Latent/prepatent period (days). Incubation averages ~7 days (range 2-10) "
                    "across standard public-health references. Literature-grounded - not a "
                    "tuning knob.",
    )
    recovery_rate: float = Field(
        0.0476,
        description="Daily probability of clearing infection (~21-day mean shedding). Korpe et "
                    "al. 2023 (Clin Infect Dis 76:2178-86; prospective, 100 Bangladeshi families) "
                    "measured mean shedding of 19.9 days (SD 8.3; 19 in the abstract) in index "
                    "infants, 16.1 days in children 1-5y and 16.2 days in adults (Table 4); an "
                    "Ethiopian clinical case series (n=53, <5y) found a median "
                    "of 31 days (95% CI 26-36) from diarrhoea onset. 21 days sits inside that "
                    "range. Literature-grounded; excluded from calibration for the same reason "
                    "as rotavirus/campylobacter recovery_rate.",
    )
    # --- Infection -> disease (MAL-ED; see abm/evidence/maled.py) ---
    # Revised from 0.33 (Korpe et al. 2016; Kattula et al. 2016). Those
    # figures count infections DETECTED in a diarrhoeal stool, whether or not
    # Cryptosporidium caused the diarrhoea; MAL-ED reports episodes
    # ATTRIBUTABLE to it, which is what this model's per-pathogen illness
    # episode represents: 245 attributable episodes / 1364 infections in
    # children 0-24 months = 0.180, by 6-month band 0.22, 0.20, 0.17, 0.17.
    symptomatic_fraction: float = Field(
        0.102,
        description="P(diarrhoea | FIRST infection) at the age profile's reference; later infections "
                    "x exp(-0.48) each. History-matched, campaign v4 (v3: 0.132). Through the "
                    "observation layer it gives 9% of DETECTED infections attributable to diarrhoea "
                    "(MAL-ED 245/1364 = 18%; site range 2-33%; the ~1/3 DETECTED in diarrhoeal stools in "
                    "Korpe et al. 2016 / Kattula et al. 2016 includes co-detections in diarrhoea "
                    "caused by other pathogens).",
        ge=0.0, le=1.0,
    )
    symptomatic_fraction_age_profile: list[tuple[float, float]] | None = Field(
        default_factory=lambda: _maled_age_profile((0.22, 0.20, 0.17, 0.17), 0.180),
        description="MAL-ED Supplementary Table 5 by 6-month band, as multipliers on "
                    "symptomatic_fraction; held at the 18-24 month value beyond 24 months.",
    )
    duration_scale: float = Field(
        1.0,
        description="Multiplier on the shared illness-duration window. Revised from 1.6: MAL-ED "
                    "(Supplementary Table 11) found Cryptosporidium-attributable diarrhoea in "
                    "children 0-24 months lasted no longer than rotavirus or campylobacter "
                    "diarrhoea - median 3 days (IQR 2-7) at 0-6 months, 2 days (1-3) at 18-24 "
                    "months. The ~10-day figure previously used comes from case series of "
                    "clinically presenting (often immunocompromised or adult) patients.",
        gt=0.0,   # redeclared: a subclass field replaces the base field outright in
                  # pydantic v2, so the base class's gt=0.0 is otherwise lost here and a
                  # zero or negative scale would be silently accepted.
    )
    treatment_clears_infection_prob: float = Field(
        0.0,
        description="Standard care (ORS + zinc) does not shorten oocyst shedding, and "
                    "nitazoxanide - the only licensed anti-cryptosporidial - is neither part of "
                    "IMCI diarrhoea management nor licensed under 12 months. A successful visit "
                    "ends the episode's symptoms, not the infection.",
        ge=0.0, le=1.0,
    )
    # Observation layer only (see PathogenConfig). Oocyst shedding lasts
    # ~2-3 weeks, but only part of it above MAL-ED's cut-off.
    detectable_shedding_days_symptomatic: float = Field(
        27.5,
        description="Observation layer only: mean days a symptomatic infection's stool tests positive "
                    "at MAL-ED's cut-off. History-matched, campaign v4 (v3: 11.4).",
        gt=0.0,
    )
    detectable_shedding_days_asymptomatic: float = Field(
        16.5,
        description="Observation layer only: mean days a symptomless infection's stool tests positive "
                    "at MAL-ED's cut-off. History-matched, campaign v4 (v3: 7.81).",
        gt=0.0,
    )
    symptomatic_reduction_per_prior_infection: float = Field(
        0.48,
        description="P(diarrhoea | infection) x exp(-0.48) per prior infection. Rogawski McQuade "
                    "et al. 2020 (MAL-ED): Cryptosporidium diarrhoea calibrated HR 0.35 after one "
                    "prior infection, 0.25 after two or more; divided by the protection against "
                    "infection set below (exp(-0.29 n)) these are 0.47 and 0.44 - least-squares "
                    "0.48. Literature value (history_matching.V3_LITERATURE_FIXED).",
        ge=0.0,
    )
    asymptomatic_relative_infectiousness: float = Field(
        0.397,
        description="Infectiousness of a symptomless infection relative to one causing illness. "
                    "History-matched, campaign v4; shared by the three pathogens (see RotavirusConfig).",
        ge=0.0, le=1.0,
    )
    immunity_decay_per_infection: float | None = Field(
        0.29,
        description="Per-prior-infection immunity decay, overriding the shared 0.15. Kattula et "
                    "al. 2016 found protection against infection rose with infection order but "
                    "reached only 69% after four infections; exp(-kappa*4)=0.31 gives kappa=0.29. "
                    "Reinfection is the norm (81% of children reinfected; 97% infected by age 3).",
    )
    immunity_floor: float | None = Field(
        0.31,
        description="Plateau of relative susceptibility, reached at four infections (Kattula et "
                    "al. 2016: 69% protection, the most measured). Adults have their own "
                    "susceptibility (adult_relative_susceptibility; until campaign v4 they took this "
                    "value).",
        ge=0.0, le=1.0,
    )
    adult_relative_susceptibility: float | None = Field(
        0.270,
        description="Relative susceptibility of every adult (agent aged 5+), on every route. "
                    "History-matched, campaign v4, against an adult : young-child (12-59 months) "
                    "carriage ratio of 0.5 (Krumkamp et al. 2021: qPCR prevalence among household "
                    "members about half that among neighbouring under-5s; plausible range 0.2-1.0): "
                    "0.44 at the default. Korpe et al. 2023 documented shedding in adult household "
                    "members. v3: None (= immunity_floor).",
        ge=0.0, le=1.0,   # redeclared: a subclass field replaces the base field outright
    )
    # Korpe et al. 2019 (Clin Infect Dis 68:1073): shedding in people over two
    # 0.41 times as long as in younger children (1.7 vs 4.1 weeks); Korpe et
    # al. 2023 (Clin Infect Dis 76:2178): 7.0-7.2 days at 6-17 years and 16.2
    # in adults vs 19.9 in infants (0.36, 0.81). Weighted by Ghana's age
    # structure (41.8% under 18, 2021 census), people aged 5+ shed ~0.65 times
    # as long as infants. Held at that value in campaign v4 (the pilots
    # admitted 0.42-0.84).
    adult_infectious_duration_scale: float = Field(
        0.65,
        description="Multiplier on the mean infectious period of an adult's infection (adults "
                    "recover each day with probability min(1, recovery_rate / scale)). Literature "
                    "value, Korpe et al. 2019 and 2023 weighted by Ghana's age structure. v3: 1.0.",
        gt=0.0,   # redeclared: a subclass field replaces the base field outright
    )
    # MAL-ED Supplementary Table 2: 0-3 month rate only 0.08x the 15-18 month
    # peak (0.45 vs 5.96 per 100 child-months) - the strongest early-life
    # protection of the three. A linear ramp from 0 reaching 1 at 16 months
    # gives 0.09 at 1.5 months. First-pass. Kattula et al. 2016 also found
    # exclusive breastfeeding protective. v3 and v4 both leave the
    # second-year rise too weak and the first-half-year incidence too high
    # (detected 0-6 : 6-12 months 0.63 at the v4 default vs MAL-ED 0.25, the
    # default's one target outside 3 SD); v4's ramp sits near its 36-month
    # search bound.
    infant_susceptibility_at_birth: float = Field(
        0.188, description="Newborn relative susceptibility. History-matched, campaign v4 (v3: 0.0127; "
                           "first-pass value 0).",
        ge=0.0, le=1.0,
    )
    infant_susceptibility_ramp_months: float = Field(
        34.7, description="Age (months) of full susceptibility. History-matched, campaign v4 (v3: 15.3; "
                          "first-pass value 16, MAL-ED's 15-18 month peak). Near the 36-month search "
                          "bound.", ge=0.0,
    )

    # ------------------------------------------------------------------
    # Calibrated route parameters
    # ------------------------------------------------------------------
    # History-matched with the other pathogens (see RotavirusConfig's
    # CALIBRATION note). At the v4 default, post burn-in: 35% of infections
    # household, 62% community, 0.6% waterborne, 2.8% zoonotic, 0.2%
    # imported (v3: 36%, 40%, 18%, 6.5%, <1%). The water share is not
    # constrained by any target (in v3 the water dose's admissible range
    # spanned more than an order of magnitude), so water-shock scenarios
    # should vary it.
    #
    # Superseded first-generation values (LHS search against
    # LEGACY_CRYPTO_TARGETS, experiments/calibration/run_crypto_calibration.py,
    # reproduced by experiments/legacy.py::LEGACY_V1_OVERRIDES): household
    # 0.011182, community 0.00047129, water 0.00090348 (legacy point-source
    # water model), zoonotic 0.014249.

    # --- Route 1: household person-to-person (dominant) ---
    household_transmission_prob: float = Field(
        0.0952,
        description="Daily per-susceptible-member infection probability in a household "
                    "containing an infectious member. Anchored (not merely guessed) on Korpe et "
                    "al. 2019, Clin Infect Dis 68(7):1073-9: 8-week secondary attack rate 35.8% "
                    "(19/53) urban and 7.8% (5/64) rural vs 0% in control families. Over a "
                    "~21-day infectious window, 1-(1-p)^21 = 0.36 implies p ~ 0.021, the urban end of that "
                    "range; the rural SAR implies ~0.004. History-matched, campaign v4, against a "
                    "0.3-0.6 household share (0.35 at the default; v3: 0.144). Higher than that "
                    "anchor because symptomless infections - most of Cryptosporidium's - transmit at "
                    "asymptomatic_relative_infectiousness (0.40) of the rate: from a symptomless "
                    "index the per-day probability is ~0.038. The 8-week SAR this implies should be "
                    "checked against Korpe et al.",
    )

    # --- Route 2: community person-to-person (co-location) ---
    community_infection_prob: float = Field(
        0.15,
        description="Per-exposure infection probability from each co-located infectious agent, "
                    "using the same co-location machinery as rotavirus H2H (so schools, worship, "
                    "social visits and neighbours all contribute), per close contact under the "
                    "contact cap (steering_parameters.max_contacts_per_visit). This is the route "
                    "carrying Krumkamp et al.'s neighbouring-children signal (RR 2.9; 23% prevalence "
                    "in neighbour children, the highest of any contact group). Calibrated tuning "
                    "knob. Unlike rotavirus this is a fixed value rather than a daily Gaussian "
                    "draw - one fewer stochastic knob, and nothing in the evidence motivates "
                    "day-to-day variation specifically for this route. History-matched, campaign v4 "
                    "(v3: 0.00271, before the cap). AT THE UPPER SEARCH BOUND (0.15): the best fit "
                    "may lie beyond it.",
    )

    # --- Route 3: waterborne, via a persistent oocyst reservoir ---
    # Cryptosporidium is the reason the water subsystem became per-pathogen.
    # Rotavirus's water state is binary CLEAN/CONTAMINATED with a ~5-day
    # memory; oocysts instead accumulate and decay slowly, so the model needs
    # a continuous load with its own die-off rate and a dose-response.
    dose_response_r: float = Field(
        0.018,
        description="Exponential dose-response parameter: P(infection | dose) = 1 - exp(-r*dose). "
                    "The exponential model is standard for Cryptosporidium in QMRA. Human-"
                    "challenge ID50s vary widely by isolate: 132 oocysts (C. parvum Iowa, DuPont "
                    "et al. 1995, N Engl J Med 332:855-9), 87 (Iowa), 9 (TAMU) and 1042 (UCP) "
                    "(Okhuysen et al. 1999, J Infect Dis 180:1275-81), 300 (Moredun, Okhuysen et "
                    "al. 2002), and 10-83 for C. hominis TU502 (Chappell et al. 2006, Am J Trop "
                    "Med Hyg 75:851-7) - the species dominating in Ghana. Via ID50 = ln2/r that "
                    "is r ~ 0.005-0.07; 0.018 corresponds to ID50 ~ 39 oocysts. Literature-"
                    "grounded, but note the dose below is in model units, so water_dose_scale "
                    "carries the absolute scaling (exactly as human_animal_interaction_rate does "
                    "for campylobacter's Beta-Poisson route).",
    )
    oocyst_decay_rate: float = Field(
        0.15,
        description="Daily first-order die-off rate of oocysts in a water source "
                    "(load *= exp(-k)); 0.15/day is T90 ~ 15 days. King et al. 2005, Appl "
                    "Environ Microbiol 71:3848-57 measured a 4-log10 infectivity reduction at 8 "
                    "weeks at 25C (k ~ 0.16/day) and no measurable loss at 15C over 12 weeks; Li "
                    "et al. 2009, Appl Environ Microbiol 75:1548-54 estimated 1-log10 reduction "
                    "in 4.9 days at 30C, 28.7 days at 20C and 71.5 days at 10C. Akuse is "
                    "tropical, so the warm end applies. NOTE the often-quoted '2-6 months "
                    "survival' figure refers to cool, moist conditions and does NOT apply here - "
                    "but 15 days is still ~3x rotavirus's effective water memory, which is the "
                    "qualitative difference that matters.",
    )
    oocyst_shedding_rate: float = Field(
        1.0,
        description="Oocyst load added to a water source per infectious agent present per day. "
                    "Definitionally 1.0: it fixes the reservoir's unit as 'infectious-agent-days "
                    "of shedding', so that water_dose_scale carries all the real scaling and "
                    "this is not a second, redundant knob. Change it only to rescale the units.",
    )
    water_dose_scale: float = Field(
        0.00574,
        description="Converts a source's load (infectious-person-days of shedding reaching it, "
                    "dilution-weighted) into the dose entering the exponential dose-response. The "
                    "calibrated knob for the waterborne route; absorbs oocysts shed per "
                    "person-day, the fraction mobilised by runoff, dilution, volume ingested and "
                    "treatment, none of which is separately measured for Akuse. The value depends "
                    "on the water model's units. History-matched for the catchment model, campaign v4 "
                    "(weakly identified; 0.6% of infections waterborne at the default; v3: 0.408, "
                    "18%); the legacy point-source model was calibrated at 0.00090348. Grid- and "
                    "population-size dependent.",
    )
    shock_oocyst_load: float = Field(
        100.0,
        description="LEGACY water model only. Load every water source is raised to (if currently lower) when a water-"
                    "contamination shock fires - the continuous-reservoir analogue of rotavirus's "
                    "'all clean sources become contaminated'. 100 infectious-agent-days is a few "
                    "times the typical equilibrium load. A documented assumption to be swept "
                    "(like shock_daily_prob itself), not a fitted value.",
    )

    importation_rate: float = Field(
        2e-5,
        description="Daily per-person probability of a Cryptosporidium infection acquired outside "
                    "Akuse. ASSUMPTION, as for rotavirus. See PathogenConfig.",
        ge=0.0,
    )

    # --- Route 4: zoonotic (ruminant only, small) ---
    # Reads the SAME ruminant density layer campylobacter uses (built from
    # household ownership draws, abm/environment/animal_density.py), so the two
    # pathogens see one consistent livestock landscape. Poultry are
    # deliberately excluded: birds are not hosts of C. parvum/C. hominis.
    # C. meleagridis is avian-associated but minor (0.6% of GEMS isolates).
    zoonotic_interaction_rate: float = Field(
        0.0073,
        description="Scales local ruminant density into a zoonotic dose for the exponential "
                    "dose-response. Calibrated, and expected to stay SMALL: Krumkamp et al. 2021 "
                    "found no elevated risk of positive animal contacts around positive index "
                    "children (RR 1.2, 95% CI 0.8-1.9) and concluded zoonotic transmission "
                    "contributes only marginally in rural sub-Saharan Africa. The lever exists "
                    "because C. parvum zoonotic subtypes do circulate (Oduro et al. 2024 found "
                    "C. parvum in 41% of Ghanaian animal samples) and because livestock-focused "
                    "intervention scenarios need it, not because it is expected to dominate. "
                    "History-matched, campaign v4, against a 2-15% zoonotic share (2.8% at the "
                    "default; v3: 0.0236, 6.5%).",
    )
    ruminant_weight: float = Field(
        1.0,
        description="Relative zoonotic-risk weight for ruminants within the crypto zoonotic "
                    "route. Kept at 1.0 (poultry weight is structurally 0 for this pathogen), so "
                    "this is effectively a units choice folded into zoonotic_interaction_rate.",
    )


# --- Pathogen config registry ---
# Single source of truth for pathogen-name -> config-class. Used by
# SVEIRConfig's validator (below) and, via abm/pathogens/registry.py, by model
# construction - replacing the three separate hardcoded {'rota': ..., 'campy':
# ...} dicts that previously had to be kept in agreement by hand.
PATHOGEN_CONFIG_TYPES: dict[str, type[PathogenConfig]] = {
    "rota": RotavirusConfig,
    "campy": CampylobacterConfig,
    "crypto": CryptosporidiumConfig,
}


# --- Illness Mechanics Constants ---

class IllnessMechanicsConfig(BaseModel):
    """Parameters controlling how illness severity and duration are calculated."""
    # Base severity per pathogen (0-1 scale)
    base_severity_rota: float = Field(0.4, description="Base rotavirus illness severity (0-1 scale) before age/immunity adjustments.")
    base_severity_campy: float = Field(0.3, description="Base campylobacter illness severity (0-1 scale) before age/immunity adjustments.")
    # Placed between campylobacter and rotavirus. GEMS (Sow et al. 2016) found
    # Cryptosporidium-attributable MSD more often dehydrating than other causes
    # (16.5% vs 8.3%) with an attributable extended case-fatality of 5.8%
    # (4.4-7.6) in Africa - so more severe than campylobacter - but rotavirus
    # remains the archetypal acutely-dehydrating pathogen, and much of
    # cryptosporidiosis' burden is carried by episode DURATION
    # (CryptosporidiumConfig.duration_scale) rather than peak severity. The
    # ordering is literature-grounded; the exact value is an assumption.
    base_severity_crypto: float = Field(0.35, description="Base cryptosporidiosis illness severity (0-1 scale) before age/immunity adjustments.")

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
    # Revised from 2 / 12 / 1.5 days against MAL-ED Supplementary Table 11
    # (abm/evidence/maled.py: median duration of attributable diarrhoea 2-3 days,
    # IQR within 1-7, for all three pathogens and every 6-month band to 24
    # months). The old window gave medians of 4-8 days. MAL-ED durations barely
    # change with age (3, 3, 3, 2 days) although severity falls steeply with
    # age, so duration must depend only weakly on severity: a 1-5 day window
    # (slope 4 days per unit severity, down from 10) keeps every pathogen x
    # age-band median within 1 day of MAL-ED's for the severity mix the model
    # produces with prior infections, and within 1.5 days even for first
    # episodes (highest severity; worst case rotavirus at 0-6 months, 4-5
    # days vs MAL-ED's 3, IQR 2-6). These are NATURAL (untreated) durations; see
    # ChildIllnessSystem for the realized duration once care-seeking acts.
    # Re-checked with ageing and the calibrated defaults (experiments/
    # calibration/validate_calibration.py, 0-24 months, post burn-in): median
    # natural durations rota 3.6, campy 2.6, crypto 3.0 days; realized (after
    # treatment) 3.2, 1.8, 2.9 - against MAL-ED's 2-3. Rotavirus runs ~0.5-1
    # day long (its severity is highest); not adjusted.
    duration_min_days: float = Field(1.0, description="Expected illness duration (days) at severity = 0; also the minimum duration. MAL-ED-aligned (see comment).")
    duration_max_days: float = Field(5.0, description="Expected illness duration (days) at severity = 1; also the maximum duration. MAL-ED-aligned (see comment).")
    duration_noise_std: float = Field(2.0, description="Stochastic spread (days) around the expected illness duration. MAL-ED-aligned (see comment).")
    # Hard ceiling on a single episode's duration, decoupled from
    # duration_max_days (the EXPECTED duration at severity 1). Previously the
    # two were the same number, so high-severity episodes piled up at the
    # ceiling - with a 5-day expected maximum, over half of first-episode
    # rotavirus durations in young infants would sit exactly at 5 days.
    # 14 days is the WHO boundary between acute and persistent diarrhoea; an
    # episode reaching it is an acute episode that has become persistent,
    # which this model does not represent separately. None restores the old
    # coupling (ceiling = duration_max_days).
    duration_cap_days: float | None = Field(
        14.0,
        description="Maximum duration (days) of one illness episode. 14 = WHO acute/persistent "
                    "diarrhoea boundary. None = use duration_max_days as the ceiling (original "
                    "behaviour).",
    )
    # When a newly infectious child's episode (or its absence) is decided.
    # Originally ChildIllnessSystem did it at the end of the day, after
    # transmission, so on its first infectious day every child counted as
    # symptomless - a 1-in-3 error over rotavirus's ~3-day infectious period
    # once symptomless infections are less infectious. True moves the decision
    # to straight after disease progression, before the day's transmission.
    # It changes the order of random draws, so results differ in detail (not
    # in distribution) from False, which is the original order.
    onset_before_transmission: bool = Field(
        True,
        description="Decide each newly infectious child's illness episode before the day's "
                    "transmission (True) or after it (False, original order).",
    )

# --- Observation layer ---

class ObservationConfig(BaseModel):
    """MAL-ED's surveillance design, applied to the model's log of child
    infections by experiments/observation.py so that the model is compared
    with what MAL-ED would have DETECTED rather than with every infection.
    Nothing here affects the simulation.

    MAL-ED (Amin et al. 2026, BMC Infect Dis 26:106): a surveillance stool
    every month of life and a stool at every diarrhoeal episode, tested by
    TaqMan array; a detection is Ct <= 30 (rotavirus <= 32.6); "consecutive
    detections" (child aged <= 1 month between them, no negative between)
    are one infection; every pathogen-attributable diarrhoeal episode is a
    distinct infection, absorbing detections in the 5 days before it and
    consecutive ones after it.
    """
    surveillance_start_months: int = Field(1, ge=0, description="Age (months) of the first monthly surveillance stool.")
    surveillance_end_months: int = Field(24, ge=1, description="Age (months) of the last monthly surveillance stool.")
    diarrhoeal_stool_tested_prob: float = Field(
        0.8, ge=0.0, le=1.0,
        description="Probability that a diarrhoeal episode yields a tested stool. MAL-ED's rotavirus "
                    "paper (Mohan et al. 2017) tested 79.2% of episodes; the TaqMan re-analysis does "
                    "not report the share.",
    )
    shedding_duration_shape: float = Field(
        2.0, gt=0.0,
        description="Gamma shape of the detectable-shedding duration (PathogenConfig."
                    "detectable_shedding_days_*), giving its spread around the mean. Assumption.",
    )
    consecutive_gap_months: float = Field(1.0, gt=0.0, description="MAL-ED: detections at most this far apart in age (with no negative between) are one infection.")
    incubation_shedding_days: int = Field(5, ge=0, description="MAL-ED: detections this many days before an attributable diarrhoeal stool belong to that episode's infection.")
    rng_seed_offset: int = Field(7919, description="Added to the run's seed for the observation layer's own random stream (shedding durations, stool collection), so that it never perturbs the simulation's.")


# --- Demography ---

class DemographyConfig(BaseModel):
    """Ageing, under-5 turnover and age-dependent activity.

    The modelled children are the under-5s. Without turnover the population is
    closed, so immunity accumulates without limit and incidence declines for
    as long as the model runs - there is no endemic steady state to calibrate
    against. With turnover, a child reaching child_max_age_months leaves the
    modelled under-5 population and its agent slot is reused for a newborn in
    the same household, keeping the number of under-5s, their (uniform) age
    distribution and the household structure fixed - the renewal that makes a
    stationary endemic state possible. This is a births = graduations
    approximation: n_u5 / 5 births per year, ~80 per 4000 agents against
    Ghana's ~27 per 1000 crude birth rate (~108) - close enough for a
    population whose total size is fixed anyway.
    """
    ageing: bool = Field(
        True,
        description="Advance every agent's age daily and renew children reaching "
                    "child_max_age_months as newborns. False = static ages (original model).",
    )
    child_max_age_months: float = Field(
        60.0, description="Age (months) at which a child leaves the modelled under-5 population.", gt=0.0,
    )
    # Ghana has two years of compulsory kindergarten from age 4 and widespread
    # nursery/creche attendance from age 3. Before this, every under-5 -
    # including newborns - spent ~30% of days at the nearest school.
    school_entry_age_months: float = Field(
        36.0,
        description="Youngest age (months) at which a child can attend school/early-childhood "
                    "education. Younger children stay home.",
        ge=0.0,
    )
    school_attendance_prob: float = Field(
        0.7,
        description="Probability a child slot attends once old enough. ASSUMPTION pending the "
                    "Ghana MICS 2017/18 indicator 'attendance to early childhood education, "
                    "36-59 months'.",
        ge=0.0, le=1.0,
    )
    initial_prior_infections: str = Field(
        "maled",
        description="'maled': children start with Poisson-distributed prior infection counts "
                    "whose mean is MAL-ED's cumulative age-specific incidence at their age, "
                    "extrapolated at the 21-24 month rate beyond two years. 'none': every agent "
                    "starts naive (original model).",
    )

    @field_validator("initial_prior_infections")
    @classmethod
    def _check_initial(cls, v):
        if v not in ("maled", "none"):
            raise ValueError("initial_prior_infections must be 'maled' or 'none'")
        return v


# --- Water ---

class WaterConfig(BaseModel):
    """Household drinking water, sanitation and the catchment contamination model.

    model="catchment" (default) - see abm/environment/water_network.py:
      * Every household has a PRIMARY DRINKING-WATER SOURCE TYPE, drawn from
        the shares below, and every member drinks from it every day. Surface
        users drink from their nearest canal/river reach (from the OSM water
        layer), groundwater users from their nearest well/borehole (from the
        SPRINGS water-sampling points), piped users from the treated network,
        and packaged (sachet/bottled) users from a source with no local
        contamination.
      * Each source holds a CONTINUOUS per-pathogen concentration that decays
        at the pathogen's environmental die-off rate and is loaded by faecal
        RUNOFF from infectious people's households - attenuated with
        distance and scaled by the household's sanitation - not by people
        visiting the source.
      * A contamination shock is a pulse of that loading (a flood washing
        accumulated faecal material in), which can also reach groundwater
        and the piped network to a lesser degree.
    model="legacy": the original point-source model - 11 water points,
    shedding and exposure only by people standing at them, rotavirus's binary
    CLEAN/CONTAMINATED state. Kept for reproducing earlier results.

    Source and sanitation shares: households with children under 5 in the
    DHS/MICS clusters within 30 km of Akuse, surveys from 2014 onwards (Ghana
    DHS 2014 and 2022, MIS 2016 and 2019, MICS 2017-18; 197 households in 26
    clusters; from the SPRINGS compiled survey file ghana_data.csv). JMP
    categories: piped = ImpW_Pipe; ground = ImpW_Grd (protected well,
    borehole) + UnimpW_Oth (unprotected well/spring, 0.5%); packaged =
    ImpW_Oth (sachet/bottled, rainwater, delivered); surface = UnimpW_Srf.
    """
    model: str = Field("catchment", description="'catchment' (default) or 'legacy' (original point-source model).")

    share_piped: float = Field(0.645, ge=0.0, le=1.0, description="Share of households whose primary drinking water is piped (DHS/MICS within 30 km of Akuse).")
    share_surface: float = Field(0.157, ge=0.0, le=1.0, description="Share drinking surface water (canal, river) - same survey basis.")
    share_ground: float = Field(0.061, ge=0.0, le=1.0, description="Share drinking groundwater (well, borehole) - same survey basis.")
    share_packaged: float = Field(0.137, ge=0.0, le=1.0, description="Share drinking packaged (sachet/bottled) or other improved non-local water - same survey basis.")
    source_choice_length_m: float = Field(
        500.0, gt=0.0,
        description="Households nearer a surface (or ground) source are more likely to use it: the "
                    "required number of users of each type is drawn with weight exp(-distance / "
                    "this). An assumption; the survey gives shares, not their spatial pattern.",
    )

    share_improved_onsite: float = Field(0.548, ge=0.0, le=1.0, description="Share with improved non-sewered sanitation (pit with slab, VIP, septic) - same survey basis.")
    share_sewer: float = Field(0.178, ge=0.0, le=1.0, description="Share with a sewer connection - same survey basis.")
    share_unimproved: float = Field(0.168, ge=0.0, le=1.0, description="Share with unimproved sanitation - same survey basis.")
    share_open_defecation: float = Field(0.107, ge=0.0, le=1.0, description="Share practising open defecation - same survey basis.")

    # Relative release of excreted pathogens to SURFACE water by sanitation
    # type. GloWPa-type emission models (Okaali & Hofstra 2018, J Environ Qual
    # 47:1130) treat pit latrines and septic tanks as non-sources for surface
    # water and open defecation as the diffuse source (2.5% mobilised by
    # runoff - absorbed here into each pathogen's water_dose_scale). The
    # unimproved value (shared/hanging latrines, pits without slabs, some
    # direct discharge) is an assumption between the two.
    surface_release_open_defecation: float = Field(1.0, ge=0.0, description="Relative surface-water release, open defecation (reference).")
    surface_release_unimproved: float = Field(0.1, ge=0.0, description="Relative surface-water release, unimproved sanitation (assumption).")
    surface_release_improved_onsite: float = Field(0.0, ge=0.0, description="Relative surface-water release, improved onsite sanitation (GloWPa: non-source).")
    surface_release_sewer: float = Field(0.01, ge=0.0, description="Relative surface-water release, sewer (assumption: ~99% contained/treated).")
    # Relative release to GROUNDWATER: pits are the classic threat to shallow
    # wells, which surface-water models ignore. Assumptions.
    ground_release_open_defecation: float = Field(1.0, ge=0.0, description="Relative groundwater release, open defecation (assumption).")
    ground_release_unimproved: float = Field(1.0, ge=0.0, description="Relative groundwater release, unimproved pit (assumption).")
    ground_release_improved_onsite: float = Field(0.3, ge=0.0, description="Relative groundwater release, improved pit/septic (assumption).")
    ground_release_sewer: float = Field(0.0, ge=0.0, description="Relative groundwater release, sewer (assumption).")
    child_stool_unsafe_frac: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Fraction of an infectious child's stools disposed of unsafely (released as if "
                    "by open defecation) whatever the household's sanitation. ASSUMPTION pending "
                    "the DHS 'safe disposal of child's stools' indicator; children are the main "
                    "shedders, so this matters more than household sanitation.",
    )

    runoff_length_m: float = Field(
        150.0, gt=0.0,
        description="Distance attenuation of faecal runoff to surface water: the fraction of a "
                    "household's released load reaching water falls as exp(-d / this), d = "
                    "distance to the nearest water cell. An assumption to be explored in "
                    "sensitivity analysis.",
    )
    groundwater_length_m: float = Field(
        30.0, gt=0.0,
        description="Distance attenuation of pit/faecal contamination to a well, exp(-d / this). "
                    "Of the order of the 15-30 m latrine-well setback distances in WHO guidance.",
    )
    borehole_protection: float = Field(
        0.1, ge=0.0, le=1.0,
        description="Load reaching a (deep, sealed) borehole relative to a shallow well. Assumption.",
    )
    river_dilution: float = Field(
        0.01, ge=0.0, le=1.0,
        description="Concentration in a Volta river reach relative to an irrigation-canal reach "
                    "receiving the same load - the river's discharge is orders of magnitude "
                    "larger. Assumption.",
    )
    reach_block_cells: int = Field(
        8, ge=1,
        description="Surface water is split into reaches by tiling the grid into blocks of this "
                    "many cells (8 cells ~ 450 m); each block's canal cells and river cells form "
                    "one reach each.",
    )
    river_min_width_cells: int = Field(
        5, ge=2,
        description="Natural-water cells that survive a morphological opening with a square of "
                    "this many cells are classed as river; the rest (thin features) as canal.",
    )
    reach_block_m: float | None = Field(
        442.0, gt=0.0, description="Reach block size in metres; overrides reach_block_cells when set. "
                                   "442 m = 8 cells of the original grid.",
    )
    river_min_width_m: float | None = Field(
        276.0, gt=0.0, description="Minimum river width in metres; overrides river_min_width_cells when "
                                   "set. 276 m = 5 cells of the original grid.",
    )

    shock_magnitude: float = Field(
        10.0, ge=0.0,
        description="A shock adds to every source this many days' worth of the current mean "
                    "daily surface-water loading (times the source type's vulnerability). An "
                    "assumption, like shock_daily_prob itself.",
    )
    shock_vulnerability_ground: float = Field(0.5, ge=0.0, description="Shock pulse reaching wells/boreholes relative to surface water (assumption).")
    shock_vulnerability_piped: float = Field(0.1, ge=0.0, description="Shock pulse reaching the piped network relative to surface water (assumption: pipe breaks, intermittent-supply intrusion).")
    piped_flush_rate: float = Field(
        1.0, ge=0.0,
        description="Extra daily removal of contamination from the piped network (flushing and "
                    "residual chlorine), on top of the pathogen's own die-off. Assumption.",
    )

    @field_validator("model")
    @classmethod
    def _check_model(cls, v):
        if v not in ("catchment", "legacy"):
            raise ValueError("water.model must be 'catchment' or 'legacy'")
        return v

    def source_shares(self) -> tuple[float, float, float, float]:
        total = self.share_piped + self.share_surface + self.share_ground + self.share_packaged
        return tuple(x / total for x in (self.share_piped, self.share_surface, self.share_ground, self.share_packaged))

    def sanitation_shares(self) -> tuple[float, float, float, float]:
        """(improved onsite, sewer, unimproved, open defecation), normalised."""
        v = (self.share_improved_onsite, self.share_sewer, self.share_unimproved, self.share_open_defecation)
        total = sum(v)
        return tuple(x / total for x in v)


# --- General Model Configuration ---

class GridCreationParams(BaseModel):
    """Arguments for creating the spatial grid environment."""
    method: str = Field("realistic_import", description="Grid-creation method (internal - not user-configurable).")
    grid_id: str | None = Field(None, description="ID of the cached spatial grid to load (internal - not user-configurable).")
    properties: dict | None = Field(None, description="Extra grid-creation properties (internal - not user-configurable).")
    # Where households live. "uniform": each household in a different,
    # uniformly drawn residence cell (original model) - which spreads the
    # town evenly over the whole domain, empty floodplain included.
    # "population": households allocated to residence cells in proportion to
    # the grid's GHS-POP population layer (grids/<id>/population.npz, made by
    # scripts/add_population_layer.py) for population_epoch - each cell gets
    # the integer part of its expected number of households, and the rest go to
    # distinct cells drawn in proportion to the fractional parts
    # (EnvironmentFactory); dense cells hold several households.
    household_placement: str = Field(
        "population",
        description="'population' (default since campaign v3: residence cells allocated households "
                    "in proportion to the grid's GHS-POP population layer, grids/<id>/population.npz) "
                    "or 'uniform' (one household per randomly drawn residence cell, original).",
    )
    population_epoch: int = Field(
        2025, description="GHS-POP epoch of the population layer used by household_placement='population'.",
    )
    model_config = ConfigDict(validate_default=True)

    @field_validator("household_placement")
    @classmethod
    def _check_placement(cls, v):
        if v not in ("uniform", "population"):
            raise ValueError("household_placement must be 'uniform' or 'population'")
        return v

class SteeringParamsSVEIR(BaseModel):
    """Steering parameters used within each step of the SVEIR model."""
    # Shared / Global Parameters
    prior_infection_immunity_factor: float = Field(
        0.15,
        description="Per-prior-infection decay kappa in exp(-kappa N) for pathogens without their "
                    "own immunity_decay_per_infection - currently only campylobacter (rotavirus and "
                    "Cryptosporidium have cohort-based overrides). An assumption for campylobacter.",
    )

    # Between-household heterogeneity in ENVIRONMENTAL exposure (water source
    # and handling, animal contact, food hygiene). Each household draws a
    # log-normal multiplier exp(sdlog*z - sdlog^2/2), z ~ N(0,1) - mean 1 - on
    # its members' infection risk via the environmental routes (water,
    # zoonotic, food-borne) of every pathogen. Person-to-person routes are not
    # scaled. 0 = homogeneous (original model).
    #
    # Motivated by MAL-ED Supplementary Table 3 (abm/evidence/maled.py,
    # prior_infection_rate_ratio): Campylobacter incidence among children
    # with a prior infection is 1.5x, 2.3x and 3.7x that of naive children at
    # 6-12, 12-18 and 18-24 months - the reverse of what acquired immunity
    # alone produces, and the signature of high-exposure children being the
    # ones infected early - while for rotavirus (mostly person-to-person) the
    # ratio is ~0.8-1.4 and for Cryptosporidium ~1. A shared multiplier on
    # ALL routes cannot reproduce that contrast (a test sweep pushed the
    # rotavirus ratio to ~5 by the time Campylobacter's matched); restricting
    # it to environmental routes lets the route mix do the work. (That sweep
    # predates rotavirus importation; part of the inflation it showed was
    # rotavirus's near-threshold outbreak dynamics, see RotavirusConfig.
    # importation_rate.)
    #
    # History-matched with everything else (see RotavirusConfig's CALIBRATION
    # note): 0.8 in campaign v2, 0.906 in v3, 0.224 in v4. At 0.224 the median
    # household has 0.97x the mean environmental exposure and the top 5% more
    # than 1.4x (at v3's 0.906: 0.66x and 2.9x). The low value leaves
    # Campylobacter's prior-infection rate ratio short of MAL-ED's (0.93 vs
    # 2.62 at 12-24 months, 2.5 SD) - one of the default's weaker fits.
    household_exposure_sdlog: float = Field(
        0.224,
        description="SD (log scale) of the household-level log-normal (mean 1) multiplier on "
                    "infection risk via environmental routes (water, zoonotic, food-borne), "
                    "shared by all members and pathogens. 0 = homogeneous. History-matched, campaign "
                    "v4 (see RotavirusConfig's CALIBRATION note), identified mainly by MAL-ED's "
                    "prior-infection rate ratios. v3: 0.906; v2: 0.8.",
        ge=0.0,
    )

    # Water Parameters (Shared Reservoir)
    human_to_water_infection_prob: float = Field(
        0.0001, description="LEGACY water model only: daily probability an infectious agent contaminates its local water source."
    )
    # Calibrated (see RotavirusConfig's calibration note; same search/output).
    water_to_human_infection_prob: float = Field(
        0.01272,
        description="LEGACY water model only: daily probability a susceptible agent is infected via "
                    "contaminated water. LHS-calibrated tuning knob, not independently measured for "
                    "Akuse.",
    )
    water_recovery_prob: float = Field(
        0.2, description="LEGACY water model only: daily probability a contaminated water source naturally reverts to clean."
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
    # Distances below are kept in grid cells for the original grid; a
    # "_m" twin, when set, gives the same quantity in metres and is
    # converted with the loaded grid's own cell size at initialisation
    # (SVEIRModel._resolve_metric_parameters), so that a finer grid keeps the
    # same physical meaning. None = use the cell value as is.
    social_interaction_radius_m: float | None = Field(
        276.0, description="Social-visit radius in metres; overrides social_interaction_radius when set. "
                           "276 m = 5 cells of the original ~55 m grid (campaign v2's value), so the "
                           "radius means the same on the 25 m settlement grid.", gt=0.0,
    )
    # Person-to-person co-location routes (rotavirus, Cryptosporidium
    # community) expose an agent to every infectious person in its cell. In a
    # home or a compound that is a handful of people; at a place of worship -
    # OSM maps 8 for the ~11,500 people of the settlement grid, each one 25 m
    # cell - it is 300-500, and in v3 95% of adult rotavirus infections and
    # 40% of adult Cryptosporidium ones were acquired there. Contact surveys
    # report nothing like it: 7.2 contacts a day (conversations of 5+ minutes;
    # 5.1 physical), 52% of them household members, 6.1-6.7 for under-5s
    # (le Polain de Waroux et al. 2018, rural southwest Uganda). When set, an
    # agent's exposure to members of OTHER households in its cell is that of
    # this many contacts drawn from them: the infectious weight among them is
    # scaled by min(1, cap / their number). Exposure to its own household is
    # never capped. None = everyone in the cell is a contact (original model,
    # and campaigns v1-v3).
    #
    # 3.5 = the survey's 7.2 daily contacts x 48% outside the household: the
    # cap applies to the one place an agent goes in the day phase. The
    # adult-transmission pilots fitted the rotavirus and Cryptosporidium
    # targets and the adult : child carriage ratios equally well at any cap
    # from 3 to 30, so campaign v4 held it at the survey value.
    max_contacts_per_visit: float | None = Field(
        3.5,
        description="Close contacts an agent makes with members of other households at one location "
                    "per day phase; above it, exposure scales with the share infectious rather than "
                    "their number. 3.5 from le Polain de Waroux et al. 2018 (7.2 contacts a day, 48% "
                    "outside the household). None = everyone in the cell is a contact (original "
                    "model; campaigns v1-v3).",
        gt=0.0,
    )

    # --- Care-Seeking Parameters ---
    # cost_of_care is history-matched with the transmission parameters (see
    # RotavirusConfig's JOINT CALIBRATION note) against the Ghana DHS
    # childhood-diarrhoea care-seeking rate (69.2%, 95% CI 65.6-72.8%, n=621),
    # under the supportive treatment model (one visit per episode). With one
    # decision per episode, affordability no longer binds: could_not_afford_
    # rate is ~0.2-2% across the admissible region (it was 27% under the
    # "cure" model, where an episode could prompt a visit every day), so the
    # care-seeking rate is set by the decision utility and pins cost_of_care
    # (admissible 0.044-0.175) but not daily_income_rate - the emulated rate
    # changes by 0.003 between incomes of 0.046 and 0.074. Income is therefore
    # left at its earlier value, and cost_of_care (0.12) is the emulator's
    # solution for 0.692 at that income.
    #
    # First-generation calibration (experiments/care_seeking/
    # run_care_seeking_calibration.py, "cure" model): cost 0.0702 with income
    # 0.0462, the lowest-could_not_afford_rate point of the (income, cost)
    # frontier; reproduced by experiments/legacy.py.
    cost_of_care: float = Field(
        0.12,
        description="Cost of seeking medical care, as a fraction of max household wealth. "
                    "History-matched against the Ghana DHS childhood-diarrhoea care-seeking rate "
                    "(69.2%, 95% CI 65.6-72.8%, n=621; admissible 0.044-0.175). The rate is "
                    "sensitive to this and to the treatment model, much less to income.",
    )
    treatment_success_prob: float = Field(
        0.80, description="Probability that a care-seeking visit delivers effective treatment "
                          "(under treatment_effect_model='cure', that it cures the illness)."
    )
    # --- What effective treatment DOES (CareSeekingSystem._apply_treatment_outcome) ---
    # "cure" (the original model): every active episode ends on the spot. With
    # ~70-80% of episodes treated, realized episodes then last a median of 1
    # day, against MAL-ED's 3 (Supplementary Table 11) - and standard care for
    # childhood diarrhoea does not end it.
    # "supportive" (default): what WHO/IMCI care actually achieves. ORS
    # prevents the HARM of the episode (dehydration) without ending it; zinc
    # shortens it by about half a day in children over 6 months; an effective
    # antibiotic (campylobacter only, via treatment_clears_infection_prob)
    # also clears the infection and shortens symptoms by ~1.3 days. A
    # successfully treated illness spell faces no further care-seeking
    # decisions (one visit per episode, as DHS counts care-seeking).
    treatment_effect_model: str = Field(
        "supportive",
        description="'supportive' (default): treatment reduces the episode's harm and shortens "
                    "it modestly (ORS, zinc, antibiotics - see the treatment_* parameters). "
                    "'cure': treatment ends every active episode immediately (original model).",
    )
    treatment_harm_reduction: float = Field(
        0.93, ge=0.0, le=1.0,
        description="Fraction of an episode's severity (its daily health toll, and how sick the "
                    "child appears) removed by effective treatment. 0.93: ORS may prevent 93% of "
                    "diarrhoea deaths (Munos, Fischer Walker & Black 2010, Int J Epidemiol "
                    "39:i75-87) - dehydration, not diarrhoea itself, is what ORS treats.",
    )
    treatment_duration_reduction_days: float = Field(
        0.48, ge=0.0,
        description="Days by which effective treatment shortens an episode in children aged at "
                    "least treatment_duration_min_age_months (applied by stochastic rounding). "
                    "Zinc: mean difference -11.46 h (95% CI -19.72 to -3.19) in children over 6 "
                    "months; no effect under 6 months (Lazzerini & Wanzira 2016, Cochrane "
                    "CD005436).",
    )
    treatment_duration_min_age_months: float = Field(
        6.0, ge=0.0,
        description="Youngest age (months) at which treatment shortens episodes (zinc: no effect "
                    "under 6 months).",
    )
    antibiotic_duration_reduction_days: float = Field(
        1.32, ge=0.0,
        description="Additional shortening of an episode of a pathogen whose infection the "
                    "treatment clears (treatment_clears_infection_prob; campylobacter). "
                    "Antibiotics reduce the duration of Campylobacter intestinal symptoms by 1.32 "
                    "days (95% CI 0.64-1.99; Ternhag et al. 2007, Clin Infect Dis 44:696-700).",
    )

    @field_validator("treatment_effect_model")
    @classmethod
    def _check_treatment_model(cls, v):
        if v not in ("supportive", "cure"):
            raise ValueError("treatment_effect_model must be 'supportive' or 'cure'")
        return v
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
    # daily_income_rate: NOT identified by the joint calibration (see the
    # cost_of_care note), so kept at its first-generation value.
    daily_income_rate: float = Field(
        0.0462,
        description="Daily household income, as a fraction of max wealth. First-generation "
                    "calibrated value; not identified by the current care-seeking target (see "
                    "cost_of_care), so the wealth outputs it drives are uncalibrated.",
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


class EconomyConfig(BaseModel):
    """The household economy (docs/economy_design_proposal.md).

    model="legacy" (default): the original dimensionless pooled wealth in [0, 1]
    (SteeringParamsSVEIR.daily_income_rate, daily_cost_of_living, cost_of_care),
    reproducing campaign v4 byte for byte. Nothing below is read.

    model="cedi": money in 2026 Ghana cedis (GH¢) per household per day.
      * Roles by age. Agents aged 5+ get an economic age group, 5-14, 15-64 or
        65+, from the census shares below. The first member of every household
        (its head, and the parent who decides on care) is aged 15+. Everyone aged
        15+ earns. The epidemiology still treats every agent aged 5+ as one
        "adult" group, so transmission is untouched.
      * Wealth strata. Each household gets a fixed DHS wealth quintile (Eastern
        Region's 2022 distribution across national quintiles), which sets its
        earning level.
      * Income. Earner i's full-health earning potential is y_i = ybar_q * f_h, with
        f_h log-normal (mean 1, SD earning_spread_sdlog on the log scale), one draw
        per household. ybar_q is set at initialisation so that quintile q's
        households earn, in total, (1 + savings_rate) times the GLSS 7 per-capita
        consumption of that quintile for every member. Daily income is
        y_i * H_i (health; SteeringParamsSVEIR.health_based_income) * g_h(t), less
        any working time lost to a care visit. g_h(t) = exp(e_h(t) - sd^2/2) is a
        persistent household income shock: e_h follows a stationary AR(1) on the
        log scale with SD income_shock_sd and half-life income_shock_halflife_days
        (irregular informal and farm work, lost jobs and harvests). Without it
        earnings never vary, every household fills its buffer and the economy is
        inert (pilot, 2026-10-28).
      * Spending. A household's consumption target is its full-health income
        divided by (1 + savings_rate), so a quintile consumes GLSS 7's per-capita
        level on average. Cash is a buffer: each day
        cash <- clip(cash + income - target, 0, buffer_days * target). Below the
        buffer the household saves; at the buffer it spends any excess; with no
        cash it consumes what it earns, below target (poverty). No debt (decision
        D5). Initial cash is uniform on [0, initial_buffer_fraction] of the buffer.
      * Care. Each household has a usual provider for a child's diarrhoea -
        public primary care, a public hospital, a private facility or a drug
        seller (pharmacy, chemical or drug store, shop, market or drug peddler) -
        drawn by quintile from where Ghanaian parents first sought care (DHS
        2022). A visit costs that provider's out-of-pocket price (at a public
        facility, by NHIS cover, drawn per household with insured_prob), payable
        only from cash, plus the share of the parent's working day the visit takes
        (GLSS 7 travel and waiting times), whose earnings the parent loses. The CPT decision is
        unchanged, with its wealth term
        W = clip(cash / (wealth_scale_days * target), 0, 1) - a month of needs is
        "full" wealth - and the cost of a visit, out of pocket plus lost earnings,
        in the same units. WEALTH holds W, so health recovery (x (1 + W)) and every
        wealth output read it.

    Initialisation draws from its own numpy generator (seed + rng_seed_offset)
    after every draw the original model makes, so the population, its homes,
    health, water and infection histories are those of the legacy model for the
    same seed; the income shocks have their own run-time generator too. Care
    decisions still draw from the global stream.

    Values are 2026 GH¢. Price level 2016/17 -> 2026 x 3.7 (chained CPI, World
    Bank / CEIC / Bank of Ghana); 2011 -> 2026 x 8.2 (2011 CPI 108.7 on 2010 =
    100). PLACEHOLDERS to calibrate (design §4, step 4): earning_spread_sdlog,
    savings_rate, buffer_days, income_shock_sd, income_shock_halflife_days,
    cost_primary_insured and the time a drug-seller visit takes.
    """
    model: str = Field(
        "legacy",
        description="'legacy' (default; campaign v4's dimensionless pooled wealth) or 'cedi' "
                    "(2026 Ghana cedis, roles by age, wealth quintiles; see EconomyConfig).",
    )
    rng_seed_offset: int = Field(
        104729, description="Added to the run's seed for the economy's own initialisation stream.",
    )
    age_group_shares_5plus: list[float] = Field(
        [0.225, 0.712, 0.063],
        description="Shares of agents aged 5+ who are 5-14, 15-64 and 65+. Lower Manya Krobo, 2021 "
                    "census (30.3% aged 0-14, 64.0% 15-64, 5.7% 65+; 0-4 about 10%), rescaled to the "
                    "population aged 5+.",
    )
    quintile_shares: list[float] = Field(
        [0.079, 0.197, 0.264, 0.281, 0.178],
        description="Households by national DHS wealth quintile, lowest to highest: Eastern Region, "
                    "Ghana DHS 2022 household recode, survey-weighted (1,216 households).",
    )
    consumption_per_capita_daily: list[float] = Field(
        [8.8, 17.3, 26.2, 40.0, 91.1],
        description="Per-capita consumption by quintile, GH¢ a day at 2026 prices: GLSS 7 (2016/17) "
                    "Table 10.1, GH¢872 / 1,704 / 2,581 / 3,944 / 8,987 a year, x 3.7. GLSS quintiles "
                    "rank consumption, the DHS strata assets (design §3.4).",
    )
    earning_spread_sdlog: float = Field(
        0.5, ge=0.0,
        description="SD (log scale) of the household earning factor within a quintile. PLACEHOLDER, to "
                    "calibrate against poverty incidence.",
    )
    savings_rate: float = Field(
        0.05, ge=0.0,
        description="Full-health income in excess of the consumption target, as a fraction of the "
                    "target. PLACEHOLDER.",
    )
    income_shock_sd: float = Field(
        0.3, ge=0.0,
        description="Stationary SD (log scale) of the persistent household income shock. 0 = earnings "
                    "never vary. PLACEHOLDER, to calibrate with the visit costs against care-seeking.",
    )
    income_shock_halflife_days: float = Field(
        30.0, gt=0.0,
        description="Half-life (days) of the household income shock's AR(1) process. PLACEHOLDER.",
    )
    buffer_days: float = Field(
        30.0, gt=0.0,
        description="Cash a household holds at most, in days of its consumption target; income above it "
                    "is spent. PLACEHOLDER.",
    )
    initial_buffer_fraction: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Initial cash is uniform on [0, this] times the buffer. The burn-in removes its "
                    "influence.",
    )
    wealth_scale_days: float = Field(
        30.0, gt=0.0,
        description="Days of consumption that count as full wealth (W = 1) in the care-seeking utility "
                    "(design §3.6: 'can this household cover a month of needs?').",
    )
    provider_shares_by_quintile: list[list[float]] = Field(
        [[0.472, 0.141, 0.019, 0.368], [0.377, 0.127, 0.024, 0.473], [0.212, 0.134, 0.040, 0.615],
         [0.238, 0.222, 0.097, 0.443], [0.206, 0.339, 0.091, 0.364]],
        description="Usual provider by wealth quintile (lowest first): shares of public primary care (CHPS, "
                    "health centre, government clinic, outreach), public hospital (incl. polyclinic), private "
                    "facility and drug seller (pharmacy or drug store, shop or market, drug peddler, "
                    "traditional, other). Ghana DHS 2022, place first sought advice or treatment for a child "
                    "under five with diarrhoea (h44a), survey-weighted; n = 286 / 205 / 126 / 87 / 58 "
                    "children with care sought. The poorest use primary care most, the richest hospitals.",
    )
    insured_prob: float = Field(
        0.9, ge=0.0, le=1.0,
        description="Probability a household holds NHIS cover: DHS 2022, 93% of women in Eastern Region, "
                    "flat by wealth (88-95%). GLSS 7's valid-card share (47.8%) is the low-cover case.",
    )
    cost_primary_insured: float = Field(
        10.0, ge=0.0,
        description="Out-of-pocket cost, GH¢, of a public primary-care visit with NHIS cover. ASSUMPTION, to "
                    "calibrate within [0, cost_hospital_insured]: NHIS covers consultation and essential "
                    "medicines, and CHPS compounds and health centres are mostly within walking distance.",
    )
    cost_hospital_insured: float = Field(
        55.0, ge=0.0,
        description="Out-of-pocket cost, GH¢, of a public-hospital visit with NHIS cover: transport and drugs "
                    "bought outside (GH¢6.65 per outpatient episode in 2011, Kusi et al. 2015, x 8.2; GLSS 7 "
                    "Table 4.8 gives GH¢12.89 for transport alone in the rural forest zone, x 3.7 = 48).",
    )
    cost_facility_uninsured: float = Field(
        210.0, ge=0.0,
        description="Out-of-pocket cost, GH¢, of a public-facility visit without cover (GH¢25.50 per outpatient "
                    "episode in 2011, Kusi et al. 2015, x 8.2).",
    )
    cost_private: float = Field(
        210.0, ge=0.0,
        description="Out-of-pocket cost, GH¢, of a private-facility visit, insured or not. ASSUMPTION: the "
                    "uninsured public cost.",
    )
    cost_drug_seller: float = Field(
        7.5, ge=0.0,
        description="Cost, GH¢, of ORS and zinc from a drug seller: 2 ORS sachets (GH¢1.47 each) and 10 "
                    "zinc tablets (GH¢0.17 each) at NHIS Medicines List prices (March 2025), x 1.05 to "
                    "2026, x 1.5 retail margin (ASSUMPTION).",
    )
    public_visit_day_fraction: float = Field(
        0.36, ge=0.0, le=1.0,
        description="Share of an 8-hour working day a public-facility visit takes; the parent loses that "
                    "share of the day's earnings. GLSS 7: about 44 minutes' travel to and from a facility "
                    "in Eastern Region (Table 4.9) and about 131 minutes at a public facility (Table 4.10; "
                    "category midpoints, 240 minutes for '181+').",
    )
    private_visit_day_fraction: float = Field(
        0.29, ge=0.0, le=1.0,
        description="Same for a private facility: about 44 minutes' travel and 97 minutes at the facility "
                    "(GLSS 7 Tables 4.9-4.10).",
    )
    drug_seller_visit_day_fraction: float = Field(
        0.1, ge=0.0, le=1.0,
        description="Share of the parent's working day a visit to a drug seller takes. ASSUMPTION.",
    )
    poverty_line_daily: float = Field(
        17.9, ge=0.0,
        description="Upper poverty line, GH¢ per person a day (GLSS 7: GH¢1,760.80 a year per adult "
                    "equivalent in 2016/17, x 3.7). Output threshold only.",
    )
    extreme_poverty_line_daily: float = Field(
        10.0, ge=0.0,
        description="Extreme (food) poverty line, GH¢ per person a day (GLSS 7: GH¢982.10 a year, x 3.7). "
                    "Output threshold only.",
    )
    model_config = ConfigDict(validate_default=True, validate_assignment=True)

    @field_validator("model")
    @classmethod
    def _check_model(cls, v):
        if v not in ("legacy", "cedi"):
            raise ValueError("economy.model must be 'legacy' or 'cedi'")
        return v

    @field_validator("age_group_shares_5plus", "quintile_shares")
    @classmethod
    def _check_shares(cls, v, info):
        n = 3 if info.field_name == "age_group_shares_5plus" else 5
        # rounded published shares may sum to 0.999; they are normalised where used
        if len(v) != n or any(x < 0 for x in v) or abs(sum(v) - 1.0) > 0.01:
            raise ValueError(f"{info.field_name}: {n} non-negative shares summing to 1")
        return v

    @field_validator("provider_shares_by_quintile")
    @classmethod
    def _check_providers(cls, v):
        if len(v) != 5 or any(len(r) != 4 or any(x < 0 for x in r) or abs(sum(r) - 1.0) > 0.01 for r in v):
            raise ValueError("provider_shares_by_quintile: 5 rows of 4 non-negative shares summing to 1")
        return v

    @field_validator("consumption_per_capita_daily")
    @classmethod
    def _check_consumption(cls, v):
        if len(v) != 5 or any(x <= 0 for x in v):
            raise ValueError("consumption_per_capita_daily: 5 positive values, lowest quintile first")
        return v


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
    pathogens: List[Union[RotavirusConfig, CampylobacterConfig, CryptosporidiumConfig]] = Field(
        default_factory=lambda: [RotavirusConfig(), CampylobacterConfig(), CryptosporidiumConfig()],
        description="Which pathogens are active in this simulation.",
    )

    # Demographic Parameters
    average_household_size: float = Field(
        3.2, description="Mean household size (Poisson-distributed). Empirically sourced from Ghana census/DHS-MICS-style survey data."
    )
    child_probability: float = Field(
        0.145, description="Probability a non-first household member is a child under five. The first "
                           "member is always aged 5+, so the realised under-five share is "
                           "0.145 x 2.2 / 3.2 = 0.10, close to the about 10% estimated for Lower "
                           "Manya Krobo from the 2021 census (docs/economy_design_proposal.md §3.2)."
    )

    # Parameters for Agent Personas
    num_agent_personas: int = Field(32, description="Number of distinct behavioral (CPT) personas sampled via Latin Hypercube (internal - not user-configurable).")
    alpha_range: list[float] = Field([0.1, 0.9], description="Sampling range for the CPT wealth/health utility weight (alpha). Plausible, exploratory range - not fit to measured risk preferences.")
    gamma_range: list[float] = Field([0.4, 0.9], description="Sampling range for the CPT probability-distortion parameter (gamma). Plausible, exploratory range - not fit to measured risk preferences.")
    lambda_range: list[float] = Field([1.0, 3.0], description="Sampling range for the CPT loss-aversion parameter (lambda). Plausible, exploratory range - not fit to measured risk preferences.")

    demography: DemographyConfig = Field(default_factory=DemographyConfig, description="Ageing, under-5 turnover and age-dependent activity.")
    water: WaterConfig = Field(default_factory=WaterConfig, description="Household water sources, sanitation and the catchment contamination model.")
    observation: ObservationConfig = Field(default_factory=ObservationConfig, description="MAL-ED surveillance design for the observation layer (analysis only).")
    economy: EconomyConfig = Field(default_factory=EconomyConfig, description="Household economy: 'legacy' dimensionless wealth (default) or 2026 Ghana cedis.")

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
        return [
            PATHOGEN_CONFIG_TYPES[p['name']](**p) if isinstance(p, dict) else p
            for p in v
        ]

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

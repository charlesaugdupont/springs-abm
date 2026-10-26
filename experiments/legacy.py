# experiments/legacy.py
"""
Configuration overrides that reproduce the FIRST-GENERATION model.

The disease-scale, demography and water work (MAL-ED symptomatic fractions
and durations, treatment that does not clear rotavirus/Cryptosporidium,
ageing and under-5 turnover, age-specific susceptibility, the catchment water
model, importation) all default ON. Each was added behind a switch whose "off"
value reproduces the earlier code path exactly. LEGACY_V1_OVERRIDES sets every
switch to "off", so that

  * experiments/perf/regression_check.py --legacy-v1 can prove, byte for byte
    against a golden captured from the first-generation three-pathogen model,
    that the refactors underneath those features changed nothing else; and
  * the superseded calibration scripts (run_calibration.py,
    run_crypto_calibration.py) can still reproduce the fits they produced.

Kept in its own module (rather than in regression_check.py, where it started)
because that module pins torch to one thread at import.
"""

#: Dot-path overrides that switch every disease-scale, demography and water
#: feature back to the first-generation (v1, three-pathogen) model.
#: Running the harness with --legacy-v1 against a golden captured from that
#: model proves the refactors underneath those features - age-structured
#: bookkeeping, the susceptibility function, the symptomatic-probability
#: function, the decoupled duration cap, the treatment-clearance switch, the
#: demography system - are exact no-ops when the features are off.
LEGACY_V1_OVERRIDES: dict = {
    "demography.ageing": False,
    "demography.initial_prior_infections": "none",
    "demography.school_entry_age_months": 0.0,
    "demography.school_attendance_prob": 1.0,
    "illness_mechanics.duration_min_days": 2.0,
    "illness_mechanics.duration_max_days": 12.0,
    "illness_mechanics.duration_noise_std": 1.5,
    "illness_mechanics.duration_cap_days": None,
    "illness_mechanics.onset_before_transmission": False,
    "pathogens[crypto].duration_scale": 1.6,
}
for _p, _frac in (("rota", 1.0), ("campy", 1.0), ("crypto", 0.33)):
    LEGACY_V1_OVERRIDES.update({
        f"pathogens[{_p}].symptomatic_fraction": _frac,
        f"pathogens[{_p}].symptomatic_fraction_age_profile": None,
        f"pathogens[{_p}].treatment_clears_infection_prob": 1.0,
        f"pathogens[{_p}].immunity_floor": None,
        f"pathogens[{_p}].infant_susceptibility_at_birth": 1.0,
        f"pathogens[{_p}].infant_susceptibility_ramp_months": 0.0,
    })
LEGACY_V1_OVERRIDES["pathogens[rota].immunity_decay_per_infection"] = None   # shared 0.15
LEGACY_V1_OVERRIDES.update({
    "water.model": "legacy",
    "pathogens[rota].importation_rate": 0.0,
    "pathogens[crypto].importation_rate": 0.0,
    "pathogens[crypto].water_dose_scale": 0.00090348,   # v1 calibrated value (legacy water units)
    "steering_parameters.household_exposure_sdlog": 0.0,
    "steering_parameters.treatment_effect_model": "cure",
    # First-generation calibrated values of every parameter the joint
    # recalibration (experiments/calibration/history_matching.py) may move,
    # pinned so that a new default can never silently break reproduction.
    "pathogens[rota].infection_prob_mean": 0.01263,
    "pathogens[campy].human_animal_interaction_rate": 0.0048,
    "pathogens[campy].fecal_oral_prob": 0.0074,
    "pathogens[campy].food_borne_prob": 0.00186,
    "pathogens[crypto].household_transmission_prob": 0.011182,
    "pathogens[crypto].community_infection_prob": 0.00047129,
    "pathogens[crypto].zoonotic_interaction_rate": 0.014249,
    "steering_parameters.cost_of_care": 0.0702,
    "steering_parameters.daily_income_rate": 0.0462,
})


#: Dot-path overrides that reproduce the SECOND-GENERATION model: the joint
#: history-matching calibration on the original grid (history_matching.py
#: campaign "v2", 4000 agents; the model as committed in 40fb731). The
#: settlement-grid recalibration (campaign "v3") changed the defaults of
#: every parameter listed here and switched on mechanisms v2 did not have
#: (illness decided before the day's transmission, reduced infectiousness of
#: symptomless infections, illness probability falling with prior
#: infections, population-weighted household placement, distances in
#: metres). These overrides pin all of it back, so that
#:   * the v2 calibration and its verification runs reproduce, and
#:   * regression_check --v2 matches a golden captured from 40fb731 byte for
#:     byte on the original grid.
#: The observation layer's detectability values are included for
#: completeness (they do not affect the simulation).
V2_OVERRIDES: dict = {
    "illness_mechanics.onset_before_transmission": False,
    "spatial_creation_args.household_placement": "uniform",
    "steering_parameters.social_interaction_radius_m": None,
    "pathogens[campy].poultry_roam_sigma_m": None,
    "pathogens[campy].ruminant_roam_sigma_m": None,
    "water.reach_block_m": None,
    "water.river_min_width_m": None,
    "pathogens[rota].infection_prob_mean": 0.0261,
    "pathogens[rota].water_dose_scale": 0.0438,
    "pathogens[rota].importation_rate": 0.00042,
    "pathogens[rota].symptomatic_fraction": 0.643,
    "pathogens[rota].infant_susceptibility_at_birth": 0.44,
    "pathogens[rota].infant_susceptibility_ramp_months": 7.5,
    "pathogens[rota].detectable_shedding_days_symptomatic": 14.0,
    "pathogens[rota].detectable_shedding_days_asymptomatic": 3.0,
    "pathogens[campy].human_animal_interaction_rate": 0.0036,
    "pathogens[campy].fecal_oral_prob": 0.000884,
    "pathogens[campy].food_borne_prob": 0.00243,
    "pathogens[campy].symptomatic_fraction": 0.171,
    "pathogens[campy].infant_susceptibility_at_birth": 0.18,
    "pathogens[campy].infant_susceptibility_ramp_months": 10.0,
    "pathogens[campy].detectable_shedding_days_symptomatic": 14.0,
    "pathogens[campy].detectable_shedding_days_asymptomatic": 10.0,
    "pathogens[crypto].household_transmission_prob": 0.0257,
    "pathogens[crypto].community_infection_prob": 0.00221,
    "pathogens[crypto].water_dose_scale": 0.329,
    "pathogens[crypto].zoonotic_interaction_rate": 0.0147,
    "pathogens[crypto].symptomatic_fraction": 0.18,
    "pathogens[crypto].infant_susceptibility_at_birth": 0.0,
    "pathogens[crypto].infant_susceptibility_ramp_months": 16.0,
    "pathogens[crypto].detectable_shedding_days_symptomatic": 21.0,
    "pathogens[crypto].detectable_shedding_days_asymptomatic": 14.0,
    "steering_parameters.household_exposure_sdlog": 0.8,
    "steering_parameters.cost_of_care": 0.12,
    "steering_parameters.daily_income_rate": 0.0462,
}
for _p in ("rota", "campy", "crypto"):
    V2_OVERRIDES[f"pathogens[{_p}].asymptomatic_relative_infectiousness"] = 1.0
    V2_OVERRIDES[f"pathogens[{_p}].symptomatic_reduction_per_prior_infection"] = 0.0
    # adults at the children's floor, with children's infectious period
    V2_OVERRIDES[f"pathogens[{_p}].adult_relative_susceptibility"] = None
    V2_OVERRIDES[f"pathogens[{_p}].adult_infectious_duration_scale"] = 1.0
# everyone in a cell is a contact
V2_OVERRIDES["steering_parameters.max_contacts_per_visit"] = None

# The first-generation model predates everything V2_OVERRIDES pins as well:
# every key LEGACY_V1_OVERRIDES does not set itself takes its v2 value, which
# is what the code defaults were when the --legacy-v1 regression was last
# shown byte-identical (before the v3 defaults were adopted).
for _k, _v in V2_OVERRIDES.items():
    LEGACY_V1_OVERRIDES.setdefault(_k, _v)

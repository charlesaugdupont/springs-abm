"""The config defaults are campaign v5's adopted candidate (e7 at 3 significant
figures, validated out of sample with validate_calibration --campaign v5), on
v5's base settings, and every value v5 changed has its v4 value pinned in
experiments.legacy.V4_OVERRIDES."""
import re

from config import SVEIRCONFIG
from experiments.calibration.history_matching import V5_BASE_OVERRIDES, V5_PARAMS
from experiments.legacy import V2_OVERRIDES, V3_OVERRIDES, V4_OVERRIDES

#: e7 (verify_v5b) at 3 significant figures, as validated (validation_values.json)
ADOPTED_V5 = {
    "pathogens[rota].infection_prob_mean": 0.673,
    "pathogens[rota].adult_relative_susceptibility": 0.167,
    "pathogens[rota].water_dose_scale": 0.00456,
    "pathogens[rota].importation_rate": 0.000191,
    "pathogens[rota].symptomatic_fraction": 0.388,
    "pathogens[rota].infant_susceptibility_at_birth": 0.407,
    "pathogens[rota].infant_susceptibility_ramp_months": 12.8,
    "pathogens[rota].detectable_shedding_days_symptomatic": 4.33,
    "pathogens[rota].detectable_shedding_days_asymptomatic": 2.44,
    "pathogens[rota].post_infection_refractory_days": 49.1,
    "pathogens[campy].human_animal_interaction_rate": 0.00351,
    "pathogens[campy].fecal_oral_prob": 0.000165,
    "pathogens[campy].food_borne_prob": 0.00388,
    "pathogens[campy].symptomatic_fraction": 0.057,
    "pathogens[campy].infant_susceptibility_at_birth": 0.838,
    "pathogens[campy].infant_susceptibility_ramp_months": 7.36,
    "pathogens[campy].detectable_shedding_days_symptomatic": 53.3,
    "pathogens[campy].detectable_shedding_days_asymptomatic": 7.66,
    "pathogens[crypto].household_transmission_prob": 0.0924,
    "pathogens[crypto].community_infection_prob": 0.104,
    "pathogens[crypto].adult_relative_susceptibility": 0.283,
    "pathogens[crypto].water_dose_scale": 0.0418,
    "pathogens[crypto].zoonotic_interaction_rate": 0.000436,
    "pathogens[crypto].symptomatic_fraction": 0.277,
    "pathogens[crypto].infant_susceptibility_at_birth": 0.0471,
    "pathogens[crypto].infant_susceptibility_ramp_months": 12.0,
    "pathogens[crypto].detectable_shedding_days_symptomatic": 16.9,
    "pathogens[crypto].detectable_shedding_days_asymptomatic": 15.3,
    "pathogens[rota].asymptomatic_relative_infectiousness": 0.385,
    "pathogens[campy].asymptomatic_relative_infectiousness": 0.385,
    "pathogens[crypto].asymptomatic_relative_infectiousness": 0.385,
    "steering_parameters.household_exposure_sdlog": 0.695,
    "economy.income_shock_sd": 0.438,
    "economy.income_shock_halflife_days": 180.0,
    "economy.buffer_days": 83.9,
    "economy.savings_rate": 0.0,
    "economy.earning_spread_sdlog": 0.203,
    "economy.cost_primary_insured": 20.9,
    "economy.buffer_wealth_elasticity": 0.377,
}


def _get(path):
    obj = SVEIRCONFIG
    for part in path.split("."):
        m = re.match(r"(\w+)\[(\w+)\]", part)
        obj = next(p for p in getattr(obj, m.group(1)) if p.name == m.group(2)) if m else getattr(obj, part)
    return obj


def test_defaults_are_the_adopted_v5_values():
    assert {k: _get(k) for k in ADOPTED_V5} == ADOPTED_V5
    # every searched dimension of v5 is among them (tied paths split)
    searched = {part for k in V5_PARAMS for part in k.split("|")}
    assert searched == set(ADOPTED_V5)


def test_defaults_run_v5s_base_settings_except_vaccination():
    """v5 was calibrated with rotavirus vaccination off (the constant hazard at
    rate 0); the default is vaccination by schedule as in Akuse today (the
    user's decision, 2026-11-05), whose hazard rate is unused."""
    differ = {k: (_get(k), v) for k, v in V5_BASE_OVERRIDES.items() if _get(k) != v}
    assert differ == {"pathogens[rota].vaccination_model": ("schedule", "hazard"),
                      "pathogens[rota].vaccination_rate": (0.001, 0.0)}


def test_every_changed_default_is_pinned_for_earlier_generations():
    assert set(V4_OVERRIDES) == set(ADOPTED_V5) | {"economy.model", "pathogens[rota].vaccination_model",
                                                    "pathogens[rota].vaccine_counts_as_prior_infection"}
    assert V4_OVERRIDES["economy.model"] == "legacy"
    assert all(V4_OVERRIDES[k] != v for k, v in ADOPTED_V5.items()), "v5 changed every one of them"
    for older in (V3_OVERRIDES, V2_OVERRIDES):
        assert set(V4_OVERRIDES) <= set(older)

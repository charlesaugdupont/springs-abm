# experiments/gsa/gsa_metrics.py
"""
The GSA metrics function + the output-metric taxonomy.

One sweep computes ALL observables (per-metric index analysis is cheap post-hoc,
so a single expensive sweep serves every output). gsa_metrics_fn is a top-level,
picklable function composed from the shared experiments.metrics building blocks -
the same ones calibration and the other experiments use, so results stay
comparable.

HEADLINE_METRICS drive the UI-facing parameter-importance ranking (the user
chose epidemic burden + care-seeking). ALSO_REPORTED are still computed and get
their own per-metric sensitivity tables, they just don't feed the headline
aggregate.
"""
from __future__ import annotations

from experiments.metrics import (
    epidemic_metrics,
    care_seeking_metrics,
    wellbeing_metrics,
    calibration_metrics,
)


def gsa_metrics_fn(model) -> dict:
    """All GSA observables for one finished run (top-level -> picklable)."""
    out = {}
    out.update(epidemic_metrics(model))       # *_peak_u5_prevalence, *_cumulative_u5_days,
                                              #   *_extinct, *_attack_rate(_u5), n_u5
    out.update(care_seeking_metrics(model))   # episode_care_seeking_rate, could_not_afford_rate, ...
    out.update(wellbeing_metrics(model))      # mean_final_health, mean_household_wealth, mean_parent_wealth
    out.update(calibration_metrics(model))    # *_episodes_per_child_year, campy_*_fraction
    return out


# Epidemic burden + care-seeking (the user's chosen headline set).
#
# Revised 2026-10-20 to the burn-in-aware metrics (experiments.metrics.
# age_window_rates; 730-day burn-in), so a GSA run must be at least ~1460
# days long (run_gsa.py's default). The earlier whole-run keys
# (*_peak_u5_prevalence, *_cumulative_u5_days, *_episodes_per_child_year)
# included the seeded initial epidemic wave of the first-generation model and,
# for *_episodes_per_child_year, counted infections rather than episodes; they
# are still computed and listed in ALSO_REPORTED for continuity.
PATHOGENS = ("rota", "campy", "crypto")
HEADLINE_METRICS = (
    # Disease burden: attributable episodes per under-5 child-year ...
    [f"{p}_symptomatic_episodes_per_child_year_0_60m" for p in PATHOGENS]
    # ... the fraction of under-5s ill on an average day ...
    + [f"{p}_mean_u5_symptomatic_prevalence_post_burnin" for p in PATHOGENS]
    # ... and infection (carriage) burden.
    + [f"{p}_infections_per_child_year_0_60m" for p in PATHOGENS]
    + ["episode_care_seeking_rate", "could_not_afford_rate"]
)

# Every headline metric is now a per-child rate, a prevalence or a fraction,
# so none is mechanically proportional to the number of under-5s (the reason
# the absolute *_cumulative_u5_days metrics used to be excluded here).
TIER_METRICS = list(HEADLINE_METRICS)

# Computed and reported, but not folded into the headline importance aggregate.
ALSO_REPORTED = (
    [f"{p}_infections_per_child_year_0_24m" for p in PATHOGENS]
    + [f"{p}_symptomatic_episodes_per_child_year_0_24m" for p in PATHOGENS]
    + [f"{p}_mean_u5_prevalence_post_burnin" for p in PATHOGENS]
    + [
        "campy_zoonotic_fraction_post_burnin",
        "crypto_household_fraction_post_burnin",
        "crypto_community_fraction_post_burnin",
        "crypto_water_fraction_post_burnin",
        "crypto_zoonotic_fraction_post_burnin",
        "rota_water_fraction_post_burnin",
        "mean_household_wealth",
        "mean_final_health",
        "rota_extinct",
        "campy_extinct",
        "crypto_extinct",
        # first-generation (whole-run) keys, for continuity
        "rota_peak_u5_prevalence",
        "campy_peak_u5_prevalence",
        "crypto_peak_u5_prevalence",
        "rota_cumulative_u5_days",
        "campy_cumulative_u5_days",
        "crypto_cumulative_u5_days",
        "rota_episodes_per_child_year",
        "campy_episodes_per_child_year",
    ]
)

ALL_GSA_METRICS = HEADLINE_METRICS + ALSO_REPORTED

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
HEADLINE_METRICS = [
    "rota_peak_u5_prevalence",
    "campy_peak_u5_prevalence",
    "rota_cumulative_u5_days",
    "campy_cumulative_u5_days",
    "rota_episodes_per_child_year",
    "campy_episodes_per_child_year",
    "episode_care_seeking_rate",
    "could_not_afford_rate",
]

# Computed and reported, but not folded into the headline importance aggregate.
ALSO_REPORTED = [
    "campy_zoonotic_fraction",
    "mean_household_wealth",
    "mean_final_health",
    "rota_extinct",
    "campy_extinct",
]

ALL_GSA_METRICS = HEADLINE_METRICS + ALSO_REPORTED

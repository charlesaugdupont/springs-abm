# experiments/care_seeking/targets.py
"""
Empirical target for the care-seeking/economics calibration.

Single source of truth for the DHS-derived target band that
run_care_seeking_calibration.py scores episode_care_seeking_rate against - see
experiments/calibration/targets.py for the analogous epidemic-transmission
targets, and the care_seeking_empirical_target memory for how this figure was
sourced.

The Ghana 2014 DHS reports a single point estimate (69.2%, n=621 weighted
under-5 diarrhea cases) rather than a stated range, unlike the epidemic
targets which are literature ranges. The band here is instead the survey's
own 95% CI on that proportion (normal approximation to a binomial CI,
1.96 * sqrt(p*(1-p)/n)), so "in range" means "statistically consistent with
the Ghana DHS estimate" rather than an arbitrary tolerance.
"""
import math

DHS_GHANA_CARE_SEEKING_RATE = 0.692
DHS_GHANA_N = 621

# Pooled sub-Saharan Africa estimate (95% CI), reported directly by its
# source study rather than derived here - shown alongside the Ghana-specific
# target as context, not used in the loss.
DHS_SSA_RATE_CI = (0.5539, 0.6204)

_se = math.sqrt(DHS_GHANA_CARE_SEEKING_RATE * (1 - DHS_GHANA_CARE_SEEKING_RATE) / DHS_GHANA_N)

TARGETS = {
    "episode_care_seeking_rate": (
        round(DHS_GHANA_CARE_SEEKING_RATE - 1.96 * _se, 4),
        round(DHS_GHANA_CARE_SEEKING_RATE + 1.96 * _se, 4),
    ),
}

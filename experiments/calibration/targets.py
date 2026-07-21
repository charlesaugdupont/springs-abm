# experiments/calibration/targets.py
"""
Empirical target ranges for SPRINGS-ABM calibration.

Single source of truth for the literature-derived plausibility bands that both
the OAT sensitivity sweep (sensitivity.py, a per-parameter visual check) and
the LHS calibration search (run_calibration.py, a joint scored search) hold
model output against - so a parameter set that looks good under one lens
stays comparable under the other.

Source: sub-Saharan Africa / Ghana literature; GEMS study; WHO rotavirus
bulletins. These are literature RANGES, not point estimates fit to an
observed time series - treat calibration as "land inside the plausible
band", not curve-fitting. Revisit these if better local data (e.g. an Akuse-
specific surveillance estimate) becomes available.
"""

TARGETS = {
    "rota_episodes_per_child_year":  (1.5,  2.5),
    "campy_episodes_per_child_year": (1.0,  3.0),
    "rota_peak_prevalence":          (0.02, 0.10),
    "campy_peak_prevalence":         (0.02, 0.15),
    "campy_zoonotic_fraction":       (0.50, 0.80),
}

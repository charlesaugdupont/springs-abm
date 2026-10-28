# experiments/calibration/history_matching.py
"""
Joint calibration of all three pathogens and the care-seeking economics by
history matching (Craig et al. 1997; Vernon, Goldstein & Bower 2010; Andrianakis
et al. 2015 for an epidemic-model application).

Why history matching rather than "minimise a loss"
--------------------------------------------------
The targets are not measurements of Akuse: they are MAL-ED's pooled estimates
for comparable settings, with a between-site spread that is as large as the
signal (Campylobacter infection rates span 0.6-2.1 per child-year across the
high-burden sites). The scientific question is therefore not "which single
parameter set fits best" but "which parameter sets are NOT RULED OUT given
that uncertainty". History matching answers exactly that: it discards the
implausible part of parameter space in waves, using cheap statistical
emulators of the model, and reports the non-implausible region (NROY), from
which a representative default can be chosen and whose extent is the honest
uncertainty to carry into a GSA.

Implausibility of parameter set x for target j (on the target's scale -
log for rates and rate ratios):

    I_j(x) = |E[f_j(x)] - z_j| / sqrt(V_obs,j + V_md,j + V_em,j(x))

V_obs is the target's uncertainty (between-site variance for MAL-ED rates, the
band for route fractions, the survey CI for DHS care-seeking); V_md a model-
discrepancy allowance; V_em the emulator's predictive variance (which includes
the replicate-to-replicate stochasticity). x is non-implausible if
max_j I_j(x) < CUTOFF (3, Pukelsheim's 3-sigma rule).

Usage
-----
    # wave 1: Latin hypercube over the full box
    python -m experiments.calibration.history_matching --wave 1 --n 160 --reps 2
    # wave k: emulate all previous waves, sample the current NROY region
    python -m experiments.calibration.history_matching --wave 2 --n 100 --reps 2
    # report NROY, the best simulated points and write best_params.json
    python -m experiments.calibration.history_matching --report
    # emulator-optimal default candidates, then verify them on independent seeds
    python -m experiments.calibration.history_matching --select
    python -m experiments.calibration.history_matching --verify --reps 8 \
        --candidates-json experiments/outputs/history_matching/emulator_candidates.json --tag verify_final

Campaigns
---------
Every option above takes --campaign NAME (default "v2"), which selects the
grid, population, parameter box, targets and output folder:

  v2  the 4000-agent calibration on the original grid against MAL-ED's
      DETECTED infections read as true ones (waves 1-5, adopted in config.py).
      Kept exactly reproducible.
  v3  the data-defined Akuse settlement (25 m grid, GHS-POP households on
      OSM buildings, ~11,500 agents), scored through the MAL-ED observation
      layer (experiments/observation.py) on incidence, its age shape and
      infection-history structure, with the model's undetected infections
      anchored by the Vellore and Mexico City cohorts. Output in
      experiments/outputs/history_matching_v3. See CAMPAIGNS["v3"].
  adult_pilot  one wave on the v3 settlement: adults' own susceptibility
      (rotavirus, Cryptosporidium), Cryptosporidium adults' shorter
      shedding, and the person-to-person transmission children then need,
      against the v3 child targets plus adult : young-child carriage ratios.
      Analysed by experiments/calibration/adult_pilot_report.py; output in
      experiments/outputs/adult_pilot. See ADULT_PILOT_PARAMS.
  adult_pilot_cap  the same with a cap on close contacts per place visited
      (steering_parameters.max_contacts_per_visit), shared by both
      pathogens; output in experiments/outputs/adult_pilot_cap. See
      ADULT_PILOT_CAP_PARAMS.
  v4  v3 recalibrated with the contact cap (3.5) and adults' own
      susceptibility, against v3's targets plus the carriage ratios, with
      v3's final selection rules from the start. Output in
      experiments/outputs/history_matching_v4. See V4_PARAMS.
  econ  the household economy in cedis (economy.model = "cedi") on the v4
      epidemiology: six economic parameters against DHS 2022 care-seeking by
      wealth quintile. Run the waves with --criterion max (five targets, all
      must fit). Verification also reports the v4 targets (econ_epi_check).
      Output in experiments/outputs/history_matching_econ. See ECON_PARAMS.

A parameter path may tie several config fields to one value ("a|b|c"): it is
one dimension of the box, and every field is set to it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import qmc

from abm.evidence import maled, rotavirus_cohorts
from experiments.calibration import targets as T
from experiments.legacy import V2_OVERRIDES, V3_OVERRIDES
from experiments.metrics import calibration_metrics, care_seeking_metrics
from experiments.observation import observe
from experiments.orchestrator import SweepSpec, run_sweep

GRID_ID = "7d9ce7c720a6"
AGENTS = 4000
STEPS = 1460          # 730-day burn-in (experiments.metrics.DEFAULT_BURN_IN_DAYS) + 2 years measured
OUT_DIR = os.path.join("experiments", "outputs", "history_matching")
CUTOFF = 3.0

#: The first wave run with the current treatment model
#: (treatment_effect_model="supportive"). Wave 1 used the original "cure"
#: model, which changes the care-seeking rate (more decision days per
#: episode) but not the epidemiological outputs, so wave 1 still guides where
#: to sample for the epidemiological targets, while the care-seeking target -
#: and every final judgement - uses only waves from this one on.
FIRST_CURRENT_WAVE = 2
TREATMENT_DEPENDENT = {"episode_care_seeking_rate"}

#: MAL-ED targets describe natural history in UNVACCINATED sites, so the
#: calibration world has no rotavirus vaccination (see targets.py).
BASE_OVERRIDES = {"pathogens[rota].vaccination_rate": 0.0}

# ---------------------------------------------------------------------------
# Parameter space: path -> (low, high, scale)
# ---------------------------------------------------------------------------
PARAMS: dict[str, tuple[float, float, str]] = {
    "pathogens[rota].infection_prob_mean":            (0.003,   0.06,  "log"),
    "pathogens[rota].water_dose_scale":               (0.02,    3.0,   "log"),
    "pathogens[campy].human_animal_interaction_rate": (0.0003,  0.012, "log"),
    "pathogens[campy].fecal_oral_prob":               (0.0005,  0.03,  "log"),
    "pathogens[campy].food_borne_prob":               (0.0001,  0.004, "log"),
    "pathogens[crypto].household_transmission_prob":  (0.003,   0.08,  "log"),
    "pathogens[crypto].community_infection_prob":     (0.00002, 0.004, "log"),
    "pathogens[crypto].water_dose_scale":             (0.02,    5.0,   "log"),
    "pathogens[crypto].zoonotic_interaction_rate":    (0.0002,  0.03,  "log"),
    "steering_parameters.household_exposure_sdlog":   (0.0,     1.2,   "lin"),
    "steering_parameters.cost_of_care":               (0.02,    0.2,   "lin"),
    "steering_parameters.daily_income_rate":          (0.03,    0.08,  "lin"),
    # Added after wave 3 (see WAVE3_FIXED): infections acquired outside the town.
    "pathogens[rota].importation_rate":               (2e-5,    3e-3,  "log"),
}
PATHS = list(PARAMS)

#: Parameters added to PARAMS after some waves had run, with the value they
#: were held at in those waves. Waves 1-3 ran with rotavirus importation fixed
#: at 2e-5/person/day - the lower bound of its range - so they are design
#: points on the lower face of the extended box, and remain valid training
#: data for the 13-parameter emulators.
#:
#: Why it was added: at every wave-3 non-implausible point, rotavirus sat near
#: its persistence threshold in a closed town of 4000 - it faded out in 1-3 of
#: 8 seeds and otherwise ran as stochastic outbreaks, and that temporal
#: clustering inflated its prior-infection rate ratio to 2-12 (MAL-ED 1.008).
#: A scan at two emulator-optimal points (experiments/outputs/history_matching/
#: import_scan) showed importation at 1e-3 removes the fade-out (between-seed
#: CV ~10%), brings the ratio to 0.91-1.0 with imported infections only 3-4%
#: of the total, and leaves the Campylobacter and Cryptosporidium outputs
#: unchanged.
WAVE3_FIXED = {"pathogens[rota].importation_rate": 2e-5}
FIRST_IMPORTATION_WAVE = 4

#: Targets whose outputs the wave-3 scan showed do not depend on the rotavirus
#: block (infection probability, water dose, importation). Wave 4 conditions
#: only on these, evaluated with importation at its wave-3 value - the one
#: value the earlier emulators have seen - and samples the rotavirus block
#: freely, since importation moves where rotavirus's plausible region is.
ROTA_INDEPENDENT_TARGETS = {
    "campy_infections_per_child_year_0_24m", "campy_symptomatic_episodes_per_child_year_0_24m",
    "campy_prior_infection_rate_ratio_12_24m", "crypto_infections_per_child_year_0_24m",
    "crypto_symptomatic_episodes_per_child_year_0_24m", "crypto_prior_infection_rate_ratio_12_24m",
    "campy_zoonotic_fraction_post_burnin", "crypto_household_fraction_post_burnin",
    "crypto_zoonotic_fraction_post_burnin",
}
ROTA_BLOCK = ("pathogens[rota].infection_prob_mean", "pathogens[rota].water_dose_scale",
              "pathogens[rota].importation_rate")


def to_unit(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for j, p in enumerate(PATHS):
        lo, hi, sc = PARAMS[p]
        v = values[:, j]
        out[:, j] = (np.log(v / lo) / np.log(hi / lo)) if sc == "log" else (v - lo) / (hi - lo)
    return out


def from_unit(u: np.ndarray) -> np.ndarray:
    out = np.empty_like(u, dtype=float)
    for j, p in enumerate(PATHS):
        lo, hi, sc = PARAMS[p]
        out[:, j] = lo * (hi / lo) ** u[:, j] if sc == "log" else lo + u[:, j] * (hi - lo)
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

@dataclass
class HMTarget:
    metric: str
    z: float              # observed value, on the transformed scale
    sd_obs: float         # observation / setting uncertainty, transformed scale
    sd_md: float          # model-discrepancy allowance, transformed scale
    transform: str        # "log" or "lin"
    eps: float = 0.0      # added before log, so that a fade-out maps to a finite value

    def tx(self, v):
        v = np.asarray(v, dtype=float)
        return np.log(v + self.eps) if self.transform == "log" else v


def build_targets() -> list[HMTarget]:
    out: list[HMTarget] = []
    for p in ("rota", "campy", "crypto"):
        for kind, eps in (("infections", 0.01), ("symptomatic_episodes", 0.003)):
            m = f"{p}_{kind}_per_child_year_0_24m"
            out.append(HMTarget(m, math.log(T.ANCHORS[m] + eps), T.LOG_SD[m], 0.10, "log", eps))
        # MAL-ED Supplementary Table 3, pooled 12-24 months. sd 0.4 on the log
        # scale (~ +/-50%): the published counts give a sampling SD of ~0.1-0.15,
        # widened for between-site heterogeneity (not reported). The ratio also
        # responds to how a pathogen persists: in a closed town, stochastic
        # outbreaks cluster infections in time and inflate it - which is what
        # identified rotavirus importation (see WAVE3_FIXED).
        rr = maled.prior_infection_rate_ratio(p, 12, 24)
        out.append(HMTarget(f"{p}_prior_infection_rate_ratio_12_24m", math.log(rr), 0.40, 0.10, "log", 0.0))
    for m, (lo, hi) in (("campy_zoonotic_fraction_post_burnin", T.TARGETS["campy_zoonotic_fraction_post_burnin"]),
                        ("crypto_household_fraction_post_burnin", T.CRYPTO_TARGETS["crypto_household_fraction_post_burnin"]),
                        ("crypto_zoonotic_fraction_post_burnin", T.CRYPTO_TARGETS["crypto_zoonotic_fraction_post_burnin"])):
        out.append(HMTarget(m, 0.5 * (lo + hi), (hi - lo) / 4.0, 0.02, "lin"))
    # Ghana DHS childhood-diarrhoea care-seeking: 69.2% (95% CI 65.6-72.8, n=621).
    out.append(HMTarget("episode_care_seeking_rate", 0.692, 0.036 / 1.96, 0.03, "lin"))
    return out


TARGETS_HM = build_targets()


def hm_metrics(model) -> dict:
    out = calibration_metrics(model)
    out.update(care_seeking_metrics(model))
    return out


# ---------------------------------------------------------------------------
# Campaign v3: the Akuse settlement grid, scored through the observation layer
# ---------------------------------------------------------------------------

#: Grid from scripts/make_settlement_grid (the GHS-POP settlement domain at
#: 25 m, with 4,580 OSM buildings) and its population layer
#: (scripts/add_population_layer). AGENTS = the 2025 GHS-POP population on
#: its residence cells.
V3_GRID_ID = "6855dd54618c"
V3_AGENTS = 11_476

#: Distances that were set in cells of the original ~55 m grid, now in
#: metres so they mean the same on the 25 m grid (5, 1, 2, 8 and 5 cells).
V3_BASE_OVERRIDES = {
    "pathogens[rota].vaccination_rate": 0.0,
    "spatial_creation_args.household_placement": "population",
    "spatial_creation_args.population_epoch": 2025,
    "steering_parameters.social_interaction_radius_m": 276.0,
    "pathogens[campy].poultry_roam_sigma_m": 55.0,
    "pathogens[campy].ruminant_roam_sigma_m": 110.0,
    "water.reach_block_m": 442.0,
    "water.river_min_width_m": 276.0,
}

_RHO = "|".join(f"pathogens[{p}].asymptomatic_relative_infectiousness" for p in ("rota", "campy", "crypto"))

#: 30 dimensions (28 in wave 1, see V3_WAVE1_FIXED). Transmission ranges come from a scan on the settlement grid
#: at the v2 values (all person-to-person routes x 0.03/0.1/0.3/1: true
#: rotavirus infections 0.13/0.20/1.5/10.8 per child-year, Cryptosporidium
#: 0.16/0.13/0.64/3.3, Campylobacter ~1.8 throughout since its animal and food
#: routes dominate), widened upwards because a low infectiousness of
#: symptomless infections (rho) needs proportionally more transmission.
#: symptomatic_fraction is P(episode) for a FIRST infection at the age-profile
#: reference; detectable_shedding_days_* act only on the observation layer.
V3_PARAMS: dict[str, tuple[float, float, str]] = {
    "pathogens[rota].infection_prob_mean":                        (0.002,   0.06,  "log"),
    "pathogens[rota].water_dose_scale":                           (0.002,   0.3,   "log"),
    "pathogens[rota].importation_rate":                           (1e-5,    1e-3,  "log"),
    "pathogens[rota].symptomatic_fraction":                       (0.15,    0.9,   "lin"),
    "pathogens[rota].symptomatic_reduction_per_prior_infection":  (0.0,     1.5,   "lin"),
    "pathogens[rota].infant_susceptibility_at_birth":             (0.05,    1.0,   "lin"),
    "pathogens[rota].infant_susceptibility_ramp_months":          (3.0,     12.0,  "lin"),
    "pathogens[rota].detectable_shedding_days_symptomatic":       (3.0,     21.0,  "log"),
    "pathogens[rota].detectable_shedding_days_asymptomatic":      (0.5,     14.0,  "log"),
    "pathogens[campy].human_animal_interaction_rate":             (0.0005,  0.02,  "log"),
    "pathogens[campy].fecal_oral_prob":                           (0.0001,  0.01,  "log"),
    "pathogens[campy].food_borne_prob":                           (0.0003,  0.01,  "log"),
    "pathogens[campy].symptomatic_fraction":                      (0.03,    0.6,   "log"),
    "pathogens[campy].symptomatic_reduction_per_prior_infection": (0.0,     1.0,   "lin"),
    "pathogens[campy].infant_susceptibility_at_birth":            (0.05,    1.0,   "lin"),
    "pathogens[campy].infant_susceptibility_ramp_months":         (3.0,     14.0,  "lin"),
    "pathogens[campy].detectable_shedding_days_symptomatic":      (3.0,     60.0,  "log"),
    "pathogens[campy].detectable_shedding_days_asymptomatic":     (2.0,     60.0,  "log"),
    "pathogens[crypto].household_transmission_prob":              (0.001,   0.3,   "log"),
    "pathogens[crypto].community_infection_prob":                 (0.0001,  0.005, "log"),
    "pathogens[crypto].water_dose_scale":                         (0.01,    1.0,   "log"),
    "pathogens[crypto].zoonotic_interaction_rate":                (0.001,   0.05,  "log"),
    "pathogens[crypto].symptomatic_fraction":                     (0.05,    0.8,   "log"),
    "pathogens[crypto].symptomatic_reduction_per_prior_infection": (0.0,    2.0,   "lin"),
    "pathogens[crypto].infant_susceptibility_at_birth":           (0.0,     0.8,   "lin"),
    "pathogens[crypto].infant_susceptibility_ramp_months":        (6.0,     24.0,  "lin"),
    "pathogens[crypto].detectable_shedding_days_symptomatic":     (5.0,     30.0,  "log"),
    "pathogens[crypto].detectable_shedding_days_asymptomatic":    (3.0,     30.0,  "log"),
    _RHO:                                                         (0.1,     1.0,   "log"),
    "steering_parameters.household_exposure_sdlog":               (0.0,     1.5,   "lin"),
}


#: Changes after wave 1 (300 points x 2 seeds; experiments/outputs/
#: history_matching_v3/wave1):
#:  * crypto household_transmission_prob upper bound 0.05 -> 0.15: the
#:    household share of Cryptosporidium infections (target 0.30-0.60) was
#:    reached only at the top of the range, with rho high and the other
#:    routes low (median 0.04; 6% of points within 3 SD) - symptomless
#:    infections that are less infectious weaken the household route most.
#:  * Campylobacter and Cryptosporidium infant_susceptibility_ramp_months
#:    added (held at 10 and 16 months in wave 1): the best wave-1 points
#:    missed Cryptosporidium's age profile (0-6 vs 6-12 month ratio, second-
#:    year rise, age at first detection), which the ramp sets.
#: Wave-1 points are design points on the slice ramp = held value. Wave 2
#: therefore judges only V3_RAMP_INDEPENDENT_TARGETS, evaluated at the held
#: ramps, and samples the ramps freely (as v2's wave 4 did for importation).
#: Changes after wave 2 (the best-fitting 10% of runs, 58 of them from
#: wave 2, sat against a range edge in >25% of cases):
#:  * Campylobacter detectable_shedding_days_{symptomatic,asymptomatic}
#:    upper bound 30 -> 60 days: MAL-ED's 30% excess of crude over
#:    consecutive-detection infections needs infections that stay detectable
#:    across consecutive monthly stools. Observation layer only.
#:  * crypto household_transmission_prob upper bound 0.15 -> 0.3 (median of
#:    the best runs 0.06, a quarter above 0.095): symptomless infections,
#:    most of Cryptosporidium's, transmit at rho (median 0.46) of the rate.
#: Waves 1-2 lie inside the widened box, so they remain training data.
V3_WAVE1_FIXED = {"pathogens[campy].infant_susceptibility_ramp_months": 10.0,
                  "pathogens[crypto].infant_susceptibility_ramp_months": 16.0}


def build_targets_v3() -> list[HMTarget]:
    """Observed (MAL-ED, through the observation layer) and true (cohort)
    targets. SDs for the age-shape and history ratios are judgements: the
    pooled MAL-ED counts give sampling SDs of 0.04-0.13 on the log scale, but
    MAL-ED does not report these ratios by site, so 0.25-0.4 allows for
    between-site variation of the size seen in the overall rates (0.4-1.0)."""
    out: list[HMTarget] = []
    for p in ("rota", "campy", "crypto"):
        pre = f"{p}_obs_"
        for kind, anchor_key, eps in (("infections", "infections", 0.01),
                                      ("attributable_episodes", "symptomatic_episodes", 0.003)):
            a = f"{p}_{anchor_key}_per_child_year_0_24m"
            out.append(HMTarget(pre + f"{kind}_per_child_year_0_24m", math.log(T.ANCHORS[a] + eps),
                                T.LOG_SD[a], 0.10, "log", eps))
        out.append(HMTarget(pre + "infant_ratio_0_6_vs_6_12m", math.log(maled.infant_ratio(p) + 0.02),
                            0.25, 0.10, "log", 0.02))
        out.append(HMTarget(pre + "second_to_first_year_ratio",
                            math.log(maled.second_to_first_year_ratio(p) + 0.02), 0.25, 0.10, "log", 0.02))
        out.append(HMTarget(pre + "naive_decline_15_24_vs_6_12m", math.log(maled.naive_decline(p) + 0.02),
                            0.30, 0.10, "log", 0.02))
        out.append(HMTarget(pre + "prior_infection_rate_ratio_12_24m",
                            math.log(maled.prior_infection_rate_ratio(p, 12, 24) + 0.02), 0.40, 0.10, "log", 0.02))
        out.append(HMTarget(pre + "crude_to_consecutive_ratio", maled.crude_to_consecutive_ratio(p),
                            0.03, 0.02, "lin"))
        # eps 0.1 month: in a run with almost no detections the median can
        # be 0 (one child positive at its first stool) - 1 of 600 wave-1 runs
        out.append(HMTarget(pre + "median_age_first_detected_infection_months",
                            math.log(maled.median_age_at_first_infection_months(p) + 0.1), 0.20, 0.07, "log", 0.1))
    # TRUE rotavirus infection (Vellore, fortnightly stools + serology)
    out.append(HMTarget("rota_true_frac_infected_by_6m", rotavirus_cohorts.VELLORE_FRAC_INFECTED_BY_6M,
                        0.10, 0.05, "lin"))
    out.append(HMTarget("rota_true_primary_share_0_36m", rotavirus_cohorts.VELLORE_PRIMARY_SHARE,
                        0.07, 0.03, "lin"))
    for m, (lo, hi) in (("campy_zoonotic_fraction_post_burnin", T.TARGETS["campy_zoonotic_fraction_post_burnin"]),
                        ("crypto_household_fraction_post_burnin", T.CRYPTO_TARGETS["crypto_household_fraction_post_burnin"]),
                        ("crypto_zoonotic_fraction_post_burnin", T.CRYPTO_TARGETS["crypto_zoonotic_fraction_post_burnin"])):
        out.append(HMTarget(m, 0.5 * (lo + hi), (hi - lo) / 4.0, 0.02, "lin"))
    return out


#: Targets judged in v3's wave 2 (see V3_WAVE1_FIXED): everything except the
#: Campylobacter and Cryptosporidium age-shape and infection-history ratios.
V3_RAMP_DEPENDENT = {f"{p}_obs_{k}" for p in ("campy", "crypto") for k in (
    "infant_ratio_0_6_vs_6_12m", "second_to_first_year_ratio", "naive_decline_15_24_vs_6_12m",
    "prior_infection_rate_ratio_12_24m", "median_age_first_detected_infection_months")}
V3_RAMP_INDEPENDENT_TARGETS = {t.metric for t in build_targets_v3()} - V3_RAMP_DEPENDENT


#: Emulator inputs (from wave 4). Each pathogen's outputs depend almost only on
#: its own parameters and the two shared ones (rho, exposure heterogeneity),
#: so each target is emulated on those 11-12 inputs instead of all 30.
#: Checked out of sample - emulators fitted on waves 1-2 predicting the 300
#: wave-3 points: lower RMSE for 26 of 29 targets and a smaller latent SD
#: (e.g. crypto 0-6 vs 6-12 month ratio 0.22 -> 0.20 RMSE, 0.29 -> 0.20 SD),
#: and ~6x faster to fit. The three targets that predicted better with every
#: input keep them: both crude/consecutive ratios (a Campylobacter or
#: rotavirus infection is also caught in the diarrhoeal stool of another
#: pathogen's episode) and the Campylobacter naive decline.
V3_SHARED_PARAMS = [_RHO, "steering_parameters.household_exposure_sdlog"]
V3_ALL_INPUT_TARGETS = {"rota_obs_crude_to_consecutive_ratio", "campy_obs_crude_to_consecutive_ratio",
                        "campy_obs_naive_decline_15_24_vs_6_12m"}


def v3_emulator_inputs(metric: str) -> list[str]:
    if metric in V3_ALL_INPUT_TARGETS:
        return list(V3_PARAMS)
    p = metric.split("_")[0]
    return [q for q in V3_PARAMS if q.startswith(f"pathogens[{p}].") and "|" not in q] + V3_SHARED_PARAMS


#: Parameters set from the literature for the final selection (after wave 4)
#: rather than chosen by fit: the targets leave them weakly identified (the
#: emulated admissible ranges after wave 4 are given in brackets), and the
#: emulator-optimal point had put rotavirus's at 0.21, below every estimate.
#: lambda multiplies P(illness | infection) by exp(-lambda) per prior infection.
V3_LITERATURE_FIXED = {
    # Velazquez et al. 1996 (Mexico City, weekly stools + serology): P(diarrhoea
    # | infection) after 1, 2, 3 infections relative to a first infection 0.37,
    # 0.42, 0.24 (abm/evidence/rotavirus_cohorts.py); least-squares fit of
    # exp(-lambda n): 0.50. MAL-ED's calibrated hazard ratios (Rogawski McQuade
    # et al. 2020, J Infect Dis 222:1858; diarrhoea 0.53, infection 0.74 after
    # one prior infection) give 0.33 - a lower bound, since MAL-ED's monthly
    # stools missed most prior infections and so diluted the contrast. [0.07-1.46]
    "pathogens[rota].symptomatic_reduction_per_prior_infection": 0.50,
    # Rogawski McQuade et al. 2020: "no evidence of protection after
    # infections with the other bacterial pathogens", C. jejuni/coli included. [0.03-0.98]
    "pathogens[campy].symptomatic_reduction_per_prior_infection": 0.0,
    # Rogawski McQuade et al. 2020: Cryptosporidium diarrhoea cHR 0.35 after one
    # prior infection, 0.25 after two or more; divided by the model's
    # protection against infection, exp(-0.29 n) (Kattula et al. 2016), these
    # are 0.47 (n = 1) and 0.44 (n = 2): least-squares lambda 0.48. [0.16-1.94]
    "pathogens[crypto].symptomatic_reduction_per_prior_infection": 0.48,
}


#: Detected infections per child-year, 0-24 months, across MAL-ED's
#: high-burden sites (targets.py): rota 0.22-0.71, campy 0.61-2.07, crypto
#: 0.35-0.88. The calibration's SD on log incidence (0.40-0.53, from the same
#: sites) admits values below every site at implausibility ~2; the first
#: literature-anchored selection (verify_v3_lit) landed there for
#: Campylobacter (0.34-0.37) and Cryptosporidium (0.18-0.19). A town whose
#: detected incidence matches no comparable site is a poor basis for burden
#: and intervention estimates, so the final selection can require the range.
V3_INCIDENCE_SITE_RANGES = {f"{p}_obs_infections_per_child_year_0_24m": T._MALED_BANDS[f"{p}_infections_per_child_year_0_24m"]
                            for p in ("rota", "campy", "crypto")}

#: P(attributable diarrhoea | detected infection) across the same sites
#: (targets.SYMPTOMATIC_FRACTION_SITE_RANGE): rota 0.24-0.74, campy 0.02-0.32,
#: crypto 0.02-0.33. Infections and episodes are separate targets, each with a
#: wide between-site SD, so their ratio can drift far outside anything MAL-ED
#: saw while both pass (verify_v3_lit_sites: Campylobacter 0.77). Required
#: together with the incidence range in the final selection: as
#: (episodes metric, infections metric, lo, hi).
V3_ATTRIBUTABLE_FRACTION_SITE_RANGES = {
    p: (f"{p}_obs_attributable_episodes_per_child_year_0_24m", f"{p}_obs_infections_per_child_year_0_24m",
        *T.SYMPTOMATIC_FRACTION_SITE_RANGE[p])
    for p in ("rota", "campy", "crypto")
}


def v3_physical_constraint(values: pd.DataFrame) -> np.ndarray:
    """For rotavirus and Cryptosporidium, an infection that causes illness
    stays detectable at least as long as one that does not (it sheds far
    more: Vellore rotavirus, median Cq 17 at presentation vs 31 in
    symptomless infections).

    Not applied to Campylobacter. It was at first (verify_v3_lit,
    verify_v3_lit_sites), and it made Campylobacter's symptomless infections
    detectable for only 2-5 days, so that 57-77% of detected infections were
    attributable to diarrhoea against MAL-ED's 17% (site range 2-32%) and
    diarrhoea reached 0.57 episodes per child-year, above every MAL-ED site.
    MAL-ED's own pattern - many detected infections, few causing diarrhoea,
    a 30% excess of crude over consecutive-detection counts - points to long
    symptomless carriage, which in endemic settings is the norm for
    Campylobacter; the rule rested on rotavirus data and does not transfer."""
    ok = np.ones(len(values), dtype=bool)
    for p in ("rota", "crypto"):
        ok &= (values[f"pathogens[{p}].detectable_shedding_days_symptomatic"].to_numpy()
               >= values[f"pathogens[{p}].detectable_shedding_days_asymptomatic"].to_numpy())
    return ok


def hm_metrics_v3(model) -> dict:
    """True-infection metrics (calibration_metrics, incl. route fractions and
    the imported share) plus the observation layer; care-seeking is not
    calibrated in this campaign (the economy is recalibrated separately)."""
    out = calibration_metrics(model)
    out.update(observe(model))
    return out


# ---------------------------------------------------------------------------
# Adult-transmission pilot (branch adult-transmission)
# ---------------------------------------------------------------------------

#: Validation of v3 found adults (every agent aged 5+) far more often infected
#: than young children - rotavirus 6.6 true infections per person-year at 5+,
#: adult carriage ~10x that of children 12-59 months, 98% of the infectious
#: pressure - because adults sat at the children's immunity plateau and move
#: about far more. The pilot asks whether, with adults given their own
#: susceptibility (and Cryptosporidium adults a shorter infectious period),
#: the model can put adult carriage where surveys find it AND still meet the
#: child targets, which then have to be reached through transmission among
#: and from children. Rotavirus and Cryptosporidium only (Campylobacter's
#: adult carriage was already about half the children's, and its animal and
#: food routes do not depend on infectious people); every other parameter at
#: its v3 value (config.py). Rotavirus and Cryptosporidium share no parameter
#: here, so each run is a design point for both.
#:
#: Ranges from four probe runs on the settlement grid (1460 days, one seed):
#:   v3 values                     rota ratio 13.7, detected 0.26/child-yr;
#:                                 crypto ratio 1.23, detected 0.37
#:   rota adults 0.03, ipm 0.08    rota faded to imports (71% imported, 0.06)
#:   rota adults 0.03, ipm 0.3     rota ratio 2.1, detected 0.46
#:   rota adults 0.08, ipm 0.15    rota ratio 4.9, detected 0.43
#:   crypto adults 0.06, household 0.8, community 0.03, duration 0.6:
#:                                 ratio 0.38, detected 0.60
#:   crypto adults 0.12, household 0.5, community 0.012, duration 0.6:
#:                                 ratio 0.44, detected 0.41
#: Rotavirus adults sustain their own chains once ipm x adult susceptibility
#: exceeds ~0.003-0.009, so the box reaches far lower adult susceptibility
#: and higher ipm than the probes.
ADULT_PILOT_PARAMS: dict[str, tuple[float, float, str]] = {
    # v3: adults at the floor, 0.26
    "pathogens[rota].adult_relative_susceptibility":          (0.001, 0.26, "log"),
    # v3: 0.0405 (admissible 0.0047-0.048); children must now carry it
    "pathogens[rota].infection_prob_mean":                    (0.03,  1.0,  "log"),
    # v3: adults at the floor, 0.31
    "pathogens[crypto].adult_relative_susceptibility":        (0.01,  0.31, "log"),
    # v3: 0.144 and 0.00271
    "pathogens[crypto].household_transmission_prob":          (0.05,  1.0,  "log"),
    "pathogens[crypto].community_infection_prob":             (0.001, 0.05, "log"),
    # Mean shedding of people aged 5+ relative to young children: 1.7 vs 4.1
    # weeks for >2 vs <2 years (Korpe et al. 2019, Clin Infect Dis 68:1073;
    # weekly stools, 8 weeks) = 0.41; 7.0-7.2 days at 6-17 years and 16.2 days
    # in adults vs 19.9 days in infants (Korpe et al. 2023, Clin Infect Dis
    # 76:2178; weekly stools; 16 episodes at 6+) = 0.36 and 0.81 - weighted by
    # Ghana's age structure (2021 census: 41.8% under 18, so about a third of
    # those aged 5+) about 0.65.
    "pathogens[crypto].adult_infectious_duration_scale":      (0.4,   0.85, "lin"),
}

#: Adult : young-child carriage. Adults are every agent aged 5+; children
#: 12-59 months not in an illness episode, as a community stool survey
#: samples them. Model values are TRUE carriage; surveys detect, and detect
#: adults' lower-load shedding less often, so the observed ratios are if
#: anything below the true ones - part of the model-discrepancy allowance.
#:  rotavirus: antigen in symptomless people, adults ~3% vs 4.2% at 1-5 years
#:    (Endara et al. 2007, Esmeraldas, Ecuador) = 0.71; household secondary
#:    infection, adults 30+ 53% vs 67% at 0-9 years (Lopman et al. 2013,
#:    Quininde) = 0.79. Plausible 0.4-1.1: log SD 0.25.
#:  Cryptosporidium (qPCR): household members (median age 15) 4% vs
#:    neighbouring children <5 8% around children without the infection
#:    (8/215 vs 13/162; ratio 0.46, 95% CI 0.20-1.08), 13% vs 23% around
#:    infected children (0.58, 0.41-0.82), Ghana 10% vs 19% (Eibach,
#:    Krumkamp et al. 2021, Clin Infect Dis 72:1358). 0.5, log SD 0.30. The
#:    Bangladeshi cohort's 5% of adults vs 35% at 1-5 years ever infected in
#:    8 months (Korpe et al. 2023, ELISA) would be far lower; ELISA misses
#:    low-load shedding, so it is not used.
ADULT_CARRIAGE_TARGETS = {
    "rota_adult_to_child_carriage_ratio_post_burnin": (0.70, 0.25),
    "crypto_adult_to_child_carriage_ratio_post_burnin": (0.50, 0.30),
}


def build_targets_adult_pilot() -> list[HMTarget]:
    out = [t for t in build_targets_v3() if t.metric.split("_")[0] in ("rota", "crypto")]
    for m, (z, sd) in ADULT_CARRIAGE_TARGETS.items():
        out.append(HMTarget(m, math.log(z + 0.02), sd, 0.10, "log", 0.02))
    return out


def adult_pilot_emulator_inputs(metric: str) -> list[str]:
    p = metric.split("_")[0]
    return [q for q in ADULT_PILOT_PARAMS if q.startswith(f"pathogens[{p}].")]


#: Pilot 2 (after pilot 1, experiments/outputs/adult_pilot): Cryptosporidium
#: passed, rotavirus failed - none of 96 runs had both adult carriage at
#: 0.4-1.1x young children's and detected incidence in MAL-ED's site range.
#: Below ipm x adult susceptibility ~0.007 adults were barely infected and
#: children alone reached at most 0.22 detected infections per child-year;
#: above it adults ran their own epidemic (carriage ratio >= 1.3, or child
#: infections of 6-35 per child-year). Tracing v3's infections showed why: 95%
#: of adult rotavirus infections (40% of adult Cryptosporidium) were acquired at
#: one of the 8 OSM places of worship, each a single cell holding 300-500
#: people on a given day, every one of them a contact of every other
#: (steering_parameters.max_contacts_per_visit, which caps that, has the
#: evidence). Probes with the cap (one seed each; detected incidence per
#: child-year, MAL-ED site ranges rota 0.22-0.71, crypto 0.35-0.88):
#:   cap 7, v3 values                rota faded (85% imported); crypto ratio 0.64, 0.23
#:   cap 7, rota ipm 0.3; crypto household 0.3, community 0.02, duration 0.65
#:                                   rota ratio 0.46, 0.16, 20% imported; crypto 0.49, 0.30
#:   cap 15, same values             rota ratio 0.79, 0.34, 4% imported;
#:                                   crypto 0.50, 0.34, household share 0.50
#: so with the cap, adults at the children's floor may already be right; the
#: box keeps their susceptibility free below it. The cap is shared by the two
#: pathogens: each one's emulators take its own parameters and the cap.
ADULT_PILOT_CAP_PARAMS: dict[str, tuple[float, float, str]] = {
    # 7.2 contacts a day, 48% outside the household; 64% report under 10
    # casual (<5 min) contacts (le Polain de Waroux et al. 2018) - per place
    # visited, so a wide range
    "steering_parameters.max_contacts_per_visit":         (3.0,   30.0, "log"),
    "pathogens[rota].adult_relative_susceptibility":       (0.02,  0.26, "log"),
    "pathogens[rota].infection_prob_mean":                 (0.05,  1.0,  "log"),
    "pathogens[crypto].adult_relative_susceptibility":     (0.03,  0.31, "log"),
    "pathogens[crypto].household_transmission_prob":       (0.05,  1.0,  "log"),
    # the cap cuts community exposure, so higher than pilot 1
    "pathogens[crypto].community_infection_prob":          (0.001, 0.1,  "log"),
    "pathogens[crypto].adult_infectious_duration_scale":   (0.4,   0.85, "lin"),
}
ADULT_PILOT_SHARED = ["steering_parameters.max_contacts_per_visit"]


def adult_pilot_cap_emulator_inputs(metric: str) -> list[str]:
    p = metric.split("_")[0]
    return ADULT_PILOT_SHARED + [q for q in ADULT_PILOT_CAP_PARAMS if q.startswith(f"pathogens[{p}].")]


# ---------------------------------------------------------------------------
# Campaign v4: v3 with the contact cap and adults' own susceptibility
# ---------------------------------------------------------------------------

#: The adult-transmission pilots (experiments/outputs/adult_pilot,
#: adult_pilot_cap; adult_pilot_report.py) found the v3 adult epidemic to be
#: a crowd artifact, removed by capping close contacts per place visited, and
#: verified 4 of 4 points fitting every rotavirus and Cryptosporidium target
#: plus the adult : child carriage ratios (verify_both: max I 2.5-2.9 on 3
#: fresh seeds, caps 3.1-19.7). v4 recalibrates everything with that
#: structure. Held at evidence values, since the targets leave them flat:
#:  * max_contacts_per_visit 3.5: 7.2 contacts a day, 48% outside the
#:    household (le Polain de Waroux et al. 2018) - the cap applies to the one
#:    place an agent goes in the day phase. Fit was indifferent over 3-30.
#:  * Cryptosporidium adult_infectious_duration_scale 0.65: Korpe et al. 2019
#:    (0.41, >2 vs <2 years) and 2023 (0.36 at 6-17 years, 0.81 adults),
#:    weighted by Ghana's age structure; admissible 0.42-0.84 in the pilots.
#:  * lambda (symptomatic_reduction_per_prior_infection) at
#:    V3_LITERATURE_FIXED from the start, as in v3's final selection.
V4_BASE_OVERRIDES = {
    **V3_BASE_OVERRIDES,
    **V3_LITERATURE_FIXED,
    "steering_parameters.max_contacts_per_visit": 3.5,
    "pathogens[crypto].adult_infectious_duration_scale": 0.65,
}

#: 29 dimensions. Person-to-person transmission and adult susceptibility from
#: the pilots (at a cap of 3.5 their emulators admitted rotavirus ipm
#: 0.43-0.99 and adults 0.13-0.26, Cryptosporidium household 0.19-0.83 and
#: community 0.001-0.094); infant ramps, importation, water, zoonotic and the
#: shared parameters over v3's full box, since they trade off against
#: transmission; illness fractions at v3's admissible range (2.5-97.5%,
#: nroy_marginals_lit_sites_attr.csv) widened by ~10%; Campylobacter - whose
#: routes the cap and the adult changes do not touch - likewise. Cryptosporidium's
#: infant ramp may now run to 30 months: v3 and the pilots left its second-year
#: rise too weak (detected second : first year 1.1-1.3 vs MAL-ED 2.3).
#:
#: Widened after wave 2 (verify_v4: one candidate, e0, met the adoption rule -
#: max I 2.96 on 8 seeds - but the emulator-optimal points and the
#: admissible set pressed against the box: rotavirus ipm (a quarter of the
#: admissible set within 0.10-0.12 of the 0.1 bound), rotavirus infant ramp
#: (11.8 of 12 months), Cryptosporidium infant ramp (28 of 30), and the lower
#: bounds of Cryptosporidium household transmission, water dose and zoonotic
#: rate and of Campylobacter's infant susceptibility at birth; a stricter
#: selection, <= 5% imported with the household share in its band, reached
#: rho = 1, its physical bound). Waves 1-2 lie inside the widened box, so
#: they remain training data; wave 3 is the planned last wave. Bounds:
#: rotavirus ipm 0.1 -> 0.03, ramp 12 -> 18 months; Cryptosporidium ramp
#: 30 -> 36 months, household 0.02 -> 0.005, water 0.01 -> 0.002, zoonotic
#: 0.001 -> 0.0002; Campylobacter infant susceptibility at birth 0.05 -> 0.
#: Rotavirus importation keeps its 1e-3 ceiling: the imported-fraction cap
#: (MAX_IMPORTED_FRACTION) is what limits it, and rho cannot exceed 1.
V4_PARAMS: dict[str, tuple[float, float, str]] = {
    "pathogens[rota].infection_prob_mean":                        (0.03,    1.0,   "log"),
    "pathogens[rota].adult_relative_susceptibility":              (0.05,    0.26,  "log"),
    "pathogens[rota].water_dose_scale":                           (0.002,   0.3,   "log"),
    "pathogens[rota].importation_rate":                           (1e-5,    1e-3,  "log"),
    "pathogens[rota].symptomatic_fraction":                       (0.2,     0.9,   "lin"),
    "pathogens[rota].infant_susceptibility_at_birth":             (0.05,    1.0,   "lin"),
    "pathogens[rota].infant_susceptibility_ramp_months":          (3.0,     18.0,  "lin"),
    "pathogens[rota].detectable_shedding_days_symptomatic":       (3.0,     21.0,  "log"),
    "pathogens[rota].detectable_shedding_days_asymptomatic":      (0.5,     14.0,  "log"),
    "pathogens[campy].human_animal_interaction_rate":             (0.0006,  0.015, "log"),
    "pathogens[campy].fecal_oral_prob":                           (0.00015, 0.01,  "log"),
    "pathogens[campy].food_borne_prob":                           (0.0003,  0.01,  "log"),
    "pathogens[campy].symptomatic_fraction":                      (0.03,    0.4,   "log"),
    "pathogens[campy].infant_susceptibility_at_birth":            (0.0,     1.0,   "lin"),
    "pathogens[campy].infant_susceptibility_ramp_months":         (3.0,     14.0,  "lin"),
    "pathogens[campy].detectable_shedding_days_symptomatic":      (3.0,     60.0,  "log"),
    "pathogens[campy].detectable_shedding_days_asymptomatic":     (5.0,     60.0,  "log"),
    "pathogens[crypto].household_transmission_prob":              (0.005,   1.0,   "log"),
    "pathogens[crypto].community_infection_prob":                 (0.0005,  0.15,  "log"),
    "pathogens[crypto].adult_relative_susceptibility":            (0.05,    0.31,  "log"),
    "pathogens[crypto].water_dose_scale":                         (0.002,   1.0,   "log"),
    "pathogens[crypto].zoonotic_interaction_rate":                (0.0002,  0.05,  "log"),
    "pathogens[crypto].symptomatic_fraction":                     (0.04,    0.4,   "log"),
    "pathogens[crypto].infant_susceptibility_at_birth":           (0.0,     0.4,   "lin"),
    "pathogens[crypto].infant_susceptibility_ramp_months":        (6.0,     36.0,  "lin"),
    "pathogens[crypto].detectable_shedding_days_symptomatic":     (5.0,     30.0,  "log"),
    "pathogens[crypto].detectable_shedding_days_asymptomatic":    (3.0,     30.0,  "log"),
    _RHO:                                                         (0.1,     1.0,   "log"),
    "steering_parameters.household_exposure_sdlog":               (0.0,     1.5,   "lin"),
}

#: Plausible adult : young-child carriage (ADULT_CARRIAGE_TARGETS): rotavirus
#: 0.4-1.1 (Endara et al. 2007; Lopman et al. 2013), Cryptosporidium 0.2-1.0
#: (Krumkamp et al. 2021: 0.46, 95% CI 0.20-1.08). Required in the
#: final selection, as the MAL-ED site ranges are.
V4_CARRIAGE_RANGES = {
    "rota_adult_to_child_carriage_ratio_post_burnin": (0.4, 1.1),
    "crypto_adult_to_child_carriage_ratio_post_burnin": (0.2, 1.0),
}


#: Route shares that are not targets but are required of the default (from
#: the wave-3 selection on). Nothing in v4 constrained rotavirus's routes, and
#: with the contact cap removing the crowd transmission, the waterborne route
#: filled the gap: 34% of rotavirus infections in verify_v4 e0, 60-68% in the
#: first wave-3 candidates (v3: 0.2%). Rotavirus spreads mainly by direct
#: faecal-oral contact (PAHO), with similar rates in high- and low-income
#: settings, and hygiene and sanitation improvements reduce it far less than
#: bacterial or parasitic diarrhoea (Parashar et al. 2006, Emerg Infect Dis
#: 12:304). The 10% bound - a minor route - is a judgement; no published
#: estimate exists. Rotavirus's own targets do not need the water route: in
#: waves 1-3, runs with at most 5% waterborne fit them as well (best
#: rotavirus-only max I 1.51) as runs with more.
V4_ROUTE_RANGES = {"rota_water_fraction_post_burnin": (0.0, 0.10)}


def build_targets_v4() -> list[HMTarget]:
    """v3's 29 targets and the two adult : young-child carriage ratios."""
    return build_targets_v3() + [t for t in build_targets_adult_pilot() if t.metric in ADULT_CARRIAGE_TARGETS]


def v4_emulator_inputs(metric: str) -> list[str]:
    if metric in V3_ALL_INPUT_TARGETS:
        return list(V4_PARAMS)
    p = metric.split("_")[0]
    return [q for q in V4_PARAMS if q.startswith(f"pathogens[{p}].") and "|" not in q] + V3_SHARED_PARAMS


# ---------------------------------------------------------------------------
# Campaign econ: the cedi household economy, epidemiology fixed at v4
# ---------------------------------------------------------------------------

#: The economy in Ghana cedis (config.EconomyConfig, docs/economy_design_proposal.md
#: §7b) on the v4 epidemiology. Only economic parameters vary; every
#: epidemiological parameter stays at its v4 default (config.py) with v4's base
#: settings (vaccination off, as in the v4 calibration).
ECON_BASE_OVERRIDES = {**V4_BASE_OVERRIDES, "economy.model": "cedi"}

#: Six dimensions; the rest of EconomyConfig is set from data (GLSS 7, DHS 2022,
#: NHIS Medicines List) or held at its stated assumption. Ranges from the pilots
#: (§7b): shock SD 0.15-0.3 put care-seeking at 0.90-0.74, so the box reaches
#: lower; the half-life spans a week to half a year (irregular work to a bad
#: season); the buffer a week to three months of consumption; the savings
#: margin 0-15%; the earning spread within a quintile 0.2-0.8 (log SD); the
#: insured primary-care cost from free to the hospital cost (GH¢55).
ECON_PARAMS: dict[str, tuple[float, float, str]] = {
    "economy.income_shock_sd": (0.1, 0.6, "lin"),
    "economy.income_shock_halflife_days": (7.0, 180.0, "log"),
    "economy.buffer_days": (7.0, 90.0, "log"),
    "economy.savings_rate": (0.0, 0.15, "lin"),
    "economy.earning_spread_sdlog": (0.2, 0.8, "lin"),
    "economy.cost_primary_insured": (0.0, 55.0, "lin"),
}

#: Care-seeking for a child under five with diarrhoea in the two weeks before the
#: survey, by national wealth quintile: Ghana DHS 2022 (children's recode, any
#: source of advice or treatment), survey-weighted, 95% cluster-bootstrap
#: intervals (Claude outputs/economy/dhs_tables.py); n = 458 / 300 / 209 / 150 / 90
#: children with diarrhoea. SD = interval width / 3.92; model discrepancy 0.03 as
#: for the earlier single care-seeking target. Compared with the model's
#: per-episode rate in each household quintile (economy_metrics). The gradient
#: is flat: the mechanism by which richer parents can afford care more easily
#: has to be offset by where they go and what it costs (D4, D8).
ECON_CARE_TARGETS = {1: (0.630, 0.587, 0.670), 2: (0.663, 0.593, 0.715), 3: (0.550, 0.463, 0.633),
                     4: (0.547, 0.456, 0.622), 5: (0.633, 0.514, 0.749)}


def build_targets_econ() -> list[HMTarget]:
    return [HMTarget(f"econ_care_seeking_rate_q{q}", z, (hi - lo) / 3.92, 0.03, "lin")
            for q, (z, lo, hi) in ECON_CARE_TARGETS.items()]


def hm_metrics_econ(model) -> dict:
    """The economy's outputs, the overall care-seeking rate, and the v4
    epidemiological metrics (true infections, routes and the observation
    layer), so that verification can show the epidemiology has not moved."""
    from experiments.metrics import economy_metrics
    out = calibration_metrics(model)
    out.update(observe(model))
    out.update(care_seeking_metrics(model))
    out.update(economy_metrics(model))
    return out


def econ_epi_check(df: pd.DataFrame) -> pd.DataFrame:
    """For verification runs of the econ campaign: each v4 target's seed mean
    and its implausibility (as in the v4 verification), next to the v4 default's
    own fit (docs/calibration_v4_adults.md §6), so that a change in the
    epidemiology from the economy shows."""
    rows = []
    for t in build_targets_v4():
        if t.metric not in df:
            continue
        y = t.tx(df[t.metric].to_numpy(dtype=float))
        y = y[np.isfinite(y)]
        if not y.size:
            continue
        I = abs(y.mean() - t.z) / np.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + (y.var(ddof=1) / len(y) if len(y) > 1 else 0.0))
        rows.append({"target": t.metric, "model": float(np.nanmean(df[t.metric])), "I": float(I)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Campaign registry
# ---------------------------------------------------------------------------

@dataclass
class Campaign:
    name: str
    grid_id: str
    agents: int
    steps: int
    out_dir: str
    params: dict
    base_overrides: dict
    targets: list
    metrics_fn: Callable
    first_current_wave: int = 1
    treatment_dependent: set = field(default_factory=set)
    #: parameters added after some waves had run -> the value they were held at
    wave_fixed: dict = field(default_factory=dict)
    #: the wave after parameters were added (wave_fixed): it judges only
    #: conditional_targets, at the held values, and samples the new ones freely
    first_importation_wave: int | None = None
    conditional_targets: set = field(default_factory=set)
    #: write partial results every N runs and resume an interrupted wave
    checkpoint_every: int | None = None
    #: target metric -> the parameter paths its emulator uses (None: all)
    emulator_inputs: Callable | None = None
    #: (first wave, fraction): from that wave on, this share of a wave's
    #: points comes from the core of the NROY (see sample_nroy)
    core_design: tuple | None = None


CAMPAIGNS: dict[str, Campaign] = {
    # v2's runs used the defaults of their day; experiments.legacy.V2_OVERRIDES
    # pins those, so the campaign still reproduces once v3's are adopted.
    "v2": Campaign("v2", GRID_ID, AGENTS, STEPS, OUT_DIR, dict(PARAMS), {**V2_OVERRIDES, **BASE_OVERRIDES},
                   TARGETS_HM, hm_metrics, FIRST_CURRENT_WAVE, set(TREATMENT_DEPENDENT),
                   dict(WAVE3_FIXED), FIRST_IMPORTATION_WAVE, conditional_targets=set(ROTA_INDEPENDENT_TARGETS)),
    # v3's and the pilots' runs used v3's defaults, since replaced by v4's;
    # experiments.legacy.V3_OVERRIDES pins them, so these still reproduce.
    "v3": Campaign("v3", V3_GRID_ID, V3_AGENTS, 1460, os.path.join("experiments", "outputs", "history_matching_v3"),
                   V3_PARAMS, {**V3_OVERRIDES, **V3_BASE_OVERRIDES}, build_targets_v3(), hm_metrics_v3, wave_fixed=dict(V3_WAVE1_FIXED),
                   first_importation_wave=2, conditional_targets=set(V3_RAMP_INDEPENDENT_TARGETS),
                   checkpoint_every=16, emulator_inputs=v3_emulator_inputs, core_design=(4, 0.5)),
    "adult_pilot": Campaign("adult_pilot", V3_GRID_ID, V3_AGENTS, 1460,
                            os.path.join("experiments", "outputs", "adult_pilot"), ADULT_PILOT_PARAMS,
                            {**V3_OVERRIDES, **V3_BASE_OVERRIDES}, build_targets_adult_pilot(), hm_metrics_v3,
                            checkpoint_every=8, emulator_inputs=adult_pilot_emulator_inputs),
    "adult_pilot_cap": Campaign("adult_pilot_cap", V3_GRID_ID, V3_AGENTS, 1460,
                                os.path.join("experiments", "outputs", "adult_pilot_cap"), ADULT_PILOT_CAP_PARAMS,
                                {**V3_OVERRIDES, **V3_BASE_OVERRIDES}, build_targets_adult_pilot(), hm_metrics_v3,
                                checkpoint_every=8, emulator_inputs=adult_pilot_cap_emulator_inputs),
    "v4": Campaign("v4", V3_GRID_ID, V3_AGENTS, 1460, os.path.join("experiments", "outputs", "history_matching_v4"),
                   V4_PARAMS, V4_BASE_OVERRIDES, build_targets_v4(), hm_metrics_v3,
                   checkpoint_every=16, emulator_inputs=v4_emulator_inputs, core_design=(2, 0.5)),
    "econ": Campaign("econ", V3_GRID_ID, V3_AGENTS, 1460, os.path.join("experiments", "outputs", "history_matching_econ"),
                     ECON_PARAMS, ECON_BASE_OVERRIDES, build_targets_econ(), hm_metrics_econ,
                     checkpoint_every=16),
}
CAMPAIGN = "v2"
CHECKPOINT_EVERY: int | None = None
EMULATOR_INPUTS: Callable | None = None
CORE_DESIGN: tuple | None = None


def use_campaign(name: str) -> Campaign:
    """Point the module-level settings every function here reads at a campaign."""
    global CAMPAIGN, GRID_ID, AGENTS, STEPS, OUT_DIR, PARAMS, PATHS, BASE_OVERRIDES, TARGETS_HM
    global FIRST_CURRENT_WAVE, TREATMENT_DEPENDENT, WAVE3_FIXED, FIRST_IMPORTATION_WAVE, CHECKPOINT_EVERY
    global ROTA_INDEPENDENT_TARGETS, EMULATOR_INPUTS, CORE_DESIGN
    c = CAMPAIGNS[name]
    CAMPAIGN, GRID_ID, AGENTS, STEPS, OUT_DIR = c.name, c.grid_id, c.agents, c.steps, c.out_dir
    PARAMS, PATHS, BASE_OVERRIDES, TARGETS_HM = c.params, list(c.params), c.base_overrides, c.targets
    FIRST_CURRENT_WAVE, TREATMENT_DEPENDENT = c.first_current_wave, c.treatment_dependent
    WAVE3_FIXED, FIRST_IMPORTATION_WAVE = c.wave_fixed, c.first_importation_wave
    CHECKPOINT_EVERY = c.checkpoint_every
    ROTA_INDEPENDENT_TARGETS = c.conditional_targets
    EMULATOR_INPUTS, CORE_DESIGN = c.emulator_inputs, c.core_design
    return c


def _expand(combo: dict) -> dict:
    """Split tied paths ("a|b") into one entry per config field."""
    out = {}
    for k, v in combo.items():
        for part in k.split("|"):
            out[part] = v
    return out


def _param_column(df: pd.DataFrame, path: str) -> pd.Series:
    """A design parameter's column in sweep results (tied paths are stored
    expanded, so read the first field)."""
    return df[path] if path in df else df[path.split("|")[0]]


# ---------------------------------------------------------------------------
# Running waves
# ---------------------------------------------------------------------------

def _spec(name: str, reps: int, base_seed: int | None = None, n_cores: int | None = None) -> SweepSpec:
    return SweepSpec(name=name, grid_id=GRID_ID, params=[], metrics_fn=CAMPAIGNS[CAMPAIGN].metrics_fn,
                     reps=reps, steps=STEPS, agents=AGENTS, base_overrides=dict(BASE_OVERRIDES),
                     output_dir=OUT_DIR, n_cores=n_cores, base_seed=base_seed,
                     checkpoint_every=CHECKPOINT_EVERY, resume=CHECKPOINT_EVERY is not None)


def combos_from_unit(u: np.ndarray, wave: int) -> list[dict]:
    vals = from_unit(u)
    return [{"gsa.design_id": f"w{wave}_{i:04d}", **{p: float(v) for p, v in zip(PATHS, row)}}
            for i, row in enumerate(vals)]


def load_all_waves() -> pd.DataFrame:
    frames = []
    for name in sorted(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else []:
        path = os.path.join(OUT_DIR, name, "results.parquet")
        if name.startswith("wave") and os.path.exists(path):
            df = pd.read_parquet(path)
            df["wave"] = int(name[len("wave"):])
            for p, v in WAVE3_FIXED.items():
                if p not in df:
                    df[p] = v
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per design point: transformed replicate means and variances."""
    rows = []
    for design_id, g in df.groupby("gsa.design_id"):
        rec = {"gsa.design_id": design_id, "wave": int(g["wave"].iloc[0]), "n_rep": len(g)}
        for p in PATHS:
            rec[p] = float(_param_column(g, p).iloc[0])
        for t in TARGETS_HM:
            y = t.tx(g[t.metric].to_numpy(dtype=float)) if t.metric in g else np.array([np.nan])
            y = y[np.isfinite(y)]
            rec[t.metric] = float(y.mean()) if y.size else np.nan
            rec[t.metric + "__var"] = float(y.var(ddof=1)) if y.size > 1 else np.nan
            raw = g[t.metric].to_numpy(dtype=float) if t.metric in g else np.array([np.nan])
            rec[t.metric + "__raw"] = float(np.nanmean(raw)) if np.isfinite(raw).any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def simulated_implausibility(summary: pd.DataFrame) -> pd.DataFrame:
    """Implausibility of each SIMULATED point, using its replicate mean and
    variance (no emulator)."""
    out = summary.copy()
    cols = []
    for t in TARGETS_HM:
        var_ens = out[t.metric + "__var"].fillna(0.0) / out["n_rep"]
        denom = np.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + var_ens)
        col = "I__" + t.metric
        out[col] = (out[t.metric] - t.z).abs() / denom
        out.loc[out[t.metric].isna(), col] = np.inf
        cols.append(col)
    out["I_max"] = out[cols].max(axis=1)
    out["I_second"] = out[cols].apply(lambda r: np.sort(r.to_numpy())[-2], axis=1)
    return out.sort_values("I_max").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Emulation
# ---------------------------------------------------------------------------

def fit_emulators(summary: pd.DataFrame, current_only: set[str] | None = None,
                  n_jobs: int | None = None) -> dict:
    """One GP per target. Targets in `current_only` are fitted only on waves
    >= FIRST_CURRENT_WAVE (skipped if fewer than 10 such points exist).

    n_jobs > 1 fits the targets in parallel processes (one BLAS thread each):
    a 30-dimensional ARD GP on 600 points takes minutes, and 29 of them in
    sequence dominate a wave's design step. The fits are identical either way."""
    current_only = TREATMENT_DEPENDENT if current_only is None else current_only
    full = summary
    current = summary[summary["wave"] >= FIRST_CURRENT_WAVE]
    if EMULATOR_INPUTS is not None or (n_jobs is not None and n_jobs > 1):
        jobs = []
        for t in TARGETS_HM:
            X, y = _training_data(current if t.metric in current_only else full, t)
            cols = None
            if EMULATOR_INPUTS is not None:
                cols = [PATHS.index(p) for p in EMULATOR_INPUTS(t.metric)]
                X = X[:, cols]
            jobs.append((t.metric, X, y, cols))
        if n_jobs is not None and n_jobs > 1:
            from joblib import Parallel, delayed
            fits = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_fit_gp_single_thread)(X, y) for _, X, y, _ in jobs)
        else:
            fits = [_fit_gp(X, y) for _, X, y, _ in jobs]
        return {m: (gp if cols is None else SubsetGP(gp, cols))
                for (m, _, _, cols), gp in zip(jobs, fits) if gp is not None}
    ems = {}
    for t in TARGETS_HM:
        data = current if t.metric in current_only else full
        gp = _fit_one(data, t)
        if gp is not None:
            ems[t.metric] = gp
    return ems


class SubsetGP:
    """An emulator fitted on a subset of the parameter columns, predicting
    from full unit-cube rows (Campaign.emulator_inputs)."""

    def __init__(self, gp, cols: list[int]):
        self.gp, self.cols = gp, list(cols)

    def predict(self, X, return_std: bool = False):
        return self.gp.predict(np.asarray(X)[:, self.cols], return_std=return_std)


def _training_data(summary: pd.DataFrame, t: "HMTarget") -> tuple[np.ndarray, np.ndarray]:
    X = to_unit(summary[PATHS].to_numpy(dtype=float))
    y = summary[t.metric].to_numpy(dtype=float)
    return X, y


def _fit_gp_single_thread(X: np.ndarray, y: np.ndarray):
    """_fit_gp in a worker process, pinned to one BLAS thread so that n_jobs
    workers do not oversubscribe the cores."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _fit_gp(X, y)


def _fit_one(summary: pd.DataFrame, t: "HMTarget"):
    X, y = _training_data(summary, t)
    return _fit_gp(X, y)


def _fit_gp(X: np.ndarray, y: np.ndarray):
    """The GP emulator itself (no module state, so it can run in a worker)."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    # Length scales of parameters an output does not depend on run to the
    # upper bound - expected (that is how ARD reports irrelevance), not a
    # fitting problem.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ok = np.isfinite(y)
    if ok.sum() < 10:
        return None
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.full(X.shape[1], 0.5), length_scale_bounds=(0.05, 20.0), nu=2.5)
              + WhiteKernel(0.05, (1e-6, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    gp.fit(X[ok], y[ok])
    return gp


def emulated_implausibility(ems: dict, U: np.ndarray, chunk: int = 20000,
                            only: set[str] | None = None, n_jobs: int | None = None) -> np.ndarray:
    """(n, n_targets) emulator implausibility at unit-cube points U. With
    `only`, the other targets' columns are left at 0. n_jobs > 1 evaluates
    the targets in parallel processes (same numbers)."""
    I = np.zeros((len(U), len(TARGETS_HM)))
    if n_jobs is not None and n_jobs > 1:
        from joblib import Parallel, delayed
        cols = [(j, ems[t.metric], t.z, t.sd_obs ** 2 + t.sd_md ** 2) for j, t in enumerate(TARGETS_HM)
                if t.metric in ems and (only is None or t.metric in only)]
        out = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_implausibility_column)(gp, U, z, v, chunk) for _, gp, z, v in cols)
        for (j, _, _, _), col in zip(cols, out):
            I[:, j] = col
        return I
    for j, t in enumerate(TARGETS_HM):
        gp = ems.get(t.metric)
        if gp is None or (only is not None and t.metric not in only):
            continue
        for a in range(0, len(U), chunk):
            mu, sd = gp.predict(U[a:a + chunk], return_std=True)
            I[a:a + chunk, j] = np.abs(mu - t.z) / np.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + sd ** 2)
    return I


def _implausibility_column(gp, U: np.ndarray, z: float, var_fixed: float, chunk: int) -> np.ndarray:
    """One target's implausibility at U, in a worker with one BLAS thread."""
    from threadpoolctl import threadpool_limits
    col = np.empty(len(U))
    with threadpool_limits(1):
        for a in range(0, len(U), chunk):
            mu, sd = gp.predict(U[a:a + chunk], return_std=True)
            col[a:a + chunk] = np.abs(mu - z) / np.sqrt(var_fixed + sd ** 2)
    return col


def _criterion(I: np.ndarray, which: str) -> np.ndarray:
    """max ("max") or second-largest ("second") implausibility per row. The
    second-max rule - one target may miss - is the usual choice for early
    waves with many outputs (Vernon et al. 2010), so that one poorly emulated
    or structurally hard output cannot empty the space on its own."""
    if which == "max":
        return I.max(axis=1)
    return np.sort(I, axis=1)[:, -2]


def sample_nroy(ems: dict, n: int, seed: int, n_candidates: int = 200_000,
                criterion: str = "second", conditional_on_rota_independent: bool = False,
                n_jobs: int | None = None, core_fraction: float = 0.0) -> tuple[np.ndarray, float]:
    """n space-filling points from the emulated NROY region, and its volume fraction.

    core_fraction > 0: that share of the points (listed first) is a maximin
    sample of the NROY's core - its 10% of candidates with the smallest sum
    of squared implausibilities - and the rest a maximin sample of the whole
    NROY. Sampling only the whole region characterises it; but the pathogens
    are calibrated by largely separate parameters, so a point that suits all
    three at once is rare unless it is sought, and those are the runs that
    sharpen the emulators where the defaults will be chosen.

    conditional_on_rota_independent: judge only the campaign's conditional
    targets (ROTA_INDEPENDENT_TARGETS), with the parameters added after
    earlier waves (WAVE3_FIXED) set to their held values for the emulators,
    and sample those parameters freely. v2 wave 4: rotavirus importation;
    v3 wave 2: the Campylobacter and Cryptosporidium infant ramps."""
    rng = np.random.default_rng(seed)
    U = qmc.LatinHypercube(d=len(PATHS), seed=seed).random(n_candidates)
    if conditional_on_rota_independent:
        U_eval = U.copy()
        for p, v in WAVE3_FIXED.items():
            U_eval[:, PATHS.index(p)] = to_unit(_row_with(p, v))[0, PATHS.index(p)]
        I = emulated_implausibility(ems, U_eval, only=ROTA_INDEPENDENT_TARGETS, n_jobs=n_jobs)
    else:
        I = emulated_implausibility(ems, U, n_jobs=n_jobs)
    score = _criterion(I, criterion)
    keep = score < CUTOFF
    frac = float(keep.mean())
    pool = U[keep]
    if len(pool) == 0:
        # Fall back to the least-implausible candidates so the wave can proceed.
        order = np.argsort(score)
        pool = U[order[: max(n * 20, 1000)]]
    if len(pool) <= n:
        return pool, frac
    n_core = int(round(n * core_fraction)) if keep.any() else 0
    if n_core > 0:
        ss = (I[keep] ** 2).sum(axis=1)
        core = pool[np.argsort(ss)[: max(len(pool) // 10, 5 * n_core)]]
        first = _maximin(core, n_core, rng)
        rest = _maximin(pool, n - n_core, rng, start=first)
        return np.vstack([first, rest]), frac
    return _maximin(pool, n, rng), frac


def _maximin(pool: np.ndarray, n: int, rng, start: np.ndarray | None = None) -> np.ndarray:
    """Greedy maximin subsample of n rows (away from `start` rows too)."""
    if len(pool) <= n:
        return pool
    if start is None:
        chosen = [int(rng.integers(len(pool)))]
        d = np.linalg.norm(pool - pool[chosen[0]], axis=1)
        n_more = n - 1
    else:
        chosen = []
        d = np.full(len(pool), np.inf)
        for row in start:
            d = np.minimum(d, np.linalg.norm(pool - row, axis=1))
        n_more = n
    for _ in range(n_more):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(pool - pool[k], axis=1))
    return pool[chosen]


def _emulated_means_and_I(ems: dict, U: np.ndarray) -> list[dict]:
    """Per design point: emulated mean (transformed scale), SD and
    implausibility for every target."""
    out = [dict() for _ in range(len(U))]
    for t in TARGETS_HM:
        gp = ems.get(t.metric)
        if gp is None:
            continue
        mu, sd = gp.predict(U, return_std=True)
        I = np.abs(mu - t.z) / np.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + sd ** 2)
        for i in range(len(U)):
            out[i][t.metric] = [float(mu[i]), float(sd[i]), float(I[i])]
    return out


def _row_with(path: str, value: float) -> np.ndarray:
    """A 1-row parameter array at the box's lower corner with `path` = value."""
    row = from_unit(np.zeros((1, len(PATHS))))
    row[0, PATHS.index(path)] = value
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_wave(wave: int, n: int, reps: int, seed: int, n_cores: int | None, criterion: str = "second"):
    os.makedirs(OUT_DIR, exist_ok=True)
    # Campaigns that checkpoint (v3) save the design before running and reuse
    # it on a restart, so a resumed wave runs exactly the points it started
    # with: refitting the emulators can shift the design by floating-point
    # noise (BLAS threading), and the resume matches finished runs by their
    # parameter values.
    design_path = os.path.join(OUT_DIR, f"wave{wave}_design.json")
    if CHECKPOINT_EVERY is not None and os.path.exists(design_path):
        with open(design_path) as f:
            saved = json.load(f)
        combos, frac = saved["combos"], saved["nroy_fraction_before"]
        if (saved["n"], saved["seed"], saved["criterion"]) != (n, seed, criterion):
            raise SystemExit(f"{design_path} was made with n={saved['n']}, seed={saved['seed']}, "
                             f"criterion={saved['criterion']}; rerun with those or delete it.")
        print(f"wave {wave}: reusing the saved design ({len(combos)} points, NROY fraction {frac:.4f})")
    else:
        if wave == 1:
            U = qmc.LatinHypercube(d=len(PATHS), seed=seed).random(n)
            frac = 1.0
        else:
            summary = summarise(load_all_waves())
            n_jobs = n_cores if CAMPAIGN != "v2" else None
            ems = fit_emulators(summary, n_jobs=n_jobs)
            core = CORE_DESIGN[1] if CORE_DESIGN is not None and wave >= CORE_DESIGN[0] else 0.0
            U, frac = sample_nroy(ems, n, seed=seed + wave, criterion=criterion,
                                  conditional_on_rota_independent=(wave == FIRST_IMPORTATION_WAVE),
                                  n_jobs=n_jobs, core_fraction=core)
            if CHECKPOINT_EVERY is not None:
                # what the emulators expected at each design point, so the
                # wave can be checked against its predictions afterwards
                predicted = _emulated_means_and_I(ems, U)
        print(f"wave {wave}: emulated NROY fraction of the original box = {frac:.4f}; running {len(U)} points")
        combos = combos_from_unit(U, wave)
        if CHECKPOINT_EVERY is not None:
            extra = {}
            if wave > 1:
                extra = {"core_fraction": core, "n_core": int(round(len(combos) * core)),
                         "predicted": predicted}
            with open(design_path, "w") as f:
                json.dump({"wave": wave, "n": n, "seed": seed, "criterion": criterion,
                           "nroy_fraction_before": frac, "combos": combos, **extra}, f, indent=1)
    U = combos
    run_sweep(_spec(f"wave{wave}", reps, n_cores=n_cores), [_expand(c) for c in combos])
    meta = {"wave": wave, "n": len(U), "reps": reps, "criterion": criterion, "nroy_fraction_before": frac}
    if CAMPAIGN != "v2":
        meta.update({"campaign": CAMPAIGN, "grid_id": GRID_ID, "agents": AGENTS, "steps": STEPS, "seed": seed})
    with open(os.path.join(OUT_DIR, f"wave{wave}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def report(top: int = 10) -> pd.DataFrame:
    """Implausibility of every simulated point of the CURRENT model
    (waves >= FIRST_CURRENT_WAVE)."""
    summary = summarise(load_all_waves())
    summary = summary[summary["wave"] >= FIRST_CURRENT_WAVE]
    imp = simulated_implausibility(summary)
    imp.to_csv(os.path.join(OUT_DIR, "simulated_implausibility.csv"), index=False)
    ok = imp[imp["I_max"] < CUTOFF]
    print(f"{len(imp)} simulated points; {len(ok)} with every target non-implausible (I < {CUTOFF}).")
    show = ["gsa.design_id", "I_max", "I_second"] + [t.metric + "__raw" for t in TARGETS_HM]
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(imp[show].head(top).T)
    if len(ok):
        print("\nNon-implausible simulated points - parameter ranges:")
        for p in PATHS:
            print(f"  {p:52s} {ok[p].min():.5g} .. {ok[p].max():.5g} (median {ok[p].median():.5g})")
    best = imp.iloc[0]
    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump({"design_id": best["gsa.design_id"], "I_max": float(best["I_max"]),
                   "params": {p: float(best[p]) for p in PATHS},
                   "base_overrides": BASE_OVERRIDES}, f, indent=2)
    return imp


#: Upper limit on the share of rotavirus infections that are imported, applied
#: when choosing the default (not a calibration target). The MAL-ED targets do
#: not distinguish a town that sustains rotavirus locally from one that is a
#: sink for infection from outside: in the wave-5 NROY the imported share runs
#: from <5% to >40% at equal fit. The two make opposite predictions for any
#: intervention applied in Akuse, which cannot reduce an external force of
#: infection. Importation was introduced to stop stochastic fade-out (a spark,
#: not a route), so the default is held to mostly-local transmission. This is
#: a modelling judgement - report it, and vary importation_rate in any GSA.
MAX_IMPORTED_FRACTION = 0.15
AUX_IMPORTED = "rota_imported_fraction_post_burnin"


def fit_imported_emulator(df: pd.DataFrame, summary: pd.DataFrame):
    """GP of the replicate-mean rotavirus imported fraction (linear scale);
    on the rotavirus inputs only when the campaign restricts emulator inputs."""
    return fit_aux_emulator(df, summary, AUX_IMPORTED)


def fit_aux_emulator(df: pd.DataFrame, summary: pd.DataFrame, metric: str):
    """GP of the replicate mean of an output that is not a calibration target
    (linear scale), on that pathogen's inputs when the campaign restricts
    emulator inputs - for constraints on outputs MAL-ED does not measure."""
    s = summary.copy()
    s[metric] = s["gsa.design_id"].map(df.groupby("gsa.design_id")[metric].mean())
    t = HMTarget(metric, 0.0, 1.0, 0.0, "lin")
    if EMULATOR_INPUTS is None:
        return _fit_one(s, t)
    X, y = _training_data(s, t)
    cols = [PATHS.index(p) for p in EMULATOR_INPUTS(metric)]
    gp = _fit_gp(X[:, cols], y)
    return None if gp is None else SubsetGP(gp, cols)


def _predict_chunked(gp, U: np.ndarray, chunk: int = 20000) -> np.ndarray:
    # Chunked: an ARD kernel against ~500 training points materialises an
    # (n x 500 x d) array, which for n = 3e5 does not fit in memory.
    return np.concatenate([gp.predict(U[a:a + chunk]) for a in range(0, len(U), chunk)])


def select_candidates(n: int = 3, seed: int = 11, n_candidates: int = 300_000,
                      min_separation: float = 0.15, centrality_weight: float = 0.0,
                      max_imported_fraction: float = MAX_IMPORTED_FRACTION,
                      n_jobs: int | None = None, n_simulated: int = 0,
                      refocus_from_wave: int | None = None, fixed: dict | None = None,
                      constraint: Callable | None = None, adjusted: list[dict] | None = None,
                      suffix: str = "", output_ranges: dict | None = None,
                      ratio_ranges: dict | None = None, aux_ranges: dict | None = None) -> list[dict]:
    """Default candidates: a regularised best fit within the emulated NROY.

    The admissible set is the emulated NROY (max_j I_j < CUTOFF) with the
    emulated rotavirus imported fraction below `max_imported_fraction`. Within
    it, candidates minimise

        sum_j I_j(u)^2  +  centrality_weight * sum_k ((u_k - m_k) / s_k)^2

    over the unit cube, where m_k and s_k are the median and SD of parameter k
    across the admissible set. The first term is the standardised squared
    distance to every target at once. The second (off by default) pulls
    parameters the targets leave flat towards the centre of their admissible
    range. Tried at weight 1, it dominated the objective (12.3 vs a fit term of
    5) because marginal medians ignore the strong correlations within the NROY
    (e.g. rota infection probability vs water dose, r = -0.8), and the chosen
    points fitted worse on fresh seeds (verify_final2: max I 1.7-2.1 vs 1.1-1.3).
    The default is therefore the plain best fit; weakly identified parameters
    are reported as admissible ranges (nroy_marginals.csv), not resolved.

    Refined by two rounds of local random search around the 20 best; `n`
    mutually separated points (unit-cube distance > min_separation) are
    returned and written to emulator_candidates.json for
    `--verify --candidates-json`.

    v3 additions: refocus_from_wave fits the emulators only on waves from
    that one on (the selection only concerns the current NROY, and dropping
    the first, box-filling wave's mostly implausible points makes the fits
    both faster and local); n_jobs fits and evaluates them in parallel; and
    n_simulated appends the best SIMULATED points (lowest max implausibility
    over their replicates, imported fraction below the cap) as candidates
    "s0", "s1", ... - their two-seed results may be lucky, which is what
    --verify on fresh seeds tests, but for targets whose emulators are still
    vague (the age and history ratios) they are direct evidence.

    fixed {path: value} holds parameters at values taken from the
    literature instead of selecting them (the emulators are evaluated there);
    constraint(values) -> bool mask rejects physically implausible
    combinations; adjusted lists earlier candidates to re-verify with the
    fixed values and constraint applied ({"gsa.design_id": ..., <paths>: ...},
    written as given); suffix names the output files
    (emulator_candidates{suffix}.json, nroy_marginals{suffix}.csv);
    output_ranges {target metric: (lo, hi)} additionally requires the
    emulated value of those targets (back on the natural scale) to lie in
    [lo, hi]; ratio_ranges {name: (numerator metric, denominator metric, lo,
    hi)} the same for the ratio of two emulated targets; aux_ranges {metric:
    (lo, hi)} the same for outputs that are not targets, each emulated by
    fit_aux_emulator.
    """
    df = load_all_waves()
    summary = summarise(df)
    if refocus_from_wave is not None:
        summary = summary[summary["wave"] >= refocus_from_wave]
    ems = fit_emulators(summary, n_jobs=n_jobs)
    gp_imp = fit_imported_emulator(df, summary)
    gp_aux = {m: fit_aux_emulator(df, summary, m) for m in (aux_ranges or {})}
    rng = np.random.default_rng(seed)
    U = qmc.LatinHypercube(d=len(PATHS), seed=seed).random(n_candidates)
    fixed = fixed or {}
    fixed_u = {PATHS.index(p): to_unit(_row_with(p, v))[0, PATHS.index(p)] for p, v in fixed.items()}

    def hold(X):
        for j, u in fixed_u.items():
            X[:, j] = u
        return X

    U = hold(U)
    crit = {"which": "max"}

    def admissible(X):
        I = emulated_implausibility(ems, X, n_jobs=n_jobs)
        imp = _predict_chunked(gp_imp, X)
        ok_ = (_criterion(I, crit["which"]) < CUTOFF) & (imp < max_imported_fraction)
        if constraint is not None:
            ok_ &= constraint(pd.DataFrame(from_unit(X), columns=PATHS))
        for metric, (lo, hi) in (output_ranges or {}).items():
            t = next(t for t in TARGETS_HM if t.metric == metric)
            mu = _predict_chunked(ems[metric], X)
            val = np.exp(mu) - t.eps if t.transform == "log" else mu
            ok_ &= (val >= lo) & (val <= hi)

        def natural(metric):
            t = next(t for t in TARGETS_HM if t.metric == metric)
            mu = _predict_chunked(ems[metric], X)
            return np.exp(mu) - t.eps if t.transform == "log" else mu

        for num, den, lo, hi in (ratio_ranges or {}).values():
            r = natural(num) / np.maximum(natural(den), 1e-9)
            ok_ &= (r >= lo) & (r <= hi)
        for metric, (lo, hi) in (aux_ranges or {}).items():
            v = _predict_chunked(gp_aux[metric], X)
            ok_ &= (v >= lo) & (v <= hi)
        return I, imp, ok_

    I, imp, ok = admissible(U)
    if ok.sum() < 20:
        # Too few candidates fit every target by the emulators: admit those
        # that miss at most one (the second-max rule used to design waves).
        crit["which"] = "second"
        print(f"only {int(ok.sum())} candidates with every target within {CUTOFF}; admitting one miss")
        I, imp, ok = admissible(U)
    if not ok.any():
        raise SystemExit("No admissible candidate: the emulators predict no parameter set meeting the "
                         "targets under these fixed values and constraints. Nothing written.")
    print(f"emulated NROY fraction (max I < {CUTOFF}): {(I.max(axis=1) < CUTOFF).mean():.5f}; "
          f"admissible ({crit['which']}-max I < {CUTOFF}, imported fraction < {max_imported_fraction}): "
          f"{ok.mean():.5f}")
    m, sd = np.median(U[ok], axis=0), U[ok].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)          # parameters held fixed

    def objective(X):
        I_, imp_, ok_ = admissible(X)
        val = (I_ ** 2).sum(axis=1) + centrality_weight * (((X - m) / sd) ** 2).sum(axis=1)
        return I_, imp_, np.where(ok_, val, np.inf)

    _, _, obj = objective(U)
    best = U[np.argsort(obj)[:20]]
    pool = [best]
    for step in (0.05, 0.02):
        c = hold(np.clip(np.repeat(best, 200, axis=0) + rng.normal(0, step, (len(best) * 200, len(PATHS))), 0, 1))
        _, _, obj_c = objective(c)
        best = c[np.argsort(obj_c)[:20]]
        pool.append(best)
    allc = np.vstack(pool)
    I_all, imp_all, obj_all = objective(allc)
    picked: list[int] = []
    for k in np.argsort(obj_all):
        if not np.isfinite(obj_all[k]):
            break
        if all(np.linalg.norm(allc[k] - allc[j]) > min_separation for j in picked):
            picked.append(int(k))
        if len(picked) == n:
            break
    values = from_unit(allc[picked])
    out = []
    for i, (k, row) in enumerate(zip(picked, values)):
        print(f"\ncandidate e{i}: objective {obj_all[k]:.2f} (sum I^2 {(I_all[k] ** 2).sum():.2f}), "
              f"emulated max I {I_all[k].max():.2f}, imported fraction {imp_all[k]:.3f}"
              + "".join(f", {m} {float(gp_aux[m].predict(allc[k:k + 1])[0]):.3f}" for m in gp_aux))
        for t, Ij in zip(TARGETS_HM, I_all[k]):
            mu = ems[t.metric].predict(allc[k:k + 1])[0]
            val = np.exp(mu) - t.eps if t.transform == "log" else mu
            print(f"   {t.metric:52s} emulated {val:7.3f}   I {Ij:.2f}")
        for j, p in enumerate(PATHS):
            print(f"   {p:52s} {row[j]:.4g}   ({(allc[k, j] - m[j]) / sd[j]:+.2f} NROY SD from median)")
        out.append({"gsa.design_id": f"e{i}", **{p: float(v) for p, v in zip(PATHS, row)}})
    if n_simulated:
        sim = simulated_implausibility(summarise(df))
        frac_imp = df.groupby("gsa.design_id")[AUX_IMPORTED].mean()
        sim = sim[sim["gsa.design_id"].map(frac_imp) < max_imported_fraction]
        for i, (_, r) in enumerate(sim.head(n_simulated).iterrows()):
            print(f"\ncandidate s{i}: simulated point {r['gsa.design_id']} (wave {int(r['wave'])}), "
                  f"max I {r['I_max']:.2f} over {int(r['n_rep'])} seeds, imported fraction "
                  f"{frac_imp[r['gsa.design_id']]:.3f}")
            out.append({"gsa.design_id": f"s{i}", "source": r["gsa.design_id"],
                        **{p: float(r[p]) for p in PATHS}})
    for a in adjusted or []:
        row = dict(a)
        row.update(fixed)
        X = hold(to_unit(np.array([[float(row[p]) for p in PATHS]])))
        I_a, imp_a, ok_a = admissible(X)
        print(f"\ncandidate {row['gsa.design_id']}: earlier candidate with the fixed values/constraint applied; "
              f"emulated max I {I_a[0].max():.2f}, sum I^2 {(I_a[0] ** 2).sum():.2f}, admissible {bool(ok_a[0])}")
        out.append(row)
    with open(os.path.join(OUT_DIR, f"emulator_candidates{suffix}.json"), "w") as f:
        json.dump(out, f, indent=1)
    nroy = pd.DataFrame(from_unit(U[ok]), columns=PATHS).quantile([0.025, 0.25, 0.5, 0.75, 0.975]).T
    nroy.to_csv(os.path.join(OUT_DIR, f"nroy_marginals{suffix}.csv"))
    print("\nadmissible-set marginals (2.5/25/50/75/97.5%):")
    with pd.option_context("display.width", 200):
        print(nroy.to_string(float_format=lambda x: f"{x:.4g}"))
    return out


def verify(reps: int, n_cores: int | None, design_ids: list[str] | None = None, top: int = 3,
           candidates_json: str | None = None, tag: str = "verify"):
    """Re-run chosen points on seeds disjoint from the waves (base_seed 10023).

    Points come either from simulated_implausibility.csv (by design id, or the
    `top` best) or from a JSON list of {"gsa.design_id": ..., <PATHS>: value}
    dicts - e.g. emulator-optimised points that were never simulated."""
    if candidates_json:
        with open(candidates_json) as f:
            combos = [{"gsa.design_id": f"{tag}_" + c["gsa.design_id"], **{p: float(c[p]) for p in PATHS}}
                      for c in json.load(f)]
    else:
        imp = pd.read_csv(os.path.join(OUT_DIR, "simulated_implausibility.csv"))
        sel = imp[imp["gsa.design_id"].isin(design_ids)] if design_ids else imp.head(top)
        combos = [{"gsa.design_id": f"{tag}_" + r["gsa.design_id"], **{p: float(r[p]) for p in PATHS}}
                  for _, r in sel.iterrows()]
    df = run_sweep(_spec(tag, reps, base_seed=10_023, n_cores=n_cores), [_expand(c) for c in combos])
    df["wave"] = 99
    v = simulated_implausibility(summarise(df))
    # Between-seed stability: a default that fades out in some seeds is not a
    # usable baseline however good its mean, so report it alongside I.
    stab = df.groupby("gsa.design_id").agg(
        rota_fadeout_reps=("rota_infections_per_child_year_0_24m", lambda x: int((x < 0.05).sum())),
        rota_infections_cv=("rota_infections_per_child_year_0_24m", lambda x: float(x.std() / x.mean())),
        crypto_infections_cv=("crypto_infections_per_child_year_0_24m", lambda x: float(x.std() / x.mean())),
        rota_imported_fraction=(AUX_IMPORTED, "mean"),
    ).reset_index()
    v = v.merge(stab, on="gsa.design_id", how="left")
    v.to_csv(os.path.join(OUT_DIR, f"{tag}_implausibility.csv"), index=False)
    if CAMPAIGN == "econ":
        # the epidemiology should not have moved: the v4 targets at each verified point
        for design_id, g in df.groupby("gsa.design_id"):
            chk = econ_epi_check(g)
            chk.to_csv(os.path.join(OUT_DIR, f"{tag}_{design_id}_v4_targets.csv"), index=False)
            print(f"\n{design_id}: v4 targets, worst I {chk['I'].max():.2f} "
                  f"({chk.loc[chk['I'].idxmax(), 'target']}); over 3: {int((chk['I'] > 3).sum())}")
        econ_cols = [c for c in df.columns if c.startswith("econ_") and not c.endswith(("_q1", "_q2", "_q3", "_q4", "_q5"))]
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print(df.groupby("gsa.design_id")[econ_cols].mean().T)
    show = (["gsa.design_id", "I_max", "I_second", "rota_fadeout_reps", "rota_infections_cv",
             "crypto_infections_cv", "rota_imported_fraction"] + [t.metric + "__raw" for t in TARGETS_HM])
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(v[show].T)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--wave", type=int)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--select", action="store_true",
                    help="write emulator-optimal default candidates to emulator_candidates.json")
    ap.add_argument("--centrality-weight", type=float, default=0.0, help="see select_candidates")
    ap.add_argument("--design-ids", nargs="*", default=None)
    ap.add_argument("--candidates-json", default=None,
                    help="verify points listed in this JSON file instead of simulated ones")
    ap.add_argument("--tag", default="verify", help="output name for --verify")
    ap.add_argument("--criterion", choices=("max", "second"), default="second")
    ap.add_argument("--literature", action="store_true",
                    help="v3 --select: hold V3_LITERATURE_FIXED and apply v3_physical_constraint")
    ap.add_argument("--incidence-in-site-range", action="store_true",
                    help="with --literature: also require detected incidence within MAL-ED's "
                         "high-burden-site range (V3_INCIDENCE_SITE_RANGES)")
    ap.add_argument("--attributable-in-site-range", action="store_true",
                    help="with --literature: also require P(attributable | detected infection) within "
                         "MAL-ED's site range (V3_ATTRIBUTABLE_FRACTION_SITE_RANGES)")
    ap.add_argument("--route-ranges", action="store_true",
                    help="v4 --select: also require V4_ROUTE_RANGES (rotavirus at most 10%% waterborne)")
    ap.add_argument("--campaign", choices=sorted(CAMPAIGNS), default="v2",
                    help="grid, parameter box, targets and output folder (see module docstring)")
    args = ap.parse_args()
    use_campaign(args.campaign)
    if args.wave:
        run_wave(args.wave, args.n, args.reps, args.seed, args.workers, args.criterion)
    if args.report:
        report()
    if args.select:
        if args.campaign == "v2":
            select_candidates(centrality_weight=args.centrality_weight)
        elif args.campaign == "v4":
            # v3's final rules from the start: literature lambda (in
            # V4_BASE_OVERRIDES), illness-causing infections detectable at
            # least as long, detected incidence and P(attributable | detected)
            # inside MAL-ED's site ranges - plus adult : child carriage in its
            # plausible range, and (from the wave-3 selection on) rotavirus
            # mostly person-to-person (V4_ROUTE_RANGES)
            select_candidates(centrality_weight=args.centrality_weight, n_jobs=args.workers,
                              n_simulated=0 if args.route_ranges else 3, refocus_from_wave=2,
                              constraint=v3_physical_constraint,
                              output_ranges={**V3_INCIDENCE_SITE_RANGES, **V4_CARRIAGE_RANGES},
                              ratio_ranges=V3_ATTRIBUTABLE_FRACTION_SITE_RANGES,
                              aux_ranges=V4_ROUTE_RANGES if args.route_ranges else None,
                              suffix="_routes" if args.route_ranges else "")
        elif args.campaign == "econ":
            select_candidates(centrality_weight=args.centrality_weight, n_jobs=args.workers,
                              n_simulated=2, refocus_from_wave=None)
        elif args.literature:
            # literature values held (V3_LITERATURE_FIXED), illness-causing
            # infections detectable at least as long, and the earlier best
            # candidate e0 re-verified with both applied ("a0")
            adjusted = []
            prev = os.path.join(OUT_DIR, "emulator_candidates.json")
            if os.path.exists(prev):
                with open(prev) as f:
                    e0 = next(c for c in json.load(f) if c["gsa.design_id"] == "e0")
                a0 = {"gsa.design_id": "a0", **{p: e0[p] for p in PATHS}}
                for p in ("rota", "campy", "crypto"):
                    sym = f"pathogens[{p}].detectable_shedding_days_symptomatic"
                    a0[sym] = max(a0[sym], a0[f"pathogens[{p}].detectable_shedding_days_asymptomatic"])
                adjusted.append(a0)
            ranges = V3_INCIDENCE_SITE_RANGES if args.incidence_in_site_range else None
            ratios = V3_ATTRIBUTABLE_FRACTION_SITE_RANGES if args.attributable_in_site_range else None
            suffix = "_lit" + ("_sites" if ranges else "") + ("_attr" if ratios else "")
            select_candidates(centrality_weight=args.centrality_weight, n_jobs=args.workers,
                              refocus_from_wave=2, fixed=V3_LITERATURE_FIXED,
                              constraint=v3_physical_constraint,
                              adjusted=[] if (ranges or ratios) else adjusted,
                              suffix=suffix, output_ranges=ranges, ratio_ranges=ratios)
        else:
            select_candidates(centrality_weight=args.centrality_weight, n_jobs=args.workers,
                              n_simulated=3, refocus_from_wave=2)
    if args.verify:
        verify(args.reps, args.workers, args.design_ids, candidates_json=args.candidates_json, tag=args.tag)


if __name__ == "__main__":
    main()

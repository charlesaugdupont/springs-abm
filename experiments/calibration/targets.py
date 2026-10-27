# experiments/calibration/targets.py
"""
Empirical targets for SPRINGS-ABM calibration.

Single source of truth for the plausibility bands that the OAT sensitivity
sweep (sensitivity.py), the LHS calibration searches (run_calibration.py,
run_crypto_calibration.py) and any history-matching wave hold model output
against.

Two generations of targets live here:

  TARGETS / CRYPTO_TARGETS / ALL_TARGETS   (current)
      Infection and disease incidence in children 0-24 months from the MAL-ED
      birth cohort (abm/evidence/maled.py), scored on the burn-in-aware,
      age-windowed metrics in experiments.metrics.age_window_rates, plus the
      route-attribution bands. Infection and disease are separate targets:
      they are linked by the (literature-fixed) symptomatic fraction, and
      conflating them is what made the original rotavirus band ~10x too high
      for disease.

  LEGACY_TARGETS / LEGACY_CRYPTO_TARGETS   (superseded; kept for reproducing
      earlier results)
      The original bands, scored on whole-run lifetime infection counts in all
      under-5s. Two problems were found with them:
        * "*_episodes_per_child_year" counted INFECTIONS over the WHOLE run,
          and a run starts from a fully naive population seeded with 1%
          exposed, so most of the counted rotavirus infections came from the
          seeded initial wave: ~4.9 infections/child-year in the first 60
          days versus ~0.3-0.6 once that wave has passed (4 seeds, 4000
          agents, 730 days). The calibrated rotavirus rate of ~2 was
          therefore mostly an artefact of the initial condition.
        * Read as DISEASE, 1.5-2.5 rotavirus episodes per child-year is ~10x
          MAL-ED's 0.23 attributable episodes (and GEMS-scale burdens are
          lower still); read as INFECTION it is ~5x MAL-ED's 0.36.

How the MAL-ED bands are built
------------------------------
  * Age window 0-24 months (MAL-ED's follow-up), post burn-in.
  * Central value ("anchor"): the pooled MAL-ED estimate (Supplementary
    Table 1; algorithm attribution, consecutive-detection infections).
  * Band: the range across MAL-ED's HIGH-BURDEN sites - Bangladesh, India,
    Nepal, Pakistan, Tanzania (Haydom) and, except for rotavirus (vaccinated
    site, excluded by the paper), Peru. Brazil (Fortaleza) and South Africa
    (Venda) are excluded as the two upper-middle-income sites with markedly
    better WASH, least like rural south-eastern Ghana. No MAL-ED site is in
    Ghana; the band is an honest statement that Akuse's true value is
    unknown within the range seen across comparable settings, not a
    confidence interval.
  * For history matching, ANCHORS and LOG_SD give a central value and a
    between-site SD on the log scale for each metric.

What these bands do NOT capture
-------------------------------
  * Children 2-5 years: MAL-ED stops at 24 months. The model's 24-60 month
    children are constrained only indirectly (through transmission).
  * Rotavirus vaccination: MAL-ED's rotavirus estimates come from sites that
    had NOT introduced the vaccine. Ghana introduced Rotarix in 2012, so these
    are natural-history targets: calibrate with vaccination switched off
    (pathogens[rota].vaccination_rate = 0) or treat the vaccinated scenario
    as a separate validation against post-introduction Ghanaian data.
  * Monthly sampling under-detects short subclinical infections, so the
    infection bands are, if anything, LOW. The disease bands do not have this
    problem (every diarrhoeal episode was sampled).
"""
from __future__ import annotations

import math

from abm.evidence import maled

#: MAL-ED sites used for the band (see module docstring).
HIGH_BURDEN_SITES: dict[str, tuple[str, ...]] = {
    "rota": ("Bangladesh", "India", "Nepal", "Pakistan", "Tanzania"),
    "campy": ("Bangladesh", "India", "Nepal", "Pakistan", "Tanzania", "Peru"),
    "crypto": ("Bangladesh", "India", "Nepal", "Pakistan", "Tanzania", "Peru"),
}

_PER_100CM = maled.PER_100_CHILD_MONTHS_TO_PER_CHILD_YEAR


def _site_values(table: dict, pathogen: str) -> list[float]:
    return [table[pathogen][s] * _PER_100CM for s in HIGH_BURDEN_SITES[pathogen]]


def _band(values: list[float]) -> tuple[float, float]:
    return (round(min(values), 4), round(max(values), 4))


def _log_sd(values: list[float]) -> float:
    logs = [math.log(v) for v in values]
    mean = sum(logs) / len(logs)
    return math.sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1))


#: Pooled MAL-ED central values, per child-year, 0-24 months.
ANCHORS: dict[str, float] = {}
#: Between-site SD of log(rate) across HIGH_BURDEN_SITES, for history matching.
LOG_SD: dict[str, float] = {}
_MALED_BANDS: dict[str, tuple[float, float]] = {}
for _p in ("rota", "campy", "crypto"):
    _inf = _site_values(maled.INFECTION_RATE_BY_SITE_PER_100CM, _p)
    _epi = _site_values(maled.ATTRIBUTABLE_RATE_BY_SITE_PER_100CM, _p)
    _ki = f"{_p}_infections_per_child_year_0_24m"
    _ke = f"{_p}_symptomatic_episodes_per_child_year_0_24m"
    _MALED_BANDS[_ki] = _band(_inf)
    _MALED_BANDS[_ke] = _band(_epi)
    ANCHORS[_ki] = maled.overall_infection_rate_per_child_year(_p)
    ANCHORS[_ke] = maled.overall_attributable_rate_per_child_year(_p)
    LOG_SD[_ki] = _log_sd(_inf)
    LOG_SD[_ke] = _log_sd(_epi)

# Resulting bands (per child-year, 0-24 months), for reference:
#   rota   infections (0.224, 0.713)  anchor 0.364 | episodes (0.053, 0.528) anchor 0.229
#   campy  infections (0.613, 2.074)  anchor 1.052 | episodes (0.042, 0.484) anchor 0.164
#   crypto infections (0.350, 0.883)  anchor 0.461 | episodes (0.013, 0.236) anchor 0.080
# The disease bands are ~10-fold wide because P(disease | infection) itself
# varies ~10-fold between sites (e.g. Cryptosporidium 0.016 in Haydom, 0.33
# in Pakistan). See SYMPTOMATIC_FRACTION_SITE_RANGE.

#: Range of site-level P(attributable diarrhoea | infection) across
#: HIGH_BURDEN_SITES (attributable rate / infection rate per site). The model
#: fixes symptomatic_fraction at the pooled value; this is the uncertainty
#: range to propagate in a GSA, since Akuse could sit anywhere in it.
SYMPTOMATIC_FRACTION_SITE_RANGE: dict[str, tuple[float, float]] = {
    _p: _band([
        maled.ATTRIBUTABLE_RATE_BY_SITE_PER_100CM[_p][s] / maled.INFECTION_RATE_BY_SITE_PER_100CM[_p][s]
        for s in HIGH_BURDEN_SITES[_p]
    ])
    for _p in ("rota", "campy", "crypto")
}

# ---------------------------------------------------------------------------
# Current targets
# ---------------------------------------------------------------------------
TARGETS = {
    "rota_infections_per_child_year_0_24m":            _MALED_BANDS["rota_infections_per_child_year_0_24m"],
    "rota_symptomatic_episodes_per_child_year_0_24m":  _MALED_BANDS["rota_symptomatic_episodes_per_child_year_0_24m"],
    "campy_infections_per_child_year_0_24m":           _MALED_BANDS["campy_infections_per_child_year_0_24m"],
    "campy_symptomatic_episodes_per_child_year_0_24m": _MALED_BANDS["campy_symptomatic_episodes_per_child_year_0_24m"],
    # Unchanged from the original band (source-attribution literature: poultry
    # and ruminant reservoirs dominate C. jejuni/coli attribution), now scored
    # post burn-in.
    "campy_zoonotic_fraction_post_burnin":             (0.50, 0.80),
}

# Kept as a SEPARATE dict because the crypto calibration is a crypto-only
# search (run_crypto_calibration.py) that must not trade off against the
# rotavirus/campylobacter bands. The route-fraction bands are unchanged from
# the first Cryptosporidium implementation; their sources are recorded in
# LEGACY_CRYPTO_TARGETS below.
CRYPTO_TARGETS = {
    "crypto_infections_per_child_year_0_24m":           _MALED_BANDS["crypto_infections_per_child_year_0_24m"],
    "crypto_symptomatic_episodes_per_child_year_0_24m": _MALED_BANDS["crypto_symptomatic_episodes_per_child_year_0_24m"],
    "crypto_zoonotic_fraction_post_burnin":             (0.02, 0.15),
    "crypto_household_fraction_post_burnin":            (0.30, 0.60),
}

#: Every current band, for drift checks across all three pathogens at once.
ALL_TARGETS = {**TARGETS, **CRYPTO_TARGETS}

# ---------------------------------------------------------------------------
# Diagnostic targets - reported, NOT scored
# ---------------------------------------------------------------------------
# MAL-ED Supplementary Table 11: median duration of attributable diarrhoea
# 2-3 days at every 6-month band to 24 months, for all three pathogens. The
# model's NATURAL durations are set to match this (IllnessMechanicsConfig).
# Its REALIZED durations do not: a successful care-seeking visit ends an
# episode at once (CareSeekingSystem._apply_treatment_outcome), which with
# ~70-80% of episodes treated puts the realized median at ~1 day. MAL-ED's
# children received standard care (ORS/zinc), which barely shortens
# diarrhoea, so the mismatch is a property of the care-seeking model's
# treatment-outcome assumption, not of the duration window. Not scored
# because no transmission or disease parameter can fix it.
DIAGNOSTIC_TARGETS = {
    f"{p}_median_realized_duration": (2.0, 3.0) for p in ("rota", "campy", "crypto")
}

# ---------------------------------------------------------------------------
# Legacy targets (superseded - see module docstring)
# ---------------------------------------------------------------------------
# Source: sub-Saharan Africa / Ghana literature; GEMS study; WHO rotavirus
# bulletins. Scored on whole-run lifetime infection counts in all under-5s.
LEGACY_TARGETS = {
    "rota_episodes_per_child_year":  (1.5,  2.5),
    "campy_episodes_per_child_year": (1.0,  3.0),
    "rota_peak_prevalence":          (0.02, 0.10),
    "campy_peak_prevalence":         (0.02, 0.15),
    "campy_zoonotic_fraction":       (0.50, 0.80),
}

LEGACY_CRYPTO_TARGETS = {
    # Kattula et al. 2016, Clin Infect Dis 62(1):47-54 (Vellore birth cohort,
    # n=497 followed 3 years, twice-weekly visits, PCR + serology): overall
    # incidence 0.86 infections per child-year; 97% of children infected by
    # age 3; 81% reinfected. Consistent with MAL-ED's pooled 0.46 per
    # child-year to 24 months given MAL-ED's monthly (less sensitive) sampling.
    "crypto_infections_per_child_year": (0.5, 1.5),
    # Gilchrist et al. 2025, J Infect Dis (Dhaka cohort): diarrhoea with
    # Cryptosporidium peaked at 0.19 episodes per child at 1-2 years.
    "crypto_symptomatic_episodes_per_child_year": (0.08, 0.35),
    # CARRIAGE prevalence - cross-sectional molecular surveys: Krumkamp et al.
    # 2020 23% among neighbouring children <5y (Gabon/Ghana/Madagascar/
    # Tanzania); Botchie et al. (Ghana, school-aged) 28.3%; Cossa-Moiane et
    # al. 2021 (Maputo, <5y) 35.4%; Adjei et al. 2004 (Accra, <5y without
    # diarrhoea) 15.6%. NOT carried forward: at a ~3-week shedding duration,
    # carriage ~ incidence x 21/365, so the incidence evidence implies ~3-5%
    # carriage, inconsistent with these 15-35% figures (see the caveat in the
    # first-generation notes, preserved in git history). Unresolved.
    "crypto_peak_u5_prevalence": (0.05, 0.25),
    # Krumkamp et al. 2021: no elevated risk of positive animal contacts
    # around positive index children (RR 1.2, 95% CI 0.8-1.9); GEMS: 92% of
    # C. parvum isolates anthroponotic (IIc); Eibach et al. 2015 (Ghana):
    # dominant gp60 subtypes all anthroponotic. Deliberately low, non-zero.
    "crypto_zoonotic_fraction": (0.02, 0.15),
    # Korpe et al. 2019: 8-week household secondary attack rates 35.8%
    # (urban) and 7.8% (rural); Newman et al. 1994: secondary cases in 58% of
    # Brazilian households; Krumkamp's household RR (3.6) exceeds the
    # neighbouring-children RR (2.9).
    "crypto_household_fraction": (0.30, 0.60),
}

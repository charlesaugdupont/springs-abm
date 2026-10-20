"""
MAL-ED natural-history estimates for rotavirus, Campylobacter and Cryptosporidium.

Source
------
Amin AB, Garcia Quesada M, Liu J, Sivakumar R, Conyers S, Lopman BA, Houpt E,
Platts-Mills JA, Rogawski McQuade E. "Natural history parameters for enteric
pathogens to inform modeling studies of diarrhea among children in
low-resource settings: results from the MAL-ED longitudinal birth cohort."
BMC Infectious Diseases (2026) 26:106. doi:10.1186/s12879-025-12265-8.
Values below are transcribed from its Supplementary Material 1 (tables cited
per constant). Code and data: clinepidb.org; github.com/AAmin23/paper-replications.

Study: 8-site birth cohort (Bangladesh, India, Nepal, Pakistan, Tanzania,
Brazil, Peru, South Africa), children followed from birth to 24 months with
monthly surveillance stools and stools from every diarrhoeal episode, TaqMan
Array Card qPCR. Rotavirus analyses exclude the three sites that had already
introduced rotavirus vaccine (Brazil, Peru, South Africa).

Which variant of the estimates is used, and why
-----------------------------------------------
The paper reports each quantity four ways: two etiologic-attribution methods
(the "algorithm", which uses the attributable fraction AFe plus longitudinal
changes in pathogen quantity; or AFe > 0.5 alone) crossed with two
unique-infection definitions ("crude" or "consecutive detections"). This
module uses ALGORITHM attribution with the CONSECUTIVE-DETECTIONS definition
throughout, because:
  * the algorithm is what the paper's main figures use;
  * consecutive detections merges successive positives into one infection,
    which is the closer match to what this model calls an infection (an
    S/R -> E transition followed by one infectious period). The crude
    definition counts each positive stool separately and therefore yields
    higher infection rates and LOWER symptomatic proportions.

Scope cautions - read before using these as targets
---------------------------------------------------
* AGE: 0-24 months only. Nothing here constrains children aged 2-5 years.
* SETTING: no Ghanaian site. The only unvaccinated sub-Saharan site is
  Haydom, Tanzania, which had unusually LOW diarrhoea: e.g. Cryptosporidium
  6.68 infections but only 0.11 attributable diarrhoeal episodes per 100
  child-months there. Ghana's DHS two-week diarrhoea prevalence points to a
  higher-burden setting than Haydom, so the POOLED values are used as the
  central estimates and the across-site spread informs the target bands.
* DISEASE IS ATTRIBUTABLE DIARRHOEA, i.e. episodes the pathogen CAUSED, not
  episodes in which it was merely detected. That is the right concept for
  this model's per-pathogen illness episodes.
* INFECTION DURATION (Supplementary Table 9) is "1 month (IQR 1-1)" for every
  pathogen and age band. That is an artefact of monthly sampling plus the
  paper's midpoint convention for infection start/end, not a measurement, so
  it is deliberately NOT encoded here and must not be used for recovery
  rates.
* PRIOR INFECTION: children with a prior infection did NOT have lower
  infection rates (Campylobacter: substantially HIGHER at every age from
  4 months; Cryptosporidium: about equal; rotavirus: roughly 25% lower at
  7-15 months) and did NOT have a lower probability of diarrhoea per
  infection (Supplementary Tables 3 and 6). Comparisons of previously
  infected vs naive children within an age band are confounded by
  heterogeneous exposure - the children who are infected early are the
  high-exposure children - so this does not show that immunity is absent. It
  does show that a model with homogeneous exposure and strong per-infection
  protection would predict the WRONG sign for Campylobacter.
"""
from __future__ import annotations

#: Rates per 100 child-months -> per child-year.
PER_100_CHILD_MONTHS_TO_PER_CHILD_YEAR = 12.0 / 100.0

#: 3-month age bands used in Supplementary Table 2, as (lower, upper] months.
AGE_BANDS_3MO: tuple[tuple[int, int], ...] = (
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24),
)

#: 6-month age bands used in Supplementary Tables 5, 7, 11.
AGE_BANDS_6MO: tuple[tuple[int, int], ...] = ((0, 6), (6, 12), (12, 18), (18, 24))

# ---------------------------------------------------------------------------
# Supplementary Table 1 - overall counts (algorithm attribution, consecutive
# detections definition), all sites pooled (rotavirus: unvaccinated sites only).
# ---------------------------------------------------------------------------
OVERALL = {
    "rota": {
        "infections": 734, "infection_child_months": 24_228.0,
        "attributable_episodes": 472, "attributable_child_months": 24_716.8,
        "severe_episodes": 142,
    },
    "campy": {
        "infections": 2_938, "infection_child_months": 33_524.7,
        "attributable_episodes": 502, "attributable_child_months": 36_813.2,
        "severe_episodes": 61,
    },
    "crypto": {
        "infections": 1_364, "infection_child_months": 35_513.8,
        "attributable_episodes": 245, "attributable_child_months": 36_848.2,
        "severe_episodes": 33,
    },
}

# ---------------------------------------------------------------------------
# Supplementary Table 2 - infection incidence per 100 child-months by 3-month
# age band (algorithm attribution, consecutive detections).
# ---------------------------------------------------------------------------
INFECTION_RATE_BY_AGE_PER_100CM = {
    "rota":   (2.58, 3.80, 4.67, 4.51, 3.49, 1.87, 1.93, 1.38),
    "campy":  (3.80, 7.95, 12.60, 12.69, 10.60, 9.09, 7.27, 6.26),
    "crypto": (0.45, 1.35, 2.90, 4.58, 5.42, 5.96, 5.04, 4.97),
}

# ---------------------------------------------------------------------------
# Supplementary Table 5 - proportion of infections resulting in pathogen-
# attributable diarrhoea, by 6-month age band (algorithm, consecutive).
# ---------------------------------------------------------------------------
SYMPTOMATIC_FRACTION_BY_AGE = {
    "rota":   (0.58, 0.72, 0.67, 0.50),
    "campy":  (0.22, 0.20, 0.15, 0.11),
    "crypto": (0.22, 0.20, 0.17, 0.17),
}

# ---------------------------------------------------------------------------
# Supplementary Table 7 - proportion of attributable diarrhoeal episodes that
# were severe (score > 6), by 6-month age band (algorithm attribution).
# ---------------------------------------------------------------------------
SEVERE_FRACTION_BY_AGE = {
    "rota":   (0.376, 0.322, 0.279, 0.100),
    "campy":  (0.086, 0.196, 0.079, 0.030),
    "crypto": (0.278, 0.152, 0.155, 0.065),
}

# ---------------------------------------------------------------------------
# Supplementary Table 11 - duration (days) of attributable diarrhoea, median
# (IQR) by 6-month age band (algorithm attribution).
# ---------------------------------------------------------------------------
DIARRHOEA_DURATION_DAYS_BY_AGE = {
    "rota":   ((3.0, 2.0, 6.0), (3.0, 2.0, 4.0), (3.0, 1.0, 4.0), (2.0, 1.0, 3.0)),
    "campy":  ((3.0, 1.0, 5.0), (3.0, 2.0, 5.0), (2.0, 1.0, 4.0), (2.0, 1.0, 3.8)),
    "crypto": ((3.0, 2.0, 7.0), (3.0, 1.2, 5.0), (3.0, 1.0, 6.0), (2.0, 1.0, 3.0)),
}

# ---------------------------------------------------------------------------
# Supplementary Table 4 - age (days) at first infection, median (IQR), all sites.
# ---------------------------------------------------------------------------
AGE_AT_FIRST_INFECTION_DAYS = {
    "rota":   (273.0, 169.0, 396.0),
    "campy":  (244.0, 157.5, 351.5),
    "crypto": (439.5, 307.0, 575.8),
}

# ---------------------------------------------------------------------------
# Supplementary Tables 13 and 14 - per-site infection incidence and
# attributable-diarrhoea incidence per 100 child-months (algorithm attribution;
# infection = consecutive detections). Rotavirus: unvaccinated sites only.
# ---------------------------------------------------------------------------
INFECTION_RATE_BY_SITE_PER_100CM = {
    "rota": {"Bangladesh": 5.94, "India": 2.37, "Nepal": 2.84, "Pakistan": 2.27,
             "Tanzania": 1.87},
    "campy": {"Bangladesh": 16.20, "Brazil": 1.11, "India": 5.11, "Nepal": 11.79,
              "Peru": 12.46, "Pakistan": 5.48, "South Africa": 1.85, "Tanzania": 17.28},
    "crypto": {"Bangladesh": 4.36, "Brazil": 0.52, "India": 2.92, "Nepal": 3.01,
               "Peru": 7.36, "Pakistan": 3.48, "South Africa": 1.98, "Tanzania": 6.68},
}

ATTRIBUTABLE_RATE_BY_SITE_PER_100CM = {
    "rota": {"Bangladesh": 4.40, "India": 1.33, "Nepal": 1.85, "Pakistan": 1.53,
             "Tanzania": 0.44},
    "campy": {"Bangladesh": 1.76, "Brazil": 0.24, "India": 0.78, "Nepal": 1.32,
              "Peru": 4.03, "Pakistan": 1.77, "South Africa": 0.24, "Tanzania": 0.35},
    "crypto": {"Bangladesh": 0.81, "Brazil": 0.03, "India": 0.50, "Nepal": 0.40,
               "Peru": 1.97, "Pakistan": 1.15, "South Africa": 0.06, "Tanzania": 0.11},
}


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def overall_infection_rate_per_child_year(pathogen: str) -> float:
    o = OVERALL[pathogen]
    return o["infections"] / o["infection_child_months"] * 12.0


def overall_attributable_rate_per_child_year(pathogen: str) -> float:
    o = OVERALL[pathogen]
    return o["attributable_episodes"] / o["attributable_child_months"] * 12.0


def overall_symptomatic_fraction(pathogen: str) -> float:
    o = OVERALL[pathogen]
    return o["attributable_episodes"] / o["infections"]


def overall_severe_fraction(pathogen: str) -> float:
    o = OVERALL[pathogen]
    return o["severe_episodes"] / o["attributable_episodes"]


def cumulative_infections_by_age(pathogen: str, age_months):
    """Expected number of infections experienced by age `age_months`.

    Integrates the 3-month-band infection rates of Supplementary Table 2
    (per 100 child-months) from birth, and extrapolates beyond 24 months at the
    21-24 month rate. Accepts a float, numpy array or torch tensor and returns
    the same kind (float64 for numpy, the input dtype for torch).

    Because MAL-ED's monthly sampling under-detects short subclinical
    infections, this is a lower bound on true cumulative infection. It is used
    only to give children a plausible starting immunity at t=0, which shortens
    the burn-in; it is not a target.
    """
    import numpy as np

    rates = [r / 100.0 for r in INFECTION_RATE_BY_AGE_PER_100CM[pathogen]]   # per child-month
    lows = [lo for lo, _ in AGE_BANDS_3MO]
    widths = [hi - lo for lo, hi in AGE_BANDS_3MO]
    last_hi = AGE_BANDS_3MO[-1][1]

    def _accumulate(a, clip_between, clip_below):
        total = 0.0 * a
        for lo, w, r in zip(lows, widths, rates):
            total = total + r * clip_between(a - lo, 0.0, float(w))
        return total + rates[-1] * clip_below(a - last_hi, 0.0)

    try:
        import torch
    except ImportError:   # pragma: no cover
        torch = None
    if torch is not None and isinstance(age_months, torch.Tensor):
        a = age_months.double()
        out = _accumulate(a, lambda x, lo, hi: torch.clamp(x, lo, hi),
                          lambda x, lo: torch.clamp(x, min=lo))
        return out.to(age_months.dtype) if age_months.is_floating_point() else out
    a = np.asarray(age_months, dtype=float)
    out = _accumulate(a, np.clip, lambda x, lo: np.maximum(x, lo))
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Supplementary Table 3 - infections (infant-months) by 3-month age band and
# prior-infection history (algorithm attribution, consecutive detections).
# Each entry: ((naive_infections, naive_infant_months),
#              (prior_infections, prior_infant_months)) for bands
# 1-3, 4-6, ..., 22-24 months of life, i.e. ages [0,3), [3,6), ... months.
# ---------------------------------------------------------------------------
INFECTIONS_BY_PRIOR_HISTORY = {
    "rota": (
        ((70, 2768.5), (3, 60.2)), ((106, 2714.1), (8, 289.7)),
        ((119, 2417.8), (23, 626.0)), ((102, 2120.4), (36, 941.3)),
        ((74, 1892.3), (34, 1199.8)), ((29, 1749.3), (29, 1353.2)),
        ((25, 1663.4), (34, 1395.5)), ((21, 1576.3), (21, 1460.0)),
    ),
    "campy": (
        ((153, 3997.8), (2, 86.1)), ((273, 3674.9), (60, 513.9)),
        ((331, 2878.7), (186, 1224.7)), ((203, 2166.8), (317, 1929.6)),
        ((116, 1751.8), (330, 2454.9)), ((62, 1509.5), (323, 2728.2)),
        ((37, 1390.6), (272, 2858.2)), ((27, 1332.1), (246, 3026.9)),
    ),
    "crypto": (
        ((19, 4238.8), (0, 11.5)), ((58, 4429.7), (3, 93.4)),
        ((118, 4275.5), (14, 283.9)), ((165, 3836.3), (40, 641.0)),
        ((181, 3348.3), (61, 1115.6)), ((166, 2826.0), (97, 1583.2)),
        ((128, 2413.5), (94, 1993.2)), ((118, 2099.0), (102, 2325.0)),
    ),
}


def prior_infection_rate_ratio(pathogen: str, age_lo_months: int, age_hi_months: int) -> float:
    """Incidence in children with >=1 prior infection / incidence in naive
    children, pooled over the 3-month bands within [age_lo, age_hi) months.

    Values > 1 (Campylobacter: 1.5, 2.3, 3.8 at 6-12, 12-18, 18-24 months)
    cannot arise from acquired immunity in a homogeneous population - they are
    the signature of heterogeneous exposure, with the children infected early
    being the high-exposure ones. Values near 1 (Cryptosporidium) or below
    (rotavirus) mean immunity offsets that selection.
    """
    n0 = m0 = n1 = m1 = 0.0
    for (lo, hi), ((a, am), (b, bm)) in zip(AGE_BANDS_3MO, INFECTIONS_BY_PRIOR_HISTORY[pathogen]):
        if lo >= age_lo_months and hi <= age_hi_months:
            n0 += a; m0 += am; n1 += b; m1 += bm
    return (n1 / m1) / (n0 / m0)

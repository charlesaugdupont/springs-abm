# experiments/scenarios/paper1.py
"""
Paper 1 (PLOS Computational Biology): the basic experiments and checks that
need runs of their own, registered as parts of experiments.scenarios.run:

    p1_obs       experiment 8, the observation process
    p1_routes    experiment 7, the transmission routes
    p1_decision  experiment 9, the decision model
    p1_morris    analysis 5, Morris screening of the fixed parameters
    p1_long      analysis 3, ten-year runs for the warm-up

Settings. Every part runs the model's defaults (the calibrated model, the
adopted vaccine of designs.VACCINE_REVISION) on the settlement grid, one
measured year after two years of warm-up (p1_long: eight years after them),
with the experiments' seeds from 61001 (every scenario the same seeds), in the
setting of the data its outputs are compared with (as the calibration did,
decision V5-D2):

* the calibration's setting, without the vaccine ("vacc_none" of 3A), for
  infection, detection and the routes (p1_obs, p1_routes, p1_morris): the
  model was fitted to MAL-ED's unvaccinated cohorts there;
* today's setting, with the vaccine ("vacc_default"), for care-seeking
  (p1_decision): the DHS 2022 targets were checked with the vaccine;
* both for the warm-up (p1_long).

p1_obs's runs are the baseline of p1_routes (the same seeds and settings),
and they repeat the first 16 seeds of p1_3a's "vacc_none" exactly;
p1_decision carries its own baseline, which repeats p1_3a's "vacc_default".

Nothing here changes the model: the route knock-outs set the multipliers the
experiments' hooks already use (Pathogen.route_multipliers), the decision
variants are settings (persona ranges with equal bounds, the CPT exponents,
credit), and the observation designs are arguments of the observation layer.
"""
from __future__ import annotations

import json
import os

import numpy as np

#: label column (designs.LABEL; repeated here so this module does not import designs at load time)
LABEL = "gsa.design_id"
#: seeds of the 16-seed parts (designs.REST_REPS) and of the 40-seed parts (designs.KEY_REPS)
REPS_16 = 16
REPS_40 = 40


def _p1_3a(label: str, reps: int) -> dict:
    """3A's scenario `label` exactly as p1_3a runs it, on `reps` seeds."""
    from experiments.orchestrator import REPS_KEY
    from experiments.scenarios.designs import p1_3a_combos
    by = {c[LABEL]: c for c in p1_3a_combos()}
    return {**by[label], REPS_KEY: reps}


def _body(combo: dict) -> dict:
    """A combo without its label and number of seeds."""
    from experiments.orchestrator import REPS_KEY
    return {k: v for k, v in combo.items() if k not in (LABEL, REPS_KEY)}


# ---------------------------------------------------------------------------
# Experiment 8: the observation process
# ---------------------------------------------------------------------------

#: the observation designs: keyword arguments of experiments.observation.observe
OBS_DESIGNS: dict[str, dict] = {
    "maled": {},                                          # MAL-ED: a stool a month and diarrhoeal stools
    "fortnightly": {"stools_per_month": 2},
    "weekly": {"stools_per_month": 4},
    "daily": {"stools_per_month": 30},                    # every day: the detection windows' limit
    "no_diarrhoeal_stools": {"diarrhoeal_stool_tested_prob": 0.0},
    "all_diarrhoeal_stools": {"diarrhoeal_stool_tested_prob": 1.0},
    "windows_half": {"shedding_scale": 0.5},              # a less sensitive assay
    "windows_double": {"shedding_scale": 2.0},            # a more sensitive one
}
#: the outputs kept for each design ("{p}_obs_<key>" of observe)
OBS_KEYS = (
    "infections_per_child_year_0_24m", "attributable_episodes_per_child_year_0_24m",
    "attributable_fraction_0_24m", "crude_to_consecutive_ratio", "naive_decline_15_24_vs_6_12m",
    "prior_infection_rate_ratio_6_12m", "prior_infection_rate_ratio_12_24m",
    "median_age_first_detected_infection_months", "second_to_first_year_ratio", "infant_ratio_0_6_vs_6_12m",
    *(f"infections_per_child_year_{lo}_{hi}m" for lo, hi in
      ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24))),
)
OBS_PREFIX = "obsd_"


def observation_designs(model, burn_in_days: int = 730) -> dict:
    """Every OBS_DESIGNS design's view of one finished run, keys
    "obsd_<design>_<pathogen>_<key>" (OBS_KEYS). "maled" equals the run's own
    "<pathogen>_obs_<key>"."""
    from experiments.observation import observe
    out = {}
    for name, kw in OBS_DESIGNS.items():
        o = observe(model, burn_in_days, **kw)
        for p in model.pathogens:
            for key in OBS_KEYS:
                out[f"{OBS_PREFIX}{name}_{p.name}_{key}"] = o.get(f"{p.name}_obs_{key}", float("nan"))
    return out


def route_counts(model, burn_in_days: int = 730) -> dict:
    """Infections after the warm-up, everyone, by pathogen and route (counts;
    Pathogen.daily_by_route): "{p}_{route}_infections_post_burnin" and their
    total "{p}_infections_post_burnin". The population's size is fixed, so
    counts compare across scenarios."""
    out = {}
    for p in model.pathogens:
        total = 0.0
        for r, v in (getattr(p, "daily_by_route", None) or {}).items():
            n = float(np.sum(np.asarray(v, dtype=np.float64)[burn_in_days:]))
            out[f"{p.name}_{r}_infections_post_burnin"] = n
            total += n
        out[f"{p.name}_infections_post_burnin"] = total
    return out


def p1_metrics(model) -> dict:
    """The experiments' metrics (common.experiment_metrics) and the infections by route."""
    from experiments.scenarios.common import experiment_metrics
    out = experiment_metrics(model)
    out.update(route_counts(model))
    return out


def p1_obs_metrics(model) -> dict:
    """p1_metrics and the observation designs."""
    out = p1_metrics(model)
    out.update(observation_designs(model))
    return out


def p1_obs_combos() -> list[dict]:
    """The calibration's setting (3A's "vacc_none") on 16 seeds, labelled "baseline"."""
    return [{**_p1_3a("vacc_none", REPS_16), LABEL: "baseline"}]


# ---------------------------------------------------------------------------
# Experiment 7: the transmission routes
# ---------------------------------------------------------------------------

#: every route of every pathogen (Pathogen.ROUTES), as (pathogen, route)
ROUTES: tuple[tuple[str, str], ...] = (
    ("rota", "h2h"), ("rota", "water"), ("rota", "imported"),
    ("campy", "zoonotic"), ("campy", "fecal_oral"), ("campy", "food_borne"),
    ("crypto", "household"), ("crypto", "community"), ("crypto", "water"), ("crypto", "zoonotic"),
    ("crypto", "imported"),
)
ROUTE_OFF_KEY = "hook.route_off"


def route_off(model, day: int, combo: dict) -> None:
    """p1_routes' step_callback: from day 0, the route named by combo["hook.route_off"]
    ("pathogen:route") makes no infection (its multiplier is 0 for the whole run).
    Its random draws are still taken, and what feeds it (shedding into water,
    the animals) goes on."""
    if day != 0 or ROUTE_OFF_KEY not in combo:
        return
    name, route = str(combo[ROUTE_OFF_KEY]).split(":")
    p = next((p for p in model.pathogens if p.name == name), None)
    if p is None or route not in p.ROUTES:
        raise ValueError(f"{ROUTE_OFF_KEY}: no route {route!r} for {name!r}")
    p.route_multipliers[route] = 0.0


def p1_routes_combos() -> list[dict]:
    """Each route switched off in turn, in the calibration's setting, on 16 seeds (the baseline is p1_obs)."""
    base = _p1_3a("vacc_none", REPS_16)
    return [{**base, LABEL: f"off_{p}_{r}", ROUTE_OFF_KEY: f"{p}:{r}"} for p, r in ROUTES]


# ---------------------------------------------------------------------------
# Experiment 9: the decision model
# ---------------------------------------------------------------------------

_EU = {"gamma_range": [1.0, 1.0], "lambda_range": [1.0, 1.0],
       "steering_parameters.cpt_theta": 1.0, "steering_parameters.cpt_eta": 1.0}
#: CPT and its variants. With gamma = lambda = theta = eta = 1 the CPT rule is
#: expected utility (CareSeekingSystem._evaluate_ev; model_tests/test_paper1.py).
#: Without the cash constraint, a visit the household's cash does not cover is
#: bought on credit (EconomyConfig.emergency_credit_days without a limit): the
#: cost still enters the decision, but cash no longer blocks it.
DECISION_VARIANTS: dict[str, dict] = {
    "cpt": {},
    "no_probability_weighting": {"gamma_range": [1.0, 1.0]},
    "no_loss_aversion": {"lambda_range": [1.0, 1.0]},
    "linear_value": {"steering_parameters.cpt_theta": 1.0, "steering_parameters.cpt_eta": 1.0},
    "expected_utility": _EU,
    "no_cash_constraint": {"economy.emergency_credit_days": 1.0e6},
}


def p1_decision_combos() -> list[dict]:
    """The variants in today's setting (3A's "vacc_default"), 40 seeds each; "cpt" is the model."""
    base = _p1_3a("vacc_default", REPS_40)
    out = []
    for name, over in DECISION_VARIANTS.items():
        clash = set(over) & set(_body(base))
        if clash:
            raise ValueError(f"{name} overrides a scenario setting: {sorted(clash)}")
        out.append({**base, LABEL: name, **over})
    return out


# ---------------------------------------------------------------------------
# Analysis 5: Morris screening of the fixed parameters
# ---------------------------------------------------------------------------

#: the fixed (not calibrated) parameters screened, with their ranges.
#: (name, low, high, scale, kind): the value is low..high on a linear or log
#: scale; kind maps it to config overrides (_factor_overrides). Ranges: the
#: literature range in the parameter's description where there is one,
#: otherwise about half to twice the value (log) or +-50% (linear).
MORRIS_FACTORS: tuple[tuple[str, float, float, str, str], ...] = (
    # the decision (CPT and the parent's stakes)
    ("cpt_alpha_centre", 0.4, 0.6, "lin", "alpha_centre"),        # alpha_range [c-h, c+h] in [0.05, 0.95]; 0.5
    ("cpt_gamma_centre", 0.5, 0.75, "lin", "gamma_centre"),       # gamma_range [c-0.25, c+0.25]; 0.65
    ("cpt_lambda_upper", 2.0, 4.0, "lin", "lambda_upper"),        # lambda_range [1, u]; 3
    ("cpt_theta_eta", 0.7, 1.0, "lin", "theta_eta"),              # 0.88 (Tversky & Kahneman 1992)
    ("steering_parameters.child_health_weight", 0.3, 0.7, "lin", "float"),           # 0.5
    ("steering_parameters.natural_worsening_prob", 0.2, 0.5, "lin", "float"),        # 0.35
    ("steering_parameters.parent_stress_health_impact", 0.15, 0.45, "lin", "float"),  # 0.3
    ("steering_parameters.untreated_severity_penalty", 0.1, 0.3, "lin", "float"),    # 0.2
    ("economy.wealth_scale_days", 15.0, 60.0, "log", "float"),                       # 30
    # treatment
    ("steering_parameters.treatment_success_prob", 0.6, 0.95, "lin", "float"),       # 0.8
    ("steering_parameters.treatment_harm_reduction", 0.8, 1.0, "lin", "float"),      # 0.93
    ("pathogens[campy].treatment_clears_infection_prob", 0.5, 1.0, "lin", "float"),  # 1.0
    # natural history: latent periods (4 levels fall on whole days) and infectious periods
    ("pathogens[rota].exposure_period", 1.0, 4.0, "lin", "int"),                     # 2
    ("pathogens[campy].exposure_period", 2.0, 5.0, "lin", "int"),                    # 3
    ("pathogens[crypto].exposure_period", 4.0, 10.0, "lin", "int"),                  # 7 (range 2-10)
    ("pathogens[rota].recovery_rate", 1.0 / 7.0, 1.0 / 3.0, "log", "float"),         # 0.3005 (3-7 days)
    ("pathogens[campy].recovery_rate", 0.1, 0.22, "log", "float"),                   # 0.1466
    ("pathogens[crypto].recovery_rate", 1.0 / 31.0, 1.0 / 16.0, "log", "float"),     # 0.0476 (16-31 days)
    # immunity against infection and against illness
    ("pathogens[rota].immunity_decay_per_infection", 0.3, 0.6, "lin", "float"),      # 0.45
    ("steering_parameters.prior_infection_immunity_factor", 0.075, 0.3, "log", "float"),  # 0.15 (campy)
    ("pathogens[crypto].immunity_decay_per_infection", 0.2, 0.4, "lin", "float"),    # 0.29
    ("pathogens[rota].symptomatic_reduction_per_prior_infection", 0.3, 0.7, "lin", "float"),    # 0.5
    ("pathogens[crypto].symptomatic_reduction_per_prior_infection", 0.3, 0.7, "lin", "float"),  # 0.48
    ("illness_mechanics.severity_reduction_per_infection", 0.1, 0.3, "lin", "float"),           # 0.2
    # the environment: die-off in water and the shape of Campylobacter's dose-response
    ("pathogens[rota].water_decay_rate", 0.125, 0.5, "log", "float"),                # 0.25
    ("pathogens[crypto].oocyst_decay_rate", 0.075, 0.3, "log", "float"),             # 0.15
    ("campy_beta_poisson_alpha", 0.019, 0.076, "log", "beta_poisson_alpha"),         # 0.038, alpha/beta held
    # contacts
    ("steering_parameters.max_contacts_per_visit", 2.0, 7.2, "log", "float"),        # 3.5
)
MORRIS_LEVELS = 4
MORRIS_TRAJECTORIES = 9
MORRIS_CANDIDATES = 100
MORRIS_SEED = 61_001
MORRIS_REPS = 2
MORRIS_DESIGN_FILE = os.path.join(os.path.dirname(__file__), "p1_morris_design.json")
#: Campylobacter's Beta-Poisson low-dose slope alpha / beta, held when alpha varies
_CAMPY_ALPHA_OVER_BETA = 0.038 / 0.022


def _factor_value(u: float, low: float, high: float, scale: str) -> float:
    if scale == "log":
        return float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
    return float(low + u * (high - low))


def _factor_overrides(name: str, value: float, kind: str) -> dict:
    if kind == "float":
        return {name: value}
    if kind == "int":
        return {name: int(round(value))}
    if kind == "alpha_centre":
        # the model's half-width 0.4 ([0.1, 0.9] at 0.5), narrowed to keep alpha in [0.05, 0.95]
        h = min(0.4, value - 0.05, 0.95 - value)
        return {"alpha_range": [value - h, value + h]}
    if kind == "gamma_centre":
        return {"gamma_range": [value - 0.25, value + 0.25]}
    if kind == "lambda_upper":
        return {"lambda_range": [1.0, value]}
    if kind == "theta_eta":
        return {"steering_parameters.cpt_theta": value, "steering_parameters.cpt_eta": value}
    if kind == "beta_poisson_alpha":
        return {"pathogens[campy].beta_poisson_alpha": value,
                "pathogens[campy].beta_poisson_beta": value / _CAMPY_ALPHA_OVER_BETA}
    raise ValueError(f"unknown factor kind {kind!r}")


def morris_problem() -> dict:
    """SALib's problem: every factor on [0, 1] (mapped by MORRIS_FACTORS)."""
    return {"num_vars": len(MORRIS_FACTORS), "names": [f[0] for f in MORRIS_FACTORS],
            "bounds": [[0.0, 1.0]] * len(MORRIS_FACTORS)}


def make_morris_design() -> dict:
    """Draws the trajectories (SALib, Campolongo et al. 2007's selection of the
    MORRIS_TRAJECTORIES most spread out of MORRIS_CANDIDATES). Run once; the
    design is kept in p1_morris_design.json, so the runs and their analysis never
    depend on the installed SALib's sampler."""
    from SALib.sample import morris as morris_sample
    X = morris_sample.sample(morris_problem(), N=MORRIS_CANDIDATES, num_levels=MORRIS_LEVELS,
                             optimal_trajectories=MORRIS_TRAJECTORIES, local_optimization=True,
                             seed=MORRIS_SEED)
    return {"factors": [list(f) for f in MORRIS_FACTORS], "levels": MORRIS_LEVELS,
            "trajectories": MORRIS_TRAJECTORIES, "candidates": MORRIS_CANDIDATES, "seed": MORRIS_SEED,
            "X": (np.round(np.asarray(X, dtype=float) * (MORRIS_LEVELS - 1)) / (MORRIS_LEVELS - 1)).tolist()}


def load_morris_design(path: str = MORRIS_DESIGN_FILE) -> np.ndarray:
    """The saved trajectories (rows: points, columns: MORRIS_FACTORS, on [0, 1]);
    refuses a file made for other factors."""
    with open(path) as f:
        d = json.load(f)
    if [tuple(x) for x in d["factors"]] != [tuple(f) for f in MORRIS_FACTORS] or d["levels"] != MORRIS_LEVELS:
        raise ValueError(f"{path} was made for other factors; rerun make_morris_design()")
    X = np.asarray(d["X"], dtype=float)
    if X.shape != (d["trajectories"] * (len(MORRIS_FACTORS) + 1), len(MORRIS_FACTORS)):
        raise ValueError(f"{path}: the design has shape {X.shape}")
    return X


def morris_point(u_row) -> dict:
    """Config overrides of one design point."""
    out = {}
    for u, (name, low, high, scale, kind) in zip(u_row, MORRIS_FACTORS):
        out.update(_factor_overrides(name, _factor_value(float(u), low, high, scale), kind))
    return out


def p1_morris_combos() -> list[dict]:
    """Every design point in the calibration's setting, on 2 seeds (the same two for every point)."""
    from experiments.orchestrator import REPS_KEY
    base = _p1_3a("vacc_none", MORRIS_REPS)
    X = load_morris_design()
    out = []
    for i, row in enumerate(X):
        over = morris_point(row)
        clash = set(over) & set(_body(base))
        if clash:
            raise ValueError(f"a factor overrides a scenario setting: {sorted(clash)}")
        out.append({**base, LABEL: f"morris_{i:03d}", **over, REPS_KEY: MORRIS_REPS})
    return out


def morris_metrics(model) -> dict:
    """The validation metrics of campaign v5, the care outcomes and the infections by route, without the
    weekly series."""
    from experiments.calibration.validate_calibration import validation_metrics_v5
    from experiments.scenarios.common import care_metrics, rota_by_quintile
    out = validation_metrics_v5(model)
    out.update(care_metrics(model))
    out.update(rota_by_quintile(model))
    out.update(route_counts(model))
    return out


# ---------------------------------------------------------------------------
# Analysis 3: the warm-up
# ---------------------------------------------------------------------------

LONG_STEPS = 3_650
LONG_REPS = 8


def _weekly_sum(daily) -> list[float]:
    from experiments.scenarios.common import _weekly
    return _weekly(np.asarray(daily, dtype=np.float64))


def epi_weekly_series(model) -> dict[str, list[float]]:
    """Weekly series of the epidemiological state over the whole run (week 0
    starts on day 0): infections in everyone and in children under 5 and under
    2, the child-years they are counted over, and under-5 carriage."""
    out: dict[str, list[float]] = {}
    child_days = np.asarray(getattr(model, "child_days_by_age_month", []), dtype=np.float64)
    if child_days.ndim != 2 or child_days.shape[0] == 0:
        return out
    out["wk_child_years_0_60m"] = (np.asarray(_weekly_sum(child_days.sum(axis=1))) / 365.0).tolist()
    out["wk_child_years_0_24m"] = (np.asarray(_weekly_sum(child_days[:, :24].sum(axis=1))) / 365.0).tolist()
    for p in model.pathogens:
        inf = np.asarray(model.new_infections_by_age_month.get(p.name, []), dtype=np.float64)
        if inf.ndim == 2 and inf.shape[0]:
            out[f"wk_inf_{p.name}_0_60m"] = _weekly_sum(inf.sum(axis=1))
            out[f"wk_inf_{p.name}_0_24m"] = _weekly_sum(inf[:, :24].sum(axis=1))
        if p.daily_by_route:
            total = np.sum([np.asarray(v, dtype=np.float64) for v in p.daily_by_route.values()], axis=0)
            out[f"wk_inf_{p.name}_all"] = _weekly_sum(total)
        prev = np.asarray(model.u5_prevalence_history.get(p.name, []), dtype=np.float64)
        if prev.size:
            from experiments.scenarios.common import _weekly
            out[f"wk_prev_{p.name}_u5"] = _weekly(prev, "mean")
    return out


def long_run_metrics(model) -> dict:
    """p1_metrics and the weekly epidemiological series."""
    out = p1_metrics(model)
    out.update(epi_weekly_series(model))
    return out


def p1_long_combos() -> list[dict]:
    """Ten years in both settings, 8 seeds each."""
    return [{**_p1_3a(lab, LONG_REPS), LABEL: f"long_{lab}"} for lab in ("vacc_none", "vacc_default")]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="paper 1's Morris design")
    ap.add_argument("command", choices=["morris-design"])
    args = ap.parse_args()
    if args.command == "morris-design":
        d = make_morris_design()
        with open(MORRIS_DESIGN_FILE, "w") as f:
            json.dump(d, f, indent=1)
        print(f"wrote {MORRIS_DESIGN_FILE}: {len(d['X'])} points x {len(MORRIS_FACTORS)} factors")

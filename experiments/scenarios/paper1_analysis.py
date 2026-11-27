# experiments/scenarios/paper1_analysis.py
"""
Paper 1's numbers and tables, from the runs of p1_3a, p1_ps and the parts of
experiments.scenarios.paper1, and the docking's output:

    python -m experiments.scenarios.paper1_analysis              # every part with results
    python -m experiments.scenarios.paper1_analysis 3a routes    # some parts

Parts: calibration (1: the history matching's diagnostics, from the tables
experiments.calibration.diagnostics_v5 writes), 3a (the vaccine, and the check
that p1_3a reproduces the 3A rerun's reported effects), ps (the calibrated parameter sets), obs (experiment 8),
routes (7), decision (9), morris (5), seeds (3: the number of seeds), warmup (3:
the ten-year runs), docking (6).

Each part writes CSV tables to experiments/outputs/experiments_v5/paper1/ and
adds the numbers quoted in the text to values_p1.tex there
(\\newcommand{\\pv<Name>}{<value>}). Contrasts are paired by seed (common random
numbers): a relative change is mean(a - b) / mean(b), its standard error the SD
of the paired differences / sqrt(n) / mean(b).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import numpy as np
import pandas as pd

LABEL = "gsa.design_id"
RESULTS = os.path.join("experiments", "outputs", "experiments_v5")
OUT = os.path.join(RESULTS, "paper1")
DOCKING = os.path.join("experiments", "outputs", "verification", "seir_docking")
CALIBRATION_DIAG = os.path.join("experiments", "outputs", "history_matching_v5", "diagnostics")
PATHOGENS = ("rota", "campy", "crypto")
NAMES = {"rota": "Rota", "campy": "Campy", "crypto": "Crypto"}

DESIGN_DIRS = {"3a": "p1_3a_vaccine", "ps": "p1_ps_param_sets", "obs": "p1_obs_observation",
               "routes": "p1_routes_knockouts", "decision": "p1_decision_model", "morris": "p1_morris_fixed",
               "long": "p1_long_warmup"}

VALUES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(part: str, results: str | None = None) -> pd.DataFrame | None:
    path = os.path.join(results or RESULTS, DESIGN_DIRS[part], "results.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def by_seed(df: pd.DataFrame, label: str) -> pd.DataFrame:
    g = df[df[LABEL] == label]
    if g.empty:
        raise KeyError(f"no runs for {label}")
    return g.set_index("seed").sort_index()


def pair(a: pd.Series, b: pd.Series) -> dict:
    """Paired contrast of two series indexed by seed."""
    s = a.index.intersection(b.index)
    a, b = a.loc[s].astype(float), b.loc[s].astype(float)
    d = a - b
    n = int(d.notna().sum())
    ref = float(b.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    rel = float(d.mean() / ref) if ref else float("nan")
    return {"a": float(a.mean()), "b": ref, "diff": float(d.mean()), "se": se, "rel": rel,
            "rel_se": se / abs(ref) if ref else float("nan"), "n": n, "lower": int((d < 0).sum()),
            "higher": int((d > 0).sum())}


def macro(name: str, value, nd: int = 0) -> None:
    """A number for the text: \\pv<name>, rounded to nd decimals (nd < 0: to tens, ...)."""
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"a LaTeX macro name has letters only: {name!r}")
    if isinstance(value, str):
        VALUES[name] = value
        return
    v = float(value)
    if not math.isfinite(v):
        VALUES[name] = "NaN"
        return
    VALUES[name] = f"{round(v, nd):,.{max(nd, 0)}f}".replace(",", "{,}")


def save(df: pd.DataFrame, name: str, out: str | None = None) -> str:
    out = out or OUT
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def write_values(out: str | None = None) -> str:
    out = out or OUT
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "values_p1.tex")
    old = {}
    if os.path.exists(path):           # keep the numbers of parts not rerun this time
        for line in open(path):
            m = re.match(r"\\newcommand\{\\pv([A-Za-z]+)\}\{(.*)\}$", line.strip())
            if m:
                old[m.group(1)] = m.group(2)
    old.update(VALUES)
    with open(path, "w") as f:
        f.write("%% Numbers quoted in paper 1, written by experiments/scenarios/paper1_analysis.py - do not edit.\n")
        for k in sorted(old):
            f.write(f"\\newcommand{{\\pv{k}}}{{{old[k]}}}\n")
    return path


def targets() -> dict:
    """Campaign v5's targets on the observed scale: {metric: (value, sd_obs, sd_md, transform, eps)}."""
    from experiments.calibration.history_matching import build_targets_v5
    out = {}
    for t in build_targets_v5():
        v = math.exp(t.z) - t.eps if t.transform == "log" else t.z
        out[t.metric] = (v, t.sd_obs, t.sd_md, t.transform, t.eps)
    return out


def add_severe(df: pd.DataFrame) -> pd.DataFrame:
    """sev59: rotavirus episodes of initial severity 0.6 or more, 0-59 months (count after the warm-up)."""
    cols = [f"rota_episodes_sev06_{b}" for b in ("0_12m", "12_24m", "24_60m")]
    if all(c in df for c in cols):
        df = df.copy()
        df["sev59"] = df[cols].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# 3A: the vaccine, and the check against the 3A rerun's reported effects
# ---------------------------------------------------------------------------

EP24 = "rota_symptomatic_episodes_per_child_year_0_24m"
EP59 = "rota_symptomatic_episodes_per_child_year_0_60m"
INF24 = "rota_infections_per_child_year_0_24m"
DET24 = "rota_obs_attributable_episodes_per_child_year_0_24m"
ADULT = "rota_adult_prevalence_post_burnin"
VACC_OUTCOMES = {"episodes 0-24 m": EP24, "episodes 0-59 m": EP59, "severe episodes 0-59 m": "sev59",
                 "infections 0-24 m": INF24, "detected episodes 0-24 m": DET24, "adults' carriage": ADULT}

#: the 3A rerun's numbers as reported (check_claims.py, part 3a): id -> (quote, cited, decimals)
REPORTED_3A = {
    "r1": ("today: episodes 0-24 m fall by 38%", 38, 0), "r1se": ("(+-3%)", 3, 0),
    "r1n": ("lower in 38 of 40 seeds", 38, 0), "r2": ("all episodes 0-59 m by 35%", 35, 0),
    "r2se": ("(+-3%)", 3, 0), "r3": ("severe 0-59 m by 56%", 56, 0), "r3se": ("(+-3%)", 3, 0),
    "r4": ("infections 0-24 m by only 14%", 14, 0), "r4se": ("(+-3%)", 3, 0),
    "r5": ("adults' carriage -5%", -5, 0), "r5se": ("(+-3%)", 3, 0),
    "m1": ("detected episodes 0-24 m fall by 36%", 36, 0), "m1se": ("(+-3%)", 3, 0),
    "c1": ("95% protection lowers episodes 0-24 m by 22%", 22, 0), "c1se": ("(+-3%)", 3, 0),
    "c1n": ("lower in 36 of 40", 36, 0),
    "s1": ("without the extra prior infection episodes 0-24 m fall by 13%", 13, 0),
    "s2": ("about as much as infections (11%)", 11, 0),
    "s3": ("two thirds of the effect rests on it", 66, -1),
    "s4": ("severe episodes still fall by 48%", 48, 0),
    "s5": ("without the cut in severity severe fall by 38%", 38, 0),
    "s6": ("a third of that effect rests on it", 33, -1),
    "s8": ("waning over four years: episodes fall by 32%", 32, 0),
    "s9": ("and severe ones by 49%", 49, 0), "s10": ("10% more episodes", 10, 0), "s10se": ("(+-4%)", 4, 0),
    "s11": ("17% more severe ones", 17, 0), "s11se": ("(+-5%)", 5, 0),
}


def part_3a(df: pd.DataFrame) -> dict:
    df = add_severe(df)
    none = by_seed(df, "vacc_none")
    rows = []
    for sc in sorted(set(df[LABEL]) - {"vacc_none"}):
        a = by_seed(df, sc)
        for name, m in VACC_OUTCOMES.items():
            p = pair(a[m], none[m])
            rows.append({"scenario": sc, "outcome": name, "metric": m, "none": p["b"], "with": p["a"],
                         "change_pct": 100 * p["rel"], "se_pct": 100 * p["rel_se"], "lower": p["lower"], "n": p["n"]})
    eff = pd.DataFrame(rows)
    save(eff, "p1_3a_effects")

    def fall(m, sc, ref="vacc_none"):
        return -pair(by_seed(df, sc)[m], by_seed(df, ref)[m])["rel"]
    got = {}
    p = pair(by_seed(df, "vacc_default")[EP24], none[EP24])
    got.update(r1=-100 * p["rel"], r1se=100 * p["rel_se"], r1n=p["lower"])
    for k, m in (("r2", EP59), ("r3", "sev59"), ("r4", INF24), ("m1", DET24)):
        p = pair(by_seed(df, "vacc_default")[m], none[m])
        got[k], got[k + "se"] = -100 * p["rel"], 100 * p["rel_se"]
    p = pair(by_seed(df, "vacc_default")[ADULT], none[ADULT])
    got["r5"], got["r5se"] = 100 * p["rel"], 100 * p["rel_se"]
    if "grid_v0.95_d616" in set(df[LABEL]):
        p = pair(by_seed(df, "grid_v0.95_d616")[EP24], by_seed(df, "vacc_default")[EP24])
        got.update(c1=-100 * p["rel"], c1se=100 * p["rel_se"], c1n=p["lower"])
    if "vacc_no_prior_infection_equivalent" in set(df[LABEL]):
        sc = "vacc_no_prior_infection_equivalent"
        got.update(s1=100 * fall(EP24, sc), s2=100 * fall(INF24, sc),
                   s3=100 * (1 - fall(EP24, sc) / fall(EP24, "vacc_default")), s4=100 * fall("sev59", sc))
    if "vacc_no_severity_reduction" in set(df[LABEL]):
        sc = "vacc_no_severity_reduction"
        got.update(s5=100 * fall("sev59", sc), s6=100 * (1 - fall("sev59", sc) / fall("sev59", "vacc_default")))
    if "vacc_illness_wanes_4y" in set(df[LABEL]):
        sc = "vacc_illness_wanes_4y"
        got.update(s8=100 * fall(EP24, sc), s9=100 * fall("sev59", sc))
        for k, m in (("s10", EP24), ("s11", "sev59")):
            p = pair(by_seed(df, sc)[m], by_seed(df, "vacc_default")[m])
            got[k], got[k + "se"] = 100 * p["rel"], 100 * p["rel_se"]
    chk = []
    for k, (quote, cited, nd) in REPORTED_3A.items():
        if k not in got:
            continue
        v = got[k]
        ok = bool(np.isfinite(v) and abs(v - cited) <= 0.5 * 10 ** (-nd) + 1e-9)
        chk.append({"id": k, "quote": quote, "cited": cited, "computed": round(float(v), 3), "ok": ok})
    chk = pd.DataFrame(chk)
    save(chk, "p1_3a_reproduction")
    n_seeds = int(df.groupby(LABEL).size().min())
    macro("VaccSeeds", n_seeds)
    macro("VaccEpisodesFall", got["r1"]); macro("VaccEpisodesFallSE", got["r1se"])
    macro("VaccEpisodesLowerSeeds", got["r1n"])
    macro("VaccSevereFall", got["r3"]); macro("VaccSevereFallSE", got["r3se"])
    macro("VaccInfectionsFall", got["r4"]); macro("VaccDetectedEpisodesFall", got["m1"])
    # experiment 10: what the vaccine's effect rests on (the 3A rerun's s-numbers)
    for k, name in (("s1", "VaccNoPriorEpisodesFall"), ("s2", "VaccNoPriorInfectionsFall"), ("s3", "VaccPriorSharePct"),
                    ("s4", "VaccNoPriorSevereFall"), ("s5", "VaccNoCutSevereFall"), ("s6", "VaccCutSharePct"),
                    ("s8", "VaccWaningEpisodesFall"), ("s9", "VaccWaningSevereFall"), ("s10", "VaccWaningEpisodesMore"),
                    ("s10se", "VaccWaningEpisodesMoreSE"), ("s11", "VaccWaningSevereMore"),
                    ("s11se", "VaccWaningSevereMoreSE"), ("c1", "VaccFullProtectionEpisodesFall")):
        if k in got:
            macro(name, got[k])
    return {"checked": len(chk), "reproduced": int(chk["ok"].sum()) if len(chk) else 0,
            "not_reproduced": chk.loc[~chk["ok"], "id"].tolist() if len(chk) else []}


# ---------------------------------------------------------------------------
# Parameter uncertainty: the calibrated parameter sets
# ---------------------------------------------------------------------------

def _route_fractions() -> list[str]:
    from experiments.scenarios.paper1 import ROUTES
    return [f"{p}_{r}_fraction_post_burnin" for p, r in ROUTES]


PS_OUTPUTS_NONE = ([f"{p}_infections_per_child_year_0_24m" for p in PATHOGENS]
                   + [f"{p}_symptomatic_episodes_per_child_year_0_24m" for p in PATHOGENS]
                   + [f"{p}_obs_infections_per_child_year_0_24m" for p in PATHOGENS]
                   + [f"{p}_obs_attributable_fraction_0_24m" for p in PATHOGENS]
                   + _route_fractions())
PS_OUTPUTS_TODAY = ["episode_care_seeking_rate"] + [f"econ_care_seeking_rate_q{q}" for q in range(1, 6)]


def _blocked_spread(Y: pd.DataFrame) -> dict:
    """The spread of the sets' means (rows: sets, columns: seeds) against the seeds' noise. Every set
    ran on the same seeds (common random numbers), so each seed is a block: the part of a seed's
    effect that every set shares cancels between sets, and the noise in a comparison of sets is the
    sets-by-seeds interaction, the residual of a two-way decomposition without replication (as the
    Morris part's noise floor takes it from its seed pairs). The one-way estimate, which counts the
    seeds' whole variance as noise, is kept for comparison (it overstates the noise when seeds are
    shared). Sets with a missing run are left out."""
    Y = Y[np.isfinite(Y.to_numpy(dtype=float)).all(axis=1)]
    A = Y.to_numpy(dtype=float)
    S, R = A.shape
    if S < 2 or R < 2:
        return {"sets_used": S, "sd_between_sets": float("nan"), "sd_parameters": float("nan"),
                "se_set_mean": float("nan"), "se_set_mean_independent": float("nan"),
                "share_from_parameters": float("nan")}
    means = A.mean(axis=1)
    resid = A - means[:, None] - A.mean(axis=0)[None, :] + A.mean()
    noise = float((resid ** 2).sum() / ((S - 1) * (R - 1))) / R
    indep = float(np.mean(A.var(axis=1, ddof=1))) / R
    between = float(np.var(means, ddof=1))
    params = max(between - noise, 0.0)
    return {"sets_used": S, "sd_between_sets": math.sqrt(between), "sd_parameters": math.sqrt(params),
            "se_set_mean": math.sqrt(noise), "se_set_mean_independent": math.sqrt(indep),
            "share_from_parameters": params / between if between > 0 else float("nan")}


def part_ps(ps: pd.DataFrame, a3: pd.DataFrame) -> dict:
    """Each set's outputs (no vaccine for infection and detection, today's for care-seeking) and its
    vaccine effect, the adopted set's from p1_3a's first 8 seeds; then the spread between the sets
    against the seeds' noise (_blocked_spread)."""
    from experiments.scenarios.param_sets import ADOPTED, PS_REPS, SCENARIO_KEY, SET_KEY
    ps, a3 = add_severe(ps), add_severe(a3)
    seeds = sorted(set(ps["seed"]))
    adopted = a3[a3["seed"].isin(seeds[:PS_REPS])].copy()
    adopted[SET_KEY], adopted[SCENARIO_KEY] = ADOPTED, adopted[LABEL]
    allp = pd.concat([adopted[adopted[LABEL].isin(["vacc_none", "vacc_default"])], ps], ignore_index=True)
    rows = []
    for s, g in allp.groupby(SET_KEY):
        none = g[g[SCENARIO_KEY] == "vacc_none"].set_index("seed")
        today = g[g[SCENARIO_KEY] == "vacc_default"].set_index("seed")
        row = {"set": s, "adopted": s == ADOPTED, "n": len(none)}
        for m in PS_OUTPUTS_NONE:
            row[m] = float(none[m].mean())
        for m in PS_OUTPUTS_TODAY:
            row[m] = float(today[m].mean())
        for m, name in ((EP24, "vacc_fall_episodes_0_24m"), ("sev59", "vacc_fall_severe_0_59m")):
            p = pair(today[m], none[m])
            row[name], row[name + "_se"] = -100 * p["rel"], 100 * p["rel_se"]
        rows.append(row)
    sets = pd.DataFrame(rows).sort_values(["adopted", "set"], ascending=[False, True])
    save(sets, "p1_ps_sets")
    # the spread between the sets against the seeds' noise, seeds as blocks
    def grid(scen, m):
        return allp[allp[SCENARIO_KEY] == scen].pivot(index=SET_KEY, columns="seed", values=m).astype(float)
    rows = []
    for m in PS_OUTPUTS_NONE + PS_OUTPUTS_TODAY + ["vacc_fall_episodes_0_24m", "vacc_fall_severe_0_59m"]:
        if m.startswith("vacc_fall"):
            src = EP24 if m == "vacc_fall_episodes_0_24m" else "sev59"
            none, today = grid("vacc_none", src), grid("vacc_default", src)
            Y = -100 * (today - none).div(none.mean(axis=1), axis=0)     # each seed's share of the set's fall
        else:
            Y = grid("vacc_default" if m in PS_OUTPUTS_TODAY else "vacc_none", m)
        rows.append({"output": m, "adopted": float(sets.loc[sets.adopted, m].iloc[0]), "min": float(sets[m].min()),
                     "max": float(sets[m].max()), **_blocked_spread(Y)})
    spread = pd.DataFrame(rows)
    # the rule set before the runs (experiments.scenarios.param_sets): an effect is tested if it exceeds
    # 2 SE at the adopted set over all its seeds (p1_3a), and is robust if, in at least ROBUST_SHARE of
    # the sets (the adopted one on the same 8 seeds included), it has that sign and exceeds 2 SE
    from experiments.scenarios.param_sets import ROBUST_SHARE
    for m, src in (("vacc_fall_episodes_0_24m", EP24), ("vacc_fall_severe_0_59m", "sev59")):
        full = pair(by_seed(a3, "vacc_default")[src], by_seed(a3, "vacc_none")[src])
        meets = (np.sign(-sets[m]) == np.sign(full["diff"])) & (sets[m].abs() > 2 * sets[m + "_se"])
        sets[m + "_meets_rule"] = meets
        i = spread.index[spread.output == m][0]
        spread.loc[i, "adopted_all_seeds"] = -100 * full["rel"]
        spread.loc[i, "tested"] = bool(abs(full["diff"]) > 2 * full["se"])
        spread.loc[i, "sets_meeting"] = int(meets.sum())
        spread.loc[i, "robust"] = bool(spread.loc[i, "tested"] and meets.mean() >= ROBUST_SHARE)
    save(sets, "p1_ps_sets")
    save(spread, "p1_ps_spread")
    macro("PsRobustSharePct", 100 * ROBUST_SHARE)
    from experiments.scenarios.paper1 import ROUTES
    camel = {"h2h": "PersonToPerson", "fecal_oral": "FecalOral", "food_borne": "FoodBorne"}
    for p, r in ROUTES:
        col = f"{p}_{r}_fraction_post_burnin"
        name = f"Ps{NAMES[p]}{camel.get(r, r.capitalize())}"
        macro(name + "Min", 100 * sets[col].min())
        macro(name + "Max", 100 * sets[col].max())
    for p in PATHOGENS:
        col = f"{p}_infections_per_child_year_0_24m"
        macro(f"Ps{NAMES[p]}InfectionsMin", sets[col].min(), 2)
        macro(f"Ps{NAMES[p]}InfectionsMax", sets[col].max(), 2)
    # what the data do not observe (true incidence, the main routes): the share of the spread between sets
    # that comes from the parameters rather than the seeds
    unobserved = ([f"{p}_infections_per_child_year_0_24m" for p in PATHOGENS]
                  + ["campy_zoonotic_fraction_post_burnin", "crypto_household_fraction_post_burnin",
                     "crypto_community_fraction_post_burnin", "crypto_water_fraction_post_burnin"])
    macro("PsUnobservedShareParamsMin",
          100 * spread.set_index("output").loc[unobserved, "share_from_parameters"].min())
    macro("PsVaccFallSetsMeeting", spread.loc[spread.output == "vacc_fall_episodes_0_24m", "sets_meeting"].iloc[0])
    macro("PsVaccSevereFallSetsMeeting", spread.loc[spread.output == "vacc_fall_severe_0_59m", "sets_meeting"].iloc[0])
    v = sets["vacc_fall_episodes_0_24m"]
    macro("PsSets", len(sets)); macro("PsVaccFallMin", v.min()); macro("PsVaccFallMax", v.max())
    s = sets["vacc_fall_severe_0_59m"]
    macro("PsVaccSevereFallMin", s.min()); macro("PsVaccSevereFallMax", s.max())
    sp = spread.set_index("output")
    for name, m in (("PsVaccFall", "vacc_fall_episodes_0_24m"), ("PsVaccSevereFall", "vacc_fall_severe_0_59m")):
        macro(name + "ShareParams", 100 * sp.loc[m, "share_from_parameters"])
        macro(name + "SDParams", sp.loc[m, "sd_parameters"])        # percentage points
        macro(name + "SESet", sp.loc[m, "se_set_mean"])             # one set's 8 seeds, percentage points
    return {"sets": len(sets), "complete_sets": int(spread["sets_used"].min())}


# ---------------------------------------------------------------------------
# Experiment 8: the observation process
# ---------------------------------------------------------------------------

def part_obs(obs: pd.DataFrame) -> dict:
    from experiments.scenarios.paper1 import OBS_DESIGNS, OBS_KEYS, OBS_PREFIX
    tg = targets()
    base = obs[obs[LABEL] == "baseline"]
    rows = []
    for d in OBS_DESIGNS:
        for p in PATHOGENS:
            true_inf = base[f"{p}_infections_per_child_year_0_24m"].astype(float)
            true_ep = base[f"{p}_symptomatic_episodes_per_child_year_0_24m"].astype(float)
            det = base[f"{OBS_PREFIX}{d}_{p}_infections_per_child_year_0_24m"].astype(float)
            att = base[f"{OBS_PREFIX}{d}_{p}_attributable_episodes_per_child_year_0_24m"].astype(float)
            row = {"design": d, "pathogen": p, "true_infections": true_inf.mean(), "detected_infections": det.mean(),
                   "detected_sd": det.std(ddof=1), "detected_over_true": (det / true_inf).mean(),
                   "true_episodes": true_ep.mean(), "detected_episodes": att.mean(),
                   "detected_episodes_over_true": (att / true_ep).mean()}
            for k in OBS_KEYS:
                if k in ("infections_per_child_year_0_24m", "attributable_episodes_per_child_year_0_24m"):
                    continue
                row[k] = float(base[f"{OBS_PREFIX}{d}_{p}_{k}"].astype(float).mean())
            if d == "maled":
                for k in ("infections_per_child_year_0_24m", "attributable_episodes_per_child_year_0_24m",
                          "crude_to_consecutive_ratio", "naive_decline_15_24_vs_6_12m",
                          "median_age_first_detected_infection_months"):
                    t = tg.get(f"{p}_obs_{k}")
                    row[f"maled_{k}"] = t[0] if t else float("nan")
            rows.append(row)
    tab = pd.DataFrame(rows)
    save(tab, "p1_obs_designs")
    # by age: true infections, those MAL-ED's design detects, and MAL-ED's own rates (its Suppl. Table 3:
    # the calibration used two ratios of these bands, not the curves)
    from abm.evidence import maled
    from experiments.calibration.validate_calibration import BANDS
    rows = []
    for p in PATHOGENS:
        for lo, hi in BANDS:
            true_inf = base[f"{p}_infections_per_child_year_{lo}_{hi}m"].astype(float)
            det = base[f"{OBS_PREFIX}maled_{p}_infections_per_child_year_{lo}_{hi}m"].astype(float)
            rows.append({"pathogen": p, "age_lo_months": lo, "age_hi_months": hi, "true": true_inf.mean(),
                         "true_se": true_inf.std(ddof=1) / math.sqrt(len(true_inf)), "detected": det.mean(),
                         "detected_se": det.std(ddof=1) / math.sqrt(len(det)),
                         "maled": maled.infection_rate(p, lo, hi)})
    by_age = pd.DataFrame(rows)
    save(by_age, "p1_obs_by_age")
    m = tab.set_index(["design", "pathogen"])
    for p in PATHOGENS:
        g = by_age[by_age.pathogen == p]
        macro(f"Obs{NAMES[p]}AgeCorr", float(np.corrcoef(g.detected, g.maled)[0, 1]), 2)
        # the share of infections that cause an episode, true and as each design would measure it
        macro(f"Obs{NAMES[p]}TrueSymptomaticPct", 100 * m.loc[("maled", p), "true_episodes"]
              / m.loc[("maled", p), "true_infections"])
        macro(f"Obs{NAMES[p]}AttributablePct", 100 * m.loc[("maled", p), "attributable_fraction_0_24m"])
        macro(f"Obs{NAMES[p]}DailyAttributablePct", 100 * m.loc[("daily", p), "attributable_fraction_0_24m"])
    for p in PATHOGENS:
        macro(f"Obs{NAMES[p]}TrueInfections", m.loc[("maled", p), "true_infections"], 2)
        macro(f"Obs{NAMES[p]}DetectedInfections", m.loc[("maled", p), "detected_infections"], 2)
        macro(f"Obs{NAMES[p]}DetectedShare", 100 * m.loc[("maled", p), "detected_over_true"])
        macro(f"Obs{NAMES[p]}DailyShare", 100 * m.loc[("daily", p), "detected_over_true"])
        macro(f"Obs{NAMES[p]}WeeklyShare", 100 * m.loc[("weekly", p), "detected_over_true"])
    macro("ObsSeeds", len(base))
    return {"designs": len(OBS_DESIGNS), "seeds": len(base)}


# ---------------------------------------------------------------------------
# Experiment 7: the transmission routes
# ---------------------------------------------------------------------------

def part_routes(routes: pd.DataFrame, obs: pd.DataFrame) -> dict:
    """Each route switched off against its share of infections: if the others did not respond, the
    fall would equal the share; a smaller fall means they make up for it, a larger one that the
    route's infections seed the others."""
    from experiments.scenarios.paper1 import ROUTES
    base = by_seed(obs, "baseline")
    rows = []
    for p, r in ROUTES:
        off = by_seed(routes, f"off_{p}_{r}")
        tot = f"{p}_infections_post_burnin"
        share = float((base[f"{p}_{r}_infections_post_burnin"] / base[tot]).mean())
        row = {"pathogen": p, "route": r, "share_pct": 100 * share, "n": len(off)}
        for name, m in (("all_infections", tot), ("child_infections_0_24m", f"{p}_infections_per_child_year_0_24m"),
                        ("episodes_0_24m", f"{p}_symptomatic_episodes_per_child_year_0_24m"),
                        ("detected_0_24m", f"{p}_obs_infections_per_child_year_0_24m")):
            q = pair(off[m], base[m])
            row[f"fall_{name}_pct"], row[f"fall_{name}_se"] = -100 * q["rel"], 100 * q["rel_se"]
        row["fall_minus_share_pct"] = row["fall_all_infections_pct"] - row["share_pct"]
        # what the other routes did, in % of the baseline's infections
        for r2 in sorted({rr for pp, rr in ROUTES if pp == p} - {r}):
            m = f"{p}_{r2}_infections_post_burnin"
            q = pair(off[m], base[m])
            row[f"other_{r2}_change_pct_of_all"] = 100 * q["diff"] / float(base[tot].mean())
        row["extinct_runs"] = int((off[tot] == 0).sum())
        rows.append(row)
    tab = pd.DataFrame(rows)
    save(tab, "p1_routes")
    t = tab.set_index(["pathogen", "route"])
    from experiments.scenarios.paper1 import ROUTES
    camel = {"h2h": "PersonToPerson", "fecal_oral": "FecalOral", "food_borne": "FoodBorne"}
    for p, r in ROUTES:
        short = f"{NAMES[p]}{camel.get(r, r.capitalize())}"
        share = t.loc[(p, r), "share_pct"]
        macro(f"Route{short}Share", share, 1 if share >= 1 else 2)
        macro(f"Route{short}Fall", t.loc[(p, r), "fall_all_infections_pct"], 1)
    macro("RouteSeeds", int(tab["n"].min()))
    macro("RouteRotaImportedExtinct", int(t.loc[("rota", "imported"), "extinct_runs"]))
    return {"routes": len(tab)}


# ---------------------------------------------------------------------------
# Experiment 9: the decision model
# ---------------------------------------------------------------------------

DECISION_OUTPUTS = (["episode_care_seeking_rate", "econ_episode_care_seeking_rate"]
                    + [f"econ_care_seeking_rate_q{q}" for q in range(1, 6)]
                    + ["could_not_afford_rate", "econ_could_not_afford_rate", "econ_could_not_afford_rate_q1",
                       "econ_could_not_afford_rate_q5", "exp_treated_share", "exp_oop_per_episode",
                       "exp_visits_on_credit", "exp_households_in_debt_end", "econ_poverty_headcount"]
                    + [f"{p}_symptomatic_episodes_per_child_year_0_24m" for p in PATHOGENS]
                    + ["rota_obs_attributable_episodes_per_child_year_0_24m"])


def part_decision(dec: pd.DataFrame) -> dict:
    from experiments.calibration.history_matching import ECON_CARE_TARGETS
    from experiments.scenarios.paper1 import DECISION_VARIANTS
    cpt = by_seed(dec, "cpt")
    rows = []
    for v in DECISION_VARIANTS:
        a = by_seed(dec, v)
        for m in DECISION_OUTPUTS:
            if m not in a:
                continue
            q = pair(a[m], cpt[m])
            rows.append({"variant": v, "output": m, "value": q["a"], "cpt": q["b"], "diff": q["diff"], "se": q["se"],
                         "rel_pct": 100 * q["rel"], "n": q["n"]})
        g = a["econ_care_seeking_rate_q5"] - a["econ_care_seeking_rate_q1"]
        q = pair(g, cpt["econ_care_seeking_rate_q5"] - cpt["econ_care_seeking_rate_q1"])
        rows.append({"variant": v, "output": "care_seeking_gradient_q5_minus_q1", "value": q["a"], "cpt": q["b"],
                     "diff": q["diff"], "se": q["se"], "rel_pct": float("nan"), "n": q["n"]})
    tab = pd.DataFrame(rows)
    tab["dhs"] = tab["output"].map({"episode_care_seeking_rate": 0.692,
                                    **{f"econ_care_seeking_rate_q{q}": v[0] for q, v in ECON_CARE_TARGETS.items()},
                                    "care_seeking_gradient_q5_minus_q1": ECON_CARE_TARGETS[5][0] - ECON_CARE_TARGETS[1][0]})
    save(tab, "p1_decision")
    t = tab.set_index(["variant", "output"])
    macro("DecSeeds", int(dec.groupby(LABEL)["seed"].nunique().min()))
    gr = t.xs("care_seeking_gradient_q5_minus_q1", level="output")
    macro("DecGradientCPT", 100 * gr.loc["cpt", "value"], 0)
    macro("DecGradientNoCash", 100 * gr.loc["no_cash_constraint", "value"], 0)
    macro("DecGradientDHS", 100 * gr.loc["cpt", "dhs"], 0)
    # validation: two-week diarrhoea prevalence by wealth quintile (the model's three pathogens, today's
    # vaccine, as in the DHS 2022 population) against the DHS's all-cause prevalence, both relative to the
    # richest quintile (V5-D1: a check, not a target)
    from experiments.calibration.validate_calibration import DHS_DIARRHOEA_2WK_BY_QUINTILE as DHS
    rows = []
    for q in range(1, 6):
        prev = cpt[f"econ_diarrhoea_2wk_prevalence_q{q}"].astype(float)
        rel = prev / cpt["econ_diarrhoea_2wk_prevalence_q5"].astype(float)
        rows.append({"quintile": q, "model": prev.mean(), "model_se": prev.std(ddof=1) / math.sqrt(len(prev)),
                     "model_rel_q5": rel.mean(), "model_rel_q5_se": rel.std(ddof=1) / math.sqrt(len(rel)),
                     **{f"dhs{y}": v[q - 1] for y, v in DHS.items()},
                     **{f"dhs{y}_rel_q5": v[q - 1] / v[4] for y, v in DHS.items()}})
    wealth = pd.DataFrame(rows)
    save(wealth, "p1_wealth_gradient")
    macro("WealthDiarrhoeaRatioModel", wealth.loc[0, "model_rel_q5"], 2)
    macro("WealthDiarrhoeaRatioModelSE", wealth.loc[0, "model_rel_q5_se"], 2)
    macro("WealthDiarrhoeaRatioDHS", wealth.loc[0, "dhs2022_rel_q5"], 2)
    macro("DecCareCPT", 100 * t.loc[("cpt", "episode_care_seeking_rate"), "value"], 1)
    for v, short in (("no_probability_weighting", "NoPW"), ("no_loss_aversion", "NoLA"),
                     ("linear_value", "Linear"), ("expected_utility", "EU"), ("no_cash_constraint", "NoCash")):
        macro(f"DecCare{short}", 100 * t.loc[(v, "episode_care_seeking_rate"), "value"], 1)
        macro(f"DecCare{short}Diff", 100 * t.loc[(v, "episode_care_seeking_rate"), "diff"], 1)
        macro(f"DecCare{short}DiffSE", 100 * t.loc[(v, "episode_care_seeking_rate"), "se"], 1)
    return {"variants": len(DECISION_VARIANTS)}


# ---------------------------------------------------------------------------
# Analysis 5: Morris screening of the fixed parameters
# ---------------------------------------------------------------------------

MORRIS_EXTRA = ([f"{p}_infections_per_child_year_0_24m" for p in PATHOGENS]
                + [f"{p}_symptomatic_episodes_per_child_year_0_24m" for p in PATHOGENS]
                + ["episode_care_seeking_rate", "exp_treated_share", "econ_poverty_headcount"])


def _step_noise(mo: pd.DataFrame, y_runs: pd.Series, labels: list[str], k: int) -> tuple[float, int]:
    """The seed noise in one Morris step, from the seeds themselves: the expected |difference| of a
    step's two points' 2-seed means if the factor did nothing. Every point runs the same seeds (common
    random numbers), so a step's two per-seed differences d1, d2 share the factor's effect and differ
    only by noise: var((d1 + d2) / 2) = E[(d1 - d2)^2] / 4, averaged over the steps (those with both
    seeds' outputs defined). Returns that expected |difference| and the number of steps used."""
    w = (pd.DataFrame({"p": mo[LABEL].to_numpy(), "r": mo["rep"].to_numpy(), "y": np.asarray(y_runs, float)})
         .pivot(index="p", columns="r", values="y").reindex(labels).to_numpy())
    a = np.array([t * (k + 1) + j for t in range(len(labels) // (k + 1)) for j in range(k)])
    dd = (w[a + 1, 0] - w[a, 0]) - (w[a + 1, 1] - w[a, 1])
    ok = np.isfinite(dd)
    return math.sqrt(2.0 / math.pi) * math.sqrt(float(np.mean(dd[ok] ** 2)) / 4.0), int(ok.sum())


#: mu*'s bootstrap confidence interval (SALib): seeded, so the table is the same on every rerun
MORRIS_CONF_RESAMPLES = 1000
MORRIS_CONF_SEED = 61_001


def part_morris(mo: pd.DataFrame) -> dict:
    """mu* and sigma (SALib) for the 41 targets on their calibration scale and for further outputs
    (relative to their mean). For a target, mu* x Delta (Delta = 2/3 of a factor's range) is in units
    of its calibration tolerance sqrt(sd_obs^2 + sd_md^2): above 1, moving the parameter across most
    of its range moves the target by more than the calibration allowed. The noise floor is what the
    seeds alone give to mu* x Delta: the expected |difference| of a step's two points' 2-seed means if
    the factor did nothing, estimated from the seed pairs' own differences (_step_noise), which
    accounts for the seeds the points share. noise_floor_independent is the same assuming independent
    points, sqrt(2/pi) x the SD of one run, kept for comparison. A factor's effect on an output counts
    as above the noise where mu* > 2 x the floor: under no effect, the mean of nine |differences| lies
    within 1 +- 0.25 floors (one SD), so 2 floors is four SDs above."""
    from SALib.analyze import morris as morris_analyze
    from experiments.scenarios.paper1 import MORRIS_FACTORS, MORRIS_LEVELS, load_morris_design, morris_problem
    X = load_morris_design()
    problem = morris_problem()
    delta = MORRIS_LEVELS / (2.0 * (MORRIS_LEVELS - 1))
    tg = targets()
    labels = [f"morris_{i:03d}" for i in range(len(X))]
    if mo.groupby("rep")["seed"].nunique().max() != 1 or mo.groupby(LABEL)["rep"].nunique().min() != 2:
        raise ValueError("the Morris runs must share their two seeds across points")
    outputs = list(tg) + [m for m in MORRIS_EXTRA if m not in tg]
    rows, noise_rows = [], []
    for m in outputs:
        if m not in mo:
            continue
        if m in tg:
            _, sd_obs, sd_md, tr, eps = tg[m]
            y_runs = np.log(mo[m].astype(float) + eps) if tr == "log" else mo[m].astype(float)
            unit = math.sqrt(sd_obs ** 2 + sd_md ** 2)
            scale = "target tolerance"
        else:
            y_runs = mo[m].astype(float)
            unit = float(y_runs.mean())
            scale = "mean"
        y = pd.Series(y_runs.to_numpy(), index=mo[LABEL]).groupby(level=0).mean().reindex(labels).to_numpy()
        if not np.all(np.isfinite(y)):
            continue
        si = morris_analyze.analyze(problem, X, y, num_levels=MORRIS_LEVELS, print_to_console=False,
                                    num_resamples=MORRIS_CONF_RESAMPLES, seed=MORRIS_CONF_SEED)
        pairs = pd.Series(y_runs.to_numpy(), index=mo[LABEL]).groupby(level=0)
        sd_run = math.sqrt(float(np.nanmean(pairs.var(ddof=1))))
        floor_indep = math.sqrt(2.0 / math.pi) * sd_run / unit
        step_noise, steps = _step_noise(mo, y_runs, labels, X.shape[1])
        floor = step_noise / unit
        noise_rows.append({"output": m, "scale": scale, "sd_one_run": sd_run, "noise_floor": floor,
                           "noise_floor_independent": floor_indep, "steps_used": steps,
                           "runs_undefined": int(y_runs.isna().sum())})
        for j, (name, *_rest) in enumerate(MORRIS_FACTORS):
            rows.append({"output": m, "factor": name, "scale": scale, "mu_star": si["mu_star"][j] * delta / unit,
                         "mu": si["mu"][j] * delta / unit, "sigma": si["sigma"][j] * delta / unit,
                         "mu_star_conf": si["mu_star_conf"][j] * delta / unit, "noise_floor": floor})
    tab = pd.DataFrame(rows)
    tab["above_noise"] = tab.mu_star > 2 * tab.noise_floor
    save(tab, "p1_morris")
    save(pd.DataFrame(noise_rows), "p1_morris_noise")
    # per factor: on how many targets its effect is above the noise, on how many of those it also passes
    # one tolerance unit, and its largest effect above the noise
    t = tab[tab.scale == "target tolerance"]
    t = t.assign(past_one=t.above_noise & (t.mu_star > 1.0), mu_star_real=t.mu_star.where(t.above_noise))
    summ = (t.groupby("factor").agg(targets_above_noise=("above_noise", "sum"), targets_past_one=("past_one", "sum"),
                                    max_mu_star_above_noise=("mu_star_real", "max"), max_mu_star=("mu_star", "max"))
            .sort_values(["targets_above_noise", "max_mu_star_above_noise"], ascending=False).reset_index())
    save(summ, "p1_morris_factors")
    macro("MorrisFactors", len(MORRIS_FACTORS)); macro("MorrisPoints", len(X))
    macro("MorrisFactorsAboveNoise", int((summ.targets_above_noise > 0).sum()))
    macro("MorrisFactorsPastOne", int((summ.targets_past_one > 0).sum()))
    sf = summ.set_index("factor")
    if len(sf):                                                        # targets among the outputs
        for f, short in (("steering_parameters.max_contacts_per_visit", "Contacts"),
                         ("pathogens[rota].recovery_rate", "RotaRecovery"),
                         ("pathogens[crypto].recovery_rate", "CryptoRecovery")):
            macro(f"Morris{short}PastOne", int(sf.loc[f, "targets_past_one"]))
            macro(f"Morris{short}MaxMuStar", sf.loc[f, "max_mu_star_above_noise"], 1)
        decision = [f for f, *_ in MORRIS_FACTORS[:9]]                  # the decision's factors come first
        macro("MorrisDecisionPastOne", int(sf.reindex(decision)["targets_past_one"].fillna(0).sum()))
    return {"outputs": int(tab.output.nunique()) if len(tab) else 0,
            "factors_above_noise": int((summ.targets_above_noise > 0).sum()),
            "factors_past_one": int((summ.targets_past_one > 0).sum())}


# ---------------------------------------------------------------------------
# Analysis 3: seeds and warm-up
# ---------------------------------------------------------------------------

SEED_OUTPUTS = ([f"{p}_infections_per_child_year_0_24m" for p in PATHOGENS]
                + [f"{p}_symptomatic_episodes_per_child_year_0_24m" for p in PATHOGENS]
                + [f"{p}_obs_infections_per_child_year_0_24m" for p in PATHOGENS]
                + ["episode_care_seeking_rate", "sev59"])


def part_seeds(a3: pd.DataFrame, e_crit: float = 0.01) -> dict:
    """Lorscheid et al. (2012): the coefficient of variation over the first n seeds, n = 2..40, and the
    smallest n after which it changes by less than e_crit from one n to the next; and the standard
    error of the vaccine's paired effect at 8, 16 and 40 seeds."""
    a3 = add_severe(a3)
    rows, curve = [], []
    for sc in ("vacc_none", "vacc_default"):
        g = by_seed(a3, sc)
        for m in SEED_OUTPUTS:
            x = g[m].astype(float).to_numpy()
            cv = np.array([np.std(x[:n], ddof=1) / np.mean(x[:n]) if np.mean(x[:n]) else np.nan
                           for n in range(2, len(x) + 1)])
            for n, c in zip(range(2, len(x) + 1), cv):
                curve.append({"scenario": sc, "output": m, "n": n, "cv": c})
            change = np.abs(np.diff(cv))
            stable = next((n for n in range(3, len(x) + 1) if np.all(change[n - 3:] < e_crit)), None)
            rows.append({"scenario": sc, "output": m, "mean": float(np.mean(x)), "cv_40": float(cv[-1]),
                         "n_stable": stable, "se_rel_16": float(np.std(x[:16], ddof=1) / math.sqrt(16) / np.mean(x[:16])),
                         "se_rel_40": float(np.std(x, ddof=1) / math.sqrt(len(x)) / np.mean(x))})
    none, today = by_seed(a3, "vacc_none"), by_seed(a3, "vacc_default")
    for n in (8, 16, 40):
        s = none.index[:n]
        for m in (EP24, "sev59"):
            p = pair(today.loc[s, m], none.loc[s, m])
            rows.append({"scenario": "vaccine effect", "output": m, "mean": -100 * p["rel"], "n_stable": None,
                         "cv_40": float("nan"), "se_rel_16": float("nan"), "se_rel_40": float("nan"),
                         "n": n, "effect_se_pct": 100 * p["rel_se"]})
    tab = pd.DataFrame(rows)
    save(tab, "p1_seeds"); save(pd.DataFrame(curve), "p1_seeds_cv_curve")
    st = tab[tab.scenario != "vaccine effect"]
    macro("SeedsNStableMax", st["n_stable"].max())
    macro("SeedsEcrit", e_crit, 2)
    return {"outputs": len(SEED_OUTPUTS)}


def _yearly(series: list, weeks: int = 52) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    n = len(x) // weeks
    return x[: n * weeks].reshape(n, weeks)


def part_warmup(lg: pd.DataFrame) -> dict:
    """Year-by-year rates of the ten-year runs (years of 52 weeks): children's infections and episodes
    per child-year, under-5 carriage, care-seeking and poverty; and for each, year 3 (the first measured
    year of every other part) against years 4-10, and the trend over years 3-10."""
    rows = []
    for _, r in lg.iterrows():
        cy24 = _yearly(r["wk_child_years_0_24m"]).sum(axis=1)
        cy60 = _yearly(r["wk_child_years_0_60m"]).sum(axis=1)
        out = {}
        for p in PATHOGENS:
            out[f"{p}_infections_0_24m"] = _yearly(r[f"wk_inf_{p}_0_24m"]).sum(axis=1) / cy24
            out[f"{p}_episodes_0_60m"] = _yearly(r[f"wk_episodes_{p}"]).sum(axis=1) / cy60
            out[f"{p}_u5_carriage"] = _yearly(r[f"wk_prev_{p}_u5"]).mean(axis=1)
        dec = _yearly(r["wk_episodes_ended"]).sum(axis=1)
        out["care_seeking"] = _yearly(r["wk_episodes_ended_with_care"]).sum(axis=1) / np.where(dec > 0, dec, np.nan)
        out["poverty"] = _yearly(r["wk_poverty"]).mean(axis=1)
        for k, v in out.items():
            for y, val in enumerate(v, start=1):
                rows.append({"scenario": r[LABEL], "seed": r["seed"], "output": k, "year": y, "value": float(val)})
    yr = pd.DataFrame(rows)
    save(yr, "p1_warmup_yearly")
    summ = []
    for (sc, k), g in yr.groupby(["scenario", "output"]):
        w = g.pivot(index="seed", columns="year", values="value")
        later = w.loc[:, [c for c in w.columns if c >= 4]].mean(axis=1)
        d = w[3] - later
        years = np.array([c for c in w.columns if c >= 3], dtype=float)
        slope = [np.polyfit(years, w.loc[s, years.astype(int)].to_numpy(), 1)[0] for s in w.index]
        mean_later = float(later.mean())
        summ.append({"scenario": sc, "output": k, "year3": float(w[3].mean()), "years4_10": mean_later,
                     "year3_minus_later_pct": 100 * float(d.mean()) / mean_later if mean_later else float("nan"),
                     "se_pct": 100 * float(d.std(ddof=1) / math.sqrt(len(d))) / mean_later if mean_later else float("nan"),
                     "trend_pct_per_year": 100 * float(np.mean(slope)) / mean_later if mean_later else float("nan"),
                     "trend_se_pct": 100 * float(np.std(slope, ddof=1) / math.sqrt(len(slope))) / mean_later
                     if mean_later else float("nan"), "years1": float(w[1].mean()), "years2": float(w[2].mean())})
    summ = pd.DataFrame(summ)
    save(summ, "p1_warmup")
    macro("WarmupMaxYearThreeGap", summ["year3_minus_later_pct"].abs().max(), 1)
    ws = summ.set_index(["scenario", "output"])
    for sc, sname in (("long_vacc_none", "NoVacc"), ("long_vacc_default", "Vacc")):
        for p in PATHOGENS:
            k = (sc, f"{p}_infections_0_24m")
            if k in ws.index:
                macro(f"Warmup{NAMES[p]}{sname}Gap", ws.loc[k, "year3_minus_later_pct"], 0)
                macro(f"Warmup{NAMES[p]}{sname}GapSE", ws.loc[k, "se_pct"], 0)
    return {"outputs": int(summ.output.nunique())}


# ---------------------------------------------------------------------------
# Analysis 6: docking
# ---------------------------------------------------------------------------

def part_calibration(diag: str | None = None) -> dict:
    """Analysis 1's numbers for the text, from experiments.calibration.diagnostics_v5's tables and the
    calibrated sets (experiments/scenarios/param_sets.json)."""
    from experiments.scenarios.param_sets import ADOPTED, I_MAX_KEEP, SETS_JSON
    d = diag or CALIBRATION_DIAG
    nroy = pd.read_csv(os.path.join(d, "nroy.csv")).sort_values("wave").reset_index(drop=True)
    ranges = pd.read_csv(os.path.join(d, "design_ranges.csv"))
    box = pd.read_csv(os.path.join(d, "sets_box.csv"))
    fit = pd.read_csv(os.path.join(d, "fit_sets.csv"))
    emu = pd.read_csv(os.path.join(d, "validation.csv"))
    sets = json.load(open(SETS_JSON))
    n_params = box.parameter.nunique()
    macro("CalWaves", len(nroy))
    macro("CalPoints", int(nroy.points.sum()))
    macro("CalRunsPerPoint", int(nroy.runs_per_point.max()))
    for i, name in ((1, "Two"), (2, "Three"), (3, "Four")):
        v = 100 * float(nroy.nroy_share_before.iloc[i])
        macro(f"CalNroyWave{name}", v, 2 if v >= 0.01 else 4)
    macro("CalPerDimensionPct", 100 * float(nroy.nroy_share_before.iloc[-1]) ** (1 / n_params))
    w4 = ranges[ranges.wave == ranges.wave.max()]
    macro("CalWaveFourWidthMedianPct", 100 * float((w4.q975 - w4.q025).median()))
    g = box.groupby("parameter").unit
    macro("CalSetsSpreadMedianPct", 100 * float((g.max() - g.min()).median()))
    macro("CalSetsAtBound", int(((g.max() < 0.1) | (g.min() > 0.9)).sum()))
    imax = fit.groupby("set").implausibility.max()
    macro("CalCandidates", len(sets["sets"]) + len(sets["excluded"]))
    macro("CalSets", len(sets["sets"]))
    macro("CalKeepRule", I_MAX_KEEP, 1)
    macro("CalSetsImaxMin", imax.min(), 2)
    macro("CalSetsImaxMax", imax.max(), 2)
    macro("CalAdoptedImax", imax[ADOPTED], 2)
    new = os.path.join(d, "fit.csv")                     # the adopted defaults on 8 new seeds (--validation)
    if os.path.exists(new):
        f = pd.read_csv(new)
        macro("CalNewSeedsWithin", int((f.implausibility < 3).sum()))
        macro("CalNewSeedsImax", f.implausibility.max(), 2)
    macro("CalEmuRmsMedian", emu.rms_standardised_error.median(), 2)
    macro("CalEmuRmsMin", emu.rms_standardised_error.min(), 2)
    macro("CalEmuRmsMax", emu.rms_standardised_error.max(), 2)
    macro("CalEmuBeyondThreePct", 100 * emu.share_beyond_3.mean(), 1)
    macro("CalEmuRTwoMedian", emu.r2.median(), 2)
    macro("CalEmuNegativeRTwo", int((emu.r2 < 0).sum()))
    return {"waves": len(nroy), "sets": len(sets["sets"])}


def part_docking(path: str | None = None) -> dict:
    s = pd.read_csv(os.path.join(path or DOCKING, "summary.csv"))
    save(s, "p1_docking")
    se = s["attack_sd"] / np.sqrt(s["reps"])
    macro("DockMaxAttackZ", float(np.max(np.abs(s["attack_mean"] - s["attack_final_size_eq"]) / se)), 1)
    macro("DockMaxCurveRMSE", 100 * float(s["curve_rmse_rel_mean_field"].max()), 1)
    macro("DockRuns", int(s["reps"].sum()))
    return {"levels": len(s)}


# ---------------------------------------------------------------------------

PARTS = ("calibration", "3a", "ps", "obs", "routes", "decision", "morris", "seeds", "warmup", "docking")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="paper 1's numbers and tables")
    ap.add_argument("parts", nargs="*", help=f"some of {', '.join(PARTS)} (default: all)")
    ap.add_argument("--results", default=None, help=f"default {RESULTS}")
    args = ap.parse_args(argv)
    bad = sorted(set(args.parts) - set(PARTS))
    if bad:
        ap.error(f"unknown parts {bad}; choose from {list(PARTS)}")
    data = {k: load(k, args.results) for k in DESIGN_DIRS}
    need = {"3a": ["3a"], "ps": ["ps", "3a"], "obs": ["obs"], "routes": ["routes", "obs"], "decision": ["decision"],
            "morris": ["morris"], "seeds": ["3a"], "warmup": ["long"]}
    for part in args.parts or PARTS:
        if part == "calibration":
            if os.path.exists(os.path.join(CALIBRATION_DIAG, "fit_sets.csv")):
                print("calibration", part_calibration())
            else:
                print("calibration: no diagnostics (python -m experiments.calibration.diagnostics_v5)")
            continue
        if part == "docking":
            if os.path.exists(os.path.join(DOCKING, "summary.csv")):
                print("docking", part_docking())
            else:
                print("docking: no results")
            continue
        if any(data[k] is None for k in need[part]):
            print(f"{part}: no results yet")
            continue
        fn = {"3a": lambda: part_3a(data["3a"]), "ps": lambda: part_ps(data["ps"], data["3a"]),
              "obs": lambda: part_obs(data["obs"]), "routes": lambda: part_routes(data["routes"], data["obs"]),
              "decision": lambda: part_decision(data["decision"]), "morris": lambda: part_morris(data["morris"]),
              "seeds": lambda: part_seeds(data["3a"]), "warmup": lambda: part_warmup(data["long"])}[part]
        print(part, fn())
    print("values:", write_values())


if __name__ == "__main__":
    main()

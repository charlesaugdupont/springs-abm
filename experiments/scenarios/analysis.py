# experiments/scenarios/analysis.py
"""
Analysis of the v5 experiments, applying the
rules set before the runs. Every comparison is paired by seed (every
scenario runs with the same seeds): a difference is the mean over seeds of
(scenario - reference), its SE the SD of those differences / sqrt(n).

    python -m experiments.scenarios.analysis 3a [--results <results.parquet>] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

LABEL = "gsa.design_id"
OUT_ROOT = os.path.join("experiments", "outputs", "experiments_v5")


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def paired(df: pd.DataFrame, metric: str, scenario: str, reference: str) -> dict:
    """Mean of `metric` in each scenario, and the paired difference (scenario -
    reference): mean, SE, the number of seeds in which it is negative, n."""
    a = df[df[LABEL] == scenario].set_index("seed")[metric]
    b = df[df[LABEL] == reference].set_index("seed")[metric]
    seeds = a.index.intersection(b.index)
    d = (a.loc[seeds] - b.loc[seeds]).astype(float)
    n = len(d)
    return {"scenario": float(a.mean()), "reference": float(b.mean()), "diff": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "n_lower": int((d < 0).sum()), "n": n,
            "rel": float(d.mean() / b.loc[seeds].mean()) if b.loc[seeds].mean() else float("nan")}


def seeds_needed(n: int, of_eight: int = 7) -> int:
    """The rules set beforehand count seeds out of 8 ("in at least 7 of 8 seeds"); with n seeds (16 or 40
    in the reruns of the revision) the same share: ceil(of_eight / 8 * n). 7 for n = 8, as before."""
    return int(np.ceil(of_eight * n / 8.0 - 1e-9))


def summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean and SE (over seeds) of each metric by scenario."""
    g = df.groupby(LABEL)[metrics]
    out = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_se"))
    return out


# ---------------------------------------------------------------------------
# Experiment 3A: what limits the vaccine
# ---------------------------------------------------------------------------

#: Ghana: rotavirus hospitalisations of under-fives fell by 49% (95% CI 32-63) after
#: Rotarix's introduction (Armah et al. 2016)
GHANA_FALL = (0.49, 0.32, 0.63)
V_EFFS = (0.30, 0.45, 0.55, 0.70, 0.85, 0.95)
DURATIONS = (475, 616, 861, "life")
PLAUSIBLE_V = (0.30, 0.45, 0.55, 0.70, 0.85)
PLAUSIBLE_D = (475, 616, 861)


def cell_label(v: float, d) -> str:
    return "vacc_default" if (v, d) == (0.55, 616) else f"grid_v{v:.2f}_d{d}"


def add_vaccine_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rota_severe_0_59m"] = df[[f"rota_episodes_sev06_{b}" for b in ("0_12m", "12_24m", "24_60m")]].sum(axis=1)
    df["rota_severe_0_12m"] = df["rota_episodes_sev06_0_12m"]
    return df


VACCINE_METRICS = {
    "episodes_0_24m": "rota_symptomatic_episodes_per_child_year_0_24m",
    "episodes_0_59m": "rota_symptomatic_episodes_per_child_year_0_60m",
    "detected_episodes_0_24m": "rota_obs_attributable_episodes_per_child_year_0_24m",
    "infections_0_24m": "rota_infections_per_child_year_0_24m",
    "infections_0_59m": "rota_infections_per_child_year_0_60m",
    "severe_0_59m": "rota_severe_0_59m",
    "severe_0_12m": "rota_severe_0_12m",
    "adult_carriage": "rota_adult_prevalence_post_burnin",
    "protected_now_12_23m": "rota_vaccine_in_V_12_23m",
    "ever_protected_12_23m": "rota_vaccine_protected_ever_12_23m",
    "care_seeking": "econ_episode_care_seeking_rate",
    "carer_days_per_episode": "econ_carer_days_per_episode",
}


def analyse_3a(df: pd.DataFrame, out_dir: str) -> dict:
    df = add_vaccine_outcomes(df)
    os.makedirs(out_dir, exist_ok=True)
    res: dict = {"n_runs": int(len(df)), "seeds": sorted(int(s) for s in df["seed"].unique())}

    # 1. every scenario against no vaccination and against the default schedule
    rows = []
    for sc in sorted(df[LABEL].unique()):
        row = {"scenario": sc}
        for name, col in VACCINE_METRICS.items():
            if col not in df:
                continue
            row[name] = float(df.loc[df[LABEL] == sc, col].mean())
            if sc != "vacc_none":
                p = paired(df, col, sc, "vacc_none")
                row[f"{name}_vs_none"], row[f"{name}_vs_none_se"] = p["rel"], p["se"] / p["reference"]
            if sc != "vacc_default":
                p = paired(df, col, sc, "vacc_default")
                row[f"{name}_vs_default"], row[f"{name}_vs_default_se"] = p["diff"], p["se"]
                row[f"{name}_vs_default_n_lower"] = p["n_lower"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("scenario")
    table.to_csv(os.path.join(out_dir, "table_3a.csv"))

    # 2. the grid: impact (fall against no vaccination) and Ghana's bounds
    m_all, m_sev = VACCINE_METRICS["episodes_0_59m"], VACCINE_METRICS["severe_0_59m"]
    m_main = VACCINE_METRICS["episodes_0_24m"]
    grid = []
    for v in V_EFFS:
        for d in DURATIONS:
            sc = cell_label(v, d)
            if sc not in set(df[LABEL]):
                continue
            fall_all = -paired(df, m_all, sc, "vacc_none")["rel"]
            fall_sev = -paired(df, m_sev, sc, "vacc_none")["rel"]
            fall_main = -paired(df, m_main, sc, "vacc_none")["rel"]
            grid.append({"v_eff": v, "duration": d, "scenario": sc, "fall_episodes_0_24m": fall_main,
                         "fall_episodes_0_59m": fall_all, "fall_severe_0_59m": fall_sev,
                         "consistent_with_ghana": bool(fall_sev >= GHANA_FALL[1] and fall_all <= GHANA_FALL[2])})
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(out_dir, "grid_3a.csv"), index=False)
    res["grid_cells_inconsistent"] = grid.loc[~grid["consistent_with_ghana"], "scenario"].tolist()

    # 3. components: the gain from making each ideal, rule: > 2 SE and > 10% of the default's episodes
    def component(sc: str, ref: str) -> dict:
        p = paired(df, m_main, sc, ref)
        base = p["reference"]
        limits = (-p["diff"] > 2 * p["se"]) and (-p["diff"] > 0.10 * base)
        return {"scenario": sc, "reference": ref, "reduction": -p["diff"], "se": p["se"],
                "reduction_share": -p["diff"] / base if base else float("nan"), "n_lower": p["n_lower"],
                "limits": bool(limits)}

    comps = [component("vacc_coverage_ideal", "vacc_default"), component("vacc_take_ideal", "vacc_default"),
             component(cell_label(0.95, 616), "vacc_default"), component(cell_label(0.55, "life"), "vacc_default"),
             component("vacc_all_ideal", "vacc_default"),
             component("vacc_take_ideal_at_v0.30_d475", cell_label(0.30, 475)),
             component("vacc_take_ideal_at_v0.85_d861", cell_label(0.85, 861))]
    # depth and duration at every plausible cell consistent with Ghana
    ok = set(grid.loc[grid["consistent_with_ghana"], "scenario"])
    rank = []
    for v in PLAUSIBLE_V:
        for d in PLAUSIBLE_D:
            sc = cell_label(v, d)
            if sc not in ok:
                continue
            depth = component(cell_label(0.95, d), sc)
            dur = component(cell_label(v, "life"), sc)
            rank.append({"cell": sc, "depth_gain": depth["reduction_share"], "depth_limits": depth["limits"],
                         "duration_gain": dur["reduction_share"], "duration_limits": dur["limits"],
                         "first": "depth" if depth["reduction"] > dur["reduction"] else "duration"})
    comps = pd.DataFrame(comps)
    comps.to_csv(os.path.join(out_dir, "components_3a.csv"), index=False)
    rank = pd.DataFrame(rank)
    rank.to_csv(os.path.join(out_dir, "ranking_3a.csv"), index=False)
    res["components"] = comps.to_dict("records")
    res["ranking_robust"] = bool(rank["first"].nunique() == 1) if len(rank) else None
    res["ranking_first"] = rank["first"].value_counts().to_dict() if len(rank) else {}

    # 4. structural assumptions: the vaccine's lifelong protection against illness
    res["structural"] = {sc: {k: paired(df, VACCINE_METRICS[k], sc, "vacc_default")
                              for k in ("episodes_0_24m", "severe_0_59m")}
                         for sc in ("vacc_no_severity_reduction", "vacc_no_prior_infection_equivalent",
                                    "vacc_illness_wanes_4y", "vacc_illness_lifelong")
                         if sc in set(df[LABEL])}
    with open(os.path.join(out_dir, "summary_3a.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return {"table": table, "grid": grid, "components": comps, "ranking": rank, "summary": res}


# ---------------------------------------------------------------------------
# Experiment 1A: the edges of care-seeking and the policies that cross them
# ---------------------------------------------------------------------------

CARE_METRICS = {
    "care_seeking": "econ_episode_care_seeking_rate",
    "care_seeking_q1": "econ_care_seeking_rate_q1",
    "care_seeking_q5": "econ_care_seeking_rate_q5",
    "blocked_share": "econ_blocked_episode_share",
    "treated_share": "exp_treated_share",
    "treated_share_q1": "exp_treated_share_q1",
    "oop_per_episode": "exp_oop_per_episode",
    "zero_cash_share": "econ_zero_cash_share",
    "zero_cash_share_q1": "econ_zero_cash_share_q1",
    "poverty": "econ_poverty_headcount",
    "episodes_per_child_year": "econ_episodes_per_child_year",
    "carer_days_per_episode": "econ_carer_days_per_episode",
    "mean_health": "mean_final_health",
    "visits_on_credit": "exp_visits_on_credit",
}
#: the sweeps whose shape the edge rule reads: (name, [(x, scenario)], reference scenario)
PRICE_SWEEP = [(0.0, "price_free"), (0.01, "price_x0.01"), (0.1, "price_x0.1"), (0.5, "price_x0.5"),
               (1.0, "baseline"), (2.0, "price_x2")]
PERCEIVED_SWEEP = [(0.05, "perceived_0.05"), (0.1, "perceived_0.1"), (0.15, "perceived_0.15"),
                   (0.2, "perceived_0.2"), (0.25, "perceived_0.25"), (0.3, "perceived_0.3"), (0.4, "perceived_0.4"),
                   (0.6, "perceived_0.6"), (0.8, "baseline"), (1.0, "perceived_1")]
ACTUAL_SWEEP = [(0.4, "actual_0.4"), (0.6, "actual_0.6"), (0.8, "baseline"), (0.95, "actual_0.95")]


def credit_sweep(days: float) -> list:
    if days == 0:
        return [(0.1, "price_x0.1"), (0.5, "price_x0.5"), (1.0, "baseline"), (2.0, "price_x2")]
    lab = lambda m: f"credit_{days:g}d_price_x{m:g}"
    return [(m, lab(m)) for m in (0.1, 0.5, 1.0, 2.0)]


def edges(df: pd.DataFrame, sweep: list, metric: str) -> dict:
    """The edge rule (set beforehand): between adjacent levels, a change of at
    least a third of the sweep's total range, with a slope at least 3 times
    the average slope, in at least 7 of 8 seeds (the same share of more seeds:
    seeds_needed). Per seed, then counted."""
    xs = np.array([x for x, _ in sweep], dtype=float)
    wide = pd.DataFrame({sc: df[df[LABEL] == sc].set_index("seed")[metric] for _, sc in sweep}).dropna()
    hits = np.zeros(len(sweep) - 1, dtype=int)
    for _, row in wide.iterrows():
        y = row.to_numpy(dtype=float)
        rng = y.max() - y.min()
        if rng <= 0:
            continue
        avg_slope = rng / (xs.max() - xs.min())
        for i in range(len(xs) - 1):
            dy = abs(y[i + 1] - y[i])
            slope = dy / (xs[i + 1] - xs[i])
            hits[i] += int(dy >= rng / 3.0 and slope >= 3.0 * avg_slope)
    n = len(wide)
    found = [(float(xs[i]), float(xs[i + 1]), int(hits[i])) for i in range(len(hits))
             if hits[i] >= max(1, seeds_needed(n))]
    return {"n_seeds": n, "hits_by_interval": [(float(xs[i]), float(xs[i + 1]), int(hits[i]))
                                               for i in range(len(hits))], "edges": found,
            "means": [float(wide[sc].mean()) for _, sc in sweep],
            "ses": [float(wide[sc].sem()) for _, sc in sweep]}


def analyse_1a(df: pd.DataFrame, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    res: dict = {"n_runs": int(len(df))}
    metrics = {k: v for k, v in CARE_METRICS.items() if v in df}
    rows = []
    for sc in sorted(df[LABEL].unique()):
        row = {"scenario": sc}
        for name, col in metrics.items():
            row[name] = float(df.loc[df[LABEL] == sc, col].mean())
            if sc != "baseline":
                p = paired(df, col, sc, "baseline")
                row[f"{name}_diff"], row[f"{name}_se"], row[f"{name}_n_lower"] = p["diff"], p["se"], p["n_lower"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("scenario")
    table["gap_q5_q1"] = table["care_seeking_q5"] - table["care_seeking_q1"]
    table.to_csv(os.path.join(out_dir, "table_1a.csv"))
    sweeps = {"price": PRICE_SWEEP, "perceived": PERCEIVED_SWEEP, "actual": ACTUAL_SWEEP}
    sweeps.update({f"credit_{d:g}d": credit_sweep(d) for d in (0, 1, 3, 10)})
    res["edges"] = {name: edges(df, sw, CARE_METRICS["care_seeking"]) for name, sw in sweeps.items()
                    if all(sc in set(df[LABEL]) for _, sc in sw)}
    with open(os.path.join(out_dir, "summary_1a.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return {"table": table, "summary": res}


# ---------------------------------------------------------------------------
# Time series: deviation from the counterfactual, recovery
# ---------------------------------------------------------------------------

WEEK = 7
ROLL = 4                 # weeks in the rolling mean
STAY = 8                 # weeks a series must stay within the band to count as recovered


def weekly(row: pd.Series, key: str) -> np.ndarray:
    """A weekly series of a run; a ratio for keys "a/b" (e.g. care per episode)."""
    if "/" in key:
        num, den = key.split("/")
        n, d = np.asarray(row[num], dtype=float), np.asarray(row[den], dtype=float)
        return np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)
    return np.asarray(row[key], dtype=float)


def rolling(x: np.ndarray, k: int = ROLL) -> np.ndarray:
    s = pd.Series(x).rolling(k, min_periods=k).mean()
    return s.to_numpy()


def resilience(x: np.ndarray, ref: np.ndarray, start_wk: int, end_wk: int, pre: tuple[int, int]) -> dict:
    """x against its counterfactual ref (same seed, no shock), weekly.
    resistance: the largest |deviation| of the 4-week rolling means from the
    event's start; recovery: weeks from the event's end until the deviation
    stays within +-2 SD of ref's own 4-week rolling mean over `pre` (weeks) for
    8 weeks (nan if it never does); excess: the summed deviation from the
    event's start; persistence: the mean deviation over the 13 weeks starting a
    year after the event's end (at least 8 of them within the run)."""
    rx, rr = rolling(x), rolling(ref)
    dev = rx - rr
    band = 2.0 * np.nanstd(rr[pre[0]:pre[1]])
    n = len(x)
    after = dev[start_wk:]
    resistance = float(after[np.nanargmax(np.abs(after))]) if np.isfinite(after).any() else float("nan")
    recovery = float("nan")
    for t in range(end_wk, n - STAY + 1):
        w = dev[t:t + STAY]
        if np.all(np.isfinite(w)) and np.all(np.abs(w) <= band):
            recovery = float(t - end_wk)
            break
    p0 = end_wk + 52
    p1 = min(p0 + 13, n)
    persistence = float(np.nanmean((x - ref)[p0:p1])) if p1 - p0 >= 8 else float("nan")
    return {"resistance": resistance, "recovery_weeks": recovery, "band": float(band),
            "excess": float(np.nansum((x - ref)[start_wk:])), "persistence": persistence}


def censored_median(v, left: float) -> float:
    """Median recovery time where nan (not recovered within the run) counts as `left`, the latest
    recovery the run could still show: the median itself while at least half the seeds recovered,
    otherwise a lower bound."""
    v = np.asarray(v, dtype=float)
    return float(np.median(np.where(np.isfinite(v), v, left))) if len(v) else float("nan")


# ---------------------------------------------------------------------------
# Experiment 2: shocks and strategies
# ---------------------------------------------------------------------------

#: event windows in weeks: (start, end of the shock itself, or None for the chronic cycles); recovery is
#: counted from the end of the shock, so a slow recovery shows as a long recovery time
def exp2_windows() -> dict:
    from experiments.scenarios import hooks as H
    wk = lambda d: int(d // WEEK)
    flood = (wk(H.FLOOD_START_DAY), wk(H.FLOOD_START_DAY + H.FLOOD_DAYS))
    drought = (wk(H.DROUGHT_START_DAY), wk(H.DROUGHT_START_DAY + H.DROUGHT_DAYS))
    return {
        "C-lo": (wk(H.SHOCK_START_DAY), None), "C-hi": (wk(H.SHOCK_START_DAY), None),
        "C-rare": (wk(H.SHOCK_START_DAY), wk(H.SHOCK_START_DAY + 365)),
        "F-fast": flood, "F-slow": flood, "C+F": flood,
        "D-fast": drought, "D-slow": drought, "C+D": drought,
    }


EXP2_OUTCOMES = {
    "episodes": "wk_episodes_started",
    "care_per_episode": "wk_episodes_ended_with_care/wk_episodes_ended",
    "blocked_per_episode": "wk_episodes_ended_blocked/wk_episodes_ended",
    "zero_cash": "wk_zero_cash_share",
    "zero_cash_q1": "wk_zero_cash_share_q1",
    "poverty": "wk_poverty",
    "episodes_rota": "wk_episodes_rota",
    "episodes_crypto": "wk_episodes_crypto",
    "episodes_campy": "wk_episodes_campy",
    "fz_episodes": "wk_fz_episodes",
    "fz_zero_cash": "wk_fz_zero_cash_share",
}


#: outcomes for which more is better: their shortfall is the burden
BURDEN_SIGN = {"care_per_episode": -1}


def exp2_variant(label: str) -> str:
    """A scenario's variant in the reruns: "main" (the regime, with the environmental pathways where it has
    a drought or a flood), "noenv" (without them, the reference), "credit" (one day of credit) or "mild" (the
    drought hygiene of a mild drought, a sensitivity)."""
    if label.endswith("_noenv"):
        return "noenv"
    if "_credit" in label:
        return "credit"
    if label.endswith("_mild"):
        return "mild"
    return "main"


def _pre_window(H) -> tuple:
    """Weeks over which the counterfactual's own variability sets the recovery band: the measured year before
    the shocks when the runs had one (shocks from day 1,095), otherwise the measured year itself (the
    counterfactual has no shock; shocks from day 730, the user's decision R1 of 2026-11-10)."""
    if H.SHOCK_START_DAY - 730 >= 26 * WEEK:
        return (int(730 // WEEK), int(H.SHOCK_START_DAY // WEEK))
    return (int(H.SHOCK_START_DAY // WEEK), None)


def analyse_2(df: pd.DataFrame, out_dir: str) -> dict:
    from experiments.scenarios import hooks as H
    os.makedirs(out_dir, exist_ok=True)
    windows = exp2_windows()
    pre = _pre_window(H)
    df = df.copy()
    df["variant"] = [exp2_variant(lab) for lab in df[LABEL]]
    rows = []
    for label, g in df.groupby(LABEL):
        regime, strategy, variant = g["hook.regime"].iloc[0], g["hook.strategy"].iloc[0], g["variant"].iloc[0]
        if regime == "R0":
            continue
        ref_runs = df[(df["hook.regime"] == "R0") & (df["hook.strategy"] == strategy)
                      & (df["variant"] == "main")].set_index("seed")
        start, end = windows[regime]
        for _, r in g.iterrows():
            if r["seed"] not in ref_runs.index:
                continue
            ref = ref_runs.loc[r["seed"]]
            for name, key in EXP2_OUTCOMES.items():
                if key.split("/")[0] not in r or not isinstance(r[key.split("/")[0]], (list, np.ndarray)):
                    continue
                x, xr = weekly(r, key), weekly(ref, key)
                m = resilience(x, xr, start, end if end is not None else len(x), pre)
                if end is None:                      # a chronic cycle: no recovery, no persistence
                    m["recovery_weeks"], m["persistence"] = float("nan"), float("nan")
                rows.append({"scenario": label, "regime": regime, "strategy": strategy, "variant": variant,
                             "seed": int(r["seed"]), "outcome": name,
                             "event_weeks": (end - start) if end is not None else float("nan"),
                             "weeks_after_event": (len(x) - end) if end is not None else float("nan"),
                             # the level itself from the shock's start to the end of the run (for the absolute
                             # comparison of strategies, against no strategy under the same shock)
                             "level": float(np.nanmean(x[start:])),
                             # the deviation summed over the event itself (and to the run's end)
                             "excess_event": float(np.nansum((x - xr)[start:end if end is not None else len(x)])),
                             **m})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(out_dir, "resilience_runs_2.csv"), index=False)
    keys = ["scenario", "regime", "strategy", "variant", "outcome"]
    agg = per_run.groupby(keys).agg(
        resistance=("resistance", "mean"), recovery_weeks=("recovery_weeks", "median"),
        recovered=("recovery_weeks", lambda x: int(np.isfinite(x).sum())), excess=("excess", "mean"),
        excess_se=("excess", "sem"), excess_event=("excess_event", "mean"), excess_event_se=("excess_event", "sem"),
        persistence=("persistence", "mean"), persistence_se=("persistence", "sem"),
        n=("seed", "count")).reset_index()
    # rules (set beforehand): recovery is "slow" if it exceeds the event's duration by more than 26 weeks in at
    # least 7 of 8 seeds (never within the run counts as slow when the run left that much time); "persistent" if
    # the deviation a year after the event exceeds 2 SE. In one-year runs neither can be judged after the
    # drought or the flood (17 and 9 weeks are left): both come out as not met, and recovery is reported as
    # whether it happened within the year.
    def _slow(g):
        rec, dur, left = g["recovery_weeks"], g["event_weeks"], g["weeks_after_event"]
        slow = (rec - dur > 26) | (rec.isna() & (left - dur > 26 + 8))
        return int(slow.sum())
    flags = per_run[per_run["event_weeks"].notna()].groupby(keys)[
        ["recovery_weeks", "event_weeks", "weeks_after_event"]].apply(_slow)
    agg = agg.merge(flags.rename("seeds_slow").reset_index(), how="left", on=keys)
    agg["slow"] = agg["seeds_slow"] >= agg["n"].map(seeds_needed)
    agg["persistent"] = (agg["persistence"].abs() > 2 * agg["persistence_se"]) & agg["persistence"].notna()
    agg.to_csv(os.path.join(out_dir, "resilience_2.csv"), index=False)
    # strategy comparisons (rule set beforehand): a lower burden added by the shock (its cumulative excess,
    # sign-adjusted so that more is worse: care lost counts as a burden) in >= 7 of 8 paired seeds (the same
    # share of more seeds), > 2 SE. Only the main variant of each regime.
    main = per_run[per_run["variant"] == "main"]
    comp = []
    for (regime, outcome), g in main.groupby(["regime", "outcome"]):
        piv = g.pivot(index="seed", columns="strategy", values="excess") * BURDEN_SIGN.get(outcome, 1)
        if "none" not in piv:
            continue
        for strategy in piv.columns:
            if strategy == "none":
                continue
            d = (piv[strategy] - piv["none"]).dropna()
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            comp.append({"regime": regime, "outcome": outcome, "strategy": strategy, "diff_burden": d.mean(),
                         "se": se, "n_lower": int((d < 0).sum()), "n": len(d),
                         "better": bool((d < 0).sum() >= seeds_needed(len(d)) and -d.mean() > 2 * se),
                         "worse": bool((d > 0).sum() >= seeds_needed(len(d)) and d.mean() > 2 * se)})
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(out_dir, "strategies_2.csv"), index=False)
    # absolute: the outcome's level from the shock's start, strategy against none, same regime and seed
    lev = []
    for (regime, outcome), g in main.groupby(["regime", "outcome"]):
        piv = g.pivot(index="seed", columns="strategy", values="level")
        if "none" not in piv:
            continue
        for strategy in piv.columns:
            if strategy == "none":
                continue
            d = (piv[strategy] - piv["none"]).dropna()
            if len(d) < 2:
                continue
            lev.append({"regime": regime, "outcome": outcome, "strategy": strategy, "level_none": piv["none"].mean(),
                        "level": piv[strategy].mean(), "diff": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)),
                        "n_lower": int((d < 0).sum()), "n": len(d)})
    pd.DataFrame(lev).to_csv(os.path.join(out_dir, "strategies_level_2.csv"), index=False)
    # the environmental pathways (with against without, no strategy) and one day of credit (against none), paired
    other = []
    for (scenario, ref) in ([(f"{r}_none", f"{r}_none_noenv") for r in sorted(set(df["hook.regime"]))]
                            + [("D-slow_none_credit1d", "D-slow_none"), ("D-slow_none_mild", "D-slow_none"),
                               ("D-slow_none_mild", "D-slow_none_noenv")]):
        a = per_run[per_run["scenario"] == scenario]
        b = per_run[per_run["scenario"] == ref]
        if a.empty or b.empty:
            continue
        for outcome in sorted(set(a["outcome"])):
            for col in ("excess_event", "excess"):
                pa = a[a["outcome"] == outcome].set_index("seed")[col]
                pb = b[b["outcome"] == outcome].set_index("seed")[col]
                d = (pa - pb).dropna()
                if len(d) < 2:
                    continue
                other.append({"scenario": scenario, "reference": ref, "outcome": outcome, "measure": col,
                              "scenario_mean": float(pa.mean()), "reference_mean": float(pb.mean()),
                              "diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": len(d)})
    pd.DataFrame(other).to_csv(os.path.join(out_dir, "pathways_and_credit_2.csv"), index=False)
    res = {"n_runs": int(len(df)), "pre_window_weeks": pre}
    with open(os.path.join(out_dir, "summary_2.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return {"per_run": per_run, "resilience": agg, "strategies": comp, "summary": res}


# ---------------------------------------------------------------------------
# Experiment 1B: learning; B1 stock-outs
# ---------------------------------------------------------------------------

CARE_PER_EPISODE = "wk_episodes_ended_with_care/wk_episodes_ended"


def _stockout_start(df: pd.DataFrame) -> int:
    """The stock-outs' first day, as the scenarios set it (1,095 in the first runs, 730 in the reruns)."""
    v = pd.to_numeric(df.get("hook.stockout_start"), errors="coerce").dropna()
    if len(v):
        return int(v.iloc[0])
    from experiments.scenarios.designs import STOCKOUT_START
    return int(STOCKOUT_START)


def analyse_1b_rev(df: pd.DataFrame, out_dir: str) -> dict:
    """The rerun of 1B in one year (B1 and B4 in one design): B1 as analyse_1b1, B4 as analyse_1b4 against
    B1's run without a stock-out at the strongest social weight."""
    b1 = analyse_1b1(df[df[LABEL].str.startswith("B1_")], out_dir)
    b4 = analyse_1b4(df[df[LABEL].str.startswith("B4_")], out_dir, ref_df=df[df[LABEL].str.startswith("B1_")])
    return {"b1": b1, "b4": b4, "summary": {"b1": b1["summary"], "b4": b4["summary"].to_dict("records")}}


def analyse_1b1(df: pd.DataFrame, out_dir: str) -> dict:
    """Recovery of care-seeking after stock-outs, against the same seed and
    social weight without one. Rule (set beforehand): a tipping point only if
    recovery time grows faster than the stock-out's duration (here: from 16 to
    26 weeks it grows by more than 10 weeks) and exceeds a year, in at least 7
    of 8 seeds; a series that never recovers within the run counts as longer
    than any that does."""
    from experiments.scenarios.designs import STOCKOUT_WEEKS
    os.makedirs(out_dir, exist_ok=True)
    start_day = _stockout_start(df)
    start = start_day // WEEK
    pre = (730 // WEEK, start) if start_day - 730 >= 26 * WEEK else (start, None)
    rows = []
    for w in sorted(df["steering_parameters.belief_social_weight"].dropna().unique()):
        sub = df[df["steering_parameters.belief_social_weight"] == w]
        ref = sub[sub[LABEL] == f"B1_s{w:g}_none"].set_index("seed")
        for weeks in STOCKOUT_WEEKS:
            runs = sub[sub[LABEL] == f"B1_s{w:g}_stockout{weeks}w"]
            if runs.empty:
                continue
            for _, r in runs.iterrows():
                if r["seed"] not in ref.index:
                    continue
                x, xr = weekly(r, CARE_PER_EPISODE), weekly(ref.loc[r["seed"]], CARE_PER_EPISODE)
                m = resilience(x, xr, start, start + weeks, pre)
                b, br = weekly(r, "wk_belief_success"), weekly(ref.loc[r["seed"]], "wk_belief_success")
                mb = resilience(b, br, start, start + weeks, pre)       # trust: a smooth series
                rows.append({"social_weight": w, "weeks": weeks, "seed": int(r["seed"]), **m,
                             "min_belief": float(np.nanmin(b[start:])), "belief_gap_1y_after":
                             float(np.nanmean((b - br)[start + weeks + 52:start + weeks + 65])),
                             "belief_drop": float(-mb["resistance"]), "belief_recovery_weeks": mb["recovery_weeks"],
                             "low_belief_share_max": float(np.nanmax(weekly(r, "wk_belief_low_share")[start:]))})
                rows[-1].update({"care_rate_vs_none": float(r["econ_episode_care_seeking_rate"])
                                 - float(ref.loc[r["seed"], "econ_episode_care_seeking_rate"]),
                                 "weeks_left": int(len(b) - (start + weeks) - STAY),
                                 # one-year runs: the gaps left at the end of the year, and care per episode
                                 # from the stock-out's end to the end of the run, against no stock-out
                                 "belief_gap_end": float(np.nanmean((b - br)[-4:])),
                                 "care_after_stockout": float(np.nanmean(x[start + weeks:]))
                                 - float(np.nanmean(xr[start + weeks:]))})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(out_dir, "recovery_1b1.csv"), index=False)
    # the table of the results note: means over seeds, and the median recovery of trust (censored)
    summary = per_run.groupby(["social_weight", "weeks"]).apply(lambda g: pd.Series({
        "min_belief": g["min_belief"].mean(), "low_belief_share_max": g["low_belief_share_max"].mean(),
        "belief_gap_1y_after": g["belief_gap_1y_after"].mean(), "belief_gap_1y_after_se": g["belief_gap_1y_after"].sem(),
        "belief_recovery_weeks_median": censored_median(g["belief_recovery_weeks"], g["weeks_left"].iloc[0]),
        "belief_not_recovered": int((~np.isfinite(g["belief_recovery_weeks"])).sum()),
        "care_rate_vs_none": g["care_rate_vs_none"].mean(), "care_rate_vs_none_se": g["care_rate_vs_none"].sem(),
        "belief_gap_end": g["belief_gap_end"].mean(), "belief_gap_end_se": g["belief_gap_end"].sem(),
        "care_after_stockout": g["care_after_stockout"].mean(),
        "care_after_stockout_se": g["care_after_stockout"].sem(),
        "n": len(g)}), include_groups=False).reset_index()
    summary.to_csv(os.path.join(out_dir, "recovery_1b1_summary.csv"), index=False)
    tipping = {}
    for key, col in (("tipping", "recovery_weeks"), ("tipping_trust", "belief_recovery_weeks")):
        tipping[key] = {}
        for w, g in per_run.groupby("social_weight"):
            piv = g.pivot(index="seed", columns="weeks", values=col).fillna(np.inf)
            if {16, 26} <= set(piv.columns):
                ok = (piv[26] - piv[16] > 10) & (piv[26] > 52)
                # the rule needs a recovery of more than a year to be observable after the 26-week stock-out
                left = g.loc[g["weeks"] == 26, "weeks_left"]
                evaluable = bool(len(left) and left.min() > 52)
                tipping[key][float(w)] = {"seeds_meeting_rule": int(ok.sum()), "n": int(len(piv)),
                                          "evaluable": evaluable,
                                          "tipping_point": bool(evaluable and ok.sum() >= seeds_needed(len(piv)))}
    res = {"n_runs": int(len(df)), **tipping}
    with open(os.path.join(out_dir, "summary_1b1.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return {"per_run": per_run, "summary": res}


def analyse_1b4(df: pd.DataFrame, out_dir: str, ref_df: pd.DataFrame | None = None) -> dict:
    """B4: after a 26-week stock-out (social weight 0.6), what restores trust. Against
    the same seed without a stock-out (B1_s0.6_none, from B1's results in ref_df):
    the time until trust is back within its band, the belief gap 13 and 52 weeks
    after supplies return, and care-seeking over the measured period."""
    from experiments.scenarios.designs import SOCIAL_STRONG
    os.makedirs(out_dir, exist_ok=True)
    start_day = _stockout_start(df)
    start, end = start_day // WEEK, (start_day + 182) // WEEK
    pre = (730 // WEEK, start) if start_day - 730 >= 26 * WEEK else (start, None)
    ref = (ref_df if ref_df is not None else df)
    ref = ref[ref[LABEL] == f"B1_s{SOCIAL_STRONG:g}_none"].set_index("seed")
    rows = []
    for _, r in df.iterrows():
        if r["seed"] not in ref.index:
            continue
        b, br = weekly(r, "wk_belief_success"), weekly(ref.loc[r["seed"]], "wk_belief_success")
        mb = resilience(b, br, start, end, pre)
        x, xr = weekly(r, CARE_PER_EPISODE), weekly(ref.loc[r["seed"]], CARE_PER_EPISODE)
        rows.append({"scenario": r[LABEL], "seed": int(r["seed"]), "belief_recovery_weeks": mb["recovery_weeks"],
                     "belief_gap_13w": float(np.nanmean((b - br)[end + 13:end + 17])),
                     "belief_gap_52w": float(np.nanmean((b - br)[end + 52:end + 56])),
                     "low_share_13w": float(np.nanmean(weekly(r, "wk_belief_low_share")[end + 13:end + 17])),
                     "care_after": float(np.nanmean(x[end:end + 52])) - float(np.nanmean(xr[end:end + 52])),
                     # care per episode (mean of weekly values) while free care lasts, and 3-12 and 12-24 months on
                     "care_0_13w": float(np.nanmean(x[end:end + 13])), "care_13_52w": float(np.nanmean(x[end + 13:end + 52])),
                     "care_52_104w": float(np.nanmean(x[end + 52:end + 104])) if len(x) > end + 52 else float("nan"),
                     # one-year runs: from 13 weeks after supplies return to the end of the run, and the gap in
                     # trust left at the end
                     "care_13w_end": float(np.nanmean(x[end + 13:])), "belief_gap_end": float(np.nanmean((b - br)[-4:])),
                     "care_rate": float(r["econ_episode_care_seeking_rate"]), "weeks_left": int(len(b) - end - STAY)})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(out_dir, "recovery_1b4.csv"), index=False)
    agg = per_run.groupby("scenario").agg(
        recovered=("belief_recovery_weeks", lambda v: int(np.isfinite(v).sum())),
        belief_gap_13w=("belief_gap_13w", "mean"), belief_gap_13w_se=("belief_gap_13w", "sem"),
        belief_gap_52w=("belief_gap_52w", "mean"), belief_gap_52w_se=("belief_gap_52w", "sem"),
        low_share_13w=("low_share_13w", "mean"), care_after=("care_after", "mean"), care_after_se=("care_after", "sem"),
        care_0_13w=("care_0_13w", "mean"), care_13w_end=("care_13w_end", "mean"),
        belief_gap_end=("belief_gap_end", "mean"), belief_gap_end_se=("belief_gap_end", "sem"),
        n=("seed", "count")).reset_index()
    # median recovery of trust, censored at the run's end (a lower bound when fewer than half recovered)
    agg.insert(1, "belief_recovery_weeks_median", [censored_median(
        per_run.loc[per_run["scenario"] == sc, "belief_recovery_weeks"],
        per_run.loc[per_run["scenario"] == sc, "weeks_left"].iloc[0]) for sc in agg["scenario"]])
    # against no intervention, paired by seed: care-seeking over the measured period and after the interventions
    base = per_run[per_run["scenario"] == "B4_collapse"].set_index("seed")
    for col in ("care_rate", "care_0_13w", "care_13_52w", "care_52_104w", "care_13w_end", "belief_gap_end"):
        mean, se, higher = [], [], []
        for sc in agg["scenario"]:
            dlt = (per_run[per_run["scenario"] == sc].set_index("seed")[col] - base[col]).dropna()
            mean.append(float(dlt.mean())), se.append(float(dlt.sem())), higher.append(int((dlt > 0).sum()))
        agg[f"{col}_vs_none"], agg[f"{col}_vs_none_se"], agg[f"{col}_seeds_higher"] = mean, se, higher
    agg.to_csv(os.path.join(out_dir, "recovery_1b4_summary.csv"), index=False)
    return {"per_run": per_run, "summary": agg}


def hysteresis_1b2(paths: pd.DataFrame) -> pd.DataFrame:
    """B2's hysteresis rule on the binned down and up paths (see analyse_1b2)."""
    from itertools import groupby
    from experiments.scenarios.designs import SOCIAL_STRONG

    def _gaps(g: pd.DataFrame, value: str) -> pd.Series:
        """Per seed and efficacy level: the up path minus the down path."""
        piv = g.pivot_table(index=["seed", "efficacy"], columns="leg", values=value).dropna()
        return piv["up"] - piv["down"]

    hyst, flagged = [], {}
    for sc, g in paths.groupby("scenario"):
        row = {"scenario": sc}
        for value in ("care", "belief"):
            gap = _gaps(g, value)
            by = gap.groupby(level="efficacy")
            diff, se = by.mean(), by.sem()
            sig = (diff.abs() > 3 * se).to_numpy()
            run = max((len(list(gr)) for k, gr in groupby(sig) if k), default=0)
            flagged[(sc, value)] = list(diff.index[sig]) or list(diff.index)
            row.update({f"mean_gap_{value}": float(diff.mean()), f"max_gap_{value}": float(diff.abs().max()),
                        f"levels_beyond_3se_{value}": int(sig.sum()),
                        f"consecutive_levels_beyond_3se_{value}": int(run), f"paths_differ_{value}": bool(run >= 3)})
        hyst.append(row)
    hyst = pd.DataFrame(hyst)
    # the half-speed check (rule), at SOCIAL_STRONG, where the ramp is also run at half speed: over the levels
    # where the full-speed paths differ by more than 3 SE (all levels if none do), the gap must not shrink.
    # Operationalised on 2026-11-07, before B2's results were read: the half-speed mean gap is at least 3/4 of
    # the full-speed one (halfway between no shrinking and the halving a pure lag would give) and not smaller
    # by more than 2 SE (paired by seed).
    full, half = f"B2_s{SOCIAL_STRONG:g}_ramp365", f"B2_s{SOCIAL_STRONG:g}_ramp730"
    for value in ("care", "belief"):
        for col in (f"half_speed_ratio_{value}", f"gap_shrinks_{value}", f"hysteresis_{value}"):
            hyst[col] = np.nan if "ratio" in col else False
        if full in set(paths["scenario"]) and half in set(paths["scenario"]):
            levels = flagged[(full, value)]
            gf = _gaps(paths[paths["scenario"] == full], value)
            gh = _gaps(paths[paths["scenario"] == half], value)
            pf = gf[gf.index.get_level_values("efficacy").isin(levels)].groupby(level="seed").mean()
            ph = gh[gh.index.get_level_values("efficacy").isin(levels)].groupby(level="seed").mean()
            s = np.sign(pf.mean()) or 1.0
            ratio = float((s * ph).mean() / (s * pf).mean()) if (s * pf).mean() != 0 else float("nan")
            d = (s * (pf - ph)).dropna()
            shrinks = bool(not (ratio >= 0.75) or (d.mean() > 2 * d.sem()))
            i = hyst.index[hyst["scenario"] == full]
            hyst.loc[i, f"half_speed_ratio_{value}"] = ratio
            hyst.loc[i, f"gap_shrinks_{value}"] = shrinks
            hyst.loc[i, f"hysteresis_{value}"] = bool(hyst.loc[i, f"paths_differ_{value}"].iloc[0] and not shrinks)
    return hyst


def analyse_1b2(df: pd.DataFrame, out_dir: str) -> dict:
    """B2/B3: efficacy lowered linearly to 0.1 and raised back. Hysteresis (rule): the
    down and up paths of care-seeking (and, descriptively, of trust), binned by the
    actual efficacy, differ by more than 3 SE over at least three consecutive levels,
    and the gap does not shrink when the ramp is twice as slow. Early warning (rule):
    Kendall's tau of the rolling lag-1 autocorrelation of weekly care-seeking, before
    its steepest fall on the way down, exceeds 0.5 in at least 6 of 8 seeds."""
    from scipy.stats import kendalltau
    from experiments.scenarios.hooks import efficacy_on
    os.makedirs(out_dir, exist_ok=True)
    bins = np.round(np.arange(0.1, 0.81, 0.1), 2)
    num_key, den_key = CARE_PER_EPISODE.split("/")
    paths, ews = [], []
    for _, r in df.iterrows():
        combo = {k: r[k] for k in ("hook.ramp_start", "hook.ramp_days", "hook.ramp_low") if k in r and pd.notna(r[k])}
        n_days = int(combo["hook.ramp_days"])
        start = int(combo["hook.ramp_start"])
        belief = weekly(r, "wk_belief_success")
        num, den = weekly(r, num_key), weekly(r, den_key)
        weeks = np.arange(len(belief))
        eff = np.array([efficacy_on(int(w * WEEK + 3), combo, 0.8) for w in weeks])     # mid-week
        down = (weeks * WEEK >= start) & (weeks * WEEK < start + n_days)
        up = (weeks * WEEK >= start + n_days) & (weeks * WEEK < start + 2 * n_days)
        for leg, mask in (("down", down), ("up", up)):
            for lo in bins:
                sel = mask & (np.abs(eff - lo) < 0.05)
                if sel.any() and np.nansum(den[sel]) > 0:
                    # care per episode pooled over the level's weeks (a ratio of sums: weeks hold ~6 episodes)
                    paths.append({"scenario": r[LABEL], "seed": int(r["seed"]), "leg": leg, "efficacy": float(lo),
                                  "weeks": int(sel.sum()), "episodes": float(np.nansum(den[sel])),
                                  "care": float(np.nansum(num[sel]) / np.nansum(den[sel])),
                                  "belief": float(np.nanmean(belief[sel]))})
        # early warning (rule): lag-1 autocorrelation of weekly care-seeking (detrended by a centred 13-week
        # mean) in a 26-week rolling window, from the ramp's start to the steepest fall of the 13-week mean;
        # descriptively, the same for the rolling SD, and the size of the fall
        x = pd.Series(weekly(r, CARE_PER_EPISODE))
        smooth_s = x.rolling(13, center=True, min_periods=7).mean()
        resid, smooth = x - smooth_s, smooth_s.to_numpy()
        w0, w1 = start // WEEK, (start + n_days) // WEEK
        drop = w0 + int(np.nanargmin(np.diff(smooth[w0:w1 + 1]))) if w1 > w0 + 1 else w1
        ac = resid.rolling(26, min_periods=20).apply(lambda v: pd.Series(v).autocorr(1), raw=False).to_numpy()
        sd = resid.rolling(26, min_periods=20).std().to_numpy()

        def _tau(series):
            seg = series[w0:drop]
            ok = np.isfinite(seg)
            return float(kendalltau(np.arange(ok.sum()), seg[ok])[0]) if ok.sum() >= 8 else float("nan")
        ews.append({"scenario": r[LABEL], "seed": int(r["seed"]), "drop_week": int(drop),
                    "weeks_before_drop": int(drop - w0), "kendall_tau_ac1": _tau(ac), "kendall_tau_sd": _tau(sd),
                    "care_13w_before_ramp": float(np.nanmean(x[max(w0 - 13, 0):w0])),
                    "care_13w_at_bottom": float(np.nanmean(x[w1 - 6:w1 + 7]))})
    paths = pd.DataFrame(paths)
    paths.to_csv(os.path.join(out_dir, "paths_1b2.csv"), index=False)
    ews = pd.DataFrame(ews)
    ews.to_csv(os.path.join(out_dir, "early_warning_1b2.csv"), index=False)

    hyst = hysteresis_1b2(paths)
    hyst.to_csv(os.path.join(out_dir, "hysteresis_1b2.csv"), index=False)
    ew = ews.groupby("scenario").agg(tau_mean=("kendall_tau_ac1", "mean"),
                                     seeds_tau_above_half=("kendall_tau_ac1", lambda v: int((v > 0.5).sum())),
                                     tau_sd_mean=("kendall_tau_sd", "mean"),
                                     weeks_before_drop=("weeks_before_drop", "median"),
                                     care_before=("care_13w_before_ramp", "mean"),
                                     care_at_bottom=("care_13w_at_bottom", "mean"),
                                     n=("seed", "count")).reset_index()
    ew["early_warning"] = ew["seeds_tau_above_half"] >= ew["n"] - 2
    ew.to_csv(os.path.join(out_dir, "early_warning_1b2_summary.csv"), index=False)
    return {"paths": paths, "hysteresis": hyst, "early_warning": ew}


# ---------------------------------------------------------------------------
# Experiment 3B: interventions, by quintile
# ---------------------------------------------------------------------------

def analyse_3b(df: pd.DataFrame, out_dir: str) -> dict:
    """Each intervention against today's schedule (paired by seed); rule (set
    beforehand): it "helps" if it reduces episodes at 0-24 months in at least 7
    of 8 seeds and by more than 2 SE; it "narrows the gap" if the lowest-to-
    highest-quintile ratio of episodes (under five) falls by more than 2 SE.
    D4: the WASH upgrade's effect with take depending on WASH, against its
    effect through exposure alone."""
    df = add_vaccine_outcomes(df)
    os.makedirs(out_dir, exist_ok=True)
    q1, q5 = "exp_rota_episodes_per_child_year_q1", "exp_rota_episodes_per_child_year_q5"
    df["rota_q1_q5_ratio"] = df[q1] / df[q5]
    main = VACCINE_METRICS["episodes_0_24m"]
    rows = []
    pairs = [(sc, "vacc_default") for sc in sorted(df[LABEL].unique())
             if sc not in ("vacc_default", "takewash_default", "takewash_wash_q12", "booster_9m_wanes_4y")]
    pairs.append(("takewash_wash_q12", "takewash_default"))
    pairs.append(("takewash_default", "vacc_default"))
    # the sensitivity on the duration of protection against illness: the booster against the same default
    pairs.append(("booster_9m_wanes_4y", "vacc_default_wanes_4y"))
    for sc, ref in pairs:
        if sc not in set(df[LABEL]) or ref not in set(df[LABEL]):
            continue
        p = paired(df, main, sc, ref)
        g = paired(df, "rota_q1_q5_ratio", sc, ref)
        row = {"scenario": sc, "reference": ref, "episodes_0_24m": p["scenario"], "diff": p["diff"], "se": p["se"],
               "rel": p["rel"], "n_lower": p["n_lower"], "n": p["n"],
               "helps": bool(p["n_lower"] >= seeds_needed(p["n"]) and -p["diff"] > 2 * p["se"]),
               "q1_q5_ratio": g["scenario"], "ratio_diff": g["diff"], "ratio_se": g["se"],
               "narrows_gap": bool(-g["diff"] > 2 * g["se"])}
        for name in ("episodes_0_59m", "severe_0_59m", "infections_0_24m", "adult_carriage"):
            pp = paired(df, VACCINE_METRICS[name], sc, ref)
            row[f"{name}_rel"], row[f"{name}_rel_se"] = pp["rel"], pp["se"] / pp["reference"] if pp["reference"] else np.nan
        for k in range(1, 6):
            pq = paired(df, f"exp_rota_episodes_per_child_year_q{k}", sc, ref)
            row[f"q{k}_rel"] = pq["rel"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "table_3b.csv"), index=False)
    res = {"n_runs": int(len(df)), "helps": table.loc[table["helps"], "scenario"].tolist(),
           "narrows_gap": table.loc[table["narrows_gap"], "scenario"].tolist()}
    with open(os.path.join(out_dir, "summary_3b.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return {"table": table, "summary": res}


ANALYSES = {"3a": ("exp3a_vaccine_limits", analyse_3a), "1a": ("exp1a_care_levers", analyse_1a),
            "2": ("exp2_shocks", analyse_2), "1b1": ("exp1b1_stockouts", analyse_1b1),
            "1b2": ("exp1b2_ramps", analyse_1b2), "1b4": ("exp1b4_recovery", analyse_1b4),
            "3b": ("exp3b_vaccine_interventions", analyse_3b),
            # the reruns of the revision (one measured year; experiments.scenarios.designs, *_rev)
            "3a_rev": ("rev_3a_vaccine_limits", analyse_3a), "1a_rev": ("rev_1a_care_levers", analyse_1a),
            "1b_rev": ("rev_1b_learning", analyse_1b_rev), "2_rev": ("rev_2_shocks", analyse_2),
            "3b_rev": ("rev_3b_vaccine_interventions", analyse_3b)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=sorted(ANALYSES))
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    name, fn = ANALYSES[a.part]
    path = a.results or os.path.join(OUT_ROOT, name, "results.parquet")
    out = a.out or os.path.join(OUT_ROOT, name, "analysis")
    res = fn(pd.read_parquet(path), out)
    print(json.dumps(res["summary"], indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()

# experiments/metrics.py
"""
Reusable metrics for SPRINGS-ABM sweep experiments.

Two kinds of functions live here:

  Per-run scalar metrics (epidemic_metrics, care_seeking_metrics, ...)
      Take a finished SVEIRModel and return a flat dict of scalars. These
      are the building blocks for a SweepSpec.metrics_fn - every experiment
      script composes the pieces it needs (e.g. epidemic outcomes +
      care-seeking outcomes) into one top-level function and passes that in.
      MUST stay side-effect-free and picklable-friendly (called in worker
      processes).

  Post-hoc complex-systems indicators (replicate_dispersion,
  early_warning_signals)
      Operate on a results/timeseries DataFrame AFTER the sweep has
      finished, not per-run - they need many replicates or many days to be
      meaningful (variance/skewness across replicates, autocorrelation
      within a time series).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from abm.constants import AgentPropertyKeys
from abm.systems.care_seeking import CareSeekingSystem
from abm.pathogens.campylobacter import Campylobacter


# ---------------------------------------------------------------------------
# Per-run scalar metrics
# ---------------------------------------------------------------------------

def epidemic_metrics(model, pathogen_names=None) -> dict:
    """Peak prevalence, cumulative child-days, attack rate, extinction flag,
    per pathogen (under-5s unless noted). `pathogen_names=None` covers every
    pathogen configured in the model, not just the one(s) being swept - handy
    since e.g. a vaccination sweep still has campy circulating in the
    background as an un-manipulated control.
    """
    if pathogen_names is None:
        pathogen_names = [p.name for p in model.pathogens]

    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    n_u5 = int(is_child.sum())

    out = {}
    for pname in pathogen_names:
        prev = np.array(model.u5_prevalence_history.get(pname, []))
        out[f"{pname}_peak_u5_prevalence"] = float(prev.max()) if prev.size else 0.0
        out[f"{pname}_peak_day"] = int(prev.argmax()) if prev.size else -1
        out[f"{pname}_cumulative_u5_days"] = float(prev.sum()) * n_u5 if prev.size else 0.0
        out[f"{pname}_extinct"] = bool(prev.size == 0 or prev.max() < 0.01)

        num_inf = model.graph.ndata[AgentPropertyKeys.num_infections(pname)].cpu().numpy()
        out[f"{pname}_attack_rate"] = float((num_inf > 0).mean())
        out[f"{pname}_attack_rate_u5"] = (
            float((num_inf[is_child] > 0).mean()) if n_u5 else 0.0
        )
    return out


def campy_route_fractions(model) -> dict:
    """Zoonotic / fecal-oral / food-borne attribution fractions for Campylobacter."""
    campy = next((p for p in model.pathogens if isinstance(p, Campylobacter)), None)
    if campy is None:
        return {}
    total = campy.total_zoonotic + campy.total_fecal_oral + campy.total_food_borne
    if total == 0:
        return {"campy_zoonotic_fraction": 0.0, "campy_fecal_oral_fraction": 0.0,
                "campy_food_borne_fraction": 0.0}
    return {
        "campy_zoonotic_fraction": campy.total_zoonotic / total,
        "campy_fecal_oral_fraction": campy.total_fecal_oral / total,
        "campy_food_borne_fraction": campy.total_food_borne / total,
    }


def care_seeking_metrics(model) -> dict:
    care_system = next((s for s in model.systems if isinstance(s, CareSeekingSystem)), None)
    if care_system is None or care_system.decisions_faced == 0:
        return {"conditional_care_rate": 0.0, "could_not_afford_rate": 0.0, "decisions_faced": 0}
    return {
        "conditional_care_rate": care_system.conditional_care_rate,
        "could_not_afford_rate": care_system.could_not_afford / care_system.decisions_faced,
        "decisions_faced": care_system.decisions_faced,
    }


def wellbeing_metrics(model) -> dict:
    """
    Wealth is a per-HOUSEHOLD pooled value (see EconomicSystem) duplicated
    across every member's agent record, not an independent per-agent
    quantity. mean_household_wealth weights every household equally
    regardless of size - the "typical household" perspective, avoiding
    double-counting large families. mean_parent_wealth is scoped
    differently, not just weighted differently: it covers only households
    that have a child (there is at most one IS_PARENT agent per household),
    since those are the only ones that ever face a care-seeking decision.
    The two can diverge substantially (e.g. parent-headed vs childless
    households - see config.py's child_cost_weight notes).
    """
    final = model.get_final_agent_states()
    is_parent = final["is_parent"].astype(bool)

    _, hh_first_idx = np.unique(final["household_id"], return_index=True)
    household_wealth = final["wealth"][hh_first_idx]

    return {
        "mean_final_health": float(final["health"].mean()),
        "mean_household_wealth": float(household_wealth.mean()),
        "mean_parent_wealth": float(final["wealth"][is_parent].mean()) if is_parent.any() else 0.0,
    }


def calibration_metrics(model) -> dict:
    """
    Mirrors sensitivity.py's OAT metrics: episodes per child-year and peak
    under-5 prevalence per pathogen, plus the Campylobacter zoonotic-route
    fraction. This is the metric set experiments/calibration/ scores against
    the empirical target ranges in experiments/calibration/targets.py, so a
    calibration search and a follow-up OAT sensitivity check stay comparable.
    """
    out = {}
    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    n_u5 = int(is_child.sum())
    sim_years = model.config.step_target / 365.0

    for p in model.pathogens:
        pname = p.name
        num_inf_u5 = model.graph.ndata[AgentPropertyKeys.num_infections(pname)].cpu().numpy()[is_child]
        out[f"{pname}_episodes_per_child_year"] = (
            float(num_inf_u5.mean()) / sim_years if n_u5 > 0 and sim_years > 0 else 0.0
        )
        prev = np.array(model.u5_prevalence_history.get(pname, []))
        out[f"{pname}_peak_u5_prevalence"] = float(prev.max()) if prev.size else 0.0

    out.update(campy_route_fractions(model))
    return out


# ---------------------------------------------------------------------------
# Post-hoc complex-systems indicators
# ---------------------------------------------------------------------------

def replicate_dispersion(df: pd.DataFrame, group_cols: list[str], metric_col: str) -> pd.DataFrame:
    """
    For each parameter combination, compute the across-replicate mean / std /
    skewness / bimodality coefficient of `metric_col`.

    Rising variance and emergent bimodality (two clusters of outcomes at the
    SAME parameter value - e.g. some replicates go extinct, others don't) are
    classic signatures of a system sitting near a tipping point, where
    stochastic noise gets amplified rather than averaged out.
    """
    def _bimodality_coefficient(x: np.ndarray) -> float:
        # Sarle's bimodality coefficient; > 0.555 is a common (informal)
        # rule-of-thumb threshold suggesting bimodality for non-normal data.
        n = len(x)
        if n < 4 or np.std(x) == 0:
            return np.nan
        s = pd.Series(x)
        skew = s.skew()
        excess_kurt = s.kurt()  # pandas already reports EXCESS kurtosis
        return (skew ** 2 + 1) / (excess_kurt + 3 + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))

    def _agg(g):
        x = g[metric_col].dropna().values
        return pd.Series({
            "mean": np.mean(x) if len(x) else np.nan,
            "std": np.std(x) if len(x) else np.nan,
            "skew": pd.Series(x).skew() if len(x) > 2 else np.nan,
            "bimodality_coef": _bimodality_coefficient(x),
            "n": len(x),
        })

    return df.groupby(group_cols).apply(_agg).reset_index()


def calibration_loss(df: pd.DataFrame, targets: dict, group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Score each swept parameter combination against empirical target ranges and
    rank by goodness-of-fit.

    `targets` is a {metric_name: (lo, hi)} dict of literature ranges - use the
    TARGETS dict from experiments/calibration/targets.py so a calibration
    search and the OAT sensitivity check (sensitivity.py) stay comparable.

    For every combination (rows sharing the same swept-parameter values), each
    metric is averaged across replicates and given a penalty of 0 when it lands
    inside [lo, hi], else its distance to the nearest band edge normalised by
    the band width. Penalties sum to `loss`; the returned frame also carries
    the mean value and an `<metric>_in_range` flag per target plus
    `n_targets_met`.

    Ranked by `n_targets_met` descending first, `loss` ascending as tiebreaker -
    NOT by raw `loss` alone. A single badly-out-of-range metric (e.g. one
    pathogen's episode rate off by 10x) can dominate a plain summed loss and
    bury a combo that actually satisfies more targets, since penalties aren't
    capped - normalising by band width keeps penalties comparable in scale
    across metrics, but doesn't bound how large a single one can get.

    Metric-name reconciliation: sensitivity.py's TARGETS use `*_peak_prevalence`
    while calibration_metrics() emits `*_peak_u5_prevalence`; the former is
    mapped onto the latter automatically. Targets whose metric is absent from
    `df` are skipped.

    `group_cols` defaults to the swept parameter columns, inferred as any column
    whose name contains "." or "[" (config dot-paths like
    "pathogens[rota].infection_prob_mean" - metric columns never do).
    """
    if group_cols is None:
        group_cols = [c for c in df.columns if ("." in c) or ("[" in c)]
    if not group_cols:
        raise ValueError("No swept-parameter columns found to group on; pass group_cols explicitly.")

    # Resolve each target metric onto an actual column in df (handle the
    # peak_prevalence -> peak_u5_prevalence naming difference).
    resolved = {}
    for metric, (lo, hi) in targets.items():
        col = metric if metric in df.columns else metric.replace("_peak_prevalence", "_peak_u5_prevalence")
        if col in df.columns:
            resolved[metric] = (col, float(lo), float(hi))
    if not resolved:
        raise ValueError("None of the target metrics are present in the results DataFrame.")

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys))
        total = 0.0
        for metric, (col, lo, hi) in resolved.items():
            val = float(g[col].mean())
            width = (hi - lo) or 1.0
            penalty = (lo - val) / width if val < lo else (val - hi) / width if val > hi else 0.0
            rec[metric] = val
            rec[f"{metric}_in_range"] = bool(lo <= val <= hi)
            total += penalty
        rec["n_targets_met"] = int(sum(rec[f"{m}_in_range"] for m in resolved))
        rec["loss"] = total
        rows.append(rec)

    return pd.DataFrame(rows).sort_values(
        ["n_targets_met", "loss"], ascending=[False, True]
    ).reset_index(drop=True)


def early_warning_signals(ts_df: pd.DataFrame, value_col: str = "u5_prevalence",
                           window: int = 30) -> pd.DataFrame:
    """
    Rolling variance and lag-1 autocorrelation of a prevalence time series
    within each run - 'critical slowing down' indicators that tend to rise
    as a system approaches a bifurcation, independent of whether the
    transition is actually crossed within that particular run.

    ts_df must have columns: run_id, day, <value_col>, and optionally
    'pathogen' (grouped on automatically if present). Pre-filter to the
    parameter combinations you care about before calling this on a large
    sweep - it's O(runs) and not vectorised across runs.
    """
    out = []
    group_cols = ["run_id"] + (["pathogen"] if "pathogen" in ts_df.columns else [])
    for keys, g in ts_df.sort_values("day").groupby(group_cols):
        g = g.reset_index(drop=True)
        roll_var = g[value_col].rolling(window).var()
        roll_ac1 = g[value_col].rolling(window).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        )
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, key_tuple))
        rec["max_rolling_variance"] = roll_var.max()
        rec["max_rolling_ac1"] = roll_ac1.max()
        rec["final_ac1"] = roll_ac1.iloc[-1] if len(roll_ac1) else np.nan
        out.append(rec)
    return pd.DataFrame(out)
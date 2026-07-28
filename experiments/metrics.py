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

    out = {"n_u5": n_u5}
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
    """
    conditional_care_rate / could_not_afford_rate are scoped to per-day
    decisions (decisions_faced), not illness episodes.
    episode_care_seeking_rate (episodes_with_care_sought / total_episodes)
    is the DHS-comparable figure instead - see
    abm/systems/care_seeking.py's module docstring and
    CareSeekingSystem._track_episodes for the distinction.
    """
    care_system = next((s for s in model.systems if isinstance(s, CareSeekingSystem)), None)
    if care_system is None:
        return {
            "conditional_care_rate": 0.0, "could_not_afford_rate": 0.0, "decisions_faced": 0,
            "episode_care_seeking_rate": 0.0, "total_episodes": 0, "episodes_with_care_sought": 0,
        }
    return {
        "conditional_care_rate": care_system.conditional_care_rate,
        "could_not_afford_rate": (
            care_system.could_not_afford / care_system.decisions_faced
            if care_system.decisions_faced > 0 else 0.0
        ),
        "decisions_faced": care_system.decisions_faced,
        "episode_care_seeking_rate": care_system.episode_care_seeking_rate,
        "total_episodes": care_system.total_episodes,
        "episodes_with_care_sought": care_system.episodes_with_care_sought,
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


def shock_response_metrics(
    ts_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    start_day: int,
    pathogen: str = "rota",
    baseline_window: int = 30,
    recovery_tolerance: float = 0.005,
    recovery_sustain_days: int = 14,
) -> pd.DataFrame:
    """
    Per-(combo, rep) sensitivity summary of a water-contamination shock
    window (see abm/systems/environment.py, experiments/shocks/run_shock_sweep.py):
    pre-shock baseline prevalence, post-shock peak, time-to-recovery, and
    excess illness-days vs. a paired magnitude=1 control run sharing the
    same rep (and therefore the same seed - orchestrator._run_one's
    seed = base_seed + rep is independent of combo, so the control and every
    shocked combo at a given rep share an identical stochastic environment
    until the shock diverges them).

    ts_df   : long time-series frame from run_sweep(..., record_timeseries=True)
              - columns run_id, rep, pathogen, day, u5_prevalence.
    meta_df : the companion results.parquet frame - must carry run_id, rep,
              "shock.duration", "shock.magnitude".
    start_day : the (fixed, non-swept) day the shock window begins.

    There is exactly one magnitude==1 combo in the shocks design (the
    explicit control, duration=0) - not one per duration value - so each
    rep has exactly one control run to match against.

    Recovery is judged against the paired control's OWN trajectory, not a
    static pre-shock baseline: this model produces one large epidemic wave
    from initial conditions that keeps declining naturally for ~150-200 days
    regardless of any shock, so a fixed start_day=60 sits inside that decline
    rather than at a quiescent equilibrium - a static-baseline definition
    would read as "recovered" almost immediately everywhere, since prevalence
    keeps falling on its own either way (confirmed empirically: it returned
    ~0 days for nearly every combo in the first pass). Comparing against the
    paired control's trajectory instead nets out that shared natural decline,
    so "recovered" means the shock's marginal effect has faded, not that the
    background epidemic has moved on. `recovered` = the trailing
    `recovery_sustain_days`-day rolling mean of (shocked - control) first
    drops to within +/- `recovery_tolerance` (absolute prevalence units) of
    zero, sustained rather than a single-day crossing, since the two runs'
    RNG streams decorrelate once the shock changes their infection counts
    (same seed only guarantees identical draws up to first divergence), so a
    single day's difference is noisy.

    `excess_illness_days` is reported in absolute under-5 child-days, not raw
    prevalence-fraction-days, since "0.7 excess prevalence-days" is not a
    directly interpretable quantity (prevalence is a fraction of the u5
    population, so summing it over days gives fraction-days, not days).
    n_u5 (under-5 population size, roughly constant per run since the
    population is closed - see abm/model/initialize_model.py) is recovered
    per run as `{pathogen}_cumulative_u5_days / (full-run sum of daily
    prevalence)`, both already present in meta_df/ts_df, rather than
    requiring metrics_fn to have recorded n_u5 explicitly - this keeps the
    function usable on older saved results too.
    """
    meta = meta_df[["run_id", "rep", "shock.duration", "shock.magnitude",
                     f"{pathogen}_cumulative_u5_days"]].drop_duplicates("run_id")
    ts = ts_df[ts_df["pathogen"] == pathogen].drop(columns=["rep"], errors="ignore").merge(
        meta, on="run_id", how="inner"
    )

    full_run_prevalence_sum = ts.groupby("run_id")["u5_prevalence"].sum()
    n_u5_by_run = (
        meta.set_index("run_id")[f"{pathogen}_cumulative_u5_days"] / full_run_prevalence_sum
    )

    controls = meta[meta["shock.magnitude"] == 1].set_index("rep")["run_id"]
    control_series = {
        rep: ts.loc[ts["run_id"] == run_id].set_index("day")["u5_prevalence"].sort_index()
        for rep, run_id in controls.items()
    }

    rows = []
    for run_id, g in ts.groupby("run_id"):
        g = g.sort_values("day")
        rep = g["rep"].iloc[0]
        duration = g["shock.duration"].iloc[0]
        magnitude = g["shock.magnitude"].iloc[0]
        end_day = start_day + duration
        n_u5 = float(n_u5_by_run[run_id])

        pre = g.loc[(g["day"] >= start_day - baseline_window) & (g["day"] < start_day), "u5_prevalence"]
        baseline = float(pre.mean()) if len(pre) else float("nan")

        post = g.loc[g["day"] >= start_day]
        if len(post):
            peak_idx = post["u5_prevalence"].idxmax()
            peak_prevalence = float(post.loc[peak_idx, "u5_prevalence"])
            peak_day = int(post.loc[peak_idx, "day"])
        else:
            peak_prevalence, peak_day = float("nan"), -1

        control = control_series.get(rep)
        if control is not None:
            window = g.loc[g["day"] >= start_day, ["day", "u5_prevalence"]].set_index("day")["u5_prevalence"]
            paired = window - control.reindex(window.index)
            excess_illness_days = float(paired.sum()) * n_u5

            after_window = paired.loc[paired.index >= end_day]
            rolling = after_window.rolling(recovery_sustain_days, min_periods=recovery_sustain_days).mean().abs()
            recovered_mask = rolling <= recovery_tolerance
            if len(rolling) and recovered_mask.any():
                recovery_idx = recovered_mask.idxmax()  # first True (chronological order preserved)
                time_to_recovery = int(recovery_idx) - end_day
                recovered = True
            else:
                time_to_recovery, recovered = float("nan"), False
        else:
            excess_illness_days = float("nan")
            time_to_recovery, recovered = float("nan"), False

        rows.append({
            "run_id": run_id, "rep": rep,
            "shock.duration": duration, "shock.magnitude": magnitude,
            "baseline_prevalence": baseline,
            "peak_prevalence": peak_prevalence, "peak_day": peak_day,
            "time_to_recovery": time_to_recovery, "recovered_by_sim_end": recovered,
            "excess_illness_days": excess_illness_days,
        })
    return pd.DataFrame(rows)
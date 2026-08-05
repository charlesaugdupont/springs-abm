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
    pairing_cols: tuple = ("rep",),
    recovery_search_from: str = "window_end",
    control_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Per-(combo, rep) sensitivity summary of a water-contamination shock
    window (see abm/systems/environment.py, experiments/shocks/run_shock_sweep.py,
    experiments/shocks/run_shock_scenarios.py): pre-shock baseline prevalence,
    post-shock peak, time-to-recovery, and excess illness-days vs. a paired
    magnitude=1 control run sharing the same rep (and therefore the same seed -
    orchestrator._run_one's seed = base_seed + rep is independent of combo, so
    the control and every shocked combo at a given rep share an identical
    stochastic environment until the shock diverges them).

    ts_df   : long time-series frame from run_sweep(..., record_timeseries=True)
              - columns run_id, rep, pathogen, day, u5_prevalence.
    meta_df : the companion results.parquet frame - must carry run_id, rep,
              "shock.duration", "shock.magnitude", plus every column in
              `pairing_cols`.
    start_day : the (fixed, non-swept) day the shock window begins.

    pairing_cols : which meta_df columns identify a run's matching control
        (duration==0, magnitude==1 run). Default ("rep",) reproduces the
        original run_shock_sweep.py design, which has exactly one control
        combo total, so `rep` alone is a unique key. run_shock_scenarios.py's
        design has *multiple* controls (one flat control plus one per cyclical
        background, e.g. the low-stress and high-stress backgrounds each
        double as the control for their own shock-on-background scenarios) -
        pass e.g. pairing_cols=("shock.bg_freq","shock.bg_amp","shock.bg_anchor","rep")
        there so each scenario pairs against its own matching background,
        not an unrelated one.

    control_mask : boolean Series aligned with meta_df's index identifying
        which rows count as "the control" to pair against. Default (None)
        uses (shock.duration==0) & (shock.magnitude==1), reproducing the
        original behavior. run_shock_scenarios.py needs this to be
        overridable because it has *multiple* duration==0/magnitude==1 rows
        (a flat control plus each cyclical-background-only scenario) that
        serve different comparison purposes depending on the question asked:
        pass the default mask with pairing_cols=("shock.bg_freq",...,"rep")
        to isolate a shock's marginal effect against its own matching
        background, or pass a mask selecting only the single true flat
        control (e.g. (bg_freq==0)&(bg_amp==0)&is_control) with
        pairing_cols=("rep",) to rank every scenario's *total* severity
        against one common baseline (the GoodBYE-Fig-3-style ranking). Two
        separate calls, not one call trying to do both at once - a static
        pairing_cols can't express "scenario 5 pairs with scenario 1, but
        scenario 1 itself pairs with scenario 0" in a single pass, since
        scenario 1 is simultaneously a control (for 5/6) and a non-control
        (relative to scenario 0).

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

    recovery_search_from : where the recovery search window begins.
        "window_end" (default, original behavior) = start_day + duration -
        correct for a "persistent" shock shape, which holds flat at peak
        magnitude for the whole window before an instant revert, so recovery
        can only begin once the hold ends. "start_day" = search the whole
        post-onset period instead - needed for a "punctuated" shape (gradual/
        continuous decay from the moment of onset, no flat hold to wait out)
        and harmless for pure-cyclical scenarios (duration=0, see below).

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

    For duration=0 combos (pure-cyclical background scenarios, no discrete
    shock event), `time_to_recovery`/`recovered_by_sim_end` are reported as
    NaN/False rather than a degenerate value - there is no discrete event to
    recover from, so "time to recovery" isn't a meaningful quantity there;
    `excess_illness_days` (a paired-control diff over the whole post-onset
    window) still is, and remains the primary comparison metric.
    """
    pairing_cols = list(pairing_cols)
    meta_cols = ["run_id", "shock.duration", "shock.magnitude", f"{pathogen}_cumulative_u5_days"]
    meta_cols += [c for c in pairing_cols if c not in meta_cols]
    meta_cols += [c for c in meta_df.columns if c.startswith("shock.") and c not in meta_cols]
    meta = meta_df[meta_cols].drop_duplicates("run_id")
    ts = ts_df[ts_df["pathogen"] == pathogen].drop(columns=["rep"], errors="ignore").merge(
        meta, on="run_id", how="inner"
    )

    full_run_prevalence_sum = ts.groupby("run_id")["u5_prevalence"].sum()
    n_u5_by_run = (
        meta.set_index("run_id")[f"{pathogen}_cumulative_u5_days"] / full_run_prevalence_sum
    )

    if control_mask is None:
        is_control = (meta["shock.duration"] == 0) & (meta["shock.magnitude"] == 1)
    else:
        is_control = control_mask.reindex(meta.index).fillna(False).astype(bool)
    controls = meta.loc[is_control].set_index(pairing_cols)["run_id"]
    if controls.index.has_duplicates:
        dupes = controls.index[controls.index.duplicated()].unique().tolist()
        raise ValueError(
            f"shock_response_metrics: pairing_cols={pairing_cols} does not uniquely identify a "
            f"control run for key(s) {dupes[:5]} - multiple control rows share the same pairing "
            f"key. Narrow control_mask (e.g. to a single scenario) or widen pairing_cols."
        )
    control_series = {
        key: ts.loc[ts["run_id"] == run_id].set_index("day")["u5_prevalence"].sort_index()
        for key, run_id in controls.items()
    }

    rows = []
    for run_id, g in ts.groupby("run_id"):
        g = g.sort_values("day")
        rep = g["rep"].iloc[0]
        duration = g["shock.duration"].iloc[0]
        magnitude = g["shock.magnitude"].iloc[0]
        end_day = start_day + duration
        n_u5 = float(n_u5_by_run[run_id])

        pairing_key = g[pairing_cols].iloc[0]
        pairing_key = pairing_key.iloc[0] if len(pairing_cols) == 1 else tuple(pairing_key)

        pre = g.loc[(g["day"] >= start_day - baseline_window) & (g["day"] < start_day), "u5_prevalence"]
        baseline = float(pre.mean()) if len(pre) else float("nan")

        post = g.loc[g["day"] >= start_day]
        if len(post):
            peak_idx = post["u5_prevalence"].idxmax()
            peak_prevalence = float(post.loc[peak_idx, "u5_prevalence"])
            peak_day = int(post.loc[peak_idx, "day"])
        else:
            peak_prevalence, peak_day = float("nan"), -1

        control = control_series.get(pairing_key)
        if control is not None:
            window = g.loc[g["day"] >= start_day, ["day", "u5_prevalence"]].set_index("day")["u5_prevalence"]
            paired = window - control.reindex(window.index)
            excess_illness_days = float(paired.sum()) * n_u5

            if duration == 0:
                time_to_recovery, recovered = float("nan"), False
            else:
                # "window_end" (original, unchanged): search starts at
                # end_day. By then a flat-held ("persistent") shock has had
                # its full `duration` to diverge the two trajectories, so
                # there's no risk of the rolling check firing before real
                # divergence has built up.
                #
                # "start_day" (new, for "punctuated"): searching from the
                # literal moment of onset is unsafe - right at start_day the
                # two runs are still identical (same seed, no divergence
                # yet), so the rolling-mean-near-zero check can spuriously
                # fire within the very first evaluable window (found via
                # run_shock_scenarios.py's pilot: this produced NEGATIVE
                # time_to_recovery for punctuated combos - "recovered"
                # before the window had even ended). Fixed by giving the
                # effect `recovery_sustain_days` to build up before
                # searching - NOT by hunting for the trajectory-wide peak
                # |paired diff| (tried first, reverted: the two runs' RNG
                # streams decorrelate over the whole rest of the run once
                # the shock changes infection counts, so a late, unrelated
                # noise spike can dominate a global argmax and push the
                # search arbitrarily far into the future - confirmed on the
                # real shocks_day200 data, where it inflated several
                # time_to_recovery values from ~13-18 days to 50-100+ days).
                if recovery_search_from == "start_day":
                    search_from = start_day + recovery_sustain_days
                else:
                    search_from = end_day
                after_window = paired.loc[paired.index >= search_from]
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
# experiments/shocks/run_shock_scenarios.py
"""
GoodBYE-Inspired Environmental Scenarios (Rotavirus only)
===========================================================
Extends the one existing shock mechanism (config.py's
SteeringParamsSVEIR.shock_daily_prob, read in abm/systems/environment.py's
EnvironmentSystem.update() - see run_shock_sweep.py's docstring for the base
mechanic) with two new axes, both inspired by Wren, Romanowska & Riede 2025
(Sci. Adv., "Bad year econometrics" - the "GoodBYE" agent-based model). Only
GoodBYE's *environmental-scenario design* (its Table 1 / Fig. 3: resource
richness as a function of frequency, amplitude, and an optional sudden
event) is adapted here - not its agent-level resilience strategies
(mobility/storage/exchange/adjustment), which have no SPRINGS analog and are
out of scope. This script sits alongside (does not modify) run_shock_sweep.py,
which remains the authoritative source of experiments/outputs/shocks_day200/.

New mechanic (both multiplicatively applied to shock_daily_prob; see
`scenario_step_callback`)
------------------------------------------------------------------------
1. Cyclical background stress: a sinusoidal modulation with `bg_freq`
   (cycles per BG_REFERENCE_DAYS) and `bg_amp` (GoodBYE's raw 0/1/5 units,
   scaled by `bg_amp_scale` into a fractional swing - deliberately NOT
   amp**sin(), which would collapse amp=1 to a constant 1.0 regardless of
   freq and make GoodBYE's low-amp/high-freq vs low-amp/low-freq scenarios
   mathematically identical). Active only from `start_day` onward
   (`bg_anchor="start_day"`, the default - see DEFAULT_ANCHOR below), not
   from day 0, so every scenario's analysis window begins at the same
   controlled phase and doesn't reopen the already-solved shocks_day200
   transient-timing confound on a new axis.

   `bg_freq`'s units: the paper's Supplementary eq. 1 defines frequency as
   "the number of times the climate curve repeats per 365 time steps" - a
   FIXED calendar-like reference period, not scaled to run length. Adopted
   here as BG_REFERENCE_DAYS, but set to 100 (not 365): with 365, freq=2
   only completes ~1.4 cycles within our ~250-day post-onset analysis
   window (GoodBYE's own runs are ~1200-1790 steps, long enough for freq=2
   to look genuinely cyclical there; ours isn't) - so freq=2 would read as
   a near-monotonic drift rather than oscillating stress. BG_REFERENCE_DAYS
   =100 preserves the paper's 2:10 (1:5) frequency RATIO while rescaling
   the absolute period so freq=2 completes 5 cycles and freq=10 completes
   25 cycles within a 250-day window - genuinely oscillating at both
   settings, chosen deliberately over extending the run length (which
   would need ~3x longer runs for freq=2 to get comparably many cycles
   under the paper's literal 365-day reference).

   `bg_amp_scale`'s calibration: the paper's shock scenarios drop the
   environment's shift (b) from 10 to 0, which collapses MEAN resource
   richness by ~99% regardless of amplitude (cos term averages to zero
   over a cycle either way) - a categorically different, far larger
   perturbation than the pure-stress scenarios' amplitude-driven swing
   (mean richness unchanged, only variance/range changes). A literal
   numeric port of that ~99%-collapse target onto `bg_amp_scale` doesn't
   work: shock_daily_prob is a bounded [0,1] probability with a small
   baseline (1/30), so pushing the *stress* peak far enough to match the
   *shock*:*stress* severity ratio implied by the paper's own formula
   (roughly 2:1, comparing the shock's ~99% mean-richness reduction to the
   high-amplitude stress scenario's ~50% peak-trough dip) would also push
   stress into saturation,
   making it indistinguishable from the shock. `bg_amp_scale` is instead
   calibrated empirically (see CALIBRATED note below) so pure-cyclical
   stress produces a real, non-zero, resolvable illness-burden effect that
   stays clearly smaller than the shock scenarios' - matching the paper's
   qualitative framing ("continuous stress had a comparatively small
   impact... relative to shocks") without chasing an unportable exact ratio.
2. Discrete shock recovery shape: "persistent" reuses run_shock_sweep.py's
   existing instant-jump/instant-revert window unchanged (matches GoodBYE's
   "rapid drop, rapid recovery after an extended hold"). "punctuated" is new:
   instant jump to peak `magnitude`, then exponential decay back toward
   baseline (time constant = duration * RECOVERY_TAU_FRAC) instead of an
   abrupt revert (GoodBYE's "rapid drop, gradual recovery"). `duration` is
   the *total event footprint* for both shapes, so a punctuated and a
   persistent event of equal `duration`/`magnitude` are directly comparable.
   `magnitude` is set to 30 (EVENT_MAGNITUDE), which exactly saturates
   shock_daily_prob at 1.0 (30 * 1/30 baseline) - deliberately mirroring
   GoodBYE's own shock representing a near-total collapse (~99% reduction),
   not an arbitrary "big" number.

The two components are multiplied, not added - this is deliberate and does
real scientific work: it naturally reproduces GoodBYE's finding that a shock
on an already-unstable background is worst, via compounding, with no
special-cased logic needed for "shock during a trough".

DEFAULT_ANCHOR ("start_day" vs "t0") turned out to be moot for this design:
with BG_REFERENCE_DAYS=100 and START_DAY=200 (an exact multiple), every
integer bg_freq we use gives phase=0 at day 200 under EITHER anchor
(verified numerically for freq in {2,4,6,10} - t0-anchor phase at day 200 is
an exact multiple of 2*pi in all four cases). The anchor machinery and
ANCHOR_COMPARISON_SCENARIO pilot check are kept (harmless, cheap, and
document the original judgment call), but no longer need to be resolved
before a full run - re-check numerically if START_DAY or BG_REFERENCE_DAYS
ever change, since the coincidence is specific to 200 being an exact
multiple of 100.

Scenario grid
--------------
`GOODBYE_SCENARIOS`: the 8 GoodBYE-named scenarios (4 pure-cyclical stress +
4 shock-on-background, using duration=14/magnitude=EVENT_MAGNITUDE=30,
saturating) plus 1 true-flat control. Scenarios 1 and 4 (the duration=0
low-/high-stress backgrounds) double as the paired controls for scenarios
5-8 - no extra runs needed for that pairing.

(An earlier version of this script also had a supplementary bg_freq x
bg_amp sensitivity grid, varying the cyclical background alone at
intermediate frequency/amplitude points beyond the 2 named levels. Removed
after review: the amplitude axis showed a clean gradient but the frequency
axis was noisy/non-monotonic at intermediate points, adding complexity
without a clean payoff - the 9 named scenarios' ranking is the clearer,
more interpretable result on its own.)

Metrics: reuses experiments.metrics.shock_response_metrics (generalized for
this script - see its docstring for `pairing_cols`/`control_mask`/
`recovery_search_from`). Two distinct comparisons are computed from the same
run: `compute_scenario_ranking()` pairs every scenario against the single
flat control (total severity, for the Fig-3-style ranking bar chart), while
`compute_shock_marginal_effect()` pairs scenarios 5-8 against their matching
background-only scenario (isolates the shock's marginal effect, netting out
the background).

Usage
-----
    python -m experiments.shocks.run_shock_scenarios --grid-id <GRID_ID> --pilot   # first
    python -m experiments.shocks.run_shock_scenarios --grid-id <GRID_ID>           # full
    python -m experiments.shocks.run_shock_scenarios --plot-only

Optional flags:
    --reps    N   replicates per combo    (default: 20, or 3 with --pilot)
    --steps   N   simulation length       (default: start_day + ANALYSIS_WINDOW_DAYS = 450;
                                            kept full-length even in --pilot so cycle
                                            visibility can actually be checked)
    --agents  N   number of agents        (default: 4000, or 800 with --pilot)
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import SVEIRCONFIG
from experiments.orchestrator import SweepSpec, run_sweep, load_results, load_timeseries
from experiments.metrics import (
    epidemic_metrics, care_seeking_metrics, wellbeing_metrics, shock_response_metrics,
)

BASELINE_SHOCK_PROB = SVEIRCONFIG.steering_parameters.shock_daily_prob  # 1/30

START_DAY = 200                 # reuse the already-validated post-transient timing (shocks_day200)
ANALYSIS_WINDOW_DAYS = 250      # post-start_day observation window (steps = START_DAY + this)
BG_REFERENCE_DAYS = 100         # fixed reference period for bg_freq (like GoodBYE's 365) - see docstring
EVENT_DURATION = 14             # discrete "sudden event" footprint (matches NAMED_COMBOS "Step" duration)
EVENT_MAGNITUDE = 30            # peak multiplier: saturates shock_daily_prob at 1.0 (30 * 1/30) - see docstring

# CALIBRATED (empirically, via a two-round pilot sweep comparing candidate
# scales against the flat control at the extreme low-freq/high-amp scenario,
# n=10 reps, 4000 agents, 450 steps): fraction-of-baseline swing per unit of
# GoodBYE amplitude (1 or 5). Round 1 (0.15/0.5/1.0/2.0/3.0) showed the
# relationship is sharply nonlinear and not resolvable by simple
# extrapolation: 0.15 -> ~0 excess illness-days (statistically indistinguishable
# from no perturbation, t~0.1), but 0.5 already jumps to ~49 (already
# comparable to the discrete shock scenarios' own ~39 pooled mean), and 1.0+
# clearly exceeds even the most severe shock scenario. Round 2 narrowed the
# search to 0.2-0.35, landing on 0.3 (~30 mean excess illness-days, t~1.6 at
# n=10) - comfortably below the shock scenarios' 39-60 range (so the two
# scenario types stay clearly distinguishable) while producing a real,
# non-zero effect, unlike the original arbitrary 0.15. Not hyper-precisely
# tuned (10 reps only resolves ~t~1.6, not >2) - the full 20-rep run has
# more resolving power and is the authoritative read.
BG_AMPLITUDE_SCALE = 0.3

RECOVERY_TAU_FRAC = 1 / 3       # punctuated shape: exponential decay time-constant as a fraction of duration
DEFAULT_ANCHOR = "start_day"    # "start_day" or "t0" - numerically equivalent for our START_DAY/BG_REFERENCE_DAYS, see docstring

ANCHOR_COMPARISON_SCENARIO = "3_low_freq_high_amp"  # longest period + highest amplitude -> most
                                                     # visible phase-alignment difference between anchors


def _cyclical_multiplier(day: int, start_day: int, ref_days: float, freq: float, amp: float,
                          anchor: str, amp_scale: float) -> float:
    """Sinusoidal background-stress multiplier on shock_daily_prob. freq =
    number of complete cycles per `ref_days` (GoodBYE's eq. 1: cycles per a
    fixed reference period, not scaled to run length - here 100, not the
    paper's 365, so freq=2/freq=10 both complete multiple cycles within our
    much shorter analysis window; see module docstring). amp = GoodBYE's raw
    amplitude units (0/1/5), scaled by `amp_scale` into a fractional swing.
    anchor="start_day": off (multiplier=1) for day < start_day, phase=0
    exactly at start_day. anchor="t0": runs continuously from day 0,
    phase-referenced to day 0 - numerically equivalent to "start_day" for
    our START_DAY/BG_REFERENCE_DAYS choice, see module docstring.
    """
    if not freq or not amp:
        return 1.0
    if anchor == "start_day" and day < start_day:
        return 1.0
    t0 = 0 if anchor == "t0" else start_day
    phase = 2 * math.pi * freq * (day - t0) / ref_days
    return max(0.05, 1.0 + amp_scale * amp * math.sin(phase))


def _discrete_shock_multiplier(day: int, start_day: int, duration: float, magnitude: float,
                                shape: str) -> float:
    """"persistent"/"none": today's existing flat-hold-then-instant-revert
    behavior (unchanged) - flat at `magnitude` for elapsed in [0, duration),
    then an instant revert. "punctuated": instant jump then exponential
    decay back toward baseline starting immediately (no plateau) - the
    decay is NOT clipped at `elapsed >= duration` the way the persistent
    hold is; `duration` only sets the decay time-constant (tau =
    duration * RECOVERY_TAU_FRAC). Recovery is asymptotic (the exponential
    naturally -> 1.0 without a hard cutoff), which is the actual point of
    "gradual recovery" - clipping it at `duration` would truncate the decay
    mid-curve and snap it to baseline, reproducing the same sharp edge as
    "persistent" and defeating the purpose of a separate shape.

    KNOWN LIMITATION (not fixed - documented, low-impact): this multiplier
    is combined multiplicatively with the cyclical background (see
    scenario_step_callback) and the combined product is clipped at 1.0.
    EVENT_MAGNITUDE=30 exactly saturates probability at 1.0 only when the
    background multiplier is at its neutral value (1.0) - so during a
    "persistent" hold on a high-amplitude/high-frequency background (e.g.
    scenario 8), the background can briefly dip the multiplier below its
    floor's threshold and pull the supposedly-flat hold back down toward
    baseline for a day or two mid-window (visible as a dip-then-recover in
    plot_scenario_curves()' bottom-right panel), rather than a clean
    constant plateau for the full duration. Confirmed this is real
    simulated behavior, not just a plotting artifact (traced the actual
    day-by-day shock_daily_prob values). Doesn't change the qualitative
    results - scenario 8 was still the most severe scenario in the ranking
    despite the dip - so left as a documented limitation rather than
    reworking the combination logic (e.g. having "persistent" ignore the
    background entirely during its hold) and re-running the grid again.
    """
    if duration <= 0 or magnitude <= 1:
        return 1.0
    elapsed = day - start_day
    if elapsed < 0:
        return 1.0
    if shape == "punctuated":
        tau = max(1.0, duration * RECOVERY_TAU_FRAC)
        return 1.0 + (magnitude - 1.0) * math.exp(-elapsed / tau)
    return magnitude if elapsed < duration else 1.0


def scenario_step_callback(model, day: int, combo: dict) -> None:
    """Mid-run driver combining the cyclical background and discrete-shock
    multipliers (multiplicatively - see module docstring). MUST stay a
    top-level function (not a lambda/closure) - pickled by reference to
    worker processes, same constraint as metrics_fn (see orchestrator.py)."""
    start = combo["shock.start_day"]
    cyc = _cyclical_multiplier(
        day, start, combo["shock.bg_ref_days"], combo["shock.bg_freq"], combo["shock.bg_amp"],
        combo["shock.bg_anchor"], combo["shock.bg_amp_scale"],
    )
    disc = _discrete_shock_multiplier(
        day, start, combo["shock.duration"], combo["shock.magnitude"], combo["shock.shape"],
    )
    model.config.steering_parameters.shock_daily_prob = min(1.0, BASELINE_SHOCK_PROB * cyc * disc)


def metrics_fn(model) -> dict:
    """Composed metric set. Must stay a top-level function (not a lambda/
    closure) so it can be pickled to worker processes."""
    out = {}
    out.update(epidemic_metrics(model))
    out.update(care_seeking_metrics(model))
    out.update(wellbeing_metrics(model))
    return out


def _full_combo(scenario: str, bg_freq: float = 0, bg_amp: float = 0, duration: float = 0,
                 magnitude: float = 1, shape: str = "none", start_day: int | None = None,
                 bg_ref_days: float | None = None, bg_anchor: str | None = None,
                 bg_amp_scale: float | None = None) -> dict:
    """Every shock.* key explicit, filled with defaults - avoids partial-dict
    surprises downstream (same principle as run_shock_sweep.py's combos).
    `bg_amp_scale` defaults to the calibrated BG_AMPLITUDE_SCALE but is
    exposed as an explicit, sweepable combo key (not just a hardcoded module
    constant) so the calibration pilot can vary it directly via run_sweep."""
    return {
        "shock.scenario": scenario,
        "shock.start_day": START_DAY if start_day is None else start_day,
        "shock.bg_ref_days": BG_REFERENCE_DAYS if bg_ref_days is None else bg_ref_days,
        "shock.bg_anchor": DEFAULT_ANCHOR if bg_anchor is None else bg_anchor,
        "shock.bg_freq": bg_freq,
        "shock.bg_amp": bg_amp,
        "shock.bg_amp_scale": BG_AMPLITUDE_SCALE if bg_amp_scale is None else bg_amp_scale,
        "shock.duration": duration,
        "shock.magnitude": magnitude,
        "shock.shape": shape,
    }


# The 8 GoodBYE-named scenarios + 1 true-flat control. Scenarios 1 and 4
# double as the paired controls for 5/6 and 7/8 respectively.
GOODBYE_SCENARIOS: dict[str, dict] = {
    key: _full_combo(key, **kwargs) for key, kwargs in {
        "0_true_flat_control":     dict(),
        "1_low_freq_low_amp":      dict(bg_freq=2,  bg_amp=1),
        "2_high_freq_low_amp":     dict(bg_freq=10, bg_amp=1),
        "3_low_freq_high_amp":     dict(bg_freq=2,  bg_amp=5),
        "4_high_freq_high_amp":    dict(bg_freq=10, bg_amp=5),
        "5_lowstress_punctuated":  dict(bg_freq=2,  bg_amp=1, duration=EVENT_DURATION,
                                         magnitude=EVENT_MAGNITUDE, shape="punctuated"),
        "6_lowstress_persistent":  dict(bg_freq=2,  bg_amp=1, duration=EVENT_DURATION,
                                         magnitude=EVENT_MAGNITUDE, shape="persistent"),
        "7_highstress_punctuated": dict(bg_freq=10, bg_amp=5, duration=EVENT_DURATION,
                                         magnitude=EVENT_MAGNITUDE, shape="punctuated"),
        "8_highstress_persistent": dict(bg_freq=10, bg_amp=5, duration=EVENT_DURATION,
                                         magnitude=EVENT_MAGNITUDE, shape="persistent"),
    }.items()
}


def build_anchor_comparison_combos() -> dict[str, dict]:
    """Both bg_anchor variants of ANCHOR_COMPARISON_SCENARIO, as two
    explicitly-labeled extra combos - included in --pilot so the actual
    trajectories can inform the DEFAULT_ANCHOR choice before a full run.
    "shock.scenario" MUST be overridden to a distinct label for each (not
    just bg_anchor) - otherwise the "start_day"-anchor variant is a
    byte-for-byte duplicate combo of the scenario-3 entry (same
    run_id via _combo_id, since every value including the label would
    match), which silently duplicates rows in results/timeseries.parquet
    and breaks shock_response_metrics' per-run_id control lookup (found via
    the pilot's own sanity check - a "cannot reindex on an axis with
    duplicate labels" crash)."""
    base = GOODBYE_SCENARIOS[ANCHOR_COMPARISON_SCENARIO]
    combos = {}
    for anchor in ("start_day", "t0"):
        label = f"{ANCHOR_COMPARISON_SCENARIO}__anchor_{anchor}"
        combos[label] = {**base, "shock.scenario": label, "shock.bg_anchor": anchor}
    return combos


def build_combos(pilot: bool = False) -> list[dict]:
    """Explicit (non-factorial) combo list - same pattern as
    run_shock_sweep.py's build_combos() / experiments/calibration's LHS
    design. pilot=True adds the anchor-comparison pair."""
    combos = list(GOODBYE_SCENARIOS.values())
    if pilot:
        combos += list(build_anchor_comparison_combos().values())
    return combos


def _spec_name(pilot: bool = False) -> str:
    return "shocks_scenarios_named_pilot" if pilot else "shocks_scenarios_named"


def build_spec(grid_id: str, reps: int, steps: int, agents: int, pilot: bool = False,
               n_cores: int | None = None) -> SweepSpec:
    return SweepSpec(
        name=_spec_name(pilot),
        grid_id=grid_id,
        params=[],  # non-factorial: combos built by build_combos(), passed explicitly to run_sweep
        metrics_fn=metrics_fn,
        step_callback=scenario_step_callback,
        reps=reps,
        steps=steps,
        agents=agents,
        record_timeseries=True,   # required: shock_response_metrics + trajectory plot both need it
        n_cores=n_cores,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_scenario_ranking(meta_df: pd.DataFrame, ts_df: pd.DataFrame, start_day: int) -> pd.DataFrame:
    """Total severity of every named scenario vs. the single flat control
    (0_true_flat_control) - the GoodBYE-Fig-3-style ranking. Every scenario
    pairs against the SAME control here (not against each other), via
    shock_response_metrics' control_mask override."""
    is_flat_control = (
        (meta_df["shock.bg_freq"] == 0) & (meta_df["shock.bg_amp"] == 0) &
        (meta_df["shock.duration"] == 0) & (meta_df["shock.magnitude"] == 1)
    )
    resp = shock_response_metrics(
        ts_df, meta_df, start_day=start_day, pathogen="rota",
        pairing_cols=("rep",), control_mask=is_flat_control,
        recovery_search_from="start_day",
    )
    return resp.merge(
        meta_df[["run_id", "shock.scenario", "shock.bg_freq", "shock.bg_amp", "shock.shape"]]
        .drop_duplicates("run_id"),
        on="run_id", how="left",
    )


def compute_shock_marginal_effect(meta_df: pd.DataFrame, ts_df: pd.DataFrame, start_day: int) -> pd.DataFrame:
    """Marginal effect of JUST the discrete shock (scenarios 5-8), isolated
    from its cyclical background by pairing against the matching
    background-only scenario (1 for 5/6, 4 for 7/8) instead of the flat
    control - answers "how much worse does the shock make things, holding
    the background fixed" rather than "how bad is this scenario overall"."""
    is_bg_control = (meta_df["shock.duration"] == 0) & (meta_df["shock.magnitude"] == 1)
    resp = shock_response_metrics(
        ts_df, meta_df, start_day=start_day, pathogen="rota",
        pairing_cols=("shock.bg_freq", "shock.bg_amp", "shock.bg_anchor", "rep"),
        control_mask=is_bg_control, recovery_search_from="start_day",
    )
    resp = resp.merge(
        meta_df[["run_id", "shock.scenario", "shock.shape"]].drop_duplicates("run_id"),
        on="run_id", how="left",
    )
    return resp[resp["shock.duration"] > 0]  # keep only the 4 actual shock-on-background rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _shock_prob_trace(combo: dict, days: list[int]) -> list[float]:
    """Reconstructs the actual shock_daily_prob(t) trace scenario_step_callback
    would drive during a run - pure function evaluation over `days`, no
    simulation needed. Used by plot_scenario_curves() for a Fig-3-style
    illustration of the input signals themselves."""
    start = combo["shock.start_day"]
    out = []
    for day in days:
        cyc = _cyclical_multiplier(
            day, start, combo["shock.bg_ref_days"], combo["shock.bg_freq"], combo["shock.bg_amp"],
            combo["shock.bg_anchor"], combo["shock.bg_amp_scale"],
        )
        disc = _discrete_shock_multiplier(
            day, start, combo["shock.duration"], combo["shock.magnitude"], combo["shock.shape"],
        )
        out.append(min(1.0, BASELINE_SHOCK_PROB * cyc * disc))
    return out


def plot_scenario_curves():
    """Fig-3-style illustration of the environmental scenario INPUT signals
    themselves (the shock_daily_prob(t) trace each scenario actually drives),
    not simulation output - mirrors the paper's Fig. 3 layout as a 2x2 grid:
    top row = pure-cyclical stress (low vs high frequency, each at low vs
    high amplitude); bottom row = shock recovery shape (punctuated vs
    persistent), shown separately on the low-stress background (scenarios
    5/6) and the high-stress background (7/8) - all 9 named scenarios (the
    flat control is the trivial flat-line case, not separately plotted) are
    represented across the 4 panels. All 4 panels share a y-axis, which
    makes the peak-height gap between the cyclical-stress panels (~0.08 max)
    and the shock panels (saturating at 1.0) directly visible - see the
    module's calibration notes on BG_AMPLITUDE_SCALE for why that gap is
    real and deliberate (empirically calibrated at the illness-OUTCOME
    level, not the input-peak level - the two aren't meant to match).

    Pure function evaluation, no simulation runs needed - regenerates
    instantly and doesn't require --plot-only to have real results on disk.
    """
    days = list(range(START_DAY - 30, START_DAY + ANALYSIS_WINDOW_DAYS + 1))

    sns.set_theme(style="white", font_scale=1.3)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)

    panels = [
        ("Low frequency (freq=2)", [
            ("Low amplitude (amp=1)", "1_low_freq_low_amp"),
            ("High amplitude (amp=5)", "3_low_freq_high_amp"),
        ]),
        ("High frequency (freq=10)", [
            ("Low amplitude (amp=1)", "2_high_freq_low_amp"),
            ("High amplitude (amp=5)", "4_high_freq_high_amp"),
        ]),
        ("Shock recovery shape\n(low-stress background)", [
            ("Punctuated (gradual recovery)", "5_lowstress_punctuated"),
            ("Persistent (abrupt recovery)", "6_lowstress_persistent"),
        ]),
        ("Shock recovery shape\n(high-stress background)", [
            ("Punctuated (gradual recovery)", "7_highstress_punctuated"),
            ("Persistent (abrupt recovery)", "8_highstress_persistent"),
        ]),
    ]
    colors = ["steelblue", "firebrick"]

    for ax, (title, series) in zip(axes.flat, panels):
        for (label, key), color in zip(series, colors):
            trace = _shock_prob_trace(GOODBYE_SCENARIOS[key], days)
            ax.plot(days, trace, label=label, color=color, linewidth=2)
        ax.axvline(START_DAY, color="gray", linestyle="--", linewidth=1, label="Onset (start_day)")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Day")
        ax.legend(fontsize=8.5, loc="upper right")

    axes[0, 0].set_ylabel("shock_daily_prob")
    axes[1, 0].set_ylabel("shock_daily_prob")
    plt.tight_layout()
    out_dir = os.path.join("experiments", "outputs", _spec_name())
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scenario_curves.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.close(fig)


def plot_trajectories(pilot: bool = False):
    spec_name = _spec_name(pilot=pilot)
    ts_df = load_timeseries(spec_name)
    meta_df = load_results(spec_name)
    if ts_df.empty or meta_df.empty:
        print(f"No successful runs found in '{spec_name}' - nothing to plot.")
        return

    actual_start_day = int(meta_df["shock.start_day"].iloc[0])  # read from data, not the module constant

    rota_ts = ts_df[ts_df["pathogen"] == "rota"]
    meta = meta_df[["run_id", "shock.scenario", "shock.duration"]].drop_duplicates("run_id")
    rota_ts = rota_ts.merge(meta, on="run_id", how="inner")

    labels = [l for l in GOODBYE_SCENARIOS if l in meta["shock.scenario"].values]
    labels += sorted(set(meta["shock.scenario"]) - set(labels))  # anchor-comparison extras, if present

    sns.set_theme(style="white", font_scale=1.6)
    ncols = 2
    nrows = -(-len(labels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 3.6 * nrows), sharey=True, squeeze=False)
    flat_axes = axes.flat

    for ax, label in zip(flat_axes, labels):
        sub = rota_ts[rota_ts["shock.scenario"] == label]
        agg = sub.groupby("day")["u5_prevalence"].agg(["mean", "std"])
        ax.plot(agg.index, agg["mean"], color="firebrick")
        ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                         alpha=0.25, color="firebrick")
        duration = sub["shock.duration"].iloc[0] if len(sub) else 0
        ax.axvline(actual_start_day, color="gray", linestyle="--", linewidth=1)
        if duration > 0:
            ax.axvspan(actual_start_day, actual_start_day + duration, color="steelblue", alpha=0.15)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Day")
        ax.set_ylabel("Rota u5 prevalence")

    for ax in list(flat_axes)[len(labels):]:
        ax.set_visible(False)

    plt.tight_layout()
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scenario_trajectories.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.close(fig)


def plot_scenario_ranking(pilot: bool = False):
    spec_name = _spec_name(pilot=pilot)
    ts_df = load_timeseries(spec_name)
    meta_df = load_results(spec_name)
    if ts_df.empty or meta_df.empty:
        print(f"No successful runs found in '{spec_name}' - nothing to plot.")
        return
    actual_start_day = int(meta_df["shock.start_day"].iloc[0])

    ranking = compute_scenario_ranking(meta_df, ts_df, start_day=actual_start_day)
    ranking = ranking[ranking["shock.scenario"].isin(GOODBYE_SCENARIOS.keys())]  # drop anchor-check extras
    agg = ranking.groupby("shock.scenario")["excess_illness_days"].agg(["mean", "std"])
    agg = agg.reindex([k for k in GOODBYE_SCENARIOS if k in agg.index])

    sns.set_theme(style="white", font_scale=1.3)
    fig, ax = plt.subplots(figsize=(8, 5))
    order = agg["mean"].sort_values().index
    ax.barh(order, agg.loc[order, "mean"], xerr=agg.loc[order, "std"].fillna(0), color="firebrick", alpha=0.85)
    ax.set_xlabel("Excess under-5 illness-days vs. flat control")
    ax.set_title("Severity ranking of GoodBYE-inspired environmental scenarios")
    plt.tight_layout()
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scenario_ranking.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.close(fig)


def plot_results(pilot: bool = False):
    plot_scenario_curves()
    plot_trajectories(pilot=pilot)
    plot_scenario_ranking(pilot=pilot)


def main():
    parser = argparse.ArgumentParser(description="GoodBYE-inspired environmental shock scenarios (Rotavirus)")
    parser.add_argument("-g", "--grid-id", required=False)
    parser.add_argument("-r", "--reps", type=int, default=None, help="Default: 20 (full) or 3 (--pilot)")
    parser.add_argument("-s", "--steps", type=int, default=None,
                         help=f"Default: {START_DAY + ANALYSIS_WINDOW_DAYS} (start_day + ANALYSIS_WINDOW_DAYS) - "
                              "kept full-length even in --pilot so cycle visibility can be checked")
    parser.add_argument("-n", "--agents", type=int, default=None, help="Default: 4000 (full) or 800 (--pilot)")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                         help="Fast smoke test: the 9 named scenarios + an anchor-comparison "
                              "pair, few reps, fewer agents (steps stays full-length). Use this first "
                              "to sanity-check the cyclical/decay mechanic and decide the bg_anchor "
                              "default before committing to a full run.")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.plot_only:
        plot_results(pilot=args.pilot)
        return

    if not args.grid_id:
        parser.error("--grid-id is required unless --plot-only is set.")

    reps = args.reps if args.reps is not None else (3 if args.pilot else 20)
    steps = args.steps if args.steps is not None else (START_DAY + ANALYSIS_WINDOW_DAYS)
    agents = args.agents if args.agents is not None else (800 if args.pilot else 4000)

    if args.pilot:
        print("*** PILOT MODE: 9 named scenarios + anchor-comparison pair, reduced reps/agents "
              "(steps kept full-length). For a mechanism/anchor sanity check only - do not draw "
              "conclusions from these results. ***\n")

    spec = build_spec(args.grid_id, reps, steps, agents, pilot=args.pilot, n_cores=args.workers)
    combos = build_combos(pilot=args.pilot)
    print(f"Design: {len(combos)} combos, {reps} reps each ({len(combos) * reps} runs).")
    run_sweep(spec, combos=combos)
    plot_results(pilot=args.pilot)


if __name__ == "__main__":
    main()

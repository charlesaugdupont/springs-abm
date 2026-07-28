# experiments/shocks/run_shock_sweep.py
"""
Water-Contamination Shock Sweep (Rotavirus only)
=================================================
Models ecological-style disturbance events on the model's one existing shock
mechanism: config.py's SteeringParamsSVEIR.shock_daily_prob, read in
abm/systems/environment.py's EnvironmentSystem.update(). Every day, a
Bernoulli(shock_daily_prob) roll either fires or doesn't; if it fires, EVERY
currently-clean water cell is contaminated (abm/systems/environment.py's
_water_shock() - full, all-or-nothing; only the daily firing probability is
stochastic, never the extent). This only affects Rotavirus (shared water
reservoir - see abm/pathogens/rotavirus.py's _water_to_human/_human_to_water
and abm/systems/household.py's water-carrying pathway) - Campylobacter has no
water route and is out of scope here.

"Pulse", "step", and "continuous" are not separate mechanisms - they're
archetypal regions of one shared (duration, magnitude) space: during
[start_day, start_day + duration), shock_daily_prob is multiplied by
`magnitude` (reverting to baseline outside the window). Pulse = short +
extreme (flash flood), step = medium duration + extreme with sharp edges
(major flood/infrastructure failure), continuous = long duration + moderate
(rainy season). start_day is held fixed across the sweep; duration and
magnitude are the two swept axes, giving a genuine dose-response surface
with pulse/step/continuous as illustrative points on it.

Mid-run config mutation is new orchestrator infrastructure - see
experiments/orchestrator.py's SweepSpec.step_callback and _run_one(). When
step_callback is None (every other experiment), behavior is unchanged.

Metrics recorded per run: same composed scalar set as the other experiments
(epidemic_metrics + care_seeking_metrics + wellbeing_metrics) PLUS a
post-hoc dose-response frame (experiments.metrics.shock_response_metrics)
computed from the recorded time series after the sweep finishes: pre-shock
baseline, post-shock peak, time-to-recovery, and excess-prevalence-days vs.
a paired magnitude=1 control run sharing the same rep/seed.

Usage
-----
    python -m experiments.shocks.run_shock_sweep --grid-id <GRID_ID>
    python -m experiments.shocks.run_shock_sweep --plot-only

Optional flags:
    --reps    N   replicates per combo          (default: 20, or 2 with --pilot)
    --steps   N   simulation length in days      (default: 300, or 150 with --pilot)
    --agents  N   number of agents               (default: 4000, or 800 with --pilot)
"""
import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

from config import SVEIRCONFIG
from experiments.orchestrator import SweepSpec, run_sweep, load_results, load_timeseries
from experiments.metrics import (
    epidemic_metrics, care_seeking_metrics, wellbeing_metrics, shock_response_metrics,
)

SPEC_NAME = "shocks"

BASELINE_SHOCK_PROB = SVEIRCONFIG.steering_parameters.shock_daily_prob  # 1/30

START_DAY = 60
DURATIONS = [3, 7, 14, 30, 60, 120]      # days
MAGNITUDES = [2, 5, 10, 20, 30]          # multiplier on shock_daily_prob

PILOT_DURATIONS = [7, 30]
PILOT_MAGNITUDES = [10, 30]

# Illustrative points on the dose-response surface for the trajectory figure -
# all members of the full grid above (no extra runs needed).
NAMED_COMBOS = {
    "Control (no shock)":    {"shock.duration": 0,   "shock.magnitude": 1},
    "Pulse (3d, 30x)":       {"shock.duration": 3,   "shock.magnitude": 30},
    "Step (14d, 20x)":       {"shock.duration": 14,  "shock.magnitude": 20},
    "Continuous (120d, 5x)": {"shock.duration": 120, "shock.magnitude": 5},
}


def shock_step_callback(model, day: int, combo: dict) -> None:
    """Mid-run driver: sets steering_parameters.shock_daily_prob to
    BASELINE_SHOCK_PROB * magnitude for day in [start_day, start_day+duration),
    and back to BASELINE_SHOCK_PROB outside that window. MUST stay a
    top-level function (not a lambda/closure) - pickled by reference to
    worker processes, same constraint as metrics_fn (see orchestrator.py).
    duration=0 (the control combo) makes the window permanently empty, so
    shock_daily_prob never leaves baseline regardless of magnitude.
    """
    start = combo["shock.start_day"]
    end = start + combo["shock.duration"]
    magnitude = combo["shock.magnitude"] if start <= day < end else 1.0
    model.config.steering_parameters.shock_daily_prob = min(1.0, BASELINE_SHOCK_PROB * magnitude)


def metrics_fn(model) -> dict:
    """Composed metric set. Must stay a top-level function (not a lambda/
    closure) so it can be pickled to worker processes."""
    out = {}
    out.update(epidemic_metrics(model))
    out.update(care_seeking_metrics(model))
    out.update(wellbeing_metrics(model))
    return out


def _spec_name(pilot: bool = False, start_day: int | None = None, named_only: bool = False) -> str:
    start_day = START_DAY if start_day is None else start_day
    suffix = (
        ("_named" if named_only else "")
        + (f"_day{start_day}" if start_day != START_DAY else "")
        + ("_pilot" if pilot else "")
    )
    return f"{SPEC_NAME}{suffix}"


def build_combos(pilot: bool = False, start_day: int | None = None, named_only: bool = False) -> list[dict]:
    """Explicit (non-factorial) combo list: build_spec()'s params=[] leaves
    this to run_sweep(spec, combos=...) - same pattern as
    experiments/calibration/run_calibration.py's LHS design. One control
    combo (duration=0, magnitude=1) plus the full duration x magnitude grid -
    NOT magnitude=1 folded into the grid across every duration, which would
    produce (for a fixed rep) bit-identical redundant runs (seed depends only
    on rep, and magnitude=1 collapses the schedule to a no-op regardless of
    duration).

    named_only=True instead runs just the 4 NAMED_COMBOS illustrative points
    (control/pulse/step/continuous) - e.g. for a cheap targeted comparison
    of shock timing (start_day) without re-running the full grid."""
    start_day = START_DAY if start_day is None else start_day
    if named_only:
        return [
            {"shock.start_day": start_day, "shock.duration": c["shock.duration"],
             "shock.magnitude": c["shock.magnitude"]}
            for c in NAMED_COMBOS.values()
        ]
    durations = PILOT_DURATIONS if pilot else DURATIONS
    magnitudes = PILOT_MAGNITUDES if pilot else MAGNITUDES
    combos = [
        {"shock.start_day": start_day, "shock.duration": d, "shock.magnitude": m}
        for d in durations for m in magnitudes
    ]
    combos.append({"shock.start_day": start_day, "shock.duration": 0, "shock.magnitude": 1})
    return combos


def build_spec(grid_id: str, reps: int, steps: int, agents: int, pilot: bool = False,
               start_day: int | None = None, named_only: bool = False,
               n_cores: int | None = None) -> SweepSpec:
    return SweepSpec(
        name=_spec_name(pilot, start_day, named_only),
        grid_id=grid_id,
        params=[],  # non-factorial: combos built by build_combos(), passed explicitly to run_sweep
        metrics_fn=metrics_fn,
        step_callback=shock_step_callback,
        reps=reps,
        steps=steps,
        agents=agents,
        record_timeseries=True,   # required: shock_response_metrics + trajectory plot both need it
        n_cores=n_cores,
    )


def plot_trajectories(pilot: bool = False, start_day: int | None = None, named_only: bool = False):
    spec_name = _spec_name(pilot, start_day, named_only)
    ts_df = load_timeseries(spec_name)
    meta_df = load_results(spec_name)
    if ts_df.empty or meta_df.empty:
        print(f"No successful runs found in '{spec_name}' - nothing to plot.")
        return

    actual_start_day = int(meta_df["shock.start_day"].iloc[0])  # read from data, not the module constant

    rota_ts = ts_df[ts_df["pathogen"] == "rota"]
    meta = meta_df[["run_id", "rep", "shock.duration", "shock.magnitude"]].drop_duplicates("run_id")
    rota_ts = rota_ts.merge(meta, on="run_id", how="inner")

    combos_to_plot = NAMED_COMBOS if (not pilot or named_only) else {
        f"duration={c['shock.duration']}, magnitude={c['shock.magnitude']}": c
        for c in build_combos(pilot=True)
    }

    sns.set_theme(style="white", font_scale=1.0)
    n = len(combos_to_plot)
    ncols = 2
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), sharey=True, squeeze=False)
    flat_axes = axes.flat

    for ax, (label, c) in zip(flat_axes, combos_to_plot.items()):
        sub = rota_ts[(rota_ts["shock.duration"] == c["shock.duration"]) &
                      (rota_ts["shock.magnitude"] == c["shock.magnitude"])]
        agg = sub.groupby("day")["u5_prevalence"].agg(["mean", "std"])
        ax.plot(agg.index, agg["mean"], color="firebrick")
        ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                         alpha=0.25, color="firebrick")
        if c["shock.duration"] > 0:
            ax.axvspan(actual_start_day, actual_start_day + c["shock.duration"], color="steelblue",
                       alpha=0.15, label="Shock window")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("Day")
        ax.set_ylabel("Rota u5 prevalence")

    for ax in list(flat_axes)[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "shock_trajectories.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.close(fig)


def plot_dose_response(pilot: bool = False, start_day: int | None = None):
    spec_name = _spec_name(pilot, start_day, named_only=False)
    meta_df = load_results(spec_name)
    ts_df = load_timeseries(spec_name)
    if meta_df.empty or ts_df.empty:
        print(f"No successful runs found in '{spec_name}' - nothing to plot.")
        return

    actual_start_day = int(meta_df["shock.start_day"].iloc[0])  # read from data, not the module constant
    resp = shock_response_metrics(ts_df, meta_df, start_day=actual_start_day, pathogen="rota")
    grid = resp[resp["shock.duration"] > 0]   # control has no (duration, magnitude) grid cell

    def _pivot(metric, agg="mean"):
        return grid.pivot_table(index="shock.duration", columns="shock.magnitude",
                                 values=metric, aggfunc=agg).sort_index(ascending=False)

    sns.set_theme(style="white", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    sns.heatmap(_pivot("peak_prevalence"), annot=True, fmt=".3f", cmap="Reds", ax=axes[0, 0],
                cbar_kws={"label": "Mean peak u5 prevalence"})
    axes[0, 0].set_title("Peak Under-5 Prevalence (Rotavirus)")
    axes[0, 0].set_xlabel("Magnitude (x baseline shock prob)")
    axes[0, 0].set_ylabel("Duration (days)")

    sns.heatmap(_pivot("excess_prevalence_days"), annot=True, fmt=".1f", cmap="Oranges",
                ax=axes[0, 1], cbar_kws={"label": "Excess prevalence-days vs. control"})
    axes[0, 1].set_title("Excess Prevalence-Days\n(vs. paired magnitude=1 control, same rep/seed)")
    axes[0, 1].set_xlabel("Magnitude (x baseline shock prob)")
    axes[0, 1].set_ylabel("")

    sns.heatmap(_pivot("time_to_recovery"), annot=True, fmt=".0f", cmap="Purples", ax=axes[1, 0],
                cbar_kws={"label": "Days after window-end to recover"})
    axes[1, 0].set_title("Time to Recovery\n(blank cell = not recovered by sim end, mean over reps that did)")
    axes[1, 0].set_xlabel("Magnitude (x baseline shock prob)")
    axes[1, 0].set_ylabel("Duration (days)")

    sns.heatmap(_pivot("recovered_by_sim_end", agg="mean"), annot=True, fmt=".0%", cmap="Greens",
                ax=axes[1, 1], vmin=0, vmax=1, cbar_kws={"label": "Fraction of reps recovered by sim end"})
    axes[1, 1].set_title("Recovery Reliability")
    axes[1, 1].set_xlabel("Magnitude (x baseline shock prob)")
    axes[1, 1].set_ylabel("")

    plt.tight_layout()
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "shock_dose_response.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.close(fig)


def plot_results(pilot: bool = False, start_day: int | None = None, named_only: bool = False):
    plot_trajectories(pilot=pilot, start_day=start_day, named_only=named_only)
    if named_only:
        print("named_only run: skipping the dose-response heatmap (needs the full duration x "
              "magnitude grid) - see the trajectory figure and/or query shock_response_metrics "
              "directly for a numeric comparison.")
    else:
        plot_dose_response(pilot=pilot, start_day=start_day)


def main():
    parser = argparse.ArgumentParser(description="Water-Contamination Shock Sweep (Rotavirus)")
    parser.add_argument("-g", "--grid-id", required=False)
    parser.add_argument("-r", "--reps", type=int, default=None,
                        help="Default: 20 (full sweep) or 2 (--pilot)")
    parser.add_argument("-s", "--steps", type=int, default=None,
                        help="Default: 300 (full sweep) or 150 (--pilot) - must stay "
                             ">= start_day + max(duration) + a post-shock recovery window")
    parser.add_argument("-n", "--agents", type=int, default=None,
                        help="Default: 4000 (full sweep) or 800 (--pilot)")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="Fast end-to-end smoke test: 2x2 duration/magnitude grid + "
                             "control, few reps, shorter run, fewer agents. Use this first "
                             "to confirm the shock mechanism/timing before the full sweep.")
    parser.add_argument("--start-day", type=int, default=None,
                        help=f"Day the shock window begins (default: {START_DAY}). Writes to "
                             "a separate output dir (suffixed _dayN) so it doesn't clobber the "
                             "default-timing results.")
    parser.add_argument("--named-only", action="store_true",
                        help="Run only the 4 illustrative NAMED_COMBOS (control/pulse/step/"
                             "continuous) instead of the full duration x magnitude grid - e.g. "
                             "for a cheap targeted comparison of shock timing (--start-day) "
                             "without re-running the whole grid.")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.plot_only:
        plot_results(pilot=args.pilot, start_day=args.start_day, named_only=args.named_only)
        return

    if not args.grid_id:
        parser.error("--grid-id is required unless --plot-only is set.")

    reps = args.reps if args.reps is not None else (2 if args.pilot else 20)
    steps = args.steps if args.steps is not None else (150 if args.pilot else 300)
    agents = args.agents if args.agents is not None else (800 if args.pilot else 4000)

    if args.pilot:
        print("*** PILOT MODE: 2x2 grid + control, reduced reps/steps/agents. "
              "For a mechanism/timing sanity check only - do not draw conclusions "
              "from these results. ***\n")

    spec = build_spec(args.grid_id, reps, steps, agents, pilot=args.pilot,
                       start_day=args.start_day, named_only=args.named_only, n_cores=args.workers)
    combos = build_combos(pilot=args.pilot, start_day=args.start_day, named_only=args.named_only)
    print(f"Design: {len(combos)} combos, {reps} reps each ({len(combos) * reps} runs).")
    run_sweep(spec, combos=combos)
    plot_results(pilot=args.pilot, start_day=args.start_day, named_only=args.named_only)


if __name__ == "__main__":
    main()

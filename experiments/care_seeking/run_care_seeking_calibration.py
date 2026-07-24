# experiments/care_seeking/run_care_seeking_calibration.py
"""
Care-Seeking Calibration: Fitting episode_care_seeking_rate to Ghana DHS
=========================================================================
The OAT and interaction sweeps (run_care_seeking_oat.py,
run_care_seeking_interaction.py) established that daily_income_rate and
cost_of_care are the two parameters that directly and dominantly drive
episode_care_seeking_rate - the other four behavioral/economic parameters
tested there barely move it directly. This script formalizes that into an
actual calibration search, mirroring experiments/calibration/run_calibration.py's
LHS + rank + adopt workflow: Latin-Hypercube sample the (income, cost) plane,
score each point's episode_care_seeking_rate against the Ghana DHS-derived
target band in experiments/care_seeking/targets.py via the same generic
experiments.metrics.calibration_loss used for the epidemic calibration, and
rank.

The interaction sweep already covers this same 2D space on a coarse 7x7 grid
(see experiments/outputs/care_seeking_interaction/) and found points landing
close to the DHS figure (e.g. income~0.0375-0.045, cost~0.025-0.075). This
search covers the space more finely via LHS (not grid-locked) and with enough
replicates to nail down a precise best-fit point suitable for adoption into
config.py, the same way experiments/calibration/ produced the adopted
epidemic-transmission parameters.

could_not_afford_rate, illness burden, and household wealth are recorded and
reported for context (the interaction sweep showed near-DHS-fit points can
carry a high could_not_afford_rate) but are NOT part of the loss - there's no
empirical target for them, unlike episode_care_seeking_rate.

Outputs (experiments/outputs/care_seeking_calibration[_pilot]/)
-----------------------------------------------------------------
  results.parquet                       one row per (combo x rep)
  care_seeking_calibration_ranked.csv   combos ranked by target fit
  best_params.json                      best-fit (income, cost_of_care), ready to adopt
  care_seeking_calibration_scatter.png  episode_care_seeking_rate over the (income, cost) plane
  care_seeking_calibration_best_fit.png best set's metric against the DHS target band

Usage
-----
    python -m experiments.care_seeking.run_care_seeking_calibration --grid-id <GRID_ID> --pilot
    python -m experiments.care_seeking.run_care_seeking_calibration --grid-id <GRID_ID> \
        --samples 80 --reps 15 --steps 250
    python -m experiments.care_seeking.run_care_seeking_calibration --plot-only
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

from config import SVEIRCONFIG
from experiments.orchestrator import SweepSpec, run_sweep, load_results, get_param
from experiments.metrics import epidemic_metrics, care_seeking_metrics, wellbeing_metrics, calibration_loss
from experiments.care_seeking.targets import TARGETS, DHS_GHANA_CARE_SEEKING_RATE, DHS_SSA_RATE_CI

SPEC_NAME = "care_seeking_calibration"
DEFAULT_GRID_ID = "7d9ce7c720a6"

# Same window already explored by run_care_seeking_interaction.py's full grid -
# near-DHS points landed solidly interior (income~0.0375-0.045, cost~0.025-
# 0.075), not at either edge, so there's no reason to widen these bounds.
CALIB_BOUNDS = {
    "steering_parameters.daily_income_rate": (0.015, 0.06),
    "steering_parameters.cost_of_care":      (0.0,   0.075),
}

# Diagnostic-only metrics (not scored against a target, just reported for
# context alongside the best-fit point).
DIAGNOSTIC_METRICS = ["could_not_afford_rate", "rota_cumulative_u5_days",
                       "campy_cumulative_u5_days", "mean_parent_wealth"]


def metrics_fn(model) -> dict:
    """Composed metric set. Must stay a top-level function (not a lambda/
    closure) so it can be pickled to worker processes."""
    out = {}
    out.update(epidemic_metrics(model))
    out.update(care_seeking_metrics(model))
    out.update(wellbeing_metrics(model))
    return out


def baseline_combo() -> dict:
    """The current config defaults for the swept params, as an anchor point."""
    return {path: float(get_param(SVEIRCONFIG, path)) for path in CALIB_BOUNDS}


def sample_lhs(n_samples: int, seed: int) -> list[dict]:
    """Latin-Hypercube sample the parameter space, scaled to CALIB_BOUNDS."""
    paths = list(CALIB_BOUNDS.keys())
    l_bounds = np.array([CALIB_BOUNDS[p][0] for p in paths])
    u_bounds = np.array([CALIB_BOUNDS[p][1] for p in paths])
    sampler = qmc.LatinHypercube(d=len(paths), seed=seed)
    scaled = qmc.scale(sampler.random(n=n_samples), l_bounds, u_bounds)
    return [dict(zip(paths, (float(v) for v in row))) for row in scaled]


def build_spec(grid_id: str, reps: int, steps: int, agents: int, pilot: bool = False,
               n_cores: int | None = None) -> SweepSpec:
    return SweepSpec(
        name=f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME,
        grid_id=grid_id,
        params=[],  # non-factorial: combos are passed explicitly to run_sweep
        metrics_fn=metrics_fn,
        reps=reps,
        steps=steps,
        agents=agents,
        record_timeseries=False,
        n_cores=n_cores,
    )


def score_and_report(pilot: bool = False):
    spec_name = f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME
    df = load_results(spec_name)

    if df.empty or "episode_care_seeking_rate" not in df.columns:
        print(f"\nNo successful runs found in '{spec_name}' results - nothing to score.")
        print("Check the per-run tracebacks printed during the sweep above.")
        return

    ranked = calibration_loss(df, TARGETS)
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)

    # Attach diagnostic-only metrics (mean per combo) alongside the ranked
    # target columns, since calibration_loss() only carries target columns.
    param_cols = list(CALIB_BOUNDS.keys())
    diag = df.groupby(param_cols, dropna=False)[DIAGNOSTIC_METRICS].mean().reset_index()
    ranked = ranked.merge(diag, on=param_cols, how="left")

    ranked.to_csv(os.path.join(out_dir, "care_seeking_calibration_ranked.csv"), index=False)

    best = ranked.iloc[0]
    best_params = {p: float(best[p]) for p in param_cols}
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({
            "params": best_params,
            "loss": float(best["loss"]),
            "n_targets_met": int(best["n_targets_met"]),
            "n_targets": len(TARGETS),
            "targets": {k: list(v) for k, v in TARGETS.items()},
            "diagnostics": {m: float(best[m]) for m in DIAGNOSTIC_METRICS},
        }, f, indent=2)

    # Console summary
    metric_cols = [m for m in TARGETS if m in ranked.columns]
    print(f"\n--- Care-seeking calibration ranking ({len(ranked)} combos) ---")
    print(f"  Best loss = {best['loss']:.4f}  |  targets met = {int(best['n_targets_met'])}/{len(metric_cols)}")
    print("  Best-fit parameters:")
    for p in param_cols:
        print(f"    {p:<45} = {best[p]:.5g}")
    print("  Best-fit metric vs DHS target band:")
    for m in metric_cols:
        lo, hi = TARGETS[m]
        flag = "OK " if best[f"{m}_in_range"] else "OUT"
        print(f"    [{flag}] {m:<26} = {best[m]:.4g}   target [{lo}, {hi}] "
              f"(Ghana DHS point estimate: {DHS_GHANA_CARE_SEEKING_RATE:.3f})")
    print("  Diagnostics at best-fit point (not scored, context only):")
    for m in DIAGNOSTIC_METRICS:
        print(f"    {m:<30} = {best[m]:.4g}")
    print(f"  (Pooled sub-Saharan Africa reference: {DHS_SSA_RATE_CI[0]:.3f}-{DHS_SSA_RATE_CI[1]:.3f})")
    print(f"\n  Ranked table -> {os.path.join(out_dir, 'care_seeking_calibration_ranked.csv')}")
    print(f"  Best params  -> {os.path.join(out_dir, 'best_params.json')}")

    _plot_scatter(ranked, param_cols, metric_cols[0], out_dir)
    _plot_best_fit(best, metric_cols, out_dir)


def _plot_scatter(ranked, param_cols, target_metric, out_dir):
    """episode_care_seeking_rate over the (income, cost) plane sampled by
    LHS - not a grid, so a scatter rather than a heatmap."""
    income_col, cost_col = param_cols
    lo, hi = TARGETS[target_metric]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(ranked[income_col], ranked[cost_col], c=ranked[target_metric],
                     cmap="Blues", s=90, edgecolors="k", linewidths=0.4)
    in_band = ranked[f"{target_metric}_in_range"]
    ax.scatter(ranked.loc[in_band, income_col], ranked.loc[in_band, cost_col],
               facecolors="none", edgecolors="red", s=160, linewidths=1.6,
               label=f"in DHS band [{lo}, {hi}]")
    best = ranked.iloc[0]
    ax.scatter([best[income_col]], [best[cost_col]], marker="*", s=400,
               color="gold", edgecolors="k", linewidths=0.8, label="best fit", zorder=5)
    fig.colorbar(sc, ax=ax, label=target_metric)
    ax.set_xlabel("Daily income rate")
    ax.set_ylabel("Cost of care")
    ax.set_title(f"Care-Seeking Calibration: {target_metric}\nover LHS-sampled (income, cost) plane")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "care_seeking_calibration_scatter.png")
    plt.savefig(path, dpi=170, bbox_inches="tight")
    print(f"  Figure saved -> {path}")
    plt.close(fig)


def _plot_best_fit(best, metric_cols, out_dir):
    """Best set's metric value against the DHS target band."""
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    for i, m in enumerate(metric_cols):
        ax = axes[0][i]
        lo, hi = TARGETS[m]
        val = best[m]
        ax.axhspan(lo, hi, color="green", alpha=0.18, label="DHS 95% CI")
        ax.axhline(DHS_GHANA_CARE_SEEKING_RATE, color="green", linestyle="--", linewidth=1,
                   label="DHS point estimate")
        ax.bar([0], [val], width=0.5,
               color="#2196F3" if best[f"{m}_in_range"] else "#E53935")
        ax.set_xticks([])
        ax.set_title(m.replace("_", "\n"), fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right", fontsize=6.5)
    fig.suptitle(f"Best-fit care-seeking rate vs Ghana DHS target  (loss={best['loss']:.4f})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "care_seeking_calibration_best_fit.png")
    plt.savefig(path, dpi=170, bbox_inches="tight")
    print(f"  Figure saved -> {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Care-seeking calibration against Ghana DHS for SPRINGS-ABM.")
    parser.add_argument("-g", "--grid-id", default=DEFAULT_GRID_ID,
                        help=f"Grid ID (default: {DEFAULT_GRID_ID})")
    parser.add_argument("--samples", type=int, default=None,
                        help="LHS sample points. Default: 80 (full) or 8 (--pilot)")
    parser.add_argument("-r", "--reps", type=int, default=None,
                        help="Replicates per point. Default: 15 (full) or 3 (--pilot)")
    parser.add_argument("-s", "--steps", type=int, default=None,
                        help="Simulation days. Default: 250 (full) or 60 (--pilot)")
    parser.add_argument("-n", "--agents", type=int, default=None,
                        help="Number of agents. Default: 4000 (full) or 800 (--pilot)")
    parser.add_argument("--seed", type=int, default=12345, help="LHS sampler seed (default: 12345)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes. Default: orchestrator's own default.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip the sweep; re-score and re-plot existing results.")
    parser.add_argument("--pilot", action="store_true",
                        help="Fast end-to-end smoke test: few LHS points, few reps, "
                             "short run, fewer agents. Sanity-checks the pipeline "
                             "and timing before a full calibration search.")
    args = parser.parse_args()

    if args.plot_only:
        score_and_report(pilot=args.pilot)
        return

    samples = args.samples if args.samples is not None else (8 if args.pilot else 80)
    reps = args.reps if args.reps is not None else (3 if args.pilot else 15)
    steps = args.steps if args.steps is not None else (60 if args.pilot else 250)
    agents = args.agents if args.agents is not None else (800 if args.pilot else 4000)

    if args.pilot:
        print("*** PILOT MODE: few LHS points, reduced reps/steps/agents. "
              "For a timing/sanity check only - do not adopt these parameters. ***\n")

    combos = [baseline_combo()] + sample_lhs(samples, args.seed)
    print(f"Design: 1 baseline anchor + {samples} LHS points = {len(combos)} combos, "
          f"{reps} reps each ({len(combos) * reps} runs).")

    spec = build_spec(args.grid_id, reps, steps, agents, pilot=args.pilot, n_cores=args.workers)
    run_sweep(spec, combos=combos)
    score_and_report(pilot=args.pilot)


if __name__ == "__main__":
    main()

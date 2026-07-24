# experiments/care_seeking/run_care_seeking_oat.py
"""
Care-Seeking & Household-Economics OAT Sensitivity
=====================================================
One-at-a-time sweep over 6 behavioral/economic parameters, holding every
other parameter at its live config.py baseline, to see how each shapes
care-seeking behavior, resulting illness burden, and household wealth.

This is exploratory, not calibration: there's no fitted target here (the
one sourced empirical figure - Ghana DHS childhood-diarrhea care-seeking,
69.2% - is shown only as a reference band on the episode_care_seeking_rate
panel, not something this sweep tries to hit).

Parameters swept (baseline in parentheses)
-------------------------------------------
  daily_income_rate (0.03), cost_of_care (0.025)
      - affordability axis. Swept as multipliers of the live baseline (not
        hardcoded absolutes) so this grid re-centers automatically if these
        values are ever recalibrated later - sensitivity.py's hardcoded
        epidemic-parameter grids went stale like this twice in one session.
  treatment_success_prob (0.80), natural_worsening_prob (0.35)
      - treatment-quality axis: how much a given care-seeking rate actually
        translates into averted illness.
  parent_stress_health_impact (0.30), untreated_severity_penalty (0.20)
      - consequence-severity axis: how costly each branch of the decision
        (in abm/systems/care_seeking.py::_evaluate_cpt / _evaluate_ev) looks.

cpt_theta/cpt_eta (value-function curvature) and the alpha/gamma/lambda
persona-heterogeneity ranges are deliberately NOT swept here.

Metrics recorded per run
-------------------------
  episode_care_seeking_rate, conditional_care_rate, could_not_afford_rate,
  decisions_faced   (care_seeking_metrics - see its docstring for why
                     episode_care_seeking_rate, not conditional_care_rate,
                     is the DHS-comparable figure)
  rota_/campy_ peak_u5_prevalence, cumulative_u5_days, attack_rate_u5, extinct
                     (epidemic_metrics - care-seeking feeds back into
                     transmission, not just individual illness severity)
  mean_final_health, mean_household_wealth, mean_parent_wealth
                     (wellbeing_metrics)

Usage
-----
    python -m experiments.care_seeking.run_care_seeking_oat --grid-id <GRID_ID>
    python -m experiments.care_seeking.run_care_seeking_oat --plot-only

Optional flags:
    --reps    N   replicates per parameter value      (default: 15)
    --steps   N   simulation length in days           (default: 250)
    --agents  N   number of agents                     (default: 4000)
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from experiments.orchestrator import SweepSpec, run_sweep, load_results, get_param
from experiments.metrics import epidemic_metrics, care_seeking_metrics, wellbeing_metrics
from config import SVEIRCONFIG

SPEC_NAME = "care_seeking_oat"

# Ghana DHS 2014 childhood-diarrhea care-seeking rate (n=621 weighted cases),
# and the pooled sub-Saharan Africa estimate across 35 DHS surveys since
# 2010 (95% CI 55.39-62.04%) - see the care_seeking_empirical_target memory
# for the source. Shown as a reference only; not a fitted target.
DHS_GHANA_RATE = 0.692
DHS_SSA_RATE_CI = (0.5539, 0.6204)

# path -> (grid values, is_multiplier). Multiplier grids are scaled against
# the live config.py baseline at combo-build time (see build_oat_combos), so
# they can't silently stop bracketing the baseline the way a hardcoded
# absolute grid can after a later recalibration.
OAT_PARAMS: dict[str, tuple[list[float], bool]] = {
    "steering_parameters.daily_income_rate":          ([0.5, 0.75, 1.0, 1.5, 2.0],   True),
    "steering_parameters.cost_of_care":                ([0.0, 0.5, 1.0, 2.0, 3.0],   True),
    "steering_parameters.treatment_success_prob":      ([0.5, 0.65, 0.80, 0.90, 0.95], False),
    "steering_parameters.natural_worsening_prob":      ([0.1, 0.2, 0.35, 0.5, 0.65],  False),
    "steering_parameters.parent_stress_health_impact": ([0.1, 0.2, 0.3, 0.4, 0.5],    False),
    "steering_parameters.untreated_severity_penalty":  ([0.05, 0.1, 0.2, 0.3, 0.4],   False),
}

# Coarse pilot: just the low/baseline/high points of each grid, to exercise
# the pipeline before committing to the full 5-point run.
PILOT_INDICES = [0, 2, 4]

PARAM_LABELS = {
    "steering_parameters.daily_income_rate": "Daily income rate",
    "steering_parameters.cost_of_care": "Cost of care",
    "steering_parameters.treatment_success_prob": "Treatment success prob",
    "steering_parameters.natural_worsening_prob": "Natural worsening prob",
    "steering_parameters.parent_stress_health_impact": "Parent stress health impact",
    "steering_parameters.untreated_severity_penalty": "Untreated severity penalty",
}

METRIC_COLUMNS = [
    "episode_care_seeking_rate",
    "could_not_afford_rate",
    "rota_cumulative_u5_days",
    "campy_cumulative_u5_days",
    "mean_parent_wealth",
]

METRIC_LABELS = {
    "episode_care_seeking_rate": "Episode care-seeking rate",
    "could_not_afford_rate": "Could-not-afford rate",
    "rota_cumulative_u5_days": "Rota cumulative u5 illness-days",
    "campy_cumulative_u5_days": "Campy cumulative u5 illness-days",
    "mean_parent_wealth": "Mean parent-household wealth",
}


def metrics_fn(model) -> dict:
    """Composed metric set. Must stay a top-level function (not a lambda/
    closure) so it can be pickled to worker processes."""
    out = {}
    out.update(epidemic_metrics(model))
    out.update(care_seeking_metrics(model))
    out.update(wellbeing_metrics(model))
    return out


def build_oat_combos(pilot: bool = False) -> list[dict]:
    """One combo per (parameter, grid value) pair - baseline everywhere else.
    Every grid already includes its own baseline value as one of its points
    (by construction, see OAT_PARAMS), so no separate anchor combo is needed.
    """
    combos = []
    for path, (grid, is_multiplier) in OAT_PARAMS.items():
        values = [grid[i] for i in PILOT_INDICES] if pilot else grid
        baseline = float(get_param(SVEIRCONFIG, path))
        for v in values:
            actual = baseline * v if is_multiplier else v
            combos.append({path: actual})
    return combos


def build_spec(grid_id: str, reps: int, steps: int, agents: int, pilot: bool = False,
               n_cores: int | None = None) -> SweepSpec:
    return SweepSpec(
        name=f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME,
        grid_id=grid_id,
        params=[],  # non-factorial: explicit OAT combos passed to run_sweep
        metrics_fn=metrics_fn,
        reps=reps,
        steps=steps,
        agents=agents,
        record_timeseries=False,
        n_cores=n_cores,
    )


def plot_results(pilot: bool = False):
    spec_name = f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME
    df = load_results(spec_name)

    if df.empty:
        print(f"\nNo successful runs found in '{spec_name}' results - nothing to plot.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.0)
    param_paths = list(OAT_PARAMS.keys())
    n_params = len(param_paths)
    n_metrics = len(METRIC_COLUMNS)

    fig, axes = plt.subplots(
        n_params, n_metrics,
        figsize=(4.2 * n_metrics, 3.0 * n_params),
        squeeze=False,
    )
    fig.suptitle("Care-Seeking & Household-Economics OAT Sensitivity", fontsize=15, y=1.01)

    for row_idx, path in enumerate(param_paths):
        sub = df[df[path].notna()].copy()
        if sub.empty:
            for col_idx in range(n_metrics):
                axes[row_idx, col_idx].set_visible(False)
            continue

        baseline = float(get_param(SVEIRCONFIG, path))
        param_label = PARAM_LABELS.get(path, path)
        agg = sub.groupby(path)[METRIC_COLUMNS].agg(["mean", "std"])

        for col_idx, metric in enumerate(METRIC_COLUMNS):
            ax = axes[row_idx, col_idx]
            x = agg.index.values
            y = agg[(metric, "mean")].values
            yerr = agg[(metric, "std")].values

            if metric == "episode_care_seeking_rate":
                ax.axhspan(DHS_SSA_RATE_CI[0], DHS_SSA_RATE_CI[1], color="green", alpha=0.12)
                ax.axhline(DHS_GHANA_RATE, color="green", linestyle=":", linewidth=1.2)

            ax.axvline(baseline, color="grey", linestyle="--", linewidth=1.2)
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.8, color="#2196F3", zorder=3)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.20, color="#2196F3")

            if row_idx == 0:
                ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(param_label, fontsize=8)
            ax.set_xlabel("Parameter value", fontsize=8)
            ax.tick_params(labelsize=7)

    dhs_patch = mpatches.Patch(color="green", alpha=0.3, label="Pooled SSA DHS 95% CI")
    dhs_line = plt.Line2D([0], [0], color="green", linestyle=":", label="Ghana DHS (69.2%)")
    baseline_line = plt.Line2D([0], [0], color="grey", linestyle="--", label="Baseline value")
    fig.legend(handles=[dhs_patch, dhs_line, baseline_line],
               loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "care_seeking_oat.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Care-Seeking & Household-Economics OAT Sensitivity")
    parser.add_argument("-g", "--grid-id", required=False)
    parser.add_argument("-r", "--reps", type=int, default=None,
                        help="Default: 15 (full sweep) or 2 (--pilot)")
    parser.add_argument("-s", "--steps", type=int, default=None,
                        help="Default: 250 (full sweep) or 60 (--pilot)")
    parser.add_argument("-n", "--agents", type=int, default=None,
                        help="Default: 4000 (full sweep) or 800 (--pilot)")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="Fast end-to-end smoke test: 3 of the 5 grid "
                             "points per parameter, few reps, short run, "
                             "fewer agents. Use this first.")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.plot_only:
        plot_results(pilot=args.pilot)
        return

    if not args.grid_id:
        parser.error("--grid-id is required unless --plot-only is set.")

    reps = args.reps if args.reps is not None else (2 if args.pilot else 15)
    steps = args.steps if args.steps is not None else (60 if args.pilot else 250)
    agents = args.agents if args.agents is not None else (800 if args.pilot else 4000)

    if args.pilot:
        print("*** PILOT MODE: 3-point grids, reduced reps/steps/agents. "
              "For a timing/sanity check only - do not draw conclusions "
              "from these results. ***\n")

    spec = build_spec(args.grid_id, reps, steps, agents, pilot=args.pilot, n_cores=args.workers)
    combos = build_oat_combos(pilot=args.pilot)
    run_sweep(spec, combos=combos)
    plot_results(pilot=args.pilot)


if __name__ == "__main__":
    main()

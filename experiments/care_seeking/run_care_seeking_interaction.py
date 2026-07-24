# experiments/care_seeking/run_care_seeking_interaction.py
"""
Care-Seeking Interaction Sweep: Cost of Care x Daily Income Rate
===================================================================
Affordability is jointly determined by income and the cost of care, not by
either alone - an OAT sweep (see run_care_seeking_oat.py) holds one fixed
while varying the other, and can miss exactly the interaction that matters
most for a policy question like "subsidize care, or raise incomes?". This
sweep maps the full 2D plane instead, mirroring
experiments/vaccination/run_vaccination_sweep.py's structure.

Metrics recorded per run: same composed set as run_care_seeking_oat.py
(epidemic_metrics + care_seeking_metrics + wellbeing_metrics).

Usage
-----
    python -m experiments.care_seeking.run_care_seeking_interaction --grid-id <GRID_ID>
    python -m experiments.care_seeking.run_care_seeking_interaction --plot-only

Optional flags:
    --reps    N   replicates per (cost, income) combination (default: 20)
    --steps   N   simulation length in days                  (default: 250)
    --agents  N   number of agents                            (default: 4000)
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.orchestrator import SweepParam, SweepSpec, run_sweep, load_results
from experiments.metrics import epidemic_metrics, care_seeking_metrics, wellbeing_metrics

SPEC_NAME = "care_seeking_interaction"

# Ghana DHS 2014 childhood-diarrhea care-seeking rate, and the pooled
# sub-Saharan Africa estimate (95% CI) - see the care_seeking_empirical_target
# memory. Reference only, not a fitted target.
DHS_GHANA_RATE = 0.692
DHS_SSA_RATE_CI = (0.5539, 0.6204)

COST_OF_CARE_VALUES = np.round(np.linspace(0.0, 0.075, 7), 5)
DAILY_INCOME_RATE_VALUES = np.round(np.linspace(0.015, 0.06, 7), 5)

PILOT_COST_VALUES = np.round(np.linspace(0.0, 0.075, 3), 5)
PILOT_INCOME_VALUES = np.round(np.linspace(0.015, 0.06, 3), 5)

# Zoomed-in region: the full grid showed a sharp cliff between cost=0 and
# cost=0.0125 at low income (a hard affordability-gate/wealth-floor effect,
# not a smooth gradient) and a similarly sharp wealth transition between
# income=0.0225 and 0.03 (the adult_fraction >= cost/income breakeven ratio
# crossing 1.0). This grid resolves both at full density instead of
# bracketing them with only 1-2 coarse points each.
ZOOM_COST_OF_CARE_VALUES = np.round(np.linspace(0.0, 0.0125, 7), 6)
ZOOM_DAILY_INCOME_RATE_VALUES = np.round(np.linspace(0.015, 0.03, 7), 5)

PILOT_ZOOM_COST_VALUES = np.round(np.linspace(0.0, 0.0125, 3), 6)
PILOT_ZOOM_INCOME_VALUES = np.round(np.linspace(0.015, 0.03, 3), 5)


def metrics_fn(model) -> dict:
    """Composed metric set. Must stay a top-level function (not a lambda/
    closure) so it can be pickled to worker processes."""
    out = {}
    out.update(epidemic_metrics(model))
    out.update(care_seeking_metrics(model))
    out.update(wellbeing_metrics(model))
    return out


def _spec_name(pilot: bool, zoom: bool) -> str:
    suffix = ("_zoom" if zoom else "") + ("_pilot" if pilot else "")
    return f"{SPEC_NAME}{suffix}"


def build_spec(grid_id: str, reps: int, steps: int, agents: int, pilot: bool = False,
               zoom: bool = False, n_cores: int | None = None) -> SweepSpec:
    if zoom:
        cost_values, income_values = (
            (PILOT_ZOOM_COST_VALUES, PILOT_ZOOM_INCOME_VALUES) if pilot
            else (ZOOM_COST_OF_CARE_VALUES, ZOOM_DAILY_INCOME_RATE_VALUES)
        )
    else:
        cost_values, income_values = (
            (PILOT_COST_VALUES, PILOT_INCOME_VALUES) if pilot
            else (COST_OF_CARE_VALUES, DAILY_INCOME_RATE_VALUES)
        )
    return SweepSpec(
        name=_spec_name(pilot, zoom),
        grid_id=grid_id,
        params=[
            SweepParam("steering_parameters.cost_of_care", cost_values.tolist(),
                       "Cost of care"),
            SweepParam("steering_parameters.daily_income_rate", income_values.tolist(),
                       "Daily income rate"),
        ],
        metrics_fn=metrics_fn,
        reps=reps,
        steps=steps,
        agents=agents,
        record_timeseries=False,
        n_cores=n_cores,
    )


def plot_results(pilot: bool = False, zoom: bool = False):
    spec_name = _spec_name(pilot, zoom)
    df = load_results(spec_name)

    if df.empty or "episode_care_seeking_rate" not in df.columns:
        print(f"\nNo successful runs found in '{spec_name}' results - nothing to plot.")
        print("Check the per-run tracebacks printed during the sweep above.")
        return

    cost_col = "steering_parameters.cost_of_care"
    income_col = "steering_parameters.daily_income_rate"

    df = df.copy()
    df["illness_burden_u5_days"] = df["rota_cumulative_u5_days"] + df["campy_cumulative_u5_days"]

    def _pivot(metric):
        return df.pivot_table(
            index=cost_col, columns=income_col, values=metric, aggfunc="mean",
        ).sort_index(ascending=False)

    sns.set_theme(style="white", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    title_prefix = "Zoomed-In " if zoom else ""

    sns.heatmap(_pivot("could_not_afford_rate"), annot=True, fmt=".2f", cmap="Reds",
                ax=axes[0, 0], cbar_kws={"label": "Could-not-afford rate"})
    axes[0, 0].set_title(f"{title_prefix}Could-Not-Afford Rate")
    axes[0, 0].set_xlabel("Daily income rate")
    axes[0, 0].set_ylabel("Cost of care")

    sns.heatmap(_pivot("episode_care_seeking_rate"), annot=True, fmt=".0%", cmap="Blues",
                ax=axes[0, 1], cbar_kws={"label": "Episode care-seeking rate"})
    axes[0, 1].set_title(
        f"{title_prefix}Episode Care-Seeking Rate\n(Ghana DHS: {DHS_GHANA_RATE:.0%}, "
        f"pooled SSA: {DHS_SSA_RATE_CI[0]:.0%}-{DHS_SSA_RATE_CI[1]:.0%})"
    )
    axes[0, 1].set_xlabel("Daily income rate")
    axes[0, 1].set_ylabel("")

    sns.heatmap(_pivot("illness_burden_u5_days"), annot=True, fmt=".0f", cmap="Oranges",
                ax=axes[1, 0], cbar_kws={"label": "Illness-days (u5, both pathogens)"})
    axes[1, 0].set_title(f"{title_prefix}Combined Illness Burden (Rota + Campy)")
    axes[1, 0].set_xlabel("Daily income rate")
    axes[1, 0].set_ylabel("Cost of care")

    sns.heatmap(_pivot("mean_parent_wealth"), annot=True, fmt=".2f", cmap="Greens",
                ax=axes[1, 1], cbar_kws={"label": "Mean parent-household wealth"})
    axes[1, 1].set_title(f"{title_prefix}Household Wealth (Parent-Headed Households)")
    axes[1, 1].set_xlabel("Daily income rate")
    axes[1, 1].set_ylabel("")

    plt.tight_layout()
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "care_seeking_interaction.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Care-Seeking Interaction Sweep")
    parser.add_argument("-g", "--grid-id", required=False)
    parser.add_argument("-r", "--reps", type=int, default=None,
                        help="Default: 20 (full sweep) or 2 (--pilot)")
    parser.add_argument("-s", "--steps", type=int, default=None,
                        help="Default: 250 (full sweep) or 60 (--pilot)")
    parser.add_argument("-n", "--agents", type=int, default=None,
                        help="Default: 4000 (full sweep) or 800 (--pilot)")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="Fast end-to-end smoke test: coarse 3x3 grid, "
                             "few reps, short run, fewer agents. Use this "
                             "first to sanity-check the pipeline and timing "
                             "before committing to the full sweep.")
    parser.add_argument("--zoom", action="store_true",
                        help="Zoom into cost_of_care [0, 0.0125] x "
                             "daily_income_rate [0.015, 0.03] - the region "
                             "where the full sweep showed sharp cliffs - "
                             "at the same 7x7 resolution as the full sweep.")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.plot_only:
        plot_results(pilot=args.pilot, zoom=args.zoom)
        return

    if not args.grid_id:
        parser.error("--grid-id is required unless --plot-only is set.")

    reps = args.reps if args.reps is not None else (2 if args.pilot else 20)
    steps = args.steps if args.steps is not None else (60 if args.pilot else 250)
    agents = args.agents if args.agents is not None else (800 if args.pilot else 4000)

    if args.pilot:
        print("*** PILOT MODE: coarse 3x3 grid, reduced reps/steps/agents. "
              "For a timing/sanity check only - do not draw conclusions "
              "from these results. ***\n")

    spec = build_spec(args.grid_id, reps, steps, agents, pilot=args.pilot, zoom=args.zoom,
                       n_cores=args.workers)
    run_sweep(spec)
    plot_results(pilot=args.pilot, zoom=args.zoom)


if __name__ == "__main__":
    main()

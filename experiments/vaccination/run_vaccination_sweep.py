# experiments/vaccination/run_vaccination_sweep.py
"""
Vaccination & Immunology Sweep
===============================
Maps the Rotavirus herd-immunity surface over vaccination_rate x
vaccine_efficacy, looking for the extinction / persistence phase boundary.

Campylobacter is left running in the background (unmanipulated) as a
control - it shares no vaccination mechanism, so its outcomes should be
flat across this sweep; if they're not, something else is confounding
the comparison (e.g. shared household/economic dynamics).

Metrics recorded per run
-------------------------
  rota_peak_u5_prevalence, rota_peak_day, rota_cumulative_u5_days,
  rota_extinct (peak < 1%), rota_attack_rate, rota_attack_rate_u5
  campy_* (same, as a control)
  conditional_care_rate, could_not_afford_rate, decisions_faced
  mean_final_health, mean_household_wealth, mean_parent_wealth

Time series (u5 prevalence per pathogen per day) are also recorded, so you
can compute early-warning-signal indicators (rolling variance / lag-1
autocorrelation) near the extinction boundary - see experiments/metrics.py.

Usage
-----
    python -m experiments.vaccination.run_vaccination_sweep --grid-id <GRID_ID>
    python -m experiments.vaccination.run_vaccination_sweep --plot-only

Optional flags:
    --reps    N   replicates per (rate, efficacy) combination (default: 20)
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

SPEC_NAME = "vaccination"

# Daily vaccination probability and vaccine efficacy, each 9 values -> 81
# combinations. Widened relative to the old exp3 (5x5) so the extinction
# boundary can actually be localised rather than just bracketed.
VACC_RATES = np.round(np.linspace(0.0, 0.12, 9), 5)
VACC_EFFICACIES = np.round(np.linspace(0.10, 0.95, 9), 4)

# Coarse grid used by --pilot: just enough points to exercise both ends of
# the sweep (no vaccination, low, high) and confirm the orchestrator/plotting
# pipeline works end-to-end before committing to the full 81-combo sweep.
PILOT_VACC_RATES = np.round(np.linspace(0.0, 0.12, 3), 5)
PILOT_VACC_EFFICACIES = np.round(np.linspace(0.10, 0.95, 3), 4)


def metrics_fn(model) -> dict:
    """Composed metric set for this experiment. Must stay a top-level
    function (not a lambda/closure) so it can be pickled to worker processes.
    """
    out = {}
    out.update(epidemic_metrics(model))          # all pathogens (rota + campy control)
    out.update(care_seeking_metrics(model))
    out.update(wellbeing_metrics(model))
    return out


def build_spec(grid_id: str, reps: int, steps: int, agents: int, pilot: bool = False,
               n_cores: int | None = None) -> SweepSpec:
    rates, efficacies = (PILOT_VACC_RATES, PILOT_VACC_EFFICACIES) if pilot else (VACC_RATES, VACC_EFFICACIES)
    return SweepSpec(
        name=f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME,
        grid_id=grid_id,
        params=[
            SweepParam("pathogens[rota].vaccination_rate", rates.tolist(),
                       "Vaccination rate (daily)"),
            SweepParam("pathogens[rota].vaccine_efficacy", efficacies.tolist(),
                       "Vaccine efficacy"),
        ],
        metrics_fn=metrics_fn,
        reps=reps,
        steps=steps,
        agents=agents,
        record_timeseries=True,
        n_cores=n_cores,
    )


def plot_results(pilot: bool = False):
    spec_name = f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME
    df = load_results(spec_name)

    if df.empty or "rota_peak_u5_prevalence" not in df.columns:
        print(f"\nNo successful runs found in '{spec_name}' results - nothing to plot.")
        print("Check the per-run tracebacks printed during the sweep above.")
        return

    rate_col = "pathogens[rota].vaccination_rate"
    eff_col = "pathogens[rota].vaccine_efficacy"

    pivot_peak = df.pivot_table(
        index=rate_col, columns=eff_col, values="rota_peak_u5_prevalence", aggfunc="mean",
    ).sort_index(ascending=False)

    pivot_extinct = df.pivot_table(
        index=rate_col, columns=eff_col, values="rota_extinct", aggfunc="mean",
    ).sort_index(ascending=False)  # mean of a bool column -> extinction probability

    sns.set_theme(style="white", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    sns.heatmap(pivot_peak, annot=True, fmt=".2f", cmap="Reds", ax=axes[0],
                cbar_kws={"label": "Mean peak u5 prevalence"})
    axes[0].set_title("Peak Under-5 Prevalence (Rotavirus)")
    axes[0].set_xlabel("Vaccine efficacy")
    axes[0].set_ylabel("Vaccination rate (daily)")

    sns.heatmap(pivot_extinct, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                cbar_kws={"label": "P(extinction)"}, vmin=0, vmax=1)
    axes[1].set_title("Extinction Probability\n(peak u5 prevalence < 1%)")
    axes[1].set_xlabel("Vaccine efficacy")
    axes[1].set_ylabel("")

    plt.tight_layout()
    out_dir = os.path.join("experiments", "outputs", spec_name)
    out_path = os.path.join(out_dir, "vaccination_phase_map.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Vaccination & Immunology Sweep")
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
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes. Default: orchestrator's own default "
                             "(min(6, cpu_count())). Override to use more cores.")
    args = parser.parse_args()

    if args.plot_only:
        plot_results(pilot=args.pilot)
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

    spec = build_spec(args.grid_id, reps, steps, agents, pilot=args.pilot, n_cores=args.workers)
    run_sweep(spec)
    plot_results(pilot=args.pilot)


if __name__ == "__main__":
    main()
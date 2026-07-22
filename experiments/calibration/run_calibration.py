# experiments/calibration/run_calibration.py
"""
Baseline Transmission Calibration
=================================
Searches the core rotavirus + campylobacter transmission parameters so that the
model's *baseline* (no-vaccination) epidemiology lands inside the empirical
target ranges defined in sensitivity.py (Ghana / GEMS / WHO literature).

Why this exists
---------------
The vaccination sweep hunts a herd-immunity extinction boundary, but that
boundary is only meaningful if the un-vaccinated baseline is plausible. As of
the vaccination pilot it was not: rota peak u5 prevalence ~13% (target 2-10%)
and campy peak ~50% with 100% of under-5s infected (target 2-15%). Both
pathogens transmit too hot, so the phase boundary sits at the wrong coverage.
Calibrate here first, then run the full vaccination sweep on the tuned baseline.

Method
------
Latin-Hypercube sample the parameter space (scipy.stats.qmc), run `reps`
stochastic replicates per point through the shared orchestrator with
vaccination turned OFF, score each point against the empirical TARGETS in
experiments/calibration/targets.py via experiments.metrics.calibration_loss,
and rank. The current config defaults are
included as an explicit anchor point (combo 0) so the ranking shows how far the
un-calibrated baseline sits from the empirical bands.

Outputs (experiments/outputs/calibration[_pilot]/)
--------------------------------------------------
  results.parquet            one row per (combo x rep) with calibration_metrics
  calibration_ranked.csv     combos ranked by targets-met then loss, with per-target in/out flags
  best_params.json           the best-fit parameter set, ready to adopt
  calibration_param_scatter.png   loss vs each swept parameter
  calibration_best_fit.png        best set's metrics against their target bands

Usage
-----
    python -m experiments.calibration.run_calibration --grid-id <GRID_ID> --pilot
    python -m experiments.calibration.run_calibration --grid-id <GRID_ID> \
        --samples 200 --reps 8 --steps 300
    python -m experiments.calibration.run_calibration --plot-only
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

from config import SVEIRCONFIG
from experiments.orchestrator import SweepSpec, run_sweep, load_results, get_param
from experiments.metrics import calibration_metrics, calibration_loss
from experiments.calibration.targets import TARGETS

SPEC_NAME = "calibration"
DEFAULT_GRID_ID = "7d9ce7c720a6"

# ---------------------------------------------------------------------------
# Calibration parameter space (LHS bounds)
# ---------------------------------------------------------------------------
# Six transmission levers - the ones sensitivity.py already treats as the
# influential knobs for the target metrics. Kept small so the LHS design stays
# tractable at a few hundred points.
#
# NOTE on rota infection_prob_mean: the daily H2H prob is drawn as
# max(0.001, Normal(infection_prob_mean, std)) (abm/pathogens/rotavirus.py:44),
# so the config default 0.0005 sits BELOW the 0.001 floor and never actually
# binds. The lower bound starts at the floor so this lever is meaningful.
#
# NOTE on campy bounds (round 3, post round-1/round-2 analysis):
# round 1 (interaction_rate 0.05-0.35) showed campy_episodes_per_child_year
# correlates 0.95 with interaction_rate alone, ~9/year even at the former
# floor (0.05) against a 1.0-3.0 target. The zoonotic dose-response (Beta-
# Poisson, campylobacter.py:218) has no hidden floor and scales to 0 as this
# rate -> 0, so round 2 narrowed it to (0.0, 0.03).
# Round 2 got 3/5 targets (both rota + campy_zoonotic_fraction), but revealed
# a 3-way tension: the best points (interaction~0.024-0.026, fecal_oral~
# 0.013-0.015, food_borne~0.005-0.008) still ran ~2x over on campy episodes/
# peak, while a near-zero-interaction point hit the episode target but
# collapsed zoonotic_fraction to 0.07 (starving the zoonotic route breaks the
# route-mix target instead). Fix: scale all three campy bounds down together
# (~2x), preserving the successful ratio rather than just lowering one lever.
# NOTE on round 4 (post code-review epidemic-core fix): the same-day
# E->I->R bug (pathogen.py - _infectious_to_recovered used to catch agents
# _exposed_to_infectious had JUST converted this same call, before any
# transmission phase ran) meant ~recovery_rate fraction of new infections
# never actually transmitted. Fixed by reordering the two calls, which
# makes transmission somewhat MORE effective per the same nominal
# recovery_rate than before - the old calibrated params no longer apply.
# Round 4 widened rota recovery_rate to (0.15, 0.45) and added campy
# recovery_rate as (0.1, 0.3) purely to give the search more room, WITHOUT
# checking either against illness-duration literature the way the other
# transmission parameters were. Round 4's 5/5 fit exploited exactly that:
# rota recovery_rate=0.4035 implies a ~2.5 day infectious duration, faster
# than the typical 3-7 day rotavirus illness range; campy's 0.1221 implies
# ~8.2 days, slightly past the typical 5-7 day range.
#
# NOTE on round 5 (literature-constrained recovery rates): tested directly
# (holding all other round-4 params fixed) whether rota's targets survive a
# realistic recovery_rate - they don't: at 0.25 (4-day duration) episodes/yr
# jumps to 3.44 and peak to 0.15, both well out of range; only 0.33 (3-day,
# the fast edge of 3-7d) still clears both targets (barely). Campy is far
# more forgiving: episodes/peak/zoonotic_fraction all stayed comfortably
# in-range across the entire tested 0.14-0.18 (5-7 day) range, since campy's
# other route parameters already have headroom. Bounds narrowed to the
# literature-plausible ranges for both, letting the search retune the other
# transmission parameters (already has room: rota water_to_human_infection_
# prob isn't floor-limited) to compensate rather than exploiting biologically
# implausible recovery speeds.
CALIB_BOUNDS = {
    # --- Rotavirus transmission ---
    "pathogens[rota].infection_prob_mean":                (0.001, 0.010),
    "pathogens[rota].recovery_rate":                      (0.14,  0.33),
    "steering_parameters.water_to_human_infection_prob":  (0.0,   0.02),
    # --- Campylobacter transmission (all three routes) ---
    "pathogens[campy].human_animal_interaction_rate":     (0.0,   0.018),
    "pathogens[campy].fecal_oral_prob":                   (0.0,   0.02),
    "pathogens[campy].food_borne_prob":                   (0.0,   0.01),
    "pathogens[campy].recovery_rate":                     (0.14,  0.2),
}

# Calibrate the BASELINE: vaccination off, matching the pre/peri-vaccine
# empirical targets. (campy has no vaccination mechanism; only rota needs it.)
BASE_OVERRIDES = {"pathogens[rota].vaccination_rate": 0.0}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

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
        metrics_fn=calibration_metrics,
        reps=reps,
        steps=steps,
        agents=agents,
        base_overrides=BASE_OVERRIDES,
        record_timeseries=False,
        n_cores=n_cores,
    )


# ---------------------------------------------------------------------------
# Scoring + reporting
# ---------------------------------------------------------------------------

def score_and_report(pilot: bool = False):
    spec_name = f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME
    df = load_results(spec_name)

    if df.empty or "rota_peak_u5_prevalence" not in df.columns:
        print(f"\nNo successful runs found in '{spec_name}' results - nothing to score.")
        print("Check the per-run tracebacks printed during the sweep above.")
        return

    ranked = calibration_loss(df, TARGETS)
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)

    ranked.to_csv(os.path.join(out_dir, "calibration_ranked.csv"), index=False)

    param_cols = list(CALIB_BOUNDS.keys())
    best = ranked.iloc[0]
    best_params = {p: float(best[p]) for p in param_cols}
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({
            "params": best_params,
            "base_overrides": BASE_OVERRIDES,
            "loss": float(best["loss"]),
            "n_targets_met": int(best["n_targets_met"]),
            "n_targets": len(TARGETS),
            "targets": {k: list(v) for k, v in TARGETS.items()},
        }, f, indent=2)

    # Console summary
    metric_cols = [m for m in TARGETS if m in ranked.columns]
    n_targets = len(metric_cols)
    print(f"\n--- Calibration ranking ({len(ranked)} combos, {n_targets} targets) ---")
    print(f"  Best loss = {best['loss']:.3f}  |  targets met = {int(best['n_targets_met'])}/{n_targets}")
    print("  Best-fit parameters:")
    for p in param_cols:
        print(f"    {p:<52} = {best[p]:.5g}")
    print("  Best-fit metrics vs target bands:")
    for m in metric_cols:
        lo, hi = TARGETS[m]
        flag = "OK " if best[f"{m}_in_range"] else "OUT"
        print(f"    [{flag}] {m:<34} = {best[m]:.4g}   target [{lo}, {hi}]")
    print(f"\n  Ranked table -> {os.path.join(out_dir, 'calibration_ranked.csv')}")
    print(f"  Best params  -> {os.path.join(out_dir, 'best_params.json')}")

    _plot_param_scatter(ranked, param_cols, out_dir)
    _plot_best_fit(best, metric_cols, out_dir)


def _plot_param_scatter(ranked, param_cols, out_dir):
    """Loss vs each swept parameter - shows which levers drive the fit."""
    n = len(param_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    for i, p in enumerate(param_cols):
        ax = axes[i // ncols][i % ncols]
        ax.scatter(ranked[p], ranked["loss"], c=ranked["loss"], cmap="viridis_r", s=28)
        lo, hi = CALIB_BOUNDS[p]
        ax.axvspan(lo, hi, color="grey", alpha=0.05)
        ax.set_xlabel(p.split("]")[-1].lstrip(".") or p, fontsize=8)
        ax.set_ylabel("loss", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Calibration loss vs swept parameter", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_param_scatter.png")
    plt.savefig(path, dpi=170, bbox_inches="tight")
    print(f"  Figure saved -> {path}")
    plt.close(fig)


def _plot_best_fit(best, metric_cols, out_dir):
    """Best set's metric values against their empirical target bands."""
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.4), squeeze=False)
    for i, m in enumerate(metric_cols):
        ax = axes[0][i]
        lo, hi = TARGETS[m]
        val = best[m]
        ax.axhspan(lo, hi, color="green", alpha=0.18, label="target")
        ax.bar([0], [val], width=0.5,
               color="#2196F3" if best[f"{m}_in_range"] else "#E53935")
        ax.set_xticks([])
        ax.set_title(m.replace("_", "\n"), fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, max(hi * 1.3, val * 1.15))
    fig.suptitle(f"Best-fit baseline vs empirical targets  (loss={best['loss']:.3f})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_best_fit.png")
    plt.savefig(path, dpi=170, bbox_inches="tight")
    print(f"  Figure saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline transmission calibration for SPRINGS-ABM.")
    parser.add_argument("-g", "--grid-id", default=DEFAULT_GRID_ID,
                        help=f"Grid ID (default: {DEFAULT_GRID_ID})")
    parser.add_argument("--samples", type=int, default=None,
                        help="LHS sample points. Default: 200 (full) or 12 (--pilot)")
    parser.add_argument("-r", "--reps", type=int, default=None,
                        help="Replicates per point. Default: 8 (full) or 3 (--pilot)")
    parser.add_argument("-s", "--steps", type=int, default=None,
                        help="Simulation days. Default: 300 (full) or 60 (--pilot)")
    parser.add_argument("-n", "--agents", type=int, default=None,
                        help="Number of agents. Default: 4000 (full) or 800 (--pilot)")
    parser.add_argument("--seed", type=int, default=12345, help="LHS sampler seed (default: 12345)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes. Default: orchestrator's own default "
                             "(min(6, cpu_count())). Override to use more cores.")
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

    samples = args.samples if args.samples is not None else (12 if args.pilot else 200)
    reps = args.reps if args.reps is not None else (3 if args.pilot else 8)
    steps = args.steps if args.steps is not None else (60 if args.pilot else 300)
    agents = args.agents if args.agents is not None else (800 if args.pilot else 4000)

    if args.pilot:
        print("*** PILOT MODE: few LHS points, reduced reps/steps/agents. "
              "For a timing/sanity check only - do not adopt these parameters. ***\n")

    combos = [baseline_combo()] + sample_lhs(samples, args.seed)
    print(f"Design: 1 baseline anchor + {samples} LHS points = {len(combos)} combos, "
          f"{reps} reps each ({len(combos) * reps} runs), vaccination OFF.")

    spec = build_spec(args.grid_id, reps, steps, agents, pilot=args.pilot, n_cores=args.workers)
    run_sweep(spec, combos=combos)
    score_and_report(pilot=args.pilot)


if __name__ == "__main__":
    main()

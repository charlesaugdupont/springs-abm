"""
Cryptosporidium Transmission Calibration
========================================
Searches the four Cryptosporidium route parameters so that the model's
*baseline* (no-vaccination) cryptosporidiosis epidemiology lands inside the
empirical target ranges in experiments/calibration/targets.py::CRYPTO_TARGETS.

Why this is a SEPARATE script from run_calibration.py
-----------------------------------------------------
run_calibration.py fits the rotavirus + campylobacter transmission parameters
against TARGETS. Those fits are settled (round 7) and this work is explicitly
NOT reopening them: the rotavirus/campylobacter parameters stay fixed, and the
drift that adding a third pathogen induces in them is MEASURED separately
(see --measure-drift below) rather than absorbed by refitting. Scoring crypto's
parameters against a combined target set would also let the search trade a
rotavirus miss against a crypto hit, which is exactly what we do not want.

What is searched, and what is not
---------------------------------
Searched (4 free knobs, one per route):
    household_transmission_prob   person-to-person within the household
    community_infection_prob      co-location outside the household
    water_dose_scale              oocyst reservoir -> ingested dose
    zoonotic_interaction_rate     ruminant density -> ingested dose

Deliberately NOT searched, because they are literature-grounded rather than
free (same principle that excludes recovery_rate from run_calibration.py - see
docs/model_description.md section 10):
    exposure_period = 7                 incubation ~7 days (range 2-10)
    recovery_rate = 0.0476              ~21-day shedding (Korpe 2022: 16-19
                                        day means; Ethiopian series: 31-day
                                        median from diarrhoea onset)
    symptomatic_fraction = 0.33         ~2/3 of childhood infections subclinical
                                        (Korpe 2016; Kattula 2016)
    duration_scale = 1.6                ~10-day mean episode
    immunity_decay_per_infection = 0.29 only 69% protection after 4 infections
                                        (Kattula 2016)
    dose_response_r = 0.018             exponential model, ID50 ~ 39 oocysts
                                        (DuPont 1995; Okhuysen 1999; Chappell
                                        2006)
    oocyst_decay_rate = 0.15            T90 ~ 15 days at tropical temperature
                                        (King 2005; Li 2009)

A note on identifiability: four route knobs against five targets is not an
over-determined fit, and two of the five targets (infections/child-year and
symptomatic episodes/child-year) are tied to each other through the fixed
symptomatic_fraction, so they are not independent constraints. The two
route-FRACTION targets are what actually pin the mix down. Expect a ridge
rather than a unique optimum, and read the ranked table, not just row 0.

Outputs (experiments/outputs/crypto_calibration[_pilot]/)
---------------------------------------------------------
  results.parquet                  one row per (combo x rep)
  crypto_calibration_ranked.csv    combos ranked by targets-met then loss
  best_params.json                 the best-fit parameter set, ready to adopt
  crypto_calibration_param_scatter.png
  crypto_calibration_best_fit.png

Usage
-----
    python -m experiments.calibration.run_crypto_calibration --grid-id <GRID_ID> --pilot
    python -m experiments.calibration.run_crypto_calibration --grid-id <GRID_ID> \
        --samples 120 --reps 6 --steps 300
    python -m experiments.calibration.run_crypto_calibration --plot-only
    python -m experiments.calibration.run_crypto_calibration --measure-drift --reps 8
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

from config import SVEIRCONFIG
from experiments.orchestrator import (
    PATHOGEN_SUBSET_KEY, SweepSpec, get_param, load_results, run_sweep,
)
from experiments.metrics import (
    calibration_loss, calibration_metrics, care_seeking_metrics, wellbeing_metrics,
)


def drift_metrics(model) -> dict:
    """Metrics for the drift comparison: epidemiology AND the coupling channel.

    calibration_metrics alone is not enough here. The mechanism by which a
    third pathogen can move the other two is the care-seeking/economics layer
    (see measure_drift), so the care-seeking counters and household wealth
    have to be measured too, or the comparison reports an effect without its
    explanation. Top-level and picklable, as SweepSpec.metrics_fn requires.
    """
    out = {}
    out.update(calibration_metrics(model))
    out.update(care_seeking_metrics(model))
    out.update(wellbeing_metrics(model))
    return out
# Superseded: the current targets (MAL-ED, burn-in-aware) are calibrated by
# experiments/calibration/history_matching.py. This script reproduces the
# FIRST-GENERATION Cryptosporidium calibration, so it scores against the legacy
# targets and runs the legacy model configuration.
from experiments.calibration.targets import (
    LEGACY_CRYPTO_TARGETS as CRYPTO_TARGETS,
    LEGACY_TARGETS as TARGETS,
)
from experiments.legacy import LEGACY_V1_OVERRIDES

SPEC_NAME = "crypto_calibration"
DEFAULT_GRID_ID = "7d9ce7c720a6"

# ---------------------------------------------------------------------------
# Calibration parameter space (LHS bounds)
# ---------------------------------------------------------------------------
# Bounds bracket a hand-probed region that already satisfies all five crypto
# targets on a single seed (household 0.012-0.016, community 0.00025-0.0003,
# water 0.0008-0.0010, zoonotic 0.014-0.016 at 4000 agents / 300 days), widened
# generously in both directions so the search is not boxed into that guess.
#
# Each lower bound is 0 where turning the route OFF is a meaningful hypothesis
# (matching how run_calibration.py's campylobacter bounds were set) -
# household is the exception: its floor is the RURAL household secondary
# attack rate from Korpe et al. 2018 (7.8% over 8 weeks => p ~ 0.004), because
# a measured household route is not a hypothesis this model should be free to
# discard, and its ceiling is the urban SAR (35.8% => p ~ 0.021).
#
# NOTE on scale dependence: the waterborne route's strength depends on agents
# per water point, and this grid has only 11 water sources, so the number of
# shedders at a source scales with `agents`. Bounds calibrated at one
# population size do NOT transfer to another - calibrate at the population you
# intend to run. These bounds are for 4000 agents.
CALIB_BOUNDS = {
    "pathogens[crypto].household_transmission_prob": (0.004, 0.030),
    "pathogens[crypto].community_infection_prob":    (0.0,   0.0008),
    "pathogens[crypto].water_dose_scale":            (0.0,   0.0030),
    "pathogens[crypto].zoonotic_interaction_rate":   (0.0,   0.040),
}

# Calibrate the BASELINE: vaccination off, matching the pre/peri-vaccine
# empirical targets. (Only rotavirus has a vaccine; crypto has none, and the
# override is kept for consistency with run_calibration.py so the two searches
# describe the same baseline world.)
BASE_OVERRIDES = {**LEGACY_V1_OVERRIDES, "pathogens[rota].vaccination_rate": 0.0}


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
               n_cores: int | None = None, name: str | None = None,
               base_overrides: dict | None = None) -> SweepSpec:
    return SweepSpec(
        name=name or (f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME),
        grid_id=grid_id,
        params=[],  # non-factorial: combos are passed explicitly to run_sweep
        metrics_fn=calibration_metrics,
        reps=reps,
        steps=steps,
        agents=agents,
        base_overrides=BASE_OVERRIDES if base_overrides is None else base_overrides,
        record_timeseries=False,
        n_cores=n_cores,
    )


# ---------------------------------------------------------------------------
# Scoring + reporting
# ---------------------------------------------------------------------------

def score_and_report(pilot: bool = False):
    spec_name = f"{SPEC_NAME}_pilot" if pilot else SPEC_NAME
    df = load_results(spec_name)

    if df.empty or "crypto_peak_u5_prevalence" not in df.columns:
        print(f"\nNo successful runs found in '{spec_name}' results - nothing to score.")
        print("Check the per-run tracebacks printed during the sweep above.")
        return

    ranked = calibration_loss(df, CRYPTO_TARGETS)
    out_dir = os.path.join("experiments", "outputs", spec_name)
    os.makedirs(out_dir, exist_ok=True)
    ranked.to_csv(os.path.join(out_dir, "crypto_calibration_ranked.csv"), index=False)

    param_cols = list(CALIB_BOUNDS.keys())
    best = ranked.iloc[0]
    best_params = {p: float(best[p]) for p in param_cols}
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({
            "params": best_params,
            "base_overrides": BASE_OVERRIDES,
            "loss": float(best["loss"]),
            "n_targets_met": int(best["n_targets_met"]),
            "n_targets": len(CRYPTO_TARGETS),
            "targets": {k: list(v) for k, v in CRYPTO_TARGETS.items()},
        }, f, indent=2)

    metric_cols = [m for m in CRYPTO_TARGETS if m in ranked.columns]
    n_targets = len(metric_cols)
    print(f"\n--- Crypto calibration ranking ({len(ranked)} combos, {n_targets} targets) ---")
    print(f"  Best loss = {best['loss']:.3f}  |  targets met = {int(best['n_targets_met'])}/{n_targets}")
    print("  Best-fit parameters:")
    for p in param_cols:
        print(f"    {p:<54} = {best[p]:.5g}")
    print("  Best-fit metrics vs target bands:")
    for m in metric_cols:
        lo, hi = CRYPTO_TARGETS[m]
        flag = "OK " if best[f"{m}_in_range"] else "OUT"
        print(f"    [{flag}] {m:<46} = {best[m]:.4g}   target [{lo}, {hi}]")

    # How many combos satisfy EVERY target - a ridge, not a point, is expected
    n_full = int((ranked["n_targets_met"] == n_targets).sum())
    print(f"\n  Combos meeting all {n_targets} targets: {n_full} / {len(ranked)}")
    if n_full > 1:
        print("  (a ridge, as anticipated - inspect the ranked table before adopting row 0)")

    print(f"\n  Ranked table -> {os.path.join(out_dir, 'crypto_calibration_ranked.csv')}")
    print(f"  Best params  -> {os.path.join(out_dir, 'best_params.json')}")

    _plot_param_scatter(ranked, param_cols, out_dir)
    _plot_best_fit(best, metric_cols, out_dir)


def _plot_param_scatter(ranked, param_cols, out_dir):
    """Loss vs each swept parameter - shows which levers drive the fit."""
    n = len(param_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)
    for i, p in enumerate(param_cols):
        ax = axes[i // ncols][i % ncols]
        ax.scatter(ranked[p], ranked["loss"], c=ranked["loss"], cmap="viridis_r", s=28)
        ax.set_xlabel(p.split("]")[-1].lstrip(".") or p, fontsize=9)
        ax.set_ylabel("loss", fontsize=9)
        ax.tick_params(labelsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Cryptosporidium calibration loss vs swept parameter", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "crypto_calibration_param_scatter.png")
    plt.savefig(path, dpi=170, bbox_inches="tight")
    print(f"  Figure saved -> {path}")
    plt.close(fig)


def _plot_best_fit(best, metric_cols, out_dir):
    """Best set's metric values against their empirical target bands."""
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.6), squeeze=False)
    for i, m in enumerate(metric_cols):
        ax = axes[0][i]
        lo, hi = CRYPTO_TARGETS[m]
        val = best[m]
        ax.axhspan(lo, hi, color="green", alpha=0.18, label="target")
        ax.bar([0], [val], width=0.5,
               color="#2196F3" if best[f"{m}_in_range"] else "#E53935")
        ax.set_xticks([])
        ax.set_title(m.replace("crypto_", "").replace("_", "\n"), fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, max(hi * 1.3, val * 1.15))
    fig.suptitle(f"Best-fit Cryptosporidium baseline vs empirical targets  (loss={best['loss']:.3f})",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "crypto_calibration_best_fit.png")
    plt.savefig(path, dpi=170, bbox_inches="tight")
    print(f"  Figure saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Drift measurement: what does adding a third pathogen do to the other two?
# ---------------------------------------------------------------------------

def measure_drift(grid_id: str, reps: int, steps: int, agents: int, n_cores: int | None):
    """Paired crypto-ON vs crypto-OFF comparison, plus a same-arm noise floor.

    Why this measurement exists
    ---------------------------
    Adding a pathogen is NOT inert in this model, and the coupling is easy to
    overlook. CareSeekingSystem._apply_treatment_outcome resolves EVERY active
    episode on a successfully treated child and forces INFECTIOUS -> RECOVERED
    for every pathogen at once; only ONE decision is made per household per
    day (for the sickest child); and household wealth is a shared pool that
    every care episode draws down. So extra cryptosporidiosis can plausibly push
    rotavirus and campylobacter burden either DOWN (more forced cures) or UP
    (wealth depleted, more households priced out of care) - the sign is not
    predictable from the mechanism alone, which is exactly why it is measured
    rather than assumed. docs/PROJECT_STATUS.md records the same channel
    operating when the care-seeking parameters were calibrated (rotavirus peak
    under-5 prevalence fell 0.099 -> 0.045).

    Pairing, and its one real limitation
    ------------------------------------
    orchestrator._run_one derives each replicate's seed as base_seed + rep
    independently of the parameter combo, so rep r uses the same seed in both
    arms. That makes household composition, child assignment and spatial
    placement IDENTICAL across arms (they are drawn before any pathogen-
    dependent code runs), and health/wealth identical too (drawn from the
    numpy stream, which the extra pathogen does not touch).

    It is NOT a perfectly paired comparison, and the reason is worth stating
    because it bounds what this measurement can claim: AgentFactory
    initialises per-pathogen state inside a loop over the configured
    pathogens, and seeding a pathogen's initial exposed cohort consumes a
    torch.randperm. A third pathogen therefore adds one draw and shifts every
    LATER torch draw in initialisation - which in practice means the
    TIME_USE activity distributions and the animal-ownership draws differ
    between arms. Those are not negligible influences on transmission.

    So the ON-OFF difference below mixes crypto's real effect with that
    residual sampling difference. To separate them, this function also
    measures a NOISE FLOOR: two crypto-OFF arms that differ only in their seed
    offset. A difference comparable to the noise floor is not evidence of a
    crypto effect; one clearly exceeding it is.
    """
    import numpy as np
    import pandas as pd

    def _arm(label, pathogens, base_seed):
        overrides = dict(BASE_OVERRIDES)
        overrides[PATHOGEN_SUBSET_KEY] = pathogens
        spec = build_spec(grid_id, reps, steps, agents, n_cores=n_cores,
                          name=f"{SPEC_NAME}_drift_{label}",
                          base_overrides=overrides)
        spec.metrics_fn = drift_metrics
        spec.base_seed = base_seed
        df = run_sweep(spec, combos=[{}])
        df["arm"] = label
        return df

    base = SVEIRCONFIG.seed
    off = _arm("crypto_off", ["rota", "campy"], base)
    on = _arm("crypto_on", ["rota", "campy", "crypto"], base)
    # Noise floor: crypto OFF again, on a disjoint seed set. Any apparent
    # "effect" of this arm versus `off` is pure sampling noise by construction.
    off2 = _arm("crypto_off_altseed", ["rota", "campy"], base + 10_000)

    both = pd.concat([off, on, off2], ignore_index=True)
    out_dir = os.path.join("experiments", "outputs", f"{SPEC_NAME}_drift")
    os.makedirs(out_dir, exist_ok=True)
    both.to_parquet(os.path.join(out_dir, "drift.parquet"), index=False)

    # Target metrics (rota/campy only - crypto has no OFF-arm counterpart),
    # plus the care-seeking/economic channel that carries any effect.
    channel = ["episode_care_seeking_rate", "could_not_afford_rate", "total_episodes",
               "decisions_faced", "mean_household_wealth", "mean_final_health"]
    rows, resolved = [], []
    for m in list(TARGETS):
        col = m if m in both.columns else m.replace("_peak_prevalence", "_peak_u5_prevalence")
        if col in both.columns:
            resolved.append((m, col, TARGETS[m]))
    for m in channel:
        if m in both.columns:
            resolved.append((m, m, None))

    print(f"\n--- Drift from adding Cryptosporidium ---")
    print(f"  {reps} replicates per arm, {agents} agents, {steps} days, seeds paired by rep.")
    print("  'noise' = the same comparison between two crypto-OFF arms differing only in seed;")
    print("  a delta not clearly exceeding it is NOT evidence of a crypto effect.\n")
    print(f"  {'metric':<36} {'OFF':>9} {'ON':>9} {'delta':>9} {'noise':>9} {'signal?':>8}  band")
    for m, col, band in resolved:
        a = off.sort_values("rep")[col].to_numpy(dtype=float)
        b = on.sort_values("rep")[col].to_numpy(dtype=float)
        c = off2.sort_values("rep")[col].to_numpy(dtype=float)
        delta = float(np.mean(b) - np.mean(a))
        noise = float(np.mean(c) - np.mean(a))
        # Paired SE of the ON-OFF difference, for the significance call.
        d = b - a
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        signal = "yes" if (abs(delta) > 2 * se and abs(delta) > abs(noise)) else "no"
        band_s = ""
        if band is not None:
            lo, hi = band
            band_s = f"[{lo}, {hi}] ON {'in' if lo <= np.mean(b) <= hi else 'OUT'}"
        print(f"  {m:<36} {np.mean(a):>9.4f} {np.mean(b):>9.4f} {delta:>+9.4f} "
              f"{noise:>+9.4f} {signal:>8}  {band_s}")
        rows.append({"metric": m, "off_mean": np.mean(a), "on_mean": np.mean(b),
                     "delta": delta, "paired_se": se, "same_arm_noise": noise,
                     "distinguishable_from_noise": signal == "yes",
                     "target_lo": band[0] if band else None,
                     "target_hi": band[1] if band else None,
                     "on_in_band": (band[0] <= np.mean(b) <= band[1]) if band else None})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "drift_summary.csv"), index=False)
    print(f"\n  Saved -> {os.path.join(out_dir, 'drift_summary.csv')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cryptosporidium transmission calibration.")
    parser.add_argument("-g", "--grid-id", default=DEFAULT_GRID_ID,
                        help=f"Grid ID (default: {DEFAULT_GRID_ID})")
    parser.add_argument("--samples", type=int, default=None,
                        help="LHS sample points. Default: 120 (full) or 12 (--pilot)")
    parser.add_argument("-r", "--reps", type=int, default=None,
                        help="Replicates per point. Default: 6 (full) or 3 (--pilot)")
    parser.add_argument("-s", "--steps", type=int, default=None,
                        help="Simulation days. Default: 300 (full) or 60 (--pilot)")
    parser.add_argument("-n", "--agents", type=int, default=None,
                        help="Number of agents. Default: 4000 (full) or 800 (--pilot). NOTE the "
                             "waterborne route is population-scale dependent (11 water sources "
                             "on this grid) - do not transfer bounds across sizes.")
    parser.add_argument("--seed", type=int, default=12345, help="LHS sampler seed (default: 12345)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes. Default: orchestrator's own default "
                             "(min(6, cpu_count())).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip the sweep; re-score and re-plot existing results.")
    parser.add_argument("--pilot", action="store_true",
                        help="Fast end-to-end smoke test: few LHS points, few reps, short run, "
                             "fewer agents. Sanity-checks the pipeline and timing before a full "
                             "search. Do NOT adopt pilot parameters.")
    parser.add_argument("--measure-drift", action="store_true",
                        help="Skip calibration; run the paired crypto-ON vs crypto-OFF comparison "
                             "quantifying what adding a third pathogen does to the rotavirus and "
                             "campylobacter targets.")
    args = parser.parse_args()

    reps = args.reps if args.reps is not None else (3 if args.pilot else 6)
    steps = args.steps if args.steps is not None else (60 if args.pilot else 300)
    agents = args.agents if args.agents is not None else (800 if args.pilot else 4000)

    if args.measure_drift:
        measure_drift(args.grid_id, reps, steps, agents, args.workers)
        return

    if args.plot_only:
        score_and_report(pilot=args.pilot)
        return

    samples = args.samples if args.samples is not None else (12 if args.pilot else 120)

    if args.pilot:
        print("*** PILOT MODE: few LHS points, reduced reps/steps/agents. "
              "For a timing/sanity check only - do not adopt these parameters. ***\n")

    combos = [baseline_combo()] + sample_lhs(samples, args.seed)
    print(f"Design: 1 baseline anchor + {samples} LHS points = {len(combos)} combos, "
          f"{reps} reps each ({len(combos) * reps} runs), rotavirus vaccination OFF.")

    spec = build_spec(args.grid_id, reps, steps, agents, pilot=args.pilot, n_cores=args.workers)
    run_sweep(spec, combos=combos)
    score_and_report(pilot=args.pilot)


if __name__ == "__main__":
    main()

"""
Experiment 5: CPT Decomposition — EV vs. Lambda-only vs. Full CPT
==================================================================
Sweeps cost_of_care across 10 values in [0.0, 0.50] under three decision
rules to decompose the individual contributions of the two CPT mechanisms:

  EV             — Plain expected utility. Objective probabilities, linear
                   value function, lambda = 1. Rational baseline.

  CPT-lambda     — Loss aversion only. Objective probabilities (gamma = 1,
                   no Prelec weighting), but asymmetric value function with
                   per-agent lambda. Isolates the loss-aversion effect.

  CPT-full       — Full CPT. Prelec probability weighting (per-agent gamma)
                   AND asymmetric value function (per-agent lambda).

Comparing EV → CPT-lambda shows the pure loss-aversion effect.
Comparing CPT-lambda → CPT-full shows the pure probability-weighting effect.
Comparing EV → CPT-full shows the net combined effect.

Experiment-level overrides (base model unchanged):
  daily_cost_of_living  = 0.015  (prevents wealth collapse)
  treatment_success_prob = 0.55  (makes seek a mixed prospect)

Metrics:
  - conditional_care_rate  : care events / decisions_faced
  - could_not_afford_rate  : priced-out decisions / decisions_faced
  - peak_u5_prevalence     : max fraction of u5s infectious (both pathogens)
  - cumulative_u5_days     : combined child-days of illness (rota + campy)
  - mean_parent_wealth     : mean final wealth of parent agents
  - mean_final_health      : mean final health across all agents

Usage
-----
    python exp5_cpt_vs_ev.py --grid-id <GRID_ID>

Optional flags:
    --reps    N   replicates per (cost, mode) combination  (default: 15)
    --steps   N   simulation length in days                (default: 250)
    --agents  N   number of agents                         (default: 4000)
    --output  DIR output directory  (default: outputs/exp5_cpt_vs_ev)
    --plot-only   skip simulation, just re-plot saved results
"""

import argparse
import io
import os
import pickle
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import SVEIRCONFIG, SVEIRConfig
from abm.model.initialize_model import SVEIRModel
from abm.systems.care_seeking import CareSeekingSystem
from abm.constants import AgentPropertyKeys
from abm.utils.rng import set_global_seed

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

COST_VALUES = np.linspace(0.0, 0.50, 10)
BASELINE    = 0.025

# Three arms: (use_cpt, gamma_override)
#   use_cpt=False, gamma_override=None  → EV
#   use_cpt=True,  gamma_override=1.0   → CPT-lambda (loss aversion only)
#   use_cpt=True,  gamma_override=None  → CPT-full
MODES = [
    ("EV",         False, 1.0),   # (label, use_cpt, gamma_override)
    ("CPT-lambda", True,  1.0),
    ("CPT-full",   True,  None),
]
MODE_COLOURS = {
    "EV":         "#FF9800",
    "CPT-lambda": "#9C27B0",
    "CPT-full":   "#2196F3",
}

OUTPUT_DIR = os.path.join("outputs", "exp5_cpt_vs_ev")
N_CORES    = max(1, min(6, cpu_count()))

# Experiment-level overrides
OVERRIDE_COST_OF_LIVING    = 0.015
OVERRIDE_TREATMENT_SUCCESS = 0.55

# ---------------------------------------------------------------------------
# Instrumented CareSeekingSystem — supports gamma override
# ---------------------------------------------------------------------------

class DecompositionCareSeekingSystem(CareSeekingSystem):
    """
    Extends CareSeekingSystem with an optional gamma_override.

    When gamma_override=1.0, the Prelec weighting function reduces to the
    identity (objective probabilities), isolating the loss-aversion effect.
    When gamma_override=None, each agent's own gamma is used (full CPT).
    """

    def __init__(self, config, use_cpt: bool, gamma_override: float | None):
        super().__init__(config, use_cpt=use_cpt)
        self.gamma_override = gamma_override

    def _evaluate_cpt(self, w, h_p, h_c, alpha, gamma, lam, params):
        # Apply gamma override if set
        g = self.gamma_override if self.gamma_override is not None else gamma
        return super()._evaluate_cpt(w, h_p, h_c, alpha, g, lam, params)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: SVEIRModel) -> dict:
    care_system = None
    for system in model.systems:
        if isinstance(system, CareSeekingSystem):
            care_system = system
            break

    if care_system is not None and care_system.decisions_faced > 0:
        conditional_care_rate = care_system.conditional_care_rate
        could_not_afford_rate = care_system.could_not_afford / care_system.decisions_faced
    else:
        conditional_care_rate = 0.0
        could_not_afford_rate = 0.0

    rota_prev  = np.array(model.u5_prevalence_history.get("rota",  []))
    campy_prev = np.array(model.u5_prevalence_history.get("campy", []))
    min_len    = min(len(rota_prev), len(campy_prev))
    peak_u5    = float((rota_prev[:min_len] + campy_prev[:min_len]).max()) if min_len > 0 else 0.0

    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age      = model.graph.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    n_u5     = int((is_child & (age < 60.0)).sum())
    cum_days = 0.0
    for pname in ("rota", "campy"):
        prev = np.array(model.u5_prevalence_history.get(pname, []))
        cum_days += float(prev.sum()) * n_u5

    final_states       = model.get_final_agent_states()
    is_parent          = final_states["is_parent"].astype(bool)
    mean_parent_wealth = float(np.mean(final_states["wealth"][is_parent])) if is_parent.any() else 0.0
    mean_final_health  = float(np.mean(final_states["health"]))

    return {
        "conditional_care_rate": conditional_care_rate,
        "could_not_afford_rate": could_not_afford_rate,
        "peak_u5_prevalence":    peak_u5,
        "cumulative_u5_days":    cum_days,
        "mean_parent_wealth":    mean_parent_wealth,
        "mean_final_health":     mean_final_health,
    }

# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------

def _run_one(args_tuple):
    cost, mode_label, use_cpt, gamma_override, rep, grid_id, steps, n_agents, base_seed, output_dir = args_tuple

    seed = base_seed + rep
    set_global_seed(seed)

    cfg: SVEIRConfig = SVEIRCONFIG.model_copy(deep=True)
    cfg.step_target   = steps
    cfg.number_agents = n_agents
    cfg.seed          = seed
    cfg.spatial_creation_args.grid_id             = grid_id
    cfg.steering_parameters.cost_of_care          = float(cost)
    cfg.steering_parameters.daily_cost_of_living  = OVERRIDE_COST_OF_LIVING
    cfg.steering_parameters.treatment_success_prob = OVERRIDE_TREATMENT_SUCCESS

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SVEIRModel(
                model_identifier=f"_exp5_{mode_label}_cost{cost:.4f}_rep{rep}",
                root_path=os.path.join(output_dir, "_tmp"),
            )
            model.set_model_parameters(**cfg.model_dump())
            model.initialize_model(verbose=False)

            # Replace CareSeekingSystem with the decomposition variant
            for idx, system in enumerate(model.systems):
                if isinstance(system, CareSeekingSystem):
                    model.systems[idx] = DecompositionCareSeekingSystem(
                        cfg, use_cpt=use_cpt, gamma_override=gamma_override
                    )
                    break

            model.run()

        return compute_metrics(model)

    except Exception:
        traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    Path(args.output).mkdir(parents=True, exist_ok=True)

    tasks = [
        (cost, label, use_cpt, gamma_ov, rep,
         args.grid_id, args.steps, args.agents, SVEIRCONFIG.seed, args.output)
        for cost                        in COST_VALUES
        for (label, use_cpt, gamma_ov) in MODES
        for rep                         in range(args.reps)
    ]

    total = len(tasks)
    print(f"\n--- Experiment 5: CPT Decomposition ---")
    print(f"  Arms         : EV | CPT-lambda (γ=1) | CPT-full")
    print(f"  Cost range   : {COST_VALUES[0]:.3f} → {COST_VALUES[-1]:.3f}  ({len(COST_VALUES)} values)")
    print(f"  Overrides    : cost_of_living={OVERRIDE_COST_OF_LIVING}, "
          f"p_success={OVERRIDE_TREATMENT_SUCCESS}")
    print(f"  Replicates   : {args.reps}")
    print(f"  Total runs   : {total}")
    print(f"  Workers      : {N_CORES}")
    print(f"  Steps/Agents : {args.steps} / {args.agents}\n")

    t0 = time.time()
    results_flat = []

    with Pool(processes=N_CORES) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            t = tasks[i - 1]
            results_flat.append((t[0], t[1], result))   # (cost, label, metrics)
            elapsed = time.time() - t0
            eta     = (elapsed / i) * (total - i)
            print(f"  [{i:>3}/{total}]  elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s", end="\r")

    print(f"\n  Sweep complete in {time.time() - t0:.0f}s")

    aggregated = {}
    for cost_val, label, metrics in results_flat:
        if metrics is None:
            continue
        key = (round(float(cost_val), 6), label)
        aggregated.setdefault(key, []).append(metrics)

    out_path = os.path.join(args.output, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "aggregated":  aggregated,
            "cost_values": COST_VALUES.tolist(),
            "baseline":    BASELINE,
            "modes":       [(l, u, g) for l, u, g in MODES],
            "overrides": {
                "daily_cost_of_living":   OVERRIDE_COST_OF_LIVING,
                "treatment_success_prob": OVERRIDE_TREATMENT_SUCCESS,
            },
        }, f)
    print(f"  Results saved → {out_path}")
    return aggregated

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRICS_CFG = {
    "conditional_care_rate": {"title": "Care-Seeking Rate",                       "ylabel": "Rate",        "pct": True},
    "could_not_afford_rate": {"title": "Could-Not-Afford Rate",                   "ylabel": "Rate",        "pct": True},
    "peak_u5_prevalence":    {"title": "Peak u5 Prevalence\n(combined pathogens)","ylabel": "Fraction",    "pct": True},
    "cumulative_u5_days":    {"title": "Cumulative u5 Child-Days\n(rota + campy)","ylabel": "Child-days",  "pct": False},
    "mean_parent_wealth":    {"title": "Mean Final Parent Wealth",                "ylabel": "Wealth (0–1)","pct": False},
    "mean_final_health":     {"title": "Mean Final Health\n(all agents)",         "ylabel": "Health (0–1)","pct": False},
}


def _build_stats(aggregated, cost_values):
    """Build mean/min/max arrays per mode per metric."""
    stats = {}
    for label, _, _ in MODES:
        stats[label] = {}
        for metric in METRICS_CFG:
            means, mins, maxes = [], [], []
            for cost in cost_values:
                key  = (round(float(cost), 6), label)
                reps = aggregated.get(key, [])
                vals = [r[metric] for r in reps if r is not None]
                means.append(np.mean(vals) if vals else np.nan)
                mins.append(np.min(vals)   if vals else np.nan)
                maxes.append(np.max(vals)  if vals else np.nan)
            stats[label][metric] = {
                "mean": np.array(means),
                "min":  np.array(mins),
                "max":  np.array(maxes),
            }
    return stats


def plot_results(args):
    results_path = os.path.join(args.output, "results.pkl")
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}. Run the sweep first.")
        return

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    aggregated  = data["aggregated"]
    cost_values = np.array(data["cost_values"])
    baseline    = data["baseline"]
    overrides   = data.get("overrides", {})
    stats       = _build_stats(aggregated, cost_values)

    override_str = (
        f"cost_of_living={overrides.get('daily_cost_of_living', '?')}, "
        f"p_success={overrides.get('treatment_success_prob', '?')}"
    )

    sns.set_theme(style="whitegrid", font_scale=1.05)

    # ------------------------------------------------------------------
    # Figure 1: All three arms on each metric
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Experiment 5: CPT Decomposition — EV vs. Lambda-only vs. Full CPT\n"
        f"(overrides: {override_str})",
        fontsize=13,
    )

    for ax, (metric, cfg) in zip(axes.flatten(), METRICS_CFG.items()):
        for label, _, _ in MODES:
            colour = MODE_COLOURS[label]
            s      = stats[label][metric]
            ax.plot(cost_values, s["mean"], marker="o", color=colour,
                    linewidth=2, label=label, zorder=4)
            ax.fill_between(cost_values, s["min"], s["max"],
                            alpha=0.12, color=colour)

        ax.axvline(baseline, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Original baseline ({baseline})")
        ax.set_title(cfg["title"], fontsize=12)
        ax.set_xlabel("Cost of Care", fontsize=11)
        ax.set_ylabel(cfg["ylabel"], fontsize=11)
        ax.legend(fontsize=8)
        if cfg["pct"]:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    plt.tight_layout()
    out_fig = os.path.join(args.output, "exp5_decomposition.png")
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    print(f"  Figure 1 saved → {out_fig}")
    plt.show()

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'Cost':>8}  {'EV care%':>10}  {'λ-only care%':>13}  "
          f"{'Full CPT care%':>15}  {'λ effect':>10}  {'γ effect':>10}")
    print("-" * 75)
    for i, cost in enumerate(cost_values):
        ev_r   = stats["EV"]["conditional_care_rate"]["mean"][i]
        lam_r  = stats["CPT-lambda"]["conditional_care_rate"]["mean"][i]
        full_r = stats["CPT-full"]["conditional_care_rate"]["mean"][i]
        lam_eff  = lam_r  - ev_r
        gam_eff  = full_r - lam_r
        print(f"  {cost:>6.3f}  {ev_r:>9.1%}  {lam_r:>12.1%}  {full_r:>14.1%}  "
              f"{lam_eff:>+9.1%}  {gam_eff:>+9.1%}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: CPT Decomposition")
    parser.add_argument("-g", "--grid-id",  required=False)
    parser.add_argument("-r", "--reps",     type=int, default=15)
    parser.add_argument("-s", "--steps",    type=int, default=250)
    parser.add_argument("-n", "--agents",   type=int, default=4000)
    parser.add_argument("-o", "--output",   type=str, default=OUTPUT_DIR)
    parser.add_argument("--plot-only",      action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_results(args)
    else:
        if not args.grid_id:
            parser.error("--grid-id is required unless --plot-only is set.")
        run_sweep(args)
        plot_results(args)


if __name__ == "__main__":
    main()

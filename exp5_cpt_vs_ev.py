"""
Experiment 5: CPT vs. Expected-Utility Care-Seeking
=====================================================
Sweeps cost_of_care across 10 values in [0.0, 0.05] under two decision
rules to isolate the behavioural contribution of Cumulative Prospect Theory:

  CPT  (use_cpt=True)  — Prelec probability weighting + asymmetric value
                          function with per-agent loss aversion (lambda).
  EV   (use_cpt=False) — Plain expected utility: objective probabilities,
                          linear value function, lambda = 1.

All other model parameters are identical between the two arms.

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

COST_VALUES = np.linspace(0.0, 0.05, 10)
BASELINE    = 0.025
MODES       = [True, False]
MODE_LABELS = {True: "CPT", False: "EV"}
OUTPUT_DIR  = os.path.join("outputs", "exp5_cpt_vs_ev")
N_CORES     = max(1, min(6, cpu_count()))

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: SVEIRModel) -> dict:
    """Extract outcome metrics from a completed model run."""

    # Care-seeking counters
    care_system: CareSeekingSystem | None = None
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

    # Peak u5 prevalence (combined pathogens)
    rota_prev  = np.array(model.u5_prevalence_history.get("rota",  []))
    campy_prev = np.array(model.u5_prevalence_history.get("campy", []))
    min_len    = min(len(rota_prev), len(campy_prev))
    peak_u5    = float((rota_prev[:min_len] + campy_prev[:min_len]).max()) if min_len > 0 else 0.0

    # Cumulative child-days
    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age      = model.graph.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    n_u5     = int((is_child & (age < 60.0)).sum())
    cum_days = 0.0
    for pname in ("rota", "campy"):
        prev = np.array(model.u5_prevalence_history.get(pname, []))
        cum_days += float(prev.sum()) * n_u5

    # Final wealth / health
    final_states = model.get_final_agent_states()
    is_parent    = final_states["is_parent"].astype(bool)
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
    cost, use_cpt, rep, grid_id, steps, n_agents, base_seed, output_dir = args_tuple

    seed = base_seed + rep
    set_global_seed(seed)

    cfg: SVEIRConfig = SVEIRCONFIG.model_copy(deep=True)
    cfg.step_target   = steps
    cfg.number_agents = n_agents
    cfg.seed          = seed
    cfg.spatial_creation_args.grid_id    = grid_id
    cfg.steering_parameters.cost_of_care = float(cost)

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SVEIRModel(
                model_identifier=f"_exp5_cost{cost:.3f}_cpt{use_cpt}_rep{rep}",
                root_path=os.path.join(output_dir, "_tmp"),
            )
            model.set_model_parameters(**cfg.model_dump())
            model.initialize_model(verbose=False)

            # Override the CareSeekingSystem with the correct use_cpt flag.
            # initialize_model() always creates it with use_cpt=True (default),
            # so we replace it here before running.
            for idx, system in enumerate(model.systems):
                if isinstance(system, CareSeekingSystem):
                    model.systems[idx] = CareSeekingSystem(cfg, use_cpt=use_cpt)
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
        (cost, use_cpt, rep, args.grid_id, args.steps, args.agents, SVEIRCONFIG.seed, args.output)
        for cost    in COST_VALUES
        for use_cpt in MODES
        for rep     in range(args.reps)
    ]

    total = len(tasks)
    print(f"\n--- Experiment 5: CPT vs. EV Care-Seeking ---")
    print(f"  Cost values  : {len(COST_VALUES)}  ({COST_VALUES[0]:.3f} → {COST_VALUES[-1]:.3f})")
    print(f"  Modes        : CPT, EV")
    print(f"  Replicates   : {args.reps}")
    print(f"  Total runs   : {total}")
    print(f"  Workers      : {N_CORES}")
    print(f"  Steps / Agents : {args.steps} / {args.agents}\n")

    t0 = time.time()
    results_flat = []

    with Pool(processes=N_CORES) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            results_flat.append((tasks[i - 1][0], tasks[i - 1][1], result))
            elapsed = time.time() - t0
            eta     = (elapsed / i) * (total - i)
            print(f"  [{i:>3}/{total}]  elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s", end="\r")

    print(f"\n  Sweep complete in {time.time() - t0:.0f}s")

    # Aggregate: key = (round(cost), use_cpt)
    aggregated = {}
    for cost_val, use_cpt, metrics in results_flat:
        if metrics is None:
            continue
        key = (round(float(cost_val), 6), bool(use_cpt))
        aggregated.setdefault(key, []).append(metrics)

    out_path = os.path.join(args.output, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "aggregated":  aggregated,
            "cost_values": COST_VALUES.tolist(),
            "baseline":    BASELINE,
        }, f)
    print(f"  Results saved → {out_path}")
    return aggregated

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRICS_CFG = {
    "conditional_care_rate": {
        "title": "Care-Seeking Rate",
        "ylabel": "Rate",
        "pct": True,
        "colour": {"CPT": "#2196F3", "EV": "#FF9800"},
    },
    "could_not_afford_rate": {
        "title": "Could-Not-Afford Rate",
        "ylabel": "Rate",
        "pct": True,
        "colour": {"CPT": "#2196F3", "EV": "#FF9800"},
    },
    "peak_u5_prevalence": {
        "title": "Peak u5 Prevalence\n(combined pathogens)",
        "ylabel": "Fraction",
        "pct": True,
        "colour": {"CPT": "#2196F3", "EV": "#FF9800"},
    },
    "cumulative_u5_days": {
        "title": "Cumulative u5 Child-Days\n(rota + campy)",
        "ylabel": "Child-days",
        "pct": False,
        "colour": {"CPT": "#2196F3", "EV": "#FF9800"},
    },
    "mean_parent_wealth": {
        "title": "Mean Final Parent Wealth",
        "ylabel": "Wealth (0–1)",
        "pct": False,
        "colour": {"CPT": "#2196F3", "EV": "#FF9800"},
    },
    "mean_final_health": {
        "title": "Mean Final Health\n(all agents)",
        "ylabel": "Health (0–1)",
        "pct": False,
        "colour": {"CPT": "#2196F3", "EV": "#FF9800"},
    },
}


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

    # Build per-mode, per-metric arrays
    stats: dict[bool, dict[str, np.ndarray]] = {}
    for use_cpt in MODES:
        label = MODE_LABELS[use_cpt]
        stats[use_cpt] = {}
        for metric in METRICS_CFG:
            means, mins, maxes = [], [], []
            for cost in cost_values:
                key  = (round(float(cost), 6), use_cpt)
                reps = aggregated.get(key, [])
                vals = [r[metric] for r in reps if r is not None]
                means.append(np.mean(vals) if vals else np.nan)
                mins.append(np.min(vals)   if vals else np.nan)
                maxes.append(np.max(vals)  if vals else np.nan)
            stats[use_cpt][metric] = {
                "mean": np.array(means),
                "min":  np.array(mins),
                "max":  np.array(maxes),
            }

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Experiment 5: CPT vs. Expected-Utility Care-Seeking", fontsize=15)

    for ax, (metric, cfg) in zip(axes.flatten(), METRICS_CFG.items()):
        for use_cpt in MODES:
            label  = MODE_LABELS[use_cpt]
            colour = cfg["colour"][label]
            s      = stats[use_cpt][metric]
            ax.plot(cost_values, s["mean"], marker="o", color=colour,
                    linewidth=2, label=label, zorder=4)
            ax.fill_between(cost_values, s["min"], s["max"],
                            alpha=0.15, color=colour)

        ax.axvline(baseline, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Baseline ({baseline})")
        ax.set_title(cfg["title"], fontsize=12)
        ax.set_xlabel("Cost of Care", fontsize=11)
        ax.set_ylabel(cfg["ylabel"], fontsize=11)
        ax.legend(fontsize=8)
        if cfg["pct"]:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    plt.tight_layout()
    out_fig = os.path.join(args.output, "exp5_cpt_vs_ev.png")
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    print(f"  Figure saved → {out_fig}")
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: CPT vs. EV Care-Seeking")
    parser.add_argument("-g", "--grid-id",  required=False)
    parser.add_argument("-r", "--reps",     type=int, default=20)
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

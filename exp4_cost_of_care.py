"""
Experiment 4: Cost-of-Care Sweep
=================================
Sweeps cost_of_care across 10 values in [0.0, 0.5] to detect how the
financial barrier to healthcare access affects care-seeking behaviour,
child disease burden, and household wealth.

Metrics:
  - conditional_care_rate  : care events / decisions_faced
                             (fraction of triggered decisions that led to care)
  - could_not_afford_rate  : could_not_afford / decisions_faced
                             (fraction of severe cases priced out of care)
  - peak_u5_prevalence     : max fraction of u5s infectious (both pathogens)
  - cumulative_u5_days     : combined child-days of illness (rota + campy)
  - mean_final_wealth      : mean final wealth across all agents
  - mean_final_health      : mean final health across all agents

Usage
-----
    python exp4_cost_of_care.py --grid-id <GRID_ID>

Optional flags:
    --reps    N   replicates per parameter value (default: 15)
    --steps   N   simulation length in days      (default: 250)
    --agents  N   number of agents               (default: 4000)
    --output  DIR output directory               (default: outputs/exp4_cost_of_care)
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

PARAM_VALUES = np.linspace(0.0, 0.03, 10)
BASELINE     = 0.025
OUTPUT_DIR   = os.path.join("outputs", "exp4_cost_of_care")
N_CORES      = max(1, min(6, cpu_count()))

OVERRIDE_COST_OF_LIVING    = 0.015
OVERRIDE_TREATMENT_SUCCESS = 0.55

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: SVEIRModel, steps: int) -> dict:
    """Extract all outcome metrics from a completed model run."""

    # --- Care-seeking counters from CareSeekingSystem ---
    care_system: CareSeekingSystem | None = None
    for system in model.systems:
        if isinstance(system, CareSeekingSystem):
            care_system = system
            break

    if care_system is not None and care_system.decisions_faced > 0:
        conditional_care_rate = care_system.conditional_care_rate
        could_not_afford_rate = (
            care_system.could_not_afford / care_system.decisions_faced
        )
        decisions_faced = care_system.decisions_faced
    else:
        conditional_care_rate = 0.0
        could_not_afford_rate = 0.0
        decisions_faced       = 0

    # --- Peak u5 prevalence (combined pathogens) ---
    rota_prev  = np.array(model.u5_prevalence_history.get("rota",  []))
    campy_prev = np.array(model.u5_prevalence_history.get("campy", []))
    min_len    = min(len(rota_prev), len(campy_prev))
    if min_len > 0:
        peak_u5 = float((rota_prev[:min_len] + campy_prev[:min_len]).max())
    else:
        peak_u5 = 0.0

    # --- Cumulative child-days (both pathogens) ---
    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age      = model.graph.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    n_u5     = int((is_child & (age < 60.0)).sum())

    cum_days = 0.0
    for pname in ("rota", "campy"):
        prev = np.array(model.u5_prevalence_history.get(pname, []))
        cum_days += float(prev.sum()) * n_u5

    # --- Final wealth and health ---
    final_states = model.get_final_agent_states()
    mean_wealth  = float(np.mean(final_states["wealth"]))
    mean_health  = float(np.mean(final_states["health"]))

    # --- Parent-specific final wealth ---
    is_parent    = final_states["is_parent"].astype(bool)
    mean_parent_wealth = float(np.mean(final_states["wealth"][is_parent])) if is_parent.any() else 0.0

    return {
        "conditional_care_rate": conditional_care_rate,
        "could_not_afford_rate": could_not_afford_rate,
        "decisions_faced":       decisions_faced,
        "peak_u5_prevalence":    peak_u5,
        "cumulative_u5_days":    cum_days,
        "mean_final_wealth":     mean_wealth,
        "mean_final_health":     mean_health,
        "mean_parent_wealth":    mean_parent_wealth,
    }


# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------

def _run_one(args_tuple):
    cost, rep, grid_id, steps, n_agents, base_seed, output_dir = args_tuple

    seed = base_seed + rep
    set_global_seed(seed)

    cfg: SVEIRConfig = SVEIRCONFIG.model_copy(deep=True)
    cfg.step_target = steps
    cfg.number_agents = n_agents
    cfg.seed = seed
    cfg.spatial_creation_args.grid_id = grid_id
    cfg.steering_parameters.cost_of_care = float(cost)
    cfg.steering_parameters.daily_cost_of_living = OVERRIDE_COST_OF_LIVING
    cfg.steering_parameters.treatment_success_prob = OVERRIDE_TREATMENT_SUCCESS

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SVEIRModel(
                model_identifier=f"_exp4_cost{cost:.3f}_rep{rep}",
                root_path=os.path.join(output_dir, "_tmp"),
            )
            model.set_model_parameters(**cfg.model_dump())
            model.initialize_model(verbose=False)
            model.run()

        return compute_metrics(model, steps)

    except Exception:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    Path(args.output).mkdir(parents=True, exist_ok=True)

    tasks = [
        (cost, rep, args.grid_id, args.steps, args.agents, SVEIRCONFIG.seed, args.output)
        for cost in PARAM_VALUES
        for rep  in range(args.reps)
    ]

    total = len(tasks)
    print(f"\n--- Experiment 4: Cost-of-Care Sweep ---")
    print(f"  Parameter values : {len(PARAM_VALUES)}  ({PARAM_VALUES[0]:.3f} → {PARAM_VALUES[-1]:.3f})")
    print(f"  Replicates       : {args.reps}")
    print(f"  Total runs       : {total}")
    print(f"  Workers          : {N_CORES}")
    print(f"  Steps / Agents   : {args.steps} / {args.agents}\n")

    t0 = time.time()
    results_flat = []

    with Pool(processes=N_CORES) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            results_flat.append((tasks[i - 1][0], result))
            elapsed = time.time() - t0
            eta     = (elapsed / i) * (total - i)
            print(f"  [{i:>3}/{total}]  elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s", end="\r")

    print(f"\n  Sweep complete in {time.time() - t0:.0f}s")

    aggregated = {}
    for cost_val, metrics in results_flat:
        if metrics is None:
            continue
        key = round(float(cost_val), 6)
        aggregated.setdefault(key, []).append(metrics)

    out_path = os.path.join(args.output, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "aggregated":   aggregated,
            "param_values": PARAM_VALUES.tolist(),
            "baseline":     BASELINE,
        }, f)
    print(f"  Results saved → {out_path}")
    return aggregated


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(args):
    results_path = os.path.join(args.output, "results.pkl")
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}. Run the sweep first.")
        return

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    aggregated = data["aggregated"]
    param_vals = sorted(aggregated.keys())
    baseline   = data["baseline"]
    x          = np.array(param_vals)

    # ---- Collect means and stds ------------------------------------------
    metrics_keys = [
        "conditional_care_rate",
        "could_not_afford_rate",
        "peak_u5_prevalence",
        "cumulative_u5_days",
        "mean_parent_wealth",
        "mean_final_health",
    ]
    means = {k: [] for k in metrics_keys}
    mins = {k: [] for k in metrics_keys}
    maxes = {k: [] for k in metrics_keys}
    raw   = {k: [] for k in metrics_keys}

    for v in param_vals:
        reps = aggregated[v]
        for k in metrics_keys:
            vals = [r[k] for r in reps if r is not None]
            means[k].append(np.mean(vals) if vals else np.nan)
            mins[k].append(np.min(vals) if vals else np.nan)
            maxes[k].append(np.max(vals) if vals else np.nan)
            raw[k].append(vals)

    for k in metrics_keys:
        means[k] = np.array(means[k])
        mins[k] = np.array(mins[k])
        maxes[k] = np.array(maxes[k])

    # ---- Titles, colours, formatters -------------------------------------
    cfg = {
        "conditional_care_rate": {
            "title": "Care-Seeking Rate",
            "colour": "#2196F3",
            "pct": True,
        },
        "could_not_afford_rate": {
            "title": "Could-Not-Afford Rate",
            "colour": "#F44336",
            "pct": True,
        },
        "peak_u5_prevalence": {
            "title": "Peak Under 5 Prevalence\n(combined pathogens)",
            "colour": "#FF5722",
            "pct": True,
        },
        "cumulative_u5_days": {
            "title": "Cumulative Under 5 Child-Days of Illness\n(rota + campy)",
            "colour": "#9C27B0",
            "pct": False,
        },
        "mean_parent_wealth": {
            "title": "Mean Final Parent Wealth",
            "colour": "#FF9800",
            "pct": False,
        },
        "mean_final_health": {
            "title": "Mean Final Health\n(all agents)",
            "colour": "#4CAF50",
            "pct": False,
        },
    }

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, k in zip(axes.flatten(), metrics_keys):
        c = cfg[k]["colour"]
        m = means[k]
        mn = mins[k]
        mx = maxes[k]

        ax.plot(x, m, marker="o", color=c, linewidth=2, zorder=4)
        ax.fill_between(x, mn, mx, alpha=0.2, color=c)

        ax.set_title(cfg[k]["title"], fontsize=16)
        ax.set_xlabel("Cost of Care", fontsize=14)

        if cfg[k]["pct"]:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    plt.tight_layout()
    out_fig = os.path.join(args.output, "exp4_cost_of_care.png")
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    print(f"  Figure saved → {out_fig}")
    plt.show()

    # # ---- Bonus: care rate vs. child-days scatter -------------------------
    # # Each point is one replicate; colour encodes cost_of_care value
    # fig2, ax2 = plt.subplots(figsize=(9, 6))
    # cmap   = plt.get_cmap("plasma", len(param_vals))
    # sm     = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=x.min(), vmax=x.max()))
    # sm.set_array([])

    # for ci, v in enumerate(param_vals):
    #     reps = aggregated[v]
    #     cr   = [r["conditional_care_rate"] for r in reps if r is not None]
    #     cd   = [r["cumulative_u5_days"]    for r in reps if r is not None]
    #     ax2.scatter(cr, cd, color=cmap(ci), alpha=0.65, s=45, zorder=3)
    #     # Profile mean as a larger marker
    #     ax2.scatter(np.mean(cr), np.mean(cd), color=cmap(ci),
    #                 s=160, marker="D", edgecolors="black", linewidths=0.8, zorder=5)

    # fig2.colorbar(sm, ax=ax2, label="cost_of_care")
    # ax2.set_title(
    #     "Conditional Care Rate vs. Cumulative u5 Child-Days\n"
    #     "(diamonds = parameter-value means, colour = cost_of_care)",
    #     fontsize=16,
    # )
    # ax2.set_xlabel("Conditional Care-Seeking Rate", fontsize=14)
    # ax2.set_ylabel("Cumulative Under 5 Child-Days of Illness", fontsize=14)
    # ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # plt.tight_layout()
    # out_fig2 = os.path.join(args.output, "exp4_care_vs_burden.png")
    # plt.savefig(out_fig2, dpi=180, bbox_inches="tight")
    # print(f"  Scatter figure saved → {out_fig2}")
    # plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Cost-of-Care Sweep")
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

"""
Experiment 3: Vaccination Herd Immunity Surface
================================================
2D grid sweep of vaccination_rate × vaccine_efficacy (5 × 5 = 25 combos)
to map the herd immunity threshold for Rotavirus.

Metrics (under-5s, Rotavirus only):
  - peak_u5_prevalence   : max fraction of u5s infectious on any single day
  - cumulative_u5_days   : area under the u5 prevalence curve (child-days of illness)
  - extinction_flag      : 1 if peak_u5_prevalence < 1%, else 0

Usage
-----
    python exp3_vaccination.py --grid-id <GRID_ID>

Optional flags:
    --reps    N   replicates per combination  (default: 15)
    --steps   N   simulation length in days   (default: 250)
    --agents  N   number of agents            (default: 4000)
    --output  DIR output directory            (default: outputs/exp3_vaccination)
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
from abm.constants import AgentPropertyKeys
from abm.utils.rng import set_global_seed

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

VACC_RATES      = np.linspace(0.0,  0.05, 5)   # daily vaccination probability
VACC_EFFICACIES = np.linspace(0.2,  0.95, 5)   # vaccine efficacy
OUTPUT_DIR      = os.path.join("outputs", "exp3_vaccination")
N_CORES         = max(1, min(6, cpu_count()))

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: SVEIRModel, steps: int) -> dict:
    u5_prev = np.array(model.u5_prevalence_history.get("rota", []))
    peak    = float(u5_prev.max()) if u5_prev.size > 0 else 0.0

    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age      = model.graph.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    n_u5     = int((is_child & (age < 60.0)).sum())
    cumulative_days = float(u5_prev.sum()) * n_u5

    return {
        "peak_u5_prevalence": peak,
        "cumulative_u5_days": cumulative_days,
        "extinction_flag":    int(peak < 0.01),
    }


# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------

def _run_one(args_tuple):
    vacc_rate, vacc_eff, rep, grid_id, steps, n_agents, base_seed, output_dir = args_tuple

    seed = base_seed + rep
    set_global_seed(seed)

    cfg: SVEIRConfig = SVEIRCONFIG.model_copy(deep=True)
    cfg.step_target   = steps
    cfg.number_agents = n_agents
    cfg.seed          = seed
    cfg.spatial_creation_args.grid_id = grid_id

    for p in cfg.pathogens:
        if p.name == "rota":
            p.vaccination_rate  = float(vacc_rate)
            p.vaccine_efficacy  = float(vacc_eff)

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SVEIRModel(
                model_identifier=f"_exp3_vr{vacc_rate:.4f}_ve{vacc_eff:.3f}_rep{rep}",
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
        (vr, ve, rep, args.grid_id, args.steps, args.agents, SVEIRCONFIG.seed, args.output)
        for vr  in VACC_RATES
        for ve  in VACC_EFFICACIES
        for rep in range(args.reps)
    ]

    total = len(tasks)
    print(f"\n--- Experiment 3: Vaccination Herd Immunity Surface ---")
    print(f"  vaccination_rate  : {VACC_RATES}")
    print(f"  vaccine_efficacy  : {VACC_EFFICACIES}")
    print(f"  Combinations      : {len(VACC_RATES) * len(VACC_EFFICACIES)}")
    print(f"  Replicates        : {args.reps}")
    print(f"  Total runs        : {total}")
    print(f"  Workers           : {N_CORES}")
    print(f"  Steps / Agents    : {args.steps} / {args.agents}\n")

    t0 = time.time()
    results_flat = []

    with Pool(processes=N_CORES) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            results_flat.append((tasks[i - 1][0], tasks[i - 1][1], result))  # (vr, ve, metrics)
            elapsed = time.time() - t0
            eta     = (elapsed / i) * (total - i)
            print(f"  [{i:>3}/{total}]  elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s", end="\r")

    print(f"\n  Sweep complete in {time.time() - t0:.0f}s")

    # Aggregate: key = (round(vr), round(ve))
    aggregated = {}
    for vr, ve, metrics in results_flat:
        if metrics is None:
            continue
        key = (round(float(vr), 6), round(float(ve), 6))
        aggregated.setdefault(key, []).append(metrics)

    out_path = os.path.join(args.output, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "aggregated":    aggregated,
            "vacc_rates":    VACC_RATES.tolist(),
            "vacc_efficacies": VACC_EFFICACIES.tolist(),
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

    aggregated      = data["aggregated"]
    vacc_rates      = np.array(data["vacc_rates"])
    vacc_efficacies = np.array(data["vacc_efficacies"])

    nr = len(vacc_rates)
    ne = len(vacc_efficacies)

    # Build 2D arrays: rows = vacc_rate, cols = vacc_efficacy
    peak_grid  = np.full((nr, ne), np.nan)
    cum_grid   = np.full((nr, ne), np.nan)
    ext_grid   = np.full((nr, ne), np.nan)

    for ri, vr in enumerate(vacc_rates):
        for ei, ve in enumerate(vacc_efficacies):
            key = (round(float(vr), 6), round(float(ve), 6))
            if key not in aggregated or not aggregated[key]:
                continue
            reps = aggregated[key]
            peak_grid[ri, ei] = np.mean([r["peak_u5_prevalence"] for r in reps])
            cum_grid[ri, ei]  = np.mean([r["cumulative_u5_days"]  for r in reps])
            ext_grid[ri, ei]  = np.mean([r["extinction_flag"]     for r in reps])

    # Axis labels
    rate_labels = [f"{v:.3f}" for v in vacc_rates]
    eff_labels  = [f"{v:.2f}" for v in vacc_efficacies]

    sns.set_theme(style="white", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    def _heatmap(ax, matrix, title, fmt, cmap, vmin=None, vmax=None):
        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=eff_labels,
            yticklabels=rate_labels,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("vaccine_efficacy", fontsize=10)
        ax.set_ylabel("vaccination_rate (per day)", fontsize=10)
        ax.invert_yaxis()

    _heatmap(axes[0], peak_grid, "Mean Peak u5 Prevalence", ".2f", "Reds", 0, None)
    _heatmap(axes[1], cum_grid, "Mean Cumulative u5 Child-Days", ".0f", "Reds", 0, None)

    plt.tight_layout()
    out_fig = os.path.join(args.output, "exp3_vaccination.png")
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    print(f"  Figure saved → {out_fig}")
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Vaccination Herd Immunity Surface")
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

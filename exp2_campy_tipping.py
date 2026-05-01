"""
Experiment 2: Campylobacter Tipping Point
==========================================
Sweeps human_animal_interaction_rate across 10 values in [0.05, 0.60]
to detect a phase transition driven by the zoonotic transmission route.

Metrics (under-5s only):
  - peak_u5_prevalence   : max fraction of u5s infectious on any single day
  - cumulative_u5_days   : area under the u5 prevalence curve (child-days of illness)
  - zoonotic_fraction    : fraction of all campy infections from the zoonotic route

Usage
-----
    python exp2_campy_tipping.py --grid-id <GRID_ID>

Optional flags:
    --reps    N   replicates per parameter value (default: 15)
    --steps   N   simulation length in days     (default: 250)
    --agents  N   number of agents              (default: 4000)
    --output  DIR output directory              (default: outputs/exp2_campy_tipping)
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
from abm.pathogens.campylobacter import Campylobacter
from abm.utils.rng import set_global_seed

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

PARAM_VALUES = np.linspace(0.01, 0.50, 10)
BASELINE     = 0.25
OUTPUT_DIR   = os.path.join("outputs", "exp2_campy_tipping")
N_CORES      = max(1, min(6, cpu_count()))

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: SVEIRModel, steps: int) -> dict:
    """Extract peak u5 prevalence, cumulative child-days, and zoonotic fraction."""
    u5_prev = np.array(model.u5_prevalence_history.get("campy", []))

    peak = float(u5_prev.max()) if u5_prev.size > 0 else 0.0

    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age      = model.graph.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    n_u5     = int((is_child & (age < 60.0)).sum())
    cumulative_days = float(u5_prev.sum()) * n_u5

    # Zoonotic fraction from the Campylobacter pathogen instance
    zoonotic_frac = 0.0
    for p in model.pathogens:
        if isinstance(p, Campylobacter):
            total = p.total_zoonotic + p.total_fecal_oral
            zoonotic_frac = p.total_zoonotic / total if total > 0 else 0.0
            break

    return {
        "peak_u5_prevalence": peak,
        "cumulative_u5_days": cumulative_days,
        "zoonotic_fraction":  zoonotic_frac,
    }


# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------

def _run_one(args_tuple):
    interaction_rate, rep, grid_id, steps, n_agents, base_seed, output_dir = args_tuple

    seed = base_seed + rep
    set_global_seed(seed)

    cfg: SVEIRConfig = SVEIRCONFIG.model_copy(deep=True)
    cfg.step_target   = steps
    cfg.number_agents = n_agents
    cfg.seed          = seed
    cfg.spatial_creation_args.grid_id = grid_id

    for p in cfg.pathogens:
        if p.name == "campy":
            p.human_animal_interaction_rate = float(interaction_rate)

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SVEIRModel(
                model_identifier=f"_exp2_rate{interaction_rate:.4f}_rep{rep}",
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
        (rate, rep, args.grid_id, args.steps, args.agents, SVEIRCONFIG.seed, args.output)
        for rate in PARAM_VALUES
        for rep  in range(args.reps)
    ]

    total = len(tasks)
    print(f"\n--- Experiment 2: Campylobacter Tipping Point ---")
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
    for rate_val, metrics in results_flat:
        if metrics is None:
            continue
        key = round(float(rate_val), 6)
        aggregated.setdefault(key, []).append(metrics)

    out_path = os.path.join(args.output, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"aggregated": aggregated, "param_values": PARAM_VALUES.tolist(), "baseline": BASELINE}, f)
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

    peak_means, peak_mins, peak_maxes = [], [], []
    cum_means, cum_mins, cum_maxes  = [], [], []
    zoo_means,  zoo_mins, zoo_maxes = [], [], []
    extinction_probs = []

    for v in param_vals:
        reps  = aggregated[v]
        peaks = [r["peak_u5_prevalence"] for r in reps]
        cums  = [r["cumulative_u5_days"]  for r in reps]
        zoos  = [r["zoonotic_fraction"]   for r in reps]
        peak_means.append(np.mean(peaks))
        peak_mins.append(np.min(peaks))
        peak_maxes.append(np.max(peaks))
        cum_means.append(np.mean(cums))
        cum_mins.append(np.min(cums))
        cum_maxes.append(np.max(cums))
        zoo_means.append(np.mean(zoos))
        zoo_mins.append(np.min(zoos))
        zoo_maxes.append(np.max(zoos))
        extinction_probs.append(np.mean([p < 0.01 for p in peaks]))

    peak_means = np.array(peak_means)
    peak_mins = np.array(peak_mins)
    peak_maxes = np.array(peak_maxes)
    cum_means = np.array(cum_means)
    cum_mins = np.array(cum_mins)
    cum_maxes = np.array(cum_maxes)
    zoo_means = np.array(zoo_means)
    zoo_mins = np.array(zoo_mins)
    zoo_maxes = np.array(zoo_maxes)
    extinction_probs = np.array(extinction_probs)
    x = np.array(param_vals)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # --- Panel 1: Peak u5 prevalence ---
    ax = axes[0]
    ax.plot(x, peak_means, marker="o", color="#2196F3", linewidth=2)
    ax.fill_between(x, peak_mins, peak_maxes, alpha=0.2, color="#2196F3")
    ax.set_ylabel("Peak Under 5 Prevalence", fontsize=16)
    ax.set_xlabel("Human-Animal Interaction Rate", fontsize=16)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # --- Panel 2: Cumulative child-days ---
    ax = axes[1]
    ax.plot(x, cum_means, marker="o", color="#FF5722", linewidth=2)
    ax.fill_between(x, cum_mins, cum_maxes, alpha=0.2, color="#FF5722")
    ax.set_ylabel("Cumulative Under 5 Child-Days of Illness", fontsize=16)
    ax.set_xlabel("Human-Animal Interaction Rate", fontsize=16)

    plt.tight_layout()
    out_fig = os.path.join(args.output, "exp2_campy_tipping.png")
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    print(f"  Figure saved → {out_fig}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Campylobacter Tipping Point")
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

"""
Experiment 6: Water-Shock Frequency Sweep
==========================================
Sweeps shock_daily_prob across 8 values spanning "never" to "every 3 days"
to quantify how the frequency of environmental water-contamination shocks
drives Rotavirus burden (the only waterborne pathogen in this model).
Campylobacter metrics are included as a negative control — they should be
largely flat because Campylobacter transmission is zoonotic / fecal-oral
and does not use the water-shock pathway.

Metrics:
  - rota_peak_u5_prevalence    : max fraction of u5s infectious (Rotavirus)
  - rota_cumulative_u5_days    : Rotavirus child-days of illness (u5)
  - campy_peak_u5_prevalence   : same for Campylobacter (control)
  - campy_cumulative_u5_days   : same for Campylobacter (control)
  - water_contamination_frac   : mean fraction of simulation days on which
                                 at least one water cell was contaminated
                                 (tracked via EnvironmentSystem counter)
  - conditional_care_rate      : care events / decisions_faced
  - mean_final_health          : mean final health (all agents)

Usage
-----
    python exp6_shock_frequency.py --grid-id <GRID_ID>

Optional flags:
    --reps    N   replicates per parameter value (default: 15)
    --steps   N   simulation length in days      (default: 250)
    --agents  N   number of agents               (default: 4000)
    --output  DIR output directory               (default: outputs/exp6_shock_frequency)
    --plot-only   skip simulation, just re-plot saved results

Implementation note
-------------------
EnvironmentSystem does not natively track contamination history, so this
experiment wraps its `update()` method with a plain closure to count
contaminated-water days.  The original method is captured in the closure
and called unchanged, keeping the core model untouched.
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
import torch

from config import SVEIRCONFIG, SVEIRConfig
from abm.model.initialize_model import SVEIRModel
from abm.systems.care_seeking import CareSeekingSystem
from abm.systems.environment import EnvironmentSystem
from abm.constants import AgentPropertyKeys, GridLayer, WaterStatus
from abm.utils.rng import set_global_seed

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Span from "no shocks" to roughly "every 3 days"
PARAM_VALUES = np.array([0.0, 1/90, 1/60, 1/30, 1/14, 1/7, 1/5, 1/3])
BASELINE     = 1 / 30
OUTPUT_DIR   = os.path.join("outputs", "exp6_shock_frequency")
N_CORES      = max(1, min(6, cpu_count()))

# ---------------------------------------------------------------------------
# Water-contamination tracker (closure-based patch, no MethodType)
# ---------------------------------------------------------------------------

def _patch_environment_system(model: SVEIRModel) -> dict:
    """
    Wraps EnvironmentSystem.update() with a plain closure to count days on
    which at least one water cell is contaminated *before* the recovery step
    runs.

    The original method is stored and called inside the closure so behaviour
    is completely unchanged.  Returns a shared counter dict that the caller
    reads after the run.

    Note: MethodType is intentionally avoided.  Assigning a plain function
    to an instance attribute works because Python looks up instance
    attributes before the class dict, so env_sys.update() will call our
    closure.  The closure receives (agent_state, **kwargs) directly —
    no implicit 'self' is prepended.
    """
    counter = {"contaminated_days": 0, "total_days": 0}

    env_sys: EnvironmentSystem | None = None
    for system in model.systems:
        if isinstance(system, EnvironmentSystem):
            env_sys = system
            break

    if env_sys is None:
        return counter

    # Capture the bound method before overwriting the instance attribute
    original_update = env_sys.update

    def _instrumented_update(agent_state, **kwargs):
        grid = kwargs.get("grid")
        if grid is not None:
            water_idx = grid.property_to_index.get(GridLayer.WATER)
            if water_idx is not None:
                water_slice = grid.grid_tensor[:, :, water_idx]
                if torch.any(water_slice == WaterStatus.CONTAMINATED):
                    counter["contaminated_days"] += 1
        counter["total_days"] += 1
        # Call the original bound method — signature is (agent_state, **kwargs)
        original_update(agent_state, **kwargs)

    # Assign as a plain instance attribute; Python resolves it before the
    # class method, so env_sys.update(...) calls our closure directly.
    env_sys.update = _instrumented_update
    return counter

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: SVEIRModel, counter: dict, steps: int) -> dict:
    """Extract all outcome metrics from a completed model run."""

    metrics = {}
    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age      = model.graph.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    n_u5     = int((is_child & (age < 60.0)).sum())

    for pname in ("rota", "campy"):
        prev = np.array(model.u5_prevalence_history.get(pname, []))
        metrics[f"{pname}_peak_u5_prevalence"] = float(prev.max()) if prev.size > 0 else 0.0
        metrics[f"{pname}_cumulative_u5_days"] = float(prev.sum()) * n_u5

    # Water contamination fraction
    total = counter["total_days"]
    metrics["water_contamination_frac"] = (
        counter["contaminated_days"] / total if total > 0 else 0.0
    )

    # Care-seeking
    care_system: CareSeekingSystem | None = None
    for system in model.systems:
        if isinstance(system, CareSeekingSystem):
            care_system = system
            break
    metrics["conditional_care_rate"] = (
        care_system.conditional_care_rate
        if care_system is not None and care_system.decisions_faced > 0
        else 0.0
    )

    # Final health
    final_states = model.get_final_agent_states()
    metrics["mean_final_health"] = float(np.mean(final_states["health"]))

    return metrics

# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------

def _run_one(args_tuple):
    shock_prob, rep, grid_id, steps, n_agents, base_seed, output_dir = args_tuple

    seed = base_seed + rep
    set_global_seed(seed)

    cfg: SVEIRConfig = SVEIRCONFIG.model_copy(deep=True)
    cfg.step_target   = steps
    cfg.number_agents = n_agents
    cfg.seed          = seed
    cfg.spatial_creation_args.grid_id        = grid_id
    cfg.steering_parameters.shock_daily_prob = float(shock_prob)

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SVEIRModel(
                model_identifier=f"_exp6_shock{shock_prob:.5f}_rep{rep}",
                root_path=os.path.join(output_dir, "_tmp"),
            )
            model.set_model_parameters(**cfg.model_dump())
            model.initialize_model(verbose=False)
            # Patch AFTER initialize_model so the system object exists,
            # but BEFORE run() so every step is instrumented.
            counter = _patch_environment_system(model)
            model.run()

        return compute_metrics(model, counter, steps)

    except Exception:
        traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    Path(args.output).mkdir(parents=True, exist_ok=True)

    tasks = [
        (prob, rep, args.grid_id, args.steps, args.agents, SVEIRCONFIG.seed, args.output)
        for prob in PARAM_VALUES
        for rep  in range(args.reps)
    ]

    total = len(tasks)
    print(f"\n--- Experiment 6: Water-Shock Frequency Sweep ---")
    print(f"  Parameter values : {len(PARAM_VALUES)}")
    print(f"    {[f'{v:.4f}' for v in PARAM_VALUES]}")
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
    for prob_val, metrics in results_flat:
        if metrics is None:
            continue
        key = round(float(prob_val), 8)
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

def _nice_label(prob: float) -> str:
    """Convert a daily probability to a human-readable recurrence string."""
    if prob == 0.0:
        return "Never"
    period = 1.0 / prob
    return f"1/{period:.0f}d"


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
    x_labels   = [_nice_label(v) for v in param_vals]

    metric_keys = [
        "rota_peak_u5_prevalence",
        "rota_cumulative_u5_days",
        "campy_peak_u5_prevalence",
        "campy_cumulative_u5_days",
        "water_contamination_frac",
        "conditional_care_rate",
        "mean_final_health",
    ]
    stats = {k: {"mean": [], "min": [], "max": []} for k in metric_keys}

    for v in param_vals:
        reps = aggregated[v]
        for k in metric_keys:
            vals = [r[k] for r in reps if r is not None]
            stats[k]["mean"].append(np.mean(vals) if vals else np.nan)
            stats[k]["min"].append(np.min(vals)   if vals else np.nan)
            stats[k]["max"].append(np.max(vals)   if vals else np.nan)

    for k in metric_keys:
        for s in ("mean", "min", "max"):
            stats[k][s] = np.array(stats[k][s])

    plot_cfg = {
        "rota_peak_u5_prevalence":  {"title": "Rota Peak u5 Prevalence",                    "colour": "#2196F3", "pct": True},
        "rota_cumulative_u5_days":  {"title": "Rota Cumulative u5 Child-Days",               "colour": "#1565C0", "pct": False},
        "campy_peak_u5_prevalence": {"title": "Campy Peak u5 Prevalence\n(control)",         "colour": "#FF5722", "pct": True},
        "campy_cumulative_u5_days": {"title": "Campy Cumulative u5 Child-Days\n(control)",   "colour": "#BF360C", "pct": False},
        "water_contamination_frac": {"title": "Fraction of Days with\nContaminated Water",   "colour": "#00897B", "pct": True},
        "conditional_care_rate":    {"title": "Conditional Care-Seeking Rate",               "colour": "#7B1FA2", "pct": True},
        "mean_final_health":        {"title": "Mean Final Health\n(all agents)",             "colour": "#388E3C", "pct": False},
    }

    sns.set_theme(style="whitegrid", font_scale=1.05)

    # --- Main 2×4 panel (7 metrics, last cell blank) ---
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Experiment 6: Water-Shock Frequency Sweep", fontsize=15)

    for ax, k in zip(axes.flatten(), metric_keys):
        cfg = plot_cfg[k]
        c   = cfg["colour"]
        ax.plot(x, stats[k]["mean"], marker="o", color=c, linewidth=2, zorder=4)
        ax.fill_between(x, stats[k]["min"], stats[k]["max"], alpha=0.2, color=c)
        ax.axvline(baseline, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Baseline ({baseline:.4f})")
        ax.set_title(cfg["title"], fontsize=11)
        ax.set_xlabel("Daily Shock Probability", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        if cfg["pct"]:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    axes.flatten()[-1].set_visible(False)

    plt.tight_layout()
    out_fig = os.path.join(args.output, "exp6_shock_frequency.png")
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    print(f"  Figure saved → {out_fig}")
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Water-Shock Frequency Sweep")
    parser.add_argument("-g", "--grid-id",  required=False)
    parser.add_argument("-r", "--reps",     type=int, default=30)
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

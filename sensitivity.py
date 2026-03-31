# sensitivity.py
"""
Sensitivity analysis for SPRINGS-ABM.

Usage
-----
# Step 1: run the sweep (saves results to outputs/sensitivity/)
python sensitivity.py sensitivity --grid-id <GRID_ID> --reps 5

# Step 2: plot results
python sensitivity.py plot-sensitivity

Optional flags for the sensitivity stage:
  --reps   N     Number of stochastic replicates per parameter set (default: 5)
  --steps  N     Simulation length in days (default: 150)
  --agents N     Number of agents (default: 5000)
  --output DIR   Output directory (default: outputs/sensitivity)

What it does
------------
A one-at-a-time (OAT) sensitivity sweep.  Each parameter is varied across a
grid of values while all others are held at their baseline.  For each
(parameter, value) combination, `--reps` independent replicates are run and
the following summary statistics are recorded:

  Per pathogen (rota / campy):
    - episodes_per_child_year       mean illness episodes per child under 5
    - peak_prevalence               maximum daily prevalence (all ages)
    - peak_day                      day on which peak prevalence occurs
    - attack_rate                   proportion of agents infected at least once

  Campylobacter-specific:
    - hh_secondary_attack_rate      proportion of susceptible household members
                                    infected within households that contain at
                                    least one infectious campy case

Empirical target ranges (sub-Saharan Africa / Ghana literature) are overlaid
on every plot so you can immediately see which parameter values produce
plausible model behaviour.
"""
import argparse
import io
import os
import pickle
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import SVEIRCONFIG, SVEIRConfig
from abm.model.initialize_model import SVEIRModel
from abm.constants import AgentPropertyKeys, Compartment
from abm.utils.rng import set_global_seed

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join("outputs", "sensitivity")

# ---------------------------------------------------------------------------
# Empirical target ranges
# Literature: sub-Saharan Africa / Ghana; GEMS study; WHO rotavirus bulletins
# ---------------------------------------------------------------------------
TARGETS = {
    "rota_episodes_per_child_year":  (1.5,  2.5),
    "campy_episodes_per_child_year": (1.0,  3.0),
    "rota_peak_prevalence":          (0.02, 0.10),
    "campy_peak_prevalence":         (0.02, 0.15),
    "campy_zoonotic_fraction":       (0.50, 0.80),
}

# ---------------------------------------------------------------------------
# Parameters to sweep and their candidate values
# ---------------------------------------------------------------------------
SWEEP_PARAMS = {
    # --- Rotavirus ---
    "pathogens[rota].infection_prob_mean": [
        0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,
    ],
    "steering_parameters.water_to_human_infection_prob": [
        0.0025, 0.005, 0.01, 0.02, 0.05, 0.10,
    ],
    "steering_parameters.human_to_water_infection_prob": [
        0.00001, 0.0001, 0.001, 0.005, 0.01,
    ],
    # --- Campylobacter ---
    "pathogens[campy].human_animal_interaction_rate": [
        0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
    ],
    "pathogens[campy].fecal_oral_prob": [
        0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12,
    ],
    # --- Shared ---
    "steering_parameters.prior_infection_immunity_factor": [
        0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
    ],
    "steering_parameters.shock_daily_prob": [
        1/60, 1/30, 1/15, 1/7,
    ],
}

PARAM_LABELS = {
    "pathogens[rota].infection_prob_mean":                  "Rota H2H infection prob",
    "steering_parameters.water_to_human_infection_prob":    "Water → human infection prob",
    "steering_parameters.human_to_water_infection_prob":    "Human → water infection prob",
    "pathogens[campy].human_animal_interaction_rate":       "Campy animal interaction rate",
    "pathogens[campy].fecal_oral_prob":                     "Campy fecal-oral prob",
    "steering_parameters.prior_infection_immunity_factor":  "Prior-infection immunity factor",
    "steering_parameters.shock_daily_prob":                 "Daily water shock prob",
}

METRIC_LABELS = {
    "rota_episodes_per_child_year":  "Rota episodes / child-year (u5)",
    "campy_episodes_per_child_year": "Campy episodes / child-year (u5)",
    "rota_peak_prevalence":          "Rota peak prevalence (u5 fraction)",
    "campy_peak_prevalence":         "Campy peak prevalence (u5 fraction)",
    "rota_peak_day":                 "Rota peak prevalence day",
    "campy_peak_day":                "Campy peak prevalence day",
    "campy_zoonotic_fraction":       "Campy zoonotic fraction",
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _set_param(config: SVEIRConfig, path: str, value):
    if path.startswith("pathogens["):
        bracket_end = path.index("]")
        p_name = path[len("pathogens["):bracket_end]
        attr   = path[bracket_end + 2:]
        for p in config.pathogens:
            if p.name == p_name:
                setattr(p, attr, value)
                return
        raise ValueError(f"Pathogen '{p_name}' not found in config.")
    else:
        parts = path.split(".", 1)
        if len(parts) == 2:
            obj = getattr(config, parts[0])
            setattr(obj, parts[1], value)
        else:
            setattr(config, path, value)


def _get_param(config: SVEIRConfig, path: str):
    if path.startswith("pathogens["):
        bracket_end = path.index("]")
        p_name = path[len("pathogens["):bracket_end]
        attr   = path[bracket_end + 2:]
        for p in config.pathogens:
            if p.name == p_name:
                return getattr(p, attr)
        raise ValueError(f"Pathogen '{p_name}' not found in config.")
    else:
        parts = path.split(".", 1)
        if len(parts) == 2:
            return getattr(getattr(config, parts[0]), parts[1])
        return getattr(config, path)

# ---------------------------------------------------------------------------
# Summary statistics — all restricted to under-5s
# ---------------------------------------------------------------------------

def _compute_metrics(model: SVEIRModel, steps: int) -> dict:
    g         = model.graph
    sim_years = steps / 365.0

    is_child = g.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy().astype(bool)
    age      = g.ndata[AgentPropertyKeys.AGE].cpu().numpy()
    under5   = is_child & (age < 60.0)   # age stored in months
    n_u5     = under5.sum()

    metrics = {}

    for p in model.pathogens:
        pname     = p.name
        count_key = AgentPropertyKeys.num_infections(pname)

        num_inf_u5 = g.ndata[count_key].cpu().numpy()[under5]

        # --- Episodes per child-year (under-5 only) ---
        metrics[f"{pname}_episodes_per_child_year"] = (
            float(num_inf_u5.mean()) / sim_years
            if n_u5 > 0 and sim_years > 0 else 0.0
        )

        # --- Peak prevalence and peak day among under-5s ---
        # Uses the per-pathogen under-5 prevalence history recorded by
        # SVEIRModel._record_u5_prevalence() on every step — no approximation.
        u5_prev = model.u5_prevalence_history.get(pname, [])
        if u5_prev and max(u5_prev) > 0:
            prev_arr = np.array(u5_prev)
            peak_idx = int(np.argmax(prev_arr))
            metrics[f"{pname}_peak_prevalence"] = float(prev_arr[peak_idx])
            metrics[f"{pname}_peak_day"]        = float(peak_idx)
        else:
            metrics[f"{pname}_peak_prevalence"] = 0.0
            metrics[f"{pname}_peak_day"]        = 0.0

    # --- Campylobacter zoonotic fraction (population-wide route split) ---
    metrics["campy_zoonotic_fraction"] = _campy_zoonotic_fraction(model)

    return metrics


def _campy_zoonotic_fraction(model: SVEIRModel) -> float:
    """
    Fraction of campylobacter infections attributable to the zoonotic route.
    Reads from lifetime counters on the Campylobacter pathogen instance.
    Returns 0.0 if no infections occurred.
    """
    from abm.pathogens.campylobacter import Campylobacter

    campy = next((p for p in model.pathogens if isinstance(p, Campylobacter)), None)
    if campy is None:
        return 0.0
    total = campy.total_zoonotic + campy.total_fecal_oral
    return campy.total_zoonotic / total if total > 0 else 0.0

# ---------------------------------------------------------------------------
# Single simulation runner
# ---------------------------------------------------------------------------

def _run_one(config: SVEIRConfig, rep: int, output_dir: str) -> dict | None:
    """
    Run a single simulation and return its metrics dict, or None on error.
    All stdout/stderr from the model is suppressed so it does not interfere
    with the progress display.  Errors are always printed.
    """
    seed = config.seed + rep
    set_global_seed(seed)

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SVEIRModel(
                model_identifier=f"_sens_rep{rep}",
                root_path=output_dir,
            )
            model.set_model_parameters(**config.model_dump())
            model.initialize_model(verbose=False)
            model.run()

        return _compute_metrics(model, config.step_target)

    except Exception:
        traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------

def run_sensitivity(args):
    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    base_config = SVEIRCONFIG.model_copy(deep=True)
    base_config.step_target   = args.steps
    base_config.number_agents = args.agents
    base_config.spatial_creation_args.grid_id = args.grid_id

    n_values   = sum(len(v) for v in SWEEP_PARAMS.values())
    total_runs = n_values * args.reps

    print(f"\n--- Sensitivity sweep ---")
    print(f"  Parameters : {len(SWEEP_PARAMS)}")
    print(f"  Total runs : {total_runs}  ({n_values} value sets × {args.reps} reps)")
    print(f"  Steps      : {args.steps}  |  Agents: {args.agents}\n")

    results = []
    run_idx  = 0
    t0       = time.time()

    for param_path, values in SWEEP_PARAMS.items():
        baseline_value = _get_param(base_config, param_path)
        label = PARAM_LABELS.get(param_path, param_path)
        print(f"\n  [{label}]  baseline = {baseline_value:.5g}")

        for val in values:
            cfg = base_config.model_copy(deep=True)
            _set_param(cfg, param_path, val)

            rep_metrics = []
            for rep in range(args.reps):
                run_idx += 1
                elapsed = time.time() - t0
                eta     = (elapsed / run_idx) * (total_runs - run_idx) if run_idx > 1 else 0
                print(
                    f"    val={val:<10.5g}  rep={rep + 1}/{args.reps}"
                    f"  run {run_idx}/{total_runs}"
                    f"  elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s   ",
                    end="\r",
                )
                m = _run_one(cfg, rep, os.path.join(output_dir, "_tmp"))
                if m is not None:
                    rep_metrics.append(m)

            n_ok = len(rep_metrics)
            print(
                f"    val={val:<10.5g}  "
                f"{n_ok}/{args.reps} reps OK"
                + ("" if n_ok == args.reps else "  *** some reps failed ***")
            )

            if not rep_metrics:
                continue

            all_keys = rep_metrics[0].keys()
            agg = {"param": param_path, "value": val, "baseline": baseline_value}
            for k in all_keys:
                vals_k = [m[k] for m in rep_metrics]
                agg[f"{k}_mean"] = float(np.mean(vals_k))
                agg[f"{k}_std"]  = float(np.std(vals_k))
            results.append(agg)

    out_path = os.path.join(output_dir, "sensitivity_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    total_time = time.time() - t0
    print(f"\nSweep complete in {total_time:.0f}s.  Results saved to {out_path}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLOT_GROUPS = [
    {
        "title": "Rotavirus — Incidence & Burden (under-5)",
        "metrics": [
            "rota_episodes_per_child_year",
            "rota_peak_prevalence",
            "rota_peak_day",
        ],
    },
    {
        "title": "Campylobacter — Incidence & Burden (under-5)",
        "metrics": [
            "campy_episodes_per_child_year",
            "campy_peak_prevalence",
            "campy_peak_day",
            "campy_zoonotic_fraction",
        ],
    },
]


def plot_sensitivity(args):
    results_path = os.path.join(args.output, "sensitivity_results.pkl")
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}. Run 'sensitivity' first.")
        return

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    df             = pd.DataFrame(results)
    params_present = df["param"].unique()
    output_dir     = args.output

    sns.set_theme(style="whitegrid", font_scale=1.0)

    for group in PLOT_GROUPS:
        group_title = group["title"]
        metrics     = group["metrics"]
        n_metrics   = len(metrics)
        n_params    = len(params_present)

        fig, axes = plt.subplots(
            n_params, n_metrics,
            figsize=(4.5 * n_metrics, 3.2 * n_params),
            squeeze=False,
        )
        fig.suptitle(f"OAT Sensitivity — {group_title}", fontsize=15, y=1.01)

        for row_idx, param_path in enumerate(params_present):
            sub         = df[df["param"] == param_path].sort_values("value")
            baseline    = sub["baseline"].iloc[0]
            param_label = PARAM_LABELS.get(param_path, param_path)

            for col_idx, metric in enumerate(metrics):
                ax       = axes[row_idx, col_idx]
                mean_col = f"{metric}_mean"
                std_col  = f"{metric}_std"

                if mean_col not in sub.columns:
                    ax.set_visible(False)
                    continue

                x    = sub["value"].values
                y    = sub[mean_col].values
                yerr = sub[std_col].values

                if metric in TARGETS:
                    lo, hi = TARGETS[metric]
                    ax.axhspan(lo, hi, color="green", alpha=0.12)

                ax.axvline(baseline, color="grey", linestyle="--", linewidth=1.2)
                ax.plot(x, y, marker="o", markersize=4, linewidth=1.8,
                        color="#2196F3", zorder=3)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.20, color="#2196F3")

                if row_idx == 0:
                    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(param_label, fontsize=8)
                ax.set_xlabel("Parameter value", fontsize=8)
                ax.tick_params(labelsize=7)

                if x.max() > 0 and (x.max() / max(x.min(), 1e-12)) > 100:
                    ax.set_xscale("log")

        target_patch  = mpatches.Patch(color="green", alpha=0.3, label="Empirical target range")
        baseline_line = plt.Line2D([0], [0], color="grey", linestyle="--", label="Baseline value")
        fig.legend(
            handles=[target_patch, baseline_line],
            loc="lower center", ncol=2, fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )

        plt.tight_layout()
        safe_title = group_title.replace(" ", "_").replace("—", "").replace("/", "")
        out_file   = os.path.join(output_dir, f"sensitivity_{safe_title}.png")
        plt.savefig(out_file, bbox_inches="tight", dpi=180)
        print(f"Saved: {out_file}")
        plt.show()

    _plot_tornado(df, params_present, output_dir)


def _plot_tornado(df: pd.DataFrame, params_present, output_dir: str):
    all_metrics = [m for g in PLOT_GROUPS for m in g["metrics"]]
    n_metrics   = len(all_metrics)

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(4.5 * n_metrics, 0.55 * len(params_present) + 2.0),
        squeeze=False,
    )
    fig.suptitle("Tornado Plot — Parameter Influence on Each Metric", fontsize=13)

    for col_idx, metric in enumerate(all_metrics):
        ax       = axes[0, col_idx]
        mean_col = f"{metric}_mean"

        influences = []
        for param_path in params_present:
            sub  = df[df["param"] == param_path]
            if mean_col not in sub.columns:
                continue
            vals = sub[mean_col].dropna()
            if len(vals) < 2:
                continue
            influences.append((PARAM_LABELS.get(param_path, param_path), vals.max() - vals.min()))

        if not influences:
            ax.set_visible(False)
            continue

        influences.sort(key=lambda x: x[1])
        labels, values = zip(*influences)
        bars = ax.barh(labels, values, color="#2196F3", edgecolor="white", height=0.6)

        if metric in TARGETS:
            lo, hi = TARGETS[metric]
            for bar, (plabel, _) in zip(bars, influences):
                matching = [p for p in params_present if PARAM_LABELS.get(p, p) == plabel]
                if not matching:
                    continue
                sub = df[df["param"] == matching[0]]
                if mean_col in sub.columns:
                    if sub[mean_col].min() <= hi and sub[mean_col].max() >= lo:
                        bar.set_color("#4CAF50")

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_xlabel("Max − Min (mean metric)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)

    blue_patch  = mpatches.Patch(color="#2196F3", label="Parameter influence")
    green_patch = mpatches.Patch(color="#4CAF50", label="Sweep range overlaps target")
    fig.legend(
        handles=[blue_patch, green_patch],
        loc="lower center", ncol=2, fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    out_file = os.path.join(output_dir, "sensitivity_tornado.png")
    plt.savefig(out_file, bbox_inches="tight", dpi=180)
    print(f"Saved: {out_file}")
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for SPRINGS-ABM.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=["sensitivity", "plot-sensitivity"],
        help=(
            "sensitivity      — run the OAT parameter sweep\n"
            "plot-sensitivity — plot results from a completed sweep"
        ),
    )
    parser.add_argument("-g", "--grid-id", type=str,  help="Grid ID (required for 'sensitivity')")
    parser.add_argument("-r", "--reps",    type=int,  default=5,    help="Replicates per parameter value (default: 5)")
    parser.add_argument("-s", "--steps",   type=int,  default=200,  help="Simulation steps / days (default: 200)")
    parser.add_argument("-n", "--agents",  type=int,  default=3000, help="Number of agents (default: 3000)")
    parser.add_argument("-o", "--output",  type=str,  default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    if args.stage == "sensitivity":
        if not args.grid_id:
            parser.error("'sensitivity' requires --grid-id.")
        run_sensitivity(args)
    elif args.stage == "plot-sensitivity":
        plot_sensitivity(args)


if __name__ == "__main__":
    main()
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
  --reps   N     Number of stochastic replicates per parameter set (default: 10)
  --steps  N     Simulation length in days (default: 730; metrics exclude a 180-day burn-in)
  --agents N     Number of agents (default: 4000)
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
from experiments.calibration.targets import ALL_TARGETS as TARGETS

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join("outputs", "sensitivity")

# Empirical target ranges: see experiments/calibration/targets.py (shared with
# the LHS calibration search so both tools score against the same bands).
# ALL_TARGETS = the rotavirus/campylobacter bands plus the Cryptosporidium
# ones, so the OAT plots overlay a target band for every pathogen. The two
# calibration searches still score against their own subsets (TARGETS and
# CRYPTO_TARGETS respectively) - only this visual overlay wants everything.

# ---------------------------------------------------------------------------
# Parameters to sweep and their candidate values
# ---------------------------------------------------------------------------
# Transmission grids are generated from the CURRENT config.py defaults rather
# than hardcoded, so they cannot go stale after a recalibration (they had to be
# re-centred by hand after every round before 2026-10-20): each spans
# 0 -> 2x baseline with the baseline as the exact centre point, the file's
# long-standing convention. Exceptions: rotavirus's H2H probability has a
# 0.001 floor (rotavirus.py draws max(0.001, Normal(mean, std))), and the
# literature-grounded natural-history parameters keep fixed grids over their
# literature uncertainty ranges.
def _centred(path: str, floor: float = 0.0) -> list[float]:
    from experiments.orchestrator import get_param
    x = float(get_param(SVEIRCONFIG, path))
    return [max(floor, 0.0), 0.5 * x, x, 1.5 * x, 2.0 * x]


SWEEP_PARAMS = {
    # --- Rotavirus ---
    "pathogens[rota].infection_prob_mean": _centred("pathogens[rota].infection_prob_mean", floor=0.001),
    "pathogens[rota].recovery_rate": [0.1005, 0.2005, 0.3005, 0.4005, 0.5005],
    "pathogens[rota].water_dose_scale": _centred("pathogens[rota].water_dose_scale"),
    # --- Campylobacter ---
    "pathogens[campy].human_animal_interaction_rate": _centred("pathogens[campy].human_animal_interaction_rate"),
    "pathogens[campy].fecal_oral_prob": _centred("pathogens[campy].fecal_oral_prob"),
    "pathogens[campy].food_borne_prob": _centred("pathogens[campy].food_borne_prob"),
    "pathogens[campy].recovery_rate": [0.0266, 0.0866, 0.1466, 0.2066, 0.2666],
    "pathogens[campy].poultry_weight": [0.25, 0.5, 0.75, 1.0, 1.5],
    "pathogens[campy].ruminant_weight": [0.0, 0.2, 0.45, 0.7, 1.0],
    # --- Cryptosporidium ---
    # dose_response_r and oocyst_decay_rate are swept over their literature
    # uncertainty ranges (isolate-to-isolate ID50 variation; temperature-
    # dependent die-off) rather than 0 -> 2x.
    "pathogens[crypto].household_transmission_prob": _centred("pathogens[crypto].household_transmission_prob"),
    "pathogens[crypto].community_infection_prob": _centred("pathogens[crypto].community_infection_prob"),
    "pathogens[crypto].water_dose_scale": _centred("pathogens[crypto].water_dose_scale"),
    "pathogens[crypto].zoonotic_interaction_rate": _centred("pathogens[crypto].zoonotic_interaction_rate"),
    "pathogens[crypto].dose_response_r": [0.005, 0.011, 0.018, 0.04, 0.07],
    "pathogens[crypto].oocyst_decay_rate": [0.05, 0.10, 0.15, 0.25, 0.47],
    # --- Shared ---
    "steering_parameters.household_exposure_sdlog": [0.0, 0.3, 0.6, 0.9, 1.2],
}

PARAM_LABELS = {
    "pathogens[rota].infection_prob_mean": "Rota H2H infection prob",
    "pathogens[rota].water_dose_scale": "Rota waterborne dose scale",
    "steering_parameters.household_exposure_sdlog": "Household environmental-exposure SD (log)",
    "steering_parameters.water_to_human_infection_prob": "Water → human infection prob",
    "pathogens[campy].human_animal_interaction_rate": "Campy animal interaction rate",
    "pathogens[campy].fecal_oral_prob": "Campy fecal-oral prob",
    "pathogens[campy].food_borne_prob": "Campy food-borne background prob",
    "pathogens[campy].poultry_weight": "Campy poultry species weight",
    "pathogens[campy].ruminant_weight": "Campy ruminant species weight",
    "pathogens[crypto].household_transmission_prob": "Crypto household P2P prob",
    "pathogens[crypto].community_infection_prob": "Crypto community P2P prob",
    "pathogens[crypto].water_dose_scale": "Crypto waterborne dose scale",
    "pathogens[crypto].zoonotic_interaction_rate": "Crypto zoonotic (ruminant) rate",
    "pathogens[crypto].dose_response_r": "Crypto dose-response r",
    "pathogens[crypto].oocyst_decay_rate": "Crypto oocyst die-off rate",
}

METRIC_LABELS = {
    "rota_episodes_per_child_year": "Rota episodes / child-year (u5)",
    "campy_episodes_per_child_year": "Campy episodes / child-year (u5)",
    "rota_peak_prevalence": "Rota peak prevalence (u5 fraction)",
    "campy_peak_prevalence": "Campy peak prevalence (u5 fraction)",
    "campy_zoonotic_fraction": "Campy zoonotic fraction",
    "rota_peak_day": "Rota peak day",
    "campy_peak_day": "Campy peak day",
    # Crypto needs BOTH prevalence metrics listed, because for this pathogen
    # carriage and disease are not interchangeable: ~2/3 of infections are
    # subclinical and shedding lasts ~3 weeks, so infectious prevalence runs
    # several times symptomatic prevalence. Labelling them explicitly stops a
    # plot being read as a diarrhoea figure when it is a carriage figure.
    "crypto_infections_per_child_year": "Crypto infections / child-year (u5)",
    "crypto_symptomatic_episodes_per_child_year": "Crypto symptomatic episodes / child-year (u5)",
    "crypto_peak_prevalence": "Crypto peak CARRIAGE prevalence (u5 fraction)",
    "crypto_peak_u5_symptomatic_prevalence": "Crypto peak SYMPTOMATIC prevalence (u5 fraction)",
    "crypto_household_fraction": "Crypto household route fraction",
    "crypto_community_fraction": "Crypto community route fraction",
    "crypto_water_fraction": "Crypto waterborne route fraction",
    "crypto_zoonotic_fraction": "Crypto zoonotic route fraction",
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
    n_u5 = is_child.sum()

    metrics = {}

    for p in model.pathogens:
        pname     = p.name
        count_key = AgentPropertyKeys.num_infections(pname)

        num_inf_u5 = g.ndata[count_key].cpu().numpy()[is_child]

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

        # --- Symptomatic (disease) prevalence, separate from carriage ---
        # Only meaningfully different from *_peak_prevalence above for a
        # pathogen with subclinical infection (Cryptosporidium); recorded for
        # all so the OAT plots can never confuse the two.
        u5_sym = getattr(model, "u5_symptomatic_prevalence_history", {}).get(pname, [])
        metrics[f"{pname}_peak_u5_symptomatic_prevalence"] = (
            float(max(u5_sym)) if u5_sym else 0.0
        )

    # --- Per-route attribution fractions, every multi-route pathogen ---
    # Emits campy_zoonotic_fraction (as before) plus crypto_household_fraction,
    # crypto_community_fraction, crypto_water_fraction,
    # crypto_zoonotic_fraction, read from the generic lifetime route counters
    # the Pathogen base class maintains.
    metrics.update(_route_fractions(model))

    # --- Infections vs symptomatic episodes per child-year ---
    # "*_episodes_per_child_year" above counts INFECTIONS. For rotavirus and
    # campylobacter that is the same as episodes; for Cryptosporidium it is
    # about 3x the symptomatic episode count, so both are recorded under
    # unambiguous names.
    episode_counts: dict[str, int] = {}
    for entry in model.get_child_episode_log():
        episode_counts[entry["pathogen"]] = episode_counts.get(entry["pathogen"], 0) + 1
    for p in model.pathogens:
        metrics[f"{p.name}_infections_per_child_year"] = metrics[
            f"{p.name}_episodes_per_child_year"
        ]
        metrics[f"{p.name}_symptomatic_episodes_per_child_year"] = (
            episode_counts.get(p.name, 0) / (n_u5 * sim_years)
            if n_u5 > 0 and sim_years > 0 else 0.0
        )

    # --- Burn-in-aware, age-windowed metrics (the current MAL-ED targets) ---
    # Shared with experiments/metrics.calibration_metrics so the OAT overlay
    # and the calibration search score identical quantities.
    from experiments.metrics import age_window_rates, route_fractions_post_burnin
    metrics.update(age_window_rates(model))
    metrics.update(route_fractions_post_burnin(model))

    return metrics


def _target_band(metric: str):
    """Resolves a metric name onto a target band, tolerating the two spellings.

    This file names the under-5 prevalence metric "<p>_peak_prevalence" while
    experiments/metrics.py names it "<p>_peak_u5_prevalence", and the target
    dicts use whichever matched their original consumer (rotavirus and
    campylobacter the former, Cryptosporidium the latter). experiments/metrics.
    calibration_loss already resolves this in one direction; this does the
    other, so a band is overlaid whichever spelling a target happens to use
    instead of silently plotting no band at all.
    """
    for candidate in (metric, metric.replace("_peak_prevalence", "_peak_u5_prevalence")):
        if candidate in TARGETS:
            return TARGETS[candidate]
    return None


def _route_fractions(model: SVEIRModel) -> dict:
    """Per-route attribution fractions for every multi-route pathogen.

    Reads Pathogen.total_by_route, the generic lifetime route counters, rather
    than campylobacter-specific attributes - so a newly added pathogen's routes
    appear here automatically.
    """
    out = {}
    for p in model.pathogens:
        if not p.total_by_route:
            continue
        total = sum(p.total_by_route.values())
        for route, n in p.total_by_route.items():
            out[f"{p.name}_{route}_fraction"] = (n / total) if total > 0 else 0.0
    return out

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
    # Scored on the current (MAL-ED, 0-24 months, post burn-in) target keys,
    # so the overlaid bands are the ones calibration uses. See
    # experiments/calibration/targets.py.
    {
        "title": "Rotavirus — Infection & Disease (0-24 months, post burn-in)",
        "metrics": [
            "rota_infections_per_child_year_0_24m",
            "rota_symptomatic_episodes_per_child_year_0_24m",
            "rota_mean_u5_prevalence_post_burnin",
            "rota_h2h_fraction_post_burnin",
        ],
    },
    {
        "title": "Cryptosporidium — Infection & Disease (0-24 months, post burn-in)",
        "metrics": [
            "crypto_infections_per_child_year_0_24m",
            "crypto_symptomatic_episodes_per_child_year_0_24m",
            "crypto_mean_u5_prevalence_post_burnin",
            "crypto_peak_u5_symptomatic_prevalence",
            "crypto_household_fraction_post_burnin",
            "crypto_community_fraction_post_burnin",
            "crypto_water_fraction_post_burnin",
            "crypto_zoonotic_fraction_post_burnin",
        ],
    },
    {
        "title": "Campylobacter — Infection & Disease (0-24 months, post burn-in)",
        "metrics": [
            "campy_infections_per_child_year_0_24m",
            "campy_symptomatic_episodes_per_child_year_0_24m",
            "campy_mean_u5_prevalence_post_burnin",
            "campy_zoonotic_fraction_post_burnin",
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

                band = _target_band(metric)
                if band is not None:
                    ax.axhspan(band[0], band[1], color="green", alpha=0.12)

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
    parser.add_argument("-r", "--reps",    type=int,  default=10,    help="Replicates per parameter value (default: 10)")
    parser.add_argument("-s", "--steps",   type=int,  default=730,  help="Simulation steps / days (default: 730; the metrics exclude a 180-day burn-in)")
    parser.add_argument("-n", "--agents",  type=int,  default=4000, help="Number of agents (default: 4000)")
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
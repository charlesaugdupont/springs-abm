# experiments/gsa/run_gsa.py
"""
Global Sensitivity Analysis driver for SPRINGS-ABM.
===================================================

Two-stage design (see the approved plan):

  morris   Screen ALL ~51 user-editable model params (over their UI slider
           ranges) with Morris elementary effects. Cheap; ranks by mu* and
           flags interactions via sigma. Selects the top-k survivors.
  sobol    Sobol variance decomposition (S1 + ST) on the survivors - quantifies
           each param's first-order and total (incl. interaction) contribution.
  export   Combine into experiments/outputs/gsa/importance.json (+ ranked CSV
           and paste-ready ParamMeta snippets) for the UI-emphasis step.
  plot     Re-render plots from saved CSVs.

Every stage runs one sweep through experiments.orchestrator.run_sweep (parallel,
Parquet output) and analyses ALL output metrics; the headline importance ranking
uses epidemic-burden + care-seeking metrics (gsa_metrics.HEADLINE_METRICS).

Pilot first (this project's convention):
    python -m experiments.gsa.run_gsa morris --pilot
    python -m experiments.gsa.run_gsa sobol  --pilot
    python -m experiments.gsa.run_gsa export --pilot
Full:
    python -m experiments.gsa.run_gsa morris --workers 8
    python -m experiments.gsa.run_gsa sobol  --workers 8 --samples 128
    python -m experiments.gsa.run_gsa export
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample

from experiments.orchestrator import SweepSpec, run_sweep, load_results
from experiments.gsa.param_space import (
    gsa_params, build_problem, problem_from_selected, combos_from_sample,
)
from experiments.gsa.gsa_metrics import (
    gsa_metrics_fn, HEADLINE_METRICS, ALSO_REPORTED, ALL_GSA_METRICS,
)
from experiments.gsa.indices import (
    align_mean_output, run_morris, run_sobol, noise_floor, DESIGN_ID_COL,
)
from experiments.gsa import plots

GSA_OUTPUT_DIR = os.path.join("experiments", "outputs", "gsa")
DEFAULT_GRID_ID = "7d9ce7c720a6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec_name(stem: str, pilot: bool) -> str:
    return f"{stem}_pilot" if pilot else stem


def build_spec(name, grid_id, reps, steps, agents, n_cores) -> SweepSpec:
    return SweepSpec(
        name=name, grid_id=grid_id, params=[], metrics_fn=gsa_metrics_fn,
        reps=reps, steps=steps, agents=agents, output_dir=GSA_OUTPUT_DIR, n_cores=n_cores,
    )


def _tag(combos: list[dict]) -> list[dict]:
    for i, c in enumerate(combos):
        c[DESIGN_ID_COL] = i
    return combos


def _save_design(out_dir, X, problem):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "design_X.npy"), X)
    with open(os.path.join(out_dir, "problem.json"), "w") as f:
        json.dump(problem, f, indent=2)


def _load_design(out_dir):
    X = np.load(os.path.join(out_dir, "design_X.npy"))
    with open(os.path.join(out_dir, "problem.json")) as f:
        problem = json.load(f)
    return X, problem


def _present_metrics(df, names):
    return [m for m in names if m in df.columns]


# ---------------------------------------------------------------------------
# Morris stage
# ---------------------------------------------------------------------------

def run_morris_stage(args):
    name = _spec_name("morris", args.pilot)
    problem = build_problem(gsa_params())
    print(f"\n=== Morris screen: {problem['num_vars']} params, "
          f"N(trajectories)={args.samples}, levels={args.levels} ===")
    X = morris_sample.sample(problem, N=args.samples, num_levels=args.levels, seed=args.seed)
    combos = _tag(combos_from_sample(X, problem["names"]))
    print(f"Design: {len(X)} points x {args.reps} reps = {len(X) * args.reps} runs")
    spec = build_spec(name, args.grid_id, args.reps, args.steps, args.agents, args.workers)
    run_sweep(spec, combos=combos)
    _save_design(os.path.join(GSA_OUTPUT_DIR, name), X, problem)
    analyze_morris(name, args)


def analyze_morris(name, args):
    out_dir = os.path.join(GSA_OUTPUT_DIR, name)
    X, problem = _load_design(out_dir)
    df = load_results(name, output_dir=GSA_OUTPUT_DIR)
    n_design = len(X)

    screening = {}   # path -> max normalized mu_star across headline metrics
    for metric in _present_metrics(df, HEADLINE_METRICS):
        Y = align_mean_output(df, n_design, metric)
        mdf = run_morris(problem, X, Y, num_levels=args.levels)
        mdf.to_csv(os.path.join(out_dir, f"morris_{metric}.csv"), index=False)
        plots.plot_morris(mdf, metric, os.path.join(out_dir, f"morris_{metric}.png"))
        mx = float(mdf["mu_star"].max())
        if mx > 0:
            for _, r in mdf.iterrows():
                screening[r["path"]] = max(screening.get(r["path"], 0.0), r["mu_star"] / mx)

    ranked = sorted(screening.items(), key=lambda kv: kv[1], reverse=True)
    selected = [p for p, _ in ranked[:args.top_k]]
    with open(os.path.join(out_dir, "selected_params.json"), "w") as f:
        json.dump({"selected": selected, "screening_scores": dict(ranked)}, f, indent=2)

    print(f"\nTop {len(selected)} survivors (by max-normalized mu* across headline metrics):")
    for i, p in enumerate(selected, 1):
        print(f"  {i:>2}. {p:<52} {screening[p]:.3f}")
    print(f"\nSelected -> {os.path.join(out_dir, 'selected_params.json')}")


# ---------------------------------------------------------------------------
# Sobol stage
# ---------------------------------------------------------------------------

def run_sobol_stage(args):
    name = _spec_name("sobol", args.pilot)
    if args.params:
        selected = args.params
    else:
        sel_path = os.path.join(GSA_OUTPUT_DIR, _spec_name("morris", args.pilot), "selected_params.json")
        with open(sel_path) as f:
            selected = json.load(f)["selected"]
    problem = problem_from_selected(selected)
    print(f"\n=== Sobol: {problem['num_vars']} params, N={args.samples}, "
          f"second_order={args.second_order} ===")
    X = sobol_sample.sample(problem, args.samples, calc_second_order=args.second_order, seed=args.seed)
    combos = _tag(combos_from_sample(X, problem["names"]))
    print(f"Design: {len(X)} points x {args.reps} reps = {len(X) * args.reps} runs")
    spec = build_spec(name, args.grid_id, args.reps, args.steps, args.agents, args.workers)
    run_sweep(spec, combos=combos)
    _save_design(os.path.join(GSA_OUTPUT_DIR, name), X, problem)
    analyze_sobol(name, args)


def analyze_sobol(name, args):
    out_dir = os.path.join(GSA_OUTPUT_DIR, name)
    X, problem = _load_design(out_dir)
    df = load_results(name, output_dir=GSA_OUTPUT_DIR)
    n_design = len(X)

    print("\n--- Sobol indices + noise floor (per metric) ---")
    for metric in _present_metrics(df, ALL_GSA_METRICS):
        Y = align_mean_output(df, n_design, metric)
        if float(np.nanstd(Y)) == 0.0:
            print(f"  [skip    ] {metric:<32} constant output (no variance to attribute)")
            continue
        try:
            sdf = run_sobol(problem, Y, calc_second_order=args.second_order)
        except Exception as exc:
            # SALib raises on (near-)constant / degenerate outputs where entire
            # Saltelli blocks coincide (e.g. campy_extinct: campy essentially
            # never goes extinct). Sobol indices are undefined there - skip it.
            print(f"  [skip    ] {metric:<32} Sobol undefined (degenerate output: {type(exc).__name__})")
            continue
        sdf.to_csv(os.path.join(out_dir, f"sobol_{metric}.csv"), index=False)
        plots.plot_sobol(sdf, metric, os.path.join(out_dir, f"sobol_{metric}.png"))
        nf = noise_floor(df, metric, args.reps)
        top = sdf.iloc[0]
        conf_ratio = (top["ST_conf"] / top["ST"]) if top["ST"] > 0 else float("nan")
        tag = "HEADLINE" if metric in HEADLINE_METRICS else "also"
        print(f"  [{tag:8}] {metric:<32} top={top['path'].split('.')[-1][:24]:<24} "
              f"ST={top['ST']:.3f}  ST_conf/ST={conf_ratio:.2f}  snr={nf['snr']:.1f}"
              + (f"  bimodal={nf['n_bimodal_points']}" if nf['n_bimodal_points'] else ""))
    print(f"\nSobol CSVs + plots -> {out_dir}")


# ---------------------------------------------------------------------------
# Export importance
# ---------------------------------------------------------------------------

# Tier thresholds on Sobol ST (max across headline metrics): the fraction of a
# headline output's variance a parameter accounts for (incl. interactions).
TIER_HIGH = 0.30      # dominates at least one headline output
TIER_MEDIUM = 0.10    # a moderate driver
# below TIER_MEDIUM, or never analyzed by Sobol (screened out by Morris) -> "low"


def _tier(st_max):
    if st_max is None:
        return "low"          # screened out by Morris -> low-sensitivity by construction
    if st_max >= TIER_HIGH:
        return "high"
    if st_max >= TIER_MEDIUM:
        return "medium"
    return "low"


def export_importance(args):
    import pandas as pd
    from webapp.parameter_registry import REGISTRY

    sobol_dir = os.path.join(GSA_OUTPUT_DIR, _spec_name("sobol", args.pilot))
    morris_dir = os.path.join(GSA_OUTPUT_DIR, _spec_name("morris", args.pilot))

    # Sobol ST per headline metric for the analyzed (survivor) params.
    sobol_st = {}   # path -> {metric: ST}
    for metric in HEADLINE_METRICS:
        p = os.path.join(sobol_dir, f"sobol_{metric}.csv")
        if not os.path.exists(p):
            continue
        for _, r in pd.read_csv(p).iterrows():
            sobol_st.setdefault(r["path"], {})[metric] = float(r["ST"])

    # Morris screening scores (all params).
    screening = {}
    sel_path = os.path.join(morris_dir, "selected_params.json")
    if os.path.exists(sel_path):
        screening = json.load(open(sel_path)).get("screening_scores", {})

    # One record per GSA param. Tier is assigned from Sobol ST (survivors);
    # screened-out params are low by construction. ST and Morris score are kept
    # as SEPARATE columns - they are different scales and must not be conflated.
    records = []
    for m in [q for q in REGISTRY if q.path in {p.path for p in _all_gsa_paths()}]:
        per_metric = sobol_st.get(m.path, {})
        st_max = max(per_metric.values()) if per_metric else None
        records.append({
            "path": m.path, "label": m.label, "category": m.category,
            "method": "sobol" if per_metric else "morris",
            "sensitivity_tier": _tier(st_max),
            "sobol_ST_max": st_max,
            "per_metric_ST": per_metric,
            "morris_screening_score": float(screening.get(m.path, 0.0)),
        })

    # Order: tier (high>medium>low); within a tier, Sobol-analyzed params first
    # by ST desc, then screened-out params by Morris score desc. (Intra-"low"
    # order is cosmetic - the whole tier is "de-emphasize".)
    tier_order = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(records, key=lambda r: (
        tier_order[r["sensitivity_tier"]],
        0 if r["method"] == "sobol" else 1,
        -(r["sobol_ST_max"] if r["sobol_ST_max"] is not None else -1.0),
        -r["morris_screening_score"],
    ))
    for rank, r in enumerate(ordered, 1):
        r["sensitivity_rank"] = rank

    os.makedirs(GSA_OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(GSA_OUTPUT_DIR, "importance.json"), "w") as f:
        json.dump({r["path"]: r for r in ordered}, f, indent=2)
    pd.DataFrame(ordered)[
        ["sensitivity_rank", "sensitivity_tier", "path", "label", "category",
         "method", "sobol_ST_max", "morris_screening_score"]
    ].to_csv(os.path.join(GSA_OUTPUT_DIR, "importance_ranked.csv"), index=False)

    # Paste-ready ParamMeta snippets (static bake-in, like evidence_tier).
    with open(os.path.join(GSA_OUTPUT_DIR, "sensitivity_snippets.txt"), "w") as f:
        for r in ordered:
            f.write(f'{r["path"]}: sensitivity_tier="{r["sensitivity_tier"]}", '
                    f'sensitivity_rank={r["sensitivity_rank"]}\n')

    counts = {t: sum(1 for r in ordered if r["sensitivity_tier"] == t) for t in ("high", "medium", "low")}
    print(f"\nImportance tiers: HIGH={counts['high']}  MEDIUM={counts['medium']}  LOW={counts['low']}")
    print("Ranked (High + Medium tiers):")
    for r in ordered:
        if r["sensitivity_tier"] == "low":
            continue
        st = f"{r['sobol_ST_max']:.3f}" if r["sobol_ST_max"] is not None else "  -  "
        print(f"  {r['sensitivity_rank']:>2}. [{r['sensitivity_tier']:6}] ST={st}  {r['path']}")
    print(f"\n-> {os.path.join(GSA_OUTPUT_DIR, 'importance.json')}")
    print(f"-> {os.path.join(GSA_OUTPUT_DIR, 'importance_ranked.csv')}")


def _all_gsa_paths():
    return gsa_params()


# ---------------------------------------------------------------------------
# Plot-only
# ---------------------------------------------------------------------------

def plot_only(args):
    for stem, analyze in (("morris", None), ("sobol", None)):
        name = _spec_name(stem, args.pilot)
        out_dir = os.path.join(GSA_OUTPUT_DIR, name)
        if not os.path.exists(os.path.join(out_dir, "problem.json")):
            continue
        if stem == "morris":
            analyze_morris(name, args)
        else:
            analyze_sobol(name, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Global sensitivity analysis for SPRINGS-ABM.")
    p.add_argument("stage", choices=["morris", "sobol", "export", "plot"])
    p.add_argument("-g", "--grid-id", default=DEFAULT_GRID_ID)
    p.add_argument("--samples", type=int, default=None,
                   help="Morris: N trajectories. Sobol: N base samples. Default depends on stage/--pilot.")
    p.add_argument("-r", "--reps", type=int, default=None)
    p.add_argument("-s", "--steps", type=int, default=None)
    p.add_argument("-n", "--agents", type=int, default=None)
    p.add_argument("--levels", type=int, default=4, help="Morris num_levels (default 4)")
    p.add_argument("--top-k", type=int, default=12, help="Morris: survivors to pass to Sobol")
    p.add_argument("--second-order", action="store_true", help="Sobol: also compute S2 (N*(2k+2) cost)")
    p.add_argument("--params", nargs="+", default=None, help="Sobol: explicit param paths (skip morris selection)")
    p.add_argument("--seed", type=int, default=12345, help="SALib sampler seed")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pilot", action="store_true", help="Fast smoke test: few samples/reps, short/small runs.")
    args = p.parse_args()

    # Stage-specific defaults (full vs pilot).
    if args.stage == "morris":
        args.samples = args.samples if args.samples is not None else (4 if args.pilot else 10)
        args.reps = args.reps if args.reps is not None else (3 if args.pilot else 8)
    else:  # sobol
        args.samples = args.samples if args.samples is not None else (16 if args.pilot else 256)
        args.reps = args.reps if args.reps is not None else (2 if args.pilot else 5)
    args.steps = args.steps if args.steps is not None else (60 if args.pilot else 250)
    args.agents = args.agents if args.agents is not None else (800 if args.pilot else 4000)

    if args.pilot:
        print("*** PILOT MODE: reduced samples/reps/steps/agents - pipeline + timing check only. ***")

    if args.stage == "morris":
        run_morris_stage(args)
    elif args.stage == "sobol":
        run_sobol_stage(args)
    elif args.stage == "export":
        export_importance(args)
    elif args.stage == "plot":
        plot_only(args)


if __name__ == "__main__":
    main()

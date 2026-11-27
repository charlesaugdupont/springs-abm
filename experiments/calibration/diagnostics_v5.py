# experiments/calibration/diagnostics_v5.py
"""
Paper 1, analyses 1 and 5: diagnostics of campaign v5's history matching, and
variance-based sensitivity indices of its emulators.

    python -m experiments.calibration.diagnostics_v5
    python -m experiments.calibration.diagnostics_v5 --validation PATH   # the adopted set's validation runs
    python -m experiments.calibration.diagnostics_v5 --skip-emulators    # only the tables without emulators

Reads the waves' runs (experiments/outputs/history_matching_v5/wave*/results.parquet,
and wave*_meta.json and wave*_design.json beside them), the calibrated parameter sets
(experiments/scenarios/param_sets.json) and their verification runs
(.../history_matching_v5/verify_v5*/results.parquet), and writes, to
.../history_matching_v5/diagnostics/:

  nroy.csv        the share of the search box not ruled out before each wave (the waves' meta)
  design_ranges.csv  each wave's design points on the box's unit scale (0 a parameter's lower bound,
                  1 its upper, log-scaled parameters on a log scale): their 2.5, 50 and 97.5%
                  quantiles per parameter
  sets_box.csv    the 11 calibrated sets on the same scale
  fit_sets.csv    the 11 sets' verification runs against the 41 targets: the standardised error
                  (mean - target) / sqrt(sd_obs^2 + sd_md^2 + var / n) on the calibration scale, the
                  implausibility of the selection rule with its sign
  validation.csv  the emulators of waves 1-3 (those that designed wave 4) against wave 4's runs,
                  per target: the standardised errors (y - mu) / sd, sd the emulator's predictive
                  SD of a two-run mean (its nugget included), their root mean square (1 if the
                  emulator's uncertainty is right), the shares beyond 2 and 3, and the share of
                  wave 4's variance the emulator explains (R^2)
  sobol.csv       first-order and total Sobol indices of each target's emulator mean (all waves)
                  over the search box, for the inputs its emulator uses (Saltelli's scheme, SALib)
  fit.csv         with --validation: the adopted set's runs against the 41 targets (implausibility
                  with the runs' own variance), as in the campaign's verification

The emulators are cached, one file per target (emulators_w1_3/, emulators_all/), so an interrupted
run resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle

import numpy as np
import pandas as pd

from experiments.calibration import history_matching as hm

SOBOL_N = 1024
SOBOL_SEED = 61_001


def _out_dir() -> str:
    d = os.path.join(hm.OUT_DIR, "diagnostics")
    os.makedirs(d, exist_ok=True)
    return d


def _cached_emulators(summary: pd.DataFrame, name: str, n_jobs: int | None = None) -> dict:
    """The campaign's emulators (as history_matching.fit_emulators fits them: one GP per target on
    the inputs Campaign.emulator_inputs gives it), one cached file per target, so that an
    interrupted run keeps the emulators already fitted."""
    d = os.path.join(_out_dir(), f"emulators_{name}")
    os.makedirs(d, exist_ok=True)
    points = sorted(summary["gsa.design_id"])
    current = summary[summary["wave"] >= hm.FIRST_CURRENT_WAVE]
    ems = {}
    for t in hm.TARGETS_HM:
        path = os.path.join(d, f"{t.metric}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("points") == points:
                if cached["emulator"] is not None:
                    ems[t.metric] = cached["emulator"]
                continue
        X, y = hm._training_data(current if t.metric in hm.TREATMENT_DEPENDENT else summary, t)
        cols = None
        if hm.EMULATOR_INPUTS is not None:
            cols = [hm.PATHS.index(p) for p in hm.EMULATOR_INPUTS(t.metric)]
            X = X[:, cols]
        gp = hm._fit_gp(X, y)
        em = None if gp is None else (gp if cols is None else hm.SubsetGP(gp, cols))
        with open(path, "wb") as f:
            pickle.dump({"points": points, "emulator": em}, f)
        if em is not None:
            ems[t.metric] = em
        print(f"  {name}: {t.metric}", flush=True)
    return ems


def nroy_table() -> pd.DataFrame:
    rows = []
    for name in sorted(os.listdir(hm.OUT_DIR)):
        if name.startswith("wave") and name.endswith("_meta.json"):
            m = json.load(open(os.path.join(hm.OUT_DIR, name)))
            rows.append({"wave": m["wave"], "points": m["n"], "runs_per_point": m["reps"],
                         "nroy_share_before": m["nroy_fraction_before"], "criterion": m["criterion"]})
    return pd.DataFrame(rows).sort_values("wave")


def design_ranges_table() -> pd.DataFrame:
    """Where each wave's design points lie in the search box. Wave 1 is a Latin hypercube of the whole
    box; each later wave drew its points from the region the earlier waves' emulators had not ruled
    out (wave 4's also meeting the campaign's design rules), half of them from that region's core
    (hm.sample_nroy), so the points' spread shows how far the region extends along each parameter."""
    rows = []
    for name in sorted(os.listdir(hm.OUT_DIR)):
        if not (name.startswith("wave") and name.endswith("_design.json")):
            continue
        d = json.load(open(os.path.join(hm.OUT_DIR, name)))
        U = hm.to_unit(np.array([[float(c[p]) for p in hm.PATHS] for c in d["combos"]]))
        for j, p in enumerate(hm.PATHS):
            q = np.quantile(U[:, j], [0.025, 0.5, 0.975])
            rows.append({"wave": int(d["wave"]), "parameter": p, "points": len(U), "q025": float(q[0]),
                         "q50": float(q[1]), "q975": float(q[2])})
    return pd.DataFrame(rows).sort_values(["wave"], kind="stable").reset_index(drop=True)


def _param_sets(path: str | None = None) -> dict:
    from experiments.scenarios.param_sets import SETS_JSON
    with open(path or SETS_JSON) as f:
        return json.load(f)


def sets_table(path: str | None = None) -> pd.DataFrame:
    """The calibrated parameter sets (experiments/scenarios/param_sets.json) on the box's unit scale."""
    from experiments.scenarios.param_sets import ADOPTED
    rows = []
    for s in _param_sets(path)["sets"]:
        u = hm.to_unit(np.array([[float(s["values"][p]) for p in hm.PATHS]]))[0]
        rows += [{"set": s["name"], "adopted": s["name"] == ADOPTED, "parameter": p,
                  "value": float(s["values"][p]), "unit": float(u[j])} for j, p in enumerate(hm.PATHS)]
    return pd.DataFrame(rows)


def fit_sets_table(path: str | None = None) -> pd.DataFrame:
    """The calibrated sets' verification runs (8 seeds apart from the waves', without the vaccine, as
    calibrated) against the 41 targets, computed as the selection rule did (hm.summarise and
    hm.simulated_implausibility): signed is (mean - target) / sqrt(sd_obs^2 + sd_md^2 + var / n) on
    the target's calibration scale, its size the implausibility. Each set's largest implausibility
    must equal the I_max recorded in param_sets.json."""
    from experiments.scenarios.param_sets import ADOPTED
    rows = []
    for s in _param_sets(path)["sets"]:
        runs = pd.read_parquet(os.path.join(hm.OUT_DIR, s["verification"], "results.parquet"))
        g = runs[runs["gsa.design_id"] == f"{s['verification']}_{s['name']}"].assign(wave=99)
        if g.empty:
            raise ValueError(f"no verification runs for {s['name']} in {s['verification']}")
        r = hm.simulated_implausibility(hm.summarise(g)).iloc[0]
        if not math.isclose(float(r["I_max"]), float(s["I_max"]), rel_tol=1e-9):
            raise ValueError(f"{s['name']}: I_max {r['I_max']} from its runs, {s['I_max']} recorded")
        for t in hm.TARGETS_HM:
            var = float(r[t.metric + "__var"]) if np.isfinite(r[t.metric + "__var"]) else 0.0
            sd = math.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + var / float(r["n_rep"]))
            back = (lambda v: math.exp(v) - t.eps) if t.transform == "log" else (lambda v: v)
            rows.append({"set": s["name"], "adopted": s["name"] == ADOPTED, "target": t.metric,
                         "runs": int(r["n_rep"]), "model": back(float(r[t.metric])), "observed": back(t.z),
                         "signed": (float(r[t.metric]) - t.z) / sd, "implausibility": float(r["I__" + t.metric])})
    return pd.DataFrame(rows)


def validation_table(summary: pd.DataFrame, ems: dict, test_wave: int) -> pd.DataFrame:
    test = summary[summary["wave"] == test_wave]
    X = hm.to_unit(test[hm.PATHS].to_numpy(dtype=float))
    rows = []
    for t in hm.TARGETS_HM:
        gp = ems.get(t.metric)
        y = test[t.metric].to_numpy(dtype=float)
        ok = np.isfinite(y)
        if gp is None or ok.sum() < 5:
            continue
        mu, sd = gp.predict(X[ok], return_std=True)
        z = (y[ok] - mu) / sd
        sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
        rows.append({"target": t.metric, "n": int(ok.sum()), "rms_standardised_error": float(np.sqrt(np.mean(z ** 2))),
                     "share_beyond_2": float(np.mean(np.abs(z) > 2)), "share_beyond_3": float(np.mean(np.abs(z) > 3)),
                     "mean_standardised_error": float(np.mean(z)),
                     "r2": 1.0 - float(np.sum((y[ok] - mu) ** 2)) / sst if sst > 0 else float("nan"),
                     "tolerance_sd": math.sqrt(t.sd_obs ** 2 + t.sd_md ** 2), "median_emulator_sd": float(np.median(sd))})
    return pd.DataFrame(rows)


def sobol_table(ems: dict, n: int = SOBOL_N, seed: int = SOBOL_SEED) -> pd.DataFrame:
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample
    rows = []
    for t in hm.TARGETS_HM:
        em = ems.get(t.metric)
        if em is None:
            continue
        cols = em.cols if isinstance(em, hm.SubsetGP) else list(range(len(hm.PATHS)))
        gp = em.gp if isinstance(em, hm.SubsetGP) else em
        names = [hm.PATHS[c] for c in cols]
        problem = {"num_vars": len(cols), "names": names, "bounds": [[0.0, 1.0]] * len(cols)}
        U = sobol_sample.sample(problem, n, calc_second_order=False, seed=seed)
        Y = np.concatenate([gp.predict(U[a:a + 20_000]) for a in range(0, len(U), 20_000)])
        si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False, seed=seed)
        for j, name in enumerate(names):
            rows.append({"target": t.metric, "parameter": name, "S1": float(si["S1"][j]),
                         "S1_conf": float(si["S1_conf"][j]), "ST": float(si["ST"][j]),
                         "ST_conf": float(si["ST_conf"][j])})
    return pd.DataFrame(rows)


def fit_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rows = []
    for t in hm.TARGETS_HM:
        if t.metric not in df:
            continue
        y = t.tx(df[t.metric].to_numpy(dtype=float))
        y = y[np.isfinite(y)]
        var = float(y.var(ddof=1) / len(y)) if len(y) > 1 else 0.0
        mean = float(y.mean())
        imp = abs(mean - t.z) / math.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + var)
        back = (math.exp(mean) - t.eps) if t.transform == "log" else mean
        target = (math.exp(t.z) - t.eps) if t.transform == "log" else t.z
        rows.append({"target": t.metric, "runs": len(y), "model": back, "observed": target,
                     "implausibility": imp, "signed": (mean - t.z) / math.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + var)})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--validation", default=None, help="the adopted set's validation runs (results.parquet)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--sobol-n", type=int, default=SOBOL_N)
    ap.add_argument("--targets", nargs="*", default=None, help="only these targets (tests)")
    ap.add_argument("--skip-emulators", action="store_true",
                    help="only the tables that need no emulator (nroy, design ranges, sets, their fit)")
    args = ap.parse_args(argv)
    hm.use_campaign("v5")
    if args.targets:
        hm.TARGETS_HM = [t for t in hm.TARGETS_HM if t.metric in set(args.targets)]
    out = _out_dir() if not args.targets else os.path.join(_out_dir(), "test")
    os.makedirs(out, exist_ok=True)
    nroy_table().to_csv(os.path.join(out, "nroy.csv"), index=False)
    design_ranges_table().to_csv(os.path.join(out, "design_ranges.csv"), index=False)
    sets_table().to_csv(os.path.join(out, "sets_box.csv"), index=False)
    fit_sets_table().to_csv(os.path.join(out, "fit_sets.csv"), index=False)
    print("nroy, design ranges, sets and their fit written", flush=True)
    if args.skip_emulators:
        if args.validation:
            fit_table(args.validation).to_csv(os.path.join(out, "fit.csv"), index=False)
        print(f"diagnostics in {out}")
        return
    summary = hm.summarise(hm.load_all_waves())
    last = int(summary["wave"].max())
    ems_prev = _cached_emulators(summary[summary["wave"] < last], f"w1_{last - 1}", args.n_jobs)
    validation_table(summary, ems_prev, last).to_csv(os.path.join(out, "validation.csv"), index=False)
    print("validation written", flush=True)
    ems = _cached_emulators(summary, "all", args.n_jobs)
    sobol_table(ems, args.sobol_n).to_csv(os.path.join(out, "sobol.csv"), index=False)
    print("sobol written", flush=True)
    if args.validation:
        fit_table(args.validation).to_csv(os.path.join(out, "fit.csv"), index=False)
    print(f"diagnostics in {out}")


if __name__ == "__main__":
    main()

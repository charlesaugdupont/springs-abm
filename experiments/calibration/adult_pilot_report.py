# experiments/calibration/adult_pilot_report.py
"""
Analysis of the adult-transmission pilots (history_matching campaigns
"adult_pilot" and "adult_pilot_cap"; see ADULT_PILOT_PARAMS and
ADULT_PILOT_CAP_PARAMS there for the question and the boxes).

For rotavirus and Cryptosporidium separately - each emulated on its own
parameters and any shared one (pilot 2: the contact cap) - it reports:

  1. emulator quality: 5-fold cross-validated RMSE of each target's emulator,
     next to the target's own SD (an emulator is useful where its error is
     well below what the target tolerates);
  2. the emulated non-implausible fraction of the pathogen's 2-4 dimensional
     box, judged on the child targets alone, on the carriage ratio alone, and
     on both - the pilot's question is whether "both" is non-empty;
  3. within the child-only region: the adult : child carriage ratio the model
     produces (if the region never reaches ~0.5-0.7, matching children
     requires adults to drive transmission);
  4. within the joint region: the admissible range of each parameter, which
     sets the box for the follow-up calibration;
  5. the best simulated runs (no emulator), with their raw outputs;

and, when the pathogens share a parameter, the region where both fit at
once and the shared parameter's range there.

Usage
-----
    python -m experiments.calibration.adult_pilot_report --campaign adult_pilot_cap [--workers 7]

Decision rule, stated before the pilot ran: a pathogen passes if its joint
region (every child target and the carriage ratio within 3 SD, max rule) is
non-empty AND at least 3 simulated runs have a joint max implausibility
below 3.5 - so the conclusion does not rest on the emulator alone - with,
for rotavirus, at most 15% of infections imported (v3's
MAX_IMPORTED_FRACTION: the town must sustain it, not importation). If both
pass, the follow-up is a short joint calibration with the adult parameters
added to the v3 box; if one fails, the report shows which child targets
conflict with plausible adult carriage, and the fix is structural (who
children meet), not more calibration. Pilot 2 adds: both must pass at a
common value of the shared parameter - the combined emulated region is
non-empty and at least 3 simulated runs pass for both pathogens at once.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import qmc

from experiments.calibration import history_matching as hm

CARRIAGE = {p: f"{p}_adult_to_child_carriage_ratio_post_burnin" for p in ("rota", "crypto")}
SHOW = ["adult_to_child_carriage_ratio_post_burnin", "adult_prevalence_post_burnin",
        "child_carriage_12_59m_post_burnin", "adult_infectious_weight_share_post_burnin",
        "infections_per_child_year_0_24m", "obs_infections_per_child_year_0_24m",
        "obs_attributable_episodes_per_child_year_0_24m", "imported_fraction_post_burnin"]


def _cv_rmse(summary: pd.DataFrame, t: hm.HMTarget, cols: list[int], k: int = 5) -> float:
    X = hm.to_unit(summary[hm.PATHS].to_numpy(dtype=float))[:, cols]
    y = summary[t.metric].to_numpy(dtype=float)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < 2 * k:
        return float("nan")
    idx = np.random.default_rng(0).permutation(len(y))
    err = []
    for f in range(k):
        test = idx[f::k]
        train = np.setdiff1d(idx, test)
        gp = hm._fit_gp(X[train], y[train])
        if gp is None:
            return float("nan")
        err.append(gp.predict(X[test]) - y[test])
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--candidates", type=int, default=200_000)
    ap.add_argument("--out-dir", default=None, help="pilot output folder (default: the campaign's)")
    ap.add_argument("--campaign", choices=("adult_pilot", "adult_pilot_cap"), default="adult_pilot_cap")
    ap.add_argument("--select-both", type=int, default=0, metavar="K",
                    help="write K points the emulators place where both pathogens fit, one per equal-log-width "
                         "bin of the shared parameter (lowest max implausibility in each), to "
                         "<out dir>/both_candidates.json for history_matching --verify")
    args = ap.parse_args()
    camp = hm.use_campaign(args.campaign)
    if args.out_dir:
        hm.OUT_DIR = args.out_dir
    df = hm.load_all_waves()
    if df.empty:
        raise SystemExit(f"no pilot results in {hm.OUT_DIR}")
    summary = hm.summarise(df)
    print(f"{len(summary)} design points, {len(df)} runs\n")
    sim = hm.simulated_implausibility(summary)
    ems = hm.fit_emulators(summary, n_jobs=args.workers)
    U = qmc.LatinHypercube(d=len(hm.PATHS), seed=7).random(args.candidates)
    I = hm.emulated_implausibility(ems, U, n_jobs=args.workers)
    names = [t.metric for t in hm.TARGETS_HM]
    V = pd.DataFrame(hm.from_unit(U), columns=hm.PATHS)
    joint_by, sim_ok_by = {}, {}

    for p in ("rota", "crypto"):
        params = camp.emulator_inputs(CARRIAGE[p])
        cols = [hm.PATHS.index(q) for q in params]
        child = [j for j, m in enumerate(names) if m.startswith(p + "_") and m != CARRIAGE[p]]
        carr = names.index(CARRIAGE[p])
        print("=" * 100)
        print(f"{p.upper()}  ({len(child)} child targets + adult:child carriage ratio; parameters: "
              + ", ".join(q.split('.')[1] for q in params) + ")")

        print("\n  emulator 5-fold CV RMSE vs target SD (transformed scale):")
        for j in child + [carr]:
            t = hm.TARGETS_HM[j]
            rmse = _cv_rmse(summary, t, cols)
            sd = float(np.hypot(t.sd_obs, t.sd_md))
            verdict = "-" if not np.isfinite(rmse) else ("ok" if rmse < sd else "POOR")
            print(f"    {t.metric:58s} RMSE {rmse:6.3f}   target SD {sd:5.3f}   {verdict}")

        Ic = I[:, child]
        child_max, child_second = Ic.max(axis=1), np.sort(Ic, axis=1)[:, -2]
        joint_max = np.maximum(child_max, I[:, carr])
        print("\n  emulated non-implausible fraction of the box (I < 3):")
        print(f"    child targets, max rule        {np.mean(child_max < 3):.4f}")
        print(f"    child targets, second-max rule {np.mean(child_second < 3):.4f}")
        print(f"    carriage ratio alone           {np.mean(I[:, carr] < 3):.4f}")
        print(f"    child targets AND carriage     {np.mean(joint_max < 3):.4f}")

        gp = ems.get(CARRIAGE[p])
        region = child_max < 3
        if gp is not None and region.any():
            mu = gp.predict(U[region][:20000])
            r = np.exp(mu) - 0.02
            print(f"\n  carriage ratio where every child target fits ({int(region.sum())} candidates): "
                  f"median {np.median(r):.2f}, 5-95% {np.quantile(r, .05):.2f}-{np.quantile(r, .95):.2f}"
                  f"  (target {hm.ADULT_CARRIAGE_TARGETS[CARRIAGE[p]][0]})")
        joint = joint_max < 3
        if joint.any():
            print(f"\n  admissible ranges, child targets AND carriage ({int(joint.sum())} candidates):")
            for q in params:
                lo, hi, _ = hm.PARAMS[q]
                v = V.loc[joint, q]
                print(f"    {q:52s} box [{lo:.4g}, {hi:.4g}]  ->  {v.min():.4g} .. {v.max():.4g}"
                      f"  (5-95% {v.quantile(.05):.4g} .. {v.quantile(.95):.4g})")
        else:
            worst = pd.Series(Ic[np.argsort(joint_max)[:2000]].mean(axis=0), index=[names[j] for j in child])
            print("\n  no joint region; at the 2000 least implausible candidates the child targets that fail most:")
            for m, v in worst.sort_values(ascending=False).head(5).items():
                print(f"    {m:58s} mean I {v:.2f}")

        icols = ["I__" + names[j] for j in child + [carr]]
        s = sim.copy()
        s["I_joint"] = s[icols].max(axis=1)
        s["I_child"] = s[["I__" + names[j] for j in child]].max(axis=1)
        s = s.sort_values("I_joint")
        raw_cols = [f"{p}_{k}" for k in SHOW if f"{p}_{k}" in df]
        raw = df.groupby("gsa.design_id")[raw_cols].mean()
        ok = s["I_joint"] < 3.5
        if p == "rota":
            imp = raw["rota_imported_fraction_post_burnin"].reindex(s["gsa.design_id"]).to_numpy()
            ok &= imp <= hm.MAX_IMPORTED_FRACTION
        joint_by[p] = joint
        sim_ok_by[p] = set(s.loc[ok.to_numpy(), "gsa.design_id"])
        print(f"\n  simulated runs with joint max I < 3.5{' and <= 15% imported' if p == 'rota' else ''}: "
              f"{int(ok.sum())} (joint < 3: {(s['I_joint'] < 3).sum()}); child-only max I < 3: "
              f"{(s['I_child'] < 3).sum()}  ->  {'PASS' if ok.sum() >= 3 and joint.any() else 'FAIL'}")
        top = s[["gsa.design_id", "I_joint", "I_child"] + params].head(6).join(raw, on="gsa.design_id")
        top.columns = [c.replace(f"pathogens[{p}].", "").replace("steering_parameters.", "")
                       .replace(f"{p}_", "").replace("_post_burnin", "") for c in top.columns]
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(top.round(4).to_string(index=False))
        print()

    shared = [q for q in hm.PATHS if not q.startswith("pathogens[")]
    if shared:
        both = joint_by["rota"] & joint_by["crypto"]
        sim_both = sim_ok_by["rota"] & sim_ok_by["crypto"]
        print("=" * 100)
        print(f"BOTH PATHOGENS at a common {', '.join(q.split('.')[-1] for q in shared)}")
        print(f"  emulated fraction of the box where both fit (child targets AND carriage): {both.mean():.4f}")
        if both.any():
            for q in hm.PATHS:
                lo, hi, _ = hm.PARAMS[q]
                v = V.loc[both, q]
                print(f"    {q:52s} box [{lo:.4g}, {hi:.4g}]  ->  {v.min():.4g} .. {v.max():.4g}"
                      f"  (5-95% {v.quantile(.05):.4g} .. {v.quantile(.95):.4g})")
        print(f"  simulated runs passing for both: {len(sim_both)} {sorted(sim_both)}  ->  "
              f"{'PASS' if len(sim_both) >= 3 and both.any() else 'FAIL'}")
        if args.select_both and both.any():
            import json
            import os
            q = shared[0]
            lo, hi, _ = hm.PARAMS[q]
            edges = np.geomspace(lo, hi, args.select_both + 1)
            worst = np.max(I, axis=1)
            picks = []
            for k in range(args.select_both):
                inbin = both & (V[q].to_numpy() >= edges[k]) & (V[q].to_numpy() < edges[k + 1])
                if inbin.any():
                    i = np.flatnonzero(inbin)[np.argmin(worst[inbin])]
                    picks.append({"gsa.design_id": f"b{k}", **{c: float(V.loc[i, c]) for c in hm.PATHS},
                                  "emulated_max_I": float(worst[i])})
            path = os.path.join(hm.OUT_DIR, "both_candidates.json")
            with open(path, "w") as f:
                json.dump(picks, f, indent=1)
            print(f"\n  {len(picks)} candidates where both fit -> {path}")
            with pd.option_context("display.width", 250, "display.max_columns", 30):
                show = pd.DataFrame(picks).set_index("gsa.design_id")
                show.columns = [c.split(".")[-1] if c.startswith("steering") else c.replace("pathogens[", "").replace("].", ".")
                                for c in show.columns]
                print(show.round(4).to_string())


if __name__ == "__main__":
    main()

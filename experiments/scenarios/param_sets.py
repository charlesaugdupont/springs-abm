# experiments/scenarios/param_sets.py
"""
Step 3 of the revision: the headline contrasts over calibrated parameter
sets, so that each headline effect is reported with its spread over the parameter values the calibration
admits, not only over seeds.

The sets
  - e7, the adopted set (the config defaults, at 3 significant figures); its runs are step 2's (the first
    8 seeds of each scenario).
  - e3, e5, e0, e1: the v5 candidates verified on the calibration's verification seeds (10023-10030,
    1,460 days; verify_v5, verify_v5b) with every target within about 3 SD (max I 2.97, 2.99, 3.01, 3.06).
  - those of e9-e23 that meet the same rule (verify_v5c). They come from the same emulator selection as
    e0-e8 (`history_matching --select`, the same seed and rules), run with n = 24 instead of 9
    (outputs/history_matching_v5_step3; its first nine reproduce e0-e8 exactly).
  Rule (stated before verify_v5c ran): a set is kept if the largest implausibility of the 41 targets is
  below 3.1 on the 8 verification seeds (the threshold that kept e0 and e1; with 8 seeds the worst
  target's I is uncertain by about +/-0.13) and rotavirus fades out in none of them. The kept sets and their
  values are written to param_sets.json, beside this file, and frozen with the code.

The scenarios (HEADLINE): each defined exactly as in step 2 (taken from its designs by label), 8 seeds
(61001-61008), 1,095 days. R0_none (no shock, no strategy: the model as it is) is the reference of every
contrast but 1B's and the strategy's.

The contrasts (CONTRASTS) and the rule (the plan's): a headline effect counts as robust if, in at least 80%
of the sets (the adopted one included, on the same 8 seeds), it has the same sign as at the adopted set and
exceeds 2 SE (paired over the 8 seeds). Only effects that exceed 2 SE at the adopted set (step 2, all its
seeds) are tested; the others are reported.

    python -m experiments.scenarios.param_sets verify --cores 2   # verify_v5c: e9-e23 x 8 seeds, and a control
    python -m experiments.scenarios.param_sets sets               # the rule; writes param_sets.json
    python -m experiments.scenarios.run ps --cores 2              # the scenarios over the kept sets but e7
    python -m experiments.scenarios.param_sets analyse --ps <results.parquet> --step2 <results.parquet> ...
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

LABEL = "gsa.design_id"
SET_KEY = "gsa.param_set"
SCENARIO_KEY = "gsa.scenario"
HM_V5_DIR = os.path.join("experiments", "outputs", "history_matching_v5")
STEP3_DIR = os.path.join("experiments", "outputs", "history_matching_v5_step3")
CANDIDATES_N24 = os.path.join(STEP3_DIR, "emulator_candidates_n24.json")
SETS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "param_sets.json")
ADOPTED = "e7"
#: the verifications behind e0-e8 (2026-11-04); the rule below is applied to them as to the new one
EARLIER_VERIFICATIONS = (("verify_v5", HM_V5_DIR), ("verify_v5b", HM_V5_DIR))
NEW = tuple(f"e{i}" for i in range(9, 24))
VERIFY_TAG = "verify_v5c"
CONTROL_TAG = "verify_v5c_control"
I_MAX_KEEP = 3.1
PS_REPS = 8
ROBUST_SHARE = 0.8

#: the scenarios run over the sets (labels of step 2's designs)
HEADLINE = (
    "R0_none",                                  # the reference: no shock, no strategy (2_rev)
    "vacc_none", "grid_v0.95_d616", "vacc_no_prior_infection_equivalent",            # 3A
    "neonatal", "booster_9m", "wash_q12",                                            # 3B
    "price_free", "credit_1d", "free_ors_zinc", "leap_q12_x1", "perceived_0.2",     # 1A
    "B1_s0.6_none", "B1_s0.6_stockout26w", "B4_campaign",                            # 1B
    "D-slow_none", "F-slow_none", "R0_adjustment", "D-slow_adjustment", "D-slow_none_noenv",   # 2
)
STEP2_PARTS = ("3a_rev", "3b_rev", "1a_rev", "1b_rev", "2_rev")


# ---------------------------------------------------------------------------
# Verification and the kept sets
# ---------------------------------------------------------------------------

def expand(values: dict) -> dict:
    """Tied paths ("a|b") to one entry per config field."""
    out = {}
    for k, v in values.items():
        for part in k.split("|"):
            out[part] = v
    return out


def _candidates() -> dict:
    with open(CANDIDATES_N24) as f:
        return {c[LABEL]: c for c in json.load(f)}


def verify_new(reps: int = 8, cores: int | None = None, control: bool = True):
    """verify_v5c: e9-e23 on the verification seeds, as e0-e8 were verified. First a control: the adopted
    e7 on the first seed must reproduce verify_v5b exactly (the same model, and the same application of a
    candidate's values, as when e0-e8 were verified), or the earlier verifications no longer hold."""
    import pandas as pd
    import experiments.calibration.history_matching as hm
    hm.use_campaign("v5")
    hm.OUT_DIR = STEP3_DIR
    hm.CHECKPOINT_EVERY = 4            # the campaign's 16 loses more runs when a batch stops
    cands = _candidates()
    os.makedirs(STEP3_DIR, exist_ok=True)
    if control:
        path = os.path.join(STEP3_DIR, "emulator_candidates_control_e7.json")
        with open(path, "w") as f:
            json.dump([cands[ADOPTED]], f, indent=1)
        hm.verify(1, cores, candidates_json=path, tag=CONTROL_TAG)
        mine = pd.read_parquet(os.path.join(STEP3_DIR, CONTROL_TAG, "results.parquet")).iloc[0]
        old = pd.read_parquet(os.path.join(HM_V5_DIR, "verify_v5b", "results.parquet"))
        old = old[(old[LABEL] == f"verify_v5b_{ADOPTED}") & (old["seed"] == int(mine["seed"]))].iloc[0]
        skip = {"run_id", LABEL}
        cols = [c for c in old.index if c in mine.index and c not in skip]
        bad = []
        for c in cols:
            a, b = mine[c], old[c]
            if isinstance(a, (list, np.ndarray)) or isinstance(b, (list, np.ndarray)):
                same = np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)
            elif isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
                same = (a == b) or (np.isnan(a) and np.isnan(b))
            else:
                same = a == b
            if not same:
                bad.append(c)
        print(f"control: {len(cols) - len(bad)} of {len(cols)} outputs identical to verify_v5b ({ADOPTED}, "
              f"seed {int(mine['seed'])})")
        if bad:
            raise RuntimeError(f"the control differs from verify_v5b in {len(bad)} outputs (e.g. {bad[:5]}): "
                               "the earlier verifications do not hold for this code")
    new = [cands[n] for n in NEW]
    path = os.path.join(STEP3_DIR, "emulator_candidates_e9_e23.json")
    with open(path, "w") as f:
        json.dump(new, f, indent=1)
    return hm.verify(reps, cores, candidates_json=path, tag=VERIFY_TAG)


def verification_table():
    """Every verified candidate (e0-e8, then e9-e23 once verify_v5c has run) with the rule applied."""
    import pandas as pd
    frames = []
    for tag, d in EARLIER_VERIFICATIONS + ((VERIFY_TAG, STEP3_DIR),):
        path = os.path.join(d, f"{tag}_implausibility.csv")
        if not os.path.exists(path):
            continue
        v = pd.read_csv(path)
        v["set"] = v[LABEL].str.replace(f"{tag}_", "", regex=False)
        v["verification"] = tag
        frames.append(v[["set", "verification", "I_max", "I_second", "rota_fadeout_reps"]])
    tab = pd.concat(frames, ignore_index=True)
    tab["kept"] = (tab["I_max"] < I_MAX_KEEP) & (tab["rota_fadeout_reps"] == 0)
    return tab.sort_values(["kept", "I_max"], ascending=[False, True]).reset_index(drop=True)


def select_sets(write: bool = True) -> dict:
    """The rule applied to every verification; the kept sets (the adopted one first) with their values."""
    tab = verification_table()
    cands = _candidates()
    tab = tab[tab["set"].isin(cands)]           # verify_v5's s0-s2 were simulated points, not candidates
    kept = tab[tab["kept"]]
    order = [ADOPTED] + [s for s in kept["set"] if s != ADOPTED]
    sets = []
    for name in order:
        r = kept[kept["set"] == name].iloc[0]
        sets.append({"name": name, "verification": r["verification"], "I_max": float(r["I_max"]),
                     "values": {k: float(v) for k, v in cands[name].items() if k != LABEL}})
    out = {"rule": f"max I < {I_MAX_KEEP} over the 41 targets on the 8 verification seeds, "
                   "and rotavirus fades out in none of them",
           "sets": sets,
           "excluded": [{"name": r["set"], "verification": r["verification"], "I_max": float(r["I_max"]),
                         "rota_fadeout_reps": int(r["rota_fadeout_reps"])} for _, r in tab[~tab["kept"]].iterrows()]}
    if write:
        with open(SETS_JSON, "w") as f:
            json.dump(out, f, indent=1)
    return out


def load_sets() -> list[dict]:
    if not os.path.exists(SETS_JSON):
        raise FileNotFoundError(f"{SETS_JSON}: run `python -m experiments.scenarios.param_sets sets` "
                                "after the verification")
    with open(SETS_JSON) as f:
        return json.load(f)["sets"]


# ---------------------------------------------------------------------------
# The scenarios over the sets
# ---------------------------------------------------------------------------

def scenarios() -> dict:
    """label -> (step 2 part, its scenario) for every HEADLINE label."""
    from experiments.scenarios import designs as D
    fns = {"3a_rev": D.exp3a_rev_combos, "3b_rev": D.exp3b_rev_combos, "1a_rev": D.exp1a_rev_combos,
           "1b_rev": D.exp1b_rev_combos, "2_rev": D.exp2_rev_combos}
    by: dict = {}
    for part, fn in fns.items():
        for c in fn():
            by.setdefault(c[LABEL], (part, c))
    missing = [lab for lab in HEADLINE if lab not in by]
    if missing:
        raise KeyError(f"headline scenarios missing from step 2's designs: {missing}")
    return {lab: by[lab] for lab in HEADLINE}


def ps_combos() -> list[dict]:
    """Every headline scenario with every kept set but the adopted one, set by set (8 seeds each). A
    scenario's own value of a calibrated parameter wins, and may only make care free (price_free)."""
    from experiments.orchestrator import REPS_KEY
    scen = scenarios()
    combos = []
    for s in load_sets():
        if s["name"] == ADOPTED:
            continue
        vals = expand(s["values"])
        for lab in HEADLINE:
            _, c = scen[lab]
            body = {k: v for k, v in c.items() if k not in (LABEL, REPS_KEY)}
            for k in set(body) & set(vals):
                if body[k] != 0.0:
                    raise ValueError(f"{lab} sets the calibrated {k} to {body[k]}: its value for another "
                                     "parameter set is not defined")
            combos.append({LABEL: f"{s['name']}|{lab}", SET_KEY: s["name"], SCENARIO_KEY: lab,
                           **vals, **body, REPS_KEY: PS_REPS})
    return combos


#: Paper 1: the scenarios run over the sets (3A's labels): the baseline outputs and the vaccine's effect
P1_HEADLINE = ("vacc_default", "vacc_none")


def p1_ps_combos() -> list[dict]:
    """Paper 1: P1_HEADLINE with every kept set but the adopted one, set by set (8 seeds each), as in the
    3A rerun; the adopted set's runs are p1_3a's first 8 seeds."""
    from experiments.orchestrator import REPS_KEY
    from experiments.scenarios import designs as D
    by = {c[LABEL]: c for c in D.exp3a_rev_combos()}
    combos = []
    for s in load_sets():
        if s["name"] == ADOPTED:
            continue
        vals = expand(s["values"])
        for lab in P1_HEADLINE:
            body = {k: v for k, v in by[lab].items() if k not in (LABEL, REPS_KEY)}
            if set(body) & set(vals):
                raise ValueError(f"{lab} sets a calibrated parameter: {sorted(set(body) & set(vals))}")
            combos.append({LABEL: f"{s['name']}|{lab}", SET_KEY: s["name"], SCENARIO_KEY: lab,
                           **vals, **body, REPS_KEY: PS_REPS})
    return combos


# ---------------------------------------------------------------------------
# The contrasts
# ---------------------------------------------------------------------------

def _windows() -> dict:
    """Weeks of the shocks and the stock-out (week w = days 7w to 7w+6)."""
    from experiments.scenarios import hooks as H
    wk = lambda d: int(d // 7)
    # 1B's stock-out in the reruns: 26 weeks from the end of the warm-up (designs.exp1b_rev_combos)
    return {"drought": (wk(H.DROUGHT_START_DAY), wk(H.DROUGHT_START_DAY + H.DROUGHT_DAYS)),
            "flood": (wk(H.FLOOD_START_DAY), wk(H.FLOOD_START_DAY + H.FLOOD_DAYS)),
            "stockout": (wk(H.SHOCK_START_DAY), wk(H.SHOCK_START_DAY + 182))}


def _series(r, key) -> np.ndarray:
    return np.asarray(r[key], dtype=float)


def _ratio(r, num, den, a, b=None) -> float:
    n, d = _series(r, num)[a:b].sum(), _series(r, den)[a:b].sum()
    return float(n / d) if d > 0 else float("nan")


def outcome(name: str, r) -> float:
    """One run's value of an outcome of the contrasts."""
    w = _windows()
    d0, d1 = w["drought"]
    s0, s1 = w["stockout"]
    if name == "episodes_0_24m":
        return float(r["rota_symptomatic_episodes_per_child_year_0_24m"])
    if name == "severe_0_59m":
        return float(sum(float(r[f"rota_episodes_sev06_{b}"]) for b in ("0_12m", "12_24m", "24_60m")))
    if name == "care_seeking":
        return float(r["econ_episode_care_seeking_rate"])
    if name == "care_seeking_q1":
        return float(r["econ_care_seeking_rate_q1"])
    if name == "zero_cash_drought":
        return float(np.nanmean(_series(r, "wk_zero_cash_share")[d0:d1]))
    if name == "zero_cash_drought_to_end":
        return float(np.nanmean(_series(r, "wk_zero_cash_share")[d0:]))
    if name == "care_drought_to_end":
        return _ratio(r, "wk_episodes_ended_with_care", "wk_episodes_ended", d0)
    if name == "poverty_end":
        return float(np.nanmean(_series(r, "wk_poverty")[-4:]))
    if name == "fz_zero_cash_flood_to_end":
        return float(np.nanmean(_series(r, "wk_fz_zero_cash_share")[w["flood"][0]:]))
    if name == "episodes_drought_to_end":
        return float(sum(_series(r, f"wk_episodes_{p}")[d0:].sum() for p in ("rota", "campy", "crypto")))
    if name == "belief_end_of_stockout":
        return float(np.nanmean(_series(r, "wk_belief_success")[s1 - 4:s1]))
    if name == "care_stockout_to_end":
        return _ratio(r, "wk_episodes_ended_with_care", "wk_episodes_ended", s0)
    if name == "belief_13w_after":
        return float(np.nanmean(_series(r, "wk_belief_success")[s1 + 13:s1 + 17]))
    raise KeyError(name)


#: (key, experiment, what, scenario, reference, outcome[, (scenario 2, reference 2)]): the effect is the mean
#: over seeds of (scenario - reference), or of (scenario - reference) - (scenario 2 - reference 2)
CONTRASTS = (
    ("vacc_episodes", "3A", "today's vaccine, rotavirus episodes 0-24 months", "R0_none", "vacc_none",
     "episodes_0_24m"),
    ("vacc_severe", "3A", "today's vaccine, severe rotavirus episodes 0-59 months", "R0_none", "vacc_none",
     "severe_0_59m"),
    ("depth", "3A", "protection against infection 0.95, episodes 0-24 months", "grid_v0.95_d616", "R0_none",
     "episodes_0_24m"),
    ("illness_protection", "3A", "without the vaccine's extra prior infection, episodes 0-24 months",
     "vacc_no_prior_infection_equivalent", "R0_none", "episodes_0_24m"),
    ("neonatal", "3B", "first dose at birth, severe episodes 0-59 months", "neonatal", "R0_none", "severe_0_59m"),
    ("booster", "3B", "booster at 9 months, episodes 0-24 months", "booster_9m", "R0_none", "episodes_0_24m"),
    ("wash", "3B", "water and sanitation, two lowest quintiles, episodes 0-24 months", "wash_q12", "R0_none",
     "episodes_0_24m"),
    ("free_care", "1A", "all care free, care sought per episode", "price_free", "R0_none", "care_seeking"),
    ("credit", "1A", "one day of credit, care sought per episode", "credit_1d", "R0_none", "care_seeking"),
    ("free_ors", "1A", "free ORS and zinc at drug sellers, care sought per episode", "free_ors_zinc", "R0_none",
     "care_seeking"),
    ("leap", "1A", "LEAP grants to the two lowest quintiles, care sought per episode, lowest quintile",
     "leap_q12_x1", "R0_none", "care_seeking_q1"),
    ("perceived", "1A", "care perceived to work with probability 0.2, care sought per episode", "perceived_0.2",
     "R0_none", "care_seeking"),
    ("stockout_belief", "1B", "26-week stock-out (social weight 0.6), mean belief over its last 4 weeks",
     "B1_s0.6_stockout26w", "B1_s0.6_none", "belief_end_of_stockout"),
    ("stockout_care", "1B", "26-week stock-out (social weight 0.6), care sought per episode from its start",
     "B1_s0.6_stockout26w", "B1_s0.6_none", "care_stockout_to_end"),
    ("campaign", "1B", "information campaign after the stock-out, mean belief 13-16 weeks after", "B4_campaign",
     "B1_s0.6_stockout26w", "belief_13w_after"),
    ("drought_cash", "2", "slow-recovery drought, households without cash during the drought", "D-slow_none",
     "R0_none", "zero_cash_drought"),
    ("drought_care", "2", "slow-recovery drought, care sought per episode from its start", "D-slow_none",
     "R0_none", "care_drought_to_end"),
    ("drought_poverty", "2", "slow-recovery drought, poverty in the last 4 weeks", "D-slow_none", "R0_none",
     "poverty_end"),
    ("flood_cash", "2", "slow-recovery flood, flood zone households without cash from the flood",
     "F-slow_none", "R0_none", "fz_zero_cash_flood_to_end"),
    ("adjustment_cash", "2", "economic adjustment, the drought's added share of households without cash "
     "(with minus without)", "D-slow_adjustment", "R0_adjustment", "zero_cash_drought_to_end",
     ("D-slow_none", "R0_none")),
    ("adjustment_care", "2", "economic adjustment, the drought's change in care sought per episode "
     "(with minus without)", "D-slow_adjustment", "R0_adjustment", "care_drought_to_end",
     ("D-slow_none", "R0_none")),
    ("pathways", "2", "the environmental pathways, children's episodes from the drought's start "
     "(slow-recovery drought)", "D-slow_none", "D-slow_none_noenv", "episodes_drought_to_end"),
)


def contrast(df, c: tuple, seeds=None) -> dict:
    """One contrast in one set's runs (a frame of rows with LABEL = scenario labels and "seed")."""
    key, exp, what, sc, ref, out = c[:6]
    pairs = [(sc, ref)] + ([c[6]] if len(c) > 6 else [])
    vals = {}
    for lab in {x for p in pairs for x in p}:
        rows = df[df[LABEL] == lab]
        vals[lab] = {int(s): outcome(out, r) for s, r in zip(rows["seed"], rows.to_dict("records"))}
    common = set.intersection(*(set(v) for v in vals.values()))
    if seeds is not None:
        common &= set(seeds)
    common = sorted(common)
    d = np.array([sum((1 if i == 0 else -1) * (vals[a][s] - vals[b][s]) for i, (a, b) in enumerate(pairs))
                  for s in common])
    ref_mean = float(np.mean([vals[ref][s] for s in common])) if common else float("nan")
    n = len(d)
    mean = float(d.mean()) if n else float("nan")
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"contrast": key, "experiment": exp, "what": what, "diff": mean, "se": se, "n": n,
            "rel": mean / ref_mean if ref_mean else float("nan"), "reference_mean": ref_mean}


def analyse(ps, step2, out_dir: str | None = None) -> dict:
    """ps: the scenarios over the sets (frame); step2: step 2's results of the headline scenarios (frame of
    every part, the adopted set). Returns the table by contrast and set, and the rule's verdict."""
    import pandas as pd
    step2 = step2[step2[LABEL].isin(HEADLINE)]
    first8 = sorted(step2["seed"].unique())[:PS_REPS]
    rows, verdict = [], []
    for c in CONTRASTS:
        full = contrast(step2, c)                                   # the adopted set, every seed of step 2
        sign = np.sign(full["diff"])
        tested = bool(np.isfinite(full["se"]) and abs(full["diff"]) > 2 * full["se"])
        per = [{**contrast(step2, c, seeds=first8), "set": ADOPTED}]
        for name, g in ps.groupby(SET_KEY):
            g = g.assign(**{LABEL: g[SCENARIO_KEY]})
            per.append({**contrast(g, c), "set": name})
        for p in per:
            p["meets"] = bool(np.sign(p["diff"]) == sign and abs(p["diff"]) > 2 * p["se"])
        rows += per
        share = float(np.mean([p["meets"] for p in per]))
        diffs = [p["diff"] for p in per]
        verdict.append({"contrast": c[0], "experiment": c[1], "what": c[2],
                        "adopted_diff": full["diff"], "adopted_se": full["se"], "adopted_n": full["n"],
                        "adopted_rel": full["rel"], "tested": tested, "sets": len(per),
                        "share_meeting": share, "robust": bool(tested and share >= ROBUST_SHARE),
                        "min_diff": float(np.min(diffs)), "max_diff": float(np.max(diffs)),
                        "min_rel": float(np.min([p["rel"] for p in per])),
                        "max_rel": float(np.max([p["rel"] for p in per]))})
    table, verdict = pd.DataFrame(rows), pd.DataFrame(verdict)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        table.to_csv(os.path.join(out_dir, "param_sets_by_set.csv"), index=False)
        verdict.to_csv(os.path.join(out_dir, "param_sets_robustness.csv"), index=False)
    return {"by_set": table, "verdict": verdict}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="step 3: the headline contrasts over calibrated parameter sets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--reps", type=int, default=8)
    v.add_argument("--cores", type=int, default=None)
    v.add_argument("--no-control", action="store_true")
    sub.add_parser("sets")
    a = sub.add_parser("analyse")
    a.add_argument("--ps", required=True, help="results.parquet of part ps")
    a.add_argument("--step2", nargs="+", required=True, help="results.parquet of step 2's parts")
    a.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        verify_new(args.reps, args.cores, control=not args.no_control)
    elif args.cmd == "sets":
        out = select_sets()
        print(verification_table().to_string(index=False))
        print(f"\nkept {len(out['sets'])}: {', '.join(s['name'] for s in out['sets'])} -> {SETS_JSON}")
    else:
        import pandas as pd
        ps = pd.read_parquet(args.ps)
        step2 = pd.concat([pd.read_parquet(p) for p in args.step2], ignore_index=True)
        out = args.out or os.path.join(os.path.dirname(args.ps), "analysis")
        res = analyse(ps, step2, out)
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(res["verdict"].to_string(index=False))
        from experiments.scenarios.paper_figures import plot_param_sets
        print(plot_param_sets(res["by_set"], res["verdict"], out))


if __name__ == "__main__":
    main()

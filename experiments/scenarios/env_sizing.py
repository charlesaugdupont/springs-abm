# experiments/scenarios/env_sizing.py
"""
Sizing the environmental pathways of experiment 2 (step 1b of the revision;
experiments.scenarios.hooks: env_settings, ENV_SIZED). Each multiplier
is set so that the model reproduces a published effect, on the same 8 seeds
for every value, and accepted when the target lies within 2 SE of the model's
effect:

  drought hygiene   child diarrhoea episodes (the three pathogens), town-wide,
                    over the drought: x1.08 against the drought without it
                    (severe six-month drought, Wang et al. 2022; 1.05 for a
                    mild one, a sensitivity)
  first rains       Cryptosporidium and Campylobacter episodes of children in
                    the FIRST_RAINS_DAYS after the drought: x1.26 against none
                    (extreme rain after a dry period, Kraay et al. 2020)
  flood             children's episodes in the flood zone during the flood and
                    the four weeks after: x1.23 against no flood (studies
                    adjusting for temperature, Kraay et al. 2020)

Round 1 brackets each target with three values (the drought without the
pathway counting as the level 1 of drought hygiene and first rains, at an IRR of
exactly 1). The value is interpolated (linearly in the log of the multiplier)
between the levels that bracket the target when the episodes' IRR rises over the
levels; where noise makes it not rise, from a weighted least-squares line in the
multiplier over every level instead (through IRR 1 at level 1, except for the
flood, whose level 1 is a flood without extra contamination). Round 2 (with the
adopted vaccine, as in the reruns) runs the values to check the rule on children's
expected episodes, the mild drought's hygiene (x1.05) as a sensitivity, and, for
the first rains and the flood, whose round-1 values rest on few episodes, levels
around the value (ROUND2_LEVELS), so that a value failing the rule can be
interpolated precisely and checked in a round 3. Infections (children's, and all
ages) and observed episodes are reported alongside.

record_env (the step_callback) counts every day's new infections by pathogen
among the flood zone's children ("zc") and adults ("za") and the rest's ("oc",
"oa"), and children's expected episodes by pathogen and zone: the sum over their
infections of each one's probability of becoming an episode, from the model's own
function and the child's age and history on the day (as the vaccine trial does;
the same expectation as the episodes, without the chance of each draw, so 3-4
times more precise for the first rains and the flood, which have about 15 and 4
children's episodes a run in their windows). env_metrics adds children's episodes
by pathogen and zone (from the episode log). Runs of SIZING_STEPS days, to four
weeks after the flood.

    python -m experiments.scenarios.run env1 --cores 2 --reps 8
    python -m experiments.scenarios.env_sizing <results.parquet>             # round 1: the values
    python -m experiments.scenarios.run env2 --cores 2 --reps 8              # round 2 (hooks.ENV_SIZED)
    python -m experiments.scenarios.env_sizing --round2 <results.parquet>    # the rule
    python -m experiments.scenarios.run env3 --cores 2 --reps 8              # round 3 (ROUND3_VALUES)
    python -m experiments.scenarios.env_sizing --round3 <results.parquet>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from abm.constants import AgentPropertyKeys as K
from experiments.scenarios import hooks as H

GROUPS = ("zc", "za", "oc", "oa")
SIZING_STEPS = H.FLOOD_START_DAY + H.FLOOD_DAYS + 28 + 1
TARGETS = {"drought_hygiene": 1.08, "first_rains": 1.26, "flood_contamination": 1.23}
#: a mild six-month drought raises child diarrhoea by 5% (Wang et al. 2022): experiment 2's sensitivity
MILD_TARGET = 1.05


def windows() -> dict:
    d_end = H.DROUGHT_START_DAY + H.DROUGHT_DAYS
    f0 = H.FLOOD_START_DAY
    return {"drought": (H.DROUGHT_START_DAY, d_end), "rains": (d_end, d_end + H.FIRST_RAINS_DAYS),
            "flood": (f0, f0 + H.FLOOD_DAYS + 28)}


def _count(model, st) -> None:
    nd = model.graph.ndata
    zone = model._exp_state["zone_agent"]
    child = nd[K.IS_CHILD].bool()
    masks = {"zc": zone & child, "za": zone & ~child, "oc": ~zone & child, "oa": ~zone & ~child}
    for p in model.pathogens:
        cur = nd[K.num_infections(p.name)]
        new = (cur - st["prev"][p.name]).clamp(min=0)
        st["prev"][p.name] = cur.clone()
        for g, m in masks.items():
            st["inf"][p.name][g].append(int(new[m].sum()))


def _illness(model):
    from abm.systems.child_illness import ChildIllnessSystem
    return next(s for s in model.systems if isinstance(s, ChildIllnessSystem))


def _expected(model, st, now: int) -> None:
    """Children's infections logged since the last call: each one's probability of becoming an episode
    (ChildIllnessSystem's rule: symptomatic_probability of the age on the day of the log and the prior
    infections, plus the vaccine's one where it counts as one), added to its day, by pathogen and zone."""
    from abm.systems.child_illness import _vaccine_illness_key, symptomatic_probability
    from abm.systems.demography import DAYS_PER_MONTH
    nd = model.graph.ndata
    zone = model._exp_state["zone_agent"]
    cfgs = {c.name: c for c in model.config.pathogens}
    log = _illness(model)._infection_log
    for name, cfg in cfgs.items():
        chunks = log.get(name, [])
        for t, idx, birth, _sym in chunks[st["n_chunks"].get(name, 0):]:
            idx = idx[nd[K.BIRTH_DAY][idx] == birth]          # the slot was not renewed since
            if idx.numel() == 0:
                continue
            prior = nd[K.num_infections(name)][idx] - 1        # unchanged while the child is infectious
            vk = _vaccine_illness_key(model.graph, name)
            if getattr(cfg, "vaccine_counts_as_prior_infection", False) and vk in nd:
                prior = prior + nd[vk][idx].to(prior.dtype)
            age = (nd[K.AGE][idx].double() - (now - int(t)) / DAYS_PER_MONTH).float()
            frac = symptomatic_probability(cfg, age, prior)
            ps = (torch.ones(idx.numel()) if frac is None else
                  torch.full((idx.numel(),), float(frac)) if isinstance(frac, float) else frac.float().cpu())
            z = zone[idx].cpu()
            ex = st["exp"][name]
            for key, m in (("zone", z), ("out", ~z)):
                ex[key][int(t)] = ex[key].get(int(t), 0.0) + float(ps[m].sum())
        st["n_chunks"][name] = len(chunks)


def record_env(model, day: int, combo: dict) -> None:
    """step_callback: the v5 hooks, then yesterday's new infections by group and children's expected
    episodes."""
    H.apply_hooks(model, day, combo)
    st = model.__dict__.get("_env_rec")
    if st is None:
        nd = model.graph.ndata
        model._env_rec = {"prev": {p.name: nd[K.num_infections(p.name)].clone() for p in model.pathogens},
                          "inf": {p.name: {g: [] for g in GROUPS} for p in model.pathogens},
                          "exp": {p.name: {"zone": {}, "out": {}} for p in model.pathogens},
                          "n_chunks": {}, "day": day}
        _expected(model, model._env_rec, day)
        return
    st["day"] = day
    _count(model, st)
    _expected(model, st, day)


def env_metrics(model) -> dict:
    st = model.__dict__.get("_env_rec")
    if st is None:
        return {}
    _count(model, st)                      # the last day's infections
    _expected(model, st, st["day"] + 1)
    out = {}
    for name, by in st["inf"].items():
        for g, v in by.items():
            out[f"env_inf_{name}_{g}"] = list(v)          # entry d = infections on day d
    days = len(next(iter(next(iter(st["inf"].values())).values())))
    for name, by in st["exp"].items():
        for z, d in by.items():
            v = np.zeros(days)
            for t, x in d.items():
                if 0 <= t < days:
                    v[t] += x
            out[f"env_exp_{name}_{z}"] = v.tolist()        # children's expected episodes, by day of onset
    zone = model._exp_state["zone_agent"].cpu().numpy()
    ep = {(p.name, z): np.zeros(days) for p in model.pathogens for z in ("zone", "out")}
    for e in model.get_child_episode_log():
        t = int(e["timestep"])
        if 0 <= t < days:
            ep[(e["pathogen"], "zone" if zone[int(e["agent_idx"])] else "out")][t] += 1
    for (name, z), v in ep.items():
        out[f"env_ep_{name}_{z}"] = v.tolist()
    return out


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------

LABEL = "gsa.design_id"
HYGIENE_LEVELS = (1.05, 1.10, 1.20)
RAINS_LEVELS = (1.5, 2.0, 3.0)
FLOOD_LEVELS = (1.0, 3.0, 10.0)


def round1_combos() -> list[dict]:
    c = [{LABEL: "R0", "hook.regime": "R0"}, {LABEL: "D_ref", "hook.regime": "D-fast"}]
    c += [{LABEL: f"D_hyg{m:g}", "hook.regime": "D-fast", "hook.drought_hygiene": m} for m in HYGIENE_LEVELS]
    c += [{LABEL: f"D_rain{m:g}", "hook.regime": "D-fast", "hook.first_rains": m} for m in RAINS_LEVELS]
    c += [{LABEL: f"F_c{m:g}", "hook.regime": "F-fast", "hook.flood_contamination": m} for m in FLOOD_LEVELS]
    return c


#: the values round 2 checked (round 1's, 2026-11-10): frozen here, so that `run env2` stays the design that
#: ran whatever hooks.ENV_SIZED becomes
ROUND2_VALUES: dict = {"drought_hygiene": 1.036, "first_rains": 1.9, "flood_contamination": 1.0,
                       "mild_drought_hygiene": 1.022}
#: round 2's levels of the pathways whose round-1 value rests on few episodes: the value is among them, and
#: the others bracket it, so that round 2 can interpolate on expected episodes if the value fails the rule
ROUND2_LEVELS: dict = {"first_rains": (1.6, 1.9, 2.3), "flood_contamination": (1.0, 3.0, 10.0)}
#: round 3: the values interpolated in round 2 where round 2's value failed the rule (the flood: x1 failed,
#: x3 and x10 bracket the target; 2026-11-10)
ROUND3_VALUES: dict = {"flood_contamination": 7.6}


def _with_vaccine(c: list[dict]) -> list[dict]:
    from experiments.scenarios.designs import _vaccine
    vacc = _vaccine()
    return [{LABEL: x[LABEL], **vacc, **{k: v for k, v in x.items() if k != LABEL}} for x in c]


def round2_combos(values: dict | None = None, levels: dict | None = None) -> list[dict]:
    """ROUND2_VALUES on the same seeds, with the references and with the adopted vaccine (as in the reruns):
    drought hygiene against the drought without it; the first rains against the drought without them, at
    ROUND2_LEVELS, and as experiment 2 runs them (with drought hygiene, against drought hygiene alone); the
    flood against none, at ROUND2_LEVELS."""
    v = dict(values or ROUND2_VALUES)
    lv = dict(levels or ROUND2_LEVELS)
    m_d, m_r, c_f = v["drought_hygiene"], v["first_rains"], v["flood_contamination"]
    if m_r not in lv["first_rains"] or c_f not in lv["flood_contamination"]:
        raise ValueError("the round-2 levels must contain the values")
    c = [{LABEL: "R0", "hook.regime": "R0"}, {LABEL: "D_ref", "hook.regime": "D-fast"},
         {LABEL: "D_hyg", "hook.regime": "D-fast", "hook.drought_hygiene": m_d},
         {LABEL: "D_hyg_mild", "hook.regime": "D-fast", "hook.drought_hygiene": float(v["mild_drought_hygiene"])},
         {LABEL: "D_both", "hook.regime": "D-fast", "hook.drought_hygiene": m_d, "hook.first_rains": m_r}]
    c += [{LABEL: f"D_rain{m:g}", "hook.regime": "D-fast", "hook.first_rains": float(m)} for m in lv["first_rains"]]
    c += [{LABEL: f"F_c{m:g}", "hook.regime": "F-fast", "hook.flood_contamination": float(m)}
          for m in lv["flood_contamination"]]
    return _with_vaccine(c)


def round3_combos(values: dict | None = None) -> list[dict]:
    """The flood at its value interpolated in round 2, against no flood, with the adopted vaccine."""
    v = dict(values or ROUND3_VALUES)
    c = [{LABEL: "R0", "hook.regime": "R0"}]
    if "flood_contamination" in v:
        c.append({LABEL: f"F_c{v['flood_contamination']:g}", "hook.regime": "F-fast",
                  "hook.flood_contamination": float(v["flood_contamination"])})
    return _with_vaccine(c)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _sum(df, cols, a, b) -> np.ndarray:
    """Per run: the sum of the listed daily series over days [a, b)."""
    return np.array([sum(float(np.sum(np.asarray(r[c])[a:b])) for c in cols) for _, r in df.iterrows()])


def irr(df, label: str, ref: str, cols: list[str], window: tuple, n_boot: int = 2000, seed: int = 1) -> dict:
    """Ratio of sums over the same seeds, and its SE (log scale) from resampling seeds."""
    t = df[df[LABEL] == label].sort_values("seed")
    r = df[df[LABEL] == ref].sort_values("seed")
    seeds = sorted(set(t["seed"]) & set(r["seed"]))
    t, r = t[t["seed"].isin(seeds)], r[r["seed"].isin(seeds)]
    a, b = _sum(t, cols, *window), _sum(r, cols, *window)
    est = a.sum() / b.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seeds), (n_boot, len(seeds)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.log(a[idx].sum(axis=1) / b[idx].sum(axis=1))
    se = float(np.nanstd(boot[np.isfinite(boot)]))
    return {"irr": float(est), "se_log": se, "lo": float(est * np.exp(-2 * se)), "hi": float(est * np.exp(2 * se)),
            "events": float(a.sum()), "events_ref": float(b.sum()), "seeds": len(seeds)}


def outcomes(pathogens=("rota", "campy", "crypto")) -> dict:
    """Series for each target: (window, children's episodes, infections of all ages, children's
    infections, children's expected episodes)."""
    ep = lambda ps, zs=("zone", "out"), kind="ep": [f"env_{kind}_{p}_{z}" for p in ps for z in zs]
    inf = lambda ps, gs=GROUPS: [f"env_inf_{p}_{g}" for p in ps for g in gs]
    cc = ("campy", "crypto")
    return {
        "drought_hygiene": ("drought", ep(pathogens), inf(pathogens), inf(pathogens, ("zc", "oc")),
                            ep(pathogens, kind="exp")),
        "first_rains": ("rains", ep(cc), inf(cc), inf(cc, ("zc", "oc")), ep(cc, kind="exp")),
        "flood_contamination": ("flood", ep(pathogens, ("zone",)), inf(pathogens, ("zc", "za")),
                                inf(pathogens, ("zc",)), ep(pathogens, ("zone",), kind="exp")),
    }


def _effects(df, lab, ref, out, win) -> dict:
    _, ep_cols, inf_cols, child_cols, exp_cols = out
    eff = {"episodes": irr(df, lab, ref, ep_cols, win), "infections": irr(df, lab, ref, inf_cols, win),
           "child_infections": irr(df, lab, ref, child_cols, win)}
    if all(c in df for c in exp_cols):
        eff["expected_episodes"] = irr(df, lab, ref, exp_cols, win)
    return eff


#: (round 1's label prefix, reference, whether the reference is the pathway's level 1)
ROUND1 = {"drought_hygiene": ("D_hyg", "D_ref", True), "first_rains": ("D_rain", "D_ref", True),
          "flood_contamination": ("F_c", "R0", False)}


def analyse(path: str) -> dict:
    """Round 1: each pathway's effects at its levels, and the value for round 2."""
    import pandas as pd
    df = pd.read_parquet(path)
    w = windows()
    res = {}
    for name, out in outcomes().items():
        prefix, ref, level_one = ROUND1[name]
        rows = {}
        for lab in sorted(l for l in df[LABEL].unique() if l.startswith(prefix)):
            rows[float(lab[len(prefix):])] = _effects(df, lab, ref, out, w[out[0]])
        res[name] = {"target": TARGETS[name], "window": w[out[0]], "reference": ref, "levels": rows,
                     **choose(rows, TARGETS[name], level_one)}
        if name == "drought_hygiene":
            res["mild_drought_hygiene"] = {"target": MILD_TARGET, **choose(rows, MILD_TARGET, level_one)}
    res["drought_vs_R0"] = {"episodes": irr(df, "D_ref", "R0", outcomes()["drought_hygiene"][1], w["drought"])}
    return res


def choose(rows: dict, target: float, level_one: bool, key: str = "episodes") -> dict:
    """The value for round 2: interpolated between the levels that bracket the target when the IRR rises
    over the levels (with the reference as level 1 at IRR 1 where it is the pathway's level 1), otherwise
    from the weighted least-squares line."""
    pts = dict(rows)
    if level_one:
        pts = {1.0: {key: {"irr": 1.0, "se_log": 0.0}}, **pts}
    ms = sorted(pts)
    vals = [pts[m][key]["irr"] for m in ms]
    rising = all(b > a for a, b in zip(vals, vals[1:]))
    line = fit_line(rows, target, level_one, key)
    interp = interpolate(pts, target, key) if rising else None
    chosen, how = (interp, "interpolated") if interp is not None else (line["value"], "line")
    return {"chosen": chosen, "how": how, "interpolated": interp, "line": line, "rising": rising}


def interpolate(rows: dict, target: float, key: str = "episodes") -> float | None:
    """The multiplier at which the episodes' IRR reaches the target: linear in
    log(multiplier) between the bracketing levels (None if not bracketed)."""
    ms = sorted(rows)
    vals = [rows[m][key]["irr"] for m in ms]
    for (m0, v0), (m1, v1) in zip(zip(ms, vals), zip(ms[1:], vals[1:])):
        if (v0 - target) * (v1 - target) <= 0 and v1 != v0:
            x = np.log(m0) + (target - v0) / (v1 - v0) * (np.log(m1) - np.log(m0))
            return float(np.exp(x))
    return None


def fit_line(rows: dict, target: float, through_one: bool, key: str = "episodes") -> dict:
    """IRR = a + b (m - 1) by weighted least squares over the levels (weights 1 / var(IRR), var from the
    bootstrap SE on the log scale), a fixed at 1 when the reference is the level 1; the value is where the
    line reaches the target."""
    if len(rows) < (1 if through_one else 2):
        return {"a": float("nan"), "b": float("nan"), "b_se": float("nan"), "value": None}
    m = np.array(sorted(rows), dtype=float)
    y = np.array([rows[k][key]["irr"] for k in sorted(rows)])
    se = np.array([max(rows[k][key]["irr"] * rows[k][key]["se_log"], 1e-6) for k in sorted(rows)])
    wts = 1.0 / se ** 2
    x = m - 1.0
    if through_one:
        b = float(np.sum(wts * x * (y - 1.0)) / np.sum(wts * x * x))
        b_se = float(1.0 / np.sqrt(np.sum(wts * x * x)))
        a = 1.0
    else:
        X = np.column_stack([np.ones_like(x), x])
        cov = np.linalg.inv(X.T @ (X * wts[:, None]))
        a, b = (cov @ X.T @ (wts * y)).tolist()
        b_se = float(np.sqrt(cov[1, 1]))
    value = float(1.0 + (target - a) / b) if b > 0 else None
    return {"a": float(a), "b": float(b), "b_se": b_se, "value": value}


def analyse_round2(path: str, values: dict | None = None) -> dict:
    """Round 2: each sized value against its target by the rule (the target within 2 SE of the model's
    effect on children's expected episodes), and, for the first rains and the flood, every level with the
    value interpolated on expected episodes (for a round 3 if the sized value fails)."""
    import pandas as pd
    df = pd.read_parquet(path)
    w = windows()
    o = outcomes()
    s = dict(values or ROUND2_VALUES)
    key = "expected_episodes"
    checks = {"drought_hygiene": ("D_hyg", "D_ref", TARGETS["drought_hygiene"], o["drought_hygiene"]),
              "mild_drought_hygiene": ("D_hyg_mild", "D_ref", MILD_TARGET, o["drought_hygiene"]),
              "first_rains": (f"D_rain{s['first_rains']:g}", "D_ref", TARGETS["first_rains"], o["first_rains"]),
              "first_rains_with_hygiene": ("D_both", "D_hyg", TARGETS["first_rains"], o["first_rains"]),
              "flood_contamination": (f"F_c{s['flood_contamination']:g}", "R0", TARGETS["flood_contamination"],
                                      o["flood_contamination"])}
    labels = set(df[LABEL])
    res = {}
    for name, (lab, ref, target, out) in checks.items():
        if lab not in labels or ref not in labels:
            continue
        eff = _effects(df, lab, ref, out, w[out[0]])
        e = eff.get(key, eff["episodes"])
        res[name] = {"label": lab, "target": target, **eff, "meets_rule": bool(e["lo"] <= target <= e["hi"])}
    for name, prefix, ref, level_one in (("first_rains", "D_rain", "D_ref", True),
                                         ("flood_contamination", "F_c", "R0", False)):
        if name not in res:
            continue
        rows = {float(l[len(prefix):]): _effects(df, l, ref, o[name], w[o[name][0]])
                for l in sorted(df[LABEL].unique()) if l.startswith(prefix)}
        res[name]["levels"] = rows
        res[name]["interpolated"] = choose(rows, TARGETS[name], level_one, key=key if rows and all(
            key in r for r in rows.values()) else "episodes")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="sizing of experiment 2's environmental pathways")
    ap.add_argument("results")
    ap.add_argument("--round2", action="store_true", help="check round 2's values against the rule")
    ap.add_argument("--round3", action="store_true", help="check round 3's values against the rule")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.round3:
        res = analyse_round2(args.results, values={**ROUND2_VALUES, **ROUND3_VALUES})
    else:
        res = analyse_round2(args.results) if args.round2 else analyse(args.results)
    out = args.out or os.path.join(os.path.dirname(args.results), "analysis")
    os.makedirs(out, exist_ok=True)
    name = ("env_sizing_round3.json" if args.round3 else "env_sizing_round2.json" if args.round2
            else "env_sizing_summary.json")
    with open(os.path.join(out, name), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

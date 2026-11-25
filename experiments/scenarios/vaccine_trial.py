# experiments/scenarios/vaccine_trial.py
"""
A randomised trial inside the model: the rotavirus vaccine's efficacy by time
since the last dose, measured as the trials pooled by Clark et al. (2019)
measured it (step 1 of the revision).

Design. Half the children are randomised to the vaccine and half to none, in
the same run: pathogens[rota].vaccine_dose_continuation = [0.5, 1, 1] gives a
child its first dose with probability 0.5 and then every later dose. Both arms
share the force of infection, as in an individually randomised trial (a run
with the vaccine against a run without it would add the indirect effect).
Every child keeps the dose ages drawn for it (the schedule's draws do not
depend on the continuation), so "time since the last dose" is defined in both
arms, as for placebo recipients. Runs of TRIAL_STEPS days; the first
TRIAL_BURN_IN are not measured.

record_trial (the step_callback) records, from the burn-in on, for children
born in the run:
- child-days by arm and week since the planned last dose (vt_pt_v, vt_pt_p:
  index 0 = week -PT_OFFSET_WEEKS);
- every rotavirus infection of a child (from the illness system's infection
  log): day, age (months), arm, the age at the dose that took (NaN for none),
  the duration of its protection against infection (lasts, days), natural
  infections before it, whether it produced an episode and the episode's
  initial severity.
trial_metrics returns them as list-valued columns (prefix vt_).

Analysis (analyse()). Efficacy = 1 - rate(vaccine arm) / rate(no vaccine), by
time since the last dose, pooled over seeds (ratio of sums), with a bootstrap
over seeds. Episodes are counted either as observed or as expected: the sum,
over infections, of each one's probability of becoming an episode given its
age and history (the model's own symptomatic_probability and severity
functions). Expected counts can be recomputed for another duration of the
vaccine's protection against illness (T) or another severity cut, from the
same infections; that ignores the feedback of illness on transmission
(episodes shed 1/0.385 times more than silent infections), so a setting chosen
this way is checked by trial runs with it on.

    python -m experiments.scenarios.run vt --cores 2 --reps 40
    python -m experiments.scenarios.vaccine_trial <results.parquet> [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from abm.constants import AgentPropertyKeys as K

TRIAL_STEPS = 2_190
TRIAL_BURN_IN = 730
#: child-days are binned by week since the planned last dose, from -PT_OFFSET_WEEKS
PT_OFFSET_WEEKS = 30
PT_WEEKS = PT_OFFSET_WEEKS + 270
#: Clark et al. (2019), high under-5 mortality: instantaneous efficacy against severe
#: rotavirus gastroenteritis 2 weeks and 12 months after the last dose (95% credible intervals)
CLARK_HIGH = {"2w": (0.66, 0.48, 0.81), "12m": (0.44, 0.27, 0.59)}


def _rota(model):
    return next(p for p in model.pathogens if p.name == "rota")


def _illness(model):
    from abm.systems.child_illness import ChildIllnessSystem
    return next(s for s in model.systems if isinstance(s, ChildIllnessSystem))


def _take_in_infections(model, st: dict, now: int) -> None:
    """Infections logged since the last call (the illness system logs them as
    they become infectious; a child's count of infections does not change
    while it is infectious). `now` = days completed; ages are taken back to the
    day of the log (children age at the end of each day)."""
    from abm.systems.demography import DAYS_PER_MONTH
    chunks = _illness(model)._infection_log.get("rota", [])
    nd = model.graph.ndata
    for t, idx, birth, sym in chunks[st["n_chunks"]:]:
        if t < TRIAL_BURN_IN:
            continue
        same = (nd[K.BIRTH_DAY][idx] == birth)          # the slot was not renewed since
        idx, birth, sym = idx[same], birth[same], sym[same]
        if idx.numel() == 0:
            continue
        st["inf"].append((np.full(idx.numel(), int(t)), idx.numpy().copy(), birth.numpy().astype(np.int64),
                          sym.numpy().copy(), nd[K.num_infections("rota")][idx].numpy().astype(np.int64),
                          nd[K.AGE][idx].double().numpy() - (now - int(t)) / DAYS_PER_MONTH))
    st["n_chunks"] = len(chunks)


def record_trial(model, day: int, combo: dict) -> None:
    """step_callback: the v5 hooks, then the trial's records."""
    from experiments.scenarios.hooks import apply_hooks
    from abm.systems.demography import DAYS_PER_MONTH
    apply_hooks(model, day, combo)
    v = getattr(_rota(model), "_vax", None)
    if v is None or v.get("birth_day") is None:
        return
    st = model.__dict__.get("_vtrial")
    if st is None:
        n = v["planned"].numel()
        st = model._vtrial = {"reg_birth": torch.full((n,), -10**9, dtype=torch.long), "reg": [], "inf": [],
                              "n_chunks": 0, "pt": np.zeros((2, PT_WEEKS))}
    st["day"] = day
    vb = v["birth_day"].long()
    new = (vb != st["reg_birth"]).nonzero(as_tuple=True)[0]
    if new.numel():          # the schedule of every new occupant of a slot (keyed by the schedule's own birth day)
        st["reg"].append((new.numpy().copy(), vb[new].numpy().copy(), v["planned"][new].numpy().copy(),
                          v["ages"][new].numpy().copy(), v["takes"][new].numpy().copy(), v["lasts"][new].numpy().copy()))
        st["reg_birth"][new] = vb[new]
    _take_in_infections(model, st, day)
    if day < TRIAL_BURN_IN:
        return
    nd = model.graph.ndata
    child = nd[K.IS_CHILD].bool() & (nd[K.BIRTH_DAY] >= 0) & (nd[K.BIRTH_DAY].long() == vb)
    tsd = nd[K.AGE].double() * DAYS_PER_MONTH - v["ages"][:, -1]
    w = torch.floor(tsd / 7.0).long() + PT_OFFSET_WEEKS
    ok = child & (w >= 0) & (w < PT_WEEKS)
    arm = v["planned"] >= 1
    for a in (0, 1):
        m = ok & (arm == bool(a))
        if torch.any(m):
            st["pt"][a] += np.bincount(w[m].numpy(), minlength=PT_WEEKS)


def trial_metrics(model) -> dict:
    """The trial's records as list-valued columns (see the module docstring)."""
    import pandas as pd
    from abm.systems.demography import DAYS_PER_MONTH
    st = model.__dict__.get("_vtrial")
    if st is None:
        return {}
    _take_in_infections(model, st, st["day"] + 1)
    reg = pd.DataFrame({
        "idx": np.concatenate([r[0] for r in st["reg"]]), "birth": np.concatenate([r[1] for r in st["reg"]]),
        "planned": np.concatenate([r[2] for r in st["reg"]]),
    })
    ages = np.concatenate([r[3] for r in st["reg"]])
    takes = np.concatenate([r[4] for r in st["reg"]])
    reg["lasts"] = np.concatenate([r[5] for r in st["reg"]])
    reg["a_last"] = ages[:, -1]
    k = ages.shape[1]
    given = np.arange(k)[None, :] < reg["planned"].to_numpy()[:, None]
    took = takes & given
    first = np.where(took.any(axis=1), took.argmax(axis=1), -1)
    reg["take_age"] = np.where(first >= 0, ages[np.arange(len(reg)), np.maximum(first, 0)], np.nan)
    reg = reg.drop_duplicates(["idx", "birth"], keep="first")
    if st["inf"]:
        inf = pd.DataFrame({name: np.concatenate([c[j] for c in st["inf"]])
                            for j, name in enumerate(("day", "idx", "birth", "sym", "ninf", "age_m"))})
    else:
        inf = pd.DataFrame(columns=["day", "idx", "birth", "sym", "ninf", "age_m"])
    inf = inf[inf["birth"] >= 0].merge(reg, on=["idx", "birth"], how="left")
    eps = [(e["agent_idx"], e["timestep"], e["initial_severity"]) for e in model.get_child_episode_log()
           if e["pathogen"] == "rota" and e["timestep"] >= TRIAL_BURN_IN]
    ep = pd.DataFrame(eps, columns=["idx", "day", "sev"]).drop_duplicates(["idx", "day"])
    inf = inf.merge(ep, on=["idx", "day"], how="left")
    age_d = inf["age_m"].to_numpy() * DAYS_PER_MONTH
    rota = _rota(model)
    out = {
        "vt_day": inf["day"].astype(np.int64).tolist(),
        "vt_age_m": inf["age_m"].astype(float).round(4).tolist(),
        "vt_arm": (inf["planned"].fillna(0) >= 1).astype(int).tolist(),
        "vt_tsd_d": (age_d - inf["a_last"].to_numpy()).round(2).tolist(),
        "vt_take_age_d": inf["take_age"].astype(float).round(2).tolist(),
        "vt_lasts_d": inf["lasts"].astype(float).round(2).tolist(),
        "vt_prior": (inf["ninf"].astype(np.int64) - 1).tolist(),
        "vt_sym": inf["sym"].astype(bool).astype(int).tolist(),
        "vt_sev": inf["sev"].astype(float).round(5).tolist(),
        "vt_pt_p": st["pt"][0].tolist(),
        "vt_pt_v": st["pt"][1].tolist(),
        "vt_protection_days": float(rota.config.vaccine_protection_days or np.inf),
        "vt_n_unmatched": int(inf["planned"].isna().sum()),
    }
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load(path: str):
    """One row per infection (with its run's seed) and child-days by run, arm and week."""
    import pandas as pd
    df = pd.read_parquet(path)
    cols = ["vt_day", "vt_age_m", "vt_arm", "vt_tsd_d", "vt_take_age_d", "vt_lasts_d", "vt_prior", "vt_sym", "vt_sev"]
    rows = []
    for _, r in df.iterrows():
        d = {c[3:]: np.asarray(r[c]) for c in cols}
        d["seed"] = np.full(len(d["day"]), r["seed"])
        d["label"] = np.full(len(d["day"]), r["gsa.design_id"])
        rows.append(pd.DataFrame(d))
    inf = pd.concat(rows, ignore_index=True)
    pt = {(r["gsa.design_id"], r["seed"]): np.vstack([np.asarray(r["vt_pt_p"]), np.asarray(r["vt_pt_v"])])
          for _, r in df.iterrows()}
    return inf, pt, df


def episode_weights(inf, *, T_days: float | None, cut: float, counts_as_prior: bool = True,
                    protection_days: float = 616.0):
    """For every infection: the probability that it becomes an episode, and the
    episode's initial severity, under a vaccine whose protection against
    illness lasts e * T_days from the dose that took (e = lasts / protection_days,
    the draw behind the protection against infection; None = for life) and cuts
    severity by `cut` while it lasts."""
    from config import SVEIRConfig
    from abm.agent.illness_mechanics import calculate_illness_severity
    from abm.systems.child_illness import symptomatic_probability
    from abm.systems.demography import DAYS_PER_MONTH
    cfg = SVEIRConfig()
    p_cfg = next(p for p in cfg.pathogens if p.name == "rota")
    ill = cfg.illness_mechanics.model_copy(update={"immunity_factor_vaccine": cut})
    age_m = inf["age_m"].to_numpy()
    since = age_m * DAYS_PER_MONTH - inf["take_age_d"].to_numpy()
    active = np.nan_to_num(since, nan=-1.0) >= 0
    if T_days is not None:
        e = inf["lasts_d"].to_numpy() / protection_days
        active &= since < e * T_days
    prior = inf["prior"].to_numpy()
    eff = prior + (active.astype(np.int64) if counts_as_prior else 0)
    p = symptomatic_probability(p_cfg, torch.as_tensor(age_m, dtype=torch.float), torch.as_tensor(eff))
    p = np.full(len(inf), 1.0) if p is None else (np.full(len(inf), float(p)) if np.isscalar(p) else p.numpy())
    sev = calculate_illness_severity("rota", torch.ones(len(inf), dtype=torch.bool),
                                     torch.as_tensor(age_m, dtype=torch.float), torch.as_tensor(active),
                                     torch.as_tensor(prior), ill).numpy()
    return p.astype(np.float64), sev.astype(np.float64), active


def ve_curve(inf, pt, w, *, bins_d, seeds=None, labels=None):
    """Efficacy 1 - rate(vaccine)/rate(none) in each bin of days since the last
    dose, pooled over runs (ratio of sums). w: each infection's weight (1 for an
    observed episode, its probability for an expected one, times 1[severe])."""
    keys = [k for k in pt if (labels is None or k[0] in labels) and (seeds is None or k[1] in seeds)]
    edges_w = (np.asarray(bins_d) / 7.0 + PT_OFFSET_WEEKS)
    num = np.zeros((2, len(bins_d) - 1))
    den = np.zeros((2, len(bins_d) - 1))
    sel = inf["label"].isin({k[0] for k in keys}) & inf["seed"].isin({k[1] for k in keys})
    tsd = inf["tsd_d"].to_numpy()
    arm = inf["arm"].to_numpy()
    b = np.digitize(tsd, bins_d) - 1
    ok = sel.to_numpy() & (b >= 0) & (b < len(bins_d) - 1)
    for a in (0, 1):
        m = ok & (arm == a)
        num[a] = np.bincount(b[m], weights=w[m], minlength=len(bins_d) - 1)
    weeks = np.arange(PT_WEEKS)
    for k in keys:
        for a in (0, 1):
            # child-days of week j counted in the bin holding the week's midpoint
            jb = np.digitize(weeks + 0.5, edges_w) - 1
            okw = (jb >= 0) & (jb < len(bins_d) - 1)
            den[a] += np.bincount(jb[okw], weights=pt[k][a][okw], minlength=len(bins_d) - 1)
    rate = num / np.maximum(den, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ve = 1.0 - rate[1] / rate[0]
    return ve, num, den


def ve_boot(inf, pt, w, *, bins_d, label, n_boot=400, seed=1):
    """Pooled efficacy by bin and a percentile interval from resampling runs."""
    seeds = sorted(k[1] for k in pt if k[0] == label)
    ve, num, den = ve_curve(inf, pt, w, bins_d=bins_d, labels={label})
    # per-run numerators and denominators, then resample runs
    per = []
    for s in seeds:
        _, n_s, d_s = ve_curve(inf, pt, w, bins_d=bins_d, labels={label}, seeds={s})
        per.append((n_s, d_s))
    N = np.stack([p[0] for p in per])
    D = np.stack([p[1] for p in per])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(seeds), len(seeds))
        n_, d_ = N[i].sum(axis=0), D[i].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            boots.append(1.0 - (n_[1] / d_[1]) / (n_[0] / d_[0]))
    boots = np.asarray(boots)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return ve, lo, hi, num, den


#: summary windows (days since the last dose): about two weeks, and about twelve months
WINDOWS = {"2w": (14.0, 45.0), "12m": (334.0, 396.0)}
#: efficacy against rotavirus gastroenteritis of any severity over efficacy against severe, in the
#: African trial of the pentavalent vaccine (Ghana, Kenya, Mali): 49.2/64.2 in the first year of
#: life and 30.5/39.3 over the whole follow-up (Armah et al. 2010; Tapia et al. 2012)
ANY_OVER_SEVERE = 0.77
#: the trials' efficacy against RVGE of any severity, as Clark et al.'s severe curve x ANY_OVER_SEVERE
TARGET_ANY = {k: tuple(round(ANY_OVER_SEVERE * x, 3) for x in v) for k, v in CLARK_HIGH.items()}


def scan_T(inf, pt, label: str, T_grid, *, cut: float = 0.4, n_boot: int = 200) -> list[dict]:
    """Efficacy against all episodes (expected counts) at about 2 weeks and 12
    months for each duration T of the protection against illness (None = for
    life), from the same recorded infections, and its distance to TARGET_ANY
    (squared, in units of the targets' half-width / 2)."""
    sub = inf[inf["label"] == label]
    rows = []
    for T in T_grid:
        p, sev, _ = episode_weights(sub, T_days=T, cut=cut)
        w = window_ve(sub, pt, p, label, n_boot=n_boot)
        d2 = 0.0
        for k, (mid, lo, hi) in TARGET_ANY.items():
            d2 += ((w[k][0] - mid) / ((hi - lo) / 4.0)) ** 2
        rows.append({"T_days": T, "ve_2w": w["2w"][:3], "ve_12m": w["12m"][:3], "d2": d2})
    return rows


def window_ve(inf, pt, w, label, n_boot=400):
    out = {}
    for name, (a, b) in WINDOWS.items():
        ve, lo, hi, num, den = ve_boot(inf, pt, w, bins_d=[a, b], label=label, n_boot=n_boot)
        out[name] = (float(ve[0]), float(lo[0]), float(hi[0]), float(num[0, 0]), float(num[1, 0]))
    return out


def analyse(path: str, out_dir: str, label: str | None = None) -> dict:
    inf, pt, df = load(path)
    labels = sorted({k[0] for k in pt}) if label is None else [label]
    res = {}
    for lab in labels:
        sub = inf[inf["label"] == lab]
        sym = sub["sym"].to_numpy().astype(float)
        p, sev, active = episode_weights(sub, T_days=None, cut=0.4)
        obs_sev = sub["sev"].to_numpy()
        check = {
            "episodes_observed": float(sym.sum()), "episodes_expected": float(p.sum()),
            "severity_max_abs_diff": float(np.nanmax(np.abs(obs_sev[sym > 0] - sev[sym > 0]))) if sym.sum() else None,
        }
        r = {"check_lifelong": check, "observed": {}, "expected": {}}
        for s_star in (None, 0.5, 0.55, 0.6):
            sev_ok = np.ones(len(sub)) if s_star is None else (sev >= s_star).astype(float)
            obs_ok = np.ones(len(sub)) if s_star is None else (np.nan_to_num(obs_sev, nan=0.0) >= s_star).astype(float)
            key = "all" if s_star is None else f"sev>={s_star}"
            r["observed"][key] = window_ve(sub, pt, sym * obs_ok, lab)
            r["expected"][key] = window_ve(sub, pt, p * sev_ok, lab)
        res[lab] = r
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vaccine_trial_summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="efficacy of the rotavirus vaccine in the model's trial runs")
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or os.path.join(os.path.dirname(args.results), "analysis")
    res = analyse(args.results, out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

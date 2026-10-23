# experiments/observation.py
"""
The MAL-ED observation layer: what MAL-ED's surveillance would have DETECTED
in a model run, so that the model is calibrated against like quantities.

Why
---
The model knows every infection; MAL-ED saw only those that a stool caught.
A surveillance stool was taken once a month and at every diarrhoeal episode,
and a sample counted as positive only above a quantity cut-off (Ct <= 30;
rotavirus <= 32.6). Symptomless rotavirus infections start near that cut-off
(median Cq 31 vs 17 in symptomatic ones, Vellore) and are shed for days, so
most fall between monthly samples: MAL-ED's own rotavirus paper found 38% of
children ever stool-positive by 24 months although serology showed >80%
exposed by 7 months (Mohan et al. 2017), and 98% of the infections it did
detect were seen in a single sample (Amin et al. 2026, Suppl. Table 1).
Comparing the model's infections with MAL-ED's therefore calibrates it to
too few infections and too little acquired immunity - which is why it could
not reproduce MAL-ED's second-year fall in rotavirus incidence among children
with no detected prior infection.

What it does
------------
Takes each child born in the run, from ChildIllnessSystem's infection log
(first infectious day, symptomatic or not) and episode log:

* surveillance: one stool in each month of life
  ObservationConfig.surveillance_start_months..end_months, mid-month;
* diarrhoeal stools: one per illness episode (any pathogen), collected with
  ObservationConfig.diarrhoeal_stool_tested_prob, at its onset;
* detection: a stool is positive for pathogen p if it falls in the
  detectable-shedding window of one of the child's p infections, which
  starts on the first infectious day and lasts a gamma-distributed number of
  days with mean PathogenConfig.detectable_shedding_days_{symptomatic,
  asymptomatic} (shape ObservationConfig.shedding_duration_shape); a
  collected stool of a p episode is always a pathogen-attributable detection;
* MAL-ED's two infection definitions (Amin et al. 2026, Methods):
  - crude: every positive is an infection, except that a symptomless and a
    symptomatic positive less than 7 days apart are one;
  - consecutive detections: every attributable diarrhoeal episode is an
    infection that also absorbs positives in the 5 days before its stool and
    the consecutive positives after it; remaining positives are grouped while
    the child ages <= 1 month between them with no negative in between;
    infections start at the midpoint between the first positive and the
    preceding negative (15 days before it if there is no test in the
    preceding month) and end symmetrically;
* rates per child-year over months of life with at least one tested stool,
  minus time spent infected (consecutive definition), by 3-month band, and
  split by detected infection history (MAL-ED Suppl. Table 3).

Only time after burn_in_days counts. Ages at first infection (detected, and
true) are Kaplan-Meier estimates with delayed entry: a child enters the risk
set at the age it reaches burn_in_days (if not yet infected) and leaves at
the end of the run, so children born late, or early in the burn-in, are
neither dropped nor allowed to bias the estimate towards young ages.

It uses its own random stream (run seed + ObservationConfig.rng_seed_offset)
for shedding durations and stool collection, so it never perturbs the
simulation, and draws in a fixed order so a run's observation is
reproducible. Nothing here changes the model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

DAYS_PER_MONTH = 365.25 / 12
BANDS_3MO = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24)]
BANDS_6MO = [(0, 6), (6, 12), (12, 18), (18, 24)]


@dataclass
class _Child:
    agent_idx: int
    birth_day: int
    # (day, positive?) for every tested stool of this pathogen's analysis,
    # with the attributable flag for this pathogen's own episodes
    tests: list = field(default_factory=list)


def _episode_index(model) -> dict:
    """{(agent_idx, pathogen, onset_day): realized-or-initial duration}."""
    out = {}
    for e in model.get_child_episode_log():
        d = e.get("realized_duration")
        if d is None:
            d = e.get("initial_duration", 0)
        out[(int(e["agent_idx"]), e["pathogen"], int(e["timestep"]))] = int(d)
    return out


def observe(model, burn_in_days: int = 730) -> dict:
    """MAL-ED-style detected outcomes of a finished run.

    Returns a flat dict of metrics, keys prefixed "{pathogen}_obs_" (detected)
    or "{pathogen}_true_" (true-infection anchors), see _summarise.
    """
    cfg = model.config
    obs = cfg.observation
    T = int(model.step_count)
    rng = np.random.default_rng(int(cfg.seed) + int(obs.rng_seed_offset))

    cohorts = [c for c in model.get_birth_cohorts() if "agent_idx" in c and c["birth_day"] >= 0]
    cohorts.sort(key=lambda c: (c["birth_day"], c["agent_idx"]))
    key_of = {(c["agent_idx"], c["birth_day"]): i for i, c in enumerate(cohorts)}
    n_child = len(cohorts)
    births = np.array([c["birth_day"] for c in cohorts], dtype=np.int64)

    # Surveillance stool days: mid-month of each month of life in range.
    months = np.arange(obs.surveillance_start_months, obs.surveillance_end_months + 1)   # 1-based
    surv_offsets = np.round((months - 0.5) * DAYS_PER_MONTH).astype(np.int64)

    # Diarrhoeal stools: one collection draw per episode, in a fixed order.
    # A slot hosts one child at a time and children leave at 60 months, so
    # within a child's first 24 months its slot's episodes are its own.
    episodes = _episode_index(model)
    by_slot_day: dict = defaultdict(lambda: defaultdict(list))   # slot -> day -> [pathogen]
    for (agent_idx, pathogen, day) in sorted(episodes, key=lambda k: (k[2], k[0], k[1])):
        if rng.random() < obs.diarrhoeal_stool_tested_prob:
            by_slot_day[agent_idx][day].append(pathogen)
    ep_stools = {slot: sorted(days.items()) for slot, days in by_slot_day.items()}

    out: dict = {}
    for p in model.pathogens:
        pc = p.config
        log = model.get_infection_log(p.name)
        # per child: infections (day, symptomatic, detectable end)
        inf_by_child = defaultdict(list)
        shape = float(obs.shedding_duration_shape)
        for a, b, d, s in zip(log["agent_idx"], log["birth_day"], log["day"], log["symptomatic"]):
            i = key_of.get((int(a), int(b)))
            mean = pc.detectable_shedding_days_symptomatic if s else pc.detectable_shedding_days_asymptomatic
            dur = rng.gamma(shape, mean / shape)
            if i is None:
                continue
            inf_by_child[i].append((int(d), bool(s), int(d) + dur))
        out.update(_analyse_pathogen(p.name, cohorts, births, inf_by_child, ep_stools,
                                     episodes, surv_offsets, T, burn_in_days, obs))
    return out


def _analyse_pathogen(name, cohorts, births, inf_by_child, ep_stools, episodes, surv_offsets, T,
                      burn_in, obs) -> dict:
    gap = obs.consecutive_gap_months * DAYS_PER_MONTH
    inc = int(obs.incubation_shedding_days)
    end_age_days = int(round(obs.surveillance_end_months * DAYS_PER_MONTH))

    # accumulators (age-month resolution 0..23)
    n_m = int(obs.surveillance_end_months)
    time_all = np.zeros(n_m)
    time_naive = np.zeros(n_m)
    inf_all = np.zeros(n_m)
    inf_naive = np.zeros(n_m)
    attrib = np.zeros(n_m)
    attrib_time = np.zeros(n_m)
    det_by_band6 = np.zeros(len(BANDS_6MO))
    attrib_by_band6 = np.zeros(len(BANDS_6MO))
    crude_total = 0
    consec_total = 0
    # Kaplan-Meier inputs (ages in days): (entry, end, event?) per child, for
    # the first DETECTED and the first TRUE infection. A child is at risk
    # from the age it reaches burn_in (or birth) to its first infection or
    # the end of the run; one already infected at entry is left out
    # (left truncation).
    km_det: list = []
    km_true: list = []
    # true primary share, cohort-equivalent: per age month 0-35, child-days
    # and (primary, all) infections after burn_in
    n36 = 36
    pt36 = np.zeros(n36)
    prim36 = np.zeros(n36)
    all36 = np.zeros(n36)

    for i, c in enumerate(cohorts):
        b = int(c["birth_day"])
        a_idx = int(c["agent_idx"])
        infs = sorted(inf_by_child.get(i, []))
        entry = max(0.0, burn_in - b)
        end = float(T - b)
        if end > entry:
            # --- true anchors ---
            first_true = (infs[0][0] - b) if infs else None
            _km_add(km_true, entry, end, first_true)
            for m in range(n36):
                lo = max(b + m * DAYS_PER_MONTH, burn_in)
                hi = min(b + (m + 1) * DAYS_PER_MONTH, T)
                if hi > lo:
                    pt36[m] += hi - lo
            for k, (d, _, _) in enumerate(infs):
                m = int((d - b) // DAYS_PER_MONTH)
                if d >= burn_in and m < n36:
                    all36[m] += 1
                    prim36[m] += (k == 0)

        # --- stools: (day, positive, attributable) ---
        tests = []
        for off in surv_offsets:
            day = b + int(off)
            if day >= T:
                break
            tests.append([day, False, False])
        for day, pathogens in ep_stools.get(a_idx, ()):
            if day < b or day >= min(T, b + end_age_days):
                continue
            tests.append([day, False, name in pathogens])
        tests.sort(key=lambda t: (t[0], not t[2]))
        for t in tests:
            day = t[0]
            if t[2]:
                t[1] = True
                continue
            for d, _s, e in infs:
                if d <= day < e:
                    t[1] = True
                    break
        if not tests:
            continue

        # --- crude definition ---
        pos = [t for t in tests if t[1]]
        crude = 0
        merged = set()
        for j, t in enumerate(pos):
            if j in merged:
                continue
            crude += 1
            # a symptomless and a symptomatic positive < 7 days apart are one
            for k2 in range(j + 1, len(pos)):
                if pos[k2][0] - t[0] >= 7:
                    break
                if pos[k2][2] != t[2]:
                    merged.add(k2)
        crude_total += crude

        # --- consecutive-detections definition ---
        groups = []          # [start_day, end_day, attributable]
        used = [False] * len(tests)
        # attributable episodes first, absorbing incubation and post-diarrhoeal shedding
        for j, t in enumerate(tests):
            if not (t[1] and t[2]):
                continue
            used[j] = True
            for k2 in range(j - 1, -1, -1):
                if tests[k2][0] < t[0] - inc:
                    break
                if tests[k2][1] and not tests[k2][2]:
                    used[k2] = True
            last = t[0]
            for k2 in range(j + 1, len(tests)):
                if tests[k2][2]:
                    break
                if not tests[k2][1] or tests[k2][0] - last > gap:
                    break
                used[k2] = True
                last = tests[k2][0]
            groups.append([t[0], last, True])
        # remaining positives: runs of consecutive positives
        j = 0
        while j < len(tests):
            if not tests[j][1] or used[j]:
                j += 1
                continue
            start_j = j
            last = tests[j][0]
            used[j] = True
            k2 = j + 1
            while k2 < len(tests) and tests[k2][1] and not used[k2] and tests[k2][0] - last <= gap:
                used[k2] = True
                last = tests[k2][0]
                k2 += 1
            first = tests[start_j][0]
            prev = [t[0] for t in tests[:start_j] if not t[1]]
            nxt = [t[0] for t in tests[k2:] if not t[1]]
            s_day = (first + prev[-1]) / 2 if prev and first - prev[-1] <= gap else first - 15
            e_day = (last + nxt[0]) / 2 if nxt and nxt[0] - last <= gap else last + 15
            groups.append([s_day, e_day, False])
            j = k2
        for g in groups:
            if g[2]:
                # An attributable infection starts at its stool (episode
                # onset) and lasts at least the episode plus MAL-ED's 2 days
                # (> 48 symptom-free hours), or to its last absorbed positive.
                g[1] = max(g[1], g[0] + max(1, episodes.get((a_idx, name, int(g[0])), 1)) + 2)
        groups.sort()
        consec_total += len(groups)
        if end > entry:
            _km_add(km_det, entry, end, (groups[0][0] - b) if groups else None)

        # --- tested months, time at risk, rates by age month ---
        tested_months = set(int((t[0] - b) // DAYS_PER_MONTH) for t in tests)
        first_det_day = groups[0][0] if groups else None
        for m in range(n_m):
            m_start = b + m * DAYS_PER_MONTH
            m_end = m_start + DAYS_PER_MONTH
            mid = (m_start + m_end) / 2
            if m not in tested_months or mid < burn_in or m_end > T:
                continue
            busy = sum(max(0.0, min(m_end, g[1]) - max(m_start, g[0])) for g in groups)
            at_risk = max(0.0, 30.5 - busy)
            time_all[m] += at_risk
            naive_here = first_det_day is None or first_det_day >= m_end
            if naive_here:
                time_naive[m] += at_risk
            elif first_det_day > m_start:
                # naive for the part of the month before the first detection
                time_naive[m] += max(0.0, min(at_risk, first_det_day - m_start))
            attrib_time[m] += 30.5
            for g in groups:
                if m_start <= g[0] < m_end:
                    inf_all[m] += 1
                    if first_det_day is not None and g[0] == first_det_day:
                        inf_naive[m] += 1
                    if g[2]:
                        attrib[m] += 1
        for g in groups:
            age_m = (g[0] - b) / DAYS_PER_MONTH
            mid = g[0]
            if mid < burn_in or age_m >= 24:
                continue
            k6 = int(age_m // 6)
            det_by_band6[k6] += 1
            attrib_by_band6[k6] += g[2]

    true_anchors = {
        "frac_infected_by_6m": _km_cdf(km_true, 6 * DAYS_PER_MONTH),
        "frac_infected_by_24m": _km_cdf(km_true, 24 * DAYS_PER_MONTH),
        # infections a fully followed 0-36 month cohort would have: rates by
        # age month, summed (pt36 > 0 for every month in a run of >= 3 years
        # past burn-in births; months with no child-time are skipped)
        "primary_share_0_36m": _cohort_share(prim36, all36, pt36),
    }
    median_first = _km_conditional_median(km_det, 24 * DAYS_PER_MONTH)
    return _summarise(name, time_all, time_naive, inf_all, inf_naive, attrib, attrib_time,
                      det_by_band6, attrib_by_band6, crude_total, consec_total, median_first,
                      true_anchors)


def _km_add(rows: list, entry: float, end: float, event_age):
    """Append a child's (entry, exit, event) unless it was infected before entry."""
    if event_age is None or event_age > end:
        rows.append((entry, end, False))
    elif event_age >= entry:
        rows.append((entry, float(event_age), True))


def _km_survival(rows: list):
    """(event ages, S just after each) of the delayed-entry Kaplan-Meier
    estimator. A child is in the risk set at age a if entry <= a <= exit."""
    if not rows:
        return np.zeros(0), np.zeros(0)
    arr = np.array(rows, dtype=float)
    entry, exit_, ev = arr[:, 0], arr[:, 1], arr[:, 2].astype(bool)
    ages = np.unique(exit_[ev])
    s, S = 1.0, []
    for a in ages:
        n = np.count_nonzero((entry <= a) & (exit_ >= a))
        d = np.count_nonzero(ev & (exit_ == a))
        if n > 0:
            s *= 1.0 - d / n
        S.append(s)
    return ages, np.array(S)


def _km_cdf(rows: list, age_days: float) -> float:
    """P(first infection by age_days); nan without data."""
    if not rows:
        return float("nan")
    ages, S = _km_survival(rows)
    k = np.searchsorted(ages, age_days, side="right")
    return float(1.0 - (S[k - 1] if k > 0 else 1.0))


def _km_conditional_median(rows: list, horizon_days: float) -> float:
    """Median age (months) at first infection among children infected by
    horizon_days: the smallest a with F(a) >= F(horizon) / 2."""
    ages, S = _km_survival(rows)
    keep = ages <= horizon_days
    ages, S = ages[keep], S[keep]
    if ages.size == 0:
        return float("nan")
    F = 1.0 - S
    k = int(np.searchsorted(F, 0.5 * F[-1], side="left"))
    return float(ages[min(k, len(ages) - 1)] / DAYS_PER_MONTH)


def _cohort_share(num, den, pt) -> float:
    ok = pt > 0
    tot = float((den[ok] / pt[ok]).sum())
    return float((num[ok] / pt[ok]).sum() / tot) if tot > 0 else float("nan")


def _rate(n, t_days):
    return float(n / t_days * 365.25) if t_days > 0 else float("nan")


def _ratio(a, b):
    return a / b if b and b == b and b > 0 and a == a else float("nan")


def _summarise(name, time_all, time_naive, inf_all, inf_naive, attrib, attrib_time, det6, att6,
               crude_total, consec_total, median_first, true_anchors) -> dict:
    out = {}
    pre = f"{name}_obs_"
    out[pre + "infections_per_child_year_0_24m"] = _rate(inf_all.sum(), time_all.sum())
    out[pre + "attributable_episodes_per_child_year_0_24m"] = _rate(attrib.sum(), attrib_time.sum())
    out[pre + "attributable_fraction_0_24m"] = float(attrib.sum() / inf_all.sum()) if inf_all.sum() else float("nan")
    for (lo, hi), n_det, n_att in zip(BANDS_6MO, det6, att6):
        out[pre + f"attributable_fraction_{lo}_{hi}m"] = float(n_att / n_det) if n_det else float("nan")
    for lo, hi in BANDS_3MO:
        sl = slice(lo, hi)
        out[pre + f"infections_per_child_year_{lo}_{hi}m"] = _rate(inf_all[sl].sum(), time_all[sl].sum())
        n0, t0 = inf_naive[sl].sum(), time_naive[sl].sum()
        n1, t1 = inf_all[sl].sum() - n0, time_all[sl].sum() - t0
        out[pre + f"naive_rate_{lo}_{hi}m"] = _rate(n0, t0)
        out[pre + f"prior_rate_{lo}_{hi}m"] = _rate(n1, t1)
    # pooled 12-24 m prior-infection rate ratio and the naive second-year decline
    for lo, hi in ((6, 12), (12, 24)):
        sl = slice(lo, hi)
        n0, t0 = inf_naive[sl].sum(), time_naive[sl].sum()
        n1, t1 = inf_all[sl].sum() - n0, time_all[sl].sum() - t0
        r0, r1 = _rate(n0, t0), _rate(n1, t1)
        out[pre + f"prior_infection_rate_ratio_{lo}_{hi}m"] = r1 / r0 if r0 and r0 == r0 and r0 > 0 else float("nan")
    r_naive_early = _rate(inf_naive[6:12].sum(), time_naive[6:12].sum())
    r_naive_late = _rate(inf_naive[15:24].sum(), time_naive[15:24].sum())
    out[pre + "naive_decline_15_24_vs_6_12m"] = (r_naive_late / r_naive_early
                                                 if r_naive_early and r_naive_early > 0 else float("nan"))
    out[pre + "crude_to_consecutive_ratio"] = crude_total / consec_total if consec_total else float("nan")
    # age shape of detected incidence (MAL-ED Suppl. Table 3 pooled)
    r = lambda lo, hi: _rate(inf_all[lo:hi].sum(), time_all[lo:hi].sum())
    out[pre + "infant_ratio_0_6_vs_6_12m"] = _ratio(r(0, 6), r(6, 12))
    out[pre + "second_to_first_year_ratio"] = _ratio(r(12, 24), r(0, 12))
    # Kaplan-Meier median among children with a detected infection by 24 m
    out[pre + "median_age_first_detected_infection_months"] = median_first
    tp = f"{name}_true_"
    for k, v in true_anchors.items():
        out[tp + k] = v
    return out

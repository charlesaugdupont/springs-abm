# experiments/scenarios/common.py
"""
Shared settings and outputs for the experiments on the v5 model:
care-seeking thresholds, weather shocks and
the rotavirus vaccine.

- experiment_spec(): a SweepSpec on the settlement grid with the model's
  defaults (Akuse today), fresh seeds from 61001 shared by every scenario
  (common random numbers), checkpointing and resume.
- weekly_series(): weekly time series built after the run from the systems'
  daily records (care-seeking, economy) and the child episode log - no change
  to the model or to its random draws.
- care_metrics(): care outcomes the validation metrics lack (effectively
  treated episodes, out-of-pocket spending, credit).
- experiment_metrics(): the validation metrics of campaign v5, care_metrics()
  and the weekly series (as list-valued columns, prefix "wk_").
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from experiments.orchestrator import SweepSpec

#: settlement grid and its 2025 population (campaign v3 onwards)
GRID_ID = "6855dd54618c"
AGENTS = 11_476
#: fresh seeds: calibration used 2026 and 10023, validation 50023, the design probes 60001-60003
BASE_SEED = 61_001
STEPS = 1_460
BURN_IN_DAYS = 730
OUTPUT_DIR = os.path.join("experiments", "outputs", "experiments_v5")


def experiment_spec(name: str, *, reps: int = 8, steps: int = STEPS, metrics_fn=None, step_callback=None,
                    base_overrides: dict | None = None, n_cores: int | None = None,
                    output_dir: str = OUTPUT_DIR, base_seed: int = BASE_SEED) -> SweepSpec:
    return SweepSpec(name=name, grid_id=GRID_ID, params=[], metrics_fn=metrics_fn or experiment_metrics,
                     reps=reps, steps=steps, agents=AGENTS, base_overrides=dict(base_overrides or {}),
                     output_dir=output_dir, n_cores=n_cores, base_seed=base_seed,
                     step_callback=step_callback, checkpoint_every=4, resume=True)


def _systems(model):
    from abm.systems.care_seeking import CareSeekingSystem
    from abm.systems.economics import EconomicSystem
    care = next((s for s in model.systems if isinstance(s, CareSeekingSystem)), None)
    eco = next((s for s in model.systems if isinstance(s, EconomicSystem)), None)
    return care, eco


def _weekly(daily: np.ndarray, how: str = "sum") -> list[float]:
    """Days -> complete weeks (a trailing partial week is dropped)."""
    n = len(daily) // 7
    if n == 0:
        return []
    w = np.asarray(daily[: n * 7], dtype=np.float64).reshape(n, 7)
    if how == "sum":
        return w.sum(axis=1).tolist()
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # an all-missing week stays nan
        return np.nanmean(w, axis=1).tolist()


def weekly_series(model) -> dict[str, list[float]]:
    """Weekly series over the whole run (week 0 starts on day 0), cedi economy only.

    Counts are sums over the week (episodes started, care decisions, visits,
    decisions blocked by cost, carer-days); shares are means of the daily
    shares (households without cash, households unable to pay for a hospital
    visit, persons below the poverty line). "_q1" is the lowest wealth quintile.
    Episodes by pathogen come from the child episode log (onset day)."""
    care, eco = _systems(model)
    out: dict[str, list[float]] = {}
    if care is None or eco is None or not getattr(care, "daily_records", None) or not eco.daily_records:
        return out
    crec, erec = care.daily_records, eco.daily_records
    days = min(len(crec), len(erec))

    def care_total(key, q=None):
        return np.array([(r[key][q] if q is not None else sum(r[key])) if key in r else 0.0 for r in crec[:days]],
                        dtype=np.float64)

    for key in ("episodes_started", "episodes_ended", "episodes_ended_with_care", "episodes_ended_blocked",
                "episodes_ended_treated", "decisions", "care_sought", "could_not_afford", "carer_days",
                "oop_paid", "visits_on_credit", "borrowed"):
        out[f"wk_{key}"] = _weekly(care_total(key))
        out[f"wk_{key}_q1"] = _weekly(care_total(key, 0))

    if "belief_success" in crec[0]:          # experiment 1B (belief_learning)
        for key in ("belief_success", "belief_worsen", "belief_low_share", "belief_success_q1"):
            out[f"wk_{key}"] = _weekly(np.array([r.get(key, np.nan) for r in crec[:days]], dtype=np.float64), "mean")

    def eco_share(num, den, q=None):
        n = np.array([float(r[num][q] if q is not None else r[num].sum()) for r in erec[:days]])
        d = np.array([float(r[den][q] if q is not None else r[den].sum()) for r in erec[:days]])
        return np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)

    for name, num, den in (("zero_cash_share", "zero_cash_households", "households"),
                           ("cannot_pay_share", "cannot_pay_facility_households", "households"),
                           ("poverty", "poor_persons", "persons")):
        out[f"wk_{name}"] = _weekly(eco_share(num, den), "mean")
        out[f"wk_{name}_q1"] = _weekly(eco_share(num, den, 0), "mean")

    onset = {p.name: np.zeros(days) for p in model.pathogens}
    for e in model.get_child_episode_log():
        t = int(e["timestep"])
        if 0 <= t < days and e["pathogen"] in onset:
            onset[e["pathogen"]][t] += 1
    for name, series in onset.items():
        out[f"wk_episodes_{name}"] = _weekly(series)
    return out


def care_metrics(model, burn_in_days: int = BURN_IN_DAYS) -> dict:
    """End-of-run care outcomes after the burn-in that the validation metrics
    lack (cedi economy only): the share of ended episodes that had at least one
    effective treatment, out-of-pocket spending per episode, visits bought on
    credit and the amount borrowed, and the debt outstanding at the end; overall
    and for the lowest quintile."""
    care, _ = _systems(model)
    if care is None or not getattr(care, "daily_records", None):
        return {}
    recs = care.daily_records[burn_in_days:]
    if not recs:
        return {}
    tot = lambda k: np.sum([r.get(k, [0] * 5) for r in recs], axis=0).astype(np.float64)
    ended, treated, oop = tot("episodes_ended"), tot("episodes_ended_treated"), tot("oop_paid")
    credit, borrowed = tot("visits_on_credit"), tot("borrowed")
    ratio = lambda a, b: float(a / b) if b > 0 else float("nan")
    out = {
        "exp_treated_share": ratio(treated.sum(), ended.sum()),
        "exp_treated_share_q1": ratio(treated[0], ended[0]),
        "exp_oop_per_episode": ratio(oop.sum(), ended.sum()),
        "exp_oop_per_episode_q1": ratio(oop[0], ended[0]),
        "exp_visits_on_credit": float(credit.sum()),
        "exp_borrowed_ghs": float(borrowed.sum()),
    }
    from abm.constants import AgentPropertyKeys as K
    nd = model.graph.ndata
    if K.DEBT in nd:
        hh = nd[K.HOUSEHOLD_ID].cpu().numpy()
        _, first = np.unique(hh, return_index=True)
        debt = nd[K.DEBT].cpu().numpy()[first]
        out["exp_households_in_debt_end"] = float(np.mean(debt > 0))
        out["exp_debt_end_ghs"] = float(debt.sum())
    return out


def rota_by_quintile(model, burn_in_days: int = BURN_IN_DAYS) -> dict:
    """Rotavirus episodes among children under five after the burn-in, per
    child-year by wealth quintile (denominator: the care-seeking system's daily
    count of children by quintile), and episodes of initial severity 0.6 or
    more (count). Agent slots stay in their household, so a slot's quintile is
    its episode's."""
    from abm.constants import AgentPropertyKeys as K
    care, _ = _systems(model)
    if care is None or not care.daily_records or "children" not in care.daily_records[0]:
        return {}
    recs = care.daily_records[burn_in_days:]
    if not recs:
        return {}
    child_days = np.sum([r["children"] for r in recs], axis=0).astype(np.float64)
    q = model.graph.ndata[K.WEALTH_QUINTILE].cpu().numpy()
    n_ep, n_severe = np.zeros(5), np.zeros(5)
    for e in model.get_child_episode_log():
        if e["pathogen"] == "rota" and int(e["timestep"]) >= burn_in_days:
            qq = q[int(e["agent_idx"])]
            n_ep[qq] += 1
            n_severe[qq] += e["initial_severity"] >= 0.6
    out = {}
    for k in range(5):
        out[f"exp_rota_episodes_per_child_year_q{k + 1}"] = (365.0 * n_ep[k] / child_days[k]) if child_days[k] else float("nan")
        out[f"exp_rota_severe_episodes_q{k + 1}"] = float(n_severe[k])
    return out


def experiment_metrics(model) -> dict:
    """Campaign v5's validation metrics, the care outcomes above and the weekly series."""
    from experiments.calibration.validate_calibration import validation_metrics_v5
    from experiments.scenarios.hooks import zone_series
    out = validation_metrics_v5(model)
    out.update(care_metrics(model))
    out.update(rota_by_quintile(model))
    out.update(weekly_series(model))
    out.update(zone_series(model))
    return out

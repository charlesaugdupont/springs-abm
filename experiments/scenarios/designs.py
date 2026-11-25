# experiments/scenarios/designs.py
"""
The scenario lists of the v5 experiments, one
Design per part. Each scenario is a dict of config overrides (orchestrator
dot-paths) and hook settings ("hook.*", read by experiments.scenarios.hooks),
labelled by "gsa.design_id". Every scenario runs with the same seeds.

    python -m experiments.scenarios.run 3a --cores 2
    python -m experiments.scenarios.run 1a --cores 2 --dry-run
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config import EconomyConfig
from experiments.scenarios.common import STEPS

#: the label column (the orchestrator skips "gsa.*" keys as config paths)
LABEL = "gsa.design_id"


@dataclass(frozen=True)
class Design:
    name: str                          # output folder under experiments/outputs/experiments_v5/
    combos: Callable[[], list[dict]]
    steps: int = STEPS
    hooks: bool = False                # run with experiments.scenarios.hooks.apply_hooks
    description: str = ""
    #: a design's own step_callback and metrics_fn (module-level, picklable); None = the defaults
    step_callback: Callable | None = None
    metrics_fn: Callable | None = None


# ---------------------------------------------------------------------------
# Experiment 3A: what limits the rotavirus vaccine (no new mechanism)
# ---------------------------------------------------------------------------

ROTA = "pathogens[rota]."
VACC_NONE = {f"{ROTA}vaccination_model": "hazard", f"{ROTA}vaccination_rate": 0.0}
#: protection against infection while protected (the default 0.55 is an assumption)
V_EFF_LEVELS = (0.30, 0.45, 0.55, 0.70, 0.85, 0.95)
#: mean duration of protection, days: 15.6, 20.2 (the default, 616 days) and 28.3 months
#: (Asare et al. 2026's 95% interval, 30.44 days a month), and lifelong (None)
DURATION_LEVELS = (475.0, 616.0, 861.0, None)
TAKE_IDEAL = {f"{ROTA}vaccine_take_by_dose": [0.95, 0.95, 0.95]}
COVERAGE_IDEAL = {f"{ROTA}vaccine_dose_continuation": [1.0, 1.0, 1.0]}


def _cell(v_eff: float, days: float | None) -> dict:
    """The overrides of one cell of the protection grid; {} for the default cell."""
    out = {}
    if v_eff != 0.55:
        out[f"{ROTA}vaccine_efficacy"] = v_eff
    if days != 616.0:
        out[f"{ROTA}vaccine_protection_days"] = days
    return out


def _cell_label(v_eff: float, days: float | None) -> str:
    return f"grid_v{v_eff:.2f}_d{'life' if days is None else int(days)}"


def exp3a_combos() -> list[dict]:
    combos = [{LABEL: "vacc_none", **VACC_NONE}]
    for v in V_EFF_LEVELS:
        for d in DURATION_LEVELS:
            label = "vacc_default" if (v, d) == (0.55, 616.0) else _cell_label(v, d)
            combos.append({LABEL: label, **_cell(v, d)})
    combos += [
        {LABEL: "vacc_coverage_ideal", **COVERAGE_IDEAL},
        {LABEL: "vacc_take_ideal", **TAKE_IDEAL},
        # take at the weakest and the strongest plausible cells
        {LABEL: "vacc_take_ideal_at_v0.30_d475", **TAKE_IDEAL, **_cell(0.30, 475.0)},
        {LABEL: "vacc_take_ideal_at_v0.85_d861", **TAKE_IDEAL, **_cell(0.85, 861.0)},
        {LABEL: "vacc_all_ideal", **COVERAGE_IDEAL, **TAKE_IDEAL, **_cell(0.95, None)},
        # the vaccine's lifelong protection against illness, part by part (structural assumptions)
        {LABEL: "vacc_no_severity_reduction", "illness_mechanics.immunity_factor_vaccine": 0.0},
        {LABEL: "vacc_no_prior_infection_equivalent", f"{ROTA}vaccine_counts_as_prior_infection": False},
    ]
    return combos


# ---------------------------------------------------------------------------
# Experiment 1B: learning how well care works (belief_learning; decision D1)
# ---------------------------------------------------------------------------

EXP1B_STEPS = 2_190            # stock-outs start on day 1,095: up to three years to recover
EXP1B_RAMP_STEPS = 2_555       # the half-speed ramp: two years down, two up, from day 1,095
LEARNING = {"steering_parameters.belief_learning": True,
            # perception pinned at the calibrated 0.8, so that a stock-out acts through outcomes only
            "steering_parameters.perceived_treatment_success_prob": 0.8}
#: social weights of B1: with 0 (own outcomes only) the pilot (2 seeds) moved the mean belief by under
#: 0.01 after a 26-week stock-out, so only the 26-week stock-out is run there
SOCIAL_B1 = (0.0, 0.3, 0.6)
STOCKOUT_WEEKS = (2, 4, 8, 16, 26)
#: B2's near-equilibrium check (the ramp at half speed) and B4's collapse use the strongest social weight
SOCIAL_STRONG = 0.6
SOCIAL_B2 = (0.0, 0.1, 0.3, 0.6)
STOCKOUT_START = 1_095


def _social(w: float) -> dict:
    return {**LEARNING, "steering_parameters.belief_social_weight": w}


def exp1b1_combos() -> list[dict]:
    """B1: stock-outs (care works with probability 0.2) of 2-26 weeks."""
    combos = []
    for w in SOCIAL_B1:
        combos.append({LABEL: f"B1_s{w:g}_none", **_social(w)})
        for weeks in (STOCKOUT_WEEKS if w > 0 else STOCKOUT_WEEKS[-1:]):
            combos.append({LABEL: f"B1_s{w:g}_stockout{weeks}w", **_social(w),
                           "hook.stockout_start": STOCKOUT_START, "hook.stockout_days": 7 * weeks})
    combos.append({LABEL: "B1_nolearning_stockout26w", "steering_parameters.perceived_treatment_success_prob": 0.8,
                   "hook.stockout_start": STOCKOUT_START, "hook.stockout_days": 182})
    return combos


def exp1b2_combos() -> list[dict]:
    """B2/B3: efficacy lowered to 0.1 over a year and raised back over a year
    (and, as the near-equilibrium check, over two years each way)."""
    combos = [{LABEL: f"B2_s{w:g}_ramp365", **_social(w), "hook.ramp_start": STOCKOUT_START, "hook.ramp_days": 365}
              for w in SOCIAL_B2]
    combos.append({LABEL: f"B2_s{SOCIAL_STRONG:g}_ramp730", **_social(SOCIAL_STRONG), "hook.ramp_start": STOCKOUT_START,
                   "hook.ramp_days": 730})
    return combos


def exp1b4_combos() -> list[dict]:
    """B4: after a 26-week stock-out (social weight SOCIAL_STRONG), what restores trust:
    an information campaign (beliefs of half the households reset), three months
    of free care, or both, from the day supplies return."""
    end = STOCKOUT_START + 182
    collapse = {**_social(SOCIAL_STRONG), "hook.stockout_start": STOCKOUT_START, "hook.stockout_days": 182}
    return [
        {LABEL: "B4_collapse", **collapse},
        {LABEL: "B4_campaign", **collapse, "hook.campaign_day": end, "hook.campaign_share": 0.5},
        {LABEL: "B4_free_care", **collapse, "hook.free_care_start": end, "hook.free_care_days": 91},
        {LABEL: "B4_both", **collapse, "hook.campaign_day": end, "hook.campaign_share": 0.5,
         "hook.free_care_start": end, "hook.free_care_days": 91},
    ]


# ---------------------------------------------------------------------------
# Experiment 3B: interventions that could improve the vaccine
# ---------------------------------------------------------------------------

def _table(*values) -> list[float]:
    """A dose-age table (VACCINE_AGE_PERCENTILES: 21 percentiles) from one value
    (every child the same) or a range (lo, hi) spread evenly over the percentiles."""
    if len(values) == 1:
        return [float(values[0])] * 21
    lo, hi = values
    return [lo + (hi - lo) * i / 20 for i in range(21)]


def exp3b_combos() -> list[dict]:
    from config import RotavirusConfig
    take = RotavirusConfig().vaccine_take_by_dose
    timely = {f"{ROTA}vaccine_dose1_age_days": _table(42.0), f"{ROTA}vaccine_dose_interval_days": _table(28.0)}
    # RV3-BB's schedule: dose 1 at 0-5 days, then about 8 and 14 weeks (one interval table: 49 days)
    neonatal = {f"{ROTA}vaccine_dose1_age_days": _table(0.0, 5.0), f"{ROTA}vaccine_dose_interval_days": _table(49.0)}
    return [
        {LABEL: "vacc_default"},
        {LABEL: "timely", **timely},
        {LABEL: "neonatal", **neonatal},
        {LABEL: "neonatal_take_ideal", **neonatal, **TAKE_IDEAL},
        {LABEL: "booster_9m", "hook.booster_age_days": 274},
        {LABEL: "wash_q12", "hook.wash_quintiles": "1,2"},
        {LABEL: "opv_separate", f"{ROTA}vaccine_take_by_dose": [round(min(1.0, 1.3 * t), 4) for t in take]},
        # sensitivity D4: SHINE's effect of household WASH on take, mean take kept
        {LABEL: "takewash_default", "hook.take_by_wash": True},
        {LABEL: "takewash_wash_q12", "hook.take_by_wash": True, "hook.wash_quintiles": "1,2"},
    ]


# ---------------------------------------------------------------------------
# Experiment 1A: the two edges of care-seeking and the policies that cross them
# ---------------------------------------------------------------------------

_E = EconomyConfig()
PRICE_FIELDS = ("cost_primary_insured", "cost_hospital_insured", "cost_facility_uninsured", "cost_private",
                "cost_drug_seller")


def price(multiple: float) -> dict:
    """Every provider's out-of-pocket price times `multiple` ({} for 1)."""
    if multiple == 1.0:
        return {}
    return {f"economy.{f}": getattr(_E, f) * multiple for f in PRICE_FIELDS}


def _mult_label(m: float) -> str:
    return "free" if m == 0 else f"x{m:g}"


PERCEIVED_LEVELS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 1.0)     # 0.8 = baseline
ACTUAL_LEVELS = (0.4, 0.6, 0.95)                                        # 0.8 = baseline
CREDIT_DAYS = (1.0, 3.0, 10.0)                                          # 0 = baseline
CREDIT_PRICES = (0.1, 0.5, 1.0, 2.0)


def exp1a_combos() -> list[dict]:
    combos = [{LABEL: "baseline"}]
    combos += [{LABEL: f"price_{_mult_label(m)}", **price(m)} for m in (0.0, 0.01, 0.1, 0.5, 2.0)]
    combos += [
        {LABEL: "free_primary_insured", "economy.cost_primary_insured": 0.0},
        {LABEL: "free_ors_zinc", "economy.cost_drug_seller": 0.0},
        {LABEL: "nhis_universal", "economy.insured_prob": 1.0},
        {LABEL: "nhis_universal_free_primary", "economy.insured_prob": 1.0, "economy.cost_primary_insured": 0.0},
    ]
    for quintiles, tag in (("1", "q1"), ("1,2", "q12")):
        for mult in (1.0, 2.0):
            combos.append({LABEL: f"leap_{tag}_x{mult:g}", "hook.transfer_quintiles": quintiles,
                           "hook.transfer_multiple": mult})
    combos += [
        {LABEL: "compensate_visit_time", "economy.public_visit_day_fraction": 0.0,
         "economy.private_visit_day_fraction": 0.0, "economy.drug_seller_visit_day_fraction": 0.0},
        {LABEL: "compensate_caring_time", "economy.carer_day_fraction": 0.0},
    ]
    combos += [{LABEL: f"perceived_{p:g}", "steering_parameters.perceived_treatment_success_prob": p}
               for p in PERCEIVED_LEVELS]
    combos += [{LABEL: f"actual_{p:g}", "steering_parameters.treatment_success_prob": p,
                "steering_parameters.perceived_treatment_success_prob": 0.8} for p in ACTUAL_LEVELS]
    for days in CREDIT_DAYS:
        for m in CREDIT_PRICES:
            combos.append({LABEL: f"credit_{days:g}d_price_{_mult_label(m)}", **price(m),
                           "economy.emergency_credit_days": days})
    return combos


def exp1a_refine_combos() -> list[dict]:
    """The edge found in 1A's price sweep (free to x0.01, in 8 of 8 seeds), refined once with
    four levels, as the rule set beforehand requires."""
    return [{LABEL: f"price_x{m:g}", **price(m)} for m in (0.001, 0.0025, 0.005, 0.0075)]


# ---------------------------------------------------------------------------
# Experiment 2: weather shocks x risk-management strategies (after Wren et al. 2025)
# ---------------------------------------------------------------------------

EXP2_STEPS = 1_826          # the shocks start on day 1,095 (experiments.scenarios.hooks)
#: the regimes every strategy is crossed with (plus R0, each strategy's own no-shock counterfactual)
EXP2_STRATEGY_REGIMES = ("R0", "C-hi", "F-slow", "D-slow", "C+D")


def exp2_combos() -> list[dict]:
    from experiments.scenarios.hooks import REGIMES, STRATEGIES
    combos = [{LABEL: f"{r}_none", "hook.regime": r, "hook.strategy": "none"} for r in REGIMES]
    for strategy in STRATEGIES[1:]:
        for r in EXP2_STRATEGY_REGIMES:
            combos.append({LABEL: f"{r}_{strategy}", "hook.regime": r, "hook.strategy": strategy})
    return combos


# ---------------------------------------------------------------------------
# Step 1 of the revision: the vaccine's efficacy by
# time since the last dose, in a trial inside the model (experiments.scenarios.vaccine_trial)
# ---------------------------------------------------------------------------

#: half the children randomised to the vaccine (dose 1 with probability 0.5, then every dose)
TRIAL_ARMS = {f"{ROTA}vaccine_dose_continuation": [0.5, 1.0, 1.0]}


def trial_combos() -> list[dict]:
    """The calibrated vaccine (protection against illness for life)."""
    return [{LABEL: "trial_lifelong", **TRIAL_ARMS}]


# ---------------------------------------------------------------------------
# Step 2 of the revision: the five experiments rerun with the
# vaccine adopted in step 1, one measured year after the two warm-up years (R1), borrowing off with
# credit as a sensitivity (R2), Experiment 2's environmental pathways (R3b), and about 40 seeds for
# the key contrasts (R4). Each scenario carries its number of seeds ("gsa.reps"); the seeds are the
# same for every scenario (61001, ...), so the first REST_REPS of a key scenario pair with the rest.
# ---------------------------------------------------------------------------

REV_STEPS = 1_095
KEY_REPS = 40
REST_REPS = 16
#: the vaccine adopted in step 1 (the user's decision of 2026-11-10, after the trial inside the model,
#: experiments.scenarios.vaccine_trial): protection against illness stays for life, and a vaccine that
#: took cuts severity as one natural infection does (0.2, IllnessMechanicsConfig's
#: severity_reduction_per_infection) instead of the unsourced 0.4; added to every scenario
VACCINE_REVISION: dict | None = {"illness_mechanics.immunity_factor_vaccine": 0.2}
#: the sensitivity on the duration of protection against illness: four years (mean), the shortest
#: inside every interval of the trial check
ILLNESS_WANES_4Y = {f"{ROTA}vaccine_illness_protection_days": 1461.0}
#: 3A's component on the duration of protection against illness: the alternative to the adopted one
ILLNESS_DURATION_KEY = f"{ROTA}vaccine_illness_protection_days"


def _vaccine() -> dict:
    if VACCINE_REVISION is None:
        raise ValueError("the revised designs need designs.VACCINE_REVISION (step 1 of the revision)")
    return dict(VACCINE_REVISION)


def _rev(combos: list[dict], key: set[str]) -> list[dict]:
    """Every scenario with the adopted vaccine (its own vaccine keys win) and its number of seeds."""
    from experiments.orchestrator import REPS_KEY
    vacc = _vaccine()
    out = []
    for c in combos:
        merged = {LABEL: c[LABEL], **{k: v for k, v in vacc.items() if k not in c},
                  **{k: v for k, v in c.items() if k != LABEL}}
        merged[REPS_KEY] = KEY_REPS if c[LABEL] in key else REST_REPS
        out.append(merged)
    return out


def exp3a_rev_combos() -> list[dict]:
    """3A: the protection grid (16 seeds) and the components (40), with the adopted vaccine; the
    component on illness takes the duration of protection against illness the other way (for life
    if the adopted one wanes, four years if it is for life)."""
    combos = [c for c in exp3a_combos()]
    t_ill = _vaccine().get(ILLNESS_DURATION_KEY)
    alt = None if t_ill else ILLNESS_WANES_4Y[ILLNESS_DURATION_KEY]
    combos.append({LABEL: "vacc_illness_" + ("lifelong" if alt is None else "wanes_4y"), ILLNESS_DURATION_KEY: alt})
    key = {"vacc_none", "vacc_default", "vacc_coverage_ideal", "vacc_take_ideal", "vacc_all_ideal",
           "vacc_no_severity_reduction", "vacc_no_prior_infection_equivalent", combos[-1][LABEL]}
    return _rev(combos, key)


def exp1a_rev_combos() -> list[dict]:
    """1A: the policies (40 seeds: free care at every provider joins them), the perceived efficacy of
    care and credit of 1, 3 and 10 days at today's prices (16). The price sweep (the old panel a), the
    actual efficacy of care and the credit x price grid are dropped."""
    old = {c[LABEL]: c for c in exp1a_combos()}
    policies = ["baseline", "price_free", "free_primary_insured", "free_ors_zinc", "nhis_universal",
                "nhis_universal_free_primary", "leap_q1_x1", "leap_q1_x2", "leap_q12_x1", "leap_q12_x2",
                "compensate_visit_time", "compensate_caring_time"]
    combos = [old[k] for k in policies]
    combos += [c for k, c in old.items() if k.startswith("perceived_")]
    combos += [{LABEL: f"credit_{d:g}d", "economy.emergency_credit_days": d} for d in CREDIT_DAYS]
    return _rev(combos, set(policies))


def exp1b_rev_combos() -> list[dict]:
    """1B in one year: B1's stock-outs from the end of the warm-up (40 seeds for none and the 26-week
    stock-out at the strongest social weight, 16 for the rest), and B4's interventions after a 26-week
    stock-out (40), followed to the end of the year (26 weeks)."""
    from experiments.scenarios.hooks import SHOCK_START_DAY
    start = SHOCK_START_DAY
    combos = []
    for w in SOCIAL_B1:
        combos.append({LABEL: f"B1_s{w:g}_none", **_social(w)})
        for weeks in (STOCKOUT_WEEKS if w > 0 else STOCKOUT_WEEKS[-1:]):
            combos.append({LABEL: f"B1_s{w:g}_stockout{weeks}w", **_social(w),
                           "hook.stockout_start": start, "hook.stockout_days": 7 * weeks})
    combos.append({LABEL: "B1_nolearning_stockout26w", "steering_parameters.perceived_treatment_success_prob": 0.8,
                   "hook.stockout_start": start, "hook.stockout_days": 182})
    end = start + 182
    collapse = {**_social(SOCIAL_STRONG), "hook.stockout_start": start, "hook.stockout_days": 182}
    combos += [
        {LABEL: "B4_collapse", **collapse},
        {LABEL: "B4_campaign", **collapse, "hook.campaign_day": end, "hook.campaign_share": 0.5},
        {LABEL: "B4_free_care", **collapse, "hook.free_care_start": end, "hook.free_care_days": 91},
        {LABEL: "B4_both", **collapse, "hook.campaign_day": end, "hook.campaign_share": 0.5,
         "hook.free_care_start": end, "hook.free_care_days": 91},
    ]
    key = {f"B1_s{SOCIAL_STRONG:g}_none", f"B1_s{SOCIAL_STRONG:g}_stockout26w",
           "B4_collapse", "B4_campaign", "B4_free_care", "B4_both"}
    return _rev(combos, key)


#: experiment 2's regimes with a drought or a flood: run with the environmental pathways at their sized
#: values, and, as the reference, without any (strategy "none" only)
ENV_REGIMES = ("F-fast", "F-slow", "D-fast", "D-slow", "C+F", "C+D")


def exp2_rev_combos() -> list[dict]:
    """2: every regime without a strategy, the strategies in five regimes, the drought and flood
    regimes with the environmental pathways (and without, as the reference), and the slow drought
    with one day of credit (R2's sensitivity). 40 seeds for no shock and the slow flood, the slow
    drought and the cycle with a drought (with the pathways, no strategy); 16 for the rest."""
    from experiments.scenarios.hooks import REGIMES, STRATEGIES, env_settings
    # the sized values written into each scenario (recorded with its results; workers need no state)
    sized = {f"hook.{k}": v for k, v in env_settings({"hook.env": True}).items()}
    env = lambda r: dict(sized) if r in ENV_REGIMES else {}
    combos = [{LABEL: f"{r}_none", "hook.regime": r, "hook.strategy": "none", **env(r)} for r in REGIMES]
    # the reference: no environmental pathway at all (the flood without its contamination pulse; its
    # latrines still overflow, as the flood's other channels stay). A regime whose pathways are all sized
    # to 1 (the flood, if its pulse is sized to none) would only repeat its main scenario: left out.
    off = {"hook.drought_hygiene": 1.0, "hook.first_rains": 1.0, "hook.flood_contamination": 1.0}
    flood_only = {"F-fast", "F-slow", "C+F"}
    combos += [{LABEL: f"{r}_none_noenv", "hook.regime": r, "hook.strategy": "none", **off} for r in ENV_REGIMES
               if not (r in flood_only and sized["hook.flood_contamination"] == 1.0)]
    for strategy in STRATEGIES[1:]:
        for r in EXP2_STRATEGY_REGIMES:
            combos.append({LABEL: f"{r}_{strategy}", "hook.regime": r, "hook.strategy": strategy, **env(r)})
    combos.append({LABEL: "D-slow_none_credit1d", "hook.regime": "D-slow", "hook.strategy": "none",
                   "economy.emergency_credit_days": 1.0, **env("D-slow")})
    # the sensitivity on drought hygiene: a mild six-month drought (x1.05 instead of x1.08)
    from experiments.scenarios.hooks import ENV_MILD_DROUGHT_HYGIENE
    if ENV_MILD_DROUGHT_HYGIENE is None:
        raise ValueError("2_rev needs hooks.ENV_MILD_DROUGHT_HYGIENE (step 1b of the revision)")
    combos.append({LABEL: "D-slow_none_mild", "hook.regime": "D-slow", "hook.strategy": "none",
                   **env("D-slow"), "hook.drought_hygiene": float(ENV_MILD_DROUGHT_HYGIENE)})
    return _rev(combos, {"R0_none", "F-slow_none", "D-slow_none", "C+D_none"})


def exp3b_rev_combos() -> list[dict]:
    """3B: every intervention against the default, 40 seeds each; and, as the sensitivity on the
    duration of protection against illness, the default and the booster with four-year waning."""
    combos = exp3b_combos()
    by = {c[LABEL]: c for c in combos}
    combos += [{**by["vacc_default"], LABEL: "vacc_default_wanes_4y", **ILLNESS_WANES_4Y},
               {**by["booster_9m"], LABEL: "booster_9m_wanes_4y", **ILLNESS_WANES_4Y}]
    return _rev(combos, {c[LABEL] for c in combos})


def ps_combos() -> list[dict]:
    """Step 3: the headline scenarios over the calibrated parameter sets (experiments.scenarios.param_sets)."""
    from experiments.scenarios.param_sets import ps_combos as combos
    return combos()


# ---------------------------------------------------------------------------
# Paper 1 (PLOS Computational Biology): the model, its calibration and validation
# ---------------------------------------------------------------------------

#: 3A's headline scenarios: today's vaccine against none (the out-of-sample validation: the calibration
#: ran without the vaccine), protection against infection 0.95 (depth), without the extra prior
#: infection, without the cut in severity, and protection against illness waning over four years
P1_3A = ("vacc_none", "vacc_default", "grid_v0.95_d616", "vacc_no_prior_infection_equivalent",
         "vacc_no_severity_reduction", "vacc_illness_wanes_4y")


def p1_3a_combos() -> list[dict]:
    """Paper 1: P1_3A, each exactly as in the 3A rerun but on 40 seeds (61001-61040)."""
    from experiments.orchestrator import REPS_KEY
    by = {c[LABEL]: c for c in exp3a_rev_combos()}
    return [{**by[lab], REPS_KEY: KEY_REPS} for lab in P1_3A]


def p1_ps_combos() -> list[dict]:
    """Paper 1: today's vaccine and none over the calibrated parameter sets (experiments.scenarios.param_sets)."""
    from experiments.scenarios.param_sets import p1_ps_combos as combos
    return combos()


def _trial_design(name: str, combos, description: str) -> "Design":
    from experiments.scenarios.vaccine_trial import TRIAL_STEPS, record_trial, trial_metrics
    return Design(name, combos, steps=TRIAL_STEPS, hooks=True, description=description,
                  step_callback=record_trial, metrics_fn=trial_metrics)


def _env_design(name: str, combos, description: str) -> "Design":
    from experiments.scenarios.env_sizing import SIZING_STEPS, env_metrics, record_env
    return Design(name, combos, steps=SIZING_STEPS, hooks=True, description=description,
                  step_callback=record_env, metrics_fn=env_metrics)


def env_round1_combos() -> list[dict]:
    from experiments.scenarios.env_sizing import round1_combos
    return round1_combos()


def env_round2_combos() -> list[dict]:
    from experiments.scenarios.env_sizing import round2_combos
    return round2_combos()


def env_round3_combos() -> list[dict]:
    from experiments.scenarios.env_sizing import round3_combos
    return round3_combos()


DESIGNS: dict[str, Design] = {
    "3a": Design("exp3a_vaccine_limits", exp3a_combos,
                 description="vaccine components and the protection grid (V_eff x duration)"),
    "1a": Design("exp1a_care_levers", exp1a_combos, hooks=True,
                 description="prices, free care, NHIS, LEAP transfers, lost earnings, perceived and actual "
                             "efficacy, emergency credit"),
    "1ar": Design("exp1a_care_levers_refine", exp1a_refine_combos, hooks=True,
                  description="1A's price edge refined: x0.001, x0.0025, x0.005, x0.0075"),
    "2": Design("exp2_shocks", exp2_combos, steps=EXP2_STEPS, hooks=True,
                description="weather-shock regimes x risk-management strategies"),
    "1b1": Design("exp1b1_stockouts", exp1b1_combos, steps=EXP1B_STEPS, hooks=True,
                  description="learning: stock-outs of 2-26 weeks, social learning off and on"),
    "1b2": Design("exp1b2_ramps", exp1b2_combos, steps=EXP1B_RAMP_STEPS, hooks=True,
                  description="learning: slow ramps of efficacy down and up (hysteresis, early warning)"),
    "1b4": Design("exp1b4_recovery", exp1b4_combos, steps=EXP1B_STEPS, hooks=True,
                  description="learning: what restores care-seeking after a collapse"),
    "3b": Design("exp3b_vaccine_interventions", exp3b_combos, hooks=True,
                 description="neonatal and timely doses, booster, WASH for the two lowest quintiles, OPV apart, "
                             "take by WASH (D4)"),
    "vt": _trial_design("vaccine_trial", trial_combos,
                        "the vaccine's efficacy by time since the last dose: half the children randomised to it"),
    "env1": _env_design("env_sizing_r1", env_round1_combos,
                        "sizing experiment 2's environmental pathways, round 1: three levels of each"),
    "env2": _env_design("env_sizing_r2", env_round2_combos,
                        "sizing experiment 2's environmental pathways, round 2: the sized values checked"),
    "env3": _env_design("env_sizing_r3", env_round3_combos,
                        "sizing experiment 2's environmental pathways, round 3: the flood's interpolated value"),
    # step 2 of the revision: one measured year, the adopted vaccine, seeds per scenario
    "3a_rev": Design("rev_3a_vaccine_limits", exp3a_rev_combos, steps=REV_STEPS,
                     description="3A rerun: the protection grid and the vaccine's components"),
    "1a_rev": Design("rev_1a_care_levers", exp1a_rev_combos, steps=REV_STEPS, hooks=True,
                     description="1A rerun: policies, perceived efficacy of care, credit"),
    "1b_rev": Design("rev_1b_learning", exp1b_rev_combos, steps=REV_STEPS, hooks=True,
                     description="1B rerun in one year: stock-outs (B1) and what restores trust (B4)"),
    "2_rev": Design("rev_2_shocks", exp2_rev_combos, steps=REV_STEPS, hooks=True,
                    description="2 rerun: regimes x strategies, with the environmental pathways"),
    "3b_rev": Design("rev_3b_vaccine_interventions", exp3b_rev_combos, steps=REV_STEPS, hooks=True,
                     description="3B rerun: interventions on the vaccine"),
    # step 3: the headline contrasts over the calibrated parameter sets (8 seeds each)
    "ps": Design("rev_ps_param_sets", ps_combos, steps=REV_STEPS, hooks=True,
                 description="step 3: the headline scenarios over the calibrated parameter sets"),
    # paper 1: the vaccine's validation and limits, and the spread over the calibrated parameter sets
    "p1_3a": Design("p1_3a_vaccine", p1_3a_combos, steps=REV_STEPS,
                    description="paper 1: 3A's headline scenarios, 40 seeds each"),
    "p1_ps": Design("p1_ps_param_sets", p1_ps_combos, steps=REV_STEPS,
                    description="paper 1: today's vaccine and none over the 10 other calibrated parameter sets"),
}

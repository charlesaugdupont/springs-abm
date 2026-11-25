"""
Tooling for the v5 experiments: emergency credit
for care (economy.emergency_credit_days, off by default), the efficacy of care
as parents perceive it, the day-by-day hooks and the run driver they need, the
scenario designs, and the experiments' extra outputs.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest
import torch

from config import SVEIRCONFIG
from abm.constants import AgentPropertyKeys as K
from abm.economy import cedi
from abm.state import AgentState
from abm.systems.care_seeking import CareSeekingSystem, _perceived_success
from abm.systems.economics import EconomicSystem
from experiments.orchestrator import SweepSpec, _run_one, set_param
from experiments.scenarios.common import care_metrics, weekly_series
from experiments.scenarios.designs import DESIGNS, LABEL, price
from experiments.scenarios.hooks import LEAP_PAYMENT_INTERVAL_DAYS, apply_hooks, leap_grant
from model_tests.conftest import GRID_ID


# ---------------------------------------------------------------------------
# Emergency credit (sensitivity D2)
# ---------------------------------------------------------------------------

def test_no_borrowing_by_default():
    assert SVEIRCONFIG.economy.emergency_credit_days == 0.0


def _sick_child_case(make_model, cash, credit_days):
    model = make_model(run=False, agents=600, overrides={"economy.emergency_credit_days": credit_days})
    nd = model.graph.ndata
    parent = int(nd[K.IS_PARENT].nonzero()[0])
    hh = nd[K.HOUSEHOLD_ID][parent]
    child = int(((nd[K.HOUSEHOLD_ID] == hh) & nd[K.IS_CHILD].bool()).nonzero()[0])
    nd[K.CASH][nd[K.HOUSEHOLD_ID] == hh] = cash
    nd[K.symptom_severity("rota")][child] = 0.6
    nd[K.illness_duration("rota")][child] = 3
    # an agent who values health only, facing a certain worsening, seeks care whenever it can
    nd[K.ALPHA][parent] = 0.1
    model.config.steering_parameters.natural_worsening_prob = 1.0
    cs = next(s for s in model.systems if isinstance(s, CareSeekingSystem))
    return model, cs, parent, child


def test_credit_pays_the_visit_of_a_household_without_cash(make_model):
    """A household without cash borrows the price. (One with a little cash, below
    the price, may borrow too, but its CPT weighs paying everything it has as the
    loss of its whole wealth term, so it rarely chooses to.)"""
    model, cs, parent, _ = _sick_child_case(make_model, cash=0.0, credit_days=10.0)
    nd = model.graph.ndata
    oop, _ = cedi.visit_cost(nd, parent, model.config.economy)
    assert oop > 0.0
    cs.update(model.graph)
    assert cs.care_sought == 1 and cs.could_not_afford == 0
    same = nd[K.HOUSEHOLD_ID] == nd[K.HOUSEHOLD_ID][parent]
    assert torch.all(nd[K.CASH][same] == 0.0)
    assert nd[K.DEBT][same].tolist() == pytest.approx([oop] * int(same.sum()), rel=1e-6)
    assert float(nd[K.DEBT][~same].abs().sum()) == 0.0
    rec, q = cs.daily_records[-1], int(nd[K.WEALTH_QUINTILE][parent])
    assert rec["visits_on_credit"][q] == 1 and rec["borrowed"][q] == pytest.approx(oop)
    assert rec["oop_paid"][q] == pytest.approx(oop)


def test_credit_has_a_limit(make_model):
    model, cs, parent, _ = _sick_child_case(make_model, cash=0.0, credit_days=1e-6)
    cs.update(model.graph)
    assert cs.care_sought == 0 and cs.could_not_afford == 1
    assert K.DEBT not in model.graph.ndata


def test_outstanding_debt_reduces_what_can_be_borrowed(make_model):
    model, cs, parent, _ = _sick_child_case(make_model, cash=0.0, credit_days=1.0)
    nd = model.graph.ndata
    target = float(nd[K.CONSUMPTION_TARGET][parent])
    cedi.add_household_debt(model.graph, parent, target)          # the whole limit is used
    cs.update(model.graph)
    assert cs.care_sought == 0 and cs.could_not_afford == 1


def _economy_state(cash, target, potential, household, debt):
    n = len(household)
    st = AgentState(n)
    st.ndata[K.HOUSEHOLD_ID] = torch.tensor(household)
    st.ndata[K.CASH] = torch.tensor(cash, dtype=torch.float32)
    st.ndata[K.CONSUMPTION_TARGET] = torch.tensor(target, dtype=torch.float32)
    st.ndata[K.EARNING_POTENTIAL] = torch.tensor(potential, dtype=torch.float32)
    st.ndata[K.HEALTH] = torch.ones(n)
    st.ndata[K.WORK_LOST] = torch.zeros(n)
    st.ndata[K.WEALTH_QUINTILE] = torch.zeros(n, dtype=torch.int64)
    st.ndata[K.WEALTH] = torch.zeros(n)
    st.ndata[K.CASH_CAP] = SVEIRCONFIG.economy.buffer_days * st.ndata[K.CONSUMPTION_TARGET]
    st.ndata[K.INSURED] = torch.ones(n, dtype=torch.bool)
    st.ndata[K.DEBT] = torch.tensor(debt, dtype=torch.float32)
    return st


def test_debt_is_repaid_before_saving():
    cfg = SVEIRCONFIG.model_copy(deep=True)
    cfg.economy.income_shock_sd = 0.0           # deterministic earnings
    # household 0 owes 20 and earns 15 above its target; household 1 owes nothing
    st = _economy_state(cash=[0.0, 0.0, 0.0], target=[10.0, 10.0, 10.0], potential=[25.0, 0.0, 25.0],
                        household=[0, 0, 1], debt=[20.0, 20.0, 0.0])
    eco = EconomicSystem(cfg)
    eco.update(st)
    assert st.ndata[K.DEBT].tolist() == pytest.approx([5.0, 5.0, 0.0])
    assert st.ndata[K.CASH].tolist() == pytest.approx([0.0, 0.0, 15.0])
    eco.update(st)
    assert st.ndata[K.DEBT].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert st.ndata[K.CASH].tolist() == pytest.approx([10.0, 10.0, 30.0])


# ---------------------------------------------------------------------------
# Perceived efficacy of care
# ---------------------------------------------------------------------------

def test_perceived_efficacy_defaults_to_the_actual_one():
    p = SVEIRCONFIG.steering_parameters
    assert p.perceived_treatment_success_prob is None
    assert _perceived_success(p) == p.treatment_success_prob
    assert _perceived_success(p.model_copy(update={"perceived_treatment_success_prob": 0.3})) == 0.3


# ---------------------------------------------------------------------------
# Hooks and the day-by-day driver
# ---------------------------------------------------------------------------

def test_leap_transfers(make_model):
    model = make_model(run=False, agents=600)
    nd = model.graph.ndata
    cash0, insured0 = nd[K.CASH].clone(), nd[K.INSURED].clone()
    q = nd[K.WEALTH_QUINTILE].numpy()
    hh = nd[K.HOUSEHOLD_ID].numpy()
    _, inv, size = np.unique(hh, return_inverse=True, return_counts=True)
    combo = {"hook.transfer_quintiles": "1,2", "hook.transfer_multiple": 2.0}
    apply_hooks(model, 0, combo)
    paid = (nd[K.CASH] - cash0).numpy()
    target = np.isin(q, [0, 1])
    assert target.any() and (~target).any()
    assert paid[target] == pytest.approx(2.0 * leap_grant(size)[inv][target], rel=1e-6)
    assert np.all(paid[~target] == 0.0)
    assert bool(nd[K.INSURED][torch.as_tensor(target)].all())
    assert bool((nd[K.INSURED][torch.as_tensor(~target)] == insured0[torch.as_tensor(~target)]).all())
    apply_hooks(model, 30, combo)
    assert np.array_equal((nd[K.CASH] - cash0).numpy(), paid), "no payment between pay days"
    apply_hooks(model, LEAP_PAYMENT_INTERVAL_DAYS, combo)
    assert (nd[K.CASH] - cash0).numpy()[target] == pytest.approx(2 * paid[target], rel=1e-6)


def test_leap_grant_schedule():
    assert leap_grant(np.array([1, 2, 3, 4, 5, 9])).tolist() == [392.0, 392.0, 474.0, 474.0, 630.0, 630.0]


def _record(tmp_path, step_callback, combo):
    spec = SweepSpec(name="driver", grid_id=GRID_ID, params=[], metrics_fn=_driver_metrics, reps=1, steps=40,
                     agents=600, output_dir=str(tmp_path), base_seed=23, step_callback=step_callback)
    out = _run_one((spec, combo, 0, "driver_rep0"))
    assert out is not None
    return out["record"]


def _driver_metrics(model):
    from experiments.perf.regression_check import fingerprint
    fp = fingerprint(model)
    return {k: (np.asarray(v).tobytes() if not isinstance(v, (int, float, dict)) else repr(v))
            for k, v in fp.items()}


def test_hook_driver_without_hooks_reproduces_a_plain_run(tmp_path):
    """The orchestrator's day-by-day driver (needed for hooks) with no hook
    settings in the scenario gives exactly the run of model.run()."""
    plain = _record(tmp_path, None, {LABEL: "x"})
    hooked = _record(tmp_path, apply_hooks, {LABEL: "x"})
    assert plain == hooked


# ---------------------------------------------------------------------------
# Designs and outputs
# ---------------------------------------------------------------------------

def _fake_sets(path, extra: dict | None = None):
    """Step 3's kept sets, as `param_sets sets` writes them: the adopted e7 and one other."""
    import json
    sets = {"rule": "test", "excluded": [], "sets": [
        {"name": "e7", "verification": "verify_v5b", "I_max": 2.68,
         "values": {"pathogens[rota].infection_prob_mean": 0.69}},
        {"name": "eA", "verification": "verify_v5c", "I_max": 2.9,
         "values": {"pathogens[rota].infection_prob_mean": 0.6,
                    "pathogens[rota].asymptomatic_relative_infectiousness|"
                    "pathogens[campy].asymptomatic_relative_infectiousness": 0.45,
                    "economy.cost_primary_insured": 30.0, **(extra or {})}}]}
    with open(path, "w") as f:
        json.dump(sets, f)


def _placeholders(monkeypatch, tmp_path=None):
    """The revised designs need the vaccine adopted in step 1, the sized pathways and step 3's kept
    parameter sets: stand-ins."""
    import os
    import experiments.scenarios.designs as D
    import experiments.scenarios.param_sets as P
    if D.VACCINE_REVISION is None:
        monkeypatch.setattr(D, "VACCINE_REVISION", {"illness_mechanics.immunity_factor_vaccine": 0.2})
    if None in H.ENV_SIZED.values():
        monkeypatch.setattr(H, "ENV_SIZED", {"drought_hygiene": 1.1, "first_rains": 2.0, "flood_contamination": 3.0})
    if H.ENV_MILD_DROUGHT_HYGIENE is None:
        monkeypatch.setattr(H, "ENV_MILD_DROUGHT_HYGIENE", 1.05)
    if tmp_path is not None and not os.path.exists(P.SETS_JSON):
        _fake_sets(tmp_path / "param_sets.json")
        monkeypatch.setattr(P, "SETS_JSON", str(tmp_path / "param_sets.json"))


@pytest.mark.parametrize("part", sorted(DESIGNS))
def test_designs_resolve(part, tmp_path, monkeypatch):
    _placeholders(monkeypatch, tmp_path)
    combos = DESIGNS[part].combos()
    labels = [c[LABEL] for c in combos]
    assert len(set(labels)) == len(labels)
    for combo in combos:
        cfg = SVEIRCONFIG.model_copy(deep=True)
        for path, value in combo.items():
            if path.startswith(("gsa.", "hook.", "shock.")):
                continue
            set_param(cfg, path, value)
    # the mixed-type columns the orchestrator will write
    pd.DataFrame.from_records(combos).to_parquet(tmp_path / "combos.parquet", index=False)


def test_price_multiples():
    assert price(1.0) == {}
    free = price(0.0)
    assert free and all(v == 0.0 for v in free.values())
    assert price(2.0)["economy.cost_drug_seller"] == 2 * SVEIRCONFIG.economy.cost_drug_seller


def test_default_cell_of_the_vaccine_grid_is_the_default_model():
    combos = {c[LABEL]: c for c in DESIGNS["3a"].combos()}
    assert set(combos["vacc_default"]) == {LABEL}
    rota = next(p for p in SVEIRCONFIG.pathogens if p.name == "rota")
    assert rota.vaccine_efficacy == 0.55 and rota.vaccine_protection_days == 616.0


def test_experiment_outputs(make_model):
    model = make_model(agents=600, steps=30)
    wk = weekly_series(model)
    assert wk and all(len(v) == 4 for v in wk.values())
    for key in ("wk_episodes_ended_treated", "wk_oop_paid", "wk_visits_on_credit", "wk_episodes_rota"):
        assert key in wk
    out = care_metrics(model, burn_in_days=0)
    assert 0.0 <= out["exp_treated_share"] <= 1.0 or math.isnan(out["exp_treated_share"])
    assert out["exp_visits_on_credit"] == 0.0 and "exp_debt_end_ghs" not in out


# ---------------------------------------------------------------------------
# Experiment 2: shock regimes and strategies
# ---------------------------------------------------------------------------

from experiments.scenarios import hooks as H  # noqa: E402


def test_regime_factors():
    one = {"income_town": 1.0, "income_zone": 1.0, "price": 1.0, "contamination": 1.0, "flood": False}
    assert H.regime_factors("R0", 1500) == one
    assert H.regime_factors("C+D", H.SHOCK_START_DAY - 1) == one, "nothing before the shocks start"
    d = H.regime_factors("D-slow", H.DROUGHT_START_DAY + 10)
    assert (d["income_town"], d["price"]) == pytest.approx((0.6, 1.2))
    end = H.DROUGHT_START_DAY + H.DROUGHT_DAYS
    half = H.regime_factors("D-slow", end + H.SLOW_RECOVERY_DAYS // 2)
    assert (half["income_town"], half["price"]) == pytest.approx((0.8, 1.1), abs=0.002)
    assert H.regime_factors("D-fast", end) == one
    fl = H.regime_factors("F-slow", H.FLOOD_START_DAY)
    assert fl["flood"] and fl["contamination"] == 10.0 and fl["income_zone"] == 0.5 and fl["income_town"] == 1.0
    fend = H.FLOOD_START_DAY + H.FLOOD_DAYS
    assert H.regime_factors("F-slow", fend + H.SLOW_RECOVERY_DAYS // 2)["income_zone"] == pytest.approx(0.75, abs=0.002)
    assert H.regime_factors("F-fast", fend) == one
    # the seasonal cycle: earnings peak in December, trough in June; contamination in the rains
    year4 = H.SHOCK_START_DAY
    assert H.regime_factors("C-hi", year4 + H.EARNINGS_PEAK_DOY)["income_town"] == pytest.approx(1.4)
    assert H.regime_factors("C-hi", year4 + (H.EARNINGS_PEAK_DOY - 182) % 365)["income_town"] == pytest.approx(0.6, abs=0.001)
    assert H.regime_factors("C-lo", year4 + 150)["contamination"] == 2.0
    assert H.regime_factors("C-lo", year4 + 230)["contamination"] == 1.0
    assert H.regime_factors("C-rare", year4 + 150)["contamination"] == 5.0
    assert H.regime_factors("C-rare", year4 + 365 + 150) == one, "one bad year in three"
    both = H.regime_factors("C+F", H.FLOOD_START_DAY + 1)
    assert both["contamination"] == 50.0 and both["flood"], "the minor rains and the flood compound"


def _exp2_model(make_model):
    return make_model(run=False, agents=600)


def test_r0_without_a_strategy_is_the_plain_model(tmp_path):
    plain = _record(tmp_path, None, {LABEL: "x"})
    hooked = _record(tmp_path, apply_hooks, {LABEL: "x", "hook.regime": "R0", "hook.strategy": "none"})
    assert plain == {k: v for k, v in hooked.items() if not k.startswith("hook.")}


def test_flood_channels_switch_on_and_off(make_model):
    from abm.environment.water_network import SAN_OPEN
    model = _exp2_model(make_model)
    net = model.grid_environment.water_network
    eco, steer = model.config.economy, model.config.steering_parameters
    base_prob, base_visit = steer.shock_daily_prob, eco.public_visit_day_fraction
    combo = {"hook.regime": "F-slow", "hook.strategy": "none"}
    apply_hooks(model, 0, combo)
    st = model._exp_state
    zone = torch.as_tensor(np.flatnonzero(st["zone_hh"]))
    san0 = net.hh_sanitation.clone()
    apply_hooks(model, H.FLOOD_START_DAY, combo)
    assert steer.shock_daily_prob == pytest.approx(10 * base_prob)
    assert eco.public_visit_day_fraction == pytest.approx(min(1.0, 2 * base_visit))
    assert bool((net.hh_sanitation[zone] == SAN_OPEN).all())
    outside = torch.as_tensor(np.flatnonzero(~st["zone_hh"]))
    assert torch.equal(net.hh_sanitation[outside], san0[outside])
    nd = model.graph.ndata
    za = st["zone_agent"]
    assert torch.allclose(nd[K.EARNING_POTENTIAL][za], 0.5 * st["potential0"][za])
    assert torch.equal(nd[K.EARNING_POTENTIAL][~za], st["potential0"][~za])
    apply_hooks(model, H.FLOOD_START_DAY + H.FLOOD_DAYS, combo)
    assert steer.shock_daily_prob == base_prob and eco.public_visit_day_fraction == base_visit
    assert torch.equal(net.hh_sanitation, san0)


def test_strategies(make_model):
    from abm.environment.water_network import HH_GROUND, HH_PIPED, HH_SURFACE, SAN_IMPROVED_ONSITE
    # storage: a larger buffer and 5% of income saved
    m = _exp2_model(make_model)
    apply_hooks(m, 0, {"hook.regime": "R0", "hook.strategy": "storage"})
    st, nd = m._exp_state, m.graph.ndata
    assert torch.allclose(nd[K.CASH_CAP], 1.5 * st["cap0"])
    assert torch.allclose(nd[K.CONSUMPTION_TARGET], st["target0"] / 1.05)
    # adjustment: consumption cut by a fifth while earnings are down (drought, prices +20%)
    m = _exp2_model(make_model)
    combo = {"hook.regime": "D-slow", "hook.strategy": "adjustment"}
    apply_hooks(m, 0, combo)
    st, nd = m._exp_state, m.graph.ndata
    assert torch.equal(nd[K.CONSUMPTION_TARGET], st["target0"])
    apply_hooks(m, H.DROUGHT_START_DAY, combo)
    assert torch.allclose(nd[K.CONSUMPTION_TARGET], st["target0"] * 1.2 * 0.8)
    assert m.config.economy.poverty_line_daily == pytest.approx(1.2 * st["poverty0"][0])
    # infrastructure: piped water and flood-proof latrines in the flood zone
    m = _exp2_model(make_model)
    net = m.grid_environment.water_network
    combo = {"hook.regime": "F-slow", "hook.strategy": "infrastructure"}
    apply_hooks(m, 0, combo)
    zone = torch.as_tensor(np.flatnonzero(m._exp_state["zone_hh"]))
    assert bool((net.hh_source[zone] == net.piped_index).all()) and bool((net.hh_type[zone] == HH_PIPED).all())
    apply_hooks(m, H.FLOOD_START_DAY, combo)
    assert bool((net.hh_sanitation[zone] == SAN_IMPROVED_ONSITE).all())
    # mobility: surface and groundwater users switch to packaged water while contamination is high
    m = _exp2_model(make_model)
    net = m.grid_environment.water_network
    combo = {"hook.regime": "F-fast", "hook.strategy": "mobility"}
    apply_hooks(m, 0, combo)
    src0, t0 = net.hh_source.clone(), m._exp_state["type0"]
    users = (t0 == HH_SURFACE) | (t0 == HH_GROUND)
    apply_hooks(m, H.FLOOD_START_DAY, combo)
    assert bool((net.hh_source[users] == net.packaged_index).all())
    assert torch.equal(net.hh_source[~users], src0[~users])
    apply_hooks(m, H.FLOOD_START_DAY + H.FLOOD_DAYS, combo)
    assert torch.equal(net.hh_source, src0)


def test_flood_zone_outputs(make_model):
    model = make_model(run=False, agents=600)
    combo = {"hook.regime": "F-fast", "hook.strategy": "none"}
    for day in range(21):
        apply_hooks(model, day, combo)
        model.step()
    out = H.zone_series(model)
    assert len(out["wk_fz_episodes"]) == 3 and len(out["wk_fz_zero_cash_share"]) == 3
    assert out["fz_households"] == float(model._exp_state["zone_hh"].sum())


# ---------------------------------------------------------------------------
# Experiment 1B: learning how well care works
# ---------------------------------------------------------------------------

def test_no_learning_by_default():
    p = SVEIRCONFIG.steering_parameters
    assert p.belief_learning is False and p.belief_social_weight == 0.0


def test_beliefs_start_at_the_probabilities_and_learn_from_outcomes(make_model):
    model = make_model(run=False, agents=600, overrides={"steering_parameters.belief_learning": True,
                                                        "steering_parameters.belief_learning_rate": 0.5,
                                                        "steering_parameters.belief_social_weight": 0.2})
    cs = next(s for s in model.systems if isinstance(s, CareSeekingSystem))
    model.step()
    b = cs._beliefs
    p = model.config.steering_parameters
    assert b is not None and len(b["success"]) == int(b["hh_of"].max()) + 1
    rec = cs.daily_records[0]
    assert rec["belief_success"] == pytest.approx(p.treatment_success_prob)
    assert rec["belief_worsen"] == pytest.approx(p.natural_worsening_prob)
    # a failed visit: the household's belief halves the gap to 0, its neighbours' move a tenth of it
    s0 = b["success"].clone()
    parent = int(model.graph.ndata[K.IS_PARENT].nonzero()[0])
    h = int(b["hh_of"][parent])
    cs._learn(parent, "success", 0.0)
    assert float(b["success"][h]) == pytest.approx(0.5 * float(s0[h]))
    nb = b["neighbours"][h]
    assert h not in set(nb.tolist())
    if nb.numel():
        assert b["success"][nb].tolist() == pytest.approx((s0[nb] * 0.9).tolist())
    others = torch.ones(len(s0), dtype=torch.bool)
    others[h] = False
    others[nb] = False
    assert torch.equal(b["success"][others], s0[others])
    # neighbours are the households within the social-visit radius
    home = model.graph.ndata[K.HOME_LOCATION][b["first_agent"]].double()
    d = (home - home[h]).norm(dim=1)
    within = set(torch.nonzero(d <= p.social_interaction_radius).flatten().tolist()) - {h}
    assert set(nb.tolist()) == within


def test_beliefs_replace_the_probabilities_in_the_decision(make_model):
    """A household convinced that care never works and waiting never harms
    does not seek care, whatever its cash."""
    model, cs, parent, _ = _sick_child_case(make_model, cash=5000.0, credit_days=0.0)
    cs.update(model.graph)
    assert cs.care_sought == 1, "without learning, this parent seeks care (test_credit_* setting)"
    model, cs, parent, _ = _sick_child_case(make_model, cash=5000.0, credit_days=0.0)
    model.config.steering_parameters.belief_learning = True
    cs._init_beliefs(model.graph)
    h = int(cs._beliefs["hh_of"][parent])
    cs._beliefs["success"][h] = 0.0
    cs._beliefs["worsen"][h] = 0.0
    cs.update(model.graph)
    assert cs.decisions_faced == 1 and cs.care_sought == 0


# ---------------------------------------------------------------------------
# Experiment 1B hooks: stock-outs, ramps, campaigns, free care
# ---------------------------------------------------------------------------

def test_efficacy_schedules():
    so = {"hook.stockout_start": 100, "hook.stockout_days": 14}
    assert [H.efficacy_on(d, so, 0.8) for d in (99, 100, 113, 114)] == [0.8, 0.2, 0.2, 0.8]
    ramp = {"hook.ramp_start": 100, "hook.ramp_days": 10, "hook.ramp_low": 0.1}
    assert H.efficacy_on(99, ramp, 0.8) == 0.8
    assert H.efficacy_on(105, ramp, 0.8) == pytest.approx(0.45)
    assert H.efficacy_on(110, ramp, 0.8) == pytest.approx(0.1)
    assert H.efficacy_on(115, ramp, 0.8) == pytest.approx(0.45)
    assert H.efficacy_on(120, ramp, 0.8) == 0.8


def test_stockout_campaign_and_free_care_hooks(make_model):
    model = make_model(run=False, agents=600, overrides={"steering_parameters.belief_learning": True})
    steer, eco = model.config.steering_parameters, model.config.economy
    combo = {"hook.stockout_start": 2, "hook.stockout_days": 3, "hook.campaign_day": 4, "hook.campaign_share": 1.0,
             "hook.free_care_start": 1, "hook.free_care_days": 2}
    prices0 = {f: getattr(eco, f) for f in H.PRICE_FIELDS}
    apply_hooks(model, 0, combo)
    model.step()
    assert steer.treatment_success_prob == 0.8 and eco.cost_drug_seller == prices0["cost_drug_seller"]
    apply_hooks(model, 1, combo)
    assert all(getattr(eco, f) == 0.0 for f in H.PRICE_FIELDS)
    apply_hooks(model, 2, combo)
    assert steer.treatment_success_prob == 0.2
    apply_hooks(model, 3, combo)
    assert {f: getattr(eco, f) for f in H.PRICE_FIELDS} == prices0
    cs = H._care_system(model)
    cs._beliefs["success"][:] = 0.1
    apply_hooks(model, 4, combo)
    assert torch.all(cs._beliefs["success"] == 0.8), "the campaign restores the true belief (not the stock-out's)"
    apply_hooks(model, 5, combo)
    assert steer.treatment_success_prob == 0.8


# ---------------------------------------------------------------------------
# Experiment 3B hooks: booster, WASH upgrade, take by WASH
# ---------------------------------------------------------------------------

def test_booster_renews_protection(make_model):
    from abm.constants import Compartment
    from abm.systems.demography import DAYS_PER_MONTH
    model = make_model(run=False, agents=600)
    model.step()
    rota = H._rota(model)
    v, nd = rota._vax, model.graph.ndata
    age_days = nd[K.AGE].double() * DAYS_PER_MONTH
    due = nd[K.IS_CHILD].bool() & (v["given"] >= 3) & (age_days >= 274)
    assert bool(due.any())
    combo = {"hook.booster_age_days": 274, "hook.booster_coverage": 1.0, "hook.booster_take": 1.0}
    apply_hooks(model, 1, combo)
    status = nd[K.status("rota")]
    assert bool(v["protected"][due].all()) and bool((v["wane_day"][due] > rota._vax_day).all())
    was_s_or_r = (status[due] == Compartment.VACCINATED)
    assert bool(was_s_or_r.any())
    wane = v["wane_day"].clone()
    apply_hooks(model, 2, combo)
    assert torch.equal(v["wane_day"], wane), "one booster per child"


def test_wash_upgrade_by_quintile(make_model):
    from abm.environment.water_network import HH_PIPED, SAN_IMPROVED_ONSITE, SAN_SEWER
    model = make_model(run=False, agents=600)
    net = model.grid_environment.water_network
    q = H._household_quintile(model)
    san0, src0 = net.hh_sanitation.clone(), net.hh_source.clone()
    apply_hooks(model, 0, {"hook.wash_quintiles": "1,2"})
    t = torch.as_tensor(np.isin(q, [0, 1]))
    assert bool(t.any())
    assert bool((net.hh_source[t] == net.piped_index).all()) and bool((net.hh_type[t] == HH_PIPED).all())
    assert bool(((net.hh_sanitation[t] == SAN_IMPROVED_ONSITE) | (net.hh_sanitation[t] == SAN_SEWER)).all())
    assert torch.equal(net.hh_sanitation[~t], san0[~t]) and torch.equal(net.hh_source[~t], src0[~t])


def test_take_by_wash_redraws_only_doses_not_given(make_model):
    model = make_model(run=False, agents=600)
    combo = {"hook.take_by_wash": True}
    apply_hooks(model, 0, combo)
    m = model._exp_take["m"]
    assert m > 1.0
    model.step()
    v = H._rota(model)._vax
    takes0, given = v["takes"].clone(), v["given"].clone()
    apply_hooks(model, 1, combo)
    k = takes0.shape[1]
    done = torch.arange(k)[None, :] < given[:, None]
    assert torch.equal(v["takes"][done], takes0[done])


def test_hysteresis_rule_needs_a_gap_that_survives_the_slower_ramp():
    """B2's rule: paths differ by > 3 SE over >= 3 consecutive levels, and the gap does not
    shrink at half speed (ratio >= 3/4, not smaller by > 2 SE). A gap that halves at half
    speed is a lag, not hysteresis."""
    from experiments.scenarios.analysis import hysteresis_1b2
    rng = np.random.default_rng(1)

    def paths(gap_full, gap_half):
        rows = []
        for sc, gap in (("B2_s0.6_ramp365", gap_full), ("B2_s0.6_ramp730", gap_half), ("B2_s0_ramp365", 0.0)):
            for seed in range(8):
                for e in np.round(np.arange(0.1, 0.81, 0.1), 2):
                    base, g = 0.5 + 0.3 * e, (gap if 0.2 <= e <= 0.6 else 0.0)
                    for leg, v in (("down", base), ("up", base - g)):
                        rows.append({"scenario": sc, "seed": seed, "leg": leg, "efficacy": e,
                                     "care": v + rng.normal(0, 0.01), "belief": v + rng.normal(0, 0.01)})
        return pd.DataFrame(rows)

    def verdict(gf, gh):
        h = hysteresis_1b2(paths(gf, gh)).set_index("scenario")
        return h.loc["B2_s0.6_ramp365"], h.loc["B2_s0_ramp365"]

    full, none = verdict(0.1, 0.1)
    assert full["paths_differ_care"] and not full["gap_shrinks_care"] and full["hysteresis_care"]
    assert not none["paths_differ_care"] and not none["hysteresis_care"]
    lag, _ = verdict(0.1, 0.05)
    assert lag["paths_differ_care"] and lag["gap_shrinks_care"] and not lag["hysteresis_care"]
    assert abs(lag["half_speed_ratio_care"] - 0.5) < 0.05
    flat, _ = verdict(0.0, 0.0)
    assert not flat["hysteresis_care"] and not flat["hysteresis_belief"]


def test_environmental_pathways_switch_on_and_off(make_model):
    """Step 1b: drought hygiene while the drought lasts, the first rains for four
    weeks after it, the flood's pulse resized; nothing outside them, and nothing
    without hook.env or an explicit multiplier."""
    model = _exp2_model(make_model)
    rota, campy, crypto = (next(p for p in model.pathogens if p.name == n) for n in ("rota", "campy", "crypto"))
    combo = {"hook.regime": "D-slow", "hook.drought_hygiene": 1.1, "hook.first_rains": 2.0}
    apply_hooks(model, 0, combo)
    assert not any(p.route_multipliers for p in model.pathogens)
    apply_hooks(model, H.DROUGHT_START_DAY, combo)
    assert rota.route_multipliers == {"h2h": 1.1}
    assert crypto.route_multipliers == {"household": 1.1, "community": 1.1}
    assert campy.route_multipliers == {"fecal_oral": 1.1, "food_borne": 1.1}
    end = H.DROUGHT_START_DAY + H.DROUGHT_DAYS
    apply_hooks(model, end, combo)
    assert rota.route_multipliers == {} and crypto.route_multipliers == {"water": 2.0}
    assert campy.route_multipliers == {"zoonotic": 2.0}
    apply_hooks(model, end + H.FIRST_RAINS_DAYS, combo)
    assert not any(p.route_multipliers for p in model.pathogens)
    # no multipliers without the settings, and the flood's pulse as before unless resized
    assert H.env_settings({}) == {"drought_hygiene": 1.0, "first_rains": 1.0, "flood_contamination": 10.0}
    assert H.regime_factors("F-fast", H.FLOOD_START_DAY, 3.0)["contamination"] == 3.0
    if None in H.ENV_SIZED.values():
        with pytest.raises(ValueError):
            H.env_settings({"hook.env": True})
    else:
        assert H.env_settings({"hook.env": True}) == {k: float(v) for k, v in H.ENV_SIZED.items()}


def test_a_route_multiplier_scales_only_its_route(make_model):
    model = make_model(run=False, agents=600)
    crypto = next(p for p in model.pathogens if p.name == "crypto")
    idx = torch.arange(50)
    base_w = crypto._relative_susceptibility(model.graph, idx, route="water")
    base_h = crypto._relative_susceptibility(model.graph, idx, route="household")
    crypto.route_multipliers["water"] = 2.0
    assert torch.allclose(crypto._relative_susceptibility(model.graph, idx, route="water"), 2.0 * base_w)
    assert torch.equal(crypto._relative_susceptibility(model.graph, idx, route="household"), base_h)
    per_agent = torch.ones(model.graph.num_nodes())
    per_agent[:10] = 3.0
    crypto.route_multipliers["water"] = per_agent
    s = crypto._relative_susceptibility(model.graph, idx, route="water")
    assert torch.allclose(s[:10], 3.0 * base_w[:10]) and torch.allclose(s[10:], base_w[10:])


def test_the_sizing_records_count_infections_and_episodes(make_model):
    from experiments.scenarios.env_sizing import env_metrics, record_env
    model = make_model(run=False, agents=600, seed=5)
    model.infection_incidence.clear()
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}
    start = {p.name: model.graph.ndata[K.num_infections(p.name)].sum().item() for p in model.pathogens}
    child_mask = model.graph.ndata[K.IS_CHILD]
    for day in range(30):
        record_env(model, day, {"hook.regime": "R0"})
        model.step(child_mask=child_mask)
    m = env_metrics(model)
    for p in model.pathogens:
        series = [m[f"env_inf_{p.name}_{g}"] for g in ("zc", "za", "oc", "oa")]
        assert all(len(s) == 30 for s in series)
        assert sum(map(sum, series)) >= model.graph.ndata[K.num_infections(p.name)].sum().item() - start[p.name]
        assert len(m[f"env_ep_{p.name}_zone"]) == 30
    n_ep = sum(sum(m[f"env_ep_{p.name}_{z}"]) for p in model.pathogens for z in ("zone", "out"))
    assert n_ep == len([e for e in model.get_child_episode_log() if e["timestep"] < 30])


def test_the_revised_designs(monkeypatch):
    """Step 2: one measured year, the adopted vaccine in every scenario (unless the scenario sets the
    same key), 40 or 16 seeds, the pathways' values written into experiment 2's scenarios."""
    import experiments.scenarios.designs as D
    from experiments.orchestrator import REPS_KEY
    _placeholders(monkeypatch)
    vacc = D.VACCINE_REVISION
    for part in ("3a_rev", "1a_rev", "1b_rev", "2_rev", "3b_rev"):
        design = DESIGNS[part]
        assert design.steps == 1_095
        for c in design.combos():
            assert c[REPS_KEY] in (D.KEY_REPS, D.REST_REPS)
            for k, v in vacc.items():
                assert k in c
    by = {c[LABEL]: c for c in DESIGNS["3a_rev"].combos()}
    assert by["vacc_no_severity_reduction"]["illness_mechanics.immunity_factor_vaccine"] == 0.0
    assert by["vacc_default"][REPS_KEY] == D.KEY_REPS and by["grid_v0.30_d475"][REPS_KEY] == D.REST_REPS
    by = {c[LABEL]: c for c in DESIGNS["2_rev"].combos()}
    assert by["D-slow_none"]["hook.drought_hygiene"] == H.ENV_SIZED["drought_hygiene"]
    assert by["D-slow_none_noenv"]["hook.flood_contamination"] == 1.0 and "hook.env" not in by["D-slow_none"]
    assert "hook.drought_hygiene" not in by["R0_none"] and "hook.drought_hygiene" not in by["C-hi_none"]
    mild = by["D-slow_none_mild"]
    assert mild["hook.drought_hygiene"] == H.ENV_MILD_DROUGHT_HYGIENE
    assert mild["hook.first_rains"] == H.ENV_SIZED["first_rains"] and mild[REPS_KEY] == D.REST_REPS
    by = {c[LABEL]: c for c in DESIGNS["1b_rev"].combos()}
    assert by["B4_campaign"]["hook.campaign_day"] == H.SHOCK_START_DAY + 182
    assert not any(k.startswith("actual_") or k.startswith("price_x") for k in
                   (c[LABEL] for c in DESIGNS["1a_rev"].combos()))


def test_a_scenario_can_set_its_own_number_of_seeds(tmp_path):
    from experiments.orchestrator import REPS_KEY, SweepSpec, _combo_id, run_sweep
    from model_tests.test_sweep_resume import stamp_metrics
    assert _combo_id({LABEL: "a", REPS_KEY: 40}, 3) == _combo_id({LABEL: "a"}, 3)
    spec = SweepSpec(name="reps", grid_id="7d9ce7c720a6", params=[], metrics_fn=stamp_metrics, reps=1, steps=2,
                     agents=200, output_dir=str(tmp_path), n_cores=1, checkpoint_every=1, resume=True)
    df = run_sweep(spec, [{LABEL: "a", REPS_KEY: 2}, {LABEL: "b"}])
    assert sorted(df[LABEL]) == ["a", "a", "b"] and sorted(df.loc[df[LABEL] == "a", "rep"]) == [0, 1]


def test_the_booster_renews_waning_protection_against_illness(make_model):
    """With vaccine_illness_protection_days set, a booster that takes renews the protection against
    illness too (the flag set, its end moved to at least today + e * T)."""
    model = make_model(run=False, agents=3000, seed=7,
                       overrides={"pathogens[rota].vaccine_illness_protection_days": 30.0})
    rota = next(p for p in model.pathogens if p.name == "rota")
    nd = model.graph.ndata
    key = K.vaccine_illness_protected("rota")
    combo = {"hook.booster_age_days": 120, "hook.booster_coverage": 1.0, "hook.booster_take": 1.0}
    rota._susceptible_to_vaccinated(model.graph)          # day 0: the doses the children's ages have passed
    v = rota._vax
    before = nd[key].clone()
    apply_hooks(model, 0, combo)                           # boosted the same day (coverage and take 1)
    renewed = model._exp_booster["done"]
    assert renewed.any() and (renewed & ~before).any(), "some boosted children had lost their protection"
    assert torch.all(nd[key][renewed]), "a booster that took protects against illness again"
    assert torch.all(v["ill_until"][renewed] > rota._vax_day)


# ---------------------------------------------------------------------------
# Step 3: the headline contrasts over calibrated parameter sets
# ---------------------------------------------------------------------------

def test_the_scenarios_over_the_parameter_sets(tmp_path, monkeypatch):
    """Every headline scenario, defined as in step 2, with each kept set but the adopted one; tied
    calibrated paths expanded; a scenario's own value wins only to make care free."""
    import experiments.scenarios.param_sets as P
    from experiments.orchestrator import REPS_KEY
    _placeholders(monkeypatch)
    _fake_sets(tmp_path / "sets.json")
    monkeypatch.setattr(P, "SETS_JSON", str(tmp_path / "sets.json"))
    combos = DESIGNS["ps"].combos()
    assert [c[LABEL] for c in combos] == [f"eA|{lab}" for lab in P.HEADLINE]
    scen = P.scenarios()
    for c in combos:
        _, s2 = scen[c[P.SCENARIO_KEY]]
        assert c[REPS_KEY] == P.PS_REPS and c[P.SET_KEY] == "eA"
        assert c["pathogens[rota].infection_prob_mean"] == 0.6
        assert c["pathogens[campy].asymptomatic_relative_infectiousness"] == 0.45
        for k, v in s2.items():
            if k not in (LABEL, REPS_KEY):
                assert c[k] == v                      # the scenario exactly as in step 2
        assert c["economy.cost_primary_insured"] == (0.0 if c[P.SCENARIO_KEY] == "price_free" else 30.0)
    _fake_sets(tmp_path / "sets.json", extra={"economy.emergency_credit_days": 2.0})
    with pytest.raises(ValueError, match="credit_1d"):
        DESIGNS["ps"].combos()


def test_the_rule_that_keeps_a_parameter_set(tmp_path, monkeypatch):
    import json
    import experiments.scenarios.param_sets as P
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(), new.mkdir()
    pd.DataFrame({LABEL: ["verify_v5b_e7", "verify_v5b_e4", "verify_v5b_e3"], "I_max": [2.68, 3.51, 3.05],
                  "I_second": [2.4, 2.9, 2.6], "rota_fadeout_reps": [0, 0, 0]}).to_csv(
        old / "verify_v5b_implausibility.csv", index=False)
    pd.DataFrame({LABEL: ["verify_v5c_e9", "verify_v5c_e10", "verify_v5c_s0"], "I_max": [2.5, 2.9, 2.0],
                  "I_second": [2.0, 2.5, 1.0], "rota_fadeout_reps": [0, 2, 0]}).to_csv(
        new / "verify_v5c_implausibility.csv", index=False)
    cands = [{LABEL: n, "pathogens[rota].infection_prob_mean": 0.5 + i / 100}
             for i, n in enumerate(("e3", "e4", "e7", "e9", "e10"))]
    (new / "cands.json").write_text(json.dumps(cands))
    monkeypatch.setattr(P, "EARLIER_VERIFICATIONS", (("verify_v5b", str(old)),))
    monkeypatch.setattr(P, "STEP3_DIR", str(new))
    monkeypatch.setattr(P, "CANDIDATES_N24", str(new / "cands.json"))
    monkeypatch.setattr(P, "SETS_JSON", str(tmp_path / "sets.json"))
    out = P.select_sets()
    # e4: I 3.51; e10: rotavirus faded out in 2 seeds; s0 is no candidate
    assert [s["name"] for s in out["sets"]] == ["e7", "e9", "e3"]
    assert {e["name"] for e in out["excluded"]} == {"e4", "e10"}
    assert P.load_sets()[2]["values"] == {"pathogens[rota].infection_prob_mean": 0.5}


def test_the_robustness_rule(monkeypatch):
    """Robust: in at least 80% of the sets (the adopted one included, on its first 8 seeds) the same sign as
    at the adopted set and beyond 2 SE; tested only if the adopted set's effect is beyond 2 SE."""
    import experiments.scenarios.param_sets as P
    monkeypatch.setattr(P, "CONTRASTS", (
        ("vacc_episodes", "3A", "x", "R0_none", "vacc_none", "episodes_0_24m"),
        ("free_care", "1A", "x", "price_free", "R0_none", "care_seeking"),
        ("adjustment_cash", "2", "x", "D-slow_adjustment", "R0_adjustment", "care_seeking",
         ("D-slow_none", "R0_none"))))
    rng = np.random.default_rng(3)

    def rows(seeds, vacc_effect, care_effect, adj):
        out = []
        for s in seeds:
            base = 0.18 + rng.normal(0, 0.005)
            for lab, ep, care in (("R0_none", base, 0.6), ("vacc_none", base - vacc_effect, 0.6),
                                  ("price_free", base, 0.6 + care_effect), ("R0_adjustment", base, 0.65),
                                  ("D-slow_adjustment", base, 0.65 - 0.2 + adj), ("D-slow_none", base, 0.4)):
                out.append({LABEL: lab, "seed": s, "rota_symptomatic_episodes_per_child_year_0_24m": ep,
                            "econ_episode_care_seeking_rate": care + rng.normal(0, 0.005)})
        return out

    step2 = pd.DataFrame(rows(range(61001, 61017), -0.09, 0.4, 0.1))
    ps = []
    for name, vacc in (("eA", -0.08), ("eB", 0.0), ("eC", -0.1), ("eD", -0.07)):
        for r in rows(range(61001, 61009), vacc, 0.35, 0.0 if name == "eB" else 0.12):
            ps.append({**r, P.SET_KEY: name, P.SCENARIO_KEY: r[LABEL], LABEL: f"{name}|{r[LABEL]}"})
    res = P.analyse(pd.DataFrame(ps), step2)
    v = res["verdict"].set_index("contrast")
    # vaccine: 4 of 5 sets (eB has none) -> 80%: robust; care: 5 of 5; adjustment: eB 0 -> 4 of 5
    assert v.loc["vacc_episodes", "tested"] and v.loc["vacc_episodes", "share_meeting"] == pytest.approx(0.8)
    assert v.loc["vacc_episodes", "robust"]
    assert v.loc["free_care", "robust"] and v.loc["free_care", "share_meeting"] == 1.0
    assert v.loc["adjustment_cash", "adopted_diff"] == pytest.approx(0.1, abs=0.01)
    assert v.loc["adjustment_cash", "share_meeting"] == pytest.approx(0.8)
    by = res["by_set"]
    assert set(by["set"]) == {"e7", "eA", "eB", "eC", "eD"} and (by[by["set"] == "e7"]["n"] == 8).all()


def test_the_outcomes_of_the_contrasts_read_their_windows():
    import experiments.scenarios.param_sets as P
    w = P._windows()
    n = 156
    zero = np.zeros(n)
    zero[w["drought"][0]:w["drought"][1]] = 0.5
    belief = np.full(n, 0.8)
    belief[w["stockout"][1] - 4:w["stockout"][1]] = 0.5
    belief[w["stockout"][1] + 13:w["stockout"][1] + 17] = 0.6
    ended = np.ones(n)
    care = np.zeros(n)
    care[w["drought"][0]:] = 1.0
    r = {"wk_zero_cash_share": zero, "wk_belief_success": belief, "wk_episodes_ended": ended,
         "wk_episodes_ended_with_care": care, "wk_poverty": np.r_[np.zeros(n - 4), np.ones(4)]}
    assert P.outcome("zero_cash_drought", r) == 0.5
    assert P.outcome("care_drought_to_end", r) == 1.0
    assert P.outcome("poverty_end", r) == 1.0
    assert P.outcome("belief_end_of_stockout", r) == 0.5
    assert P.outcome("belief_13w_after", r) == pytest.approx(0.6)
    assert w["stockout"] == (104, 130) and w["drought"] == (112, 139)


def test_choosing_a_pathway_s_value():
    """Interpolated between the bracketing levels while the IRR rises (the reference as level 1, IRR 1);
    otherwise from the weighted line through IRR 1 at level 1 (or with a free intercept for the flood)."""
    from experiments.scenarios.env_sizing import choose, fit_line
    e = lambda irr, se=0.05: {"episodes": {"irr": irr, "se_log": se}}
    rising = {1.05: e(1.11), 1.1: e(1.265), 1.2: e(1.605)}
    c = choose(rising, 1.08, level_one=True)
    assert c["how"] == "interpolated" and c["chosen"] == pytest.approx(np.exp(np.log(1.05) * 0.08 / 0.11))
    noisy = {1.5: e(0.9, 0.2), 2.0: e(1.6, 0.15), 3.0: e(2.0, 0.1)}
    c = choose(noisy, 1.26, level_one=True)
    line = fit_line(noisy, 1.26, True)
    assert c["how"] == "line" and not c["rising"] and c["chosen"] == pytest.approx(1 + 0.26 / line["b"])
    exact = {1.0: e(1.1), 3.0: e(1.3), 10.0: e(2.0)}
    f = fit_line(exact, 1.23, False)
    assert f["a"] == pytest.approx(1.1) and f["b"] == pytest.approx(0.1) and f["value"] == pytest.approx(2.3)


def test_rounds_two_and_three_run_their_values(monkeypatch):
    import experiments.scenarios.env_sizing as E
    _placeholders(monkeypatch)
    vals = {"drought_hygiene": 1.04, "first_rains": 1.6, "flood_contamination": 4.0, "mild_drought_hygiene": 1.02}
    levels = {"first_rains": (1.4, 1.6, 1.9), "flood_contamination": (3.0, 4.0, 6.0)}
    by = {c[LABEL]: c for c in E.round2_combos(vals, levels)}
    assert set(by) == {"R0", "D_ref", "D_hyg", "D_hyg_mild", "D_both", "D_rain1.4", "D_rain1.6", "D_rain1.9",
                       "F_c3", "F_c4", "F_c6"}
    assert by["D_both"]["hook.drought_hygiene"] == 1.04 and by["D_both"]["hook.first_rains"] == 1.6
    assert by["F_c4"]["hook.flood_contamination"] == 4.0 and by["D_hyg_mild"]["hook.drought_hygiene"] == 1.02
    assert "hook.first_rains" not in by["D_hyg"]
    assert all(c["illness_mechanics.immunity_factor_vaccine"] == 0.2 for c in by.values())
    with pytest.raises(ValueError, match="must contain"):
        E.round2_combos(vals, {"first_rains": (1.4, 1.9), "flood_contamination": (3.0, 4.0)})
    # the frozen designs: round 2 as it ran, round 3 the flood at its value interpolated in round 2
    r2 = {c[LABEL]: c for c in DESIGNS["env2"].combos()}
    assert {"D_rain1.6", "D_rain1.9", "D_rain2.3", "F_c1", "F_c3", "F_c10"} <= set(r2)
    assert r2["D_hyg"]["hook.drought_hygiene"] == 1.036 and r2["D_hyg_mild"]["hook.drought_hygiene"] == 1.022
    r3 = {c[LABEL]: c for c in DESIGNS["env3"].combos()}
    assert set(r3) == {"R0", "F_c7.6"} and r3["F_c7.6"]["hook.flood_contamination"] == 7.6


def test_expected_episodes_follow_the_model(make_model):
    """The sizing's expected episodes: the sum over children's infections of the model's probability that
    each becomes an episode; over a run they match the observed episodes within chance."""
    from experiments.scenarios.env_sizing import env_metrics, record_env
    model = make_model(run=False, agents=2000, seed=5)
    model.infection_incidence.clear()
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}
    child_mask = model.graph.ndata[K.IS_CHILD]
    days = 300
    for day in range(days):
        record_env(model, day, {"hook.regime": "R0"})
        model.step(child_mask=child_mask)
    m = env_metrics(model)
    total = 0.0
    for p in ("rota", "campy", "crypto"):
        obs = sum(sum(m[f"env_ep_{p}_{z}"]) for z in ("zone", "out"))
        exp = sum(sum(m[f"env_exp_{p}_{z}"]) for z in ("zone", "out"))
        assert len(m[f"env_exp_{p}_zone"]) == days
        assert abs(obs - exp) <= 4 * np.sqrt(max(exp, 1.0)) + 1, (p, obs, exp)
        total += exp
    assert total > 20


def test_the_parameter_sets_figure_draws(tmp_path):
    import experiments.scenarios.param_sets as P
    from experiments.scenarios.paper_figures import plot_param_sets
    rows, ver = [], []
    for i, c in enumerate(P.CONTRASTS[:3]):
        for s, d in (("e7", 1.0), ("eA", 0.8), ("eB", -0.1)):
            rows.append({"contrast": c[0], "set": s, "diff": d, "se": 0.2, "meets": d > 0.4})
        ver.append({"contrast": c[0], "experiment": c[1], "what": c[2], "adopted_diff": 1.0, "tested": i < 2,
                    "sets": 3, "share_meeting": 2 / 3, "robust": False})
    paths = plot_param_sets(pd.DataFrame(rows), pd.DataFrame(ver), str(tmp_path))
    assert all(os.path.getsize(p) > 0 for p in paths.values())


def test_experiment_2_variants_and_references(monkeypatch):
    """The reruns' variants; a flood whose pulse is sized to none has no separate reference (it would repeat
    its main scenario)."""
    import experiments.scenarios.designs as D
    from experiments.scenarios.analysis import exp2_variant
    assert [exp2_variant(x) for x in ("D-slow_none", "D-slow_none_noenv", "D-slow_none_credit1d",
                                      "D-slow_none_mild", "C+D_adjustment")] == ["main", "noenv", "credit", "mild",
                                                                                "main"]
    _placeholders(monkeypatch)
    monkeypatch.setattr(H, "ENV_SIZED", {**H.ENV_SIZED, "flood_contamination": 1.0})
    labels = {c[LABEL] for c in D.exp2_rev_combos()}
    assert "F-slow_none_noenv" not in labels and "C+F_none_noenv" not in labels
    assert "D-slow_none_noenv" in labels and "C+D_none_noenv" in labels and "F-slow_none" in labels
    monkeypatch.setattr(H, "ENV_SIZED", {**H.ENV_SIZED, "flood_contamination": 3.0})
    assert "F-slow_none_noenv" in {c[LABEL] for c in D.exp2_rev_combos()}


def test_paper1_parts_are_3a_scenarios_on_40_and_8_seeds():
    """Paper 1's parts reuse the 3A rerun's scenarios unchanged: p1_3a on 40 seeds, p1_ps (today's vaccine
    and none) on 8 seeds over the 10 other calibrated parameter sets."""
    from experiments.orchestrator import REPS_KEY
    from experiments.scenarios import designs as D
    by = {c[D.LABEL]: c for c in D.exp3a_rev_combos()}
    p1 = D.p1_3a_combos()
    assert [c[D.LABEL] for c in p1] == list(D.P1_3A)
    for c in p1:
        assert c[REPS_KEY] == 40
        assert {k: v for k, v in c.items() if k != REPS_KEY} == {k: v for k, v in by[c[D.LABEL]].items()
                                                                 if k != REPS_KEY}
    ps = D.p1_ps_combos()
    assert len(ps) == 20 and all(c[REPS_KEY] == 8 for c in ps)

"""
The household economy in 2026 Ghana cedis (config.EconomyConfig, abm/economy/cedi.py).

The legacy economy stays the default and is checked byte for byte by
experiments/perf/regression_check.py; these tests cover the cedi mode: its
initialisation (roles by age, strata, earnings anchored on consumption), the
daily cash rule, care costs, and its outputs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from config import SVEIRCONFIG
from abm.constants import AgentPropertyKeys as K, Compartment
from abm.economy import cedi
from abm.state import AgentState
from abm.systems.care_seeking import CareSeekingSystem
from abm.systems.economics import EconomicSystem
from experiments.metrics import economy_metrics

CEDI = {"economy.model": "cedi"}


def _hh_constant(model, key) -> bool:
    nd = model.graph.ndata
    hh, v = nd[K.HOUSEHOLD_ID].numpy(), nd[key].numpy()
    order = np.argsort(hh, kind="stable")
    hh, v = hh[order], v[order]
    starts = np.r_[0, np.flatnonzero(np.diff(hh)) + 1]
    return all(np.all(v[a:b] == v[a]) for a, b in zip(starts, np.r_[starts[1:], len(hh)]))


def test_legacy_is_the_default_and_adds_nothing(make_model):
    assert SVEIRCONFIG.economy.model == "legacy"
    model = make_model(steps=3)
    for key in (K.CASH, K.WEALTH_QUINTILE, K.EARNING_POTENTIAL, K.ECON_AGE_GROUP):
        assert key not in model.graph.ndata
    assert economy_metrics(model) == {}


def test_cedi_keeps_the_legacy_population(make_model):
    """The economy draws from its own stream, after everything else."""
    a = make_model(run=False)
    b = make_model(run=False, overrides=CEDI)
    for key in (K.HOUSEHOLD_ID, K.IS_CHILD, K.IS_PARENT, K.AGE, K.HEALTH, K.X, K.Y, K.OWNS_POULTRY,
                K.EXPOSURE_FRAILTY, K.status("rota"), K.num_infections("campy")):
        assert torch.equal(a.graph.ndata[key], b.graph.ndata[key]), key


def test_roles_by_age(make_model):
    model = make_model(run=False, agents=3000, overrides=CEDI)
    nd = model.graph.ndata
    group = nd[K.ECON_AGE_GROUP].numpy()
    child = nd[K.IS_CHILD].numpy().astype(bool)
    assert np.all(group[child] == cedi.UNDER_5) and np.all(group[~child] >= cedi.AGE_5_14)
    _, first = np.unique(nd[K.HOUSEHOLD_ID].numpy(), return_index=True)
    assert np.all(group[first] >= cedi.AGE_15_64), "every household head is aged 15+"
    assert np.all(group[nd[K.IS_PARENT].numpy().astype(bool)] >= cedi.AGE_15_64)
    shares = [np.mean(group[~child] == g) for g in (cedi.AGE_5_14, cedi.AGE_15_64, cedi.AGE_65_PLUS)]
    assert np.allclose(shares, model.config.economy.age_group_shares_5plus, atol=0.03)
    earns = nd[K.EARNING_POTENTIAL].numpy() > 0
    assert np.array_equal(earns, group >= cedi.AGE_15_64), "everyone aged 15+ earns, nobody else"


def test_household_attributes_are_shared(make_model):
    model = make_model(run=False, overrides=CEDI)
    for key in (K.WEALTH_QUINTILE, K.CASH, K.CONSUMPTION_TARGET, K.INSURED, K.WEALTH):
        assert _hh_constant(model, key), key


def test_quintile_consumption_matches_glss_without_spread(make_model):
    """With no spread within quintiles, each quintile's consumption target per
    person is exactly GLSS 7's per-capita consumption."""
    model = make_model(run=False, agents=3000, overrides={**CEDI, "economy.earning_spread_sdlog": 0.0})
    nd = model.graph.ndata
    hh, first = np.unique(nd[K.HOUSEHOLD_ID].numpy(), return_index=True)
    size = np.bincount(np.unique(nd[K.HOUSEHOLD_ID].numpy(), return_inverse=True)[1])
    q = nd[K.WEALTH_QUINTILE].numpy()[first]
    target = nd[K.CONSUMPTION_TARGET].numpy()[first]
    for k, c in enumerate(model.config.economy.consumption_per_capita_daily):
        if np.any(q == k):
            assert target[q == k].sum() / size[q == k].sum() == pytest.approx(c, rel=1e-5)


def test_initial_cash_and_wealth_term(make_model):
    model = make_model(run=False, overrides=CEDI)
    nd, eco = model.graph.ndata, model.config.economy
    cash, target = nd[K.CASH], nd[K.CONSUMPTION_TARGET]
    assert torch.all(cash >= 0) and torch.all(cash <= eco.buffer_days * target + 1e-3)
    w = nd[K.WEALTH]
    assert torch.allclose(w, torch.clamp(cash / (eco.wealth_scale_days * target), 0, 1))
    assert torch.equal(nd[K.INITIAL_WEALTH], w)


def _economy_state(cash, target, potential, health, work_lost, household):
    st = AgentState(len(household))
    st.ndata[K.HOUSEHOLD_ID] = torch.tensor(household)
    st.ndata[K.CASH] = torch.tensor(cash, dtype=torch.float32)
    st.ndata[K.CONSUMPTION_TARGET] = torch.tensor(target, dtype=torch.float32)
    st.ndata[K.EARNING_POTENTIAL] = torch.tensor(potential, dtype=torch.float32)
    st.ndata[K.HEALTH] = torch.tensor(health, dtype=torch.float32)
    st.ndata[K.WORK_LOST] = torch.tensor(work_lost, dtype=torch.float32)
    st.ndata[K.WEALTH_QUINTILE] = torch.zeros(len(household), dtype=torch.int64)
    st.ndata[K.WEALTH] = torch.zeros(len(household))
    st.ndata[K.CASH_CAP] = SVEIRCONFIG.economy.buffer_days * st.ndata[K.CONSUMPTION_TARGET]
    st.ndata[K.INSURED] = torch.ones(len(household), dtype=torch.bool)
    return st


def test_daily_cash_rule():
    """cash <- clip(cash + income - target, 0, buffer x target), income =
    potential x health x (1 - work lost); W follows; work lost resets."""
    cfg = SVEIRCONFIG.model_copy(deep=True)
    cfg.economy.model = "cedi"
    cfg.economy.income_shock_sd = 0.0     # deterministic earnings for this check
    B, S = cfg.economy.buffer_days, cfg.economy.wealth_scale_days
    # household 0: two earners, one at half health and half a day lost; saves
    # household 1: earns nothing, has 5 GH¢ against a target of 20 -> 0
    # household 2: at the buffer, earns more than its target -> stays at the buffer
    st = _economy_state(
        cash=[100.0, 100.0, 5.0, 5.0, B * 30.0], target=[50.0, 50.0, 20.0, 20.0, 30.0],
        potential=[40.0, 30.0, 0.0, 0.0, 60.0], health=[1.0, 0.5, 1.0, 1.0, 1.0],
        work_lost=[0.0, 0.5, 0.0, 0.0, 0.0], household=[0, 0, 1, 1, 2])
    sys_ = EconomicSystem(cfg)
    sys_.update(st)
    income0 = 40.0 + 30.0 * 0.5 * 0.5
    expect = [100.0 + income0 - 50.0, 0.0, B * 30.0]
    assert st.ndata[K.CASH].tolist() == pytest.approx([expect[0], expect[0], 0.0, 0.0, expect[2]])
    assert st.ndata[K.WEALTH].tolist() == pytest.approx(
        [min(1, expect[0] / (S * 50))] * 2 + [0.0, 0.0, min(1.0, expect[2] / (S * 30))])
    assert torch.all(st.ndata[K.WORK_LOST] == 0)
    rec = sys_.daily_records[-1]
    assert float(rec["zero_cash_households"].sum()) == 1.0
    assert float(rec["below_target_households"].sum()) == 1.0   # household 1 consumed 5 < 20


def test_paying_for_care_reduces_household_cash():
    st = _economy_state(cash=[100.0, 100.0, 7.0], target=[10.0, 10.0, 10.0], potential=[0, 0, 0],
                        health=[1, 1, 1], work_lost=[0, 0, 0], household=[0, 0, 1])
    cedi.add_household_cash(st, torch.tensor(1), -55.0, 30.0)
    assert st.ndata[K.CASH].tolist() == pytest.approx([45.0, 45.0, 7.0])
    assert st.ndata[K.WEALTH].tolist()[:2] == pytest.approx([45.0 / 300.0] * 2)
    cedi.add_household_cash(st, torch.tensor(2), -55.0, 30.0)
    assert st.ndata[K.CASH].tolist()[2] == 0.0


def _sick_child_case(make_model, cash):
    model = make_model(run=False, agents=600, overrides=CEDI)
    nd = model.graph.ndata
    parent = int(nd[K.IS_PARENT].nonzero()[0])
    hh = nd[K.HOUSEHOLD_ID][parent]
    child = int(((nd[K.HOUSEHOLD_ID] == hh) & nd[K.IS_CHILD].bool()).nonzero()[0])
    nd[K.CASH][nd[K.HOUSEHOLD_ID] == hh] = cash
    nd[K.symptom_severity("rota")][child] = 0.6
    nd[K.illness_duration("rota")][child] = 3
    cs = next(s for s in model.systems if isinstance(s, CareSeekingSystem))
    return model, cs, parent, child


def test_unaffordable_visit_is_counted_and_not_paid(make_model):
    model, cs, parent, child = _sick_child_case(make_model, cash=1.0)
    cs.update(model.graph)
    assert cs.could_not_afford == 1 and cs.care_sought == 0
    q = int(model.graph.ndata[K.WEALTH_QUINTILE][parent])
    assert cs.daily_records[-1]["could_not_afford"][q] == 1
    assert bool(cs._blocked_this_episode[child]), "the episode is marked as blocked"
    assert float(model.graph.ndata[K.CASH][parent]) == pytest.approx(1.0)


def test_a_visit_costs_cash_and_working_time(make_model):
    model, cs, parent, _ = _sick_child_case(make_model, cash=5000.0)
    eco = model.config.economy
    nd = model.graph.ndata
    # an agent who values health only, facing a certain worsening, always seeks care
    nd[K.ALPHA][parent] = 0.1
    model.config.steering_parameters.natural_worsening_prob = 1.0
    cs.update(model.graph)
    assert cs.care_sought == 1
    oop, day_fraction = cedi.visit_cost(nd, parent, eco)
    assert float(nd[K.CASH][parent]) == pytest.approx(5000.0 - oop)
    assert float(nd[K.WORK_LOST][parent]) == pytest.approx(day_fraction)


def test_visit_cost_by_provider_and_cover():
    eco = SVEIRCONFIG.economy
    st = AgentState(6)
    st.ndata[K.PROVIDER] = torch.tensor([cedi.PRIMARY, cedi.PRIMARY, cedi.HOSPITAL, cedi.HOSPITAL,
                                         cedi.PRIVATE, cedi.DRUG_SELLER])
    st.ndata[K.INSURED] = torch.tensor([True, False, True, False, True, True])
    got = [cedi.visit_cost(st.ndata, torch.tensor(i), eco) for i in range(6)]
    assert got == [(eco.cost_primary_insured, eco.public_visit_day_fraction),
                   (eco.cost_facility_uninsured, eco.public_visit_day_fraction),
                   (eco.cost_hospital_insured, eco.public_visit_day_fraction),
                   (eco.cost_facility_uninsured, eco.public_visit_day_fraction),
                   (eco.cost_private, eco.private_visit_day_fraction),
                   (eco.cost_drug_seller, eco.drug_seller_visit_day_fraction)]


def test_usual_provider_follows_the_quintile_shares(make_model):
    model = make_model(run=False, agents=6000, overrides=CEDI)
    assert _hh_constant(model, K.PROVIDER)
    nd = model.graph.ndata
    _, first = np.unique(nd[K.HOUSEHOLD_ID].numpy(), return_index=True)
    prov, q = nd[K.PROVIDER].numpy()[first], nd[K.WEALTH_QUINTILE].numpy()[first]
    expected = np.asarray(model.config.economy.provider_shares_by_quintile)[q].mean(axis=0)
    observed = np.bincount(prov, minlength=cedi.N_PROVIDERS) / len(prov)
    assert np.allclose(observed, expected, atol=0.04)


def test_income_shocks_are_persistent_with_mean_one():
    cfg = SVEIRCONFIG.model_copy(deep=True)
    cfg.economy.model = "cedi"
    cfg.economy.income_shock_sd, cfg.economy.income_shock_halflife_days = 0.3, 30.0
    sys_ = EconomicSystem(cfg)
    days = np.stack([sys_._income_shock(4000).numpy() for _ in range(200)])
    assert days.mean() == pytest.approx(1.0, abs=0.03)
    log = np.log(days)
    lag1 = np.corrcoef(log[:-1].ravel(), log[1:].ravel())[0, 1]
    assert lag1 == pytest.approx(0.5 ** (1 / 30), abs=0.01)
    assert np.log(days).std() == pytest.approx(0.3, abs=0.02)
    cfg.economy.income_shock_sd = 0.0
    assert EconomicSystem(cfg)._income_shock(10) is None


def test_economy_metrics(make_model):
    model = make_model(steps=40, overrides=CEDI)
    m = economy_metrics(model, burn_in_days=10)
    for key in ("econ_poverty_headcount", "econ_mean_cash_ghs", "econ_zero_cash_share", "econ_blocked_episode_share",
                "econ_episode_care_seeking_rate", "econ_care_seeking_rate_q3", "econ_poverty_headcount_q1"):
        assert key in m
    assert 0.0 <= m["econ_poverty_headcount"] <= 1.0
    eco = next(s for s in model.systems if isinstance(s, EconomicSystem))
    assert len(eco.daily_records) == 40
    nd = model.graph.ndata
    assert torch.all(nd[K.CASH] >= 0)
    assert torch.all(nd[K.CASH] <= model.config.economy.buffer_days * nd[K.CONSUMPTION_TARGET] + 1e-2)
    assert _hh_constant(model, K.CASH)


def test_econ_campaign_is_wired():
    from experiments.calibration import history_matching as hm
    from experiments.orchestrator import set_param
    c = hm.CAMPAIGNS["econ"]
    assert c.base_overrides["economy.model"] == "cedi"
    assert c.base_overrides["pathogens[rota].vaccination_rate"] == 0.0     # as in the v4 calibration
    assert [t.metric for t in c.targets] == [f"econ_care_seeking_rate_q{q}" for q in range(1, 6)]
    cfg = SVEIRCONFIG.model_copy(deep=True)
    for path, value in c.base_overrides.items():
        set_param(cfg, path, value)
    for path, (lo, hi, scale) in c.params.items():
        assert path.startswith("economy.") and lo < hi and scale in ("lin", "log")
        set_param(cfg, path, lo)          # a real, valid config field
        set_param(cfg, path, hi)


def test_buffer_grows_with_wealth_only_when_asked(make_model):
    flat = make_model(run=False, agents=3000, overrides=CEDI)
    nd = flat.graph.ndata
    assert torch.allclose(nd[K.CASH_CAP], flat.config.economy.buffer_days * nd[K.CONSUMPTION_TARGET], rtol=1e-5)
    steep = make_model(run=False, agents=3000, overrides={**CEDI, "economy.buffer_wealth_elasticity": 1.0})
    nd = steep.graph.ndata
    hh, first, inv = np.unique(nd[K.HOUSEHOLD_ID].numpy(), return_index=True, return_inverse=True)
    size = np.bincount(inv)
    per_person = nd[K.CONSUMPTION_TARGET].numpy()[first] / size
    days = nd[K.CASH_CAP].numpy()[first] / nd[K.CONSUMPTION_TARGET].numpy()[first]
    c_ref = steep.config.economy.consumption_per_capita_daily[2]
    assert np.allclose(days, steep.config.economy.buffer_days * per_person / c_ref, rtol=1e-4)
    assert torch.all(nd[K.CASH] <= nd[K.CASH_CAP] + 1e-3)


def test_econ2_campaign_adds_the_money_gradient():
    from experiments.calibration import history_matching as hm
    c = hm.CAMPAIGNS["econ2"]
    assert "economy.buffer_wealth_elasticity" in c.params and len(c.params) == 7
    metrics = [t.metric for t in c.targets]
    assert metrics[:5] == [f"econ_care_seeking_rate_q{q}" for q in range(1, 6)]
    assert metrics[5:] == [f"econ_cannot_pay_facility_share_q{q}" for q in range(1, 6)]
    zs = [t.z for t in c.targets[5:]]
    assert zs == sorted(zs, reverse=True), "the money barrier falls with wealth"

"""
Tests for adults' own susceptibility and infectious period
(PathogenConfig.adult_relative_susceptibility, adult_infectious_duration_scale)
and the adult vs young-child carriage record (SVEIRModel.carriage_by_group,
experiments.metrics.carriage_metrics).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from abm.constants import AgentPropertyKeys as K, Compartment


def _pathogen(model, name):
    return next(p for p in model.pathogens if p.name == name)


def test_adults_default_to_the_childrens_floor(make_model):
    model = make_model(steps=1, agents=600, run=False)
    for p in model.pathogens:
        assert p.config.adult_relative_susceptibility is None
        idx = torch.arange(model.graph.num_nodes())
        s = p._relative_susceptibility(model.graph, idx)
        adult = ~model.graph.ndata[K.IS_CHILD].bool()
        assert torch.allclose(s[adult], torch.full_like(s[adult], float(p.config.immunity_floor)))


def test_adult_susceptibility_sets_adults_and_leaves_children_alone(make_model):
    base = make_model(steps=1, agents=600, run=False)
    model = make_model(steps=1, agents=600, run=False,
                       overrides={"pathogens[rota].adult_relative_susceptibility": 0.04})
    idx = torch.arange(model.graph.num_nodes())
    child = model.graph.ndata[K.IS_CHILD].bool()
    s = _pathogen(model, "rota")._relative_susceptibility(model.graph, idx)
    s0 = _pathogen(base, "rota")._relative_susceptibility(base.graph, idx)
    assert torch.allclose(s[~child], torch.full_like(s[~child], 0.04))
    assert torch.equal(s[child], s0[child])
    # the other pathogens are untouched
    for name in ("campy", "crypto"):
        assert torch.equal(_pathogen(model, name)._relative_susceptibility(model.graph, idx),
                           _pathogen(base, name)._relative_susceptibility(base.graph, idx))


def test_adult_susceptibility_without_a_floor_leaves_children_unclamped(make_model):
    model = make_model(steps=1, agents=600, run=False,
                       overrides={"pathogens[rota].immunity_floor": None,
                                  "pathogens[rota].adult_relative_susceptibility": 0.1,
                                  "pathogens[rota].infant_susceptibility_at_birth": 1.0})
    rota = _pathogen(model, "rota")
    model.graph.ndata[K.num_infections("rota")][:] = 10      # exp(-0.45 * 10) ~ 0.011
    idx = torch.arange(model.graph.num_nodes())
    child = model.graph.ndata[K.IS_CHILD].bool()
    s = rota._relative_susceptibility(model.graph, idx)
    assert torch.allclose(s[child], torch.full_like(s[child], math.exp(-0.45 * 10)))
    assert torch.allclose(s[~child], torch.full_like(s[~child], 0.1))


def test_adult_duration_scale_shortens_only_adults_infections(make_model):
    # recovery 0.5 and scale 0.5: adults recover with probability 1, children 0.5
    model = make_model(steps=1, agents=2000, run=False,
                       overrides={"pathogens[crypto].recovery_rate": 0.5,
                                  "pathogens[crypto].adult_infectious_duration_scale": 0.5})
    crypto = _pathogen(model, "crypto")
    st = model.graph.ndata[K.status("crypto")]
    st[:] = Compartment.INFECTIOUS
    crypto._infectious_to_recovered(model.graph)
    child = model.graph.ndata[K.IS_CHILD].bool()
    assert bool((st[~child] == Compartment.RECOVERED).all())
    still = (st[child] == Compartment.INFECTIOUS).float().mean().item()
    assert 0.35 < still < 0.65


def test_adult_duration_scale_draws_the_same_random_numbers(make_model):
    """The scale changes adults' recovery threshold, not the draws: with
    recovery 0 for everyone the run is the default run exactly."""
    off = {"pathogens[crypto].recovery_rate": 0.0}
    a = make_model(steps=30, agents=800, overrides=off)
    b = make_model(steps=30, agents=800, overrides={**off, "pathogens[crypto].adult_infectious_duration_scale": 0.41})
    for p in ("rota", "campy", "crypto"):
        assert torch.equal(a.graph.ndata[K.status(p)], b.graph.ndata[K.status(p)])


def test_carriage_record_counts_adults_and_well_older_children(make_model):
    from experiments.metrics import carriage_metrics
    model = make_model(steps=1, agents=1500, run=False)
    nd = model.graph.ndata
    child = nd[K.IS_CHILD].bool()
    older = child & (nd[K.AGE] >= 12.0)
    adults = (~child).nonzero(as_tuple=True)[0]
    kids = older.nonzero(as_tuple=True)[0]
    st = nd[K.status("rota")]
    st[:] = Compartment.SUSCEPTIBLE
    st[adults[:30]] = Compartment.INFECTIOUS
    st[kids[:10]] = Compartment.INFECTIOUS
    nd[K.symptom_severity("rota")][:] = 0
    nd[K.symptom_severity("rota")][kids[:5]] = 0.5           # ill: not in a carriage survey
    model._init_age_structured_records()
    model._record_carriage_by_group()
    out = carriage_metrics(model, burn_in_days=0)
    n_adult, n_well = len(adults), int(older.sum()) - 5
    assert out["rota_adult_prevalence_post_burnin"] == pytest.approx(30 / n_adult)
    assert out["rota_child_carriage_12_59m_post_burnin"] == pytest.approx(5 / n_well)
    assert out["rota_adult_to_child_carriage_ratio_post_burnin"] == pytest.approx((30 / n_adult) / (5 / n_well))
    rho = _pathogen(model, "rota").config.asymptomatic_relative_infectiousness
    assert 0.0 < out["rota_adult_infectious_weight_share_post_burnin"] <= 1.0
    w = _pathogen(model, "rota").infectiousness_weights(model.graph)
    assert out["rota_adult_infectious_weight_share_post_burnin"] == pytest.approx(
        float(w[adults].sum() / w.sum()))
    assert float(w[adults[:30]].sum()) == pytest.approx(30 * rho)


def test_carriage_metrics_are_recorded_every_day(make_model):
    from experiments.metrics import calibration_metrics
    model = make_model(steps=40, agents=800)
    assert all(len(v) == 40 for v in model.carriage_by_group.values())
    out = calibration_metrics(model, burn_in_days=10)
    for p in ("rota", "campy", "crypto"):
        assert f"{p}_adult_to_child_carriage_ratio_post_burnin" in out
    assert np.isnan(calibration_metrics(model, burn_in_days=40)["rota_adult_prevalence_post_burnin"])


def test_adult_pilot_campaign_is_wired_per_pathogen():
    from experiments.calibration import history_matching as hm
    c = hm.CAMPAIGNS["adult_pilot"]
    metrics = [t.metric for t in c.targets]
    assert {m.split("_")[0] for m in metrics} == {"rota", "crypto"}
    assert set(hm.ADULT_CARRIAGE_TARGETS) <= set(metrics)
    for m in metrics:
        p = m.split("_")[0]
        inputs = c.emulator_inputs(m)
        assert inputs and all(q.startswith(f"pathogens[{p}].") for q in inputs)
    assert set(c.params) == set(hm.ADULT_PILOT_PARAMS)


# ---------------------------------------------------------------------------
# Contact cap (steering_parameters.max_contacts_per_visit)
# ---------------------------------------------------------------------------

def _cap_case(make_model):
    from types import SimpleNamespace
    model = make_model(steps=1, agents=600, run=False)
    rota = _pathogen(model, "rota")
    hh = torch.tensor([1, 1, 2, 3, 4, 5])
    state = SimpleNamespace(ndata={K.HOUSEHOLD_ID: hh})
    loc = torch.tensor([0, 0, 0, 0, 0, 1])
    w = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    total = torch.zeros(2).index_add_(0, loc, w)[loc]
    return rota, state, loc, w, total


def test_cap_scales_only_other_households(make_model):
    rota, state, loc, w, total = _cap_case(make_model)
    p = rota._capped_pressure(state, loc, 2, w, total, 2.0, exclude_same_household=False)
    # agent 1: its household member (1) in full + 2 of the 3 others' weight 2
    assert p.tolist() == pytest.approx([2 / 3 * 2, 1 + 2 / 3 * 2, 1.0, 1.0, 1.5, 0.0])
    q = rota._capped_pressure(state, loc, 2, w, total, 2.0, exclude_same_household=True)
    assert q.tolist() == pytest.approx([2 / 3 * 2, 2 / 3 * 2, 1.0, 1.0, 1.5, 0.0])


def test_a_cap_above_the_crowd_is_the_original_rule(make_model):
    rota, state, loc, w, total = _cap_case(make_model)
    p = rota._capped_pressure(state, loc, 2, w, total, 100.0, exclude_same_household=False)
    assert p.tolist() == pytest.approx((total - w).tolist())


def test_the_cap_cuts_crowd_transmission(make_model):
    on = {"steering_parameters.max_contacts_per_visit": 5.0}
    a = make_model(steps=60, agents=1500)
    b = make_model(steps=60, agents=1500, overrides=on)
    n = lambda m: int(m.graph.ndata[K.num_infections("rota")][~m.graph.ndata[K.IS_CHILD].bool()].sum())
    assert n(b) < n(a)


def test_adult_pilot_cap_campaign_shares_the_cap():
    from experiments.calibration import history_matching as hm
    c = hm.CAMPAIGNS["adult_pilot_cap"]
    for m in (t.metric for t in c.targets):
        p = m.split("_")[0]
        inputs = c.emulator_inputs(m)
        assert inputs[0] == "steering_parameters.max_contacts_per_visit"
        assert all(q.startswith(f"pathogens[{p}].") for q in inputs[1:])


def test_v4_campaign_is_complete_and_valid():
    import numpy as np
    from config import SVEIRCONFIG
    from experiments.calibration import history_matching as hm
    from experiments.orchestrator import set_param
    c = hm.CAMPAIGNS["v4"]
    assert len(c.params) == 29 and len(c.targets) == 31
    assert c.base_overrides["steering_parameters.max_contacts_per_visit"] == 3.5
    assert all(c.base_overrides[k] == v for k, v in hm.V3_LITERATURE_FIXED.items())
    assert not set(hm.V3_LITERATURE_FIXED) & set(c.params)
    for t in c.targets:
        assert set(c.emulator_inputs(t.metric)) <= set(c.params), t.metric
    cfg = SVEIRCONFIG.model_copy(deep=True)
    for k, v in c.base_overrides.items():
        set_param(cfg, k, v)
    for k, (lo, hi, _) in c.params.items():
        for part in k.split("|"):
            set_param(cfg, part, float(np.sqrt(lo * hi)) if lo > 0 else hi / 2)

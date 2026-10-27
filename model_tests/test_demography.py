"""
Tests for the age structure: daily ageing, under-5 turnover, age-dependent
school attendance, age-specific susceptibility (infant ramp, immunity plateau,
adults), initial infection histories, household environmental-exposure
heterogeneity, and the birth-cohort metrics built on them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from abm.constants import Activity, AgentPropertyKeys, Compartment
from abm.evidence.maled import cumulative_infections_by_age, INFECTION_RATE_BY_AGE_PER_100CM
from abm.systems.child_illness import ChildIllnessSystem
from abm.systems.demography import DAYS_PER_MONTH, DemographySystem
from experiments.metrics import first_infection_metrics


def _children(model):
    return model.graph.ndata[AgentPropertyKeys.IS_CHILD].bool()


# ---------------------------------------------------------------------------
# Ageing and turnover
# ---------------------------------------------------------------------------

def test_ages_advance_one_day_per_step(make_model):
    model = make_model(steps=30, agents=800, run=False)
    g = model.graph
    start = g.ndata[AgentPropertyKeys.AGE].clone()
    for _ in range(30):
        model.step()
    not_renewed = g.ndata[AgentPropertyKeys.BIRTH_DAY] < 0
    expected = start + 30 / DAYS_PER_MONTH
    assert torch.allclose(g.ndata[AgentPropertyKeys.AGE][not_renewed], expected[not_renewed], atol=1e-3)


def test_child_reaching_five_is_renewed_as_a_naive_newborn(make_model):
    model = make_model(steps=5, agents=800, run=False)
    g = model.graph
    child = _children(model).nonzero(as_tuple=True)[0][0]
    g.ndata[AgentPropertyKeys.AGE][child] = 60.0 - 0.5 / DAYS_PER_MONTH   # half a day short
    for name in ("rota", "campy", "crypto"):
        g.ndata[AgentPropertyKeys.status(name)][child] = Compartment.INFECTIOUS
        g.ndata[AgentPropertyKeys.num_infections(name)][child] = 3
        g.ndata[AgentPropertyKeys.ever_vaccinated(name)][child] = True
    g.ndata[AgentPropertyKeys.HEALTH][child] = 0.01
    n_children = int(_children(model).sum())

    model.step()

    assert int(_children(model).sum()) == n_children            # slot stays a child
    assert g.ndata[AgentPropertyKeys.AGE][child].item() < 1.0 / DAYS_PER_MONTH
    assert int(g.ndata[AgentPropertyKeys.BIRTH_DAY][child]) == 0
    assert g.ndata[AgentPropertyKeys.HEALTH][child].item() == pytest.approx(
        g.ndata[AgentPropertyKeys.INITIAL_HEALTH][child].item()
    )
    for name in ("rota", "campy", "crypto"):
        assert int(g.ndata[AgentPropertyKeys.status(name)][child]) == Compartment.SUSCEPTIBLE
        assert int(g.ndata[AgentPropertyKeys.num_infections(name)][child]) == 0
        assert not bool(g.ndata[AgentPropertyKeys.ever_vaccinated(name)][child])
        assert int(g.ndata[AgentPropertyKeys.last_episode_infection(name)][child]) == -1
        assert int(g.ndata[AgentPropertyKeys.illness_duration(name)][child]) == 0


def test_child_ages_stay_under_five_and_roughly_uniform_over_years(make_model):
    model = make_model(steps=900, agents=2000)
    ages = model.graph.ndata[AgentPropertyKeys.AGE][_children(model)].numpy()
    assert ages.min() >= 0.0 and ages.max() < 60.0
    # Turnover preserves the uniform 0-60 month distribution (mean 30).
    assert 25.0 < ages.mean() < 35.0
    assert (model.graph.ndata[AgentPropertyKeys.BIRTH_DAY] >= 0).sum() > 0


def test_ageing_off_keeps_ages_static(make_model):
    model = make_model(steps=20, agents=600, run=False, overrides={"demography.ageing": False})
    start = model.graph.ndata[AgentPropertyKeys.AGE].clone()
    model.run()
    assert torch.equal(start, model.graph.ndata[AgentPropertyKeys.AGE])


def test_open_episode_of_a_child_ageing_out_is_censored(make_model):
    model = make_model(steps=5, agents=800, run=False)
    g = model.graph
    illness = next(s for s in model.systems if isinstance(s, ChildIllnessSystem))
    child = _children(model).nonzero(as_tuple=True)[0][0]
    # Give the child an open rota episode via the illness system itself.
    g.ndata[AgentPropertyKeys.status("rota")][child] = Compartment.INFECTIOUS
    g.ndata[AgentPropertyKeys.num_infections("rota")][child] += 1
    mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    mask[child] = True
    illness._initialize_illness(g, mask, "rota", model.config.illness_mechanics,
                                next(p for p in model.config.pathogens if p.name == "rota"))
    entry = illness.episode_log[-1]
    g.ndata[AgentPropertyKeys.AGE][child] = 60.0 - 0.1 / DAYS_PER_MONTH
    demo = next(s for s in model.systems if isinstance(s, DemographySystem))
    demo.update(g, timestep=0, illness_system=illness)
    assert entry.get("censored") is True
    assert entry["realized_duration"] is None


# ---------------------------------------------------------------------------
# School attendance by age
# ---------------------------------------------------------------------------

def test_children_below_entry_age_never_go_to_school(make_model):
    model = make_model(steps=1, agents=1500, run=False)
    g = model.graph
    young = _children(model) & (g.ndata[AgentPropertyKeys.AGE] < 30.0)   # stay < 36 m for 6 months
    assert young.sum() > 10
    for _ in range(20):
        model.step()
        assert not torch.any(g.ndata[AgentPropertyKeys.ACTIVITY_CHOICE][young] == Activity.SCHOOL)


def test_school_eligibility_switches_on_at_entry_age(make_model):
    model = make_model(steps=1, agents=1500, run=False,
                       overrides={"demography.school_attendance_prob": 1.0})
    g = model.graph
    entry = model.config.demography.school_entry_age_months
    kid = _children(model).nonzero(as_tuple=True)[0][0]
    g.ndata[AgentPropertyKeys.AGE][kid] = entry - 0.5 / DAYS_PER_MONTH
    g.ndata[AgentPropertyKeys.TIME_USE][kid, Activity.SCHOOL] = 0.0   # as for an under-age child
    demo = next(s for s in model.systems if isinstance(s, DemographySystem))
    demo.update(g, timestep=0)
    assert g.ndata[AgentPropertyKeys.TIME_USE][kid, Activity.SCHOOL].item() > 0.0
    assert torch.equal(g.ndata[AgentPropertyKeys.TIME_USE][kid], g.ndata[AgentPropertyKeys.TIME_USE_BASE][kid])


def test_adults_time_use_is_untouched(make_model):
    model = make_model(steps=1, agents=800, run=False)
    g = model.graph
    adults = ~_children(model)
    assert torch.equal(g.ndata[AgentPropertyKeys.TIME_USE][adults],
                       g.ndata[AgentPropertyKeys.TIME_USE_BASE][adults])


# ---------------------------------------------------------------------------
# Age-specific susceptibility
# ---------------------------------------------------------------------------

def test_relative_susceptibility_children_adults_and_infants(make_model):
    model = make_model(steps=1, agents=800, run=False)
    g = model.graph
    rota = next(p for p in model.pathogens if p.name == "rota")
    cfg = rota.config
    is_child = _children(model)
    kids = is_child.nonzero(as_tuple=True)[0][:4]
    adult = (~is_child).nonzero(as_tuple=True)[0][:1]
    key = AgentPropertyKeys.num_infections("rota")
    g.ndata[key][kids] = torch.tensor([0, 1, 5, 0], dtype=g.ndata[key].dtype)
    g.ndata[AgentPropertyKeys.AGE][kids] = torch.tensor([20.0, 20.0, 20.0, 0.0])
    s = rota._relative_susceptibility(g, torch.cat([kids, adult]), route="h2h")
    k, f = cfg.immunity_decay_per_infection, cfg.immunity_floor
    assert s[0].item() == pytest.approx(1.0)                                   # naive, past ramp
    assert s[1].item() == pytest.approx(math.exp(-k))                          # one infection
    assert s[2].item() == pytest.approx(f)                                     # plateau
    assert s[3].item() == pytest.approx(cfg.infant_susceptibility_at_birth)    # newborn
    adult_s = cfg.adult_relative_susceptibility if cfg.adult_relative_susceptibility is not None else f
    assert s[4].item() == pytest.approx(adult_s)                               # adult (own value since v4)


def test_original_susceptibility_is_recovered_with_features_off(make_model):
    model = make_model(steps=1, agents=600, run=False,
                       overrides={"pathogens[rota].immunity_floor": None,
                                  "pathogens[rota].adult_relative_susceptibility": None,
                                  "pathogens[rota].infant_susceptibility_ramp_months": 0.0,
                                  "steering_parameters.household_exposure_sdlog": 0.0})
    rota = next(p for p in model.pathogens if p.name == "rota")
    idx = torch.arange(model.graph.num_nodes())
    assert torch.equal(rota._relative_susceptibility(model.graph, idx, route="water"),
                       rota._immunity_factor(model.graph, idx))


# ---------------------------------------------------------------------------
# Initial infection histories
# ---------------------------------------------------------------------------

def test_cumulative_infections_match_the_maled_rates():
    for p, rates in INFECTION_RATE_BY_AGE_PER_100CM.items():
        assert cumulative_infections_by_age(p, 24.0) == pytest.approx(3 * sum(rates) / 100)
        assert cumulative_infections_by_age(p, 36.0) == pytest.approx(
            3 * sum(rates) / 100 + 12 * rates[-1] / 100
        )
        assert cumulative_infections_by_age(p, 0.0) == 0.0


def test_initial_prior_infections_follow_maled_by_age(make_model):
    model = make_model(steps=1, agents=6000, run=False)
    g = model.graph
    is_child = _children(model)
    age = g.ndata[AgentPropertyKeys.AGE]
    older = is_child & (age >= 36)
    for p in ("campy", "crypto"):
        n = g.ndata[AgentPropertyKeys.num_infections(p)][older].float()
        expected = cumulative_infections_by_age(p, age[older]).mean().item()
        assert n.mean().item() == pytest.approx(expected, rel=0.15)
        # Adults are not given histories (they sit at the plateau instead).
        assert int(g.ndata[AgentPropertyKeys.num_infections(p)][~is_child].sum()) == 0
        # last_episode_infection marks every prior infection as already used.
        last = g.ndata[AgentPropertyKeys.last_episode_infection(p)][older]
        assert torch.equal(last, g.ndata[AgentPropertyKeys.num_infections(p)][older] - 1)


def test_no_initial_histories_when_switched_off(make_model):
    model = make_model(steps=1, agents=600, run=False,
                       overrides={"demography.initial_prior_infections": "none"})
    for p in model.pathogens:
        assert int(model.graph.ndata[AgentPropertyKeys.num_infections(p.name)].sum()) == 0


# ---------------------------------------------------------------------------
# Environmental-exposure heterogeneity
# ---------------------------------------------------------------------------

def test_exposure_multiplier_is_household_level_mean_one_and_environmental_only(make_model):
    model = make_model(steps=1, agents=4000, run=False,
                       overrides={"steering_parameters.household_exposure_sdlog": 0.8})
    g = model.graph
    f = g.ndata[AgentPropertyKeys.EXPOSURE_FRAILTY]
    hh = g.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
    # Shared within households.
    for h in torch.unique(hh)[:50]:
        vals = f[hh == h]
        assert torch.all(vals == vals[0])
    # Mean ~1 across households.
    _, first = np.unique(hh.numpy(), return_index=True)
    assert f[first].mean().item() == pytest.approx(1.0, abs=0.1)
    crypto = next(p for p in model.pathogens if p.name == "crypto")
    idx = torch.arange(g.num_nodes())
    person = crypto._relative_susceptibility(g, idx, route="household")
    water = crypto._relative_susceptibility(g, idx, route="water")
    assert torch.allclose(water, person * f)


def test_zero_heterogeneity_draws_nothing(make_model):
    model = make_model(steps=1, agents=600, run=False,
                       overrides={"steering_parameters.household_exposure_sdlog": 0.0})
    assert torch.all(model.graph.ndata[AgentPropertyKeys.EXPOSURE_FRAILTY] == 1.0)


# ---------------------------------------------------------------------------
# Birth-cohort metrics
# ---------------------------------------------------------------------------

def test_first_infection_days_are_recorded_only_for_children_born_in_the_run(make_model):
    model = make_model(steps=400, agents=3000)
    g = model.graph
    birth = g.ndata[AgentPropertyKeys.BIRTH_DAY]
    for p in model.pathogens:
        first = g.ndata[AgentPropertyKeys.first_infection_day(p.name)]
        assert torch.all(first[birth < 0] == -1)
        born = birth >= 0
        has = born & (first >= 0)
        assert torch.all(first[has] >= birth[has])
    cohorts = model.get_birth_cohorts()
    assert len(cohorts) == int((birth >= 0).sum())   # nobody born in-run has aged out yet
    # Followed-for-24-months cohorts need a longer run; the metric says so with NaN.
    out = first_infection_metrics(model)
    assert all(math.isnan(v) for k, v in out.items() if k.endswith("_by_24m"))

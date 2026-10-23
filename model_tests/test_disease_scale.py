"""
Tests for the disease-scale layer: MAL-ED-sourced infection -> disease
parameters, age-structured incidence bookkeeping, burn-in-aware metrics,
realized episode durations, and pathogen-specific treatment effects.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from abm.constants import AgentPropertyKeys, Compartment
from abm.evidence import maled
from abm.systems.care_seeking import CareSeekingSystem
from abm.systems.child_illness import ChildIllnessSystem, symptomatic_probability
from config import (
    CampylobacterConfig,
    CryptosporidiumConfig,
    RotavirusConfig,
    SVEIRCONFIG,
)
from experiments.metrics import age_window_rates, route_fractions, route_fractions_post_burnin

CONFIGS = {"rota": RotavirusConfig, "campy": CampylobacterConfig, "crypto": CryptosporidiumConfig}


# ---------------------------------------------------------------------------
# MAL-ED transcription and its use in the configs
# ---------------------------------------------------------------------------

def test_maled_overall_rates_reproduce_published_per_100_child_month_rates():
    """The Supplementary Table 1 counts reproduce the pooled rates quoted in the
    paper (per 100 child-months): rota 3.03 infections / 1.91 episodes, campy
    8.76 / 1.36, crypto 3.84 / 0.665."""
    expected = {"rota": (3.03, 1.91), "campy": (8.76, 1.36), "crypto": (3.84, 0.665)}
    for p, (inf, epi) in expected.items():
        assert maled.overall_infection_rate_per_child_year(p) / 12 * 100 == pytest.approx(inf, abs=0.01)
        assert maled.overall_attributable_rate_per_child_year(p) / 12 * 100 == pytest.approx(epi, abs=0.01)


def test_maled_pooled_rates_lie_within_the_site_ranges():
    for p in ("rota", "campy", "crypto"):
        pooled_inf = maled.overall_infection_rate_per_child_year(p) * 100 / 12
        pooled_epi = maled.overall_attributable_rate_per_child_year(p) * 100 / 12
        assert min(maled.INFECTION_RATE_BY_SITE_PER_100CM[p].values()) <= pooled_inf
        assert pooled_inf <= max(maled.INFECTION_RATE_BY_SITE_PER_100CM[p].values())
        assert min(maled.ATTRIBUTABLE_RATE_BY_SITE_PER_100CM[p].values()) <= pooled_epi
        assert pooled_epi <= max(maled.ATTRIBUTABLE_RATE_BY_SITE_PER_100CM[p].values())


@pytest.mark.parametrize("pathogen", ["rota", "campy", "crypto"])
def test_config_age_profile_has_maled_shape(pathogen):
    """symptomatic_fraction is calibrated (campaign v3: P(illness | first
    infection), compared with MAL-ED through the observation layer), but the
    age profile keeps MAL-ED Supplementary Table 5's relative shape: the
    multipliers times MAL-ED's overall fraction give its 6-month bands."""
    cfg = CONFIGS[pathogen]()
    assert 0.0 < cfg.symptomatic_fraction < 1.0
    profile = cfg.symptomatic_fraction_age_profile
    assert [u for u, _ in profile] == [6.0, 12.0, 18.0, 24.0]
    bands = [maled.overall_symptomatic_fraction(pathogen) * m for _, m in profile]
    assert bands == pytest.approx(list(maled.SYMPTOMATIC_FRACTION_BY_AGE[pathogen]), abs=0.001)


@pytest.mark.parametrize("pathogen", ["rota", "campy", "crypto"])
def test_age_profile_reproduces_overall_fraction_under_maled_infection_mix(pathogen):
    """Weighting the 6-month-band fractions by MAL-ED's own age distribution of
    infections recovers the overall fraction - i.e. the scalar and the profile
    are mutually consistent, not two independent numbers."""
    rates = maled.INFECTION_RATE_BY_AGE_PER_100CM[pathogen]
    weights = [rates[2 * i] + rates[2 * i + 1] for i in range(4)]
    fracs = maled.SYMPTOMATIC_FRACTION_BY_AGE[pathogen]
    weighted = sum(w * f for w, f in zip(weights, fracs)) / sum(weights)
    assert weighted == pytest.approx(maled.overall_symptomatic_fraction(pathogen), abs=0.015)


def test_symptomatic_probability_band_edges_and_extrapolation():
    # at MAL-ED's overall fraction the bands are MAL-ED's own
    cfg = CampylobacterConfig(symptomatic_fraction=maled.overall_symptomatic_fraction("campy"))
    ages = torch.tensor([0.0, 6.0, 6.01, 12.0, 17.9, 24.0, 30.0, 59.0])
    probs = symptomatic_probability(cfg, ages)
    expected = [0.22, 0.22, 0.20, 0.20, 0.15, 0.11, 0.11, 0.11]   # held flat beyond 24 months
    assert probs.tolist() == pytest.approx(expected, abs=0.001)


def test_symptomatic_probability_skips_draw_only_when_certain():
    always = RotavirusConfig(symptomatic_fraction=1.0, symptomatic_fraction_age_profile=None)
    assert symptomatic_probability(always, torch.tensor([3.0, 40.0])) is None
    flat = RotavirusConfig(symptomatic_fraction=0.5, symptomatic_fraction_age_profile=None)
    assert symptomatic_probability(flat, torch.tensor([3.0])) == 0.5
    # A profile that pushes every band to >= 1 is also certain - and is clamped.
    saturated = RotavirusConfig(symptomatic_fraction=1.0,
                                symptomatic_fraction_age_profile=[(6, 1.2), (24, 1.0)])
    assert symptomatic_probability(saturated, torch.tensor([3.0])) is None


def test_age_profile_validation():
    with pytest.raises(ValueError):
        RotavirusConfig(symptomatic_fraction_age_profile=[(12, 1.0), (6, 1.0)])
    with pytest.raises(ValueError):
        RotavirusConfig(symptomatic_fraction_age_profile=[(6, -0.1)])
    with pytest.raises(ValueError):
        RotavirusConfig(symptomatic_fraction_age_profile=[])


# ---------------------------------------------------------------------------
# Age-structured incidence bookkeeping and burn-in-aware metrics
# ---------------------------------------------------------------------------

def test_age_records_reconcile_exactly_with_lifetime_infection_counts(make_model):
    """Every S/R -> E transition among children appears exactly once in
    new_infections_by_age_month, and the person-time record counts every child
    on every day."""
    # Ageing off so no slot is renewed (renewal resets num_infections); the
    # initial (pre-t=0) infection counts are subtracted.
    model = make_model(steps=60, agents=1500, run=False, overrides={"demography.ageing": False})
    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].bool()
    initial = {p.name: int(model.graph.ndata[AgentPropertyKeys.num_infections(p.name)][is_child].sum())
               for p in model.pathogens}
    assert any(v > 0 for v in initial.values()), "MAL-ED initial histories should be non-zero"
    model.run()
    n_children = int(is_child.sum())
    days = np.asarray(model.child_days_by_age_month)
    assert days.shape == (60, model.AGE_RECORD_MONTHS)
    assert (days.sum(axis=1) == n_children).all()
    for p in model.pathogens:
        recorded = int(np.asarray(model.new_infections_by_age_month[p.name]).sum())
        lifetime = int(model.graph.ndata[AgentPropertyKeys.num_infections(p.name)][is_child].sum())
        assert recorded == lifetime - initial[p.name], p.name
        # The naive stratum is a subset of the total.
        naive = np.asarray(model.new_infections_naive_by_age_month[p.name])
        naive_days = np.asarray(model.child_days_naive_by_age_month[p.name])
        assert (naive <= np.asarray(model.new_infections_by_age_month[p.name])).all()
        assert (naive_days <= days).all()


def test_daily_route_history_sums_to_lifetime_totals(make_model):
    model = make_model(steps=40, agents=1000)
    for p in model.pathogens:
        for route, total in p.total_by_route.items():
            assert len(p.daily_by_route[route]) == 40
            assert sum(p.daily_by_route[route]) == total, (p.name, route)
    # With no burn-in the post-burn-in fractions equal the lifetime fractions.
    lifetime = route_fractions(model)
    windowed = route_fractions_post_burnin(model, burn_in_days=0)
    for k, v in lifetime.items():
        assert windowed[f"{k[:-len('_fraction')]}_fraction_post_burnin"] == pytest.approx(v)


def test_age_window_rates_match_a_manual_computation(make_model):
    model = make_model(steps=80, agents=1500)
    burn = 20
    out = age_window_rates(model, burn_in_days=burn)
    days = np.asarray(model.child_days_by_age_month)
    for p in model.pathogens:
        inf = np.asarray(model.new_infections_by_age_month[p.name])
        py = days[burn:, :24].sum() / 365.0
        assert out[f"{p.name}_infections_per_child_year_0_24m"] == pytest.approx(
            inf[burn:, :24].sum() / py
        )
        n_epi = sum(1 for e in model.get_child_episode_log()
                    if e["pathogen"] == p.name and e["timestep"] >= burn and e["age_months"] < 24)
        assert out[f"{p.name}_symptomatic_episodes_per_child_year_0_24m"] == pytest.approx(n_epi / py)


def test_age_window_rates_are_nan_when_burn_in_covers_the_run(make_model):
    model = make_model(steps=15, agents=600)
    out = age_window_rates(model, burn_in_days=15)
    assert all(math.isnan(v) for k, v in out.items() if "per_child_year" in k)


def test_age_records_need_no_setup_when_stepping_manually(make_model):
    """The webapp and orchestrator drive step() themselves; the age records
    must fill in regardless."""
    model = make_model(steps=10, agents=600, run=False)
    for _ in range(5):
        model.step()
    assert len(model.child_days_by_age_month) == 5
    assert all(len(v) == 5 for v in model.new_infections_by_age_month.values())


# ---------------------------------------------------------------------------
# Realized episode duration
# ---------------------------------------------------------------------------

def test_realized_duration_equals_initial_when_treatment_never_succeeds(make_model):
    model = make_model(steps=300, agents=3000,
                       overrides={"steering_parameters.treatment_success_prob": 0.0})
    closed = [e for e in model.get_child_episode_log() if e["realized_duration"] is not None]
    assert len(closed) > 20
    assert all(e["realized_duration"] == e["initial_duration"] for e in closed)


def test_realized_duration_never_exceeds_initial_and_open_episodes_stay_none(make_model):
    model = make_model(steps=150, agents=2000)
    log = model.get_child_episode_log()
    closed = [e for e in log if e["realized_duration"] is not None]
    assert closed and all(1 <= e["realized_duration"] <= e["initial_duration"] for e in closed)
    # Anything still open at the end must have started recently enough to be
    # still running (it cannot be older than the cap allows).
    cap = model.config.illness_mechanics.duration_cap_days
    for e in log:
        if e["realized_duration"] is None:
            assert e["timestep"] >= model.config.step_target - cap - 1


# ---------------------------------------------------------------------------
# Treatment clears infection only where configured
# ---------------------------------------------------------------------------

def test_successful_treatment_clears_only_pathogens_configured_to_clear(make_model):
    # The original "cure" treatment model: episodes end at once.
    model = make_model(steps=1, agents=800, run=False,
                       overrides={"steering_parameters.treatment_success_prob": 1.0,
                                  "steering_parameters.treatment_effect_model": "cure"})
    g = model.graph
    care = next(s for s in model.systems if isinstance(s, CareSeekingSystem))
    hh = g.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
    parents = g.ndata[AgentPropertyKeys.IS_PARENT].nonzero(as_tuple=True)[0]
    is_child = g.ndata[AgentPropertyKeys.IS_CHILD].bool()
    parent_idx = parents[0]
    child_idx = (is_child & (hh == hh[parent_idx])).nonzero(as_tuple=True)[0][0]

    for name in ("rota", "campy", "crypto"):
        g.ndata[AgentPropertyKeys.status(name)][child_idx] = Compartment.INFECTIOUS
    g.ndata[AgentPropertyKeys.illness_duration("rota")][child_idx] = 3
    g.ndata[AgentPropertyKeys.symptom_severity("rota")][child_idx] = 0.5
    care._track_episodes(torch.zeros(g.num_nodes()), is_child)   # allocate bookkeeping

    care._apply_treatment_outcome(g, parent_idx, child_idx)

    # Symptoms always resolve on success...
    assert int(g.ndata[AgentPropertyKeys.illness_duration("rota")][child_idx]) == 0
    assert float(g.ndata[AgentPropertyKeys.symptom_severity("rota")][child_idx]) == 0.0
    # ...but only campylobacter (antibiotic-treatable) is cleared.
    status = {n: int(g.ndata[AgentPropertyKeys.status(n)][child_idx]) for n in ("rota", "campy", "crypto")}
    assert status["rota"] == Compartment.INFECTIOUS
    assert status["crypto"] == Compartment.INFECTIOUS
    assert status["campy"] == Compartment.RECOVERED


def test_default_treatment_clearance_values():
    by_name = {p.name: p for p in SVEIRCONFIG.pathogens}
    assert by_name["rota"].treatment_clears_infection_prob == 0.0
    assert by_name["crypto"].treatment_clears_infection_prob == 0.0
    assert by_name["campy"].treatment_clears_infection_prob == 1.0


# ---------------------------------------------------------------------------
# Supportive treatment (default treatment_effect_model)
# ---------------------------------------------------------------------------

def _child_with_parent(model, min_age=None, max_age=None):
    g = model.graph
    hh = g.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
    is_child = g.ndata[AgentPropertyKeys.IS_CHILD].bool()
    age = g.ndata[AgentPropertyKeys.AGE]
    for parent in g.ndata[AgentPropertyKeys.IS_PARENT].nonzero(as_tuple=True)[0].tolist():
        kids = (is_child & (hh == hh[parent])).nonzero(as_tuple=True)[0].tolist()
        for k in kids:
            a = float(age[k])
            if (min_age is None or a >= min_age) and (max_age is None or a < max_age):
                return torch.tensor(parent), torch.tensor(k)
    raise AssertionError("no suitable child")


def _give_episodes(model, child, duration=4, severity=0.5):
    g = model.graph
    for name in ("rota", "campy", "crypto"):
        g.ndata[AgentPropertyKeys.status(name)][child] = Compartment.INFECTIOUS
        g.ndata[AgentPropertyKeys.illness_duration(name)][child] = duration
        g.ndata[AgentPropertyKeys.symptom_severity(name)][child] = severity


def test_supportive_treatment_reduces_harm_shortens_modestly_and_clears_only_campy(make_model):
    model = make_model(steps=1, agents=1500, run=False,
                       overrides={"steering_parameters.treatment_success_prob": 1.0})
    sp = model.config.steering_parameters
    assert sp.treatment_effect_model == "supportive"
    g = model.graph
    care = next(s for s in model.systems if isinstance(s, CareSeekingSystem))
    parent, child = _child_with_parent(model, min_age=12.0)
    _give_episodes(model, child)
    care._track_episodes(torch.zeros(g.num_nodes()), g.ndata[AgentPropertyKeys.IS_CHILD].bool())
    care._apply_treatment_outcome(g, parent, child)

    st = {n: int(g.ndata[AgentPropertyKeys.status(n)][child]) for n in ("rota", "campy", "crypto")}
    dur = {n: int(g.ndata[AgentPropertyKeys.illness_duration(n)][child]) for n in ("rota", "campy", "crypto")}
    sev = {n: float(g.ndata[AgentPropertyKeys.symptom_severity(n)][child]) for n in ("rota", "campy", "crypto")}
    assert st == {"rota": Compartment.INFECTIOUS, "campy": Compartment.RECOVERED,
                  "crypto": Compartment.INFECTIOUS}
    for n in ("rota", "crypto"):                      # zinc only: 0 or 1 day off
        assert dur[n] in (3, 4)
        assert sev[n] == pytest.approx(0.5 * (1 - sp.treatment_harm_reduction))
    assert dur["campy"] in (1, 2, 3)                  # zinc + antibiotic: 1.80 days in expectation
    assert bool(care._treated_this_episode[child])


def test_zinc_effect_is_absent_under_six_months_and_unbiased_above(make_model):
    model = make_model(steps=1, agents=1500, run=False,
                       overrides={"steering_parameters.treatment_success_prob": 1.0})
    sp = model.config.steering_parameters
    g = model.graph
    care = next(s for s in model.systems if isinstance(s, CareSeekingSystem))
    care._track_episodes(torch.zeros(g.num_nodes()), g.ndata[AgentPropertyKeys.IS_CHILD].bool())
    for age_lo, age_hi, expected in ((0.0, 6.0, 0.0), (12.0, None, sp.treatment_duration_reduction_days)):
        parent, child = _child_with_parent(model, min_age=age_lo, max_age=age_hi)
        cuts = []
        torch.manual_seed(0)
        for _ in range(400):
            g.ndata[AgentPropertyKeys.status("rota")][child] = Compartment.INFECTIOUS
            g.ndata[AgentPropertyKeys.illness_duration("rota")][child] = 5
            g.ndata[AgentPropertyKeys.symptom_severity("rota")][child] = 0.5
            care._apply_supportive_treatment(g, child)
            cuts.append(5 - int(g.ndata[AgentPropertyKeys.illness_duration("rota")][child]))
        assert np.mean(cuts) == pytest.approx(expected, abs=0.07)


def test_a_treated_child_faces_no_further_decisions_this_episode(make_model):
    model = make_model(steps=1, agents=1500, run=False,
                       overrides={"steering_parameters.treatment_success_prob": 1.0,
                                  "steering_parameters.cost_of_care": 0.0})
    g = model.graph
    care = next(s for s in model.systems if isinstance(s, CareSeekingSystem))
    for n in ("rota", "campy", "crypto"):        # nobody else is ill
        g.ndata[AgentPropertyKeys.symptom_severity(n)][:] = 0.0
        g.ndata[AgentPropertyKeys.illness_duration(n)][:] = 0
    parent, child = _child_with_parent(model, min_age=12.0)
    _give_episodes(model, child, duration=6)
    g.ndata[AgentPropertyKeys.WEALTH][g.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
                                     == g.ndata[AgentPropertyKeys.HOUSEHOLD_ID][parent]] = 1.0
    care.update(g)                               # day 1: free care, certain success
    assert care.decisions_faced == 1 and care.care_sought == 1
    assert bool(care._treated_this_episode[child])
    care.update(g)                               # day 2: still ill, already treated
    care.update(g)
    assert care.decisions_faced == 1
    assert care.total_episodes == 1


def test_treated_child_health_in_the_decision_matches_the_treatment_effect():
    from abm.systems.care_seeking import _treated_child_health
    from config import SteeringParamsSVEIR
    sup = SteeringParamsSVEIR()
    assert _treated_child_health(0.4, sup) == pytest.approx(1 - 0.6 * (1 - sup.treatment_harm_reduction))
    cure = SteeringParamsSVEIR(treatment_effect_model="cure")
    assert _treated_child_health(0.4, cure) == 1.0

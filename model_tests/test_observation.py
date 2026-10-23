"""
Tests for the MAL-ED observation layer (experiments/observation.py), the
infectiousness of symptomless infections, illness probability by infection
history, illness onset timing, population-weighted household placement and
metre-based distances.

The observation-layer tests drive experiments.observation.observe() with a
hand-built stand-in for a finished model, so that every detection is known
in advance: shedding durations are made effectively deterministic (a huge
gamma shape) and every diarrhoeal stool is collected.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from config import SVEIRCONFIG
from experiments.observation import DAYS_PER_MONTH, observe

MONTH = DAYS_PER_MONTH


def _fake_model(infections, episodes=(), births=((0, 0),), T=3000, sym_days=5.0, asym_days=5.0):
    """infections: [(agent_idx, birth_day, onset_day, symptomatic)];
    episodes: [(agent_idx, onset_day, duration)] (rota)."""
    cfg = SVEIRCONFIG.model_copy(deep=True)
    cfg.observation.shedding_duration_shape = 1e6           # ~deterministic duration
    cfg.observation.diarrhoeal_stool_tested_prob = 1.0
    rota = next(p for p in cfg.pathogens if p.name == "rota")
    rota.detectable_shedding_days_symptomatic = sym_days
    rota.detectable_shedding_days_asymptomatic = asym_days
    log = {
        "agent_idx": np.array([i[0] for i in infections], dtype=np.int64),
        "birth_day": np.array([i[1] for i in infections], dtype=np.int64),
        "day": np.array([i[2] for i in infections], dtype=np.int64),
        "symptomatic": np.array([i[3] for i in infections], dtype=bool),
    }
    empty = {k: v[:0] for k, v in log.items()}
    return SimpleNamespace(
        config=cfg, step_count=T,
        pathogens=[SimpleNamespace(name="rota", config=rota)],
        get_birth_cohorts=lambda: [{"agent_idx": a, "birth_day": b, "exit_day": None, "first_infection_day": {}}
                                   for a, b in births],
        get_infection_log=lambda name: log if name == "rota" else empty,
        get_child_episode_log=lambda: [{"agent_idx": a, "pathogen": "rota", "timestep": d,
                                        "initial_duration": dur, "realized_duration": dur}
                                       for a, d, dur in episodes],
    )


def _sample_day(birth, month_of_life):
    """Day of the surveillance stool in month_of_life (1-based)."""
    return birth + int(round((month_of_life - 0.5) * MONTH))


def test_a_symptomless_infection_is_detected_only_if_a_monthly_stool_falls_in_its_window():
    s5 = _sample_day(0, 5)
    caught = observe(_fake_model([(0, 0, s5 - 2, False)]), burn_in_days=0)
    missed = observe(_fake_model([(0, 0, s5 + 3, False)]), burn_in_days=0)
    assert caught["rota_obs_infections_per_child_year_0_24m"] > 0
    assert missed["rota_obs_infections_per_child_year_0_24m"] == 0


def test_consecutive_positives_are_one_infection_but_two_crude_ones():
    s5 = _sample_day(0, 5)
    # detectable for 40 days from just before month 5's stool: positive in months 5 and 6
    o = observe(_fake_model([(0, 0, s5 - 2, False)], asym_days=40.0), burn_in_days=0)
    assert o["rota_obs_crude_to_consecutive_ratio"] == pytest.approx(2.0)


def test_a_negative_stool_between_positives_separates_infections():
    s5, s7 = _sample_day(0, 5), _sample_day(0, 7)
    o = observe(_fake_model([(0, 0, s5 - 1, False), (0, 0, s7 - 1, False)]), burn_in_days=0)
    assert o["rota_obs_crude_to_consecutive_ratio"] == pytest.approx(1.0)
    rate = o["rota_obs_infections_per_child_year_0_24m"]
    assert rate > 0


def test_attributable_episode_absorbs_incubation_and_post_diarrhoeal_positives():
    s8 = _sample_day(0, 8)
    onset = s8 + 3          # the month-8 stool falls 3 days before the episode: incubation shedding
    o = observe(_fake_model([(0, 0, s8 - 1, True)], episodes=[(0, onset, 4)], sym_days=40.0), burn_in_days=0)
    # one attributable episode; month-8 (incubation) and month-9 (post-diarrhoeal) positives absorbed
    assert o["rota_obs_attributable_fraction_0_24m"] == pytest.approx(1.0)
    assert o["rota_obs_crude_to_consecutive_ratio"] > 1.0


def test_naive_and_prior_rates_split_on_detected_history():
    births = ((0, 0), (1, 0))
    s4, s16 = _sample_day(0, 4), _sample_day(0, 16)
    # child 0: detected at 4 and 16 months; child 1: a missed infection at 4, detected at 16
    inf = [(0, 0, s4 - 1, False), (0, 0, s16 - 1, False), (1, 0, s4 + 4, False), (1, 0, s16 - 1, False)]
    o = observe(_fake_model(inf, births=births), burn_in_days=0)
    # at 15-18 months one detection is in a naive child (child 1), one in a previously detected one
    assert o["rota_obs_naive_rate_15_18m"] > 0
    assert o["rota_obs_prior_rate_15_18m"] > 0
    # child 1's undetected first infection still counts as a TRUE infection
    assert o["rota_true_frac_infected_by_6m"] == pytest.approx(1.0)


def test_kaplan_meier_handles_censoring_and_delayed_entry():
    from experiments.observation import _km_add, _km_cdf, _km_conditional_median
    rows = []
    _km_add(rows, 0.0, 3000.0, 100.0)     # infected at 100 days
    _km_add(rows, 0.0, 50.0, None)        # followed to 50 days only (run ended)
    _km_add(rows, 200.0, 3000.0, 300.0)   # enters at 200 days (burn-in), infected at 300
    _km_add(rows, 200.0, 3000.0, 150.0)   # infected before entry: left out
    # the censored child is not in the risk set at 100 days, the late entrant
    # not yet: everyone at risk at 100 days was infected (a naive proportion
    # would say 1/3)
    assert _km_cdf(rows, 120.0) == pytest.approx(1.0)
    assert len(rows) == 3
    # conditional median of first-infection age among those infected
    rows2 = []
    for age in (90.0, 150.0, 210.0, 270.0):
        _km_add(rows2, 0.0, 3000.0, age)
    assert _km_conditional_median(rows2, 24 * MONTH) == pytest.approx(150.0 / MONTH)


def test_age_shape_ratios_compare_detected_rates_between_age_bands():
    # one child, detections in months 3 and 9 (first year) and 15 (second year)
    inf = [(0, 0, _sample_day(0, m) - 1, False) for m in (3, 9, 15)]
    o = observe(_fake_model(inf), burn_in_days=0)
    assert o["rota_obs_infant_ratio_0_6_vs_6_12m"] == pytest.approx(1.0, rel=0.1)
    assert o["rota_obs_second_to_first_year_ratio"] == pytest.approx(0.5, rel=0.1)


def test_children_born_before_the_run_are_not_observed():
    s5 = _sample_day(-100, 5)
    o = observe(_fake_model([(0, -100, s5 - 1, False)], births=((0, -100),)), burn_in_days=0)
    assert math.isnan(o["rota_obs_infections_per_child_year_0_24m"]) or o["rota_obs_infections_per_child_year_0_24m"] == 0


# ---------------------------------------------------------------------------
# Model mechanisms
# ---------------------------------------------------------------------------

def test_infectiousness_weights_are_the_infectious_mask_at_rho_one(make_model):
    from abm.constants import AgentPropertyKeys, Compartment
    model = make_model(steps=20, agents=800,
                       overrides={f"pathogens[{p}].asymptomatic_relative_infectiousness": 1.0
                                  for p in ("rota", "campy", "crypto")})
    for p in model.pathogens:
        w = p.infectiousness_weights(model.graph)
        mask = (model.graph.ndata[AgentPropertyKeys.status(p.name)] == Compartment.INFECTIOUS).float()
        assert torch.equal(w, mask)


def test_symptomless_infections_carry_reduced_weight(make_model):
    from abm.constants import AgentPropertyKeys, Compartment
    model = make_model(steps=1, agents=600, run=False,
                       overrides={"pathogens[rota].asymptomatic_relative_infectiousness": 0.3})
    rota = next(p for p in model.pathogens if p.name == "rota")
    st = model.graph.ndata[AgentPropertyKeys.status("rota")]
    sym = model.graph.ndata[AgentPropertyKeys.symptomatic_infection("rota")]
    st[:] = Compartment.SUSCEPTIBLE
    st[:4] = Compartment.INFECTIOUS
    sym[:] = False
    sym[:2] = True
    w = rota.infectiousness_weights(model.graph)
    assert w[:2].tolist() == [1.0, 1.0]
    assert w[2:4].tolist() == pytest.approx([0.3, 0.3])
    assert float(w[4:].abs().sum()) == 0.0


def test_new_exposure_clears_the_symptomatic_flag(make_model):
    from abm.constants import AgentPropertyKeys
    model = make_model(steps=1, agents=600, run=False)
    rota = next(p for p in model.pathogens if p.name == "rota")
    key = AgentPropertyKeys.symptomatic_infection("rota")
    model.graph.ndata[key][:10] = True
    rota.begin_day(model.graph)
    rota._commit_infections(model.graph, torch.arange(5))
    assert not bool(model.graph.ndata[key][:5].any())
    assert bool(model.graph.ndata[key][5:10].all())


def test_illness_probability_falls_with_prior_infections():
    from abm.systems.child_illness import symptomatic_probability
    rota = next(p for p in SVEIRCONFIG.pathogens if p.name == "rota").model_copy(
        update={"symptomatic_reduction_per_prior_infection": 1.0, "symptomatic_fraction_age_profile": None,
                "symptomatic_fraction": 0.5})
    probs = symptomatic_probability(rota, torch.tensor([9.0, 9.0, 9.0]), torch.tensor([0, 1, 2]))
    assert probs.tolist() == pytest.approx([0.5, 0.5 * math.exp(-1), 0.5 * math.exp(-2)])


def test_each_child_infection_is_logged_once_with_onset_before_transmission(make_model):
    model = make_model(steps=120, agents=1500)
    assert model.config.illness_mechanics.onset_before_transmission
    for p in model.pathogens:
        log = model.get_infection_log(p.name)
        keys = set(zip(log["agent_idx"].tolist(), log["day"].tolist()))
        assert len(keys) == len(log["day"]), f"{p.name}: an infection was logged twice"
        n_sym = int(log["symptomatic"].sum())
        n_ep = sum(1 for e in model.get_child_episode_log() if e["pathogen"] == p.name)
        assert n_sym == n_ep


def test_population_placement_follows_the_population_layer(make_model):
    from abm.constants import AgentPropertyKeys
    pop = np.load("grids/7d9ce7c720a6/population.npz")["population_2025"]
    model = make_model(steps=1, agents=3000, run=False,
                       overrides={"spatial_creation_args.household_placement": "population"})
    x = model.graph.ndata[AgentPropertyKeys.X].long().numpy()
    y = model.graph.ndata[AgentPropertyKeys.Y].long().numpy()
    occupied_pop = pop[y, x]
    # households land where people live: hardly any agent in a cell GHS-POP leaves empty
    assert (occupied_pop < 0.05).mean() < 0.01
    # and the densest tenth of cells holds far more agents than under uniform placement
    dense = pop >= np.quantile(pop[pop > 0], 0.9)
    assert dense[y, x].mean() > 0.3


def test_metre_distances_convert_with_the_grid_cell_size(make_model):
    model = make_model(steps=1, agents=600, run=False,
                       overrides={"steering_parameters.social_interaction_radius_m": 276.0,
                                  "water.reach_block_m": 442.0})
    # the original grid's cells are ~60 x 51 m (geometric mean ~55 m)
    assert 54.0 < model.cell_size_m < 57.0
    assert model.config.steering_parameters.social_interaction_radius == pytest.approx(276.0 / model.cell_size_m)
    assert model.config.water.reach_block_cells == 8

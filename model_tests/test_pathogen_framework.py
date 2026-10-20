"""
Tests for the shared pathogen framework introduced alongside Cryptosporidium.

These cover the refactors rather than any one pathogen: the registry, generic
route accounting, the per-day single-infection guard, the explicit day/night
phase, per-pathogen water reservoirs and the per-pathogen immunity override.
"""
from __future__ import annotations

import math

import pytest
import torch

from abm.constants import AgentPropertyKeys, Compartment, Phase, WaterStatus
from abm.pathogens.registry import (
    PATHOGEN_CLASSES,
    build_pathogen,
    is_implemented,
    pathogen_class,
    pathogen_config_class,
)
from config import PATHOGEN_CONFIG_TYPES, SVEIRCONFIG


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_halves_agree():
    """Every implemented pathogen has a config class and vice versa.

    The registry module asserts this at import time; this test states it
    explicitly so a failure reads as "you added one half of a pathogen".
    """
    assert set(PATHOGEN_CLASSES) == set(PATHOGEN_CONFIG_TYPES)


def test_registry_covers_the_three_pathogens():
    assert set(PATHOGEN_CLASSES) == {"rota", "campy", "crypto"}
    for name in PATHOGEN_CLASSES:
        assert is_implemented(name)
        assert pathogen_config_class(name).model_fields["name"].default == name


def test_unknown_pathogen_raises_rather_than_silently_skipping():
    with pytest.raises(KeyError):
        pathogen_class("giardia")
    with pytest.raises(KeyError):
        pathogen_config_class("giardia")


def test_default_config_has_all_three_pathogens():
    assert [p.name for p in SVEIRCONFIG.pathogens] == ["rota", "campy", "crypto"]


def test_build_pathogen_retypes_the_config():
    cfg = next(p for p in SVEIRCONFIG.pathogens if p.name == "crypto")
    pathogen = build_pathogen(cfg, SVEIRCONFIG.steering_parameters, "cpu")
    assert isinstance(pathogen, pathogen_class("crypto"))
    assert isinstance(pathogen.config, pathogen_config_class("crypto"))


# ---------------------------------------------------------------------------
# Route accounting
# ---------------------------------------------------------------------------

def test_route_counters_account_for_every_infection(make_model):
    """Per-route totals must sum exactly to the infections actually recorded.

    This is the invariant that the generic _commit_infections exists to
    guarantee: before it, each route incremented its own counters by hand in a
    separate code path, so a route could silently under- or double-count.
    """
    # Ageing and initial infection histories off: with them on, num_infections
    # also counts infections from before t=0 and is reset when a slot is
    # renewed, so it is no longer a lifetime count of committed infections.
    model = make_model(steps=40, overrides={"demography.ageing": False,
                                            "demography.initial_prior_infections": "none"})
    for p in model.pathogens:
        if not p.total_by_route:
            continue
        recorded = int(
            model.graph.ndata[AgentPropertyKeys.num_infections(p.name)].sum().item()
        )
        seeded = int(round(model.config.number_agents * p.config.initial_exposed_proportion))
        # Agents seeded as EXPOSED at t=0 never passed through a commit, so
        # they are not counted in num_infections (AgentFactory sets the
        # compartment directly). Every OTHER infection must be attributed.
        assert sum(p.total_by_route.values()) == recorded, (
            f"{p.name}: routes sum to {sum(p.total_by_route.values())} "
            f"but num_infections totals {recorded}"
        )
        assert seeded >= 0


def test_route_counters_reset_each_day_but_totals_accumulate(make_model):
    model = make_model(steps=1, run=False)
    crypto = next(p for p in model.pathogens if p.name == "crypto")

    child_mask = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}

    running = 0
    for _ in range(25):
        model.step(child_mask=child_mask)
        per_day = sum(crypto.cases_by_route.values())
        assert per_day == crypto.new_cases_this_step
        running += per_day
        assert sum(crypto.total_by_route.values()) == running


def test_an_agent_is_infected_by_at_most_one_route_per_day(make_model):
    """The per-day guard makes route attribution unambiguous.

    Checked by driving the day manually and asserting the number of agents
    whose infection count changed equals the number of infections committed -
    if an agent could be infected twice in one day, the former would be
    smaller than the latter.
    """
    model = make_model(pathogens=["crypto"], steps=1, run=False,
                       overrides={
                           # Crank every route hard, so a double infection
                           # would actually happen if it were possible.
                           "pathogens[crypto].household_transmission_prob": 0.5,
                           "pathogens[crypto].community_infection_prob": 0.05,
                           "pathogens[crypto].water_dose_scale": 1.0,
                           "pathogens[crypto].zoonotic_interaction_rate": 5.0,
                           # Slot renewal resets num_infections, which would
                           # count as a "change" that is not an infection.
                           "demography.ageing": False,
                       })
    crypto = model.pathogens[0]
    key = AgentPropertyKeys.num_infections("crypto")
    child_mask = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}

    saw_infections = False
    for _ in range(20):
        before = model.graph.ndata[key].clone()
        model.step(child_mask=child_mask)
        after = model.graph.ndata[key]
        n_changed = int((after != before).sum().item())
        n_committed = crypto.new_cases_this_step
        assert n_committed == n_changed, (
            f"{n_committed} infections committed but only {n_changed} agents changed - "
            "an agent was infected more than once in a single day"
        )
        # And no agent's count jumped by more than one in a single day.
        assert int((after - before).max().item()) <= 1
        saw_infections |= n_committed > 0

    assert saw_infections, "test is vacuous - no infections occurred at all"


# ---------------------------------------------------------------------------
# Same-household exclusion in co-location pressure
# ---------------------------------------------------------------------------

def test_community_route_excludes_same_household_contacts(make_model):
    """With everyone at home and no other route active, community must be zero.

    Cryptosporidium's community route exists to represent mixing OUTSIDE the
    household. If same-household co-location were counted, a population parked
    at home would still generate community infections, and within-household
    spread would be mislabelled as community transmission in the route
    fractions the pathogen is calibrated against.
    """
    model = make_model(
        pathogens=["crypto"], steps=25, run=False,
        overrides={
            "pathogens[crypto].community_infection_prob": 0.3,
            "pathogens[crypto].household_transmission_prob": 0.0,
            "pathogens[crypto].water_dose_scale": 0.0,
            "pathogens[crypto].zoonotic_interaction_rate": 0.0,
        },
    )
    # Force every agent to stay home every day: TIME_USE is a categorical
    # distribution over Activity, so a point mass on HOME (index 0) removes
    # school / worship / water / social movement entirely, including the
    # social visits that would otherwise bring other households onto the cell.
    time_use = model.graph.ndata[AgentPropertyKeys.TIME_USE]
    time_use.zero_()
    time_use[:, 0] = 1.0

    child_mask = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}
    for _ in range(25):
        model.step(child_mask=child_mask)

    crypto = model.pathogens[0]
    assert crypto.total_by_route["community"] == 0, (
        "community route fired with the whole population at home - same-household "
        "contacts are leaking into the co-location pressure"
    )


def test_household_route_still_fires_when_everyone_is_home(make_model):
    """Companion to the previous test: the household route must NOT be zero.

    Together these two pin down that the exclusion removed the duplicate
    channel rather than disabling within-household transmission altogether.
    """
    model = make_model(
        pathogens=["crypto"], steps=25, run=False,
        overrides={
            "pathogens[crypto].community_infection_prob": 0.0,
            "pathogens[crypto].household_transmission_prob": 0.3,
            "pathogens[crypto].water_dose_scale": 0.0,
            "pathogens[crypto].zoonotic_interaction_rate": 0.0,
        },
    )
    time_use = model.graph.ndata[AgentPropertyKeys.TIME_USE]
    time_use.zero_()
    time_use[:, 0] = 1.0

    child_mask = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}
    for _ in range(25):
        model.step(child_mask=child_mask)

    crypto = model.pathogens[0]
    assert crypto.total_by_route["household"] > 0


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------

def test_phase_is_passed_explicitly_and_both_phases_occur(make_model):
    """step_transmission receives DAY then NIGHT, once each, per simulated day."""
    model = make_model(steps=1, run=False)
    crypto = next(p for p in model.pathogens if p.name == "crypto")

    seen = []
    original = crypto.step_transmission

    def spy(agent_state, location_ids, num_locations, grid, phase):
        seen.append(phase)
        return original(agent_state, location_ids, num_locations, grid, phase)

    crypto.step_transmission = spy
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}
    model.step(child_mask=model.graph.ndata[AgentPropertyKeys.IS_CHILD])

    assert seen == [Phase.DAY, Phase.NIGHT]


# ---------------------------------------------------------------------------
# Per-pathogen water reservoirs
# ---------------------------------------------------------------------------

def test_each_waterborne_pathogen_gets_its_own_reservoir(make_model):
    model = make_model(steps=1, run=False, overrides={"water.model": "legacy"})
    grid = model.grid_environment

    rota_state = grid.get_water_state("rota")
    crypto_state = grid.get_water_state("crypto")
    assert rota_state is not None and crypto_state is not None
    assert rota_state is not crypto_state

    # Campylobacter is not waterborne and must not have one.
    assert grid.get_water_state("campy") is None

    # Rotavirus starts binary CLEAN at water points, crypto starts at zero load.
    mask = grid.water_point_mask()
    assert mask.sum().item() > 0
    assert torch.all(rota_state[mask] == WaterStatus.CLEAN)
    assert torch.all(rota_state[~mask] == 0)
    assert torch.all(crypto_state == 0)


def test_static_water_layer_is_not_mutated_by_contamination(make_model):
    """The grid's own water layer must stay a pristine 'where are the sources' mask.

    Contamination used to be written straight into grid_tensor, which meant the
    static layer doubled as mutable state. AgentFactory assigns WATER_LOCATION
    by looking for cells equal to 1 in that layer, so a contaminated cell
    (value 2) would have been invisible to it.
    """
    model = make_model(steps=30, run=False,
                       overrides={"steering_parameters.human_to_water_infection_prob": 1.0,
                                  "water.model": "legacy"})
    grid = model.grid_environment
    before = grid.water_point_mask().clone()

    child_mask = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}
    for _ in range(30):
        model.step(child_mask=child_mask)

    assert torch.equal(grid.water_point_mask(), before)


def test_water_shock_hits_every_waterborne_pathogen(make_model):
    """One shock event, applied to all waterborne reservoirs.

    A flood contaminates the same sources for everything living in them, so the
    shock is drawn once and applied to each pathogen - not drawn independently
    per pathogen.
    """
    from abm.systems.environment import EnvironmentSystem

    model = make_model(steps=1, run=False, overrides={"water.model": "legacy"})
    grid = model.grid_environment
    env = next(s for s in model.systems if isinstance(s, EnvironmentSystem))

    # Force the shock to fire with certainty.
    model.config.steering_parameters.shock_daily_prob = 1.0
    env.config.steering_parameters.shock_daily_prob = 1.0
    env.update(model.graph, grid=grid, timestep=0, pathogens=model.pathogens)

    mask = grid.water_point_mask()
    crypto_cfg = next(p for p in model.config.pathogens if p.name == "crypto")
    assert torch.all(grid.get_water_state("rota")[mask] == WaterStatus.CONTAMINATED)
    assert torch.all(grid.get_water_state("crypto")[mask] >= crypto_cfg.shock_oocyst_load)


# ---------------------------------------------------------------------------
# Per-pathogen immunity override
# ---------------------------------------------------------------------------

def test_immunity_override_applies_only_to_the_overriding_pathogen(make_model):
    model = make_model(steps=1, run=False)
    shared = model.config.steering_parameters.prior_infection_immunity_factor

    by_name = {p.name: p for p in model.pathogens}
    assert by_name["campy"].immunity_decay_per_infection == shared
    assert by_name["rota"].immunity_decay_per_infection == pytest.approx(0.45)   # Velazquez/MAL-ED
    assert by_name["crypto"].immunity_decay_per_infection == pytest.approx(0.29)  # Kattula
    assert by_name["rota"].immunity_decay_per_infection != shared
    assert by_name["crypto"].immunity_decay_per_infection != shared


def test_immunity_factor_is_exponential_in_prior_infections(make_model):
    model = make_model(steps=1, run=False)
    crypto = next(p for p in model.pathogens if p.name == "crypto")
    key = AgentPropertyKeys.num_infections("crypto")
    model.graph.ndata[key][:4] = torch.tensor([0, 1, 2, 4], dtype=model.graph.ndata[key].dtype)

    factor = crypto._immunity_factor(model.graph, torch.arange(4))
    kappa = crypto.immunity_decay_per_infection
    expected = [math.exp(-kappa * n) for n in (0, 1, 2, 4)]
    assert factor.tolist() == pytest.approx(expected, rel=1e-5)

    # Kattula et al. 2016's figure: only ~69% protection after four infections.
    assert factor[3].item() == pytest.approx(0.31, abs=0.01)

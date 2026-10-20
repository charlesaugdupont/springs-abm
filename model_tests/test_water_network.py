"""
Tests for the catchment water model (abm/environment/water_network.py):
sources from the OSM water layer and the sampling file, household source and
sanitation assignment, runoff/seepage loading from infectious households,
die-off, shocks and daily drinking-water exposure.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from abm.constants import AgentPropertyKeys, Compartment, GridLayer
from abm.environment.water_network import (
    BOREHOLE, CANAL, HH_GROUND, HH_PACKAGED, HH_PIPED, HH_SURFACE, RIVER, WELL,
    SAN_IMPROVED_ONSITE, SAN_OPEN,
)


@pytest.fixture
def model(make_model):
    return make_model(steps=1, agents=3000, run=False)


def _net(model):
    net = model.grid_environment.water_network
    assert net is not None
    return net


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_sources_cover_canals_river_and_groundwater_points(model):
    net = _net(model)
    kinds = net.source_kind
    assert int((kinds == CANAL).sum()) > 10
    assert int((kinds == RIVER).sum()) > 5
    # The SPRINGS sampling file has two wells and one borehole in Akuse.
    assert int((kinds == WELL).sum()) == 2
    assert int((kinds == BOREHOLE).sum()) == 1


def test_household_source_types_follow_the_configured_shares(model):
    net = _net(model)
    cfg = model.config.water
    H = len(net.hh_type)
    counts = torch.bincount(net.hh_type, minlength=4)
    _, p_surface, p_ground, p_packaged = cfg.source_shares()
    assert int(counts[HH_SURFACE]) == round(p_surface * H)
    assert int(counts[HH_GROUND]) == round(p_ground * H)
    assert int(counts[HH_PACKAGED]) == round(p_packaged * H)
    assert int(counts.sum()) == H


def test_surface_users_live_nearer_surface_water(model):
    net = _net(model)
    g = model.graph
    nat = model.grid_environment.get_slice(GridLayer.NATURAL_WATER).numpy() > 0
    cells = np.argwhere(nat)
    first = {}
    for i, h in enumerate(net.agent_hh.tolist()):
        first.setdefault(h, i)
    home = g.ndata[AgentPropertyKeys.HOME_LOCATION][[first[h] for h in range(len(first))]].numpy()
    d = np.min(np.hypot(home[:, None, 0] - cells[None, :, 0], home[:, None, 1] - cells[None, :, 1]), axis=1)
    surf = (net.hh_type == HH_SURFACE).numpy()
    assert d[surf].mean() < d[~surf].mean()


def test_water_locations_by_source_type(model):
    net = _net(model)
    g = model.graph
    loc = g.ndata[AgentPropertyKeys.WATER_LOCATION]
    home = g.ndata[AgentPropertyKeys.HOME_LOCATION]
    agent_type = net.hh_type[net.agent_hh]
    at_home = (agent_type == HH_PIPED) | (agent_type == HH_PACKAGED)
    assert torch.equal(loc[at_home], home[at_home])
    nat = model.grid_environment.get_slice(GridLayer.NATURAL_WATER) > 0
    surf = (agent_type == HH_SURFACE).nonzero(as_tuple=True)[0]
    assert surf.numel() > 0
    assert torch.all(nat[loc[surf, 0].long(), loc[surf, 1].long()])
    gw_cells = {tuple(c) for c in net.source_access[net.source_kind >= WELL].long().tolist()}
    for i in (agent_type == HH_GROUND).nonzero(as_tuple=True)[0].tolist():
        assert tuple(loc[i].long().tolist()) in gw_cells


def test_runoff_kernel_conserves_mass(model):
    """A household's runoff is split among reaches; the total reaching water
    is exp(-d_nearest / L) <= 1."""
    net = _net(model)
    totals = net.surface_kernel.sum(dim=1)
    assert torch.all(totals <= 1.0 + 1e-6)
    assert torch.all(totals >= 0.0)
    assert float(totals.max()) > 0.5    # households next to water lose most runoff to it


def test_release_weights_by_sanitation_and_child_stool_disposal(model):
    net = _net(model)
    cfg = model.config.water
    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].bool()
    san = net.hh_sanitation[net.agent_hh]
    adult_improved = (~is_child) & (san == SAN_IMPROVED_ONSITE)
    adult_open = (~is_child) & (san == SAN_OPEN)
    child_improved = is_child & (san == SAN_IMPROVED_ONSITE)
    assert torch.all(net.agent_surface_release[adult_improved] == cfg.surface_release_improved_onsite)
    assert torch.all(net.agent_surface_release[adult_open] == cfg.surface_release_open_defecation)
    u = cfg.child_stool_unsafe_frac
    expected = u * cfg.surface_release_open_defecation + (1 - u) * cfg.surface_release_improved_onsite
    assert torch.allclose(net.agent_surface_release[child_improved], torch.tensor(expected))


def test_legacy_model_builds_no_network(make_model):
    m = make_model(steps=1, agents=600, run=False, overrides={"water.model": "legacy"})
    assert m.grid_environment.water_network is None
    assert m.grid_environment.get_water_state("rota") is not None


# ---------------------------------------------------------------------------
# Loading, die-off, shocks
# ---------------------------------------------------------------------------

def _set_all(model, name, compartment):
    model.graph.ndata[AgentPropertyKeys.status(name)][:] = compartment


def test_loading_comes_only_from_infectious_agents_households(model):
    net = _net(model)
    _set_all(model, "crypto", Compartment.SUSCEPTIBLE)
    assert float(net.loading(model.graph, "crypto").sum()) == 0.0

    # One infectious child: its load is its release weight times its
    # household's kernel rows.
    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].bool()
    kernel_total = net.surface_kernel.sum(1) + net.ground_kernel.sum(1)
    candidates = [i for i in is_child.nonzero(as_tuple=True)[0].tolist()
                  if kernel_total[net.agent_hh[i]] > 0.1]
    i = candidates[0]
    model.graph.ndata[AgentPropertyKeys.status("crypto")][i] = Compartment.INFECTIOUS
    load = net.loading(model.graph, "crypto")
    h = net.agent_hh[i]
    expected = (net.agent_surface_release[i] * net.surface_kernel[h]
                + net.agent_ground_release[i] * net.ground_kernel[h])
    assert torch.allclose(load, expected)


def test_visiting_a_source_does_not_contaminate_it(model):
    """Contamination is from household runoff: moving an infectious agent onto
    a water cell changes nothing."""
    net = _net(model)
    _set_all(model, "crypto", Compartment.SUSCEPTIBLE)
    i = 0
    model.graph.ndata[AgentPropertyKeys.status("crypto")][i] = Compartment.INFECTIOUS
    before = net.loading(model.graph, "crypto").clone()
    access = net.source_access[0]
    model.graph.ndata[AgentPropertyKeys.Y][i] = access[0]
    model.graph.ndata[AgentPropertyKeys.X][i] = access[1]
    assert torch.equal(net.loading(model.graph, "crypto"), before)


def test_concentration_decays_first_order_and_piped_flushes_faster(model):
    net = _net(model)
    _set_all(model, "crypto", Compartment.SUSCEPTIBLE)
    c = net.concentration["crypto"]
    c[:] = 10.0
    k = 0.15
    net.advance(model.graph, "crypto", decay_rate=k, shock=False)
    S = net.num_sources
    assert torch.allclose(c[:S], torch.full((S,), 10.0 * math.exp(-k)))
    flush = model.config.water.piped_flush_rate
    assert float(c[S]) == pytest.approx(10.0 * math.exp(-(k + flush)), rel=1e-6)


def test_shock_pulse_scales_with_current_loading_and_vulnerability(model):
    net = _net(model)
    cfg = model.config.water
    _set_all(model, "crypto", Compartment.SUSCEPTIBLE)
    net.concentration["crypto"].zero_()
    net.advance(model.graph, "crypto", decay_rate=0.15, shock=True)
    assert float(net.concentration["crypto"].sum()) == 0.0   # nothing to wash in

    _set_all(model, "crypto", Compartment.INFECTIOUS)
    net.concentration["crypto"].zero_()
    load = net.loading(model.graph, "crypto")
    net.advance(model.graph, "crypto", decay_rate=0.15, shock=True)
    n_reach = int((net.source_kind <= RIVER).sum())
    pulse = cfg.shock_magnitude * float(load[:n_reach].mean())
    c = net.concentration["crypto"]
    assert torch.allclose(c[:n_reach], load[:n_reach] + pulse, rtol=1e-5)
    assert float(c[net.piped_index]) == pytest.approx(pulse * cfg.shock_vulnerability_piped, rel=1e-5)


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def test_exposure_concentration_by_source_type(model):
    net = _net(model)
    c = net.concentration["crypto"]
    c[:] = 1.0
    conc = net.agent_exposure_concentration("crypto")
    agent_type = net.hh_type[net.agent_hh]
    assert torch.all(conc[agent_type == HH_PACKAGED] == 0.0)
    assert torch.all(conc[agent_type == HH_PIPED] == 1.0)
    river_users = (net.source_kind[net.hh_source[net.agent_hh].clamp(max=net.num_sources - 1)] == RIVER) \
        & (agent_type == HH_SURFACE)
    if torch.any(river_users):
        assert torch.allclose(conc[river_users], torch.tensor(model.config.water.river_dilution))


def test_drinking_exposure_infects_only_susceptible_like_agents(model):
    crypto = next(p for p in model.pathogens if p.name == "crypto")
    n = model.graph.num_nodes()
    status = model.graph.ndata[AgentPropertyKeys.status("crypto")]
    status[:] = Compartment.SUSCEPTIBLE
    status[: n // 2] = Compartment.INFECTIOUS
    crypto.begin_day(model.graph)
    conc = torch.full((n,), 1e6)                 # dose-response saturated: P = min(1, relative susceptibility)
    torch.manual_seed(3)
    crypto.drinking_water_exposure(model.graph, conc)
    assert torch.all(status[: n // 2] == Compartment.INFECTIOUS)
    newly = status[n // 2:] == Compartment.EXPOSED
    rel = crypto._relative_susceptibility(model.graph, torch.arange(n // 2, n), route="water")
    assert not torch.any(newly[rel == 0])
    # rel includes the household exposure multiplier, which can exceed 1.
    assert float(newly.float().mean()) == pytest.approx(float(rel.clamp(max=1.0).mean()), abs=0.03)
    assert crypto.total_by_route["water"] == int(newly.sum())


def test_water_infection_probability_increases_with_concentration(model):
    for p in model.pathogens:
        if not getattr(p, "WATERBORNE", False):
            continue
        c = torch.tensor([0.0, 0.1, 1.0, 10.0])
        prob = p.water_infection_prob(c)
        assert prob[0].item() == 0.0
        assert torch.all(prob[1:] > prob[:-1])
        assert torch.all(prob <= 1.0)


def test_vaccinated_rotavirus_children_get_breakthrough_reduction(model):
    rota = next(p for p in model.pathogens if p.name == "rota")
    n = model.graph.num_nodes()
    status = model.graph.ndata[AgentPropertyKeys.status("rota")]
    status[:] = Compartment.VACCINATED
    torch.manual_seed(0)
    conc = torch.full((n,), 2.0)
    rota.drinking_water_exposure(model.graph, conc)
    frac = float((status == Compartment.EXPOSED).float().mean())
    rel = rota._relative_susceptibility(model.graph, torch.arange(n), route="water")
    expected = float((rota.water_infection_prob(conc) * rel * (1 - rota.config.vaccine_efficacy)).mean())
    assert frac == pytest.approx(expected, abs=0.03)

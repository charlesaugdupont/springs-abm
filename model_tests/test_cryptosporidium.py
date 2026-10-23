"""
Tests for the Cryptosporidium pathogen itself: its four routes, its oocyst
reservoir, its dose-response, and the subclinical-infection mechanism that
distinguishes it from the other two pathogens.
"""
from __future__ import annotations

import math

import numpy as np

import pytest
import torch

from abm.constants import AgentPropertyKeys, Compartment, Phase
from config import CryptosporidiumConfig


# ---------------------------------------------------------------------------
# Natural history / config sanity
# ---------------------------------------------------------------------------

def test_config_defaults_match_the_cited_literature():
    """Guards the literature-grounded values against accidental drift.

    Each of these is sourced in CryptosporidiumConfig's field descriptions; if
    one is changed, that citation needs revisiting, so the test is deliberately
    exact rather than a loose range.
    """
    cfg = CryptosporidiumConfig()
    assert cfg.exposure_period == 7                      # ~7-day incubation (range 2-10)
    assert cfg.recovery_rate == pytest.approx(0.0476)    # ~21-day mean shedding
    assert 1 / cfg.recovery_rate == pytest.approx(21, abs=0.1)
    # symptomatic_fraction is calibrated (campaign v3); its age profile and the
    # illness protection from prior infections are literature values
    assert cfg.symptomatic_fraction_age_profile is not None  # MAL-ED Suppl. Table 5
    assert cfg.symptomatic_reduction_per_prior_infection == pytest.approx(0.48)  # Rogawski McQuade 2020
    assert cfg.duration_scale == 1.0                     # MAL-ED: no longer than rota/campy
    assert cfg.treatment_clears_infection_prob == 0.0    # ORS/zinc do not stop shedding
    assert cfg.immunity_decay_per_infection == pytest.approx(0.29)  # 69% after 4 infections
    assert cfg.dose_response_r == pytest.approx(0.018)   # ID50 ~ 39 oocysts
    assert cfg.oocyst_decay_rate == pytest.approx(0.15)  # T90 ~ 15 days at tropical temps
    # Shedding rate is definitional - it fixes the reservoir's units.
    assert cfg.oocyst_shedding_rate == 1.0


def test_crypto_has_no_vaccine_pathway(make_model):
    """Cryptosporidium has no licensed vaccine, so the V compartment stays empty."""
    model = make_model(pathogens=["crypto"], steps=40)
    status = model.graph.ndata[AgentPropertyKeys.status("crypto")]
    assert int((status == Compartment.VACCINATED).sum().item()) == 0
    assert not hasattr(model.pathogens[0].config, "vaccination_rate")


def test_latent_period_is_exactly_exposure_period(make_model):
    """An agent exposed on day 0 becomes infectious after exposure_period days."""
    model = make_model(pathogens=["crypto"], steps=1, run=False,
                       overrides={
                           "pathogens[crypto].household_transmission_prob": 0.0,
                           "pathogens[crypto].community_infection_prob": 0.0,
                           "pathogens[crypto].water_dose_scale": 0.0,
                           "pathogens[crypto].zoonotic_interaction_rate": 0.0,
                           "pathogens[crypto].recovery_rate": 0.0,
                       })
    crypto = model.pathogens[0]
    status_key = AgentPropertyKeys.status("crypto")
    timer_key = AgentPropertyKeys.exposure_time("crypto")

    # One agent, freshly exposed, nobody else infected.
    model.graph.ndata[status_key][:] = Compartment.SUSCEPTIBLE
    model.graph.ndata[status_key][0] = Compartment.EXPOSED
    model.graph.ndata[timer_key][:] = 0

    period = crypto.config.exposure_period
    for day in range(period + 2):
        became_infectious = (
            model.graph.ndata[status_key][0].item() == Compartment.INFECTIOUS
        )
        assert became_infectious == (day >= period), (
            f"day {day}: infectious={became_infectious}, expected {day >= period}"
        )
        crypto.step_progression(model.graph)


# ---------------------------------------------------------------------------
# Subclinical infection (symptomatic_fraction)
# ---------------------------------------------------------------------------

def test_zero_symptomatic_fraction_yields_no_episodes_but_still_transmits(make_model):
    """The 'silent reservoir': infection and transmission without any illness.

    This is the mechanism the whole symptomatic_fraction refactor exists for -
    an asymptomatic carrier must still progress through INFECTIOUS and still
    infect others, while generating no episode, no health loss and no
    care-seeking decision.
    """
    model = make_model(pathogens=["crypto"], steps=50,
                       overrides={"pathogens[crypto].symptomatic_fraction": 0.0})
    crypto = model.pathogens[0]

    episodes = [e for e in model.get_child_episode_log() if e["pathogen"] == "crypto"]
    assert episodes == [], "no episode may start when symptomatic_fraction is 0"

    # ... yet transmission happened, driven by agents who were never ill.
    assert sum(crypto.total_by_route.values()) > 0
    severity = model.graph.ndata[AgentPropertyKeys.symptom_severity("crypto")]
    duration = model.graph.ndata[AgentPropertyKeys.illness_duration("crypto")]
    assert torch.all(severity == 0)
    assert torch.all(duration == 0)


def test_symptomatic_fraction_scales_episode_count(make_model):
    """Episodes should scale roughly with symptomatic_fraction, infections should not.

    Infections are driven by the compartment model, which the symptomatic draw
    does not touch, so the same seed gives a similar infection count at either
    setting while the episode count tracks the fraction.

    Deliberately run at a much larger scale than the other tests here, and
    measured empirically rather than guessed. At the default 600 agents /
    30 days the comparison is vacuous (both arms log ~0 episodes), and even at
    1500 agents / 150 days the under-5 infection count is single digits - too
    few for a ratio to mean anything. Two reasons crypto is slow to get going
    at small scale: its waterborne route strength depends on the number of
    shedders per water source (this grid has only 11), and its 7-day latent
    period delays the take-off. 2500 agents / 250 days gives ~90 under-5
    infections per arm, which is enough.
    """
    infections, episodes = {}, {}
    for frac in (0.33, 1.0):
        # Flat in age, so the configured fraction is the probability itself
        # (with the MAL-ED profile, frac=1.0 would be clamped in young bands).
        model = make_model(pathogens=["crypto"], steps=250, agents=2500,
                           overrides={"pathogens[crypto].symptomatic_fraction": frac,
                                      "pathogens[crypto].symptomatic_fraction_age_profile": None,
                                      # boosted only so both arms log enough episodes
                                      "pathogens[crypto].household_transmission_prob": 0.04})
        is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
        infections[frac] = int(
            model.graph.ndata[AgentPropertyKeys.num_infections("crypto")][is_child].sum().item()
        )
        episodes[frac] = sum(
            1 for e in model.get_child_episode_log() if e["pathogen"] == "crypto"
        )

    assert episodes[1.0] > episodes[0.33] > 0
    assert min(infections.values()) > 30, (
        f"too few infections ({infections}) for the ratio below to be meaningful"
    )
    ratio = (episodes[0.33] / episodes[1.0])
    # Generous tolerance: the two runs' RNG streams diverge once their
    # infection counts differ, so this is a scaling check, not an identity.
    assert 0.15 < ratio < 0.65, f"episode ratio {ratio:.3f} not consistent with 0.33"


def test_an_asymptomatic_infection_is_not_re_rolled_for_symptoms(make_model):
    """One symptomatic draw per infection, not one per infectious day.

    Shedding lasts ~3 weeks, so if the draw were repeated daily, almost every
    infection would eventually win it and symptomatic_fraction would be
    defeated. Checked by driving a fixed cohort of infectious children with
    symptomatic_fraction just below 1 and asserting episodes never exceed the
    number of distinct infections.
    """
    model = make_model(pathogens=["crypto"], steps=1, run=False,
                       overrides={
                           "pathogens[crypto].symptomatic_fraction": 0.5,
                           "pathogens[crypto].household_transmission_prob": 0.0,
                           "pathogens[crypto].community_infection_prob": 0.0,
                           "pathogens[crypto].water_dose_scale": 0.0,
                           "pathogens[crypto].zoonotic_interaction_rate": 0.0,
                           "pathogens[crypto].recovery_rate": 0.0,
                       })
    from abm.systems.child_illness import ChildIllnessSystem

    illness = next(s for s in model.systems if isinstance(s, ChildIllnessSystem))
    status_key = AgentPropertyKeys.status("crypto")
    count_key = AgentPropertyKeys.num_infections("crypto")

    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
    children = is_child.nonzero(as_tuple=True)[0]
    assert children.numel() > 20

    # One single infection each, all already infectious, and never recovering.
    model.graph.ndata[status_key][:] = Compartment.SUSCEPTIBLE
    model.graph.ndata[status_key][children] = Compartment.INFECTIOUS
    model.graph.ndata[count_key][:] = 0
    model.graph.ndata[count_key][children] = 1
    model.graph.ndata[AgentPropertyKeys.last_episode_infection("crypto")][:] = -1

    for day in range(30):
        illness.update(model.graph, timestep=day)

    episodes = [e for e in illness.episode_log if e["pathogen"] == "crypto"]
    assert len(episodes) <= children.numel(), (
        f"{len(episodes)} episodes from {children.numel()} infections - the symptomatic "
        "draw is being repeated while the agent stays infectious"
    )
    # And each child got at most one episode.
    counts: dict[int, int] = {}
    for e in episodes:
        counts[e["agent_idx"]] = counts.get(e["agent_idx"], 0) + 1
    assert max(counts.values(), default=0) <= 1


def test_carriage_prevalence_exceeds_symptomatic_prevalence(make_model):
    """Shedding prevalence must run well above disease prevalence for crypto.

    Confusing the two is the single easiest way to mis-score this pathogen
    against the literature, since the comparable field measurements are
    molecular carriage surveys, not diarrhoea prevalence.
    """
    # Mean, not peak, prevalence: with ~300 under-5s a single symptomatic
    # child moves the peak by 0.003, so peaks are too noisy to compare.
    model = make_model(pathogens=["crypto"], steps=300, agents=3000)
    carriage = float(np.mean(model.u5_prevalence_history["crypto"]))
    symptomatic = float(np.mean(model.u5_symptomatic_prevalence_history["crypto"]))
    assert carriage > 0
    assert symptomatic < carriage, "symptomatic prevalence should be a subset of carriage"
    assert carriage > 5 * symptomatic, (
        f"carriage {carriage:.4f} vs symptomatic {symptomatic:.4f} - expected a "
        "large gap given ~3-week shedding, ~3-day episodes and ~82% subclinical infection"
    )


# ---------------------------------------------------------------------------
# Dose-response
# ---------------------------------------------------------------------------

def test_exponential_dose_response_shape(make_model):
    model = make_model(pathogens=["crypto"], steps=1, run=False)
    crypto = model.pathogens[0]
    r = crypto.config.dose_response_r

    dose = torch.tensor([0.0, 1.0, 10.0, 100.0, 1e4])
    prob = crypto._infection_prob_from_dose(dose)

    assert prob[0].item() == 0.0                       # no dose, no infection
    assert torch.all(prob[1:] > prob[:-1])             # monotone increasing
    assert torch.all((prob >= 0) & (prob <= 1))
    assert prob[-1].item() == pytest.approx(1.0, abs=1e-6)   # saturates
    expected = [1 - math.exp(-r * d) for d in dose.tolist()]
    assert prob.tolist() == pytest.approx(expected, rel=1e-5)


def test_id50_matches_the_dose_response_parameter(make_model):
    """ID50 = ln2 / r. The default r implies ~39 oocysts.

    That sits inside the range of human-challenge ID50 estimates for the
    C. hominis / C. parvum isolates relevant here (10-83 for C. hominis TU502;
    87-132 for C. parvum Iowa; 9 for TAMU).
    """
    model = make_model(pathogens=["crypto"], steps=1, run=False)
    crypto = model.pathogens[0]
    id50 = math.log(2) / crypto.config.dose_response_r
    assert crypto._infection_prob_from_dose(torch.tensor([id50]))[0].item() == pytest.approx(0.5)
    assert 10 <= id50 <= 90


# ---------------------------------------------------------------------------
# Oocyst reservoir (LEGACY point-source water model; the catchment model is
# tested in model_tests/test_water_network.py)
# ---------------------------------------------------------------------------

def test_oocyst_load_decays_exponentially(make_model):
    """load(t) = load(0) * exp(-k t), to numerical precision over 30 days."""
    model = make_model(pathogens=["crypto"], steps=1, run=False,
                       overrides={"water.model": "legacy"})   # point-source reservoir
    crypto = model.pathogens[0]
    grid = model.grid_environment
    k = crypto.config.oocyst_decay_rate

    state = grid.get_water_state("crypto")
    mask = grid.water_point_mask()
    state[mask] = 100.0
    initial = 100.0

    for day in range(1, 31):
        crypto.decay_water_reservoir(grid)
        expected = initial * math.exp(-k * day)
        got = state[mask].max().item()
        assert got == pytest.approx(expected, rel=1e-4), f"day {day}: {got} != {expected}"

    # Sanity-check the headline interpretation: T90 ~ 15 days at k = 0.15.
    t90 = math.log(10) / k
    assert t90 == pytest.approx(15.4, abs=0.5)


def test_oocyst_decay_consumes_no_randomness(make_model):
    """Deterministic decay is what keeps rotavirus's RNG stream untouched.

    Rotavirus's reservoir recovery draws random numbers; if crypto's decay did
    too, enabling crypto would shift rotavirus's draws and silently change its
    calibrated behaviour even though nothing about rotavirus had changed.
    """
    model = make_model(pathogens=["crypto"], steps=1, run=False,
                       overrides={"water.model": "legacy"})   # point-source reservoir
    crypto = model.pathogens[0]
    grid = model.grid_environment
    grid.get_water_state("crypto")[grid.water_point_mask()] = 50.0

    before = torch.rand(1).item()
    torch.manual_seed(1234)
    a = torch.rand(5)
    torch.manual_seed(1234)
    crypto.decay_water_reservoir(grid)
    b = torch.rand(5)
    assert torch.equal(a, b), "decay_water_reservoir consumed randomness"
    assert isinstance(before, float)


def test_shedding_adds_load_only_at_water_sources(make_model):
    model = make_model(pathogens=["crypto"], steps=1, run=False,
                       overrides={"water.model": "legacy"})   # point-source reservoir
    crypto = model.pathogens[0]
    grid = model.grid_environment
    state = grid.get_water_state("crypto")
    mask = grid.water_point_mask()

    # Put every agent on a water source and make them all infectious.
    water_cells = mask.nonzero(as_tuple=False)
    target = water_cells[0]
    n = model.graph.num_nodes()
    model.graph.ndata[AgentPropertyKeys.Y][:] = float(target[0].item())
    model.graph.ndata[AgentPropertyKeys.X][:] = float(target[1].item())
    model.graph.ndata[AgentPropertyKeys.status("crypto")][:] = Compartment.INFECTIOUS

    state.zero_()
    crypto._shed_into_water(model.graph, grid)

    assert state[target[0], target[1]].item() == pytest.approx(
        n * crypto.config.oocyst_shedding_rate
    )
    # Nothing anywhere else, water source or not.
    total = state.sum().item()
    assert total == pytest.approx(n * crypto.config.oocyst_shedding_rate)


def test_only_infectious_agents_shed(make_model):
    model = make_model(pathogens=["crypto"], steps=1, run=False,
                       overrides={"water.model": "legacy"})   # point-source reservoir
    crypto = model.pathogens[0]
    grid = model.grid_environment
    state = grid.get_water_state("crypto")
    target = grid.water_point_mask().nonzero(as_tuple=False)[0]

    model.graph.ndata[AgentPropertyKeys.Y][:] = float(target[0].item())
    model.graph.ndata[AgentPropertyKeys.X][:] = float(target[1].item())
    for compartment in (Compartment.SUSCEPTIBLE, Compartment.EXPOSED, Compartment.RECOVERED):
        model.graph.ndata[AgentPropertyKeys.status("crypto")][:] = compartment
        state.zero_()
        crypto._shed_into_water(model.graph, grid)
        assert state.sum().item() == 0.0, f"{compartment!r} agents must not shed"


# ---------------------------------------------------------------------------
# Routes in isolation
# ---------------------------------------------------------------------------

ROUTE_KNOBS = {
    "household": "pathogens[crypto].household_transmission_prob",
    "community": "pathogens[crypto].community_infection_prob",
    "water": "pathogens[crypto].water_dose_scale",
    "zoonotic": "pathogens[crypto].zoonotic_interaction_rate",
    "imported": "pathogens[crypto].importation_rate",
}


@pytest.mark.parametrize("route", list(ROUTE_KNOBS))
def test_zeroing_a_route_knob_silences_exactly_that_route(make_model, route):
    overrides = {path: 0.0 for path in ROUTE_KNOBS.values()}
    model = make_model(pathogens=["crypto"], steps=60, overrides=overrides)
    crypto = model.pathogens[0]
    assert crypto.total_by_route[route] == 0


@pytest.mark.parametrize("route,knob", list(ROUTE_KNOBS.items()))
def test_each_route_can_transmit_on_its_own(make_model, route, knob):
    """With only one route enabled, that route must still produce infections.

    A route that cannot transmit in isolation is either mis-wired or
    unreachable, and the calibration search would quietly drive its parameter
    to an arbitrary value because it has no effect.
    """
    overrides = {path: 0.0 for path in ROUTE_KNOBS.values()}
    # Generous value so a short, small test run still sees cases.
    overrides[knob] = {"household": 0.25, "community": 0.03,
                       "water": 20.0, "zoonotic": 3.0, "imported": 0.002}[route]
    if route == "water":
        # The run starts with ~1% of agents exposed. At the survey shares most
        # households have improved sanitation (no surface release) and drink
        # piped water, so whether any of those few seeds can contaminate and
        # infect via water at all is luck of the draw - it went from 5
        # infections to 0 when an unrelated default changed the random stream.
        # Every household open-defecating and drinking surface water makes the
        # wiring test independent of that luck.
        overrides.update({"water.share_open_defecation": 1.0, "water.share_improved_onsite": 0.0,
                          "water.share_sewer": 0.0, "water.share_unimproved": 0.0,
                          "water.share_surface": 1.0, "water.share_piped": 0.0,
                          "water.share_ground": 0.0, "water.share_packaged": 0.0})
    model = make_model(pathogens=["crypto"], steps=80, agents=1200, overrides=overrides)
    crypto = model.pathogens[0]
    assert crypto.total_by_route[route] > 0, f"{route} produced no infections in isolation"
    others = {r: n for r, n in crypto.total_by_route.items() if r != route}
    assert all(n == 0 for n in others.values()), f"other routes fired too: {others}"


def test_zoonotic_route_reads_ruminants_not_poultry(make_model):
    """Birds are not hosts of the human-infecting Cryptosporidium species.

    Campylobacter's zoonotic dose is poultry-dominated; crypto's must ignore
    poultry entirely, even though both read the same density layers.
    """
    overrides = {path: 0.0 for path in ROUTE_KNOBS.values()}
    overrides["pathogens[crypto].zoonotic_interaction_rate"] = 5.0
    model = make_model(pathogens=["crypto"], steps=40, overrides=overrides, run=False)
    grid = model.grid_environment

    # Saturate poultry, erase ruminants.
    grid.set_dynamic_layer("poultry_density", torch.ones_like(grid.get_dynamic_layer("poultry_density")))
    grid.set_dynamic_layer("ruminant_density", torch.zeros_like(grid.get_dynamic_layer("ruminant_density")))

    child_mask = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}
    for _ in range(40):
        model.step(child_mask=child_mask)

    assert model.pathogens[0].total_by_route["zoonotic"] == 0, (
        "crypto's zoonotic route responded to poultry density"
    )


def test_zoonotic_route_responds_to_ruminant_density(make_model):
    """Companion to the previous test - ruminants must actually drive it."""
    overrides = {path: 0.0 for path in ROUTE_KNOBS.values()}
    overrides["pathogens[crypto].zoonotic_interaction_rate"] = 5.0
    model = make_model(pathogens=["crypto"], steps=40, overrides=overrides, run=False)
    grid = model.grid_environment
    grid.set_dynamic_layer("poultry_density", torch.zeros_like(grid.get_dynamic_layer("poultry_density")))
    grid.set_dynamic_layer("ruminant_density", torch.ones_like(grid.get_dynamic_layer("ruminant_density")))

    child_mask = model.graph.ndata[AgentPropertyKeys.IS_CHILD]
    model.u5_prevalence_history = {p.name: [] for p in model.pathogens}
    model.u5_symptomatic_prevalence_history = {p.name: [] for p in model.pathogens}
    for _ in range(40):
        model.step(child_mask=child_mask)

    assert model.pathogens[0].total_by_route["zoonotic"] > 0


def test_household_route_fires_in_the_night_phase_only(make_model):
    model = make_model(pathogens=["crypto"], steps=1, run=False,
                       overrides={"pathogens[crypto].household_transmission_prob": 0.4})
    crypto = model.pathogens[0]
    crypto.begin_day(model.graph)
    # Make a broad set infectious so a household route call would surely fire.
    model.graph.ndata[AgentPropertyKeys.status("crypto")][::3] = Compartment.INFECTIOUS

    crypto.step_transmission(model.graph, torch.zeros(model.graph.num_nodes(), dtype=torch.long),
                             1, model.grid_environment, Phase.DAY)
    assert crypto.cases_by_route["household"] == 0

    crypto.step_transmission(model.graph, torch.zeros(model.graph.num_nodes(), dtype=torch.long),
                             1, model.grid_environment, Phase.NIGHT)
    assert crypto.cases_by_route["household"] > 0

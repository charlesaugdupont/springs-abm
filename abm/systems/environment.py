# abm/systems/environment.py
from typing import Any
import torch

from .system import System
from abm.state import AgentState


class EnvironmentSystem(System):
    """Handles environmental dynamics: water-reservoir die-off and shocks.

    Water contamination state is PER PATHOGEN (see
    GridEnvironment.set_water_state and the rationale there): rotavirus holds a
    binary CLEAN/CONTAMINATED flag that reverts stochastically on a ~5-day
    timescale, while Cryptosporidium holds a continuous oocyst load that decays
    first-order over weeks. This system therefore does not implement either
    rule itself - it drives each waterborne pathogen's own
    decay_water_reservoir() / apply_water_shock(), so adding a pathogen with
    different environmental persistence needs no change here.

    A shock is a single whole-supply event (flooding, infrastructure failure),
    so one roll against shock_daily_prob applies to every waterborne pathogen
    at once rather than being drawn independently per pathogen - physically the
    same flood contaminates the same sources for everything living in them.
    """

    def update(self, agent_state: AgentState, **kwargs):
        grid     = kwargs.get("grid")
        timestep = kwargs.get("timestep")
        pathogens = kwargs.get("pathogens") or []
        if grid is None or timestep is None:
            return

        waterborne = [p for p in pathogens if getattr(p, "WATERBORNE", False)]

        # Catchment model: one shock roll for the day, then each pathogen's
        # sources decay, receive today's faecal loading from infectious
        # households, and (on a shock day) a loading pulse.
        network = getattr(grid, "water_network", None)
        if network is not None:
            shock = torch.rand(1).item() < self.config.steering_parameters.shock_daily_prob
            for pathogen in waterborne:
                if pathogen.name in network.concentration:
                    network.advance(agent_state, pathogen.name, pathogen.water_decay_rate, shock)
            return

        for pathogen in waterborne:
            pathogen.decay_water_reservoir(grid)

        if torch.rand(1).item() < self.config.steering_parameters.shock_daily_prob:
            for pathogen in waterborne:
                pathogen.apply_water_shock(grid)

# abm/pathogens/campylobacter.py
"""
Campylobacter transmission model.

Two independent routes operate each day:

1. Zoonotic (environmental) route
   Agents are exposed to a dose derived from the local animal-density layer.
   Infection probability follows the exact Beta-Poisson dose-response model.
   This route fires during both the Day Phase (activity location) and the
   Night Phase (home location).

2. Fecal-oral (household) route
   Infectious agents contaminate their household environment.  Susceptible
   household members are exposed with a fixed per-contact probability
   `fecal_oral_prob`.  This route runs once per day during the Night Phase.

Route tracking
--------------
Each new infection is attributed to exactly one route.  The counters
``cases_zoonotic`` and ``cases_fecal_oral`` are incremented accordingly and
are reset at the start of each day alongside ``new_cases_this_step``.
Over the full simulation, ``total_zoonotic`` and ``total_fecal_oral``
accumulate the lifetime totals, which are used to compute the zoonotic
fraction at the end of a run.
"""

from typing import Any
import torch
from scipy.special import hyp1f1

from .pathogen import Pathogen
from abm.state import AgentState
from abm.constants import Compartment, AgentPropertyKeys, GridLayer
from config import CampylobacterConfig, SteeringParamsSVEIR


class Campylobacter(Pathogen):
    """Implements the logic for Campylobacter."""

    def __init__(
        self,
        config: CampylobacterConfig,
        global_params: SteeringParamsSVEIR,
        device: torch.device,
    ):
        super().__init__(config, global_params, device)
        self.config: CampylobacterConfig = config

        # Per-step route counters (reset each day)
        self.cases_zoonotic: int = 0
        self.cases_fecal_oral: int = 0

        # Lifetime route totals (accumulate across all steps)
        self.total_zoonotic: int = 0
        self.total_fecal_oral: int = 0

        # Tracks agents newly exposed within the current day to prevent
        # double-counting across the two transmission phases.
        self._newly_exposed_this_day: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Pathogen interface
    # ------------------------------------------------------------------

    def reset_incidence(self):
        """Reset per-step counters. Called once at the start of each day."""
        super().reset_incidence()
        self.cases_zoonotic = 0
        self.cases_fecal_oral = 0

    def step_progression(self, agent_state: AgentState):
        """Disease-state progression — called once per day."""
        self._newly_exposed_this_day = torch.zeros(
            agent_state.num_nodes(), dtype=torch.bool, device=self.device
        )
        self._increment_exposure_time(agent_state)
        self._exposed_to_infectious(agent_state)
        self._infectious_to_recovered(agent_state)

    def step_transmission(
        self,
        agent_state: AgentState,
        location_ids: torch.Tensor,
        num_locations: int,
        grid: Any,
    ):
        """
        Transmission — called twice per day (Day Phase and Night Phase).

        The zoonotic route fires on both calls.
        The fecal-oral route fires only during the Night Phase.
        """
        self._zoonotic_transmission(agent_state, grid)
        if self._is_night_phase(agent_state):
            self._fecal_oral_transmission(agent_state)

    @property
    def zoonotic_fraction(self) -> float | None:
        """
        Fraction of all campylobacter infections attributable to the zoonotic
        route over the lifetime of the simulation.  Returns None if no
        infections have occurred yet.
        """
        total = self.total_zoonotic + self.total_fecal_oral
        if total == 0:
            return None
        return self.total_zoonotic / total

    # ------------------------------------------------------------------
    # Private: zoonotic route
    # ------------------------------------------------------------------

    def _zoonotic_transmission(self, agent_state: AgentState, grid: Any):
        """
        Beta-Poisson dose-response infection from the animal density layer.

        P(inf | dose) = 1 − 1F1(alpha, alpha + beta, −dose)
        """
        if grid is None or self._newly_exposed_this_day is None:
            return

        animal_idx = grid.property_to_index.get(GridLayer.ANIMAL_DENSITY)
        if animal_idx is None:
            return

        grid_shape = grid.grid_shape
        x = agent_state.ndata[AgentPropertyKeys.X].long().clamp(0, grid_shape[1] - 1)
        y = agent_state.ndata[AgentPropertyKeys.Y].long().clamp(0, grid_shape[0] - 1)

        local_density = grid.grid_tensor[y, x, animal_idx]
        dose = local_density * self.config.human_animal_interaction_rate

        status_key = AgentPropertyKeys.status(self.name)
        count_key = AgentPropertyKeys.num_infections(self.name)

        susceptible_mask = (
            (agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE)
            & ~self._newly_exposed_this_day
        )
        if not torch.any(susceptible_mask):
            return

        alpha = self.config.beta_poisson_alpha
        beta = self.config.beta_poisson_beta

        target_doses = dose[susceptible_mask].cpu().numpy()
        prob_inf_np = 1.0 - hyp1f1(alpha, alpha + beta, -target_doses)

        prob_infection = torch.zeros(agent_state.num_nodes(), device=self.device)
        prob_infection[susceptible_mask] = (
            torch.from_numpy(prob_inf_np).to(self.device).float()
        )

        num_prior = agent_state.ndata[count_key][susceptible_mask].float()
        immunity_factor = torch.exp(
            -self.global_params.prior_infection_immunity_factor * num_prior
        )
        prob_infection[susceptible_mask] *= immunity_factor

        rand_vals = torch.rand(agent_state.num_nodes(), device=self.device)
        new_infections = (rand_vals < prob_infection) & susceptible_mask

        num_new = int(torch.sum(new_infections).item())
        if num_new > 0:
            agent_state.ndata[status_key][new_infections] = Compartment.EXPOSED
            agent_state.ndata[AgentPropertyKeys.exposure_time(self.name)][new_infections] = 0
            agent_state.ndata[count_key][new_infections] += 1
            self.new_cases_this_step += num_new
            self.cases_zoonotic += num_new
            self.total_zoonotic += num_new
            self._newly_exposed_this_day[new_infections] = True

    # ------------------------------------------------------------------
    # Private: fecal-oral household route
    # ------------------------------------------------------------------

    def _fecal_oral_transmission(self, agent_state: AgentState):
        """
        Within-household fecal-oral transmission.

        For each household containing at least one infectious agent, every
        susceptible household member faces a Bernoulli trial with probability
        `fecal_oral_prob`. Agents already exposed today are excluded.
        """
        if self._newly_exposed_this_day is None:
            return

        status_key = AgentPropertyKeys.status(self.name)
        count_key  = AgentPropertyKeys.num_infections(self.name)
        hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]

        is_infectious = agent_state.ndata[status_key] == Compartment.INFECTIOUS
        if not torch.any(is_infectious):
            return

        infectious_hh = torch.unique(hh_ids[is_infectious])

        is_susceptible = (
            (agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE)
            & ~self._newly_exposed_this_day
        )
        in_infectious_hh = torch.isin(hh_ids, infectious_hh)
        target_mask = is_susceptible & in_infectious_hh

        if not torch.any(target_mask):
            return

        target_indices = target_mask.nonzero(as_tuple=True)[0]

        num_prior = agent_state.ndata[count_key][target_indices].float()
        immunity_factor = torch.exp(
            -self.global_params.prior_infection_immunity_factor * num_prior
        )

        effective_prob = self.config.fecal_oral_prob * immunity_factor
        rand_vals = torch.rand(len(target_indices), device=self.device)
        newly_infected = rand_vals < effective_prob

        infected_indices = target_indices[newly_infected]
        num_new = len(infected_indices)
        if num_new > 0:
            agent_state.ndata[status_key][infected_indices] = Compartment.EXPOSED
            agent_state.ndata[AgentPropertyKeys.exposure_time(self.name)][infected_indices] = 0
            agent_state.ndata[count_key][infected_indices] += 1
            self.new_cases_this_step += num_new
            self.cases_fecal_oral    += num_new
            self.total_fecal_oral    += num_new
            self._newly_exposed_this_day[infected_indices] = True

    # ------------------------------------------------------------------
    # Private: phase detection
    # ------------------------------------------------------------------

    def _is_night_phase(self, agent_state: AgentState) -> bool:
        """Returns True when all agents are at their home location."""
        home  = agent_state.ndata[AgentPropertyKeys.HOME_LOCATION]
        cur_y = agent_state.ndata[AgentPropertyKeys.Y]
        cur_x = agent_state.ndata[AgentPropertyKeys.X]
        at_home = (cur_y == home[:, 0]) & (cur_x == home[:, 1])
        return bool(torch.all(at_home).item())

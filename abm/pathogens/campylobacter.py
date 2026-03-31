# abm/pathogens/campylobacter.py
"""
Campylobacter transmission model.

Two independent routes operate each day:

1. Zoonotic (environmental) route
   Agents are exposed to a dose derived from the local animal-density layer.
   Infection probability follows the exact Beta-Poisson dose-response model.
   This route fires during both the Day Phase (activity location) and the
   Night Phase (home location), so agents are exposed to whatever animal
   density exists at their current grid cell.

2. Fecal-oral (household) route
   Infectious agents contaminate their household environment. Susceptible
   household members are exposed with a fixed per-contact probability
   `fecal_oral_prob`. This route runs once per day (after agents have
   returned home) and represents the dominant within-household pathway
   (contaminated hands, shared food/water, inadequate sanitation).

An agent that is already newly exposed during the Day Phase zoonotic step is
excluded from further exposure that day to avoid double-counting incidence.
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
        # Tracks agents newly exposed *within the current day* so that the
        # two transmission phases do not double-count the same infection event.
        self._newly_exposed_this_day: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Pathogen interface
    # ------------------------------------------------------------------

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

        The zoonotic route uses the agent's current grid cell each time.
        The fecal-oral household route is appended only during the Night
        Phase (when all agents are at home), identified by checking whether
        all agents share their home coordinates. We use a simple flag
        approach: the household route fires when ``_newly_exposed_this_day``
        already exists but agents are back at home (Night Phase).
        """
        self._zoonotic_transmission(agent_state, grid)
        # Fecal-oral route: run once per day, during the Night Phase.
        # We detect the Night Phase by checking whether this is the second
        # call (i.e. _newly_exposed_this_day is already populated).
        # A cleaner alternative would be an explicit phase flag passed in
        # via kwargs, but this avoids changing the Pathogen interface.
        if self._is_night_phase(agent_state):
            self._fecal_oral_transmission(agent_state)

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

        # Acquired immunity
        num_prior = agent_state.ndata[count_key][susceptible_mask].float()
        immunity_factor = torch.exp(
            -self.global_params.prior_infection_immunity_factor * num_prior
        )
        prob_infection[susceptible_mask] *= immunity_factor

        rand_vals = torch.rand(agent_state.num_nodes(), device=self.device)
        new_infections = (rand_vals < prob_infection) & susceptible_mask

        num_new = torch.sum(new_infections).item()
        if num_new > 0:
            agent_state.ndata[status_key][new_infections] = Compartment.EXPOSED
            agent_state.ndata[AgentPropertyKeys.exposure_time(self.name)][new_infections] = 0
            agent_state.ndata[count_key][new_infections] += 1
            self.new_cases_this_step += num_new
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

        This is intentionally simple: it does not scale with the number of
        infectious members (one infectious person is sufficient to contaminate
        a shared latrine / food preparation area).
        """
        if self._newly_exposed_this_day is None:
            return

        status_key = AgentPropertyKeys.status(self.name)
        count_key = AgentPropertyKeys.num_infections(self.name)
        hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]

        is_infectious = agent_state.ndata[status_key] == Compartment.INFECTIOUS
        if not torch.any(is_infectious):
            return

        # Households with at least one infectious member
        infectious_hh = torch.unique(hh_ids[is_infectious])

        # Susceptible agents in those households who have not been exposed today
        is_susceptible = (
            (agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE)
            & ~self._newly_exposed_this_day
        )
        in_infectious_hh = torch.isin(hh_ids, infectious_hh)
        target_mask = is_susceptible & in_infectious_hh

        if not torch.any(target_mask):
            return

        target_indices = target_mask.nonzero(as_tuple=True)[0]

        # Acquired immunity
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
            self._newly_exposed_this_day[infected_indices] = True

    # ------------------------------------------------------------------
    # Private: phase detection
    # ------------------------------------------------------------------

    def _is_night_phase(self, agent_state: AgentState) -> bool:
        """
        Returns True when agents are in the Night Phase (all at home).

        We compare current (x, y) against home locations. If every agent's
        current position matches their home position, we are in the Night
        Phase. This is O(N) but runs only twice per step.
        """
        home = agent_state.ndata[AgentPropertyKeys.HOME_LOCATION]
        cur_y = agent_state.ndata[AgentPropertyKeys.Y]
        cur_x = agent_state.ndata[AgentPropertyKeys.X]
        at_home = (cur_y == home[:, 0]) & (cur_x == home[:, 1])
        return bool(torch.all(at_home).item())

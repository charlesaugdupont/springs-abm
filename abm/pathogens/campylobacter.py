# abm/pathogens/campylobacter.py
from typing import Any
import torch
from scipy.special import hyp1f1

from .pathogen import Pathogen
from abm.state import AgentState
from abm.constants import Compartment, AgentPropertyKeys, GridLayer
from config import CampylobacterConfig, SteeringParamsSVEIR

class Campylobacter(Pathogen):
    """Implements the logic for Campylobacter, a bacterial, zoonotic-driven pathogen."""

    def __init__(self, config: CampylobacterConfig, global_params: SteeringParamsSVEIR, device: torch.device):
        super().__init__(config, global_params, device)
        self.config: CampylobacterConfig = config
        self._newly_exposed_this_day: torch.Tensor | None = None

    def step_progression(self, agent_state: AgentState):
        """Internal progression (Once per day)."""
        # Reset the within-day exposure tracker at the start of each new day
        self._newly_exposed_this_day = torch.zeros(
            agent_state.num_nodes(), dtype=torch.bool, device=self.device
        )
        self._increment_exposure_time(agent_state)
        self._exposed_to_infectious(agent_state)
        self._infectious_to_recovered(agent_state)

    def step_transmission(self, agent_state: AgentState, location_ids: torch.Tensor, num_locations: int, grid: Any):
        """Transmission (Day and Night)."""
        # Campylobacter is environmental. It depends on where the agent IS.
        # This works perfectly with the Day/Night loop: 
        # Day: Exposed to animals at Activity Location (e.g. School surroundings)
        # Night: Exposed to animals at Home.
        self._animal_to_human_transmission(agent_state, grid)

    def _animal_to_human_transmission(self, agent_state: AgentState, grid: Any):
        """
        Exact Beta-Poisson dose-response infection from the animal density layer.
        P(inf) = 1 − 1F1(alpha, alpha + beta, −dose)

        Agents that were already newly exposed earlier in the same day are
        excluded so that incidence is not double-counted across the two
        transmission phases.
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

        # Exclude agents already exposed today so we don't double-count them.
        susceptible_mask = (
            (agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE)
            & ~self._newly_exposed_this_day
        )

        if not torch.any(susceptible_mask):
            return

        # --- Exact Beta-Poisson probability (via scipy) ---
        alpha = self.config.beta_poisson_alpha
        beta = self.config.beta_poisson_beta

        target_doses = dose[susceptible_mask].cpu().numpy()
        prob_inf_np = 1.0 - hyp1f1(alpha, alpha + beta, -target_doses)

        prob_infection = torch.zeros(agent_state.num_nodes(), device=self.device)
        prob_infection[susceptible_mask] = (
            torch.from_numpy(prob_inf_np).to(self.device).float()
        )

        # --- Acquired immunity ---
        num_prior = agent_state.ndata[count_key][susceptible_mask].float()
        immunity_factor = torch.exp(
            -self.global_params.prior_infection_immunity_factor * num_prior
        )
        prob_infection[susceptible_mask] *= immunity_factor

        # --- Stochastic infection event ---
        rand_vals = torch.rand(agent_state.num_nodes(), device=self.device)
        new_infections = (rand_vals < prob_infection) & susceptible_mask

        num_new = torch.sum(new_infections).item()
        if num_new > 0:
            agent_state.ndata[status_key][new_infections] = Compartment.EXPOSED
            agent_state.ndata[AgentPropertyKeys.exposure_time(self.name)][new_infections] = 0
            # Increment infection count at exposure (consistent with incidence counter).
            agent_state.ndata[count_key][new_infections] += 1
            self.new_cases_this_step += num_new
            # Mark these agents so the second transmission phase skips them.
            self._newly_exposed_this_day[new_infections] = True
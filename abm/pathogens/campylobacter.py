# abm/pathogens/campylobacter.py
from typing import Any
import torch

from .pathogen import Pathogen
from abm.agent_graph import AgentGraph
from abm.constants import Compartment, AgentPropertyKeys
from config import CampylobacterConfig, SteeringParamsSVEIR

class Campylobacter(Pathogen):
    """Implements the logic for Campylobacter, a bacterial, zoonotic-driven pathogen."""

    def __init__(self, config: CampylobacterConfig, global_params: SteeringParamsSVEIR, device: torch.device):
        super().__init__(config, global_params, device)
        # Ensure config is of the correct type for type hinting
        self.config: CampylobacterConfig = config

    def update(self, agent_graph: AgentGraph, adjacency: torch.Tensor, grid: Any):
        """Runs the full update cycle for Campylobacter."""
        self.reset_incidence()

        # 1. Update internal disease progression
        self._increment_exposure_time(agent_graph)
        self._exposed_to_infectious(agent_graph)
        self._infectious_to_recovered(agent_graph)

        # 2. Handle transmission from animal reservoir
        self._animal_to_human_transmission(agent_graph, grid)

    def _animal_to_human_transmission(self, agent_graph: AgentGraph, grid: Any):
        """Handles Beta-Poisson infection from the animal density environmental layer."""
        animal_idx = grid.property_to_index.get('animal_density')
        if animal_idx is None:
            return

        # Get agent locations, ensuring they are within grid bounds
        grid_shape = grid.grid_shape
        x = agent_graph.ndata[AgentPropertyKeys.X].long().clamp(0, grid_shape[1] - 1)
        y = agent_graph.ndata[AgentPropertyKeys.Y].long().clamp(0, grid_shape[0] - 1)

        # Calculate dose from local animal density
        local_density = grid.grid_tensor[y, x, animal_idx]
        dose = local_density * self.config.human_animal_interaction_rate

        # Beta-Poisson dose-response model to calculate probability of infection
        alpha = self.config.beta_poisson_alpha
        beta = self.config.beta_poisson_beta
        prob_infection = 1.0 - torch.pow(1.0 + dose / beta, -alpha)

        # Identify susceptible agents and apply infection probability
        status_key = AgentPropertyKeys.status(self.name)
        susceptible_mask = agent_graph.ndata[status_key] == Compartment.SUSCEPTIBLE

        rand_vals = torch.rand(agent_graph.num_nodes(), device=self.device)
        new_infections = (rand_vals < prob_infection) & susceptible_mask

        if torch.any(new_infections):
            agent_graph.ndata[status_key][new_infections] = Compartment.EXPOSED
            agent_graph.ndata[AgentPropertyKeys.exposure_time(self.name)][new_infections] = 0
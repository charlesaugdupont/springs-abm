# abm/pathogens/pathogen.py
from abc import ABC, abstractmethod
from typing import Any
import torch

from abm.state import AgentGraph
from config import PathogenConfig, SteeringParamsSVEIR

class Pathogen(ABC):
    """Abstract Base Class for a pathogen in the SVEIR model."""

    def __init__(self, config: PathogenConfig, global_params: SteeringParamsSVEIR, device: torch.device):
        self.config = config
        self.name = config.name
        self.global_params = global_params
        self.device = device
        self.new_cases_this_step = 0

    def reset_incidence(self):
        """Resets the new case counter at the start of a step."""
        self.new_cases_this_step = 0

    @abstractmethod
    def update(self, agent_graph: AgentGraph, adjacency: torch.Tensor, grid: Any):
        """
        Executes all disease-specific updates for a single timestep.
        This includes disease progression, transmission, etc.
        """
        pass

    def _increment_exposure_time(self, agent_graph: AgentGraph):
        """Increments the exposure timer for agents in the Exposed state."""
        from abm.constants import Compartment, AgentPropertyKeys
        status_key = AgentPropertyKeys.status(self.name)
        timer_key = AgentPropertyKeys.exposure_time(self.name)
        exposed_mask = agent_graph.ndata[status_key] == Compartment.EXPOSED
        agent_graph.ndata[timer_key][exposed_mask] += 1

    def _exposed_to_infectious(self, agent_graph: AgentGraph):
        """Transitions agents from Exposed to Infectious after the exposure period."""
        from abm.constants import Compartment, AgentPropertyKeys
        status_key = AgentPropertyKeys.status(self.name)
        timer_key = AgentPropertyKeys.exposure_time(self.name)
        count_key = AgentPropertyKeys.num_infections(self.name)

        mask = (agent_graph.ndata[status_key] == Compartment.EXPOSED) & \
               (agent_graph.ndata[timer_key] >= self.config.exposure_period)

        if torch.any(mask):
            agent_graph.ndata[status_key][mask] = Compartment.INFECTIOUS
            agent_graph.ndata[count_key][mask] += 1
            self.new_cases_this_step += torch.sum(mask).item()

    def _infectious_to_recovered(self, agent_graph: AgentGraph):
        """Transitions agents from Infectious to Recovered based on a probability."""
        from abm.constants import Compartment, AgentPropertyKeys
        status_key = AgentPropertyKeys.status(self.name)
        infectious_mask = agent_graph.ndata[status_key] == Compartment.INFECTIOUS
        if not torch.any(infectious_mask):
            return

        recovery_chance = torch.rand(torch.sum(infectious_mask), device=self.device)
        recovered_mask = recovery_chance < self.config.recovery_rate

        agents_to_recover = infectious_mask.nonzero(as_tuple=True)[0][recovered_mask]
        agent_graph.ndata[status_key][agents_to_recover] = Compartment.RECOVERED

    def _apply_new_infections(
        self, agent_graph: AgentGraph, target_nodes_mask: torch.Tensor,
        base_prob: float, adjacency: torch.Tensor, prob_multiplier: float = 1.0
    ):
        """Helper function to calculate and apply new infections from any source."""
        from abm.constants import Compartment, AgentPropertyKeys
        if not torch.any(target_nodes_mask):
            return

        status_key = AgentPropertyKeys.status(self.name)
        count_key = AgentPropertyKeys.num_infections(self.name)
        target_indices = target_nodes_mask.nonzero(as_tuple=True)[0]

        is_infectious_mask = (agent_graph.ndata[status_key] == Compartment.INFECTIOUS).float()
        infection_pressure = torch.matmul(adjacency[target_indices].float(), is_infectious_mask)

        if torch.sum(infection_pressure) == 0:
            return

        num_prior_infections = agent_graph.ndata[count_key][target_indices].float()
        health = agent_graph.ndata[AgentPropertyKeys.HEALTH][target_indices].float()

        immunity_factor = torch.exp(-self.global_params.prior_infection_immunity_factor * num_prior_infections)
        health_factor = torch.exp(-self.global_params.infection_reduction_factor_per_health_unit * health)

        final_prob = prob_multiplier * base_prob * immunity_factor * health_factor
        final_prob.clamp_(0.0, 1.0)

        prob_not_infected = (1 - final_prob) ** infection_pressure
        prob_getting_infected = 1 - prob_not_infected

        random_samples = torch.rand(len(target_indices), device=self.device)
        newly_infected_mask = random_samples < prob_getting_infected
        infected_nodes_indices = target_indices[newly_infected_mask]

        if len(infected_nodes_indices) > 0:
            agent_graph.ndata[status_key][infected_nodes_indices] = Compartment.EXPOSED
            agent_graph.ndata[AgentPropertyKeys.exposure_time(self.name)][infected_nodes_indices] = 0
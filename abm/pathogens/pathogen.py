# abm/pathogens/pathogen.py
from abc import ABC, abstractmethod
from typing import Any, Optional
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
        self.new_cases_this_step = 0

    @abstractmethod
    def update(self, agent_graph: AgentGraph, location_ids: torch.Tensor, num_locations: int, grid: Any):
        """
        Executes updates. 
        changed: Accepts location_ids (N,) and num_locations (int) instead of adjacency matrix.
        """
        pass

    def _increment_exposure_time(self, agent_graph: AgentGraph):
        from abm.constants import Compartment, AgentPropertyKeys
        status_key = AgentPropertyKeys.status(self.name)
        timer_key = AgentPropertyKeys.exposure_time(self.name)
        exposed_mask = agent_graph.ndata[status_key] == Compartment.EXPOSED
        agent_graph.ndata[timer_key][exposed_mask] += 1

    def _exposed_to_infectious(self, agent_graph: AgentGraph):
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
        self, 
        agent_graph: AgentGraph, 
        target_nodes_mask: torch.Tensor,
        base_prob: float, 
        location_ids: Optional[torch.Tensor] = None,
        num_locations: Optional[int] = None,
        forced_pressure: Optional[torch.Tensor] = None,
        prob_multiplier: float = 1.0
    ):
        """
        Calculates infection probability using vectorized scatter/gather operations.
        
        Args:
            location_ids: (N,) tensor of location indices for each agent.
            num_locations: Total number of unique locations.
            forced_pressure: Optional (N,) tensor to manually set infection pressure (e.g. for Water).
        """
        from abm.constants import Compartment, AgentPropertyKeys      
        if not torch.any(target_nodes_mask):
            return

        status_key = AgentPropertyKeys.status(self.name)
        count_key = AgentPropertyKeys.num_infections(self.name)
        target_indices = target_nodes_mask.nonzero(as_tuple=True)[0]

        # --- Calculate Infection Pressure ---
        if forced_pressure is not None:
            # Case: Environmental/Water sources where pressure is pre-calculated
            infection_pressure = forced_pressure[target_indices]
        elif location_ids is not None and num_locations is not None:
            # Case: H2H Transmission via co-location (Optimized O(N))
            is_infectious = (agent_graph.ndata[status_key] == Compartment.INFECTIOUS).float()
            
            # 1. Sum infectious agents per location
            infected_per_location = torch.zeros(num_locations, device=self.device)
            infected_per_location.index_add_(0, location_ids, is_infectious)
            
            # 2. Map back to agents
            pressure_per_agent = infected_per_location[location_ids]
            
            # 3. Remove self-loop (if agent is infectious, they don't infect themselves)
            pressure_per_agent = pressure_per_agent - is_infectious
            
            infection_pressure = pressure_per_agent[target_indices]
        else:
            raise ValueError("Must provide either (location_ids, num_locations) or forced_pressure.")

        if torch.sum(infection_pressure) == 0:
            return

        # --- Calculate Individual Susceptibility ---
        num_prior_infections = agent_graph.ndata[count_key][target_indices].float()
        health = agent_graph.ndata[AgentPropertyKeys.HEALTH][target_indices].float()

        immunity_factor = torch.exp(-self.global_params.prior_infection_immunity_factor * num_prior_infections)
        health_factor = torch.exp(-self.global_params.infection_reduction_factor_per_health_unit * health)

        final_prob = prob_multiplier * base_prob * immunity_factor * health_factor
        final_prob.clamp_(0.0, 1.0)

        # Probability of NOT getting infected by ANY of the infectious contacts
        # P(infection) = 1 - (1 - p)^k  where k is infection_pressure
        prob_not_infected = (1 - final_prob) ** infection_pressure
        prob_getting_infected = 1 - prob_not_infected

        # --- Apply Infection ---
        random_samples = torch.rand(len(target_indices), device=self.device)
        newly_infected_mask = random_samples < prob_getting_infected
        infected_nodes_indices = target_indices[newly_infected_mask]

        if len(infected_nodes_indices) > 0:
            agent_graph.ndata[status_key][infected_nodes_indices] = Compartment.EXPOSED
            agent_graph.ndata[AgentPropertyKeys.exposure_time(self.name)][infected_nodes_indices] = 0
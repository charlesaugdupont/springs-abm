# abm/pathogens/rotavirus.py
from typing import Any
import torch

from .pathogen import Pathogen
from abm.state import AgentGraph
from abm.constants import Compartment, AgentPropertyKeys, Activity, GridLayer, WaterStatus
from config import RotavirusConfig, SteeringParamsSVEIR

class Rotavirus(Pathogen):
    """Implements the logic for Rotavirus, a viral, H2H-driven pathogen."""

    def __init__(self, config: RotavirusConfig, global_params: SteeringParamsSVEIR, device: torch.device):
        super().__init__(config, global_params, device)
        self.config: RotavirusConfig = config
        self.current_infection_prob = 0.0

    def update(self, agent_graph: AgentGraph, location_ids: torch.Tensor, num_locations: int, grid: Any):
        """Runs the full update cycle for Rotavirus."""
        self.reset_incidence()

        # 1. Update internal disease progression
        self._increment_exposure_time(agent_graph)
        self._exposed_to_infectious(agent_graph)
        self._infectious_to_recovered(agent_graph)

        # 2. Update population states (vaccination)
        self._susceptible_to_vaccinated(agent_graph)

        # 3. Handle transmission
        self._update_infection_probability()

        # 3a. H2H Transmission (Using optimized location groups)
        self._susceptible_to_exposed_h2h(agent_graph, location_ids, num_locations)
        self._vaccinated_to_exposed_h2h(agent_graph, location_ids, num_locations)

        # 3b. Waterborne Transmission
        self._human_to_water(agent_graph, grid)
        self._water_to_human(agent_graph, grid)


    def _update_infection_probability(self):
        self.current_infection_prob = max(
            0.001,
            torch.normal(mean=self.config.infection_prob_mean, std=self.config.infection_prob_std, size=(1,)).item()
        )

    def _susceptible_to_vaccinated(self, agent_graph: AgentGraph):
        status_key = AgentPropertyKeys.status(self.name)
        sus_mask = agent_graph.ndata[status_key] == Compartment.SUSCEPTIBLE
        if not torch.any(sus_mask):
            return

        chance = torch.rand(torch.sum(sus_mask), device=self.device)
        vaccinated_mask = chance < self.config.vaccination_rate
        agents_to_vaccinate = sus_mask.nonzero(as_tuple=True)[0][vaccinated_mask]
        agent_graph.ndata[status_key][agents_to_vaccinate] = Compartment.VACCINATED

    def _susceptible_to_exposed_h2h(self, agent_graph: AgentGraph, location_ids: torch.Tensor, num_locations: int):
        status_key = AgentPropertyKeys.status(self.name)
        sus_mask = agent_graph.ndata[status_key] == Compartment.SUSCEPTIBLE
        self._apply_new_infections(
            agent_graph, 
            target_nodes_mask=sus_mask, 
            base_prob=self.current_infection_prob, 
            location_ids=location_ids,
            num_locations=num_locations,
            prob_multiplier=1.0
        )

    def _vaccinated_to_exposed_h2h(self, agent_graph: AgentGraph, location_ids: torch.Tensor, num_locations: int):
        status_key = AgentPropertyKeys.status(self.name)
        vac_mask = agent_graph.ndata[status_key] == Compartment.VACCINATED
        breakthrough_multiplier = 1.0 - self.config.vaccine_efficacy
        self._apply_new_infections(
            agent_graph, 
            target_nodes_mask=vac_mask, 
            base_prob=self.current_infection_prob, 
            location_ids=location_ids,
            num_locations=num_locations,
            prob_multiplier=breakthrough_multiplier
        )

    def _human_to_water(self, agent_graph: AgentGraph, grid: Any):
        water_idx = grid.property_to_index.get(GridLayer.WATER)
        if water_idx is None: return

        status_key = AgentPropertyKeys.status(self.name)
        infectious_mask = agent_graph.ndata[status_key] == Compartment.INFECTIOUS
        at_water_mask = agent_graph.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] == Activity.WATER
        contaminator_mask = infectious_mask & at_water_mask

        if not torch.any(contaminator_mask):
            return

        prob = self.global_params.human_to_water_infection_prob
        chance = torch.rand(torch.sum(contaminator_mask), device=self.device)
        success = chance < prob
        successful_contaminators = contaminator_mask.nonzero(as_tuple=True)[0][success]

        if len(successful_contaminators) > 0:
            water_points = agent_graph.ndata[AgentPropertyKeys.WATER_LOCATION][successful_contaminators].long()
            unique_points = torch.unique(water_points, dim=0)
            if unique_points.shape[0] > 0:
                grid.grid_tensor[unique_points[:, 0], unique_points[:, 1], water_idx] = WaterStatus.CONTAMINATED

    def _water_to_human(self, agent_graph: AgentGraph, grid: Any):
        """Susceptible agents get infected from contaminated water sources."""
        water_idx = grid.property_to_index.get(GridLayer.WATER)
        if water_idx is None: return

        water_slice = grid.grid_tensor[:, :, water_idx]
        if torch.all(water_slice != WaterStatus.CONTAMINATED): return

        agent_coords = torch.stack(
            (agent_graph.ndata[AgentPropertyKeys.Y], agent_graph.ndata[AgentPropertyKeys.X]), dim=1
        ).long()
        agent_water_status = water_slice[agent_coords[:, 0], agent_coords[:, 1]]
        at_contaminated_water_mask = (agent_water_status == WaterStatus.CONTAMINATED)

        if not torch.any(at_contaminated_water_mask):
            return

        status_key = AgentPropertyKeys.status(self.name)
        target_mask = (agent_graph.ndata[status_key] != Compartment.INFECTIOUS) & \
                      (agent_graph.ndata[status_key] != Compartment.EXPOSED) & \
                      at_contaminated_water_mask

        # FIX: Explicitly pass pressure=1 for people at contaminated water
        # The base logic calculates: (1 - p) ** pressure.
        # If pressure is 0 (because we used identity matrix on Susceptibles), prob is 0.
        # Here we force pressure to 1.0 for everyone standing at contaminated water.
        pressure = torch.ones(agent_graph.num_nodes(), device=self.device)
        
        self._apply_new_infections(
            agent_graph, 
            target_nodes_mask=target_mask, 
            base_prob=self.global_params.water_to_human_infection_prob, 
            forced_pressure=pressure
        )
# abm/pathogens/rotavirus.py
from typing import Any
import torch

from .pathogen import Pathogen
from abm.state import AgentState
from abm.constants import Compartment, AgentPropertyKeys, GridLayer, WaterStatus
from config import RotavirusConfig, SteeringParamsSVEIR

class Rotavirus(Pathogen):
    """Implements the logic for Rotavirus, a viral, H2H-driven pathogen."""

    def __init__(self, config: RotavirusConfig, global_params: SteeringParamsSVEIR, device: torch.device):
        super().__init__(config, global_params, device)
        self.config: RotavirusConfig = config
        self.current_infection_prob = 0.0

    def step_progression(self, agent_state: AgentState):
        """Internal state progression (Once per day)."""
        # 1. Update internal disease progression
        self._increment_exposure_time(agent_state)
        self._exposed_to_infectious(agent_state)
        self._infectious_to_recovered(agent_state)

        # 2. Update population states (vaccination)
        self._susceptible_to_vaccinated(agent_state)

        # 3. Update stochastic global infection probability for the day
        self._update_infection_probability()

    def step_transmission(self, agent_state: AgentState, location_ids: torch.Tensor, num_locations: int, grid: Any):
        """Transmission logic (Run for Day and Night phases)."""
        
        # 1. H2H Transmission
        self._susceptible_to_exposed_h2h(agent_state, location_ids, num_locations)
        self._vaccinated_to_exposed_h2h(agent_state, location_ids, num_locations)

        # 2. Waterborne Transmission
        # Water transmission only happens if agents are actually AT the water source.
        # In Night phase (Reset to Home), no one is at water, so this naturally results in 0 infections, which is correct.
        self._human_to_water(agent_state, grid)
        self._water_to_human(agent_state, grid)

    def _update_infection_probability(self):
        self.current_infection_prob = max(
            0.001,
            torch.normal(mean=self.config.infection_prob_mean, std=self.config.infection_prob_std, size=(1,)).item()
        )

    def _susceptible_to_vaccinated(self, agent_state: AgentState):
        status_key = AgentPropertyKeys.status(self.name)
        sus_mask = agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE
        if not torch.any(sus_mask):
            return

        chance = torch.rand(torch.sum(sus_mask), device=self.device)
        vaccinated_mask = chance < self.config.vaccination_rate
        agents_to_vaccinate = sus_mask.nonzero(as_tuple=True)[0][vaccinated_mask]
        agent_state.ndata[status_key][agents_to_vaccinate] = Compartment.VACCINATED

    def _susceptible_to_exposed_h2h(self, agent_state: AgentState, location_ids: torch.Tensor, num_locations: int):
        status_key = AgentPropertyKeys.status(self.name)
        sus_mask = agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE
        self._apply_new_infections(
            agent_state, 
            target_nodes_mask=sus_mask, 
            base_prob=self.current_infection_prob, 
            location_ids=location_ids,
            num_locations=num_locations,
            prob_multiplier=1.0
        )

    def _vaccinated_to_exposed_h2h(self, agent_state: AgentState, location_ids: torch.Tensor, num_locations: int):
        status_key = AgentPropertyKeys.status(self.name)
        vac_mask = agent_state.ndata[status_key] == Compartment.VACCINATED
        breakthrough_multiplier = 1.0 - self.config.vaccine_efficacy
        self._apply_new_infections(
            agent_state, 
            target_nodes_mask=vac_mask, 
            base_prob=self.current_infection_prob, 
            location_ids=location_ids,
            num_locations=num_locations,
            prob_multiplier=breakthrough_multiplier
        )

    def _human_to_water(self, agent_state: AgentState, grid: Any):
        if grid is None:
            return

        water_idx = grid.property_to_index.get(GridLayer.WATER)
        if water_idx is None: return

        status_key = AgentPropertyKeys.status(self.name)
        infectious_mask = agent_state.ndata[status_key] == Compartment.INFECTIOUS
        
        current_x = agent_state.ndata[AgentPropertyKeys.X].long()
        current_y = agent_state.ndata[AgentPropertyKeys.Y].long()
        
        # Get water status at current agent locations
        # If agent is at home, water_layer value will be 0 (unless home is on water source, which shouldn't happen)
        water_layer = grid.grid_tensor[:, :, water_idx]
        is_at_water_loc = water_layer[current_y, current_x] > 0
        
        contaminator_mask = infectious_mask & is_at_water_loc

        if not torch.any(contaminator_mask):
            return

        prob = self.global_params.human_to_water_infection_prob
        chance = torch.rand(torch.sum(contaminator_mask), device=self.device)
        success = chance < prob
        successful_contaminators = contaminator_mask.nonzero(as_tuple=True)[0][success]

        if len(successful_contaminators) > 0:
            # We contaminate the water source at their current location
            locs = torch.stack((current_y[successful_contaminators], current_x[successful_contaminators]), dim=1)
            unique_locs = torch.unique(locs, dim=0)
            if unique_locs.shape[0] > 0:
                grid.grid_tensor[unique_locs[:, 0], unique_locs[:, 1], water_idx] = WaterStatus.CONTAMINATED

    def _water_to_human(self, agent_state: AgentState, grid: Any):
        """Susceptible agents get infected from contaminated water sources."""
        if grid is None: 
            return

        water_idx = grid.property_to_index.get(GridLayer.WATER)
        if water_idx is None: return

        water_slice = grid.grid_tensor[:, :, water_idx]
        if torch.all(water_slice != WaterStatus.CONTAMINATED): return

        current_x = agent_state.ndata[AgentPropertyKeys.X].long()
        current_y = agent_state.ndata[AgentPropertyKeys.Y].long()
        
        agent_water_status = water_slice[current_y, current_x]
        at_contaminated_water_mask = (agent_water_status == WaterStatus.CONTAMINATED)

        if not torch.any(at_contaminated_water_mask):
            return

        status_key = AgentPropertyKeys.status(self.name)
        target_mask = (agent_state.ndata[status_key] != Compartment.INFECTIOUS) & \
                      (agent_state.ndata[status_key] != Compartment.EXPOSED) & \
                      at_contaminated_water_mask

        pressure = torch.ones(agent_state.num_nodes(), device=self.device)
        
        self._apply_new_infections(
            agent_state, 
            target_nodes_mask=target_mask, 
            base_prob=self.global_params.water_to_human_infection_prob, 
            forced_pressure=pressure
        )
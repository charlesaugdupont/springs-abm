# abm/systems/environment.py
from typing import Any
import torch

from .system import System
from abm.agent_graph import AgentGraph

class EnvironmentSystem(System):
    """Handles environmental dynamics like water contamination recovery and shocks."""

    def update(self, agent_graph: AgentGraph, **kwargs):
        """
        Updates the state of the environment.
        """
        grid = kwargs.get("grid")
        timestep = kwargs.get("timestep")
        if grid is None or timestep is None:
            raise ValueError("EnvironmentSystem requires 'grid' and 'timestep'.")

        self._water_recovery(grid)

        if (timestep + 1) % self.config.steering_parameters.shock_frequency == 0:
            self._water_shock(grid)

    def _water_recovery(self, grid: Any):
        """Models the natural recovery of contaminated water sources."""
        water_idx = grid.property_to_index.get('water')
        if water_idx is None: return

        water_slice = grid.grid_tensor[:, :, water_idx]
        infected_mask = (water_slice == 2)
        if not torch.any(infected_mask):
            return

        prob = self.config.steering_parameters.water_recovery_prob
        chance = torch.rand(torch.sum(infected_mask), device=self.device)
        success = chance < prob

        coords_to_recover = infected_mask.nonzero(as_tuple=True)
        recovered_coords = (coords_to_recover[0][success], coords_to_recover[1][success])

        if len(recovered_coords[0]) > 0:
            water_slice[recovered_coords] = 1 # 1 represents clean

    def _water_shock(self, grid: Any):
        """Introduces contamination to clean water sources, simulating a shock event."""
        water_idx = grid.property_to_index.get('water')
        if water_idx is None: return

        water_slice = grid.grid_tensor[:, :, water_idx]
        clean_mask = (water_slice == 1)
        if not torch.any(clean_mask):
            return

        prob = self.config.steering_parameters.shock_infection_prob
        chance = torch.rand(torch.sum(clean_mask), device=self.device)
        success = chance < prob

        coords_to_shock = clean_mask.nonzero(as_tuple=True)
        shocked_coords = (coords_to_shock[0][success], coords_to_shock[1][success])

        if len(shocked_coords[0]) > 0:
            water_slice[shocked_coords] = 2 # 2 represents contaminated
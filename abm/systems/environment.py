# abm/systems/environment.py
from typing import Any
import torch

from .system import System
from abm.state import AgentState
from abm.constants import GridLayer, WaterStatus


class EnvironmentSystem(System):
    """Handles environmental dynamics like water contamination recovery and shocks."""

    def update(self, agent_state: AgentState, **kwargs):
        grid     = kwargs.get("grid")
        timestep = kwargs.get("timestep")
        if grid is None or timestep is None:
            return

        self._water_recovery(grid)

        if torch.rand(1).item() < self.config.steering_parameters.shock_daily_prob:
            self._water_shock(grid)

    def _water_recovery(self, grid: Any):
        """Models the natural recovery of contaminated water sources."""
        water_idx = grid.property_to_index.get(GridLayer.WATER)
        if water_idx is None:
            return

        water_slice   = grid.grid_tensor[:, :, water_idx]
        infected_mask = water_slice == WaterStatus.CONTAMINATED
        if not torch.any(infected_mask):
            return

        prob    = self.config.steering_parameters.water_recovery_prob
        chance  = torch.rand(torch.sum(infected_mask), device=self.device)
        success = chance < prob

        coords_to_recover  = infected_mask.nonzero(as_tuple=True)
        recovered_coords   = (coords_to_recover[0][success], coords_to_recover[1][success])

        if len(recovered_coords[0]) > 0:
            water_slice[recovered_coords] = WaterStatus.CLEAN

    def _water_shock(self, grid: Any):
        """
        Contaminates all currently clean water sources.

        When a shock event fires (governed by shock_daily_prob), every clean
        water source is contaminated.  This reflects a sudden environmental
        event (e.g. flooding, infrastructure failure) that affects the whole
        water supply simultaneously.  The probability of the shock occurring
        is the only stochastic element; the extent is always total.
        """
        water_idx = grid.property_to_index.get(GridLayer.WATER)
        if water_idx is None:
            return

        water_slice = grid.grid_tensor[:, :, water_idx]
        clean_mask  = water_slice == WaterStatus.CLEAN
        if not torch.any(clean_mask):
            return

        water_slice[clean_mask] = WaterStatus.CONTAMINATED

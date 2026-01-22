# abm/factories/environment_factory.py
import os
from pathlib import Path
import torch
import numpy as np

from abm.environment.grid_environment import GridEnvironment
from abm.state import AgentState
from abm.constants import AgentPropertyKeys
from config import GridCreationParams

class EnvironmentFactory:
    """A class to handle the creation of the spatial grid and agent placement."""

    def __init__(self, config: GridCreationParams):
        self.config = config
        self.grid_environment: GridEnvironment | None = None
        self.grid_tensor: np.ndarray | None = None
        self.grid_bounds: np.ndarray | None = None
        self.property_to_index: dict | None = None

    def create_grid(self):
        """Creates the grid environment based on the provided configuration."""
        if self.config.method == "realistic_import":
            if not self.config.grid_id:
                raise ValueError("A grid_id is required for the 'realistic_import' method.")

            grid_path = os.path.join("grids", self.config.grid_id, "grid.npz")
            if not Path(grid_path).exists():
                raise FileNotFoundError(f"Realistic grid file for ID '{self.config.grid_id}' not found at '{grid_path}'.")

            data = np.load(grid_path, allow_pickle=True)
            self.grid_tensor = data['grid']
            self.grid_bounds = data['bounds']
            self.property_to_index = {v: k for k, v in data['property_map'].item().items()}

            self.grid_environment = GridEnvironment(
                grid_tensor=torch.from_numpy(self.grid_tensor),
                property_index=self.property_to_index
            )
        else:
            raise Exception("Grid method should be 'realistic import'.")

    def place_agents(self, agent_state: AgentState):
        """Places agents on the grid, assigning their initial x and y coordinates."""
        if self.grid_environment is None or self.grid_tensor is None:
            raise RuntimeError("Grid has not been created. Call create_grid() first.")

        agent_state.ndata[AgentPropertyKeys.X] = torch.zeros(agent_state.num_nodes(), dtype=torch.float)
        agent_state.ndata[AgentPropertyKeys.Y] = torch.zeros(agent_state.num_nodes(), dtype=torch.float)

        household_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        unique_households, household_indices = torch.unique(household_ids, return_inverse=True)
        num_households = len(unique_households)

        if self.config.method == "realistic_import":
            residence_idx = self.property_to_index.get('residences')
            if residence_idx is None:
                raise ValueError("Grid for 'realistic_import' must contain a 'residences' layer.")

            residence_mask = self.grid_tensor[:, :, residence_idx]
            valid_cells = np.argwhere(residence_mask == 1)
            if len(valid_cells) == 0:
                raise ValueError("No valid residence cells found in the grid.")

            # Assign each household to a random valid residence cell
            assigned_cell_indices = np.random.choice(len(valid_cells), num_households)
            household_coords = valid_cells[assigned_cell_indices] # Shape: (num_households, 2) [r, c]

            # Assign the cell coordinates to all agents based on their household
            # household_coords is indexed by the mapped unique household ID
            agent_coords = torch.from_numpy(household_coords[household_indices])

            # Grid is (row, col), so coords are (y, x)
            agent_state.ndata[AgentPropertyKeys.Y] = agent_coords[:, 0].float()
            agent_state.ndata[AgentPropertyKeys.X] = agent_coords[:, 1].float()
        else:
            # Fallback to random placement for non-realistic grids
            shape = self.grid_environment.grid_shape
            hh_x = torch.randint(0, shape[0], (num_households,)).float()
            hh_y = torch.randint(0, shape[1], (num_households,)).float()
            agent_state.ndata[AgentPropertyKeys.X] = hh_x[household_indices]
            agent_state.ndata[AgentPropertyKeys.Y] = hh_y[household_indices]
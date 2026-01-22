# abm/systems/movement.py
import torch

from .system import System
from abm.state import AgentState
from abm.constants import Activity, AgentPropertyKeys

class MovementSystem(System):
    """
    Handles agent movement based on their activity choices and the Day/Night cycle.
    """
    def update(self, agent_state: AgentState, **kwargs):
        """
        Updates agent x, y coordinates based on their daily activity choice (Day Phase).
        """
        # 1. Agents choose an activity for the day
        time_use_probs = agent_state.ndata[AgentPropertyKeys.TIME_USE]
        # Add a small epsilon to prevent error with zero-probability rows
        activity_choice = torch.multinomial(time_use_probs + 1e-9, num_samples=1).squeeze()
        agent_state.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] = activity_choice

        # 2. Move agents to their chosen location
        location_map = {
            Activity.HOME: AgentPropertyKeys.HOME_LOCATION,
            Activity.SCHOOL: AgentPropertyKeys.SCHOOL_LOCATION,
            Activity.WORSHIP: AgentPropertyKeys.WORSHIP_LOCATION,
            Activity.WATER: AgentPropertyKeys.WATER_LOCATION
        }

        # Handle fixed locations
        for activity_idx, location_key in location_map.items():
            mask = (activity_choice == activity_idx)
            if torch.any(mask):
                # Locations are stored as (y, x)
                locations = agent_state.ndata[location_key][mask]
                agent_state.ndata[AgentPropertyKeys.Y][mask] = locations[:, 0]
                agent_state.ndata[AgentPropertyKeys.X][mask] = locations[:, 1]

        # Handle dynamic social locations (Purely Spatial)
        social_mask = (activity_choice == Activity.SOCIAL)
        if torch.any(social_mask):
            self._move_social_agents_spatial(agent_state, social_mask)

    def reset_to_home(self, agent_state: AgentState):
        """
        Forces all agents to return to their home location (Night Phase).
        """
        home_locations = agent_state.ndata[AgentPropertyKeys.HOME_LOCATION]
        agent_state.ndata[AgentPropertyKeys.Y] = home_locations[:, 0]
        agent_state.ndata[AgentPropertyKeys.X] = home_locations[:, 1]

    def _move_social_agents_spatial(self, agent_state: AgentState, social_mask: torch.Tensor):
        """
        Moves agents engaging in social activity to a random neighbor within the interaction radius.
        """
        params = self.config.steering_parameters
        visiting_indices = social_mask.nonzero(as_tuple=True)[0]
        
        # Get coordinates of visitors
        visitor_coords = torch.stack([
            agent_state.ndata[AgentPropertyKeys.Y][visiting_indices],
            agent_state.ndata[AgentPropertyKeys.X][visiting_indices]
        ], dim=1)

        # Potential hosts are agents currently at home.
        # Note: At the start of update(), agents moved to activity.
        # So we look for agents who chose Activity.HOME
        is_at_home = agent_state.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] == Activity.HOME
        host_indices = is_at_home.nonzero(as_tuple=True)[0]
        
        if len(host_indices) == 0:
            return # No one is home to visit

        host_coords = torch.stack([
            agent_state.ndata[AgentPropertyKeys.Y][host_indices],
            agent_state.ndata[AgentPropertyKeys.X][host_indices]
        ], dim=1)

        # Calculate distances (M visitors x N hosts)
        dists = torch.cdist(visitor_coords, host_coords)

        # Create a boolean mask of valid hosts within radius
        valid_hosts_mask = (dists <= params.social_interaction_radius)
        
        # For each visitor, are there any valid hosts?
        has_neighbors = valid_hosts_mask.any(dim=1)
        
        if not torch.any(has_neighbors):
            return

        # Filter down to visitors who actually found a neighbor
        valid_visitors_indices = visiting_indices[has_neighbors]
        valid_mask_subset = valid_hosts_mask[has_neighbors].float()

        # Sample one host per visitor from the valid ones (Uniform probability)
        # Note: multinomial expects probabilities, so we pass the binary float mask
        chosen_host_local_idx = torch.multinomial(valid_mask_subset, num_samples=1).squeeze()
        chosen_host_global_idx = host_indices[chosen_host_local_idx]

        # Move visitor to host's location
        target_locs = agent_state.ndata[AgentPropertyKeys.HOME_LOCATION][chosen_host_global_idx]
        agent_state.ndata[AgentPropertyKeys.Y][valid_visitors_indices] = target_locs[:, 0]
        agent_state.ndata[AgentPropertyKeys.X][valid_visitors_indices] = target_locs[:, 1]
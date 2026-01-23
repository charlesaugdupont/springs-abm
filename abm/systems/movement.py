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
        Optimized: Moves agents to a random neighbor using Rejection Sampling.
        Memory Complexity: O(N) instead of O(N^2).
        """
        params = self.config.steering_parameters
        visiting_indices = social_mask.nonzero(as_tuple=True)[0]
        num_visitors = len(visiting_indices)

        if num_visitors == 0: 
            return

        # 1. Identify all potential hosts (People currently at home)
        is_at_home = agent_state.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] == Activity.HOME
        host_indices = is_at_home.nonzero(as_tuple=True)[0]
        
        if len(host_indices) == 0:
            return 

        # 2. Randomly assign 1 candidate host to every visitor
        # We select indices from the host_indices array
        random_selections = torch.randint(0, len(host_indices), (num_visitors,), device=self.device)
        candidate_hosts = host_indices[random_selections]

        # 3. Check Distances (Vectorized, but only 1-to-1 comparison, not N-to-N)
        visitor_y = agent_state.ndata[AgentPropertyKeys.Y][visiting_indices]
        visitor_x = agent_state.ndata[AgentPropertyKeys.X][visiting_indices]
        
        host_y = agent_state.ndata[AgentPropertyKeys.Y][candidate_hosts]
        host_x = agent_state.ndata[AgentPropertyKeys.X][candidate_hosts]

        # Euclidean distance squared is faster (avoid sqrt)
        dist_sq = (visitor_y - host_y)**2 + (visitor_x - host_x)**2
        radius_sq = params.social_interaction_radius**2

        # 4. Filter: Who is actually within range?
        success_mask = dist_sq <= radius_sq
        
        if not torch.any(success_mask):
            return

        # 5. Move the successful visitors
        successful_visitors = visiting_indices[success_mask]
        accepted_hosts = candidate_hosts[success_mask]
        
        target_locs = agent_state.ndata[AgentPropertyKeys.HOME_LOCATION][accepted_hosts]
        
        agent_state.ndata[AgentPropertyKeys.Y][successful_visitors] = target_locs[:, 0]
        agent_state.ndata[AgentPropertyKeys.X][successful_visitors] = target_locs[:, 1]
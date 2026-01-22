# abm/systems/movement.py
import torch

from .system import System
from abm.state import AgentGraph
from abm.constants import Activity, AgentPropertyKeys

class MovementSystem(System):
    """Handles agent movement based on their activity choices."""

    def update(self, agent_graph: AgentGraph, **kwargs):
        """
        Updates agent x, y coordinates based on their daily activity choice.
        (Day Phase Movement)
        """
        edge_weights = kwargs.get("edge_weights")
        # If we haven't removed the network yet, we keep this check for now.
        if edge_weights is None:
             # Fallback if weights aren't provided, though step.py usually provides them
             pass

        # 1. Agents choose an activity for the day
        time_use_probs = agent_graph.ndata[AgentPropertyKeys.TIME_USE]
        # Add a small epsilon to prevent error with zero-probability rows
        activity_choice = torch.multinomial(time_use_probs + 1e-9, num_samples=1).squeeze()
        agent_graph.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] = activity_choice

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
                locations = agent_graph.ndata[location_key][mask]
                agent_graph.ndata[AgentPropertyKeys.Y][mask] = locations[:, 0]
                agent_graph.ndata[AgentPropertyKeys.X][mask] = locations[:, 1]

        # Handle dynamic social locations
        social_mask = (activity_choice == Activity.SOCIAL)
        if torch.any(social_mask) and edge_weights is not None:
            self._move_social_agents(agent_graph, social_mask, activity_choice, edge_weights)

    def reset_to_home(self, agent_graph: AgentGraph):
        """
        Forces all agents to their home coordinates.
        (Night Phase Movement)
        """
        # Locations are stored as (y, x) in HOME_LOCATION
        home_locs = agent_graph.ndata[AgentPropertyKeys.HOME_LOCATION]
        agent_graph.ndata[AgentPropertyKeys.Y] = home_locs[:, 0]
        agent_graph.ndata[AgentPropertyKeys.X] = home_locs[:, 1]

    def _move_social_agents(self, agent_graph: AgentGraph, social_mask: torch.Tensor,
                            activity_choice: torch.Tensor, edge_weights: torch.Tensor):
        """Helper to move agents engaging in social activity."""
        visiting_agents = social_mask.nonzero(as_tuple=True)[0]
        is_at_home_mask = (activity_choice == Activity.HOME)

        # Agents can only visit other agents who are at home
        social_weights = edge_weights[visiting_agents][:, is_at_home_mask]

        # Filter out visitors who have no one to visit
        can_visit_mask = social_weights.sum(dim=1) > 1e-9
        if not torch.any(can_visit_mask):
            return

        active_visitors = visiting_agents[can_visit_mask]
        active_weights = social_weights[can_visit_mask]

        hosts_at_home_indices = is_at_home_mask.nonzero(as_tuple=True)[0]
        if len(hosts_at_home_indices) == 0:
            return

        # Normalize weights to form a probability distribution
        active_weights /= active_weights.sum(dim=1, keepdim=True)

        chosen_local_idx = torch.multinomial(active_weights, num_samples=1).squeeze()
        visited_host_indices = hosts_at_home_indices[chosen_local_idx]

        # Move visitor to the home of the visited host
        host_locations = agent_graph.ndata[AgentPropertyKeys.HOME_LOCATION][visited_host_indices]
        agent_graph.ndata[AgentPropertyKeys.Y][active_visitors] = host_locations[:, 0]
        agent_graph.ndata[AgentPropertyKeys.X][active_visitors] = host_locations[:, 1]
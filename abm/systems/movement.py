# abm/systems/movement.py
import torch
from scipy.spatial import cKDTree
import numpy as np

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
        # .clone() is REQUIRED: without it, ndata[Y]/[X] become torch views into
        # HOME_LOCATION's columns, so the next Day Phase's in-place move writes
        # (update(): `ndata[Y][mask] = ...`) bleed back and corrupt HOME_LOCATION
        # - agents' stored homes drift to their activity locations over the run.
        agent_state.ndata[AgentPropertyKeys.Y] = home_locations[:, 0].clone()
        agent_state.ndata[AgentPropertyKeys.X] = home_locations[:, 1].clone()

    def _move_social_agents_spatial(self, agent_state: AgentState, social_mask: torch.Tensor):
        params = self.config.steering_parameters
        visiting_indices = social_mask.nonzero(as_tuple=True)[0]
        if len(visiting_indices) == 0:
            return

        is_at_home = agent_state.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] == Activity.HOME
        host_indices = is_at_home.nonzero(as_tuple=True)[0]
        if len(host_indices) == 0:
            return

        host_coords = torch.stack([
            agent_state.ndata[AgentPropertyKeys.Y][host_indices],
            agent_state.ndata[AgentPropertyKeys.X][host_indices],
        ], dim=1).cpu().numpy()

        visitor_coords = torch.stack([
            agent_state.ndata[AgentPropertyKeys.Y][visiting_indices],
            agent_state.ndata[AgentPropertyKeys.X][visiting_indices],
        ], dim=1).cpu().numpy()

        tree = cKDTree(host_coords)
        radius = params.social_interaction_radius
        neighbor_lists = tree.query_ball_point(visitor_coords, r=radius)

        # Pick a host per visitor, preserving the exact per-agent
        # np.random.randint call sequence (so the RNG stream is unchanged),
        # then apply all the moves in one batched scatter instead of a
        # per-agent tensor read/write. Visitors are distinct agents and are
        # never also hosts (SOCIAL vs HOME activity), so batching is
        # byte-for-byte identical to the per-element writes.
        chosen_visitor_slots = []
        chosen_hosts = []
        for local_i, neighbors in enumerate(neighbor_lists):
            if not neighbors:
                continue  # no one home within radius -> agent stays home
            chosen = neighbors[np.random.randint(len(neighbors))]
            chosen_visitor_slots.append(local_i)
            chosen_hosts.append(chosen)

        if not chosen_visitor_slots:
            return

        device = agent_state.device
        v_idx = visiting_indices[torch.tensor(chosen_visitor_slots, dtype=torch.long, device=device)]
        h_idx = host_indices[torch.tensor(chosen_hosts, dtype=torch.long, device=device)]
        target_locs = agent_state.ndata[AgentPropertyKeys.HOME_LOCATION][h_idx]
        agent_state.ndata[AgentPropertyKeys.Y][v_idx] = target_locs[:, 0]
        agent_state.ndata[AgentPropertyKeys.X][v_idx] = target_locs[:, 1]
# abm/systems/household.py
import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys, Activity, WaterStatus, Compartment, GridLayer

class HouseholdSystem(System):
    """
    Handles dynamics that occur within the household, such as sharing contaminated water.
    """
    def update(self, agent_state: AgentState, **kwargs):
        grid = kwargs.get("grid")
        if grid is None:
            return

        # This system models Rotavirus transmission via household water
        pathogen_name = "rota" 
        status_key = AgentPropertyKeys.status(pathogen_name)
        
        # --- 1. Identify Households with Newly Contaminated Water ---
        water_idx = grid.property_to_index.get(GridLayer.WATER)
        if water_idx is None: return

        # Find adults who are currently at a water source
        is_adult = ~agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        at_water = agent_state.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] == Activity.WATER
        water_collectors_mask = is_adult & at_water

        if not torch.any(water_collectors_mask):
            return

        water_slice = grid.grid_tensor[:, :, water_idx]
        collector_indices = water_collectors_mask.nonzero(as_tuple=True)[0]
        
        # Get the (y, x) coords of the water sources these adults are visiting
        coords = agent_state.ndata[AgentPropertyKeys.WATER_LOCATION][collector_indices].long()
        
        # Check the status of these specific water sources
        source_is_contaminated = water_slice[coords[:, 0], coords[:, 1]] == WaterStatus.CONTAMINATED

        if not torch.any(source_is_contaminated):
            return

        # Get the household IDs of only those collectors who visited a contaminated source
        successful_collectors = collector_indices[source_is_contaminated]
        contaminated_hh_ids = torch.unique(agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID][successful_collectors])

        # --- 2. Expose Susceptible Household Members ---
        if len(contaminated_hh_ids) == 0:
            return

        # Create a boolean mask for all agents who belong to one of the contaminated households
        # This is an efficient way to find all members of the target households
        is_in_contaminated_hh = torch.isin(
            agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID], contaminated_hh_ids
        )
        
        # Find agents who are susceptible and are in one of these households
        is_susceptible = agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE
        target_for_infection_mask = is_in_contaminated_hh & is_susceptible

        if not torch.any(target_for_infection_mask):
            return

        # Apply the infection probability using the existing water_to_human parameter
        rand_vals = torch.rand(torch.sum(target_for_infection_mask), device=self.device)
        newly_infected_mask = rand_vals < self.config.steering_parameters.water_to_human_infection_prob

        # Get the global indices of the agents who just became infected
        infected_agent_indices = target_for_infection_mask.nonzero(as_tuple=True)[0][newly_infected_mask]

        if len(infected_agent_indices) > 0:
            agent_state.ndata[status_key][infected_agent_indices] = Compartment.EXPOSED
            agent_state.ndata[AgentPropertyKeys.exposure_time(pathogen_name)][infected_agent_indices] = 0
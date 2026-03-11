# abm/systems/household.py
import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys, Activity, WaterStatus, Compartment, GridLayer
from abm.pathogens.rotavirus import Rotavirus

class HouseholdSystem(System):
    """
    Handles within-household dynamics, specifically the risk of Rotavirus
    transmission via a contaminated household water source.

    Mechanism:
      Each adult whose daily activity was WATER visited their assigned water
      source during the Day Phase.  By the time this system runs (after
      reset_to_home) all agents are physically back at home, but the
      ACTIVITY_CHOICE tensor still records what each agent did today.
      We use that record — together with each agent's fixed WATER_LOCATION —
      to determine which households fetched water from a contaminated source,
      then expose susceptible members of those households.
    """
    def update(self, agent_state: AgentState, **kwargs):
        grid = kwargs.get("grid")
        if grid is None:
            return
        pathogens = kwargs.get("pathogens", None)

        pathogen_name = "rota"
        status_key = AgentPropertyKeys.status(pathogen_name)

        water_idx = grid.property_to_index.get(GridLayer.WATER)
        if water_idx is None:
            return

        # --- 1. Find adults whose activity today was fetching water ---
        is_adult = ~agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        went_to_water = agent_state.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] == Activity.WATER
        water_collectors_mask = is_adult & went_to_water

        if not torch.any(water_collectors_mask):
            return

        water_slice = grid.grid_tensor[:, :, water_idx]
        collector_indices = water_collectors_mask.nonzero(as_tuple=True)[0]

        # Check the contamination status of assigned water sources
        water_locs = agent_state.ndata[AgentPropertyKeys.WATER_LOCATION][collector_indices].long()
        source_is_contaminated = (
            water_slice[water_locs[:, 0], water_locs[:, 1]] == WaterStatus.CONTAMINATED
        )

        if not torch.any(source_is_contaminated):
            return

        # --- 2. Identify households that brought back contaminated water ---
        successful_collectors = collector_indices[source_is_contaminated]
        contaminated_hh_ids = torch.unique(
            agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID][successful_collectors]
        )

        if len(contaminated_hh_ids) == 0:
            return

        # --- 3. Expose members of those households, with vaccine-modified risk ---
        status = agent_state.ndata[status_key]
        is_in_contaminated_hh = torch.isin(
            agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID], contaminated_hh_ids
        )

        # Allow re-infection of recovered agents, consistent with other water routes:
        susceptible_like = (status == Compartment.SUSCEPTIBLE) | (status == Compartment.RECOVERED)
        vaccinated = (status == Compartment.VACCINATED)

        susceptible_mask = is_in_contaminated_hh & susceptible_like
        vaccinated_mask = is_in_contaminated_hh & vaccinated

        if not torch.any(susceptible_mask) and not torch.any(vaccinated_mask):
            return

        # Find the Rotavirus pathogen instance
        rota = None
        if pathogens is not None:
            for p in pathogens:
                if isinstance(p, Rotavirus):
                    rota = p
                    break

        if rota is None:
            # Fallback: original simple Bernoulli for susceptible agents only
            base_prob = self.config.steering_parameters.water_to_human_infection_prob
            target_for_infection_mask = susceptible_mask  # vaccinated ignored here
            if not torch.any(target_for_infection_mask):
                return

            rand_vals = torch.rand(torch.sum(target_for_infection_mask), device=self.device)
            newly_infected_mask = rand_vals < base_prob

            infected_agent_indices = (
                target_for_infection_mask.nonzero(as_tuple=True)[0][newly_infected_mask]
            )

            num_new = len(infected_agent_indices)
            if num_new > 0:
                agent_state.ndata[status_key][infected_agent_indices] = Compartment.EXPOSED
                agent_state.ndata[AgentPropertyKeys.exposure_time(pathogen_name)][infected_agent_indices] = 0
                agent_state.ndata[AgentPropertyKeys.num_infections(pathogen_name)][infected_agent_indices] += 1
                return

        # Use the pathogen's own infection logic
        base_prob = self.config.steering_parameters.water_to_human_infection_prob
        pressure = torch.ones(agent_state.num_nodes(), device=self.device)

        # Unvaccinated (S + R): full base probability
        if torch.any(susceptible_mask):
            rota._apply_new_infections(
                agent_state,
                target_nodes_mask=susceptible_mask,
                base_prob=base_prob,
                forced_pressure=pressure,
            )

        # Vaccinated: reduced probability via breakthrough multiplier
        if torch.any(vaccinated_mask):
            breakthrough_multiplier = 1.0 - rota.config.vaccine_efficacy
            rota._apply_new_infections(
                agent_state,
                target_nodes_mask=vaccinated_mask,
                base_prob=base_prob,
                forced_pressure=pressure,
                prob_multiplier=breakthrough_multiplier,
            )
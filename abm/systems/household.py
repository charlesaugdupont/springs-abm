# abm/systems/household.py
import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys, Activity

class HouseholdSystem(System):
    """
    Handles within-household dynamics: specifically, drinking-water exposure.

    Catchment water model (WaterConfig.model == "catchment"): every member of
    every household drinks from the household's primary source each day; the
    exposure model is the pathogen's water_infection_prob applied to that
    source's concentration (abm/environment/water_network.py).

    Legacy water model (below): the risk carried home in fetched water.

    Mechanism:
      Each adult whose daily activity was WATER visited their assigned water
      source during the Day Phase.  By the time this system runs (after
      reset_to_home) all agents are physically back at home, but the
      ACTIVITY_CHOICE tensor still records what each agent did today.
      We use that record - together with each agent's fixed WATER_LOCATION -
      to determine which households fetched water, read the state of each
      pathogen's reservoir at those sources, and hand the result to the
      pathogen to apply its own risk model.

    This used to be rotavirus-specific (the pathogen name was hardcoded and
    the Rotavirus instance was located by isinstance). It now loops over every
    waterborne pathogen, so Cryptosporidium's carried-water exposure comes for
    free - and, because each pathogen owns its own reservoir state, the two can
    represent water contamination completely differently.

    Contract with pathogens: `water_carrying_exposure` receives ONE ROW PER
    HOUSEHOLD that fetched water today, together with the value of that
    pathogen's own water state at that household's source. Where a household
    sent several collectors to different sources, the worst (maximum) state
    value is used.
    """
    def update(self, agent_state: AgentState, **kwargs):
        grid = kwargs.get("grid")
        if grid is None:
            return
        pathogens = kwargs.get("pathogens") or []
        waterborne = [p for p in pathogens if getattr(p, "WATERBORNE", False)]
        if not waterborne:
            return

        # Catchment model: every household drinks from its own primary source
        # every day, whether or not anyone fetched water today (stored water).
        network = getattr(grid, "water_network", None)
        if network is not None:
            for pathogen in waterborne:
                if pathogen.name in network.concentration:
                    conc = network.agent_exposure_concentration(pathogen.name).to(agent_state.device)
                    pathogen.drinking_water_exposure(agent_state, conc)
            return

        # --- 1. Find adults whose activity today was fetching water ---
        is_adult = ~agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        went_to_water = agent_state.ndata[AgentPropertyKeys.ACTIVITY_CHOICE] == Activity.WATER
        water_collectors_mask = is_adult & went_to_water

        if not torch.any(water_collectors_mask):
            return

        collector_indices = water_collectors_mask.nonzero(as_tuple=True)[0]
        water_locs = agent_state.ndata[AgentPropertyKeys.WATER_LOCATION][collector_indices].long()
        collector_hh = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID][collector_indices]

        # One row per household, in sorted household order.
        unique_hh, inverse = torch.unique(collector_hh, return_inverse=True)

        # --- 2. Per pathogen: read its reservoir at those sources, then
        #        let the pathogen apply its own risk model ---
        for pathogen in waterborne:
            state = grid.get_water_state(pathogen.name)
            if state is None:
                continue

            source_values = state[water_locs[:, 0], water_locs[:, 1]].float()
            per_household = torch.zeros(
                unique_hh.numel(), dtype=source_values.dtype, device=source_values.device
            )
            # Worst-case source per household (see class docstring).
            per_household.scatter_reduce_(0, inverse, source_values, reduce="amax")

            pathogen.water_carrying_exposure(agent_state, grid, unique_hh, per_household)

    @staticmethod
    def apply_household_wealth_delta(
        agent_state: AgentState,
        member_idx: torch.Tensor,
        delta: float,
    ) -> None:
        """
        Applies `delta` to the pooled wealth of the household that `member_idx`
        belongs to, and propagates the resulting value to every member of that
        household.

        Wealth is modeled as a shared household resource (see EconomicSystem)
        rather than an individual balance, so any wealth-changing event
        experienced by one household member (e.g. paying for a child's care)
        draws on, and updates, the whole household's pooled wealth.
        """
        household_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        hh_id = household_ids[member_idx]
        same_household_mask = household_ids == hh_id

        current_wealth = agent_state.ndata[AgentPropertyKeys.WEALTH][member_idx].item()
        new_wealth = min(1.0, max(0.0, current_wealth + delta))
        agent_state.ndata[AgentPropertyKeys.WEALTH][same_household_mask] = new_wealth

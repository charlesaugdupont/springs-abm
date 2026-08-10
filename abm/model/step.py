# abm/model/step.py
from typing import Any, Dict, Tuple, List
import torch

from abm.state import AgentState
from config import SVEIRConfig
from abm.constants import Compartment, AgentPropertyKeys
from abm.pathogens.pathogen import Pathogen
from abm.systems.system import System

def _get_location_groups(agent_state: AgentState) -> Tuple[torch.Tensor, int]:
    """Returns group indices for agents based on co-location.

    Each (x, y) cell is encoded as a single integer key (y * width + x) and a
    1-D torch.unique is taken. This is much faster than a lexicographic 2-D
    torch.unique(dim=0) and yields the *same* co-location partition. The
    absolute group-label values differ from the 2-D version, but that is
    irrelevant: callers only use these as grouping keys for a scatter/gather
    (index_add_ in Pathogen._apply_new_infections), which is invariant to a
    relabeling. Grid coordinates are integer cell indices, so the .long() cast
    is exact and preserves the partition byte-for-byte.
    """
    x = agent_state.ndata[AgentPropertyKeys.X].long()
    y = agent_state.ndata[AgentPropertyKeys.Y].long()
    width = int(x.max().item()) + 1
    keys = y * width + x
    _, inverse_indices = torch.unique(keys, return_inverse=True)
    num_locations = int(inverse_indices.max().item()) + 1
    return inverse_indices, num_locations

def sveir_step(
    agent_state: AgentState,
    timestep: int,
    config: SVEIRConfig,
    grid: Any,
    pathogens: List[Pathogen],
    systems: List[System],
) -> Tuple[Dict[str, int], Dict[str, int]]:

    # Reset incidence for the new day
    for pathogen in pathogens:
        pathogen.reset_incidence()

    # --- 0. DISEASE PROGRESSION (Morning) ---
    for pathogen in pathogens:
        pathogen.step_progression(agent_state)

    # --- 1. PHASE 1: DAYTIME (Activity) ---

    # a. MOVEMENT (Go to School, Water, Social)
    systems[0].update(agent_state)

    # b. SPATIAL GROUPING (Daytime)
    location_ids, num_locations = _get_location_groups(agent_state)

    # c. TRANSMISSION (Daytime)
    for pathogen in pathogens:
        pathogen.step_transmission(agent_state, location_ids, num_locations, grid)

    # --- 2. PHASE 2: NIGHTTIME (Home) ---

    # a. MOVEMENT (Return to Home)
    systems[0].reset_to_home(agent_state)

    # b. SPATIAL GROUPING (Nighttime/Household)
    location_ids, num_locations = _get_location_groups(agent_state)

    # c. TRANSMISSION (Nighttime)
    for pathogen in pathogens:
        pathogen.step_transmission(agent_state, location_ids, num_locations, grid)

    # --- 3. DAILY SYSTEMS ---
    # Order must match self.systems in initialize_model.py
    systems[1].update(agent_state, timestep=timestep) # ChildIllnessSystem
    systems[2].update(agent_state) # CareSeekingSystem
    systems[3].update(agent_state, grid=grid, pathogens=pathogens) # HouseholdSystem
    systems[4].update(agent_state, grid=grid, timestep=timestep) # EnvironmentSystem
    systems[5].update(agent_state) # EconomicSystem

    # --- 4. GATHER STATISTICS ---
    new_cases_by_pathogen: Dict[str, int] = {}
    compartment_counts: Dict[str, int] = {}

    for p in pathogens:
        new_cases_by_pathogen[p.name] = p.new_cases_this_step

        status = agent_state.ndata[f"status_{p.name}"]
        # One bincount pass (one CPU sync) instead of five torch.sum(...).item()
        # scans. Compartment is an IntEnum, so its values index the result
        # directly: S=0, V=1, E=2, I=3, R=4 (abm/constants.py).
        counts = torch.bincount(status.long(), minlength=5).tolist()
        compartment_counts[f"{p.name}_S"] = counts[Compartment.SUSCEPTIBLE]
        compartment_counts[f"{p.name}_E"] = counts[Compartment.EXPOSED]
        compartment_counts[f"{p.name}_I"] = counts[Compartment.INFECTIOUS]
        compartment_counts[f"{p.name}_R"] = counts[Compartment.RECOVERED]
        compartment_counts[f"{p.name}_V"] = counts[Compartment.VACCINATED]

    return new_cases_by_pathogen, compartment_counts

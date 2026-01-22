# abm/model/step.py
from typing import Any, Dict, Tuple, List
import torch

from abm.state import AgentState
from config import SVEIRConfig
from abm.model.data_collection import data_collection
from abm.constants import Compartment, AgentPropertyKeys
from abm.pathogens.pathogen import Pathogen
from abm.systems.system import System

def _get_location_groups(agent_state: AgentState) -> Tuple[torch.Tensor, int]:
    """
    Returns group indices for agents based on co-location.
    """
    coords = torch.stack([agent_state.ndata[AgentPropertyKeys.X], agent_state.ndata[AgentPropertyKeys.Y]], dim=1)
    
    # Efficiently find unique locations and assign each agent a location index
    _, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
    num_locations = inverse_indices.max().item() + 1
    
    return inverse_indices, num_locations

def sveir_step(
    agent_state: AgentState,
    timestep: int,
    config: SVEIRConfig,
    grid: Any,
    pathogens: List[Pathogen],
    systems: List[System]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    
    params = config.steering_parameters

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
    systems[1].update(agent_state) # ChildIllnessSystem
    systems[2].update(agent_state) # CareSeekingSystem
    systems[3].update(agent_state, grid=grid) # HouseholdSystem
    systems[4].update(agent_state, grid=grid, timestep=timestep) # EnvironmentSystem
    systems[5].update(agent_state) # EconomicSystem

    # --- 4. DATA COLLECTION ---
    if (params.data_collection_period > 0 and (timestep % params.data_collection_period == 0)) or \
       (params.data_collection_list and (timestep in params.data_collection_list)):
        data_collection(agent_state, timestep=timestep + 1, npath=params.npath, ndata=params.ndata, mode=params.mode)

    # --- 5. GATHER STATISTICS ---
    new_cases_by_pathogen: Dict[str, int] = {}
    compartment_counts = {}
    
    for p in pathogens:
        new_cases_by_pathogen[p.name] = p.new_cases_this_step
        
        status = agent_state.ndata[f"status_{p.name}"]
        compartment_counts[f"{p.name}_S"] = torch.sum(status == Compartment.SUSCEPTIBLE).item()
        compartment_counts[f"{p.name}_E"] = torch.sum(status == Compartment.EXPOSED).item()
        compartment_counts[f"{p.name}_I"] = torch.sum(status == Compartment.INFECTIOUS).item()
        compartment_counts[f"{p.name}_R"] = torch.sum(status == Compartment.RECOVERED).item()
        compartment_counts[f"{p.name}_V"] = torch.sum(status == Compartment.VACCINATED).item()

    return new_cases_by_pathogen, compartment_counts
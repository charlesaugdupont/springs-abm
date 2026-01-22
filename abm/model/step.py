# abm/model/step.py
from typing import Any, Dict, Tuple, List
import torch

from abm.state import AgentGraph
from config import SVEIRConfig
from abm.model.data_collection import data_collection
from abm.constants import Compartment, AgentPropertyKeys
from abm.pathogens.pathogen import Pathogen
from abm.systems.system import System

def _get_location_groups(agent_graph: AgentGraph) -> Tuple[torch.Tensor, int]:
    """
    Returns group indices for agents based on co-location.
    """
    coords = torch.stack([agent_graph.ndata[AgentPropertyKeys.X], agent_graph.ndata[AgentPropertyKeys.Y]], dim=1)
    
    # Efficiently find unique locations and assign each agent a location index
    _, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
    num_locations = inverse_indices.max().item() + 1
    
    return inverse_indices, num_locations

def sveir_step(
    agent_graph: AgentGraph,
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
        pathogen.step_progression(agent_graph)

    # --- 1. PHASE 1: DAYTIME (Activity) ---
    
    # a. MOVEMENT (Go to School, Water, Social)
    systems[0].update(agent_graph)
    
    # b. SPATIAL GROUPING (Daytime)
    location_ids, num_locations = _get_location_groups(agent_graph)

    # c. TRANSMISSION (Daytime)
    for pathogen in pathogens:
        pathogen.step_transmission(agent_graph, location_ids, num_locations, grid)

    # --- 2. PHASE 2: NIGHTTIME (Home) ---

    # a. MOVEMENT (Return to Home)
    systems[0].reset_to_home(agent_graph)

    # b. SPATIAL GROUPING (Nighttime/Household)
    location_ids, num_locations = _get_location_groups(agent_graph)

    # c. TRANSMISSION (Nighttime)
    for pathogen in pathogens:
        pathogen.step_transmission(agent_graph, location_ids, num_locations, grid)

    # --- 3. DAILY SYSTEMS ---
    systems[1].update(agent_graph) # ChildIllnessSystem
    systems[2].update(agent_graph) # CareSeekingSystem
    systems[3].update(agent_graph, grid=grid, timestep=timestep) # EnvironmentSystem
    systems[4].update(agent_graph, grid=grid) # HouseholdSystem
    systems[5].update(agent_graph) # EconomicSystem

    # --- 4. DATA COLLECTION ---
    if (params.data_collection_period > 0 and (timestep % params.data_collection_period == 0)) or \
       (params.data_collection_list and (timestep in params.data_collection_list)):
        data_collection(
            agent_graph, timestep=timestep + 1, npath=params.npath,
            epath=params.epath, ndata=params.ndata, edata=params.edata, mode=params.mode
        )

    # --- 5. GATHER STATISTICS ---
    new_cases_by_pathogen: Dict[str, int] = {}
    compartment_counts = {}
    
    for p in pathogens:
        new_cases_by_pathogen[p.name] = p.new_cases_this_step
        
        status = agent_graph.ndata[f"status_{p.name}"]
        compartment_counts[f"{p.name}_S"] = torch.sum(status == Compartment.SUSCEPTIBLE).item()
        compartment_counts[f"{p.name}_E"] = torch.sum(status == Compartment.EXPOSED).item()
        compartment_counts[f"{p.name}_I"] = torch.sum(status == Compartment.INFECTIOUS).item()
        compartment_counts[f"{p.name}_R"] = torch.sum(status == Compartment.RECOVERED).item()
        compartment_counts[f"{p.name}_V"] = torch.sum(status == Compartment.VACCINATED).item()

    return new_cases_by_pathogen, compartment_counts
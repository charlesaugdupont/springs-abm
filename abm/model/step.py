# abm/model/step.py
from typing import Any, Dict, Tuple, List
import torch

from abm.state import AgentGraph
from config import SVEIRConfig
from abm.model.data_collection import data_collection
from abm.constants import Compartment, EdgePropertyKeys
from abm.pathogens.pathogen import Pathogen
from abm.systems.system import System

def _get_location_groups(agent_graph: AgentGraph) -> Tuple[torch.Tensor, int]:
    """
    Returns group indices for agents based on co-location.
    Returns:
        inverse_indices: Tensor of shape (num_agents) where value is the location ID.
        num_locations: Integer count of unique locations.
    """
    coords = torch.stack([agent_graph.ndata['x'], agent_graph.ndata['y']], dim=1)
    
    # Efficiently find unique locations and assign each agent a location index
    # This is O(N log N) or O(N) depending on implementation, much faster than O(N^2)
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

    # --- 1. MOVEMENT ---
    # We pass edge_weights if they exist, but we no longer recalc them spatially
    src, dst = agent_graph.edges()
    edge_weights = torch.zeros((agent_graph.num_nodes(), agent_graph.num_nodes()), device=config.device)
    
    # Ensure we handle the sparse edge list correctly for the MovementSystem
    # Note: If weights aren't set, MovementSystem might default to uniform.
    if EdgePropertyKeys.WEIGHT in agent_graph.edata:
        # This is still technically sparse-to-dense, but only for active edges (O(E))
        # For huge graphs, MovementSystem should also be refactored to avoid this,
        # but for now we focus on the Pathogen bottleneck.
        edge_weights[src, dst] = agent_graph.edata[EdgePropertyKeys.WEIGHT]

    systems[0].update(agent_graph, edge_weights=edge_weights) 
    
    # --- 2. SPATIAL GROUPING ---
    # Instead of an N*N matrix, we get vectors representing location groups
    location_ids, num_locations = _get_location_groups(agent_graph)

    # --- 3. PATHOGEN DYNAMICS ---
    new_cases_by_pathogen: Dict[str, int] = {}
    for pathogen in pathogens:
        # Pass the efficient location data instead of the adjacency matrix
        pathogen.update(agent_graph, location_ids, num_locations, grid)
        new_cases_by_pathogen[pathogen.name] = pathogen.new_cases_this_step

    # --- 4. ILLNESS & BEHAVIORAL UPDATES ---
    systems[1].update(agent_graph)
    systems[2].update(agent_graph)
    systems[3].update(agent_graph, grid=grid, timestep=timestep)

    # --- 5. DATA COLLECTION ---
    if (params.data_collection_period > 0 and (timestep % params.data_collection_period == 0)) or \
       (params.data_collection_list and (timestep in params.data_collection_list)):
        data_collection(
            agent_graph, timestep=timestep + 1, npath=params.npath,
            epath=params.epath, ndata=params.ndata, edata=params.edata, mode=params.mode
        )

    # --- 6. GATHER STATISTICS ---
    compartment_counts = {}
    for p in pathogens:
        status = agent_graph.ndata[f"status_{p.name}"]
        compartment_counts[f"{p.name}_S"] = torch.sum(status == Compartment.SUSCEPTIBLE).item()
        compartment_counts[f"{p.name}_E"] = torch.sum(status == Compartment.EXPOSED).item()
        compartment_counts[f"{p.name}_I"] = torch.sum(status == Compartment.INFECTIOUS).item()
        compartment_counts[f"{p.name}_R"] = torch.sum(status == Compartment.RECOVERED).item()
        compartment_counts[f"{p.name}_V"] = torch.sum(status == Compartment.VACCINATED).item()

    return new_cases_by_pathogen, compartment_counts
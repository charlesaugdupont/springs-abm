# abm/model/step.py
"""The main time-stepping logic for the SVEIR model."""
from typing import Any, Dict, Tuple, List
import torch

from abm.agent_graph import AgentGraph
from config import SVEIRConfig
from abm.model.data_collection import data_collection
from abm.constants import Compartment, EdgePropertyKeys
from abm.pathogens.pathogen import Pathogen
from abm.systems.system import System
from abm.pathogens.rotavirus import Rotavirus

def _calculate_adjacency(agent_graph: AgentGraph) -> torch.Tensor:
    """Calculates a dense adjacency matrix based on agent co-location."""
    num_nodes = agent_graph.num_nodes()
    coords = torch.stack([agent_graph.ndata['x'], agent_graph.ndata['y']], dim=1)
    
    # Efficiently find groups of agents at the same location
    unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
    
    # Create an adjacency matrix where adj[i, j] = 1 if agent i and j are at the same location
    # This is equivalent to checking if their inverse_indices are equal
    adjacency_matrix = (inverse_indices.unsqueeze(1) == inverse_indices.unsqueeze(0)).float()
    
    # Remove self-loops
    adjacency_matrix.fill_diagonal_(0)
    
    return adjacency_matrix

def sveir_step(
    agent_graph: AgentGraph,
    timestep: int,
    config: SVEIRConfig,
    grid: Any,
    policy_library: dict,
    risk_levels: torch.Tensor,
    pathogens: List[Pathogen],
    systems: List[System]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Executes a single step of the SVEIR model by invoking all systems and pathogens.
    """
    params = config.steering_parameters

    # --- 1. MOVEMENT & INTERACTION SETUP ---
    # The MovementSystem is assumed to be the first system.
    src, dst = agent_graph.edges()
    edge_weights = torch.zeros((agent_graph.num_nodes(), agent_graph.num_nodes()), device=config.device)
    if EdgePropertyKeys.WEIGHT in agent_graph.edata:
        edge_weights[src, dst] = agent_graph.edata[EdgePropertyKeys.WEIGHT]

    systems[0].update(agent_graph, edge_weights=edge_weights) # MovementSystem
    adjacency = _calculate_adjacency(agent_graph)

    # --- 2. BEHAVIORAL & ENVIRONMENTAL UPDATES ---
    # The BehavioralSystem needs a risk proxy. We use the first pathogen's (e.g., Rota) prob.
    # A more robust solution might average risks or use a dedicated risk perception model.
    risk_proxy = 0.002 # Default risk
    for p in pathogens:
        if isinstance(p, Rotavirus):
            p._update_infection_probability() # Ensure probability is current
            risk_proxy = p.current_infection_prob
            break
    
    systems[1].update( # BehavioralSystem
        agent_graph, policy_library=policy_library, risk_levels=risk_levels,
        current_risk_proxy=risk_proxy
    )
    systems[2].update(agent_graph, grid=grid, timestep=timestep) # EnvironmentSystem

    # --- 3. PATHOGEN DYNAMICS ---
    new_cases_by_pathogen: Dict[str, int] = {}
    for pathogen in pathogens:
        pathogen.update(agent_graph, adjacency, grid)
        new_cases_by_pathogen[pathogen.name] = pathogen.new_cases_this_step

    # --- 4. DATA COLLECTION ---
    if (params.data_collection_period > 0 and (timestep % params.data_collection_period == 0)) or \
       (params.data_collection_list and (timestep in params.data_collection_list)):
        data_collection(
            agent_graph, timestep=timestep + 1, npath=params.npath,
            epath=params.epath, ndata=params.ndata, edata=params.edata, mode=params.mode
        )

    # --- 5. GATHER STATISTICS ---
    compartment_counts = {}
    for p in pathogens:
        status = agent_graph.ndata[f"status_{p.name}"]
        compartment_counts[f"{p.name}_S"] = torch.sum(status == Compartment.SUSCEPTIBLE).item()
        compartment_counts[f"{p.name}_E"] = torch.sum(status == Compartment.EXPOSED).item()
        compartment_counts[f"{p.name}_I"] = torch.sum(status == Compartment.INFECTIOUS).item()
        compartment_counts[f"{p.name}_R"] = torch.sum(status == Compartment.RECOVERED).item()
        compartment_counts[f"{p.name}_V"] = torch.sum(status == Compartment.VACCINATED).item()

    return new_cases_by_pathogen, compartment_counts
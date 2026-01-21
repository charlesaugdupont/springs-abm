# abm/model/step.py

"""Time-stepping module for the SVEIR model."""
from typing import Any, Dict, Tuple
import torch
from abm.agent.agent_update import sveir_agent_update
from abm.model.data_collection import data_collection
from abm.state import AgentGraph
from abm.constants import Compartment


def _calculate_adjacency(agent_graph: AgentGraph) -> torch.Tensor:
    """Calculates dense adjacency matrix based on co-location."""
    num_nodes = agent_graph.num_nodes()
    device = agent_graph.device

    coords = torch.stack([agent_graph.ndata['x'], agent_graph.ndata['y']], dim=1)
    unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)

    src_nodes, dst_nodes = [], []
    for loc_id in range(len(unique_coords)):
        agent_indices_in_group = (inverse_indices == loc_id).nonzero(as_tuple=True)[0]
        if len(agent_indices_in_group) > 1:
            combinations = torch.combinations(agent_indices_in_group, r=2, with_replacement=False)
            src_nodes.append(combinations[:, 0])
            dst_nodes.append(combinations[:, 1])
            src_nodes.append(combinations[:, 1])
            dst_nodes.append(combinations[:, 0])

    adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    if src_nodes:
        all_src = torch.cat(src_nodes)
        all_dst = torch.cat(dst_nodes)
        adjacency_matrix[all_src, all_dst] = 1.0

    return adjacency_matrix

def sveir_step(
    agent_graph: AgentGraph,
    device: torch.device,
    timestep: int,
    params: Dict[str, Any],
    grid: Any,
    policy_library: dict,
    risk_levels: torch.Tensor
) -> Tuple[Any, Dict[str, int]]:
    
    num_nodes = agent_graph.num_nodes()
    
    # Movement is shared
    src, dst = agent_graph.edges()
    edge_weights = torch.zeros((num_nodes, num_nodes), device=device)
    if "weight" in agent_graph.edata:
        edge_weights[src, dst] = agent_graph.edata["weight"].to(device)

    sveir_agent_update("move", agent_graph, edge_weights=edge_weights)
    
    # Calculate Adjacency for H2H
    adjacency = _calculate_adjacency(agent_graph)

    # ----------------------------------------------------
    # 1. ROTAVIRUS TRACK (Viral, H2H, Vaccine)
    # ----------------------------------------------------
    rota_params = params['rotavirus']
    
    # Calculate current step infection probability
    rota_mean = rota_params.infection_prob_mean
    rota_std = rota_params.infection_prob_std
    current_rota_prob = max(0.001, torch.normal(mean=rota_mean, std=rota_std, size=(1,)).item())
    
    # Store this proxy for the Health Investment decision logic
    params['infection_probability_proxy'] = current_rota_prob 

    sveir_agent_update("exposure_increment", agent_graph, pathogen="rota")
    
    new_cases_rota = sveir_agent_update("exposed_to_infectious", agent_graph, params=rota_params, pathogen="rota")
    
    sveir_agent_update("infectious_to_recovered", agent_graph, params=rota_params, pathogen="rota")
    sveir_agent_update("susceptible_to_vaccinated", agent_graph, params=rota_params, pathogen="rota")
    
    sveir_agent_update("susceptible_to_exposed", agent_graph, 
                       global_params=params, pathogen="rota", infection_prob=current_rota_prob, adjacency=adjacency)
                       
    sveir_agent_update("vaccinated_to_exposed", agent_graph, 
                       params=rota_params, global_params=params, pathogen="rota", 
                       infection_prob=current_rota_prob, adjacency=adjacency)
    
    # Water dynamics (assigned to Rota for now)
    sveir_agent_update("water_to_human_transmission", agent_graph, params=params, grid=grid, pathogen="rota")
    sveir_agent_update("human_to_water_transmission", agent_graph, params=params, grid=grid, pathogen="rota")

    # ----------------------------------------------------
    # 2. CAMPYLOBACTER TRACK (Bacterial, Animal)
    # ----------------------------------------------------
    campy_params = params['campylobacter']
    
    sveir_agent_update("exposure_increment", agent_graph, pathogen="campy")
    
    new_cases_campy = sveir_agent_update("exposed_to_infectious", agent_graph, params=campy_params, pathogen="campy")
    
    sveir_agent_update("infectious_to_recovered", agent_graph, params=campy_params, pathogen="campy")
    
    # Animal Transmission
    sveir_agent_update("animal_to_human_transmission", agent_graph, params=campy_params, grid=grid)

    # ----------------------------------------------------
    # 3. SHARED / ENVIRONMENTAL DYNAMICS
    # ----------------------------------------------------
    
    # Health Investment Decision
    sveir_agent_update("health_investment", agent_graph, params=params, policy=policy_library, risk_levels=risk_levels)

    # Water recovery/shock
    sveir_agent_update("water_recovery", agent_graph, params=params, grid=grid)
    if (timestep + 1) % params["shock_frequency"] == 0:
        sveir_agent_update("shock", agent_graph, params=params, grid=grid)

    # Data Collection
    if (params['data_collection_period'] > 0 and (timestep % params['data_collection_period'] == 0)) or \
       (params['data_collection_list'] and (timestep in params['data_collection_list'])):
        data_collection(
            agent_graph,
            timestep=timestep + 1,
            npath=params['npath'],
            epath=params['epath'],
            ndata=params['ndata'],
            edata=params['edata'],
            mode=params['mode']
        )

    # Counts
    # Collect generic stats for both
    compartment_counts = {}
    for p in ["rota", "campy"]:
        status_tensor = agent_graph.ndata[f"status_{p}"]
        compartment_counts[f"{p}_S"] = torch.sum(status_tensor == Compartment.SUSCEPTIBLE).item()
        compartment_counts[f"{p}_E"] = torch.sum(status_tensor == Compartment.EXPOSED).item()
        compartment_counts[f"{p}_I"] = torch.sum(status_tensor == Compartment.INFECTIOUS).item()
        compartment_counts[f"{p}_R"] = torch.sum(status_tensor == Compartment.RECOVERED).item()
        compartment_counts[f"{p}_V"] = torch.sum(status_tensor == Compartment.VACCINATED).item()

    return (new_cases_rota, new_cases_campy), compartment_counts
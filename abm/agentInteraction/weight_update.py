# abm/agentInteraction/weight_update.py
import torch

from abm.agent_graph import AgentGraph
from abm.constants import AgentPropertyKeys, EdgePropertyKeys

def weight_update_sveir(agent_graph: AgentGraph, decay_rate: float, truncation_weight: float):
    """
    Update function to calculate the weight of edges based on Euclidean distance
    between agents' home locations.
    """
    u, v = agent_graph.edges()
    device = agent_graph.device

    u = u.to(device)
    v = v.to(device)

    # Extract home locations (y, x)
    loc_u = agent_graph.ndata[AgentPropertyKeys.HOME_LOCATION][u]
    loc_v = agent_graph.ndata[AgentPropertyKeys.HOME_LOCATION][v]

    # Compute pairwise Euclidean distance
    distance = torch.sqrt(torch.sum((loc_u - loc_v)**2, dim=1))

    # Apply decaying weight function
    weights = torch.exp(-decay_rate * distance)

    # Apply truncation
    truncated_weights = torch.where(weights > truncation_weight, weights, truncation_weight)

    agent_graph.edata[EdgePropertyKeys.WEIGHT] = truncated_weights.to(device)
import dgl.function as fn
import torch


def weight_update_sveir(agent_graph, device, decay_rate, truncation_weight):
    """
    Update function to calculate the weight of edges based on the physical distance 
    (Euclidean distance) between connected nodes. The formula used for weight computation is:
    
        weight = exp(-decay_rate * d(x_i, x_j))
        
    where:
        decay_rate = parameter controlling the rate of weight decay with distance
        d(x_i, x_j) = Euclidean distance between positions of connected agents

    Weights below a specified truncation value are set to that truncation value.

    Parameters:
    - agent_graph: DGLGraph object representing the network.
    - device: torch device (e.g., 'cpu' or 'cuda').
    - decay_rate: Rate at which the weights decay with increasing distance.
    - truncation_weight: Minimum allowable weight for any edge.
    """
    # Extract x and y positions of connected nodes (u = source, v = target)
    u, v = agent_graph.edges()
    x_u = agent_graph.ndata['home_location'][u, 0]  # x-coordinates of source nodes
    y_u = agent_graph.ndata['home_location'][u, 1]  # y-coordinates of source nodes
    x_v = agent_graph.ndata['home_location'][v, 0]  # x-coordinates of target nodes
    y_v = agent_graph.ndata['home_location'][v, 1]  # y-coordinates of target nodes

    # Compute pairwise Euclidean distance between connected nodes
    distance = torch.sqrt((x_u - x_v)**2 + (y_u - y_v)**2)  # Euclidean distance

    # Apply decaying weight function (exponential decay)
    weights = torch.exp(-decay_rate * distance)

    # Handle truncation (clip weights below truncation_weight)
    truncated_weights = torch.where(weights > truncation_weight, weights, truncation_weight)

    # Update the edge weights in the graph
    agent_graph.edata['weight'] = truncated_weights.to(device)

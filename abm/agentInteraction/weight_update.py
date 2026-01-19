import torch

def weight_update_sveir(agent_graph, device, decay_rate, truncation_weight):
    """
    Update function to calculate the weight of edges based on Euclidean distance.
    """
    # Extract x and y positions of connected nodes (u = source, v = target)
    # Using our new AgentGraph .edges() method
    u, v = agent_graph.edges()
    
    # Ensure indices are on the correct device
    u = u.to(device)
    v = v.to(device)
    
    x_u = agent_graph.ndata['home_location'][u, 0]
    y_u = agent_graph.ndata['home_location'][u, 1]
    x_v = agent_graph.ndata['home_location'][v, 0]
    y_v = agent_graph.ndata['home_location'][v, 1]

    # Compute pairwise Euclidean distance
    distance = torch.sqrt((x_u - x_v)**2 + (y_u - y_v)**2)

    # Apply decaying weight function
    weights = torch.exp(-decay_rate * distance)

    # Handle truncation
    truncated_weights = torch.where(weights > truncation_weight, weights, truncation_weight)

    # Update the edge weights in the graph dict
    agent_graph.edata['weight'] = truncated_weights.to(device)
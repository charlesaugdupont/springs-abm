# abm/network/network_creation.py
"""Network creation functions."""
import networkx as nx
import torch
import numpy as np
from abm.state import AgentGraph
from typing import Optional

def network_creation(num_agents: int, method: str, verbose: bool, device: str,
                     active_indices: Optional[torch.Tensor] = None, **kwargs) -> AgentGraph:
    """Creates the graph network for the model."""
    agent_graph = AgentGraph(num_agents, device=device)

    if method == 'barabasi-albert':
        seed = kwargs.get('seed', 42)
        new_node_edges = kwargs.get('new_node_edges', 1)

        # Determine which subset of agents will form the network (e.g., adults only)
        if active_indices is not None:
            num_active_nodes = len(active_indices)
            # A mapping from the local index (0 to N-1) in the nx_graph to the global agent ID
            id_map = active_indices.cpu().numpy()
        else:
            num_active_nodes = num_agents
            id_map = np.arange(num_agents)

        if verbose:
            print(f"Creating Barabasi-Albert network for {num_active_nodes} active agents (out of {num_agents} total).")

        # Generate the graph topology using networkx
        nx_graph = nx.barabasi_albert_graph(n=num_active_nodes, m=new_node_edges, seed=seed)
        edges = np.array(list(nx_graph.edges())).T

        if edges.size > 0:
            # Map local edge indices back to global agent IDs
            u_local, v_local = edges[0], edges[1]
            u_global = torch.from_numpy(id_map[u_local]).long()
            v_global = torch.from_numpy(id_map[v_local]).long()

            # Make the graph undirected by adding edges in both directions
            u_final = torch.cat([u_global, v_global])
            v_final = torch.cat([v_global, u_global])

            agent_graph.add_edges(u_final, v_final)

        return agent_graph
    else:
        raise NotImplementedError(f"Network creation method '{method}' is not implemented.")
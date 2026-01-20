"""Network creation functions."""
#!/usr/bin/env python

import networkx as nx
import torch
import numpy as np
from abm.state import AgentGraph

def network_creation(num_agents, method, verbose, active_indices=None, **kwargs):
    """Creates the graph network for the model."""
    agent_graph = AgentGraph(num_agents)

    if method == 'barabasi-albert':
        seed = kwargs.get('seed', torch.initial_seed())
        new_node_edges = kwargs.get('new_node_edges', 1)
        
        # Determine who gets edges
        if active_indices is not None:
            # We are building a sub-network (e.g., Adults only)
            num_active_nodes = len(active_indices)
            mapping = active_indices.cpu().numpy() # Maps local graph index 0..N to global Agent index
        else:
            num_active_nodes = num_agents
            mapping = np.arange(num_agents)

        if verbose:
            print(f"Creating {method} network for {num_active_nodes} active agents out of {num_agents} total.")

        # Generate topology for the active subset
        nx_graph = nx.barabasi_albert_graph(n=num_active_nodes, m=new_node_edges, seed=int(seed))

        # Extract edges (NetworkX returns list of tuples)
        edges = np.array(nx_graph.edges()).T 

        if edges.size > 0:
            # These edges refer to 0..N_Adults. We need to map them to real Agent IDs.
            u_local = edges[0]
            v_local = edges[1]

            u_global = torch.from_numpy(mapping[u_local]).long()
            v_global = torch.from_numpy(mapping[v_local]).long()

            # Make undirected
            u_final = torch.cat([u_global, v_global])
            v_final = torch.cat([v_global, u_global])

            agent_graph.add_edges(u_final, v_final)

        return agent_graph
    else:
        raise NotImplementedError('Currently only barabasi-albert model implemented!')

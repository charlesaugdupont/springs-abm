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

def barabasi_albert_graph(num_agents, new_node_edges=1, seed=1):
    """Create a barabasi-albert graph.
    
    This function creates a network graph for user-defined
    number of agents using the barabasi albert model function 
    from networkx.

    Args:
        num_agents: Number of agent nodes
        new_node_edges: Number of edges to create for each new node
        seed: random seed for function

    Return:
        agent_graph: Created agent_graph as per the chosen method
    """
    #Create graph using networkx function for barabasi albert graph 
    networkx_graph = nx.barabasi_albert_graph(n=num_agents, m=new_node_edges, seed=seed)
    barabasi_albert_coo = nx.to_scipy_sparse_array(networkx_graph,format='coo')
    
    #Return DGL graph from networkx graph
    return dgl.from_scipy(barabasi_albert_coo)

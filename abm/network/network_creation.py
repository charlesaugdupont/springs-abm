"""Network creation functions."""
#!/usr/bin/env python

import networkx as nx
import torch
import numpy as np
from abm.state import AgentGraph

def network_creation(num_agents, method, verbose, **kwargs):
    """Creates the graph network for the model."""
    if (method == 'barabasi-albert'):
        seed = kwargs.get('seed', torch.initial_seed())
        new_node_edges = kwargs.get('new_node_edges', 1)
        
        if verbose:
            print(f"Using seed {seed} for network creation with {new_node_edges} edges requested.")
        
        # Generate topology using NetworkX
        nx_graph = nx.barabasi_albert_graph(n=num_agents, m=new_node_edges, seed=int(seed))
        
        # Convert to our AgentGraph container
        agent_graph = AgentGraph(num_agents)
        
        # Extract edges (NetworkX returns list of tuples, convert to tensor)
        # NetworkX edges are often (u, v), we need 2 rows [u_list, v_list]
        edges = np.array(nx_graph.edges()).T 
        
        # Handle case where graph has no edges (rare for BA but possible with n=1)
        if edges.size == 0:
            u, v = torch.tensor([]), torch.tensor([])
        else:
            u = torch.from_numpy(edges[0]).long()
            v = torch.from_numpy(edges[1]).long()
            
            # Make undirected (add reverse edges)
            u_final = torch.cat([u, v])
            v_final = torch.cat([v, u])
            
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

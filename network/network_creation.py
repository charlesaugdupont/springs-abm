"""Network creation functions."""
#!/usr/bin/env python


import dgl
import networkx as nx
import torch

# network_creation - Creates the network between the initialized nodes
# using edges from DGL.

def network_creation(num_agents, method, verbose, **kwargs):
    """network_creation - Creates the graph network for the model.

    Args:
        num_agents: Number of agent nodes
        method: Current implemented methods include:
            barabasi_albert model: 
                This method takes the following possible keyword arguments,
                seed: random seed for networkx barabasi_albert_graph function.
                new_node_edges: number of edges to create for each new node.
        kwargs: keyword arguments to be supplied to the network creation method.

    Return:
        agent_graph: Created agent_graph as per the chosen method
    """
    if (method == 'barabasi-albert'):
        if 'seed' in kwargs.keys():
            seed  = kwargs['seed']
        else:
            seed = torch.initial_seed() 
        
        if 'new_node_edges' in kwargs.keys(): 
            new_node_edges = kwargs['new_node_edges']
        else:
            new_node_edges = 1 
        if verbose:
            print(f"Using seed {seed} for network creation with {new_node_edges} edges requested.")
        agent_graph = barabasi_albert_graph(num_agents, new_node_edges, seed)
    else:
        raise NotImplementedError('Currently only barabasi-albert model implemented!')
    
    return agent_graph

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

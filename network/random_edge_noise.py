import torch
import dgl 
from dgl.sparse import spmatrix


def random_edge_noise(graph,device,n_perturbances):
    '''This function ads noise to n graph edges by randomly sampling two nodes and assigning a random 
    weight to an edge between them, removingand reinstating edges that already exist, and tracking the number 
    of edges created as a result of this process. Note: a flow of adding nonexisting edges and using apply_edges
    was considered, but indexing the random number from the edge id was convoluted. It is unclear whether the
    speed has been improved or worsened by this choice.'''
    

    # Select neighbor node pairs randomly from graph and remove any duplicates and autoconnection suggestions
    node_pairs,_ = torch.sort(torch.stack((torch.randint(0,graph.nodes().shape[0],(n_perturbances,)),torch.randint(0,graph.nodes().shape[0],(n_perturbances,))), dim=1), dim=1)
    node_pairs = torch.unique(node_pairs, dim=0).to(device)
    node_pairs = node_pairs[(node_pairs[:, 0] != node_pairs[:, 1])]

    # Delete existing edges between node pairs
    existing_connections = graph.has_edges_between(node_pairs[:,0], node_pairs[:,1])
    existing_forward = graph.edge_ids(node_pairs[:,0][existing_connections], node_pairs[:,1][existing_connections])
    existing_reverse = graph.edge_ids(node_pairs[:,1][existing_connections], node_pairs[:,0][existing_connections])
    graph.remove_edges(torch.cat((existing_forward, existing_reverse)))

    # Assign random weights to new node_pair edges 
    random_weights = torch.rand(node_pairs.size(0)).to(device)
    graph.add_edges(node_pairs[:, 0], node_pairs[:, 1],{'weight': random_weights})
    graph.add_edges(node_pairs[:, 1], node_pairs[:, 0],{'weight': random_weights})




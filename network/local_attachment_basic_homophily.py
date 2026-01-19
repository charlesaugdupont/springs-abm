import torch
import dgl 
from dgl.sparse import spmatrix


def local_attachment_homophily(graph,device,n_FoF_links, homophily_parameter = None, characteristic_distance = None, truncation_weight = None):
    '''This function attempts to form links between two agents connected by a common neighbor
    by randomly selecting 2 neighbors of randomly selected nodes with 2 or more neighbors, 
    calculating the potential homophily weight of the new edge and forming the edge if the
    random number generated is less than the potential homophily edge weight.
    Notes: n_FoF_links represents attempted links; sampling of connecting nodes is performed 
    with replacement; if a connecting agent's sampled neighbors are already connected, or the 
    random number generated is greater than the potential homophily edge weight, a new link 
    is not formed. A potentially connecting pair of neighbors is only considered once per call.'''

    #preselect based on adjacency matrix for 2 or more neighbors
    candidates=graph.adj().sum(dim=1)>1
    if torch.sum(candidates)==0:
        print("There are no agents with two or more neighbors. No local attachment can occur.")
        return

    # Select bridge/connecting nodes randomly from candidates (with replacement)
    connecting_nodes=torch.nonzero(candidates, as_tuple=True)[0][torch.randint(0,torch.sum(candidates),(n_FoF_links,))]

    # Sample 2 neighbors of bridge/connecting nodes and remove any duplicates or false autopairs
    sample = dgl.sampling.sample_neighbors(graph,connecting_nodes, 2 , replace= False, edge_dir="out")
    node_pairs,_ = torch.sort(sample.edges(order='eid')[1].view(-1, 2), dim=1)
    node_pairs = torch.unique(node_pairs, dim=0)
    node_pairs = node_pairs[(node_pairs[:, 0] != node_pairs[:, 1])]

    # Extract node pairs and exclude existing edges
    existing_connections = graph.has_edges_between(node_pairs[:,0], node_pairs[:,1])
    even_indices_tensor = node_pairs[:,0][~existing_connections]
    odd_indices_tensor = node_pairs[:,1][~existing_connections]
    prob_tensor = torch.rand(even_indices_tensor.size(0)).to(device)

    # Compare random number for each prospective link to projected homophily edge weight
    wealth_diff = sample.ndata['wealth'][even_indices_tensor] - sample.ndata['wealth'][odd_indices_tensor]
    potential_weights = 1./(1. + torch.exp(homophily_parameter*(torch.abs(wealth_diff)-characteristic_distance)))
    finiteweights = torch.isfinite(potential_weights)
    potential_weights[~finiteweights] = 0.
    potential_weights = torch.where(potential_weights > truncation_weight, potential_weights, truncation_weight)

    successful_links = potential_weights > prob_tensor

    # Add new edges to the original graph
    graph.add_edges(even_indices_tensor[successful_links], odd_indices_tensor[successful_links], data={'weight': potential_weights[successful_links]})
    graph.add_edges(odd_indices_tensor[successful_links], even_indices_tensor[successful_links], data={'weight': potential_weights[successful_links]})






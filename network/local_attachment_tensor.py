import torch
import dgl 
from dgl.sparse import spmatrix

import dgl_ptm.util.matrix_utils as matrix_utils

# TODO: check readability variables.

# TODO: confirm that the behavior for sum(weight)=0 agents is the same as for plain local attachment
def local_attachment_tensor(graph,n_FoF_links,edge_prop=None,p_attach=1.):
    adj_matrix = adjacency_matrix_with_edge_prop(graph,eprop=edge_prop)
    norm_prop = adj_matrix.val/adj_matrix.val.sum()

    #sample n_FoF_links from the entire normalized edge property graph weighted by weight
    selected_links = norm_prop.multinomial(num_samples=n_FoF_links,replacement=False)
    selected_links_matrix = matrix_utils.apply_mask_to_sparse_matrix(adj_matrix, selected_links)
    FoF_field_matrix, FoF_field_matrix_norm_eprop = neighbour_field_matrix(selected_links_matrix,adj_matrix)
    new_FoF = FoF_field_matrix.val > 0
    new_FoF_norm_eprop = matrix_utils.apply_mask_to_sparse_matrix(FoF_field_matrix_norm_eprop, new_FoF)
    if torch.count_nonzero(new_FoF) <= n_FoF_links:
        probe_p_attach = torch.rand(new_FoF_norm_eprop.val.shape[0]) 
        to_link = probe_p_attach < p_attach
    else:
        new_FoF_renorm = new_FoF_norm_eprop.val/new_FoF_norm_eprop.val.sum()
        selected_FoF = new_FoF_renorm.flatten().multinomial(num_samples=n_FoF_links,replacement=False)
        #probe_p_attach = torch.rand(new_FoF_norm_eprop.val.shape[0])
        probe_p_attach = torch.rand(n_FoF_links)
        #to_link = torch.logical_and((probe_p_attach > p_attach),selected_FoF)
        probe_selected = probe_p_attach < p_attach
        to_link = selected_FoF[probe_selected]
    FoF_to_link_matrix = matrix_utils.apply_mask_to_sparse_matrix(new_FoF_norm_eprop, to_link)
    graph.add_edges(FoF_to_link_matrix.row,FoF_to_link_matrix.col)
    graph.add_edges(FoF_to_link_matrix.col,FoF_to_link_matrix.row)

def adjacency_matrix_with_edge_prop(graph,etype=None, eprop=None):
    etype = graph.to_canonical_etype(etype)
    indices = torch.stack(graph.all_edges(etype=etype))
    shape = (graph.num_nodes(etype[0]),graph.number_of_nodes(etype[2]))
    if eprop is not None:
        val =graph.edges[etype].data[eprop].flatten()
    else:
        val=None
    return spmatrix(
        indices,
        val=val,
        shape=shape,
    )

def construct_neighbour_field_tensors(selm,adjm):
    """
    This fucnction creates the row tensor, column tensor, link tensor, and value tensor
    needed to construct the neighbour field matrix/tensor in sparse representation. This is done by creating a list of the tensor
    representations for each element i,j of the matrix of selected edges by adding row i and row j of the adjacency
    matrix, and storing the result in row i, for each element an insttance of row i as well as the entry i,i is removed. The list is subsequently concatenated to obtain a single tensor representation.
    In addition to the link tensor, which denotes the link status with and integer, the function also returns a value tensor with
    with entries corresponding to the weight of edge j,k. neighbours wiith no direct connecton appear as > 0 values
    
    The resulting tensors can/will contain significant numbers of multiple assigments for an element i,k. This is addressed in
    subsequent processing. This also handles the combinattion of weights for elligible connections arising from multiple possible links 
    
    Input:
    
    :param: selm matrix of selected edges in sparse format (dgl.sparse.spmatrix)
    :param: adjm adjacency matrix witth edge weights as sparse values
    
    Output:
    
    :param: rowtensor (row tensor of the neighbour field matrix, compattible with sparse format)
    :param: coltensor (column tensor of the neighbour field matrix, compatible with sparse format)
    :param: ltensor (link tensor of the neighbouir field matrix, compatible with sparse format) 
    :param: valtensor (tensor with edge weights of link jk)
    
    """
    rtl = list()
    ctl = list()
    ltl = list()
    vtl = list()
    for i in range(selm.row.shape[0]):
        src = selm.row[i]
        dst = selm.col[i]
        srcten = torch.tensor([src])
        wsrcten = torch.tensor([0.])
        lsrcten = torch.tensor([-1])
        src_nids = adjm.col[adjm.row==src]
        wsrc_nids = adjm.val[adjm.row==src]
        lsrc_nids = torch.ones_like(src_nids,dtype=int)*(-1)
        dst_nids = adjm.col[adjm.row==dst]
        wdst_nids = adjm.val[adjm.row==dst]
        ldst_nids = torch.ones_like(dst_nids,dtype=int)
        rv_src_nids = torch.ones_like(src_nids)*src
        rv_dst_nids = torch.ones_like(dst_nids)*src
        rv_nids = torch.cat((rv_src_nids,rv_dst_nids,srcten))
        cv_nids = torch.cat((src_nids,dst_nids,srcten))
        lv_nids = torch.cat((lsrc_nids,ldst_nids,lsrcten))
        wv_nids = torch.cat((wsrc_nids,wdst_nids,wsrcten))
        rtl.append(rv_nids)
        ctl.append(cv_nids)
        ltl.append(lv_nids)
        vtl.append(wv_nids)
    rowtensor = torch.cat(rtl)
    coltensor = torch.cat(ctl)
    ltensor   = torch.cat(ltl)
    valtensor = torch.cat(vtl)
    return (rowtensor, coltensor, ltensor, valtensor)

def neighbour_field_matrix(selm, adjm):
    nf = construct_neighbour_field_tensors(selm,adjm)
    indices= torch.stack((nf[0],nf[1])) #stack indices into tensor for matrix construction
    nfm = dgl.sparse.spmatrix(indices,nf[2],shape=selm.shape) # create link neighbour fields
    nfmv = dgl.sparse.spmatrix(indices,nf[3],shape=selm.shape) # create value neighbour fields
    nfmc = nfm.coalesce()
    nfmvc = nfmv.coalesce()
    return nfmc, nfmvc 
"""The module to collect data from agents and edges."""
import os
from pathlib import Path
import xarray as xr

def data_collection(agent_graph, timestep, npath='./agent_data', epath='./edge_data',
                    ndata=None, edata=None, format='xarray', mode='w-', verbose=False):
    """collects data from agents and edges."""
    
    # Updated: use node_keys() instead of node_attr_schemes()
    available_ndata = agent_graph.node_keys()
    
    if ndata == ['all']:
        ndata = available_ndata
    elif ndata[0] == 'all_except':
        ndata = list(set(available_ndata) - set(ndata[1]))
    
    available_edata = agent_graph.edge_keys()
    if edata == ['all']:
        edata = available_edata

    if ndata is not None:
        _node_property_collector(agent_graph, npath, ndata, timestep, format, mode)

    if edata is not None:
        _edge_property_collector(agent_graph, epath, edata, timestep, format, mode)

def _node_property_collector(agent_graph, npath, ndata, timestep, format, mode):
    if format == 'xarray':
        agent_data_instance = xr.Dataset()
        for prop in ndata:
            _check_nprop_in_graph(agent_graph, prop)
            agent_data_instance = agent_data_instance.assign(
                prop=(['n_agents','n_time'], agent_graph.ndata[prop][:,None].cpu().numpy())
                )
            agent_data_instance = agent_data_instance.rename(name_dict={'prop':prop})
        if timestep == 0:
            agent_data_instance.to_zarr(npath, mode=mode)
        else:
            agent_data_instance.to_zarr(npath, append_dim='n_time')
    else:
        raise NotImplementedError("Only 'xarray' format currently available")

def _edge_property_collector(agent_graph, epath, edata, timestep, format, mode):
    if format == 'xarray':
        # agent_graph.edges() now returns tensors (u, v)
        u, v = agent_graph.edges()
        edge_data_instance = xr.Dataset(
            coords=dict(
                source=(["n_edges"], u.cpu()),
                dest=(["n_edges"], v.cpu()),
                )
            )
        for prop in edata:
            _check_eprop_in_graph(agent_graph, prop)
            edge_data_instance = edge_data_instance.assign(
                property=(['n_edges','time'], agent_graph.edata[prop][:,None].cpu().numpy())
                )
            edge_data_instance = edge_data_instance.rename_vars(name_dict={'property':prop})
        edge_data_instance.to_zarr(Path(epath)/(str(timestep)+'.zarr'), mode=mode)
    else:
        raise NotImplementedError("Only 'xarray' mode current available")

def _check_nprop_in_graph(agent_graph, prop):
    if prop not in agent_graph.ndata:
        raise ValueError(f"{prop} is not a node property.")

def _check_eprop_in_graph(agent_graph, prop):
    if prop not in agent_graph.edata:
        raise ValueError(f"{prop} is not an edge property.")
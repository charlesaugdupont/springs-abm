"""The module to collect data from agents and edges."""

import os
from pathlib import Path

import xarray as xr


def data_collection(agent_graph,
                    timestep,
                    npath='./agent_data',
                    epath='./edge_data',
                    ndata=None,
                    edata=None,
                    format = 'xarray',
                    mode = 'w-',
                    verbose = False):
    """data_collection - collects data from agents and edges.

    Args:
        agent_graph: DGLGraph with agent nodes and edges connecting agents.
        timestep: current timestep to name folder for edge properties
        npath: path to store node data.
        epath: path to store edge data with one file for each timestep.
        ndata: node data properties to be stored.
            [list] specifies node properties to be stored at every time step
            ['all'] implies all node properties will be saved at every time step
            ['all_except', [list]] specifies that all but the listed properties will be saved
            ['initial_only', [list]] specifies properties saved at timestep 0 only
            [list] and ['all_except', [list]] can be used with [initial_only, [list]]
            formatted as [[specification list],[specification list]]
            ['all'] should not be used together with any other specification
        edata: edge data properties to be stored.
            ['all'] implies all edge properties will be saved
        format: storage format
            ['xarray'] saves the properties in zarr format with xarray dataset
        mode: zarr write mode.
    """
    if ndata == ['all']:
        ndata = list(agent_graph.node_attr_schemes().keys())
    elif ndata[0] == 'all_except':
        ndata = list(agent_graph.node_attr_schemes().keys() - ndata[1])
    elif sum(1 for item in ndata if isinstance(item, list)) > 1:
        ndata_list = ndata
        initial_only=[]
        for specification in ndata_list:
            if specification == ['all']:
                raise ValueError('Use of "all" is not compatible with multiple data collection specification lists.')
            elif specification[0] == 'all_except':
                ndata = list(agent_graph.node_attr_schemes().keys() - specification[1])
            elif specification[0] == 'initial_only':
                if timestep == 0:
                    initial_only = specification[1]
            else:
                ndata = specification
        
    else:
        raise ValueError('Invalid node data collection specification.')
    
    if edata == ['all']:
        edata = list(agent_graph.edge_attr_schemes().keys())
    
    if ndata == None:
        if timestep == 0 and verbose:
            print("ATTENTION: No node data collection requested for this simulation!")
    else:
        if timestep == 0 and locals().get('initial_only', []) != []:
            initialpath = npath.split('.')[0] + '_initial.zarr'
            _node_property_collector(agent_graph, initialpath, initial_only, timestep, format, mode)
        _node_property_collector(agent_graph, npath, ndata, timestep, format, mode)
    if edata == None:
        if timestep == 0 and verbose:
            print("ATTENTION: No edge data collection requested for this simulation!")
    else:
        _edge_property_collector(agent_graph, epath, edata, timestep, format, mode)


def _node_property_collector(agent_graph, npath, ndata, timestep, format, mode):
    if os.environ["DGLBACKEND"] == "pytorch":
        if format == 'xarray':
            agent_data_instance = xr.Dataset()
            for prop in ndata:
                _check_nprop_in_graph(agent_graph, prop)
                agent_data_instance = agent_data_instance.assign(
                    prop=(['n_agents','n_time'], agent_graph.ndata[prop][:,None].cpu().numpy())  # noqa: E501
                    )
                agent_data_instance = agent_data_instance.rename(
                    name_dict={'prop':prop}
                    )
            if timestep == 0:
                agent_data_instance.to_zarr(npath, mode = mode)
            else:
                agent_data_instance.to_zarr(npath, append_dim='n_time')
        else:
            raise NotImplementedError("Only 'xarray' format currently available")
    else:
        raise NotImplementedError(
            "Data collection currently only implemented for pytorch backend"
            )


def _edge_property_collector(agent_graph, epath, edata, timestep, format, mode):
    if os.environ["DGLBACKEND"] == "pytorch":
        if format == 'xarray':
            edge_data_instance = xr.Dataset(
                coords=dict(
                    source=(["n_edges"], agent_graph.edges()[0].cpu()),
                    dest=(["n_edges"], agent_graph.edges()[1].cpu()),
                    )
                )
            for prop in edata:
                _check_eprop_in_graph(agent_graph, prop)
                edge_data_instance = edge_data_instance.assign(
                    property=(['n_edges','time'], agent_graph.edata[prop][:,None].cpu().numpy()) # noqa: E501
                    )

                edge_data_instance = edge_data_instance.rename_vars(
                    name_dict={'property':prop}
                    )
            edge_data_instance.to_zarr(Path(epath)/(str(timestep)+'.zarr'), mode = mode)
        else:
            raise NotImplementedError("Only 'xarray' mode current available")
    else:
        raise NotImplementedError(
            "Data collection currently only implemented for pytorch backend"
            )

def _check_nprop_in_graph(agent_graph, prop):
    if prop not in agent_graph.node_attr_schemes().keys():
        raise ValueError(
            f"{prop} is not a node property."
            f"Please choose from {agent_graph.node_attr_schemes().keys()}"
            )

def _check_eprop_in_graph(agent_graph, prop):
    if prop not in agent_graph.edge_attr_schemes().keys():
        raise ValueError(
            f"{prop} is not an edge property."
            f"Please choose from {agent_graph.edge_attr_schemes().keys()}"
            )

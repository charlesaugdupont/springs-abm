# abm/model/data_collection.py
"""The module to collect data from agents and edges."""
from pathlib import Path
import xarray as xr
from abm.state import AgentGraph

def data_collection(agent_graph: AgentGraph, timestep: int, npath: str, epath: str,
                    ndata: list | None, edata: list | None, mode: str):
    """Collects specified data from agents and edges and saves it."""
    available_ndata = agent_graph.node_keys()
    if ndata == ['all']:
        ndata_to_collect = available_ndata
    elif ndata and ndata[0] == 'all_except':
        ndata_to_collect = list(set(available_ndata) - set(ndata[1]))
    else:
        ndata_to_collect = ndata

    available_edata = agent_graph.edge_keys()
    if edata == ['all']:
        edata_to_collect = available_edata
    else:
        edata_to_collect = edata

    if ndata_to_collect:
        _node_property_collector(agent_graph, npath, ndata_to_collect, timestep, mode)

    if edata_to_collect:
        _edge_property_collector(agent_graph, epath, edata_to_collect, timestep, mode)

def _node_property_collector(agent_graph: AgentGraph, npath: str, ndata: list, timestep: int, mode: str):
    """Collects and saves node (agent) data to a Zarr store."""
    agent_data = xr.Dataset()
    for prop in ndata:
        if prop not in agent_graph.ndata:
            raise ValueError(f"Node property '{prop}' not found in graph.")
        # Ensure data is 2D (n_agents, n_time=1) for concatenation
        data_tensor = agent_graph.ndata[prop].cpu().numpy()
        if data_tensor.ndim == 1:
            data_tensor = data_tensor[:, None]
        agent_data[prop] = (['n_agents', 'n_time'], data_tensor)

    if timestep == 0:
        agent_data.to_zarr(npath, mode=mode)
    else:
        agent_data.to_zarr(npath, append_dim='n_time')

def _edge_property_collector(agent_graph: AgentGraph, epath: str, edata: list, timestep: int, mode: str):
    """Collects and saves edge data for a specific timestep to a Zarr store."""
    u, v = agent_graph.edges()
    edge_data = xr.Dataset(
        coords=dict(
            source=(["n_edges"], u.cpu().numpy()),
            dest=(["n_edges"], v.cpu().numpy()),
        )
    )
    for prop in edata:
        if prop not in agent_graph.edata:
            raise ValueError(f"Edge property '{prop}' not found in graph.")
        data_tensor = agent_graph.edata[prop].cpu().numpy()
        edge_data[prop] = (['n_edges'], data_tensor)

    # Save each timestep as a separate file
    epath_dir = Path(epath)
    epath_dir.mkdir(parents=True, exist_ok=True)
    edge_data.to_zarr(epath_dir / f"{timestep}.zarr", mode=mode)
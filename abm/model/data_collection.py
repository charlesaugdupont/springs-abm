# abm/model/data_collection.py
import xarray as xr
from abm.state import AgentState

def data_collection(agent_state: AgentState, timestep: int, npath: str,
                    ndata: list | None, mode: str):
    """Collects specified data from agents and saves it."""
    
    # 1. Determine which Node properties to collect
    available_ndata = agent_state.node_keys()
    if ndata == ['all']:
        ndata_to_collect = available_ndata
    elif ndata and ndata[0] == 'all_except':
        ndata_to_collect = list(set(available_ndata) - set(ndata[1]))
    else:
        ndata_to_collect = ndata

    # 2. Collect Node Data
    if ndata_to_collect:
        _node_property_collector(agent_state, npath, ndata_to_collect, timestep, mode)

def _node_property_collector(agent_state: AgentState, npath: str, ndata: list, timestep: int, mode: str):
    """Collects and saves node (agent) data to a Zarr store."""
    agent_data = xr.Dataset()
    for prop in ndata:
        if prop not in agent_state.ndata:
            raise ValueError(f"Node property '{prop}' not found in graph.")
        # Ensure data is 2D (n_agents, n_time=1) for concatenation
        data_tensor = agent_state.ndata[prop].cpu().numpy()
        if data_tensor.ndim == 1:
            data_tensor = data_tensor[:, None]
        agent_data[prop] = (['n_agents', 'n_time'], data_tensor)

    if timestep == 0:
        agent_data.to_zarr(npath, mode=mode)
    else:
        agent_data.to_zarr(npath, append_dim='n_time')
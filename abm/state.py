# abm/state.py
from typing import Dict, List
import torch

class AgentState:
    """
    A lightweight container for agent-based modeling state (ndata).
    Note: Explicit edges are not stored; interaction is spatial/dynamic.
    """
    def __init__(self, num_nodes: int, device: str | torch.device = 'cpu'):
        self._num_nodes: int = num_nodes
        self.device: str | torch.device = device

        # Dictionary to store node features (agent properties)
        self.ndata: Dict[str, torch.Tensor] = {}

    def num_nodes(self) -> int:
        """Returns the number of agents."""
        return self._num_nodes

    def to(self, device: str | torch.device):
        """Moves all agent data to the specified device."""
        self.device = device
        for k, v in self.ndata.items():
            if isinstance(v, torch.Tensor):
                self.ndata[k] = v.to(device)
        return self

    def node_keys(self) -> List[str]:
        """Returns a list of all node property keys."""
        return list(self.ndata.keys())
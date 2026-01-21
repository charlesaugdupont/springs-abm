# abm/agent_graph.py
from typing import Dict, List, Tuple
import torch

class AgentGraph:
    """
    A lightweight container for agent-based modeling state, including
    agent properties (ndata), edge properties (edata), and connectivity.
    """
    def __init__(self, num_nodes: int, device: str | torch.device = 'cpu'):
        self._num_nodes: int = num_nodes
        self.device: str | torch.device = device

        # Dictionary to store node features (agent properties)
        self.ndata: Dict[str, torch.Tensor] = {}

        # Dictionary to store edge features
        self.edata: Dict[str, torch.Tensor] = {}

        # Edge storage: 2xN tensor (Row 0: Source, Row 1: Destination)
        self.edge_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long, device=device)

    def num_nodes(self) -> int:
        """Returns the number of nodes (agents) in the graph."""
        return self._num_nodes

    def num_edges(self) -> int:
        """Returns the number of edges in the graph."""
        return self.edge_index.shape[1]

    def edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (source, destination) tensors for all edges."""
        return self.edge_index[0], self.edge_index[1]

    def to(self, device: str | torch.device):
        """Moves all graph data to the specified device."""
        self.device = device
        self.edge_index = self.edge_index.to(device)

        for k, v in self.ndata.items():
            if isinstance(v, torch.Tensor):
                self.ndata[k] = v.to(device)

        for k, v in self.edata.items():
            if isinstance(v, torch.Tensor):
                self.edata[k] = v.to(device)
        return self

    def add_edges(self, u: torch.Tensor, v: torch.Tensor, data: Dict[str, torch.Tensor] | None = None):
        """
        Adds edges to the graph.

        Args:
            u: Source node indices (Tensor).
            v: Destination node indices (Tensor).
            data: Optional dictionary of edge features.
        """
        u = u.to(self.device)
        v = v.to(self.device)
        new_edges = torch.stack([u, v], dim=0)
        self.edge_index = torch.cat([self.edge_index, new_edges], dim=1)

        if data:
            for k, val in data.items():
                val = val.to(self.device)
                if k not in self.edata:
                    self.edata[k] = val
                else:
                    self.edata[k] = torch.cat([self.edata[k], val])

    def node_keys(self) -> List[str]:
        """Returns a list of all node property keys."""
        return list(self.ndata.keys())

    def edge_keys(self) -> List[str]:
        """Returns a list of all edge property keys."""
        return list(self.edata.keys())
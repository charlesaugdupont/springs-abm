import torch

class AgentGraph:
    """
    A lightweight container to replace dgl.DGLGraph for Agent-Based Modeling.
    Stores agent states (ndata), edge states (edata), and connectivity.
    """
    def __init__(self, num_nodes, device='cpu'):
        self._num_nodes = num_nodes
        self.device = device
        
        # Dictionary to store node features (e.g., health, wealth, x, y)
        self.ndata = {}
        
        # Dictionary to store edge features (e.g., weight)
        self.edata = {}
        
        # Edge storage: 2xN tensor (Row 0: Source, Row 1: Destination)
        # Initialize as empty
        self.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    def num_nodes(self):
        return self._num_nodes

    def num_edges(self):
        return self.edge_index.shape[1]

    def edges(self):
        """Returns (src, dst) tensors."""
        return self.edge_index[0], self.edge_index[1]

    def to(self, device):
        """Moves all data to the specified device."""
        self.device = device
        self.edge_index = self.edge_index.to(device)
        
        for k, v in self.ndata.items():
            if isinstance(v, torch.Tensor):
                self.ndata[k] = v.to(device)
        
        for k, v in self.edata.items():
            if isinstance(v, torch.Tensor):
                self.edata[k] = v.to(device)
        return self

    def add_edges(self, u, v, data=None):
        """
        Adds edges to the graph.
        u: Source node indices (Tensor)
        v: Destination node indices (Tensor)
        data: Optional dictionary of edge features
        """
        u = u.to(self.device)
        v = v.to(self.device)
        new_edges = torch.stack([u, v], dim=0)
        self.edge_index = torch.cat([self.edge_index, new_edges], dim=1)
        
        # If data is provided, append it to edata
        # Note: This implies edata must be grown dynamically. 
        # For this refactor, we assume bulk creation mostly.
        if data:
            for k, val in data.items():
                val = val.to(self.device)
                if k not in self.edata:
                    self.edata[k] = val
                else:
                    self.edata[k] = torch.cat([self.edata[k], val])

    def node_keys(self):
        return list(self.ndata.keys())

    def edge_keys(self):
        return list(self.edata.keys())
"""These functions pertain to spatial grid creation."""
import torch

class GridEnvironment:
    """This class represents a grid environment with properties."""
    def __init__(self, grid_tensor: torch.Tensor, property_index: dict):
        self.grid_tensor = grid_tensor
        self.property_to_index = property_index
        self.grid_shape = grid_tensor.shape

    def get_slice(self, property_name: str) -> torch.Tensor:
        """Returns a 2D slice of the grid for a given property."""
        if property_name not in self.property_to_index:
            raise KeyError(f"Property '{property_name}' not found.")
        index = self.property_to_index[property_name]
        return self.grid_tensor[:, :, index]

    def __getitem__(self, property_name: str) -> torch.Tensor:
        return self.get_slice(property_name)

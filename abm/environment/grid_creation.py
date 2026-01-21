"""These functions pertain to spatial grid creation."""
import torch
import numpy as np

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

def sample_distribution_tensor(dist_type: str, dist_parameters: list, n_samples: int,
                               do_round: bool = False, decimals: int | None = None) -> torch.Tensor:
    """Creates and returns samples from different torch distributions."""
    # Convert parameters to tensors if they are not already
    params = [torch.tensor(p) if not isinstance(p, torch.Tensor) else p for p in dist_parameters if p is not None]

    if dist_type == 'uniform':
        dist = torch.distributions.Uniform(params[0], params[1]).sample((n_samples,))
    elif dist_type == 'normal':
        dist = torch.distributions.Normal(params[0], params[1]).sample((n_samples,))
    # Add other distributions as needed
    else:
        raise NotImplementedError(f"Distribution type '{dist_type}' is not supported.")

    if do_round:
        if decimals is None:
            raise ValueError("Rounding requires 'decimals' to be specified.")
        return torch.round(dist, decimals=decimals)
    return dist

def grid_creation(**kwargs) -> GridEnvironment:
    """
    Creates a representation of the model environment.

    Args:
        method (str): The method to use ('basic', 'distribution', 'custom_import').
        **kwargs: Method-specific arguments.
    """
    method = kwargs.get('method')
    if method == 'basic':
        x, y = kwargs['x'], kwargs['y']
        grid = torch.ones(x, y, 1) # Add channel dimension
        return GridEnvironment(grid, {"ones": 0})

    elif method == 'distribution':
        x, y = kwargs['x'], kwargs['y']
        properties = kwargs['properties']
        grid = torch.zeros(x, y, len(properties))
        n = x * y
        for i, (prop_name, dist_info) in enumerate(properties.items()):
            samples = sample_distribution_tensor(
                dist_info['type'], dist_info['parameters'], n,
                do_round=dist_info.get('round', False),
                decimals=dist_info.get('decimals')
            ).reshape(x, y)
            grid[:, :, i] = samples
        prop_map = {name: i for i, name in enumerate(properties.keys())}
        return GridEnvironment(grid, prop_map)

    elif method == 'custom_import':
        path = kwargs['path']
        properties = kwargs['properties']
        if path.endswith(('.np', '.npy')):
            grid_np = np.load(path)
            grid = torch.from_numpy(grid_np)
        elif path.endswith('.pt'):
            grid = torch.load(path)
        else:
            raise ValueError("Unsupported file type for grid import; use .npy or .pt.")
        return GridEnvironment(grid, properties)

    else:
        raise NotImplementedError(f"Unsupported grid creation method: {method}")
"""These functions pertain to spatial grid creation."""
from dgl_ptm.util.utils import sample_distribution_tensor
import torch
import numpy as np

# grid_creation - Creates a representation of the spatial environment 
# within which agents act and interact.

class GridEnvironment:
    """This class represents a grid environment with properties."""
    def __init__(self, grid_tensor, property_index):
        self.grid_tensor = grid_tensor
        self.property_to_index = property_index
        self.grid_shape = grid_tensor.shape

    def get_slice(self, property):
        if property not in self.property_to_index:
            raise KeyError(f"Property '{property}' not found.")
        index = self.property_to_index[property]
        if len(self.grid_tensor.shape) == 2:
            return self.grid_tensor
        else:
            return self.grid_tensor[:, :, index]

    def __getitem__(self, property):
        return self.get_slice(property)

def grid_creation(**kwargs):
    """grid_creation - Creates a representation of the model environment

    Args:
        method: Currently implementable methods include:
            basic: 
                This method creates a uniform grid environment and
                requires the following keyword arguments,
                x: length of the grid in the x-direction/longitude
                y: length of the grid in the y-direction/latitude
            distribution:
                This method creates a grid environment with properties and
                requires the following keyword arguments,
                x: length of the grid in the x-direction/longitude
                y: length of the grid in the y-direction/latitude
                properties: dictionary of property distribution dictionaries 
                    to be assigned to the grid (format example, 
                    {'property1': {'type': 'uniform', 'parameters': [0, 1], 
                            'round': False, 'decimals': None}})
            custom_import:
                This method creates a grid environment by importing a 
                numpy array or torch tensor from a file (.np or .pt or .npy)
                and requires the following keyword arguments,
                path: path at which .np or .pt file is located
                properties: dictionary of property names to be assigned 
                    to the third dimension (format example {'property1': 0, 
                    'property2': 1})

        kwargs: keyword arguments to be supplied to the grid creation method

    Return:
        grid: Grid environment created via the specified method
    """

    if kwargs['method'] == 'basic':
        x = kwargs['x']
        y = kwargs['y']
        grid = torch.ones(x, y)
        grid_environment=GridEnvironment(grid, {"ones": 0})
        return grid_environment
    
    elif kwargs['method'] == 'distribution':
        x = kwargs['x']
        y = kwargs['y']
        properties = kwargs['properties']
        grid = torch.zeros(x, y, len(properties))
        n = x * y
        for i, prop in enumerate(properties):
            distribution = properties[prop]
            
            grid[:, :, i] = sample_distribution_tensor(distribution['type'],
                            distribution['parameters'], n, round = distribution['round'],
                            decimals = distribution['decimals']).reshape(x, y)
        grid_environment=GridEnvironment(grid, {key: i for i, key in enumerate(properties.keys())})
        return grid_environment
           
    elif kwargs['method'] == 'custom_import':
        path = kwargs['path']
        properties = kwargs['properties']
        if path.endswith('.np') or path.endswith('.npy'):
            grid = torch.from_numpy(np.load(path))
        elif path.endswith('.pt'):
            grid = torch.load(path)
        else:
            raise ValueError("File type not supported for grid creation; please use .np or.pt.")
        grid_environment=GridEnvironment(grid, properties)
        return grid_environment
            
    else:
        raise NotImplementedError("Unsupported grid creation method received.")




"""These functions pertain to spatial grid creation."""
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
    
def sample_distribution_tensor(type, dist_parameters, n_samples, round=False, decimals=None):
    """Create and return samples from different distributions.

    :param type: Type of distribution to sample
    :param dist_parameters: array of parameters as required/supported by
        requested distribution type
    :param n_samples: number of samples to return (as 1d tensor)
    :param round: optional, whether the samples are to be rounded
    :param decimals: optional, required if round is specified. decimal places to
        round to
    """
    # check if each item in dist_parameters are torch tensors, if not convert them
    for i, item in enumerate(dist_parameters):
        # if item has dtype NoneType, raise error
        if item is not None and not isinstance(item, torch.Tensor):
                dist_parameters[i] = torch.tensor(item)

    if not isinstance(n_samples, torch.Tensor):
        n_samples = torch.tensor(n_samples)
     
    if type == 'uniform':
        dist = torch.distributions.uniform.Uniform(
            dist_parameters[0], dist_parameters[1]
            ).sample([n_samples])
    elif type == 'normal':
        dist = torch.distributions.normal.Normal(
            dist_parameters[0], dist_parameters[1]
            ).sample([n_samples])
    elif type == 'bernoulli':
        dist = torch.distributions.bernoulli.Bernoulli(
            probs=dist_parameters[0], logits=dist_parameters[1], validate_args=None
            ).sample([n_samples])
    elif type == 'multinomial':
        multinomial_samples = torch.multinomial(
            torch.tensor(dist_parameters[0]), n_samples, replacement=True
            )
        dist = torch.gather(torch.Tensor(dist_parameters[1]), 0, multinomial_samples)
    elif type == 'truncnorm':
        # dist_parameters are mean, standard deviation, min, and max.
        # cdf(x)=(1+erf(x/2^0.5))/2. cdf^-1(x)=2^0.5*erfinv(2*x-1).
        trunc_val_min = (dist_parameters[2]-dist_parameters[0])/dist_parameters[1]
        trunc_val_max = (dist_parameters-dist_parameters[0])/dist_parameters[1]
        cdf_min = (1 + torch.erf(trunc_val_min / torch.sqrt(torch.tensor(2.0))))/2
        cdf_max = (1 + torch.erf(trunc_val_max / torch.sqrt(torch.tensor(2.0))))/2

        uniform_samples = torch.rand(n_samples)
        inverse_transform = torch.erfinv(
            2 *(cdf_min + (cdf_max - cdf_min) * uniform_samples) - 1
            )
        sample_ppf = torch.sqrt(torch.tensor(2.0)) * inverse_transform

        dist = dist_parameters[0] + dist_parameters[1] * sample_ppf
    elif type == 'beta':
        dist = torch.distributions.beta.Beta(dist_parameters[0], dist_parameters[1]
            ).sample([n_samples])
    else:
        raise NotImplementedError(
            'Currently only uniform, normal, multinomial, truncated normal, beta, and '
            'bernoulli distributions are supported'
            )

    if round:
        if decimals is None:
            raise ValueError(
                'rounding requires decimals of rounding accuracy to be specified'
                )
        else:
            return torch.round(dist,decimals=decimals)
    else:
        return dist

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




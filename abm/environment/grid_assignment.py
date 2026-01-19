import torch
import numpy as np

def grid_assignment(graph, grid_environment, **kwargs):
    '''
    grid_assignment - Initial assignment of positions for agents on the grid
    
Args:
        graph: DGL graph object representing the agent network
        grid_environment: GridEnvironment object 
        method: Currently implementable methods include:
            random: 
                This method distributes agents randomly
            property:
                This method distributes agents based on a 
                specified property of the grid environment
                which will be normalized to unity to form 
                an assignment probability. It requires the following 
                keyword argument,
                property: property of the grid environment to be used
                    for agent assignment
            custom_import:
                This method distributes agents based on a 
                array or tensor of positions (D0 agent, D1 x,y). 
                It requires the following keyword argument,
                path: path at which .np or .pt file is located
    '''
    if kwargs['method'] == "random":
        graph.ndata['x'] = torch.randint(0, grid_environment.grid_shape[0], (graph.num_nodes(),)).float()
        graph.ndata['y'] = torch.randint(0, grid_environment.grid_shape[1], (graph.num_nodes(),)).float()

    elif kwargs['method'] == "property":
        property = kwargs['property']
        property_slice = grid_environment[property]
        property_sum = torch.sum(property_slice)
        property_slice = property_slice / property_sum
        property_slice = property_slice.view(-1)
        position = torch.multinomial(property_slice, graph.num_nodes(), replacement=True)
        graph.ndata['x'] = (position % grid_environment.grid_shape[1]).float()
        graph.ndata['y'] = (position // grid_environment.grid_shape[1]).float()

    elif kwargs['method'] == "custom_import":
        path = kwargs['path']
        if path.endswith('.np'):
            position = torch.from_numpy(np.load(path))
        elif path.endswith('.pt'):
            position = torch.load(path)
        graph.ndata['x'] = position[:, 0].float()
        graph.ndata['y'] = position[:, 1].float()


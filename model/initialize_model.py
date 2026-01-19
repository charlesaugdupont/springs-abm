# model/initialize_model.py

"""This module contains the model class and functions to initialize the model."""

import copy
import pickle
import numpy as np
import os
from pathlib import Path

import torch
from dgl.data.utils import load_graphs, save_graphs

from dgl_ptm.agentInteraction.weight_update import weight_update_sveir
from dgl_ptm.config import SVEIRCONFIG
from dgl_ptm.model.step import sveir_step
from dgl_ptm.network.network_creation import network_creation
from dgl_ptm.environment.grid_creation import grid_creation
from dgl_ptm.environment.grid_assignment import grid_assignment

# Set the seed of the random number generator
generator = torch.manual_seed(0)

class Model:
    """Abstract model class."""

    def __init__(self, model_identifier=None, root_path='.'):
        self._model_identifier = model_identifier
        self.root_path = Path(root_path)
        self.model_dir = self.root_path / Path(self._model_identifier)
        self.step_count = 0

    def save_model_parameters(self, overwrite=False):
        """Save model parameters to a yaml file."""
        filename = f"{self._model_identifier}.yaml"
        cfg_filename = self.model_dir / filename
        if overwrite:
            cfg_filename = cfg_filename.with_suffix('.yaml')
        else:
            cfg_filename = _make_path_unique(cfg_filename.as_posix(), '.yaml')
        self.config.to_yaml(cfg_filename)

    def create_network(self):
        raise NotImplementedError('network creation is not implemented for this class.')

    def step(self):
        raise NotImplementedError('step function is not implemented for this class.')

    def run(self):
        raise NotImplementedError('run method is not implemented for this class.')


class SVEIRModel(Model):

    def __init__(self, *, model_identifier, root_path='.'):
        super().__init__(model_identifier=model_identifier, root_path=root_path)

        self.config = None
        self.steering_parameters = None
        self.graph = None
        self.step_first = -1
        self.policy_library = None
        self.risk_levels_tensor = None
        self.infection_incidence = []
        self.prevalence_history = []
        self.susceptible_history = []
        self.exposed_history = []
        self.recovered_history = []
        self.vaccinated_history = []
        version_path = Path(__file__).resolve().parents[2] / 'version.md'
        self.version = version_path.read_text().splitlines()[0] if version_path.exists() else "dev"

    def set_model_parameters(self, **kwargs):
        self.config = SVEIRCONFIG.from_dict(kwargs)

        self.steering_parameters = self.config.steering_parameters.__dict__

        self.model_dir = self.root_path / Path(self._model_identifier)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.steering_parameters['npath'] = str(self.model_dir / Path(self.config.steering_parameters.npath).name)
        self.steering_parameters['epath'] = str(self.model_dir / Path(self.config.steering_parameters.epath).name)

        self.save_model_parameters(overwrite=True)

    def _load_policy_library(self):
        policy_path = Path(self.config.policy_library_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy library file not found at '{policy_path}'.")

        data = np.load(policy_path, allow_pickle=True)
        self.agent_personas = torch.from_numpy(data['agent_personas']).float()
        self.risk_levels_tensor = torch.from_numpy(data['infection_risk_levels']).float().to(self.config.device)
        
        self.policy_library = {}
        for persona_id in range(self.config.num_agent_personas):
            key = f"policies_{persona_id}"
            if key not in data:
                raise KeyError(f"Policy for persona ID {persona_id} not found...")
            self.policy_library[persona_id] = torch.from_numpy(data[key]).long().to(self.config.device)

    def initialize_model(self, restart=False, verbose=False):
        self._load_policy_library()
        self.inputs = None
        if isinstance(restart, bool) and restart:
            print(f'Loading model state from checkpoint: {self.model_dir}')
            self.inputs = _load_model(self.model_dir)
        elif isinstance(restart, tuple):
            milestone_dir = self.model_dir / f'milestone_{restart[0]}' if restart[1] == 0 else self.model_dir / f'milestone_{restart[0]}_{restart[1]}'
            print(f'Loading model state from milestone: {milestone_dir}')
            self.inputs = _load_model(milestone_dir)

        if self.inputs:
            self.graph = copy.deepcopy(self.inputs["graph"])
            generator.set_state(self.inputs["generator_state"])
            self.step_count = self.inputs["step_count"]
        else:
            torch.manual_seed(self.config.seed)
            if verbose:
                print(f"Model torch seed set to {self.config.seed}")

        self.create_network(verbose)
        if self.config.spatial:
            self.create_grid()
            self.place_agents()

        self.initialize_agent_properties()
        self.graph = self.graph.to(self.config.device)

        if verbose:
            print(f'{self.graph.number_of_nodes()} agents initialized on {self.graph.device} device')

        weight_update_sveir(self.graph, self.config.device, self.steering_parameters['proximity_decay_rate'], self.steering_parameters['truncation_weight'])
        self.generator_state = generator.get_state()

    def run(self, verbose=False):
        self.infection_incidence.clear()
        self.prevalence_history.clear()
        self.susceptible_history.clear()
        self.exposed_history.clear()
        self.recovered_history.clear()
        self.vaccinated_history.clear()
        self.step_first = self.step_count
        while self.step_count < self.config.step_target:
            self.step(verbose)

    def step(self, verbose):
        try:
            if verbose:
                print(f'performing step {self.step_count} of {self.config.step_target}')
            
            newly_infected, compartment_counts = sveir_step(
                self.graph, self.config.device, self.step_count, self.steering_parameters,
                self.grid_environment, self.policy_library, self.risk_levels_tensor
            )
            
            self.infection_incidence.append(newly_infected)
            self.prevalence_history.append(compartment_counts['I'])
            self.susceptible_history.append(compartment_counts['S'])
            self.exposed_history.append(compartment_counts['E'])
            self.recovered_history.append(compartment_counts['R'])
            self.vaccinated_history.append(compartment_counts['V'])

        except Exception as e:
            raise RuntimeError(f'Execution of step failed for step {self.step_count}') from e

        self.step_count += 1

    def get_total_infections(self) -> int:
        if "num_infections" in self.graph.ndata:
            return torch.sum(self.graph.ndata["num_infections"]).item()
        return 0
    
    def get_proportion_infected_at_least_once(self) -> float:
        """
        Calculates the proportion of the total population that was infected
        at least one time during the simulation.
        """
        if "num_infections" not in self.graph.ndata:
            return 0.0

        num_infections_per_agent = self.graph.ndata["num_infections"]

        # Create a boolean mask of agents with 1 or more infections
        infected_mask = num_infections_per_agent > 0
        
        # Count the number of unique agents who were infected
        num_unique_infected = torch.sum(infected_mask).item()
        
        total_agents = self.graph.num_nodes()
        if total_agents == 0:
            return 0.0
            
        return num_unique_infected / total_agents

    def get_time_series_data(self) -> dict:
        return {
            "incidence": self.infection_incidence, "prevalence": self.prevalence_history,
            "susceptible": self.susceptible_history, "exposed": self.exposed_history,
            "recovered": self.recovered_history, "vaccinated": self.vaccinated_history,
        }

    def get_final_agent_states(self) -> dict:
        """
        Returns the final health and wealth vectors for all agents as numpy arrays.
        """
        if "health" not in self.graph.ndata or "wealth" not in self.graph.ndata:
            return {'health': np.array([]), 'wealth': np.array([])}

        return {
            'health': self.graph.ndata['health'].cpu().numpy(),
            'wealth': self.graph.ndata['wealth'].cpu().numpy()
        }

    def get_final_infection_counts(self) -> np.ndarray:
        """
        Returns the final number of infections for all agents as a numpy array.
        """
        if "num_infections" not in self.graph.ndata:
            return np.array([])
        
        return self.graph.ndata['num_infections'].cpu().numpy()
    
    def get_agent_personas(self) -> np.ndarray:
        """
        Returns the persona ID for all agents as a numpy array.
        """
        if "persona_id" not in self.graph.ndata:
            return np.array([])
        
        return self.graph.ndata['persona_id'].cpu().numpy()

    def get_initial_wealth(self) -> np.ndarray:
        """
        Returns the initial wealth for all agents as a numpy array.
        """
        if "initial_wealth" not in self.graph.ndata:
            return np.array([])

        return self.graph.ndata['initial_wealth'].cpu().numpy()

    def get_initial_health(self) -> np.ndarray:
        """
        Returns the initial wealth for all agents as a numpy array.
        """
        if "initial_wealth" not in self.graph.ndata:
            return np.array([])

        return self.graph.ndata['initial_health'].cpu().numpy()

    def create_network(self, verbose):
        self.graph = network_creation(
            self.config.number_agents, self.config.initial_graph_type, verbose,
            **self.config.initial_graph_args.__dict__
        )

    def create_grid(self):
        """
        Creates the grid environment by either loading a pre-computed realistic
        grid or generating a new one based on config.
        """
        grid_params = self.config.spatial_creation_args
        
        if grid_params.method == "realistic_import":
            if not grid_params.grid_id:
                raise ValueError("A 'grid_id' must be provided in the configuration to load a realistic grid.")
            
            grid_path = os.path.join("grids", grid_params.grid_id, "grid.npz")
            
            if not Path(grid_path).exists():
                raise FileNotFoundError(
                    f"Realistic grid file for ID '{grid_params.grid_id}' not found at '{grid_path}'.\n"
                    "Please run the 'create-grid' stage first."
                )
            
            data = np.load(grid_path, allow_pickle=True)            
            self.grid_tensor = data['grid']
            self.grid_bounds = data['bounds']
            # The loaded property_map is a 0-d array, get the item
            self.property_to_index = {v: k for k, v in data['property_map'].item().items()}
            
            # Create a simplified grid_environment object for compatibility
            self.grid_environment = type('Grid', (), {})()
            self.grid_environment.grid_tensor = torch.from_numpy(self.grid_tensor)
            self.grid_environment.property_to_index = self.property_to_index
            
        else:
            # Fallback to original method if not using realistic import
            self.grid_environment = grid_creation(**grid_params.__dict__)

    def place_agents(self):
        """
        Places agents onto the grid. If using the realistic grid, it handles
        distributing agents according to the residential mask.
        """
        self.graph.ndata['x'] = torch.zeros(self.graph.num_nodes()).float()
        self.graph.ndata['y'] = torch.zeros(self.graph.num_nodes()).float()

        if self.config.spatial_creation_args.method == "realistic_import":
            num_agents = self.config.number_agents
            
            # Layer 0 is the residential mask
            residence_mask = self.grid_tensor[:, :, self.property_to_index['residences']]
            valid_cells = np.argwhere(residence_mask == 1)
            
            if len(valid_cells) == 0:
                raise ValueError("The loaded grid has no valid cells for placing agents.")

            # Distribute agents among valid cells
            allocations = np.random.multinomial(num_agents, np.ones(len(valid_cells)) / len(valid_cells))
            
            agent_coords = []
            minx, miny, maxx, maxy = self.grid_bounds
            x_step = (maxx - minx) / self.grid_tensor.shape[1]
            y_step = (maxy - miny) / self.grid_tensor.shape[0]

            for i, count in enumerate(allocations):
                if count > 0:
                    r, c = valid_cells[i]
                    # Get center coordinates of the cell
                    cell_x_center = minx + (c + 0.5) * x_step
                    cell_y_center = miny + (r + 0.5) * y_step
                    # Add agents for this cell
                    for _ in range(count):
                        agent_coords.append((cell_x_center, cell_y_center))
            
            # Assign coordinates to graph nodes
            coords_tensor = torch.tensor(agent_coords, dtype=torch.float)
            # Shuffle to randomize which agent gets which coordinate
            coords_tensor = coords_tensor[torch.randperm(num_agents)]

            self.graph.ndata['x'] = coords_tensor[:, 0]
            self.graph.ndata['y'] = coords_tensor[:, 1]

        else:
            # Fallback to original method
            grid_assignment(self.graph, self.grid_environment, **self.config.spatial_assignment_args.__dict__)

    def initialize_agent_properties(self):
        agent_properties = {}
        num_agents = self.graph.num_nodes()

        # Assign personas and their corresponding CPT parameters
        persona_ids = torch.randint(0, self.config.num_agent_personas, (num_agents,))
        assigned_personas = self.agent_personas[persona_ids]
        agent_properties["persona_id"] = persona_ids
        agent_properties["alpha"] = assigned_personas[:, 0]
        agent_properties["gamma"] = assigned_personas[:, 1]
        agent_properties["omega"] = assigned_personas[:, 2]
        agent_properties["eta"]   = assigned_personas[:, 3]

        # Initialize state and location properties
        agent_properties["num_infections"] = self._initialize_agent_num_infections()
        agent_properties["compartments"] = self._initialize_agents_compartment()
        agent_properties["exposure_time"] = self._initialize_agents_exposure_time()
        agent_properties["time_use"] = self._initialize_agents_time_use()
        agent_properties["home_location"] = self._initialize_agents_home_location()
        agent_properties["school_location"] = self._find_nearest_locations(agent_properties["home_location"], "school")
        agent_properties["worship_location"] = self._find_nearest_locations(agent_properties["home_location"], "place_of_worship")
        agent_properties["water_location"] = self._find_nearest_locations(agent_properties["home_location"], "water")
        agent_properties["activity_choice"] = self._initialize_agents_activity_choice()
        agent_properties["wealth"] = self._initialize_agents_wealth(min_val=1, max_val=self.config.steering_parameters.max_state_value)
        agent_properties["health"] = self._initialize_agents_health(min_val=1, max_val=self.config.steering_parameters.max_state_value)
        agent_properties["initial_wealth"] = agent_properties["wealth"].clone()
        agent_properties["initial_health"] = agent_properties["health"].clone()

        for key, value in agent_properties.items():
            self.graph.ndata[key] = value

    def _initialize_agent_num_infections(self):
        return torch.zeros(self.graph.num_nodes(), dtype=torch.int)

    def _initialize_agents_compartment(self):
        proportion = self.steering_parameters["initial_infected_proportion"]
        if not 0 <= proportion <= 1.0: 
            raise ValueError("Initial infected proportion must be between 0 and 1.")
        num_infected = round(self.graph.num_nodes() * proportion)
        tensor = torch.zeros(self.graph.num_nodes(), dtype=torch.int)
        indices = torch.randperm(self.graph.num_nodes())[:num_infected]
        tensor[indices] = 3
        return tensor
    
    def _initialize_agents_exposure_time(self):
        return torch.zeros(self.graph.num_nodes(), dtype=torch.int)

    def _initialize_agents_time_use(self):
        tensor = torch.rand(self.graph.num_nodes(), 5)
        return tensor / tensor.sum(dim=1, keepdim=True)
    
    def _initialize_agents_home_location(self):
        tensor = torch.zeros((self.graph.num_nodes(), 2), dtype=torch.int)
        tensor[:, 0] = self.graph.ndata["y"]
        tensor[:, 1] = self.graph.ndata["x"]
        return tensor.float()

    def _find_nearest_locations(self, home_locations, property_name):
        property_grid = self.grid_environment.grid_tensor[:, :, self.grid_environment.property_to_index[property_name]]
        property_locations = torch.stack(torch.where(property_grid == 1)).T.float()
        
        if property_locations.shape[0] == 0:
            print(f"No grid locations found for '{property_name}'. Defaulting to home.")
            return home_locations

        distances = torch.cdist(home_locations, property_locations)
        return property_locations[torch.argmin(distances, dim=1)]

    def _initialize_agents_activity_choice(self):
        return torch.zeros(self.graph.num_nodes(), dtype=torch.int)

    def _initialize_agents_wealth(self, min_val, max_val):
        return torch.randint(min_val, max_val + 1, (self.graph.num_nodes(),), dtype=torch.int)
    
    def _initialize_agents_health(self, min_val, max_val):
        return torch.randint(min_val, max_val + 1, (self.graph.num_nodes(),), dtype=torch.int)

def _make_path_unique(path_str, extension=''):
    path = Path(path_str)
    if not path.with_suffix(extension).exists():
        return path.with_suffix(extension)
    
    instance = 1
    while True:
        new_path = path.parent / f"{path.name}_{instance}{extension}"
        if not new_path.exists():
            return new_path
        instance += 1

def _save_model(path, inputs):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    graph_labels = {'step_count': torch.tensor([inputs["step_count"]])}
    save_graphs(str(path / "graph.bin"), inputs["graph"], graph_labels)
    with open(path / "generator_state.bin", 'wb') as f:
        pickle.dump([inputs["generator_state"], inputs["step_count"]], f)
    with open(path / "process_version.md", 'w') as f:
        f.writelines([f'{inputs["process_version"]}\n', f'step={inputs["step_count"]}\n'])

def _load_model(path):
    path = Path(path)
    graph, graph_labels = load_graphs(str(path / "graph.bin"))
    with open(path / "generator_state.bin", 'rb') as f:
        generator, generator_step = pickle.load(f)
    with open(path / "process_version.md") as f:
        process_version = f.readlines()[0].strip()

    graph_step = graph_labels['step_count'].item()
    if graph_step != generator_step:
        raise ValueError('Step count mismatch in saved model files.')

    print(f'Loading model state from step {generator_step}.')
    return {'graph': graph[0], 'generator_state': generator, 'step_count': generator_step, 'process_version': process_version}
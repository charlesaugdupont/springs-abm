"""This module contains the model class and functions to initialize the model."""
import copy
import pickle
import numpy as np
import os
from pathlib import Path
import torch

from abm.agentInteraction.weight_update import weight_update_sveir
from config import SVEIRCONFIG
from abm.model.step import sveir_step
from abm.network.network_creation import network_creation
from abm.environment.grid_creation import grid_creation
from abm.constants import Activity

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
        # Ensure dir exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(cfg_filename)

    def create_network(self):
        raise NotImplementedError
    def step(self):
        raise NotImplementedError
    def run(self):
        raise NotImplementedError

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

    def set_model_parameters(self, **kwargs):
        self.config = SVEIRCONFIG.from_dict(kwargs)
        self.steering_parameters = self.config.steering_parameters.__dict__
        self.model_dir = self.root_path / Path(self._model_identifier)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Update paths
        self.steering_parameters['npath'] = str(self.model_dir / Path(self.config.steering_parameters.npath).name)
        self.steering_parameters['epath'] = str(self.model_dir / Path(self.config.steering_parameters.epath).name)
        self.save_model_parameters(overwrite=True)

    def _load_policy_library(self):
        # (Logic remains identical to original)
        policy_path = Path(self.config.policy_library_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy library file not found at '{policy_path}'.")
        data = np.load(policy_path, allow_pickle=True)
        self.agent_personas = torch.from_numpy(data['agent_personas']).float()
        self.risk_levels_tensor = torch.from_numpy(data['infection_risk_levels']).float().to(self.config.device)
        self.policy_library = {}
        for persona_id in range(self.config.num_agent_personas):
            self.policy_library[persona_id] = torch.from_numpy(data[f"policies_{persona_id}"]).long().to(self.config.device)

    def initialize_model(self, restart=False, verbose=False):
        self._load_policy_library()
        self.inputs = None
        if isinstance(restart, bool) and restart:
            print(f'Loading model state from checkpoint: {self.model_dir}')
            self.inputs = _load_model(self.model_dir)
        elif isinstance(restart, tuple):
            milestone_dir = self.model_dir / f'milestone_{restart[0]}' if restart[1] == 0 else self.model_dir / f'milestone_{restart[0]}_{restart[1]}'
            self.inputs = _load_model(milestone_dir)

        if self.inputs:
            # Deepcopy handling for custom object
            self.graph = copy.deepcopy(self.inputs["graph"])
            generator.set_state(self.inputs["generator_state"])
            self.step_count = self.inputs["step_count"]
        else:
            torch.manual_seed(self.config.seed)

            _, is_child = self._calculate_demographics_arrays(self.config.number_agents)
            adult_indices = (~is_child).nonzero(as_tuple=True)[0]

            self.graph = network_creation(
                self.config.number_agents, 
                self.config.initial_graph_type, 
                verbose,
                active_indices=adult_indices,  # <--- Pass adults here
                **self.config.initial_graph_args.__dict__
            )

            if self.config.spatial:
                self.create_grid()
                self.place_agents()

            self.initialize_agent_properties()

            # Move AgentGraph to device
            self.graph = self.graph.to(self.config.device)

            if verbose:
                print(f'{self.graph.num_nodes()} agents initialized on {self.config.device} device')

            weight_update_sveir(self.graph, self.config.device, self.steering_parameters['proximity_decay_rate'], self.steering_parameters['truncation_weight'])
            self.generator_state = generator.get_state()

    def _calculate_demographics_arrays(self, num_agents):
        """Calculates household IDs with VARIABLE sizes and Child status."""
        avg_size = float(self.config.average_household_size)

        # 1. Generate variable household sizes
        # We sample enough households to definitely cover all agents
        # Poisson(avg-1) + 1 ensures min size is 1 and mean is avg
        estimated_num_households = int((num_agents / avg_size) * 1.5) 

        # Sample sizes: Lambda is (Average - 1) because Poisson starts at 0
        sizes = torch.poisson(torch.full((estimated_num_households,), avg_size - 1.0)) + 1.0
        sizes = sizes.int()

        # 2. Create IDs based on these sizes
        # repeat_interleave repeats index 'i' by 'sizes[i]' times e.g., sizes=[3, 2] -> ids=[0,0,0, 1,1]
        full_ids = torch.repeat_interleave(torch.arange(len(sizes)), sizes)

        # Trim to exact number of agents
        household_ids = full_ids[:num_agents]

        # 3. Determine Child Status (Same logic as before)
        is_child = torch.zeros(num_agents, dtype=torch.bool)
        unique_households = torch.unique(household_ids)

        for hid in unique_households:
            members = (household_ids == hid).nonzero(as_tuple=True)[0]
            # Household Head (First member) is always Adult
            # Others are probabilistically children
            if len(members) > 1:
                others = members[1:]
                random_draws = torch.rand(len(others))
                is_child[others] = (random_draws < self.config.child_probability)

        return household_ids, is_child

    def run(self, verbose=False):
        # (Identical to original)
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
        if "num_infections" not in self.graph.ndata:
            return 0.0
        num_infections_per_agent = self.graph.ndata["num_infections"]
        infected_mask = num_infections_per_agent > 0
        num_unique_infected = torch.sum(infected_mask).item()
        total_agents = self.graph.num_nodes()
        if total_agents == 0: return 0.0
        return num_unique_infected / total_agents

    def get_time_series_data(self) -> dict:
        return {
            "incidence": self.infection_incidence, "prevalence": self.prevalence_history,
            "susceptible": self.susceptible_history, "exposed": self.exposed_history,
            "recovered": self.recovered_history, "vaccinated": self.vaccinated_history,
        }

    def get_final_agent_states(self) -> dict:
        if "health" not in self.graph.ndata or "wealth" not in self.graph.ndata:
            return {'health': np.array([]), 'wealth': np.array([])}
        return {
            'health': self.graph.ndata['health'].cpu().numpy(),
            'wealth': self.graph.ndata['wealth'].cpu().numpy()
        }

    def get_final_infection_counts(self) -> np.ndarray:
        if "num_infections" not in self.graph.ndata: return np.array([])
        return self.graph.ndata['num_infections'].cpu().numpy()
    
    def get_agent_personas(self) -> np.ndarray:
        if "persona_id" not in self.graph.ndata: return np.array([])
        return self.graph.ndata['persona_id'].cpu().numpy()

    def get_initial_wealth(self) -> np.ndarray:
        if "initial_wealth" not in self.graph.ndata: return np.array([])
        return self.graph.ndata['initial_wealth'].cpu().numpy()

    def get_initial_health(self) -> np.ndarray:
        if "initial_wealth" not in self.graph.ndata: return np.array([])
        return self.graph.ndata['initial_health'].cpu().numpy()

    def create_grid(self):
        grid_params = self.config.spatial_creation_args
        if grid_params.method == "realistic_import":
            if not grid_params.grid_id:
                raise ValueError("Grid ID required.")
            grid_path = os.path.join("grids", grid_params.grid_id, "grid.npz")
            if not Path(grid_path).exists():
                raise FileNotFoundError(f"Realistic grid file for ID '{grid_params.grid_id}' not found.")
            data = np.load(grid_path, allow_pickle=True)            
            self.grid_tensor = data['grid']
            self.grid_bounds = data['bounds']
            self.property_to_index = {v: k for k, v in data['property_map'].item().items()}
            self.grid_environment = type('Grid', (), {})()
            self.grid_environment.grid_tensor = torch.from_numpy(self.grid_tensor)
            self.grid_environment.property_to_index = self.property_to_index
        else:
            self.grid_environment = grid_creation(**grid_params.__dict__)

    def place_agents(self):
        self.graph.ndata['x'] = torch.zeros(self.graph.num_nodes()).float()
        self.graph.ndata['y'] = torch.zeros(self.graph.num_nodes()).float()

        household_ids = self.graph.ndata["household_id"]
        unique_households = torch.unique(household_ids)
        num_households = len(unique_households)

        if self.config.spatial_creation_args.method == "realistic_import":
            # Identify valid residence cells
            residence_mask = self.grid_tensor[:, :, self.property_to_index['residences']]
            valid_cells = np.argwhere(residence_mask == 1)
            
            if len(valid_cells) == 0: 
                raise ValueError("No valid residence cells.")
            
            # Allocate households to cells, not individual agents
            allocations = np.random.multinomial(num_households, np.ones(len(valid_cells)) / len(valid_cells))
            minx, miny, maxx, maxy = self.grid_bounds
            x_step = (maxx - minx) / self.grid_tensor.shape[1]
            y_step = (maxy - miny) / self.grid_tensor.shape[0]
            
            household_centers = {}
            current_hh_idx = 0
            
            # Determine (x,y) for each household ID
            for i, count in enumerate(allocations):
                if count > 0:
                    r, c = valid_cells[i]
                    cell_x_center = minx + (c + 0.5) * x_step
                    cell_y_center = miny + (r + 0.5) * y_step
                    for _ in range(count):
                        hh_id = unique_households[current_hh_idx].item()
                        household_centers[hh_id] = (cell_x_center, cell_y_center)
                        current_hh_idx += 1
            
            # Assign coordinates to agents based on their household ID
            # This is a bit slow as a loop, can be vectorized if needed, but safe for <10k agents
            x_coords = torch.zeros(self.graph.num_nodes())
            y_coords = torch.zeros(self.graph.num_nodes())
            for hh_id, (hx, hy) in household_centers.items():
                mask = (household_ids == hh_id)
                x_coords[mask] = hx
                y_coords[mask] = hy
                
            self.graph.ndata['x'] = x_coords
            self.graph.ndata['y'] = y_coords

        else:
            # Fallback for synthetic grids (random assignment)
            # Create random positions for households
            hh_x = torch.randint(0, self.grid_environment.grid_shape[0], (num_households,)).float()
            hh_y = torch.randint(0, self.grid_environment.grid_shape[1], (num_households,)).float()
            
            # Map back to agents
            # Note: This assumes household_ids are contiguous 0..N, which they are from _assign_demographics
            self.graph.ndata['x'] = hh_x[household_ids]
            self.graph.ndata['y'] = hh_y[household_ids]

    def initialize_agent_properties(self):
        num_agents = self.graph.num_nodes()
        agent_properties = {}
        persona_ids = torch.randint(0, self.config.num_agent_personas, (num_agents,))
        assigned_personas = self.agent_personas[persona_ids]

        # behavioural profile
        agent_properties["persona_id"] = persona_ids
        agent_properties["alpha"] = assigned_personas[:, 0]
        agent_properties["gamma"] = assigned_personas[:, 1]
        agent_properties["omega"] = assigned_personas[:, 2]
        agent_properties["eta"]   = assigned_personas[:, 3]

        # demographics
        agent_properties["household_id"] = self.graph.ndata['household_id']
        agent_properties["is_child"] = self.graph.ndata['is_child'].long()

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

        # Update AgentGraph ndata
        for key, value in agent_properties.items():
            self.graph.ndata[key] = value

    def _initialize_agent_num_infections(self):
        return torch.zeros(self.graph.num_nodes(), dtype=torch.int)

    def _initialize_agents_compartment(self):
        proportion = self.steering_parameters["initial_infected_proportion"]
        num_infected = round(self.graph.num_nodes() * proportion)
        tensor = torch.zeros(self.graph.num_nodes(), dtype=torch.int)
        indices = torch.randperm(self.graph.num_nodes())[:num_infected]
        tensor[indices] = 3
        return tensor

    def _initialize_agents_exposure_time(self):
        return torch.zeros(self.graph.num_nodes(), dtype=torch.int)

    def _initialize_agents_time_use(self):
        """
        Creates time use schedules.
        Activities: 0:Home, 1:School, 2:Worship, 3:Water, 4:Social
        """
        num_agents = self.graph.num_nodes()
        is_child = self.graph.ndata['is_child']

        # Initialize tensor
        time_use = torch.zeros((num_agents, 5))

        # --- ADULT SCHEDULE ---
        # Adults split time between Home, Worship, Water, Social. (No School)
        # Weights: Home(40), School(0), Worship(10), Water(30), Social(20)
        adult_weights = torch.tensor([40.0, 0.0, 10.0, 30.0, 20.0])

        # Add some random noise to adults so they aren't robots
        adult_noise = torch.rand((~is_child).sum(), 5) * 10.0
        # Ensure School (idx 1) remains 0 for adults
        base_adult = adult_weights.expand((~is_child).sum(), 5).clone()
        base_adult += adult_noise
        base_adult[:, Activity.SCHOOL] = 0.0 

        time_use[~is_child] = base_adult
  
        # --- CHILD SCHEDULE ---
        # Children split time between Home, School, Social. (No Water, No Worship independently)
        # Weights: Home(40), School(40), Worship(0), Water(0), Social(20)
        child_weights = torch.tensor([80.0, 20.0, 0.0, 0.0, 0.0])
 
        child_noise = torch.rand(is_child.sum(), 5) * 5.0
        base_child = child_weights.expand(is_child.sum(), 5).clone()
        base_child += child_noise
        base_child[:, Activity.WORSHIP] = 0.0   # No Worship
        base_child[:, Activity.WATER] = 0.0     # No Water
        base_child[:, Activity.SOCIAL] = 0.0    # No Social Visits

        time_use[is_child] = base_child

        # Normalize to probabilities (sum to 1)
        # Add epsilon to avoid division by zero if noise makes everything 0
        return time_use / (time_use.sum(dim=1, keepdim=True) + 1e-6)

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

def _save_model(path, inputs):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Save the AgentGraph object using torch.save or pickle
    with open(path / "graph.pkl", 'wb') as f:
        pickle.dump(inputs["graph"], f)
    with open(path / "generator_state.bin", 'wb') as f:
        pickle.dump([inputs["generator_state"], inputs["step_count"]], f)
    with open(path / "process_version.md", 'w') as f:
        f.writelines([f'{inputs["process_version"]}\n', f'step={inputs["step_count"]}\n'])

def _load_model(path):
    path = Path(path)
    # Load AgentGraph
    with open(path / "graph.pkl", 'rb') as f:
        graph = pickle.load(f)
    with open(path / "generator_state.bin", 'rb') as f:
        generator, generator_step = pickle.load(f)
    with open(path / "process_version.md") as f:
        process_version = f.readlines()[0].strip()

    # step_count isn't stored in graph labels anymore, assume generator_step is truth
    print(f'Loading model state from step {generator_step}.')
    return {'graph': graph, 'generator_state': generator, 'step_count': generator_step, 'process_version': process_version}
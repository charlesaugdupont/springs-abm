# abm/model/initialize_model.py
"""This module contains the primary SVEIRModel class."""
from pathlib import Path
import torch
import numpy as np
from scipy.stats import qmc

from config import SVEIRConfig, RotavirusConfig, CampylobacterConfig
from abm.model.step import sveir_step
from abm.network.network_creation import network_creation
from abm.constants import AgentPropertyKeys, EdgePropertyKeys
from abm.factories.agent_factory import AgentFactory
from abm.factories.environment_factory import EnvironmentFactory
from abm.pathogens.rotavirus import Rotavirus
from abm.pathogens.campylobacter import Campylobacter

class Model:
    """Abstract base class for a model."""
    def __init__(self, model_identifier: str, root_path: str = '.'):
        self._model_identifier = model_identifier
        self.root_path = Path(root_path)
        self.model_dir = self.root_path / self._model_identifier
        self.step_count = 0
        self.config: SVEIRConfig | None = None

    def save_model_parameters(self):
        """Saves the model's configuration to a YAML file."""
        if self.config is None: return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cfg_filename = self.model_dir / f"{self._model_identifier}.yaml"
        self.config.to_yaml(cfg_filename)

class SVEIRModel(Model):
    """The main class for the SVEIR agent-based model."""
    def __init__(self, *, model_identifier: str, root_path: str = '.'):
        super().__init__(model_identifier=model_identifier, root_path=root_path)
        self.steering_parameters = None
        self.graph = None
        self.infection_incidence = []
        self.prevalence_history = []
        self.pathogens = []
        self.systems = []
        self.grid_environment = None
        self.agent_personas = None

    def set_model_parameters(self, **kwargs):
        """Sets model parameters from a dictionary, overriding defaults."""
        self.config = SVEIRConfig.from_dict(kwargs)
        self.steering_parameters = self.config.steering_parameters
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.steering_parameters.npath = str(self.model_dir / Path(self.steering_parameters.npath).name)
        self.steering_parameters.epath = str(self.model_dir / Path(self.steering_parameters.epath).name)
        self.save_model_parameters()
    
    def _load_agent_personas(self):
        """Generates agent behavioral personas using LH sampling."""
        sampler = qmc.LatinHypercube(d=3, seed=self.config.seed)
        samples = sampler.random(n=self.config.num_agent_personas)
        ranges = [self.config.alpha_range, self.config.gamma_range, self.config.lambda_range]
        scaled_samples = qmc.scale(samples, [r[0] for r in ranges], [r[1] for r in ranges])
        self.agent_personas = torch.from_numpy(scaled_samples).float()

    def initialize_model(self, verbose: bool = False):
        """Initializes the entire model state, including agents, network, and environment."""
        if self.config is None:
            raise RuntimeError("Model parameters have not been set. Call set_model_parameters() first.")

        torch.manual_seed(self.config.seed)
        self._load_agent_personas()

        # 1. Create Social Network
        household_ids, is_child = self._calculate_demographics(self.config.number_agents)
        adult_indices = (~is_child).nonzero(as_tuple=True)[0]
        self.graph = network_creation(
            self.config.number_agents, self.config.initial_graph_type, verbose,
            device=self.config.device, active_indices=adult_indices,
            **self.config.initial_graph_args.model_dump()
        )
        self.graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID] = household_ids
        self.graph.ndata[AgentPropertyKeys.IS_CHILD] = is_child

        # 2. Create Environment and Place Agents
        if self.config.spatial:
            env_factory = EnvironmentFactory(self.config.spatial_creation_args)
            env_factory.create_grid()
            env_factory.place_agents(self.graph)
            self.grid_environment = env_factory.grid_environment

        # 3. Initialize Agent Properties
        agent_factory = AgentFactory(self.config, self.agent_personas)
        agent_factory.initialize_agent_properties(self.graph, self.grid_environment)

        # 4. Finalize Graph and Initialize Systems
        self.graph.to(self.config.device)

        if self.graph.num_edges() > 0:
            self.graph.edata[EdgePropertyKeys.WEIGHT] = torch.ones(
                self.graph.num_edges(), device=self.config.device, dtype=torch.float
            )   
     
        self._initialize_pathogens_and_systems()

        if verbose:
            print(f'{self.graph.num_nodes()} agents initialized on {self.config.device}')

    def _initialize_pathogens_and_systems(self):
        """Instantiates the pathogen and system objects that drive the simulation."""
        from abm.systems.movement import MovementSystem
        from abm.systems.environment import EnvironmentSystem
        from abm.systems.child_illness import ChildIllnessSystem
        from abm.systems.care_seeking import CareSeekingSystem
        from abm.systems.economics import EconomicSystem
        from abm.systems.household import HouseholdSystem

        pathogen_map = {"rota": Rotavirus, "campy": Campylobacter}
        pathogen_config_map = {"rota": RotavirusConfig, "campy": CampylobacterConfig}

        for p_config in self.config.pathogens:
            if p_config.name in pathogen_map:
                pathogen_class = pathogen_map[p_config.name]
                # Ensure the config object is the specific subclass type
                typed_config = pathogen_config_map[p_config.name](**p_config.model_dump())
                self.pathogens.append(pathogen_class(typed_config, self.steering_parameters, self.config.device))
            else:
                print(f"Warning: Pathogen '{p_config.name}' not implemented.")

        self.systems = [
            MovementSystem(self.config),
            ChildIllnessSystem(self.config),
            CareSeekingSystem(self.config),
            EnvironmentSystem(self.config),
            HouseholdSystem(self.config),
            EconomicSystem(self.config),
        ]

    def _calculate_demographics(self, num_agents: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates household IDs and child status for all agents."""
        avg_size = float(self.config.average_household_size)
        est_num_hh = int((num_agents / avg_size) * 1.5)
        sizes = torch.poisson(torch.full((est_num_hh,), avg_size - 1.0)) + 1.0
        full_ids = torch.repeat_interleave(torch.arange(len(sizes)), sizes.long())
        household_ids = full_ids[:num_agents]

        is_child = torch.zeros(num_agents, dtype=torch.bool)
        for hid in torch.unique(household_ids):
            members = (household_ids == hid).nonzero(as_tuple=True)[0]
            if len(members) > 1:
                others = members[1:]
                is_child[others] = (torch.rand(len(others)) < self.config.child_probability)
        return household_ids, is_child

    def run(self, verbose: bool = False):
        """Runs the simulation from the current step_count to the step_target."""
        self.infection_incidence.clear()
        self.prevalence_history.clear()
        while self.step_count < self.config.step_target:
            self.step(verbose)

    def step(self, verbose: bool = False):
        """Executes one full timestep of the simulation."""
        if verbose:
            print(f'Performing step {self.step_count} of {self.config.step_target}')
        try:
            new_cases_by_pathogen, compartment_counts = sveir_step(
                self.graph, self.step_count, self.config,
                self.grid_environment, self.pathogens, self.systems
            )
            total_new_cases = sum(new_cases_by_pathogen.values())
            total_prevalence = sum(v for k, v in compartment_counts.items() if k.endswith("_I"))
            self.infection_incidence.append(total_new_cases)
            self.prevalence_history.append(total_prevalence)
        except Exception as e:
            raise RuntimeError(f'Execution of step {self.step_count} failed.') from e
        self.step_count += 1

    # --- Result-gathering methods ---
    def get_total_infections(self) -> int:
        return sum(torch.sum(self.graph.ndata[AgentPropertyKeys.num_infections(p.name)]).item() for p in self.pathogens)

    def get_proportion_infected_at_least_once(self) -> float:
        if not self.pathogens: return 0.0
        infected_masks = [(self.graph.ndata[AgentPropertyKeys.num_infections(p.name)] > 0) for p in self.pathogens]
        combined_mask = torch.stack(infected_masks).any(dim=0)
        return torch.sum(combined_mask).item() / self.graph.num_nodes()

    def get_time_series_data(self) -> dict:
        return {"incidence": self.infection_incidence, "prevalence": self.prevalence_history}

    def get_final_agent_states(self) -> dict:
        return {
            'health': self.graph.ndata[AgentPropertyKeys.HEALTH].cpu().numpy(),
            'wealth': self.graph.ndata[AgentPropertyKeys.WEALTH].cpu().numpy()
        }
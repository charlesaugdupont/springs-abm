# abm/factories/agent_factory.py
from typing import Dict
import torch

from abm.state import AgentState
from abm.constants import Activity, AgentPropertyKeys, Compartment, GridLayer
from config import SVEIRConfig

class AgentFactory:
    """A class to handle the initialization of agent properties."""

    def __init__(self, config: SVEIRConfig, agent_personas: torch.Tensor):
        self.config = config
        self.agent_personas = agent_personas.to(config.device)

    def initialize_agent_properties(self, agent_state: AgentState, grid_env):
        """
        Populates the agent graph with all necessary initial properties.
        """
        num_agents = agent_state.num_nodes()
        agent_properties: Dict[str, torch.Tensor] = {}

        # --- Behavioral Profile ---
        persona_ids = torch.randint(0, self.config.num_agent_personas, (num_agents,))
        assigned_personas = self.agent_personas[persona_ids]
        agent_properties[AgentPropertyKeys.PERSONA_ID] = persona_ids

        # CPT parameters from persona (alpha, gamma, lambda)
        agent_properties[AgentPropertyKeys.ALPHA] = assigned_personas[:, 0]
        agent_properties[AgentPropertyKeys.GAMMA] = assigned_personas[:, 1]
        agent_properties[AgentPropertyKeys.LAMBDA] = assigned_personas[:, 2]

        # Globally defined behavioral parameters
        agent_properties[AgentPropertyKeys.ETA] = torch.full((num_agents,), self.config.steering_parameters.eta)
        agent_properties[AgentPropertyKeys.THETA] = torch.full((num_agents,), self.config.steering_parameters.eta)

        # --- Demographics ---
        agent_properties[AgentPropertyKeys.HOUSEHOLD_ID] = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD].bool()
        num_children = is_child.sum()

        # Default age for adults (e.g., 30 years in months)
        ages = torch.full((num_agents,), 30 * 12.0)

        # Assign a uniform random age from 0 to 5 years for all children (in month units)
        child_ages = torch.rand(num_children) * (5 * 12.0)
        ages[is_child] = child_ages
        agent_properties[AgentPropertyKeys.AGE] = ages

        agent_properties[AgentPropertyKeys.IS_CHILD] = is_child
        # Assign IS_PARENT status to the first adult in any household with children
        is_parent = torch.zeros_like(is_child)
        for hh_id in torch.unique(agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]):
            in_hh = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID] == hh_id
            if torch.any(in_hh & is_child):
                 # Find first adult in that household and make them a parent
                 adults_in_hh = (in_hh & ~is_child).nonzero(as_tuple=True)[0]
                 if len(adults_in_hh) > 0:
                     is_parent[adults_in_hh[0]] = True
        agent_properties[AgentPropertyKeys.IS_PARENT] = is_parent

        # --- Disease States (Initialized for all pathogens in config) ---
        for pathogen_conf in self.config.pathogens:
            p_name = pathogen_conf.name
            p_prop = pathogen_conf.initial_infected_proportion
            agent_properties[AgentPropertyKeys.status(p_name)] = self._initialize_compartment(num_agents, p_prop)
            agent_properties[AgentPropertyKeys.exposure_time(p_name)] = torch.zeros(num_agents, dtype=torch.int)
            agent_properties[AgentPropertyKeys.num_infections(p_name)] = torch.zeros(num_agents, dtype=torch.int)

        # --- Location and Activity ---
        agent_properties[AgentPropertyKeys.TIME_USE] = self._initialize_time_use(num_agents, agent_state.ndata[AgentPropertyKeys.IS_CHILD])
        home_loc = self._initialize_home_location(agent_state)
        agent_properties[AgentPropertyKeys.HOME_LOCATION] = home_loc
        agent_properties[AgentPropertyKeys.SCHOOL_LOCATION] = self._find_nearest_locations(home_loc, GridLayer.SCHOOL, grid_env)
        agent_properties[AgentPropertyKeys.WORSHIP_LOCATION] = self._find_nearest_locations(home_loc, GridLayer.WORSHIP, grid_env)
        agent_properties[AgentPropertyKeys.WATER_LOCATION] = self._find_nearest_locations(home_loc, GridLayer.WATER, grid_env)
        agent_properties[AgentPropertyKeys.ACTIVITY_CHOICE] = torch.zeros(num_agents, dtype=torch.int)

        # --- Health and Wealth ---
        wealth = torch.rand(num_agents, dtype=torch.float)
        health = torch.rand(num_agents, dtype=torch.float)
        agent_properties[AgentPropertyKeys.WEALTH] = wealth
        agent_properties[AgentPropertyKeys.HEALTH] = health
        agent_properties[AgentPropertyKeys.INITIAL_WEALTH] = wealth.clone()
        agent_properties[AgentPropertyKeys.INITIAL_HEALTH] = health.clone()

        # --- Illness State ---
        agent_properties[AgentPropertyKeys.SYMPTOM_SEVERITY] = torch.zeros(num_agents, dtype=torch.float)
        agent_properties[AgentPropertyKeys.ILLNESS_DURATION] = torch.zeros(num_agents, dtype=torch.int)

        # --- Assign all properties to the graph ---
        for key, value in agent_properties.items():
            agent_state.ndata[key] = value.to(self.config.device)

    def _initialize_compartment(self, num_agents: int, proportion: float) -> torch.Tensor:
        num_infected = round(num_agents * proportion)
        tensor = torch.zeros(num_agents, dtype=torch.int)
        if num_infected > 0:
            indices = torch.randperm(num_agents)[:num_infected]
            tensor[indices] = Compartment.INFECTIOUS
        return tensor

    def _initialize_time_use(self, num_agents: int, is_child: torch.Tensor) -> torch.Tensor:
        time_use = torch.zeros((num_agents, 5))
        num_adults = (~is_child).sum()
        num_children = is_child.sum()

        if num_adults > 0:
            adult_weights = torch.tensor([40.0, 0.0, 10.0, 30.0, 20.0])
            adult_noise = torch.rand(num_adults, 5) * 10.0
            base_adult = adult_weights.expand(num_adults, 5).clone() + adult_noise
            base_adult[:, Activity.SCHOOL] = 0.0
            time_use[~is_child] = base_adult

        if num_children > 0:
            child_weights = torch.tensor([80.0, 20.0, 0.0, 0.0, 0.0])
            child_noise = torch.rand(num_children, 5) * 5.0
            base_child = child_weights.expand(num_children, 5).clone() + child_noise
            base_child[:, [Activity.WORSHIP, Activity.WATER, Activity.SOCIAL]] = 0.0
            time_use[is_child] = base_child

        return time_use / (time_use.sum(dim=1, keepdim=True) + 1e-9)

    def _initialize_home_location(self, agent_state: AgentState) -> torch.Tensor:
        tensor = torch.zeros((agent_state.num_nodes(), 2), dtype=torch.float)
        # Note: Grid is (row, col) which corresponds to (y, x)
        tensor[:, 0] = agent_state.ndata[AgentPropertyKeys.Y]
        tensor[:, 1] = agent_state.ndata[AgentPropertyKeys.X]
        return tensor

    def _find_nearest_locations(self, home_locations, property_name, grid_env) -> torch.Tensor:
        prop_idx = grid_env.property_to_index.get(property_name)
        if prop_idx is None:
            print(f"Warning: Property '{property_name}' not in grid. Defaulting to home location.")
            return home_locations

        property_grid = grid_env.grid_tensor[:, :, prop_idx]
        property_locations = torch.stack(torch.where(property_grid == 1)).T.float().to(self.config.device)

        if property_locations.shape[0] == 0:
            print(f"Warning: No grid locations found for '{property_name}'. Defaulting to home location.")
            return home_locations

        # Ensure home_locations is on the correct device for cdist
        home_locations_dev = home_locations.to(self.config.device)
        distances = torch.cdist(home_locations_dev, property_locations)
        return property_locations[torch.argmin(distances, dim=1)]
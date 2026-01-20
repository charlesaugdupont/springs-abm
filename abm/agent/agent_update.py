# abm/agent/agent_update.py

import torch
from .health_cpt_utils import (
    utility,
    compute_new_wealth,
    compute_health_delta,
    compute_health_decline,
    compute_health_cost
)
from abm.constants import Compartment, Activity

def sveir_agent_update(method, agent_graph, params=None, num_nodes=None, edge_weights=None, grid=None, adjacency=None, policy=None, risk_levels=None):
    """
    Dispatcher function that calls the appropriate agent update logic.
    """
    # Define a mapping from method names to functions
    update_functions = {
        "exposure_increment": (_agent_increment_exposure_time, [agent_graph]),
        "health_investment": (_agent_health_investment_vectorized, [agent_graph, params, policy, risk_levels]),
        "exposed_to_infectious": (_agent_exposed_to_infectious, [agent_graph, params]),
        "infectious_to_recovered": (_agent_infectious_to_recovered, [agent_graph, params]),
        "susceptible_to_vaccinated": (_agent_susceptible_to_vaccinated, [agent_graph, params, num_nodes]),
        "susceptible_to_exposed": (_agent_susceptible_to_exposed, [agent_graph, params, adjacency]),
        "vaccinated_to_exposed": (_agent_vaccinated_to_exposed, [agent_graph, params, adjacency]),
        "move": (_agent_move, [agent_graph, edge_weights]),
        "human_to_water_transmission": (_agent_human_to_water_transmission, [agent_graph, params, grid]),
        "water_to_human_transmission": (_agent_water_to_human_transmission, [agent_graph, params, grid]),
        "water_recovery": (_water_recovery, [params, grid]),
        "shock": (_water_shock, [params, grid]),
    }

    if method in update_functions:
        func, args = update_functions[method]
        # Only exposed_to_infectious needs to return a value (new infection count)
        if method == "exposed_to_infectious":
             return func(*args)
        else:
            filtered_args = [arg for arg in args if arg is not None]
            func(*filtered_args)
    else:
        raise ValueError(f"Unknown agent update method: {method}")

def _agent_increment_exposure_time(agent_graph):
    """Increments the exposure time for all agents currently in the 'Exposed' compartment."""
    exposed_mask = agent_graph.ndata["compartments"] == Compartment.EXPOSED
    agent_graph.ndata["exposure_time"][exposed_mask] += 1

def _agent_health_investment_vectorized(agent_graph, params, policy_library, risk_levels):
    """
    Vectorized function for agents to decide on health investment.    
    """
    num_agents = agent_graph.num_nodes()
    wealth = agent_graph.ndata["wealth"].clone().long()
    health = agent_graph.ndata["health"].clone().long()
    persona_ids = agent_graph.ndata["persona_id"].long()

    current_prob = params['infection_probability']
    risk_level_index = torch.argmin(torch.abs(risk_levels - current_prob))

    decisions = torch.zeros(num_agents, dtype=torch.long, device=agent_graph.device)
    for i in range(num_agents):
        pid = persona_ids[i].item()
        agent_policy = policy_library[pid][risk_level_index]
        decisions[i] = agent_policy[wealth[i]-1, health[i]-1]

    invest_mask = (decisions == 1)
    
    # --- Calculate potential changes and costs ---
    cost_params = {'cost_subsidy_factor': params['cost_subsidy_factor'], 'efficacy_multiplier': params['efficacy_multiplier']}
    delta_params = {'efficacy_multiplier': params['efficacy_multiplier']}

    investment_cost = compute_health_cost(health, cost_params).int()
    health_change = compute_health_delta(health, delta_params).int()
    
    # Calculate health decline separately
    health_decline = compute_health_decline(health).int()

    can_afford_mask = (wealth >= investment_cost)
    
    new_wealth = wealth.clone().int()
    new_health = health.clone().int()

    # --- Update agents who INVEST ---
    invest_and_can_afford = invest_mask & can_afford_mask
    if torch.any(invest_and_can_afford):
        new_wealth[invest_and_can_afford] -= investment_cost[invest_and_can_afford]
        
        prob_increase = torch.rand(torch.sum(invest_and_can_afford), device=agent_graph.device)
        success_increase_mask = prob_increase < params["P_H_increase"]

        health_to_update = new_health[invest_and_can_afford]
        health_to_update[success_increase_mask] += health_change[invest_and_can_afford][success_increase_mask]
        new_health[invest_and_can_afford] = health_to_update

    # --- Update agents who could NOT afford to invest (or chose not to) ---
    save_mask = ~invest_mask
    invest_but_cant_afford = invest_mask & ~can_afford_mask
    
    decrease_candidates = save_mask | invest_but_cant_afford
    if torch.any(decrease_candidates):
        prob_decrease = torch.rand(torch.sum(decrease_candidates), device=agent_graph.device)
        success_decrease_mask = prob_decrease < params["P_H_decrease"]
        
        health_to_update = new_health[decrease_candidates]
        
        health_to_update[success_decrease_mask] -= health_decline[decrease_candidates][success_decrease_mask]
        new_health[decrease_candidates] = health_to_update
        
    new_health.clamp_(min=1, max=params["max_state_value"])

    # --- Final wealth update based on new utility ---
    current_utility = utility(new_wealth, new_health, agent_graph.ndata["alpha"])
    updated_wealth = compute_new_wealth(new_wealth, params["wealth_update_A"], current_utility)

    agent_graph.ndata["wealth"] = updated_wealth.clamp(min=1, max=params['max_state_value'])
    agent_graph.ndata["health"] = new_health

def _agent_exposed_to_infectious(agent_graph, params):
    """Transitions agents from 'Exposed' to 'Infectious' and returns the count."""
    exposed_to_infections_mask = (agent_graph.ndata["compartments"] == Compartment.EXPOSED) & \
                                 (agent_graph.ndata["exposure_time"] >= params["exposure_period"])

    if torch.any(exposed_to_infections_mask):
        agent_graph.ndata["compartments"][exposed_to_infections_mask] = Compartment.INFECTIOUS
        agent_graph.ndata["num_infections"][exposed_to_infections_mask] += 1
        return torch.sum(exposed_to_infections_mask).item()

    return 0 # Return 0 if no new infections

def _agent_infectious_to_recovered(agent_graph, params):
    """Transitions agents from 'Infectious' to 'Recovered' based on a recovery probability."""
    infectious_mask = agent_graph.ndata["compartments"] == Compartment.INFECTIOUS
    if not torch.any(infectious_mask):
        return
        
    recovery_chance = torch.rand(torch.sum(infectious_mask), device=agent_graph.device)
    recovered_mask = recovery_chance < params["recovery_rate"]
    
    agents_to_recover = infectious_mask.nonzero(as_tuple=True)[0][recovered_mask]
    agent_graph.ndata["compartments"][agents_to_recover] = Compartment.RECOVERED

def _agent_susceptible_to_vaccinated(agent_graph, params):
    """Transitions agents from 'Susceptible' to 'Vaccinated' based on a vaccination probability."""
    susceptible_mask = agent_graph.ndata["compartments"] == Compartment.SUSCEPTIBLE
    if not torch.any(susceptible_mask):
        return

    vaccination_chance = torch.rand(torch.sum(susceptible_mask), device=agent_graph.device)
    vaccinated_mask = vaccination_chance < params["vaccination_rate"]
    
    agents_to_vaccinate = susceptible_mask.nonzero(as_tuple=True)[0][vaccinated_mask]
    agent_graph.ndata["compartments"][agents_to_vaccinate] = Compartment.VACCINATED

def _calculate_and_apply_new_infections(agent_graph, params, target_nodes_mask, adjacency, base_prob_multiplier=1.0):
    """
    Helper function to handle infection logic for any group of agents.
    """
    if not torch.any(target_nodes_mask):
        return

    target_nodes_indices = target_nodes_mask.nonzero(as_tuple=True)[0]
    
    # Create a mask for infectious agents to find infectious neighbors
    is_infectious_mask = agent_graph.ndata["compartments"] == Compartment.INFECTIOUS
    
    # Calculate infection pressure from infectious agents at the same location
    infection_pressure = torch.matmul(adjacency[target_nodes_indices].float(), is_infectious_mask.float())

    if torch.sum(infection_pressure) == 0:
        return # No one is exposed to an infectious agent

    # Base infection probability, reduced by prior infections and health
    num_infections = agent_graph.ndata["num_infections"][target_nodes_indices].float()
    health = agent_graph.ndata["health"][target_nodes_indices].int()
    
    prob_infection_base = params["infection_probability"] * torch.exp(-params["prior_infection_immunity_factor"] * num_infections)
    health_susceptibility = torch.exp(-params["infection_reduction_factor_per_health_unit"] * (health - 1.0))
    
    final_prob = base_prob_multiplier * prob_infection_base * health_susceptibility
    final_prob = torch.clamp(final_prob, 0.0, 1.0)

    # The probability of NOT getting infected is (1-p)^k, where k is infection_pressure
    prob_not_infected = (1 - final_prob) ** infection_pressure
    prob_getting_infected = 1 - prob_not_infected

    # Determine new infections
    random_samples = torch.rand(len(target_nodes_indices), device=agent_graph.device)
    newly_infected_mask = random_samples < prob_getting_infected

    infected_nodes_indices = target_nodes_indices[newly_infected_mask]

    if len(infected_nodes_indices) > 0:
        agent_graph.ndata["compartments"][infected_nodes_indices] = Compartment.EXPOSED
        agent_graph.ndata["exposure_time"][infected_nodes_indices] = 0

def _agent_susceptible_to_exposed(agent_graph, params, adjacency):
    """Transitions 'Susceptible' agents to 'Exposed' based on proximity to 'Infectious' agents."""
    susceptible_mask = (agent_graph.ndata["compartments"] == Compartment.SUSCEPTIBLE)
    _calculate_and_apply_new_infections(agent_graph, params, susceptible_mask, adjacency, base_prob_multiplier=1.0)

def _agent_vaccinated_to_exposed(agent_graph, params, adjacency):
    """Transitions 'Vaccinated' agents to 'Exposed' (breakthrough infection)."""
    vaccinated_mask = (agent_graph.ndata["compartments"] == Compartment.VACCINATED)
    breakthrough_multiplier = 1.0 - params["vaccine_efficacy"]
    _calculate_and_apply_new_infections(agent_graph, params, vaccinated_mask, adjacency, base_prob_multiplier=breakthrough_multiplier)

def _agent_move(agent_graph, edge_weights):
    """Moves agents to different locations based on their daily activity schedule."""
    random_activity = torch.multinomial(agent_graph.ndata["time_use"], num_samples=1).squeeze()
    agent_graph.ndata["activity_choice"] = random_activity

    # Location mapping: 0:home, 1:school, 2:worship, 3:water, 4:social
    location_map = {
        Activity.HOME: "home_location",
        Activity.SCHOOL: "school_location",
        Activity.WORSHIP: "worship_location",
        Activity.WATER: "water_location"
    }

    for activity_idx, location_key in location_map.items():
        mask = (random_activity == activity_idx)
        if torch.any(mask):
            agent_graph.ndata['x'][mask] = agent_graph.ndata[location_key][mask, 0]
            agent_graph.ndata['y'][mask] = agent_graph.ndata[location_key][mask, 1]

    # Handle social visits separately as they are more complex
    social_mask = (random_activity == Activity.SOCIAL)
    if torch.any(social_mask):
        visiting_agents = social_mask.nonzero(as_tuple=True)[0]

        # 1. Identify potential hosts (must be at home)
        is_at_home_mask = (random_activity == Activity.HOME)

        # 2. Get weights
        # Note: edge_weights comes from agent_graph.edges().
        # Since we only created edges between adults, this matrix 
        # naturally only contains Adult->Adult connections.
        social_weights = edge_weights[visiting_agents][:, is_at_home_mask]

        # 3. Selection
        can_visit_mask = social_weights.sum(dim=1) > 0
        if torch.any(can_visit_mask):
            active_visitors = visiting_agents[can_visit_mask]
            active_weights = social_weights[can_visit_mask]

            # Normalize
            active_weights /= active_weights.sum(dim=1, keepdim=True)

            # Choose Host
            hosts_at_home_indices = is_at_home_mask.nonzero(as_tuple=True)[0]
            chosen_local_idx = torch.multinomial(active_weights, num_samples=1).squeeze()
            visited_host_indices = hosts_at_home_indices[chosen_local_idx]

            # Teleport
            agent_graph.ndata['x'][active_visitors] = agent_graph.ndata["home_location"][visited_host_indices, 0]
            agent_graph.ndata['y'][active_visitors] = agent_graph.ndata["home_location"][visited_host_indices, 1]

def _agent_water_to_human_transmission(agent_graph, params, grid):
    """Handles infection of agents from contaminated water sources."""
    water_idx = grid.property_to_index['water']
    water_slice = grid.grid_tensor[:, :, water_idx]
    infected_water_coords = torch.stack(torch.where(water_slice == 2)).T
    if infected_water_coords.shape[0] == 0:
        return  # No contaminated water sources

    # Find agents at contaminated water locations
    agent_coords = torch.stack((agent_graph.ndata["x"], agent_graph.ndata["y"])).T
    at_infected_source_mask = (agent_coords.unsqueeze(1) == infected_water_coords.unsqueeze(0)).all(dim=-1).any(dim=1)

    if not torch.any(at_infected_source_mask):
        return

    # Handle Susceptible/Recovered agents
    s_r_mask = ((agent_graph.ndata["compartments"] == Compartment.SUSCEPTIBLE) | (agent_graph.ndata["compartments"] == Compartment.RECOVERED)) & at_infected_source_mask
    _calculate_and_apply_new_infections(agent_graph, params, s_r_mask, torch.eye(agent_graph.num_nodes()), base_prob_multiplier=params["water_to_human_infection_prob"])

    # Handle Vaccinated agents
    v_mask = (agent_graph.ndata["compartments"] == Compartment.VACCINATED) & at_infected_source_mask
    breakthrough_multiplier = (1.0 - params["vaccine_efficacy"]) * params["water_to_human_infection_prob"]
    _calculate_and_apply_new_infections(agent_graph, params, v_mask, torch.eye(agent_graph.num_nodes()), base_prob_multiplier=breakthrough_multiplier)

def _agent_human_to_water_transmission(agent_graph, params, grid):
    """Handles contamination of water sources by infectious agents."""
    water_idx = grid.property_to_index['water']
    water_slice = grid.grid_tensor[:, :, water_idx]

    # Find infectious agents who are at a water source
    infectious_mask = agent_graph.ndata["compartments"] == Compartment.INFECTIOUS
    
    # Retrieve the activity choice directly from the graph state
    random_activity = agent_graph.ndata["activity_choice"]
    
    at_water_source_mask = random_activity == Activity.WATER
    contaminator_mask = infectious_mask & at_water_source_mask

    if not torch.any(contaminator_mask):
        return

    # Probabilistic contamination
    contamination_chance = torch.rand(torch.sum(contaminator_mask), device=agent_graph.device)
    contamination_success = contamination_chance < params["human_to_water_infection_prob"]

    successful_contaminators = contaminator_mask.nonzero(as_tuple=True)[0][contamination_success]

    if len(successful_contaminators) > 0:
        # Get the unique locations of water points to be contaminated
        water_points_to_infect = torch.unique(agent_graph.ndata["water_location"][successful_contaminators], dim=0).int()
        if water_points_to_infect.shape[0] > 0:
            water_slice[water_points_to_infect[:, 0], water_points_to_infect[:, 1]] = 2 # Mark as contaminated

def _water_recovery(params, grid):
    """Handles the random recovery of contaminated water sources."""
    water_idx = grid.property_to_index['water']
    water_slice = grid.grid_tensor[:, :, water_idx]
    infected_water_mask = water_slice == 2
    if not torch.any(infected_water_mask):
        return

    recovery_chance = torch.rand(torch.sum(infected_water_mask), device=water_slice.device)
    recovery_success = recovery_chance < params["water_recovery_prob"]

    coords_to_recover = infected_water_mask.nonzero(as_tuple=True)
    recovered_coords = (coords_to_recover[0][recovery_success], coords_to_recover[1][recovery_success])
    
    if len(recovered_coords[0]) > 0:
        water_slice[recovered_coords] = 1 # Mark as clean

def _water_shock(params, grid):
    """Applies a cyclical shock that contaminates clean water sources."""
    water_idx = grid.property_to_index['water']
    water_slice = grid.grid_tensor[:, :, water_idx]
    clean_water_mask = water_slice == 1
    if not torch.any(clean_water_mask):
        return

    shock_chance = torch.rand(torch.sum(clean_water_mask), device=water_slice.device)
    shock_success = shock_chance < params["shock_infection_prob"]

    coords_to_shock = clean_water_mask.nonzero(as_tuple=True)
    shocked_coords = (coords_to_shock[0][shock_success], coords_to_shock[1][shock_success])

    if len(shocked_coords[0]) > 0:
        water_slice[shocked_coords] = 2 # Mark as contaminated
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

def sveir_agent_update(method, agent_graph, **kwargs):
    """
    Dispatcher function that calls the appropriate agent update logic.
    Arguments are passed via kwargs to specific functions.
    """
    update_functions = {
        # Generic Transitions
        "exposure_increment": _agent_increment_exposure_time,
        "exposed_to_infectious": _agent_exposed_to_infectious,
        "infectious_to_recovered": _agent_infectious_to_recovered,
        
        # H2H Transmission
        "susceptible_to_exposed": _agent_susceptible_to_exposed,
        "vaccinated_to_exposed": _agent_vaccinated_to_exposed,
        
        # Zoonotic Transmission
        "animal_to_human_transmission": _agent_animal_to_human_transmission,
        
        # Water Transmission
        "human_to_water_transmission": _agent_human_to_water_transmission,
        "water_to_human_transmission": _agent_water_to_human_transmission,
        
        # Other
        "health_investment": _agent_health_investment_vectorized,
        "susceptible_to_vaccinated": _agent_susceptible_to_vaccinated,
        "move": _agent_move,
        "water_recovery": _water_recovery,
        "shock": _water_shock,
    }

    if method in update_functions:
        
        # Explicit argument mapping for clarity and safety
        if method == "exposure_increment":
            _agent_increment_exposure_time(agent_graph, kwargs['pathogen'])
            
        elif method == "exposed_to_infectious":
            return _agent_exposed_to_infectious(agent_graph, kwargs['params'], kwargs['pathogen'])
            
        elif method == "infectious_to_recovered":
            _agent_infectious_to_recovered(agent_graph, kwargs['params'], kwargs['pathogen'])
            
        elif method == "susceptible_to_exposed":
            _agent_susceptible_to_exposed(
                agent_graph, kwargs['global_params'], kwargs['pathogen'], 
                kwargs['infection_prob'], kwargs['adjacency']
            )
            
        elif method == "vaccinated_to_exposed":
            _agent_vaccinated_to_exposed(
                agent_graph, kwargs['params'], kwargs['global_params'],
                kwargs['pathogen'], kwargs['infection_prob'], kwargs['adjacency']
            )
            
        elif method == "animal_to_human_transmission":
            _agent_animal_to_human_transmission(agent_graph, kwargs['params'], kwargs['grid'])
            
        elif method == "human_to_water_transmission":
             _agent_human_to_water_transmission(agent_graph, kwargs['params'], kwargs['grid'], kwargs['pathogen'])
             
        elif method == "water_to_human_transmission":
             _agent_water_to_human_transmission(agent_graph, kwargs['params'], kwargs['grid'], kwargs['pathogen'])
             
        elif method == "health_investment":
            _agent_health_investment_vectorized(
                agent_graph, kwargs['params'], kwargs['policy'], kwargs['risk_levels']
            )
            
        elif method == "susceptible_to_vaccinated":
            _agent_susceptible_to_vaccinated(agent_graph, kwargs['params'], kwargs['pathogen'])
            
        elif method == "move":
            _agent_move(agent_graph, kwargs['edge_weights'])
            
        elif method == "water_recovery":
            _water_recovery(kwargs['params'], kwargs['grid'])
            
        elif method == "shock":
            _water_shock(kwargs['params'], kwargs['grid'])

    else:
        raise ValueError(f"Unknown agent update method: {method}")

def _agent_increment_exposure_time(agent_graph, pathogen):
    """Increments the exposure time for agents in the 'Exposed' state."""
    status_key = f"status_{pathogen}"
    timer_key = f"exposure_time_{pathogen}"
    exposed_mask = agent_graph.ndata[status_key] == Compartment.EXPOSED
    agent_graph.ndata[timer_key][exposed_mask] += 1

def _agent_exposed_to_infectious(agent_graph, params, pathogen):
    """Transitions agents from 'Exposed' to 'Infectious'."""
    status_key = f"status_{pathogen}"
    timer_key = f"exposure_time_{pathogen}"
    count_key = f"num_infections_{pathogen}"

    exposed_to_infections_mask = (agent_graph.ndata[status_key] == Compartment.EXPOSED) & \
                                 (agent_graph.ndata[timer_key] >= params.exposure_period)

    if torch.any(exposed_to_infections_mask):
        agent_graph.ndata[status_key][exposed_to_infections_mask] = Compartment.INFECTIOUS
        agent_graph.ndata[count_key][exposed_to_infections_mask] += 1
        return torch.sum(exposed_to_infections_mask).item()
    return 0

def _agent_infectious_to_recovered(agent_graph, params, pathogen):
    """Transitions agents from 'Infectious' to 'Recovered'."""
    status_key = f"status_{pathogen}"
    infectious_mask = agent_graph.ndata[status_key] == Compartment.INFECTIOUS
    if not torch.any(infectious_mask):
        return
        
    recovery_chance = torch.rand(torch.sum(infectious_mask), device=agent_graph.device)
    recovered_mask = recovery_chance < params.recovery_rate
    
    agents_to_recover = infectious_mask.nonzero(as_tuple=True)[0][recovered_mask]
    agent_graph.ndata[status_key][agents_to_recover] = Compartment.RECOVERED

def _calculate_and_apply_new_infections(agent_graph, global_params, pathogen, infection_prob, target_nodes_mask, adjacency, base_prob_multiplier=1.0):
    """Helper function to handle infection logic."""
    if not torch.any(target_nodes_mask):
        return

    status_key = f"status_{pathogen}"
    count_key = f"num_infections_{pathogen}"

    target_nodes_indices = target_nodes_mask.nonzero(as_tuple=True)[0]
    
    is_infectious_mask = agent_graph.ndata[status_key] == Compartment.INFECTIOUS
    infection_pressure = torch.matmul(adjacency[target_nodes_indices].float(), is_infectious_mask.float())

    if torch.sum(infection_pressure) == 0:
        return

    num_infections = agent_graph.ndata[count_key][target_nodes_indices].float()
    health = agent_graph.ndata["health"][target_nodes_indices].int()
    
    immunity_factor = torch.exp(-global_params["prior_infection_immunity_factor"] * num_infections)
    health_factor = torch.exp(-global_params["infection_reduction_factor_per_health_unit"] * (health - 1.0))
    
    final_prob = base_prob_multiplier * infection_prob * immunity_factor * health_factor
    final_prob = torch.clamp(final_prob, 0.0, 1.0)

    prob_not_infected = (1 - final_prob) ** infection_pressure
    prob_getting_infected = 1 - prob_not_infected

    random_samples = torch.rand(len(target_nodes_indices), device=agent_graph.device)
    newly_infected_mask = random_samples < prob_getting_infected

    infected_nodes_indices = target_nodes_indices[newly_infected_mask]

    if len(infected_nodes_indices) > 0:
        agent_graph.ndata[status_key][infected_nodes_indices] = Compartment.EXPOSED
        agent_graph.ndata[f"exposure_time_{pathogen}"][infected_nodes_indices] = 0

def _agent_susceptible_to_exposed(agent_graph, global_params, pathogen, infection_prob, adjacency):
    status_key = f"status_{pathogen}"
    susceptible_mask = (agent_graph.ndata[status_key] == Compartment.SUSCEPTIBLE)
    _calculate_and_apply_new_infections(agent_graph, global_params, pathogen, infection_prob, susceptible_mask, adjacency, 1.0)

def _agent_vaccinated_to_exposed(agent_graph, params, global_params, pathogen, infection_prob, adjacency):
    status_key = f"status_{pathogen}"
    vaccinated_mask = (agent_graph.ndata[status_key] == Compartment.VACCINATED)
    breakthrough_multiplier = 1.0 - params.vaccine_efficacy
    _calculate_and_apply_new_infections(agent_graph, global_params, pathogen, infection_prob, vaccinated_mask, adjacency, breakthrough_multiplier)

def _agent_animal_to_human_transmission(agent_graph, params, grid):
    """
    Beta-Poisson infection from animal density layer.
    """
    animal_idx = grid.property_to_index.get('animal_density')
    if animal_idx is None:
        return 

    x = agent_graph.ndata['x'].long().clamp(0, grid.grid_shape[0]-1)
    y = agent_graph.ndata['y'].long().clamp(0, grid.grid_shape[1]-1)
    
    local_density = grid.grid_tensor[x, y, animal_idx]
    dose = local_density * params.human_animal_interaction_rate
    
    alpha = params.beta_poisson_alpha
    beta = params.beta_poisson_beta
    
    prob_infection = 1.0 - (1.0 + dose / beta).pow(-alpha)
    
    status_key = "status_campy"
    susceptible_mask = agent_graph.ndata[status_key] == Compartment.SUSCEPTIBLE
    
    rand_vals = torch.rand(agent_graph.num_nodes(), device=agent_graph.device)
    new_infections = (rand_vals < prob_infection) & susceptible_mask
    
    if torch.any(new_infections):
        agent_graph.ndata[status_key][new_infections] = Compartment.EXPOSED
        agent_graph.ndata["exposure_time_campy"][new_infections] = 0

def _agent_human_to_water_transmission(agent_graph, params, grid, pathogen):
    # Only Rotavirus has explicit water params in current config structure
    # To be safe, we check if the params object has the attribute
    if not hasattr(params, 'human_to_water_infection_prob'):
        return

    water_idx = grid.property_to_index['water']
    water_slice = grid.grid_tensor[:, :, water_idx]

    status_key = f"status_{pathogen}"
    infectious_mask = agent_graph.ndata[status_key] == Compartment.INFECTIOUS
    random_activity = agent_graph.ndata["activity_choice"]
    
    at_water_source_mask = random_activity == Activity.WATER
    contaminator_mask = infectious_mask & at_water_source_mask

    if not torch.any(contaminator_mask):
        return

    # Note: params here is Global dict because water params are currently Top-Level in config
    # We will assume 'params' passed here is the Dictionary from step.py for water funcs
    prob = params['human_to_water_infection_prob']

    contamination_chance = torch.rand(torch.sum(contaminator_mask), device=agent_graph.device)
    contamination_success = contamination_chance < prob

    successful_contaminators = contaminator_mask.nonzero(as_tuple=True)[0][contamination_success]

    if len(successful_contaminators) > 0:
        water_points_to_infect = torch.unique(agent_graph.ndata["water_location"][successful_contaminators], dim=0).int()
        if water_points_to_infect.shape[0] > 0:
            water_slice[water_points_to_infect[:, 0], water_points_to_infect[:, 1]] = 2 

def _agent_water_to_human_transmission(agent_graph, params, grid, pathogen):
    # Currently assumes parameters are in the Global Dict
    if 'water_to_human_infection_prob' not in params:
        return

    water_idx = grid.property_to_index['water']
    water_slice = grid.grid_tensor[:, :, water_idx]
    infected_water_coords = torch.stack(torch.where(water_slice == 2)).T
    if infected_water_coords.shape[0] == 0:
        return

    agent_coords = torch.stack((agent_graph.ndata["x"], agent_graph.ndata["y"])).T
    at_infected_source_mask = (agent_coords.unsqueeze(1) == infected_water_coords.unsqueeze(0)).all(dim=-1).any(dim=1)

    if not torch.any(at_infected_source_mask):
        return

    status_key = f"status_{pathogen}"
    s_r_mask = ((agent_graph.ndata[status_key] == Compartment.SUSCEPTIBLE) | \
                (agent_graph.ndata[status_key] == Compartment.RECOVERED)) & at_infected_source_mask

    prob = params["water_to_human_infection_prob"]
    _calculate_and_apply_new_infections(agent_graph, params, pathogen, prob, s_r_mask, torch.eye(agent_graph.num_nodes()), 1.0)

def _agent_susceptible_to_vaccinated(agent_graph, params, pathogen):
    status_key = f"status_{pathogen}"
    susceptible_mask = agent_graph.ndata[status_key] == Compartment.SUSCEPTIBLE
    if not torch.any(susceptible_mask):
        return

    vaccination_chance = torch.rand(torch.sum(susceptible_mask), device=agent_graph.device)
    vaccinated_mask = vaccination_chance < params.vaccination_rate
    
    agents_to_vaccinate = susceptible_mask.nonzero(as_tuple=True)[0][vaccinated_mask]
    agent_graph.ndata[status_key][agents_to_vaccinate] = Compartment.VACCINATED

def _agent_health_investment_vectorized(agent_graph, params, policy_library, risk_levels):
    # This remains largely unchanged but accesses GLOBAL params dict
    num_agents = agent_graph.num_nodes()
    wealth = agent_graph.ndata["wealth"].clone().long()
    health = agent_graph.ndata["health"].clone().long()
    persona_ids = agent_graph.ndata["persona_id"].long()

    # Note: Using Rota probability as proxy for general risk perception for now
    current_prob = params['infection_probability_proxy'] 
    risk_level_index = torch.argmin(torch.abs(risk_levels - current_prob))

    decisions = torch.zeros(num_agents, dtype=torch.long, device=agent_graph.device)
    for i in range(num_agents):
        pid = persona_ids[i].item()
        agent_policy = policy_library[pid][risk_level_index]
        decisions[i] = agent_policy[wealth[i]-1, health[i]-1]

    invest_mask = (decisions == 1)
    
    cost_params = {'cost_subsidy_factor': params['cost_subsidy_factor'], 'efficacy_multiplier': params['efficacy_multiplier']}
    delta_params = {'efficacy_multiplier': params['efficacy_multiplier']}

    investment_cost = compute_health_cost(health, cost_params).int()
    health_change = compute_health_delta(health, delta_params).int()
    health_decline = compute_health_decline(health).int()

    can_afford_mask = (wealth >= investment_cost)
    new_wealth = wealth.clone().int()
    new_health = health.clone().int()

    invest_and_can_afford = invest_mask & can_afford_mask
    if torch.any(invest_and_can_afford):
        new_wealth[invest_and_can_afford] -= investment_cost[invest_and_can_afford]
        prob_increase = torch.rand(torch.sum(invest_and_can_afford), device=agent_graph.device)
        success_increase_mask = prob_increase < params["P_H_increase"]
        health_to_update = new_health[invest_and_can_afford]
        health_to_update[success_increase_mask] += health_change[invest_and_can_afford][success_increase_mask]
        new_health[invest_and_can_afford] = health_to_update

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
    current_utility = utility(new_wealth, new_health, agent_graph.ndata["alpha"])
    updated_wealth = compute_new_wealth(new_wealth, params["wealth_update_A"], current_utility)

    agent_graph.ndata["wealth"] = updated_wealth.clamp(min=1, max=params['max_state_value'])
    agent_graph.ndata["health"] = new_health

def _agent_move(agent_graph, edge_weights):
    random_activity = torch.multinomial(agent_graph.ndata["time_use"], num_samples=1).squeeze()
    agent_graph.ndata["activity_choice"] = random_activity
    
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

    social_mask = (random_activity == Activity.SOCIAL)
    if torch.any(social_mask):
        visiting_agents = social_mask.nonzero(as_tuple=True)[0]
        is_at_home_mask = (random_activity == Activity.HOME)
        social_weights = edge_weights[visiting_agents][:, is_at_home_mask]

        can_visit_mask = social_weights.sum(dim=1) > 0
        if torch.any(can_visit_mask):
            active_visitors = visiting_agents[can_visit_mask]
            active_weights = social_weights[can_visit_mask]
            active_weights /= active_weights.sum(dim=1, keepdim=True)

            hosts_at_home_indices = is_at_home_mask.nonzero(as_tuple=True)[0]
            chosen_local_idx = torch.multinomial(active_weights, num_samples=1).squeeze()
            visited_host_indices = hosts_at_home_indices[chosen_local_idx]

            agent_graph.ndata['x'][active_visitors] = agent_graph.ndata["home_location"][visited_host_indices, 0]
            agent_graph.ndata['y'][active_visitors] = agent_graph.ndata["home_location"][visited_host_indices, 1]

def _water_recovery(params, grid):
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
        water_slice[recovered_coords] = 1 

def _water_shock(params, grid):
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
        water_slice[shocked_coords] = 2
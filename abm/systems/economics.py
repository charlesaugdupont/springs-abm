# abm/systems/economics.py
import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys

class EconomicSystem(System):
    def update(self, agent_state: AgentState, **kwargs):
        params = self.config.steering_parameters
        
        # --- 1. Calculate Income (No change here) ---
        income = torch.zeros(agent_state.num_nodes(), device=self.device)
        if params.daily_income_rate > 0:
            is_adult = ~agent_state.ndata[AgentPropertyKeys.IS_CHILD]
            if torch.any(is_adult):
                if params.health_based_income:
                    adult_health = agent_state.ndata[AgentPropertyKeys.HEALTH][is_adult]
                    income[is_adult] = params.daily_income_rate * adult_health
                else:
                    income[is_adult] = params.daily_income_rate
        
        # --- 2. Calculate Net Wealth Change (MODIFIED LOGIC) ---
        current_wealth = agent_state.ndata[AgentPropertyKeys.WEALTH]
        
        # Subtract cost of living for everyone
        cost_of_living = params.daily_cost_of_living
        
        # Calculate the net change and apply it
        net_change = income - cost_of_living
        new_wealth = torch.clamp(current_wealth + net_change, min=0.0, max=1.0)
        
        agent_state.ndata[AgentPropertyKeys.WEALTH] = new_wealth
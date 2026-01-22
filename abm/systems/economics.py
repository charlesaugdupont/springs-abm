# abm/systems/economics.py
import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys

class EconomicSystem(System):
    def update(self, agent_state: AgentState, **kwargs):
        params = self.config.steering_parameters
        if params.daily_income_rate <= 0:
            return

        is_adult = ~agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        if not torch.any(is_adult):
            return
        
        income = torch.zeros(agent_state.num_nodes(), device=self.device)

        if params.health_based_income:
            # Income is proportional to health for adults
            adult_health = agent_state.ndata[AgentPropertyKeys.HEALTH][is_adult]
            income[is_adult] = params.daily_income_rate * adult_health
        else:
            # All adults get a fixed income
            income[is_adult] = params.daily_income_rate
        
        # Add income and clamp wealth to the [0, 1] range
        current_wealth = agent_state.ndata[AgentPropertyKeys.WEALTH]
        new_wealth = torch.clamp(current_wealth + income, min=0.0, max=1.0)
        agent_state.ndata[AgentPropertyKeys.WEALTH] = new_wealth
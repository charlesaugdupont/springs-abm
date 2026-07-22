# abm/systems/economics.py
import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys


class EconomicSystem(System):
    """
    Updates agent wealth once per day.

    Wealth is a pooled HOUSEHOLD resource, not an independent per-agent
    balance: every member of a household shares the same WEALTH value.
    Each day, income earned by adults in a household (health-modulated, if
    configured) and the household's total cost of living (an equivalence-
    scale-style charge: adults in full, children discounted by
    child_cost_weight) are aggregated at the household level. The net
    change is applied to the shared balance, which is then re-broadcast to
    every member. This reflects money being shared within a household, and
    avoids children — who have no income of their own — silently
    accumulating individual debt that no one can act on.
    """

    def update(self, agent_state: AgentState, **kwargs):
        params = self.config.steering_parameters
        is_adult = ~agent_state.ndata[AgentPropertyKeys.IS_CHILD]

        # --- 1. Each adult's individual income contribution ---
        income = torch.zeros(agent_state.num_nodes(), device=self.device)
        if params.daily_income_rate > 0 and torch.any(is_adult):
            if params.health_based_income:
                adult_health = agent_state.ndata[AgentPropertyKeys.HEALTH][is_adult]
                income[is_adult] = params.daily_income_rate * adult_health
            else:
                income[is_adult] = params.daily_income_rate

        # --- 2. Aggregate income, cost of living, and current wealth per household ---
        household_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        unique_hh_ids, hh_index = torch.unique(household_ids, return_inverse=True)
        num_households = unique_hh_ids.numel()

        household_income = torch.zeros(num_households, device=self.device)
        household_income.index_add_(0, hh_index, income)

        household_size = torch.zeros(num_households, device=self.device)
        household_size.index_add_(0, hh_index, torch.ones_like(income))

        # Cost of living: adults charged in full, children discounted by
        # child_cost_weight (see config.py) - children contribute cost but
        # no income, and charging them a full adult share made any
        # household with a typical adult:child ratio structurally unable
        # to break even regardless of health or income.
        adult_count = torch.zeros(num_households, device=self.device)
        adult_count.index_add_(0, hh_index, is_adult.float())
        child_count = household_size - adult_count
        household_cost = (adult_count + params.child_cost_weight * child_count) * params.daily_cost_of_living

        # All members of a household share the same WEALTH value (the
        # invariant maintained here and by
        # abm.utils.household.apply_household_wealth_delta); averaging
        # recovers that shared value robustly.
        current_wealth = agent_state.ndata[AgentPropertyKeys.WEALTH]
        household_wealth_sum = torch.zeros(num_households, device=self.device)
        household_wealth_sum.index_add_(0, hh_index, current_wealth)
        household_wealth = household_wealth_sum / household_size

        net_change = household_income - household_cost
        new_household_wealth = torch.clamp(household_wealth + net_change, min=0.0, max=1.0)

        # --- 3. Broadcast the updated pooled wealth back to every member ---
        agent_state.ndata[AgentPropertyKeys.WEALTH] = new_household_wealth[hh_index]
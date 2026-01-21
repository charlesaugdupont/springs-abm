# abm/systems/behavior.py
import torch

from .system import System
from abm.state import AgentGraph
from abm.constants import AgentPropertyKeys
from abm.agent.health_cpt_utils import (
    utility,
    compute_new_wealth,
    compute_health_delta,
    compute_health_decline,
    compute_health_cost
)

class BehavioralSystem(System):
    """Handles agent decision-making regarding health investment."""

    def update(self, agent_graph: AgentGraph, **kwargs):
        """
        Agents decide whether to invest in health based on their pre-computed policy.
        Updates agent health and wealth accordingly.
        """
        policy_library = kwargs.get("policy_library")
        risk_levels = kwargs.get("risk_levels")
        current_risk_proxy = kwargs.get("current_risk_proxy")
        if policy_library is None or risk_levels is None or current_risk_proxy is None:
            raise ValueError("BehavioralSystem requires policy_library, risk_levels, and current_risk_proxy.")

        params = self.config.steering_parameters
        num_agents = agent_graph.num_nodes()

        # --- 1. Get Agent Decisions ---
        # Find the index of the risk level closest to the current proxy
        risk_level_index = torch.argmin(torch.abs(risk_levels - current_risk_proxy))

        decisions = self._get_decisions_from_policy(
            agent_graph, policy_library, risk_level_index
        )

        # --- 2. Calculate Potential Outcomes ---
        health = agent_graph.ndata[AgentPropertyKeys.HEALTH].clone()
        invest_mask = (decisions == 1)

        cost_params = {'cost_subsidy_factor': params.cost_subsidy_factor, 'efficacy_multiplier': params.efficacy_multiplier}
        delta_params = {'efficacy_multiplier': params.efficacy_multiplier}

        investment_cost = compute_health_cost(health, cost_params).int()
        health_gain = compute_health_delta(health, delta_params).int()
        health_loss = compute_health_decline(health).int()

        # --- 3. Apply Outcomes Based on Decisions ---
        can_afford_mask = (agent_graph.ndata[AgentPropertyKeys.WEALTH] >= investment_cost)
        new_wealth = agent_graph.ndata[AgentPropertyKeys.WEALTH].clone().int()
        new_health = health.clone().int()

        # Case A: Agent invests and can afford it
        invest_and_afford = invest_mask & can_afford_mask
        if torch.any(invest_and_afford):
            new_wealth[invest_and_afford] -= investment_cost[invest_and_afford]
            prob_increase = torch.rand(torch.sum(invest_and_afford), device=self.device)
            success_mask = prob_increase < params.P_H_increase
            new_health[invest_and_afford][success_mask] += health_gain[invest_and_afford][success_mask]

        # Case B: Agent saves (or invests but can't afford) -> potential decline
        decrease_candidates = ~invest_mask | (invest_mask & ~can_afford_mask)
        if torch.any(decrease_candidates):
            prob_decrease = torch.rand(torch.sum(decrease_candidates), device=self.device)
            success_mask = prob_decrease < params.P_H_decrease
            new_health[decrease_candidates][success_mask] -= health_loss[decrease_candidates][success_mask]

        # --- 4. Finalize and Update State ---
        new_health.clamp_(min=1, max=params.max_state_value)
        current_utility = utility(new_wealth, new_health, agent_graph.ndata[AgentPropertyKeys.ALPHA])
        updated_wealth = compute_new_wealth(new_wealth, params.wealth_update_A, current_utility)

        agent_graph.ndata[AgentPropertyKeys.WEALTH] = updated_wealth.clamp(min=1, max=params.max_state_value)
        agent_graph.ndata[AgentPropertyKeys.HEALTH] = new_health


    def _get_decisions_from_policy(self, agent_graph: AgentGraph,
                                   policy_library: dict, risk_idx: int) -> torch.Tensor:
        """Looks up the optimal decision for each agent from the policy library."""
        num_agents = agent_graph.num_nodes()
        decisions = torch.zeros(num_agents, dtype=torch.long, device=self.device)

        # Using tensors for indexing is faster than a Python loop
        persona_ids = agent_graph.ndata[AgentPropertyKeys.PERSONA_ID].long()
        wealth_indices = (agent_graph.ndata[AgentPropertyKeys.WEALTH].long() - 1).clamp(0, 99)
        health_indices = (agent_graph.ndata[AgentPropertyKeys.HEALTH].long() - 1).clamp(0, 99)

        # Group agents by persona to perform batched lookups
        for pid in torch.unique(persona_ids):
            agent_mask = (persona_ids == pid)
            agent_policy = policy_library[pid.item()][risk_idx]

            w_lookup = wealth_indices[agent_mask]
            h_lookup = health_indices[agent_mask]
            decisions[agent_mask] = agent_policy[w_lookup, h_lookup]

        return decisions
# abm/utils/household.py
"""Helpers for agent properties that are pooled/shared at the household level."""
import torch

from abm.state import AgentState
from abm.constants import AgentPropertyKeys


def apply_household_wealth_delta(
    agent_state: AgentState,
    member_idx: torch.Tensor,
    delta: float,
) -> None:
    """
    Applies `delta` to the pooled wealth of the household that `member_idx`
    belongs to, and propagates the resulting value to every member of that
    household.

    Wealth is modeled as a shared household resource (see EconomicSystem)
    rather than an individual balance, so any wealth-changing event
    experienced by one household member (e.g. paying for a child's care)
    draws on, and updates, the whole household's pooled wealth.
    """
    household_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
    hh_id = household_ids[member_idx]
    same_household_mask = household_ids == hh_id

    current_wealth = agent_state.ndata[AgentPropertyKeys.WEALTH][member_idx].item()
    new_wealth = min(1.0, max(0.0, current_wealth + delta))
    agent_state.ndata[AgentPropertyKeys.WEALTH][same_household_mask] = new_wealth
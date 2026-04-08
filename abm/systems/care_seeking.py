# abm/systems/care_seeking.py
"""
Parental care-seeking decisions via Cumulative Prospect Theory (CPT).

Decision frame
--------------
Each day, for each child with a non-zero severity, the parent stochastically
"notices" the child as sick with probability equal to the child's current
severity score. If noticed, the parent evaluates two prospects:

  Seek care
    Cost c is paid immediately (certain wealth loss).
    With probability p_success the child recovers faster (illness duration
    reduced by `duration_reduction_on_success` days), which the parent
    values as a future health gain for the child.
    With probability (1 - p_success) the cost is paid but the child's
    illness continues unchanged.

  Wait
    No immediate cost.
    With probability p_worsen the child's severity increases by
    `untreated_severity_penalty` and the parent suffers a stress health
    hit of `parent_stress_health_impact`.
    With probability (1 - p_worsen) nothing changes.

Both prospects are evaluated in utility space using a Cobb-Douglas utility
function U(w, h) = w^alpha * h^(1-alpha), with gains and losses measured
relative to the parent's current (reference) utility. The CPT value
function and Prelec probability weighting are applied to each outcome before
summing to form the prospect value. The parent chooses the prospect with
the higher CPT value.

If the parent cannot afford care, the waiting outcome is applied directly
without CPT evaluation.

Counters
--------
Three counters are maintained and reset between runs via reset_counters():

  decisions_faced   : number of times _parent_makes_decision() was called,
                      i.e. a parent stochastically noticed a sick child.

  care_sought       : number of times the parent chose to seek care
                      (CPT branch only).

  could_not_afford  : number of times the parent wanted to seek care
                      but could not afford it (CPT favoured care but
                      wealth < cost_of_care).

The conditional care rate is: care_sought / decisions_faced.
"""

import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys
from abm.agent.health_cpt_utils import utility, cpt_value_function, probability_weighting


class CareSeekingSystem(System):
    """Handles parent agent decision-making for sick children."""

    def __init__(self, config):
        super().__init__(config)
        self.decisions_faced:   int = 0
        self.care_sought:       int = 0
        self.could_not_afford:  int = 0

    def reset_counters(self):
        """Reset all counters — call between independent simulation runs."""
        self.decisions_faced  = 0
        self.care_sought      = 0
        self.could_not_afford = 0

    @property
    def conditional_care_rate(self) -> float:
        """Fraction of triggered decisions that resulted in care being sought."""
        if self.decisions_faced == 0:
            return 0.0
        return self.care_sought / self.decisions_faced

    def update(self, agent_state: AgentState, **kwargs):
        child_severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY]

        # Only children with non-zero severity are candidates
        candidate_mask = (
            (child_severity > 0.0)
            & agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        )
        if not torch.any(candidate_mask):
            return

        # Stochastic notice: each candidate child is "noticed" with
        # probability equal to its severity score
        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        severities = child_severity[candidate_indices]
        noticed_mask = torch.rand(len(candidate_indices), device=agent_state.device) < severities
        noticed_indices = candidate_indices[noticed_mask]

        if len(noticed_indices) == 0:
            return

        # --- Build household → noticed-children index map ---
        child_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID][noticed_indices]

        hh_to_noticed_children: dict[int, list] = {}
        for i, hh_id in enumerate(child_hh_ids):
            key = hh_id.item()
            hh_to_noticed_children.setdefault(key, []).append(noticed_indices[i])

        # --- Each parent with a noticed sick child makes a decision ---
        all_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        parent_mask = agent_state.ndata[AgentPropertyKeys.IS_PARENT]
        parent_indices = parent_mask.nonzero(as_tuple=True)[0]
        parent_hh_ids = all_hh_ids[parent_indices]

        for i, parent_idx in enumerate(parent_indices):
            hh_id = parent_hh_ids[i].item()
            if hh_id not in hh_to_noticed_children:
                continue

            # Focus on the sickest noticed child in the household
            children = hh_to_noticed_children[hh_id]
            indices_tensor = torch.tensor(
                [idx.item() for idx in children],
                device=agent_state.device,
                dtype=torch.long,
            )
            severities = child_severity[indices_tensor]
            sickest_child_idx = children[torch.argmax(severities).item()]

            self.decisions_faced += 1
            self._parent_makes_decision(agent_state, parent_idx, sickest_child_idx)

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------

    def _parent_makes_decision(
        self,
        agent_state: AgentState,
        parent_idx: torch.Tensor,
        child_idx: torch.Tensor,
    ):
        params = self.config.steering_parameters

        parent_wealth = agent_state.ndata[AgentPropertyKeys.WEALTH][parent_idx].item()
        parent_health = agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx].item()

        # --- Cannot afford care: forced to wait ---
        if parent_wealth < params.cost_of_care:
            self.could_not_afford += 1
            self._apply_waiting_outcome(agent_state, parent_idx, child_idx)
            return

        # --- CPT evaluation ---
        alpha = agent_state.ndata[AgentPropertyKeys.ALPHA][parent_idx].item()
        gamma = agent_state.ndata[AgentPropertyKeys.GAMMA][parent_idx].item()
        lam   = agent_state.ndata[AgentPropertyKeys.LAMBDA][parent_idx].item()

        ref_u = utility(
            torch.tensor(parent_wealth),
            torch.tensor(parent_health),
            torch.tensor(alpha),
        ).item()

        # ---- Prospect A: Seek Care ----------------------------------------
        w_after_cost = parent_wealth - params.cost_of_care

        u_A1 = utility(
            torch.tensor(w_after_cost),
            torch.tensor(parent_health),
            torch.tensor(alpha),
        ).item()

        u_A2 = utility(
            torch.tensor(w_after_cost),
            torch.tensor(max(0.0, parent_health - params.parent_stress_health_impact)),
            torch.tensor(alpha),
        ).item()

        pi_A1 = probability_weighting(params.treatment_success_prob, gamma)
        pi_A2 = probability_weighting(1.0 - params.treatment_success_prob, gamma)

        v_A1 = _cpt_v(u_A1 - ref_u, lam)
        v_A2 = _cpt_v(u_A2 - ref_u, lam)

        ev_seek_care = pi_A1 * v_A1 + pi_A2 * v_A2

        # ---- Prospect B: Wait ---------------------------------------------
        u_B1 = utility(
            torch.tensor(parent_wealth),
            torch.tensor(max(0.0, parent_health - params.parent_stress_health_impact)),
            torch.tensor(alpha),
        ).item()

        pi_B1 = probability_weighting(params.natural_worsening_prob, gamma)
        v_B1  = _cpt_v(u_B1 - ref_u, lam)
        ev_wait = pi_B1 * v_B1

        # ---- Choose -------------------------------------------------------
        if ev_seek_care >= ev_wait:
            self.care_sought += 1
            self._apply_treatment_outcome(agent_state, parent_idx, child_idx)
        else:
            self._apply_waiting_outcome(agent_state, parent_idx, child_idx)

    # ------------------------------------------------------------------
    # Outcome application
    # ------------------------------------------------------------------

    def _apply_treatment_outcome(
        self,
        agent_state: AgentState,
        parent_idx: torch.Tensor,
        child_idx: torch.Tensor,
    ):
        """Deduct cost, log the visit, and stochastically apply treatment."""
        params = self.config.steering_parameters

        agent_state.ndata[AgentPropertyKeys.WEALTH][parent_idx] -= params.cost_of_care
        agent_state.ndata[AgentPropertyKeys.CARE_SEEKING_COUNT][parent_idx] += 1

        if torch.rand(1).item() < params.treatment_success_prob:
            current_duration = agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION][child_idx].item()
            new_duration = max(0, current_duration - params.duration_reduction_on_success)
            agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION][child_idx] = new_duration
            if new_duration <= 0:
                agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][child_idx] = 0.0
        else:
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx].item()
            agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx] = max(
                0.0, current_health - params.parent_stress_health_impact
            )

    def _apply_waiting_outcome(
        self,
        agent_state: AgentState,
        parent_idx: torch.Tensor,
        child_idx: torch.Tensor,
    ):
        """Stochastically apply the consequences of choosing to wait."""
        params = self.config.steering_parameters

        if torch.rand(1).item() < params.natural_worsening_prob:
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx].item()
            agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx] = max(
                0.0, current_health - params.parent_stress_health_impact
            )
            current_severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][child_idx].item()
            agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][child_idx] = min(
                1.0, current_severity + params.untreated_severity_penalty
            )


def _cpt_v(delta_u: float, lam: float) -> float:
    """
    Apply the CPT value function to a utility change delta_u, scaling losses
    by the agent's personal loss-aversion coefficient lambda.
    """
    raw = cpt_value_function(delta_u)
    if delta_u < 0:
        return lam * raw
    return raw

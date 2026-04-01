# abm/systems/care_seeking.py
"""
Parental care-seeking decisions via Cumulative Prospect Theory (CPT).

Decision frame
--------------
Each day, a parent with at least one sick child above the moderate-severity
threshold evaluates two prospects:

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

Severe cases (severity >= severe_severity_threshold) bypass CPT: the parent
seeks care automatically if they can afford it, reflecting the near-certain
decision to act in a crisis.

Counters
--------
Three counters are maintained and reset between runs via reset_counters():

  decisions_faced   : number of times _parent_makes_decision() was called,
                      i.e. a parent had a child sick enough to trigger the
                      decision tree (severity > moderate_severity_threshold).

  care_sought       : number of times the parent chose to seek care
                      (both CPT branch and severe auto-seek).

  could_not_afford  : number of times the parent wanted to seek care
                      (severe case) but could not afford it.

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
        params = self.config.steering_parameters
        child_severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY]

        sick_children_mask = (
            (child_severity > params.moderate_severity_threshold)
            & agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        )
        if not torch.any(sick_children_mask):
            return

        # --- Build household → sick-children index map ---
        sick_child_indices = sick_children_mask.nonzero(as_tuple=True)[0]
        child_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID][sick_child_indices]

        hh_to_sick_children: dict[int, list] = {}
        for i, hh_id in enumerate(child_hh_ids):
            key = hh_id.item()
            hh_to_sick_children.setdefault(key, []).append(sick_child_indices[i])

        # --- Each parent with a sick child makes a decision ---
        all_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        parent_mask = agent_state.ndata[AgentPropertyKeys.IS_PARENT]
        parent_indices = parent_mask.nonzero(as_tuple=True)[0]
        parent_hh_ids = all_hh_ids[parent_indices]

        for i, parent_idx in enumerate(parent_indices):
            hh_id = parent_hh_ids[i].item()
            if hh_id not in hh_to_sick_children:
                continue

            # Focus on the sickest child in the household
            children = hh_to_sick_children[hh_id]
            indices_tensor = torch.tensor(
                [idx.item() for idx in children],
                device=agent_state.device,
                dtype=torch.long,
            )
            severities = child_severity[indices_tensor]
            sickest_child_idx = children[torch.argmax(severities).item()]

            # Increment decisions_faced before calling the decision method
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
        child_severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][child_idx].item()

        # --- Severe cases: automatic decision (no CPT) ---
        if child_severity >= params.severe_severity_threshold:
            if parent_wealth >= params.cost_of_care:
                self.care_sought += 1
                self._apply_treatment_outcome(agent_state, parent_idx, child_idx)
            else:
                self.could_not_afford += 1
            return

        # --- Cannot afford care: forced to wait ---
        if parent_wealth < params.cost_of_care:
            self._apply_waiting_outcome(agent_state, parent_idx, child_idx)
            return

        # --- CPT evaluation for moderate cases ---
        alpha  = agent_state.ndata[AgentPropertyKeys.ALPHA][parent_idx].item()
        gamma  = agent_state.ndata[AgentPropertyKeys.GAMMA][parent_idx].item()
        lam    = agent_state.ndata[AgentPropertyKeys.LAMBDA][parent_idx].item()

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

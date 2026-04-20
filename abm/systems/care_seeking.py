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
    reduced by duration_reduction_on_success days), which the parent
    values as a future health gain for the child.
    With probability (1 - p_success) the cost is paid but the child's
    illness continues unchanged.

  Wait
    No immediate cost.
    With probability p_worsen the child's severity increases by
    untreated_severity_penalty and the parent suffers a stress health
    hit of parent_stress_health_impact.
    With probability (1 - p_worsen) nothing changes.

CPT mode (use_cpt=True, default)
---------------------------------
Both prospects are evaluated in utility space using a Cobb-Douglas utility
function U(w, h) = w^alpha * h^(1-alpha), with gains and losses measured
relative to the parent's current (reference) utility. The CPT value
function and Prelec probability weighting are applied to each outcome before
summing to form the prospect value. The parent chooses the prospect with
the higher CPT value.

EV mode (use_cpt=False)
-----------------------
A plain expected-value comparison using the same Cobb-Douglas utility
function but with objective probabilities (no weighting) and a linear
value function (no curvature, no loss aversion). This is the "rational
expected-utility" baseline used in Experiment 5 to isolate the behavioural
contribution of CPT.

If the parent cannot afford care in either mode, the waiting outcome is
applied directly without any evaluation.

Counters
--------
Three counters are maintained and reset between runs via reset_counters():

  decisions_faced   : number of times _parent_makes_decision() was called.
  care_sought       : number of times the parent chose to seek care.
  could_not_afford  : number of times the parent wanted to seek care but could not afford it.

The conditional care rate is: care_sought / decisions_faced.
"""

import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys
from abm.agent.health_cpt_utils import utility, cpt_value_function, probability_weighting


class CareSeekingSystem(System):
    """Handles parent agent decision-making for sick children.

    Parameters
    ----------
    config : SVEIRConfig
    use_cpt : bool, default True
        When True (default) the full CPT decision rule is used, including
        Prelec probability weighting and the asymmetric value function with
        loss aversion (lambda).  When False a plain expected-utility rule is
        used instead: objective probabilities, linear value function, and
        lambda = 1.  This switch is the basis for Experiment 5.
    """

    def __init__(self, config, use_cpt: bool = True):
        super().__init__(config)
        self.use_cpt = use_cpt
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

        if self.use_cpt:
            seek_care = self._evaluate_cpt(agent_state, parent_idx, parent_wealth, parent_health)
        else:
            seek_care = self._evaluate_ev(parent_wealth, parent_health,
                                          agent_state.ndata[AgentPropertyKeys.ALPHA][parent_idx].item())

        if seek_care:
            self.care_sought += 1
            self._apply_treatment_outcome(agent_state, parent_idx, child_idx)
        else:
            self._apply_waiting_outcome(agent_state, parent_idx, child_idx)

    # ------------------------------------------------------------------
    # CPT evaluation
    # ------------------------------------------------------------------

    def _evaluate_cpt(
        self,
        agent_state: AgentState,
        parent_idx: torch.Tensor,
        parent_wealth: float,
        parent_health: float,
    ) -> bool:
        """Return True if CPT favours seeking care."""
        params = self.config.steering_parameters
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

        return ev_seek_care >= ev_wait

    # ------------------------------------------------------------------
    # Plain expected-utility evaluation (no CPT)
    # ------------------------------------------------------------------

    def _evaluate_ev(
        self,
        parent_wealth: float,
        parent_health: float,
        alpha: float,
    ) -> bool:
        """
        Return True if plain expected utility favours seeking care.

        Uses objective probabilities (no Prelec weighting) and a linear
        value function (no curvature, lambda = 1).  The same Cobb-Douglas
        utility function is retained so that the only difference from the
        CPT path is the *decision rule*, not the preference structure.
        """
        params = self.config.steering_parameters
        w_after = parent_wealth - params.cost_of_care

        alpha_t = torch.tensor(alpha)

        # EV(seek care) = p_success * U(w', h) + (1-p_success) * U(w', h - stress)
        u_success = utility(
            torch.tensor(w_after),
            torch.tensor(parent_health),
            alpha_t,
        ).item()
        u_fail = utility(
            torch.tensor(w_after),
            torch.tensor(max(0.0, parent_health - params.parent_stress_health_impact)),
            alpha_t,
        ).item()
        ev_seek = (params.treatment_success_prob * u_success
                   + (1.0 - params.treatment_success_prob) * u_fail)

        # EV(wait) = p_worsen * U(w, h - stress) + (1-p_worsen) * U(w, h)
        u_worsen = utility(
            torch.tensor(parent_wealth),
            torch.tensor(max(0.0, parent_health - params.parent_stress_health_impact)),
            alpha_t,
        ).item()
        u_ok = utility(
            torch.tensor(parent_wealth),
            torch.tensor(parent_health),
            alpha_t,
        ).item()
        ev_wait = (params.natural_worsening_prob * u_worsen
                   + (1.0 - params.natural_worsening_prob) * u_ok)

        return ev_seek >= ev_wait

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

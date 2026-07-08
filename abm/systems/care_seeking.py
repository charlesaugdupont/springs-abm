# abm/systems/care_seeking.py
"""
Parental care-seeking decisions via Cumulative Prospect Theory (CPT).

Decision frame
--------------
Each day, for each child with a non-zero severity, the parent stochastically
"notices" the child as sick with probability equal to the child's current
severity score. A child may be concurrently ill with more than one pathogen
(see ChildIllnessSystem, which tracks severity/duration per pathogen); the
severity used here is the *maximum* across the child's currently active
episodes — i.e. how sick the child appears overall. If noticed, the parent
evaluates two prospects:

  Seek care
    Cost c is paid immediately (certain wealth loss).
    With probability p_success the child recovers (illness duration shortened,
    child health restored, and the child transitions INFECTIOUS → RECOVERED).
    With probability (1 - p_success) the cost is paid but the child's
    illness continues and the parent suffers a stress health hit.

  Wait
    No immediate cost.
    With probability p_worsen the child's severity increases and the parent
    suffers a stress health hit.
    With probability (1 - p_worsen) nothing changes.

CPT mode (use_cpt=True, default)
---------------------------------
The parent's utility includes both their own state AND the child's health:

    U(w, h_eff) = w^alpha * h_eff^(1 - alpha)
    h_eff = (1 - child_weight) * h_parent + child_weight * h_child

The reference point is the current state (child is sick), so:
  - Seek+success: child recovers to full health → GAIN (h_eff rises)
  - Seek+fail: cost paid, child still sick → LOSS
  - Wait+ok: no change → 0 (reference)
  - Wait+worsen: child deteriorates → LOSS

This mixed-prospect structure (gain vs. loss in the seek prospect) makes
loss aversion (lambda) genuinely operative: loss-averse parents discount
the recovery gain heavily, suppressing care-seeking relative to EV.

The Prelec probability weighting function distorts both probabilities,
creating additional divergence from EV especially at intermediate costs.

EV mode (use_cpt=False)
-----------------------
Same extended utility function and same reference point, but with objective
probabilities (no Prelec weighting) and a linear value function (no
curvature, lambda = 1). This is the rational expected-utility baseline.

Wealth
------
`w` above is the household's POOLED wealth (see EconomicSystem and
abm.utils.household), not the parent's individual balance — households
share financial resources, so the affordability check and the cost of care
draw on, and are deducted from, the whole household.

Transmission link
-----------------
When treatment succeeds, ALL of the child's currently active pathogen
episodes are resolved together — duration and severity reset to zero for
every pathogen with an active episode, and any INFECTIOUS compartment
transitions immediately to RECOVERED. This treats a single care-seeking
event (e.g. a clinic visit) as addressing the child's diarrheal illness
holistically rather than one pathogen at a time, and closes the loop
between care-seeking and epidemic dynamics: higher care-seeking rates
reduce prevalence, not just illness duration.

If the parent instead chooses to wait and the child's condition worsens,
the severity bump is applied to whichever pathogen is currently the
child's "presenting" (most severe) active episode.

Counters
--------
  decisions_faced : number of times a parent noticed a sick child.
  care_sought : number of times the parent chose to seek care.
  could_not_afford : number of times the parent was priced out.
"""

import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys, Compartment
from abm.agent.health_cpt_utils import cpt_value_function, probability_weighting
from abm.systems.household import HouseholdSystem

# Weight placed on child health in the parent's utility function.
# 0.5 = equal weight on parent and child health.
CHILD_HEALTH_WEIGHT: float = 0.5


def _utility(w: float, h_eff: float, alpha: float) -> float:
    """Cobb-Douglas utility: w^alpha * h_eff^(1-alpha)."""
    return (w + 1e-9) ** alpha * (h_eff + 1e-9) ** (1.0 - alpha)


def _h_eff(h_parent: float, h_child: float) -> float:
    """Effective health combining parent and child health."""
    return (1.0 - CHILD_HEALTH_WEIGHT) * h_parent + CHILD_HEALTH_WEIGHT * h_child


def _max_active_severity(agent_state: AgentState, pathogen_names: list[str]) -> torch.Tensor:
    """
    Per-agent maximum currently-active symptom severity across all
    pathogens. A child can be concurrently ill with more than one pathogen
    (ChildIllnessSystem tracks severity/duration per pathogen so concurrent
    episodes don't overwrite one another); this is what determines how sick
    a child appears overall for notice-probability and "sickest child"
    comparisons.
    """
    severities = torch.zeros(agent_state.num_nodes(), device=agent_state.device)
    for pathogen_name in pathogen_names:
        severity_key = AgentPropertyKeys.symptom_severity(pathogen_name)
        severities = torch.maximum(severities, agent_state.ndata[severity_key])
    return severities


class CareSeekingSystem(System):
    """Handles parent agent decision-making for sick children.

    Parameters
    ----------
    config : SVEIRConfig
    use_cpt : bool, default True
        When True the full CPT decision rule is used (Prelec probability
        weighting, asymmetric value function, per-agent loss aversion lambda).
        When False a plain expected-utility rule is used instead.
    """

    def __init__(self, config, use_cpt: bool = True):
        super().__init__(config)
        self.use_cpt = use_cpt
        self.decisions_faced: int = 0
        self.care_sought: int = 0
        self.could_not_afford: int = 0

    def reset_counters(self):
        """Reset all counters — call between independent simulation runs."""
        self.decisions_faced = 0
        self.care_sought = 0
        self.could_not_afford = 0

    @property
    def conditional_care_rate(self) -> float:
        if self.decisions_faced == 0:
            return 0.0
        return self.care_sought / self.decisions_faced

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, agent_state: AgentState, **kwargs):
        pathogen_names = [p.name for p in self.config.pathogens]
        child_severity = _max_active_severity(agent_state, pathogen_names)

        candidate_mask = (
            (child_severity > 0.0)
            & agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        )
        if not torch.any(candidate_mask):
            return

        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        severities = child_severity[candidate_indices]
        noticed_mask = torch.rand(len(candidate_indices), device=agent_state.device) < severities
        noticed_indices = candidate_indices[noticed_mask]

        if len(noticed_indices) == 0:
            return

        child_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID][noticed_indices]
        hh_to_noticed: dict[int, list] = {}
        for i, hh_id in enumerate(child_hh_ids):
            hh_to_noticed.setdefault(hh_id.item(), []).append(noticed_indices[i])

        all_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        parent_mask = agent_state.ndata[AgentPropertyKeys.IS_PARENT]
        parent_indices = parent_mask.nonzero(as_tuple=True)[0]
        parent_hh_ids = all_hh_ids[parent_indices]

        for i, parent_idx in enumerate(parent_indices):
            hh_id = parent_hh_ids[i].item()
            if hh_id not in hh_to_noticed:
                continue

            children = hh_to_noticed[hh_id]
            idx_tensor = torch.tensor(
                [idx.item() for idx in children],
                device=agent_state.device, dtype=torch.long,
            )
            sickest = children[torch.argmax(child_severity[idx_tensor]).item()]

            self.decisions_faced += 1
            self._parent_makes_decision(agent_state, parent_idx, sickest)

    # ------------------------------------------------------------------
    # Presenting-illness lookup
    # ------------------------------------------------------------------

    def _presenting_illness(
        self, agent_state: AgentState, child_idx: torch.Tensor
    ) -> tuple[str | None, float]:
        """
        Returns (pathogen_name, severity) for whichever pathogen is
        currently driving this child's most severe active symptoms. Used to
        decide which specific illness worsens if care is not sought.
        """
        best_name, best_severity = None, 0.0
        for p_config in self.config.pathogens:
            severity_key = AgentPropertyKeys.symptom_severity(p_config.name)
            s = agent_state.ndata[severity_key][child_idx].item()
            if s > best_severity:
                best_name, best_severity = p_config.name, s
        return best_name, best_severity

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

        # Household-pooled wealth (see abm.utils.household) — not the
        # parent's individual balance.
        household_wealth = agent_state.ndata[AgentPropertyKeys.WEALTH][parent_idx].item()
        parent_health = agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx].item()

        presenting_pathogen, child_severity = self._presenting_illness(agent_state, child_idx)
        child_health = max(0.0, 1.0 - child_severity)

        if household_wealth < params.cost_of_care:
            self.could_not_afford += 1
            self._apply_waiting_outcome(agent_state, parent_idx, child_idx, presenting_pathogen)
            return

        alpha = agent_state.ndata[AgentPropertyKeys.ALPHA][parent_idx].item()

        if self.use_cpt:
            gamma = agent_state.ndata[AgentPropertyKeys.GAMMA][parent_idx].item()
            lam = agent_state.ndata[AgentPropertyKeys.LAMBDA][parent_idx].item()
            seek = self._evaluate_cpt(
                household_wealth, parent_health, child_health, alpha, gamma, lam, params
            )
        else:
            seek = self._evaluate_ev(
                household_wealth, parent_health, child_health, alpha, params
            )

        if seek:
            self.care_sought += 1
            self._apply_treatment_outcome(agent_state, parent_idx, child_idx)
        else:
            self._apply_waiting_outcome(agent_state, parent_idx, child_idx, presenting_pathogen)

    # ------------------------------------------------------------------
    # CPT evaluation
    # ------------------------------------------------------------------

    def _evaluate_cpt(
        self,
        w: float, h_p: float, h_c: float,
        alpha: float, gamma: float, lam: float,
        params,
    ) -> bool:
        """
        Return True if CPT favours seeking care.

        Reference point = current state (child is sick).
        Seek+success is a GAIN (child recovers to full health), making
        loss aversion (lambda) genuinely operative.
        """
        w2 = w - params.cost_of_care

        ref_u = _utility(w, _h_eff(h_p, h_c), alpha)

        u_s1 = _utility(w2, _h_eff(h_p, 1.0), alpha)
        u_s2 = _utility(w2, _h_eff(max(0.0, h_p - params.parent_stress_health_impact), h_c), alpha)

        h_c_worsened = max(0.0, h_c - params.untreated_severity_penalty)
        u_w1 = _utility(w, _h_eff(max(0.0, h_p - params.parent_stress_health_impact), h_c_worsened), alpha)

        pi_s1 = probability_weighting(params.treatment_success_prob, gamma)
        pi_s2 = 1.0 - probability_weighting(params.treatment_success_prob, gamma)
        pi_w1 = probability_weighting(params.natural_worsening_prob, gamma)

        v_s1 = _cpt_v(u_s1 - ref_u, lam)
        v_s2 = _cpt_v(u_s2 - ref_u, lam)
        v_w1 = _cpt_v(u_w1 - ref_u, lam)

        ev_seek = pi_s1 * v_s1 + pi_s2 * v_s2
        ev_wait = pi_w1 * v_w1

        return ev_seek >= ev_wait

    # ------------------------------------------------------------------
    # EV evaluation
    # ------------------------------------------------------------------

    def _evaluate_ev(
        self,
        w: float, h_p: float, h_c: float,
        alpha: float,
        params,
    ) -> bool:
        """
        Return True if plain expected utility favours seeking care.
        """
        w2 = w - params.cost_of_care

        u_s1 = _utility(w2, _h_eff(h_p, 1.0), alpha)
        u_s2 = _utility(w2, _h_eff(max(0.0, h_p - params.parent_stress_health_impact), h_c), alpha)

        h_c_worsened = max(0.0, h_c - params.untreated_severity_penalty)
        u_w1 = _utility(w, _h_eff(max(0.0, h_p - params.parent_stress_health_impact), h_c_worsened), alpha)
        u_w2 = _utility(w, _h_eff(h_p, h_c), alpha)

        ev_seek = (params.treatment_success_prob * u_s1
                   + (1.0 - params.treatment_success_prob) * u_s2)
        ev_wait = (params.natural_worsening_prob * u_w1
                   + (1.0 - params.natural_worsening_prob) * u_w2)

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
        """
        Deduct cost from the household's pooled wealth, log the visit, and
        stochastically apply treatment.

        On success: EVERY pathogen for which the child currently has an
        active episode is resolved together — severity and duration reset,
        and any INFECTIOUS compartment transitions to RECOVERED.
        """
        params = self.config.steering_parameters

        HouseholdSystem.apply_household_wealth_delta(agent_state, parent_idx, -params.cost_of_care)
        agent_state.ndata[AgentPropertyKeys.CARE_SEEKING_COUNT][parent_idx] += 1

        if torch.rand(1).item() < params.treatment_success_prob:
            for p_config in self.config.pathogens:
                status_key = AgentPropertyKeys.status(p_config.name)
                severity_key = AgentPropertyKeys.symptom_severity(p_config.name)
                duration_key = AgentPropertyKeys.illness_duration(p_config.name)

                if agent_state.ndata[duration_key][child_idx].item() > 0:
                    agent_state.ndata[severity_key][child_idx] = 0.0
                    agent_state.ndata[duration_key][child_idx] = 0

                if agent_state.ndata[status_key][child_idx].item() == Compartment.INFECTIOUS:
                    agent_state.ndata[status_key][child_idx] = Compartment.RECOVERED
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
        presenting_pathogen: str | None = None,
    ):
        """
        Stochastically apply the consequences of choosing to wait. If the
        child's condition worsens, the bump is applied to
        `presenting_pathogen` — the pathogen currently driving the child's
        most severe active symptoms.
        """
        params = self.config.steering_parameters

        if torch.rand(1).item() < params.natural_worsening_prob:
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx].item()
            agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx] = max(
                0.0, current_health - params.parent_stress_health_impact
            )

            if presenting_pathogen is None:
                presenting_pathogen, _ = self._presenting_illness(agent_state, child_idx)

            if presenting_pathogen is not None:
                severity_key = AgentPropertyKeys.symptom_severity(presenting_pathogen)
                current_severity = agent_state.ndata[severity_key][child_idx].item()
                agent_state.ndata[severity_key][child_idx] = min(
                    1.0, current_severity + params.untreated_severity_penalty
                )

def _cpt_v(delta_u: float, lam: float) -> float:
    """
    CPT value function applied to a utility change delta_u.
    Losses (delta_u < 0) are scaled by the agent's loss-aversion lambda.
    """
    raw = cpt_value_function(delta_u)
    if delta_u < 0:
        return lam * raw
    return raw
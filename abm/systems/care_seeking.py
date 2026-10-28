# abm/systems/care_seeking.py
"""
Parental care-seeking decisions via Cumulative Prospect Theory (CPT).

Decision frame
--------------
Each day, for each child with a non-zero severity, the parent evaluates a
care-seeking decision (diarrhea is highly visible, so awareness itself
isn't modeled as uncertain - unlike, say, a low-grade fever, there's no
realistic "might not notice" gate for a child with diarrhea, and severity
already drives the decision directly through the utility comparison below,
so a separate stochastic notice step would just double-count it). A child
may be concurrently ill with more than one pathogen (see ChildIllnessSystem,
which tracks severity/duration per pathogen); the severity used here is the
*maximum* across the child's currently active episodes — i.e. how sick the
child appears overall. The parent evaluates two prospects:

  Seek care
    Cost c is paid immediately (certain wealth loss).
    With probability p_success treatment is effective. What that does depends
    on steering_parameters.treatment_effect_model:
      "supportive" (default) - the episode's harm is cut by
        treatment_harm_reduction (ORS), it is shortened by about half a day in
        children over 6 months (zinc), and for a pathogen an antibiotic treats
        (treatment_clears_infection_prob) the infection is cleared and the
        episode shortened further. The episode otherwise runs its course, and
        the child faces no further decisions until it ends.
      "cure" (original model) - every active episode ends at once.
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

The Tversky-Kahneman (1992) probability weighting function distorts both probabilities,
creating additional divergence from EV especially at intermediate costs.

EV mode (use_cpt=False)
-----------------------
Same extended utility function and same reference point, but with objective
probabilities (no probability weighting) and a linear value function (no
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
every pathogen with an active episode. This treats a single care-seeking
event (e.g. a clinic visit) as addressing the child's diarrheal illness
holistically rather than one pathogen at a time.

Whether treatment ALSO ends the infection (INFECTIOUS -> RECOVERED, i.e.
stops shedding) is per pathogen: PathogenConfig.treatment_clears_infection_prob.
Originally every pathogen was cleared, so care-seeking reduced transmission
as well as illness. That is only defensible where the treatment given is
pathogen-specific and effective (antibiotics for campylobacter). Standard
care for watery diarrhoea - ORS and zinc - does not shorten rotavirus or
Cryptosporidium shedding, so for those the loop between care-seeking and
transmission is closed only through illness duration, not through
prevalence.

If the parent instead chooses to wait and the child's condition worsens,
the severity bump is applied to whichever pathogen is currently the
child's "presenting" (most severe) active episode.

Counters
--------
  decisions_faced : number of times a parent evaluated a sick child's care.
  care_sought : number of times the parent chose to seek care.
  could_not_afford : number of times the parent was priced out.
  total_episodes : number of illness episodes (any pathogen, onset-to-
      resolution).
  episodes_with_care_sought : of those, how many had care sought at least
      once. episode_care_seeking_rate (the ratio) is the DHS-comparable
      figure - conditional_care_rate is NOT, since its denominator is
      per-day decisions, not episodes.
"""

import numpy as np
import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys, Compartment
from abm.agent.health_cpt_utils import cpt_value_function, probability_weighting
from abm.systems.household import HouseholdSystem

def _utility(w: float, h_eff: float, alpha: float) -> float:
    """Cobb-Douglas utility: w^alpha * h_eff^(1-alpha)."""
    return (w + 1e-9) ** alpha * (h_eff + 1e-9) ** (1.0 - alpha)


def _h_eff(h_parent: float, h_child: float, child_health_weight: float) -> float:
    """Effective health combining parent and child health.

    child_health_weight: weight placed on child health (0.5 = equal weight
    on parent and child). Read from steering_parameters.child_health_weight
    at each call site so it's reachable via the same dot-path sweep/override
    mechanism as every other steering parameter.
    """
    return (1.0 - child_health_weight) * h_parent + child_health_weight * h_child


def _max_active_severity(agent_state: AgentState, pathogen_names: list[str]) -> torch.Tensor:
    """
    Per-agent maximum currently-active symptom severity across all
    pathogens. A child can be concurrently ill with more than one pathogen
    (ChildIllnessSystem tracks severity/duration per pathogen so concurrent
    episodes don't overwrite one another); this is what determines how sick
    a child appears overall for the utility comparison and "sickest child"
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
        When True the full CPT decision rule is used (Tversky-Kahneman
        probability weighting, asymmetric value function, per-agent loss aversion lambda).
        When False a plain expected-utility rule is used instead.
    """

    def __init__(self, config, use_cpt: bool = True):
        super().__init__(config)
        self.use_cpt = use_cpt
        # Household economy in cedis (EconomyConfig.model == "cedi"): real
        # costs, cash-based affordability and wealth term, and per-quintile
        # daily records (`daily_records`). The legacy path is unchanged.
        self._cedi = getattr(config, "economy", None) is not None and config.economy.model == "cedi"
        self.daily_records: list[dict] = []
        self.decisions_faced: int = 0
        self.care_sought: int = 0
        self.could_not_afford: int = 0

        # Episode-level tracking. Lazily sized on the first update() call
        # since num_nodes isn't known at construction time. Pure bookkeeping,
        # not shared simulation state, so this lives as instance tensors
        # rather than AgentPropertyKeys.
        self.total_episodes: int = 0
        self.episodes_with_care_sought: int = 0
        self._was_sick_prev: torch.Tensor | None = None
        self._care_sought_this_episode: torch.Tensor | None = None
        # Children whose current illness spell has been successfully treated
        # (treatment_effect_model == "supportive"): they face no further
        # decisions until the spell ends.
        self._treated_this_episode: torch.Tensor | None = None

    def reset_counters(self):
        """Reset all counters — call between independent simulation runs."""
        self.decisions_faced = 0
        self.care_sought = 0
        self.could_not_afford = 0
        self.total_episodes = 0
        self.episodes_with_care_sought = 0
        self._was_sick_prev = None
        self._care_sought_this_episode = None
        self._treated_this_episode = None
        self.daily_records = []

    @property
    def conditional_care_rate(self) -> float:
        if self.decisions_faced == 0:
            return 0.0
        return self.care_sought / self.decisions_faced

    @property
    def episode_care_seeking_rate(self) -> float:
        """Fraction of illness episodes (any pathogen, onset-to-resolution)
        for which care was sought at least once - the DHS-comparable
        figure. conditional_care_rate is NOT comparable to DHS: its
        denominator is per-day decisions, not episodes."""
        if self.total_episodes == 0:
            return 0.0
        return self.episodes_with_care_sought / self.total_episodes

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, agent_state: AgentState, **kwargs):
        pathogen_names = [p.name for p in self.config.pathogens]
        child_severity = _max_active_severity(agent_state, pathogen_names)
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD]

        if self._cedi:
            self._track_episodes(child_severity, is_child,
                                 quintile=agent_state.ndata[AgentPropertyKeys.WEALTH_QUINTILE])
        else:
            self._track_episodes(child_severity, is_child)

        candidate_mask = (
            (child_severity > 0.0)
            & agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        )
        if self.config.steering_parameters.treatment_effect_model == "supportive":
            # One successful visit per illness spell: a treated child is under
            # treatment until the spell ends (under "cure" the spell has
            # already ended, so this is only needed here).
            candidate_mask = candidate_mask & ~self._treated_this_episode
        if not torch.any(candidate_mask):
            return

        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]

        child_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID][candidate_indices]
        hh_to_sick_children: dict[int, list] = {}
        for i, hh_id in enumerate(child_hh_ids):
            hh_to_sick_children.setdefault(hh_id.item(), []).append(candidate_indices[i])

        all_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        parent_mask = agent_state.ndata[AgentPropertyKeys.IS_PARENT]
        parent_indices = parent_mask.nonzero(as_tuple=True)[0]
        parent_hh_ids = all_hh_ids[parent_indices]

        for i, parent_idx in enumerate(parent_indices):
            hh_id = parent_hh_ids[i].item()
            if hh_id not in hh_to_sick_children:
                continue

            children = hh_to_sick_children[hh_id]
            idx_tensor = torch.tensor(
                [idx.item() for idx in children],
                device=agent_state.device, dtype=torch.long,
            )
            sickest = children[torch.argmax(child_severity[idx_tensor]).item()]

            self.decisions_faced += 1
            self._parent_makes_decision(agent_state, parent_idx, sickest)

    # ------------------------------------------------------------------
    # Episode-level tracking (DHS-comparable metric)
    # ------------------------------------------------------------------

    def _track_episodes(self, child_severity: torch.Tensor, is_child: torch.Tensor,
                        quintile: torch.Tensor | None = None):
        """
        Maintains per-agent illness-episode bookkeeping for
        episode_care_seeking_rate. An episode runs from the first day
        `child_severity > 0` (any pathogen) until the first day it returns
        to zero: onset increments total_episodes and clears the per-episode
        "care sought" flag; resolution increments episodes_with_care_sought
        if that flag was ever set during the episode (set in
        _apply_treatment_outcome, regardless of treatment success/failure -
        this measures whether care was SOUGHT, not whether it worked).
        Episodes still open at simulation end are counted at onset but
        never resolve, so they don't contribute to the numerator -
        negligible given illness durations (days) are short relative to a
        sweep's step count (months).
        """
        currently_sick = (child_severity > 0.0) & is_child

        if self._was_sick_prev is None:
            self._was_sick_prev = torch.zeros_like(currently_sick)
            self._care_sought_this_episode = torch.zeros_like(currently_sick)
            self._treated_this_episode = torch.zeros_like(currently_sick)

        onset_mask = currently_sick & ~self._was_sick_prev
        resolved_mask = self._was_sick_prev & ~currently_sick

        self.total_episodes += int(onset_mask.sum().item())
        if quintile is not None:
            if getattr(self, "_blocked_this_episode", None) is None or self._blocked_this_episode.numel() != currently_sick.numel():
                self._blocked_this_episode = torch.zeros_like(currently_sick)
            # cedi mode: today's record, by household wealth quintile (decision
            # counts are added to it by _parent_makes_decision_cedi)
            from abm.economy.cedi import N_QUINTILES
            q = quintile.long()
            count = lambda m: torch.bincount(q[m], minlength=N_QUINTILES).tolist()
            self.daily_records.append({
                "episodes_started": count(onset_mask),
                "episodes_ended": count(resolved_mask),
                "episodes_ended_with_care": count(resolved_mask & self._care_sought_this_episode),
                # ended without care after at least one unaffordable decision
                "episodes_ended_blocked": count(resolved_mask & self._blocked_this_episode
                                                & ~self._care_sought_this_episode),
                "decisions": [0] * N_QUINTILES, "care_sought": [0] * N_QUINTILES,
                "could_not_afford": [0] * N_QUINTILES,
            })
            self._blocked_this_episode[onset_mask] = False
        self._care_sought_this_episode[onset_mask] = False
        self._treated_this_episode[onset_mask] = False
        self.episodes_with_care_sought += int(
            (resolved_mask & self._care_sought_this_episode).sum().item()
        )

        self._was_sick_prev = currently_sick

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
        if self._cedi:
            self._parent_makes_decision_cedi(agent_state, parent_idx, child_idx)
            return
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

    def _parent_makes_decision_cedi(
        self,
        agent_state: AgentState,
        parent_idx: torch.Tensor,
        child_idx: torch.Tensor,
    ):
        """The decision under the cedi economy (EconomyConfig): the out-of-pocket
        cost of a visit to the household's usual provider (at a public facility,
        by NHIS cover) must be payable from cash, and the CPT compares the wealth
        term before and after paying it and losing that provider's share of the
        parent's day of earnings. Otherwise exactly _parent_makes_decision."""
        from abm.economy.cedi import visit_cost, wealth_term
        K = AgentPropertyKeys
        params = self.config.steering_parameters
        eco = self.config.economy
        nd = agent_state.ndata
        q = int(nd[K.WEALTH_QUINTILE][parent_idx].item())
        today = self.daily_records[-1]
        today["decisions"][q] += 1

        cash = float(nd[K.CASH][parent_idx].item())
        target = float(nd[K.CONSUMPTION_TARGET][parent_idx].item())
        parent_health = nd[K.HEALTH][parent_idx].item()
        oop, day_fraction = visit_cost(nd, parent_idx, eco)
        lost = float(nd[K.EARNING_POTENTIAL][parent_idx].item()) * day_fraction
        if params.health_based_income:
            lost *= parent_health

        presenting_pathogen, child_severity = self._presenting_illness(agent_state, child_idx)
        child_health = max(0.0, 1.0 - child_severity)

        if cash < oop:
            self.could_not_afford += 1
            today["could_not_afford"][q] += 1
            self._blocked_this_episode[child_idx] = True
            self._apply_waiting_outcome(agent_state, parent_idx, child_idx, presenting_pathogen)
            return

        w = float(wealth_term(np.float64(cash), np.float64(target), eco.wealth_scale_days))
        w_after = float(wealth_term(np.float64(cash - oop - lost), np.float64(target), eco.wealth_scale_days))
        alpha = nd[K.ALPHA][parent_idx].item()
        if self.use_cpt:
            gamma = nd[K.GAMMA][parent_idx].item()
            lam = nd[K.LAMBDA][parent_idx].item()
            seek = self._evaluate_cpt(w, parent_health, child_health, alpha, gamma, lam, params, w_after=w_after)
        else:
            seek = self._evaluate_ev(w, parent_health, child_health, alpha, params, w_after=w_after)

        if seek:
            self.care_sought += 1
            today["care_sought"][q] += 1
            self._apply_treatment_outcome(agent_state, parent_idx, child_idx, oop_cost=oop,
                                          lost_day_fraction=day_fraction)
        else:
            self._apply_waiting_outcome(agent_state, parent_idx, child_idx, presenting_pathogen)

    # ------------------------------------------------------------------
    # CPT evaluation
    # ------------------------------------------------------------------

    def _evaluate_cpt(
        self,
        w: float, h_p: float, h_c: float,
        alpha: float, gamma: float, lam: float,
        params, w_after: float | None = None,
    ) -> bool:
        """
        Return True if CPT favours seeking care.

        Reference point = current state (child is sick). "Seek care" is a
        two-outcome prospect: success (prob=treatment_success_prob) and
        failure (prob=1-treatment_success_prob).

        Success is usually, but NOT always, a gain: for a health-weighted
        (low alpha) agent near the affordability floor with a mildly-sick
        child, cost_of_care can outweigh a small health improvement, making
        "successful" treatment a net loss too. (Numerically verified:
        failure is a loss in ~100% of the realistic parameter space, but
        success is a loss in a non-negligible - and, for mildly-sick
        children, majority - share of it.) Textbook CPT (Tversky & Kahneman
        1992) handles this with two distinct rules depending on whether the
        outcomes land in the same domain:
          - Different domains (one gain, one loss): each outcome, being
            alone in its own domain, is weighted by the probability
            weighting function applied to its OWN raw probability. These do
            NOT sum to 1 (subadditivity is deliberate, not an error).
          - Same domain (both losses here): outcomes are ranked by
            severity; the WORSE loss gets w(its own probability), and the
            less-bad loss gets the complementary w(1) - w(worse's
            probability) = 1 - w(worse's probability) - this "1 minus"
            form is only valid within a single domain, never across
            domains.

        w_after: the wealth term after paying for the visit (cedi economy);
        None = w - cost_of_care (legacy).
        """
        w2 = w - params.cost_of_care if w_after is None else w_after

        chw = params.child_health_weight
        ref_u = _utility(w, _h_eff(h_p, h_c, chw), alpha)

        u_s1 = _utility(w2, _h_eff(h_p, _treated_child_health(h_c, params), chw), alpha)
        u_s2 = _utility(w2, _h_eff(max(0.0, h_p - params.parent_stress_health_impact), h_c, chw), alpha)

        h_c_worsened = max(0.0, h_c - params.untreated_severity_penalty)
        u_w1 = _utility(w, _h_eff(max(0.0, h_p - params.parent_stress_health_impact), h_c_worsened, chw), alpha)

        d_s1 = u_s1 - ref_u
        d_s2 = u_s2 - ref_u

        p1 = params.treatment_success_prob
        p2 = 1.0 - p1

        if d_s1 >= 0:
            # Mixed gamble: success is the lone gain, failure is the lone loss.
            pi_s1 = probability_weighting(p1, gamma)
            pi_s2 = probability_weighting(p2, gamma)
        elif d_s1 <= d_s2:
            # Both losses, success is the worse (or equally bad) outcome.
            pi_s1 = probability_weighting(p1, gamma)
            pi_s2 = 1.0 - pi_s1
        else:
            # Both losses, failure is the worse outcome.
            pi_s2 = probability_weighting(p2, gamma)
            pi_s1 = 1.0 - pi_s2

        pi_w1 = probability_weighting(params.natural_worsening_prob, gamma)

        v_s1 = _cpt_v(d_s1, lam, params.cpt_theta, params.cpt_eta)
        v_s2 = _cpt_v(d_s2, lam, params.cpt_theta, params.cpt_eta)
        v_w1 = _cpt_v(u_w1 - ref_u, lam, params.cpt_theta, params.cpt_eta)

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
        params, w_after: float | None = None,
    ) -> bool:
        """
        Return True if plain expected utility favours seeking care.
        """
        w2 = w - params.cost_of_care if w_after is None else w_after
        chw = params.child_health_weight

        u_s1 = _utility(w2, _h_eff(h_p, _treated_child_health(h_c, params), chw), alpha)
        u_s2 = _utility(w2, _h_eff(max(0.0, h_p - params.parent_stress_health_impact), h_c, chw), alpha)

        h_c_worsened = max(0.0, h_c - params.untreated_severity_penalty)
        u_w1 = _utility(w, _h_eff(max(0.0, h_p - params.parent_stress_health_impact), h_c_worsened, chw), alpha)
        u_w2 = _utility(w, _h_eff(h_p, h_c, chw), alpha)

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
        oop_cost: float | None = None,
        lost_day_fraction: float = 0.0,
    ):
        """
        Deduct cost from the household's pooled wealth, log the visit, and
        stochastically apply treatment.

        On success, under treatment_effect_model == "supportive" (default):
        see _apply_supportive_treatment. Under "cure" (original model): EVERY
        pathogen for which the child currently has an active episode is
        resolved together — severity and duration reset — and each
        pathogen's INFECTIOUS compartment additionally transitions to
        RECOVERED with that pathogen's treatment_clears_infection_prob (1.0
        reproduces the original behaviour; 0 means the child keeps shedding).
        """
        params = self.config.steering_parameters

        if oop_cost is None:
            HouseholdSystem.apply_household_wealth_delta(agent_state, parent_idx, -params.cost_of_care)
        else:
            # cedi economy: pay out of pocket from cash; the parent loses part of the day
            from abm.economy.cedi import add_household_cash
            eco = self.config.economy
            add_household_cash(agent_state, parent_idx, -oop_cost, eco.wealth_scale_days)
            lost = agent_state.ndata[AgentPropertyKeys.WORK_LOST]
            lost[parent_idx] = min(1.0, float(lost[parent_idx].item()) + float(lost_day_fraction))
        agent_state.ndata[AgentPropertyKeys.CARE_SEEKING_COUNT][parent_idx] += 1
        self._care_sought_this_episode[child_idx] = True

        if torch.rand(1).item() < params.treatment_success_prob:
            if params.treatment_effect_model == "supportive":
                self._apply_supportive_treatment(agent_state, child_idx)
                return
            for p_config in self.config.pathogens:
                status_key = AgentPropertyKeys.status(p_config.name)
                severity_key = AgentPropertyKeys.symptom_severity(p_config.name)
                duration_key = AgentPropertyKeys.illness_duration(p_config.name)

                if agent_state.ndata[duration_key][child_idx].item() > 0:
                    agent_state.ndata[severity_key][child_idx] = 0.0
                    agent_state.ndata[duration_key][child_idx] = 0

                # Whether the visit also ends the INFECTION (not just the
                # symptoms) is pathogen-specific: supportive care (ORS + zinc)
                # does not shorten shedding. See
                # PathogenConfig.treatment_clears_infection_prob. The draw is
                # skipped for the deterministic 0/1 cases so they consume no RNG.
                if agent_state.ndata[status_key][child_idx].item() == Compartment.INFECTIOUS:
                    p_clear = float(getattr(p_config, "treatment_clears_infection_prob", 1.0))
                    if p_clear >= 1.0 or (p_clear > 0.0 and torch.rand(1).item() < p_clear):
                        agent_state.ndata[status_key][child_idx] = Compartment.RECOVERED
        else:
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx].item()
            agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx] = max(
                0.0, current_health - params.parent_stress_health_impact
            )

    def _apply_supportive_treatment(self, agent_state: AgentState, child_idx: torch.Tensor):
        """Effective treatment under treatment_effect_model == "supportive".

        For every pathogen with an active episode: severity is cut by
        treatment_harm_reduction (ORS prevents dehydration, the harm, not the
        diarrhoea); the remaining duration is shortened by
        treatment_duration_reduction_days if the child is old enough (zinc);
        and with that pathogen's treatment_clears_infection_prob the infection
        is cleared and the episode shortened by a further
        antibiotic_duration_reduction_days (an effective antibiotic). Day
        reductions are applied by stochastic rounding (0.48 days = one day with
        probability 0.48), so the expected effect equals the trial estimate.
        Infections without an active episode are cleared with the same
        probability, as an antibiotic would.
        """
        params = self.config.steering_parameters
        self._treated_this_episode[child_idx] = True
        age = agent_state.ndata[AgentPropertyKeys.AGE][child_idx].item()
        zinc_days = (params.treatment_duration_reduction_days
                     if age >= params.treatment_duration_min_age_months else 0.0)

        for p_config in self.config.pathogens:
            status_key = AgentPropertyKeys.status(p_config.name)
            severity_key = AgentPropertyKeys.symptom_severity(p_config.name)
            duration_key = AgentPropertyKeys.illness_duration(p_config.name)
            has_episode = agent_state.ndata[duration_key][child_idx].item() > 0

            cleared = False
            if agent_state.ndata[status_key][child_idx].item() == Compartment.INFECTIOUS:
                p_clear = float(getattr(p_config, "treatment_clears_infection_prob", 0.0))
                if p_clear >= 1.0 or (p_clear > 0.0 and torch.rand(1).item() < p_clear):
                    agent_state.ndata[status_key][child_idx] = Compartment.RECOVERED
                    cleared = True

            if not has_episode:
                continue
            agent_state.ndata[severity_key][child_idx] *= (1.0 - params.treatment_harm_reduction)
            cut = zinc_days + (params.antibiotic_duration_reduction_days if cleared else 0.0)
            if cut > 0.0:
                whole = int(cut)
                if torch.rand(1).item() < cut - whole:
                    whole += 1
                remaining = max(0, int(agent_state.ndata[duration_key][child_idx].item()) - whole)
                agent_state.ndata[duration_key][child_idx] = remaining
                if remaining == 0:
                    agent_state.ndata[severity_key][child_idx] = 0.0

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

def _treated_child_health(h_c: float, params) -> float:
    """The child's health after SUCCESSFUL treatment, as the parent weighs it.

    "cure": full health (the original model). "supportive": the episode's
    severity reduced by treatment_harm_reduction, i.e. what treatment actually
    does to the child in this model, so the decision and its consequence are
    consistent. (With the default 0.93 this is close to full health, so the
    decision frame changes little.)
    """
    if params.treatment_effect_model != "supportive":
        return 1.0
    return 1.0 - (1.0 - h_c) * (1.0 - params.treatment_harm_reduction)


def _cpt_v(delta_u: float, lam: float, theta: float, eta: float) -> float:
    """
    CPT value function applied to a utility change delta_u.
    Losses (delta_u < 0) are scaled by the agent's loss-aversion lambda.
    """
    raw = cpt_value_function(delta_u, theta, eta)
    if delta_u < 0:
        return lam * raw
    return raw
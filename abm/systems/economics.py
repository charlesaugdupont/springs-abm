# abm/systems/economics.py
import numpy as np
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

    With EconomyConfig.model == "cedi" the household economy is in 2026 Ghana
    cedis instead (_update_cedi; config.EconomyConfig): earners aged 15+ earn
    their earning potential times their health, less any working time lost
    to a care visit, times the household's persistent income shock (an AR(1)
    on the log scale, from the economy's own run-time random stream, so the
    epidemic's draws are unaffected); the household consumes its target out of income and cash,
    with cash clipped to [0, buffer]; WEALTH becomes the care-seeking wealth
    term W. Daily summaries (poverty, cash, by quintile) are kept in
    `daily_records` for experiments.metrics.economy_metrics. The legacy path
    below is unchanged.
    """

    def __init__(self, config):
        super().__init__(config)
        self._cedi = getattr(config, "economy", None) is not None and config.economy.model == "cedi"
        #: one dict of per-quintile arrays per day (cedi mode only)
        self.daily_records: list[dict] = []
        self._hh_index = None
        self._rep = None
        self._shock = None        # household log income shocks (cedi mode)
        self._shock_rng = None

    def update(self, agent_state: AgentState, **kwargs):
        if self._cedi:
            self._update_cedi(agent_state)
            return
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

    # ------------------------------------------------------------------
    # Cedi economy
    # ------------------------------------------------------------------

    def _households(self, agent_state: AgentState):
        """(agent -> household index, one representative agent per household,
        household sizes); household membership never changes within a run."""
        if self._hh_index is None:
            hh = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
            _, inv = torch.unique(hh, return_inverse=True)
            n = int(inv.max().item()) + 1
            rep = torch.full((n,), hh.numel(), dtype=torch.long, device=hh.device)
            rep.scatter_reduce_(0, inv, torch.arange(hh.numel(), device=hh.device), reduce="amin")
            self._hh_index, self._rep = inv, rep
            self._size = torch.zeros(n, device=hh.device).index_add_(0, inv, torch.ones(hh.numel(), device=hh.device))
        return self._hh_index, self._rep, self._size

    def _income_shock(self, n_households: int):
        """Today's multiplier exp(e_h - sd^2/2) on each household's earnings, mean 1;
        e_h is a stationary AR(1) with SD income_shock_sd and half-life
        income_shock_halflife_days. None when the SD is 0 (no draws)."""
        eco = self.config.economy
        sd = float(eco.income_shock_sd)
        if sd <= 0.0:
            return None
        if self._shock is None:
            self._shock_rng = np.random.default_rng(int(self.config.seed) + int(eco.rng_seed_offset) + 1)
            self._shock = self._shock_rng.normal(0.0, sd, n_households)
        else:
            rho = 0.5 ** (1.0 / float(eco.income_shock_halflife_days))
            self._shock = rho * self._shock + np.sqrt(1.0 - rho * rho) * sd * self._shock_rng.standard_normal(n_households)
        return torch.as_tensor(np.exp(self._shock - 0.5 * sd * sd), dtype=torch.float32)

    def _facility_cost(self, nd, rep):
        """Each household's out-of-pocket cost of a public-hospital visit (cached)."""
        if getattr(self, "_facility_cost_cache", None) is None:
            eco = self.config.economy
            insured = nd[AgentPropertyKeys.INSURED][rep]
            self._facility_cost_cache = torch.where(
                insured, torch.full(insured.shape, float(eco.cost_hospital_insured)),
                torch.full(insured.shape, float(eco.cost_facility_uninsured))).to(nd[AgentPropertyKeys.CASH].device)
        return self._facility_cost_cache

    def _update_cedi(self, agent_state: AgentState):
        from abm.economy.cedi import N_QUINTILES, wealth_term
        K = AgentPropertyKeys
        eco = self.config.economy
        nd = agent_state.ndata
        inv, rep, size = self._households(agent_state)

        earn = nd[K.EARNING_POTENTIAL] * (1.0 - nd[K.WORK_LOST])
        if self.config.steering_parameters.health_based_income:
            earn = earn * nd[K.HEALTH]
        income = torch.zeros(rep.numel(), device=earn.device).index_add_(0, inv, earn)
        shock = self._income_shock(rep.numel())
        if shock is not None:
            income = income * shock.to(income.device)

        cash = nd[K.CASH][rep]
        target = nd[K.CONSUMPTION_TARGET][rep]
        new_cash = torch.minimum(torch.clamp(cash + income - target, min=0.0), nd[K.CASH_CAP][rep])
        consumption = cash + income - new_cash

        nd[K.CASH] = new_cash[inv]
        nd[K.WEALTH] = wealth_term(new_cash, target, eco.wealth_scale_days)[inv]
        nd[K.WORK_LOST] = torch.zeros_like(nd[K.WORK_LOST])

        # --- daily summary, by quintile (population-weighted where per person) ---
        q = nd[K.WEALTH_QUINTILE][rep]
        per_capita = consumption / size

        def by_q(values):
            return torch.zeros(N_QUINTILES, dtype=torch.float64).index_add_(0, q.cpu(), values.double().cpu())

        self.daily_records.append({
            "households": by_q(torch.ones_like(size)),
            "persons": by_q(size),
            "poor_persons": by_q(size * (per_capita < eco.poverty_line_daily)),
            "extreme_poor_persons": by_q(size * (per_capita < eco.extreme_poverty_line_daily)),
            "below_target_households": by_q((consumption < target * (1.0 - 1e-6)).float()),
            "zero_cash_households": by_q((new_cash <= 0.0).float()),
            # could not pay today for a public-hospital visit at its NHIS status (the
            # model's analogue of DHS's "getting money for treatment is a big problem")
            "cannot_pay_facility_households": by_q((new_cash < self._facility_cost(nd, rep)).float()),
            "cash": by_q(new_cash),
            "cash_days": by_q(new_cash / torch.clamp(target, min=1e-9)),
            "w": by_q(wealth_term(new_cash, target, eco.wealth_scale_days)),
            "consumption_per_capita_x_persons": by_q(per_capita * size),
        })

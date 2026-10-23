# abm/pathogens/pathogen.py
from abc import ABC, abstractmethod
from typing import Any, Optional
import torch

from abm.state import AgentState
from abm.constants import Phase
from config import PathogenConfig, SteeringParamsSVEIR

class Pathogen(ABC):
    """Abstract Base Class for a pathogen in the SVEIR model."""

    #: Names of the transmission routes this pathogen attributes infections to.
    #: Subclasses with more than one route declare them here; the base class
    #: then maintains per-step and lifetime counters for each, so route
    #: attribution does not have to be reimplemented per pathogen (it used to
    #: be hand-rolled inside Campylobacter). A pathogen that does not
    #: distinguish routes can leave this empty and the counters stay empty.
    ROUTES: tuple[str, ...] = ()

    #: True for pathogens that maintain a per-pathogen water reservoir on the
    #: GridEnvironment (see abm/systems/environment.py, which loops over these
    #: rather than hardcoding one pathogen's water state).
    WATERBORNE: bool = False

    def __init__(self, config: PathogenConfig, global_params: SteeringParamsSVEIR, device: torch.device):
        self.config = config
        self.name = config.name
        self.global_params = global_params
        self.device = device
        self.new_cases_this_step = 0

        # Route attribution. cases_by_route is reset each day by reset_incidence();
        # total_by_route accumulates over the whole run and is what the
        # route-fraction metrics read.
        self.cases_by_route: dict[str, int] = {r: 0 for r in self.ROUTES}
        self.total_by_route: dict[str, int] = {r: 0 for r in self.ROUTES}
        # Per-day history of cases_by_route (one entry per simulated day,
        # opened by reset_incidence()), so route fractions can be computed
        # over any window - e.g. excluding the seeded initial wave.
        self.daily_by_route: dict[str, list[int]] = {r: [] for r in self.ROUTES}

        # Agents newly exposed earlier in the current simulated day. Pathogens
        # whose routes are evaluated more than once per day (or which have
        # several independent routes within one day) use this to make route
        # attribution unambiguous: an agent can be infected by at most one
        # route per day. Allocated lazily in begin_day(); pathogens that do not
        # opt in simply never consult it.
        self._newly_exposed_this_day: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Immunity
    # ------------------------------------------------------------------

    @property
    def immunity_decay_per_infection(self) -> float:
        """Per-prior-infection decay rate kappa in the susceptibility term
        exp(-kappa * N_infections).

        Defaults to the model-wide steering parameter
        (prior_infection_immunity_factor) so rotavirus and campylobacter keep
        sharing one value, but a pathogen whose acquired-immunity profile is
        separately evidenced can override it in its own config
        (PathogenConfig.immunity_decay_per_infection). Cryptosporidium does:
        protection against reinfection is well characterised and weaker-per-
        infection than the shared default implies.
        """
        override = getattr(self.config, "immunity_decay_per_infection", None)
        if override is not None:
            return float(override)
        return float(self.global_params.prior_infection_immunity_factor)

    def _immunity_factor(self, agent_state: AgentState, indices: torch.Tensor) -> torch.Tensor:
        """exp(-kappa * N_prior_infections) for the given agent indices.

        The acquired-immunity term alone. Transmission routes should use
        _relative_susceptibility(), which adds the immunity plateau, adults'
        fixed susceptibility and the infant ramp.
        """
        from abm.constants import AgentPropertyKeys
        count_key = AgentPropertyKeys.num_infections(self.name)
        num_prior = agent_state.ndata[count_key][indices].float()
        return torch.exp(-self.immunity_decay_per_infection * num_prior)

    #: Routes whose risk depends on the household's ENVIRONMENT (water,
    #: animals, food) rather than on contact with an infectious person. Only
    #: these are scaled by the household exposure multiplier.
    ENVIRONMENTAL_ROUTES: frozenset = frozenset({"water", "zoonotic", "food_borne"})

    def _relative_susceptibility(self, agent_state: AgentState, indices: torch.Tensor,
                                 route: Optional[str] = None) -> torch.Tensor:
        """Per-agent multiplier on infection probability, for every route.

            children: max(immunity_floor, exp(-kappa N)) * infant(age) * f
            adults:   immunity_floor * f
        (f only on environmental routes - see below)

        infant(age) = b + (1 - b) * min(1, age / R), with b =
        infant_susceptibility_at_birth and R = infant_susceptibility_ramp_months
        (1 everywhere when R = 0). f is the household environmental-exposure
        multiplier (EXPOSURE_FRAILTY), applied only on ENVIRONMENTAL_ROUTES and
        only when steering_parameters.household_exposure_sdlog > 0. Immunity
        (the max(...) term) and exposure (f) are separate factors: a
        low-exposure household is not pulled up to the floor, and a
        high-exposure adult is not capped at it.

        With immunity_floor None, no ramp and sdlog 0 this is exactly the
        original exp(-kappa N) for every agent, computed with the same
        operations, so outputs are bit-identical.

        Adults are held at the floor rather than accumulating immunity from
        in-simulation infections: they stand for people who have long since
        reached the plateau, and adults never leave the population, so letting
        their susceptibility keep falling would make the model drift
        indefinitely instead of reaching an endemic steady state.
        """
        from abm.constants import AgentPropertyKeys
        s = self._immunity_factor(agent_state, indices)
        floor = getattr(self.config, "immunity_floor", None)
        b = float(getattr(self.config, "infant_susceptibility_at_birth", 1.0))
        ramp = float(getattr(self.config, "infant_susceptibility_ramp_months", 0.0))
        has_ramp = ramp > 0.0 and b < 1.0
        if floor is not None or has_ramp:
            is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD][indices].bool()
            if floor is not None:
                s = torch.where(is_child, torch.clamp(s, min=float(floor)),
                                torch.full_like(s, float(floor)))
            if has_ramp:
                age = agent_state.ndata[AgentPropertyKeys.AGE][indices].float()
                infant = b + (1.0 - b) * torch.clamp(age / ramp, max=1.0)
                s = torch.where(is_child, s * infant, s)
        if (route in self.ENVIRONMENTAL_ROUTES
                and float(getattr(self.global_params, "household_exposure_sdlog", 0.0)) > 0.0):
            s = s * agent_state.ndata[AgentPropertyKeys.EXPOSURE_FRAILTY][indices]
        return s

    # ------------------------------------------------------------------
    # Per-day lifecycle
    # ------------------------------------------------------------------

    def reset_incidence(self):
        """Resets the incidence counters at the start of a simulation step."""
        self.new_cases_this_step = 0
        for route in self.cases_by_route:
            self.cases_by_route[route] = 0
            self.daily_by_route[route].append(0)

    def begin_day(self, agent_state: AgentState):
        """Allocates the per-day 'already exposed today' guard.

        Called from step_progression by pathogens that use multi-route
        attribution. Kept separate from reset_incidence() because
        reset_incidence() has no access to the agent state.
        """
        self._newly_exposed_this_day = torch.zeros(
            agent_state.num_nodes(), dtype=torch.bool, device=self.device
        )

    def susceptible_like_mask(self, agent_state: AgentState, respect_daily_guard: bool = False) -> torch.Tensor:
        """Agents eligible for a new infection: SUSCEPTIBLE or RECOVERED.

        Recovered agents are re-infection-eligible by design throughout this
        model - immunity is partial and never sterile (see
        docs/model_description.md). With respect_daily_guard=True, agents
        already infected earlier today are excluded.
        """
        from abm.constants import Compartment, AgentPropertyKeys
        status = agent_state.ndata[AgentPropertyKeys.status(self.name)]
        mask = (status == Compartment.SUSCEPTIBLE) | (status == Compartment.RECOVERED)
        if respect_daily_guard and self._newly_exposed_this_day is not None:
            mask = mask & ~self._newly_exposed_this_day
        return mask

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step_progression(self, agent_state: AgentState):
        """
        Updates internal disease states (timers, recovery).
        Should run once per full simulation day.
        """
        pass

    @abstractmethod
    def step_transmission(
        self,
        agent_state: AgentState,
        location_ids: torch.Tensor,
        num_locations: int,
        grid: Any,
        phase: Phase,
    ):
        """
        Calculates infections based on current agent locations.

        Runs twice per simulated day - once for the day phase (agents at their
        chosen activity location) and once for the night phase (agents at home).
        `phase` says which; see abm.constants.Phase.
        """
        pass

    # ------------------------------------------------------------------
    # Water reservoir interface (waterborne pathogens only)
    # ------------------------------------------------------------------

    def init_water_reservoir(self, grid: Any):
        """Seeds this pathogen's own water-contamination state on the grid.

        Only called for WATERBORNE pathogens, once per initialize_model().
        """
        raise NotImplementedError

    def decay_water_reservoir(self, grid: Any):
        """Advances this pathogen's water state one day (recovery / die-off)."""
        raise NotImplementedError

    def apply_water_shock(self, grid: Any):
        """Applies a whole-supply contamination shock to this pathogen's water state."""
        raise NotImplementedError

    # --- Catchment water model (WaterConfig.model == "catchment") ---
    # When True (set by SVEIRModel), the pathogen's own location-based water
    # routes are switched off and the WaterNetwork drives loading and
    # exposure instead (abm/environment/water_network.py).
    use_catchment_water: bool = False

    @property
    def water_decay_rate(self) -> float:
        """Daily first-order die-off rate in water (catchment model)."""
        raise NotImplementedError

    def water_infection_prob(self, concentration: torch.Tensor) -> torch.Tensor:
        """P(infection) from one day's drinking water at this concentration
        (catchment model; units: infectious-person-days of shedding reaching
        the source, dilution-weighted)."""
        raise NotImplementedError

    def drinking_water_exposure(self, agent_state: AgentState, concentration: torch.Tensor):
        """Daily drinking-water exposure of every agent (catchment model).

        `concentration` is per agent (their household source's dilution-weighted
        concentration). Susceptible and recovered agents face
        water_infection_prob x relative susceptibility on the "water" route;
        subclasses with a vaccinated compartment extend this.
        """
        target_mask = self.susceptible_like_mask(agent_state, respect_daily_guard=True) & (concentration > 0)
        if not torch.any(target_mask):
            return
        idx = target_mask.nonzero(as_tuple=True)[0]
        prob = self.water_infection_prob(concentration[idx]) * self._relative_susceptibility(
            agent_state, idx, route="water"
        )
        self._apply_bernoulli_infections(agent_state, target_mask, prob, route="water")

    def water_carrying_exposure(
        self,
        agent_state: AgentState,
        grid: Any,
        household_ids: torch.Tensor,
        source_values: torch.Tensor,
    ):
        """Infects members of households that fetched water today.

        Driven by HouseholdSystem, which knows which agents went to fetch water
        and from where; the pathogen supplies the risk model. `household_ids`
        holds one (ascending-sorted) id per water-fetching household and
        `source_values` the value of THIS pathogen's water state at that
        household's source - a WaterStatus flag for rotavirus, an oocyst load
        for Cryptosporidium.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Importation (infection acquired outside the modelled population)
    # ------------------------------------------------------------------

    def _imported_infections(self, agent_state: AgentState):
        """PathogenConfig.importation_rate x relative susceptibility, route
        "imported". No draw - and no RNG consumed - at the default rate 0."""
        rate = float(getattr(self.config, "importation_rate", 0.0))
        if rate <= 0.0:
            return
        target_mask = self.susceptible_like_mask(agent_state, respect_daily_guard=True)
        if not torch.any(target_mask):
            return
        idx = target_mask.nonzero(as_tuple=True)[0]
        prob = rate * self._relative_susceptibility(agent_state, idx, route="imported")
        self._apply_bernoulli_infections(agent_state, target_mask, prob, route="imported")

    # ------------------------------------------------------------------
    # Shared progression helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Infectiousness
    # ------------------------------------------------------------------

    def infectiousness_weights(self, agent_state: AgentState) -> torch.Tensor:
        """Per-agent infectiousness for every transmission route: 1 for an
        INFECTIOUS agent whose current infection produced an illness episode,
        asymptomatic_relative_infectiousness for any other INFECTIOUS agent
        (every adult, and children's subclinical infections), 0 otherwise.

        At the default relative infectiousness of 1.0 this is exactly the
        float of the INFECTIOUS mask - the original count of infectious
        agents - so every route is unchanged bit for bit.
        """
        from abm.constants import Compartment, AgentPropertyKeys
        is_infectious = (agent_state.ndata[AgentPropertyKeys.status(self.name)] == Compartment.INFECTIOUS).float()
        rho = float(getattr(self.config, "asymptomatic_relative_infectiousness", 1.0))
        if rho >= 1.0:
            return is_infectious
        sym = agent_state.ndata.get(AgentPropertyKeys.symptomatic_infection(self.name))
        if sym is None:
            return is_infectious * rho
        return is_infectious * torch.where(sym.bool(), torch.ones_like(is_infectious),
                                           torch.full_like(is_infectious, rho))

    def household_infectiousness(self, agent_state: AgentState):
        """(household ids with an infectious member, the largest infectiousness
        weight among each one's infectious members). For the presence-based
        household routes: a household whose infectious members are all
        symptomless transmits at asymptomatic_relative_infectiousness times
        the per-contact probability. Returns (None, None) if nobody is
        infectious."""
        from abm.constants import AgentPropertyKeys
        w = self.infectiousness_weights(agent_state)
        infectious = w > 0
        if not torch.any(infectious):
            return None, None
        hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        hh_inf, inv = torch.unique(hh_ids[infectious], return_inverse=True)
        w_max = torch.zeros(len(hh_inf), dtype=w.dtype).scatter_reduce_(
            0, inv, w[infectious], reduce="amax", include_self=False)
        return hh_inf, w_max

    def _increment_exposure_time(self, agent_state: AgentState):
        from abm.constants import Compartment, AgentPropertyKeys
        status_key = AgentPropertyKeys.status(self.name)
        timer_key = AgentPropertyKeys.exposure_time(self.name)
        exposed_mask = agent_state.ndata[status_key] == Compartment.EXPOSED
        agent_state.ndata[timer_key][exposed_mask] += 1

    def _exposed_to_infectious(self, agent_state: AgentState):
        """
        Transitions agents from EXPOSED → INFECTIOUS once the exposure period
        has elapsed.
        """
        from abm.constants import Compartment, AgentPropertyKeys
        status_key = AgentPropertyKeys.status(self.name)
        timer_key = AgentPropertyKeys.exposure_time(self.name)

        mask = (
            (agent_state.ndata[status_key] == Compartment.EXPOSED)
            & (agent_state.ndata[timer_key] >= self.config.exposure_period)
        )
        if torch.any(mask):
            agent_state.ndata[status_key][mask] = Compartment.INFECTIOUS

    def _infectious_to_recovered(self, agent_state: AgentState):
        from abm.constants import Compartment, AgentPropertyKeys
        status_key = AgentPropertyKeys.status(self.name)
        infectious_mask = agent_state.ndata[status_key] == Compartment.INFECTIOUS
        if not torch.any(infectious_mask):
            return

        recovery_chance = torch.rand(torch.sum(infectious_mask), device=self.device)
        recovered_mask = recovery_chance < self.config.recovery_rate

        agents_to_recover = infectious_mask.nonzero(as_tuple=True)[0][recovered_mask]
        agent_state.ndata[status_key][agents_to_recover] = Compartment.RECOVERED

    # ------------------------------------------------------------------
    # Shared infection commit
    # ------------------------------------------------------------------

    def _commit_infections(
        self,
        agent_state: AgentState,
        infected_indices: torch.Tensor,
        route: Optional[str] = None,
    ) -> int:
        """Moves the given agents into EXPOSED and does all the bookkeeping.

        Single place where a new infection is recorded: compartment, exposure
        timer, lifetime infection count, incidence, per-route attribution and
        the per-day 'already exposed today' guard. Returns the number of new
        infections committed.

        Every route of every pathogen funnels through here so route counters
        can never drift out of step with new_cases_this_step - something that
        had to be maintained by hand in four separate places before.
        """
        from abm.constants import Compartment, AgentPropertyKeys

        num_new = int(infected_indices.numel())
        if num_new == 0:
            return 0

        status_key = AgentPropertyKeys.status(self.name)
        agent_state.ndata[status_key][infected_indices] = Compartment.EXPOSED
        agent_state.ndata[AgentPropertyKeys.exposure_time(self.name)][infected_indices] = 0
        agent_state.ndata[AgentPropertyKeys.num_infections(self.name)][infected_indices] += 1
        sym_key = AgentPropertyKeys.symptomatic_infection(self.name)
        if sym_key in agent_state.ndata:
            agent_state.ndata[sym_key][infected_indices] = False

        self.new_cases_this_step += num_new
        if route is not None and route in self.cases_by_route:
            self.cases_by_route[route] += num_new
            self.total_by_route[route] += num_new
            if self.daily_by_route[route]:
                self.daily_by_route[route][-1] += num_new
        if self._newly_exposed_this_day is not None:
            self._newly_exposed_this_day[infected_indices] = True
        return num_new

    def _apply_bernoulli_infections(
        self,
        agent_state: AgentState,
        target_mask: torch.Tensor,
        per_agent_prob: torch.Tensor | float,
        route: Optional[str] = None,
    ) -> int:
        """One independent Bernoulli trial per targeted agent.

        `per_agent_prob` is either a scalar or a tensor already restricted to
        the targeted agents (in target-index order). Immunity is NOT applied
        here - callers pass the already-adjusted probability, because different
        routes combine immunity with their own dose model differently.
        """
        if not torch.any(target_mask):
            return 0
        target_indices = target_mask.nonzero(as_tuple=True)[0]
        rand_vals = torch.rand(len(target_indices), device=self.device)
        newly_infected = rand_vals < per_agent_prob
        return self._commit_infections(agent_state, target_indices[newly_infected], route=route)

    def _apply_new_infections(
        self,
        agent_state: AgentState,
        target_nodes_mask: torch.Tensor,
        base_prob: float,
        location_ids: Optional[torch.Tensor] = None,
        num_locations: Optional[int] = None,
        forced_pressure: Optional[torch.Tensor] = None,
        prob_multiplier: float = 1.0,
        route: Optional[str] = None,
        exclude_same_household: bool = False,
    ):
        """
        Calculates and applies new infections using vectorised scatter/gather ops.

        Susceptibility is _relative_susceptibility() (prior-infection
        immunity, the adult plateau and the infant ramp). Health is
        intentionally excluded here: its effect on outcomes is
        captured through care-seeking and economic dynamics only.

        exclude_same_household
            When True, infectious agents of the target's OWN household are
            removed from the co-location pressure it experiences. This matters
            for a pathogen that ALSO has an explicit household route: agents
            who spend the day at home are co-located with their own household,
            so without this their within-household exposure would be counted
            twice - once by the household route and once by the co-location
            route - and, worse, would be ATTRIBUTED to the community route,
            making the route-fraction metrics misreport which setting
            transmission actually happened in. Visitors from other households
            (MovementSystem's social visits) still contribute, so this does not
            simply switch off transmission for agents who stay home.
            Rotavirus leaves this False: it has no separate household route, so
            its night-phase co-location pressure IS its household route.
        """
        from abm.constants import Compartment, AgentPropertyKeys
        if not torch.any(target_nodes_mask):
            return

        status_key = AgentPropertyKeys.status(self.name)
        target_indices = target_nodes_mask.nonzero(as_tuple=True)[0]

        # --- Calculate Infection Pressure ---
        if forced_pressure is not None:
            infection_pressure = forced_pressure[target_indices]
        elif location_ids is not None and num_locations is not None:
            # Infectiousness-weighted count of infectious agents (exactly the
            # count when asymptomatic_relative_infectiousness is 1).
            is_infectious = self.infectiousness_weights(agent_state)
            infected_per_location = torch.zeros(num_locations, device=self.device)
            infected_per_location.index_add_(0, location_ids, is_infectious)
            pressure_per_agent = infected_per_location[location_ids]
            if exclude_same_household:
                # Subtract infectious agents sharing BOTH the cell and the
                # household. That also removes the agent itself (an agent is
                # always in its own household), so the plain self-subtraction
                # below is not applied in this branch.
                hh = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
                _, cell_hh_ids = torch.unique(
                    location_ids * (int(hh.max().item()) + 1) + hh.long(),
                    return_inverse=True,
                )
                n_cell_hh = torch.zeros(int(cell_hh_ids.max().item()) + 1, device=self.device)
                n_cell_hh.index_add_(0, cell_hh_ids, is_infectious)
                pressure_per_agent = pressure_per_agent - n_cell_hh[cell_hh_ids]
            else:
                # Exclude the agent themselves from the pressure they experience
                pressure_per_agent = pressure_per_agent - is_infectious
            infection_pressure = pressure_per_agent[target_indices]
        else:
            raise ValueError("Must provide either (location_ids, num_locations) or forced_pressure.")

        if torch.sum(infection_pressure) == 0:
            return

        # --- Individual susceptibility: immunity, adult plateau, infant ramp ---
        immunity_factor = self._relative_susceptibility(agent_state, target_indices, route=route)

        final_prob = prob_multiplier * base_prob * immunity_factor
        final_prob.clamp_(0.0, 1.0)

        # P(infection) = 1 - (1 - p)^k
        prob_not_infected = (1 - final_prob) ** infection_pressure
        prob_getting_infected = 1 - prob_not_infected

        # --- Apply Infection ---
        random_samples = torch.rand(len(target_indices), device=self.device)
        newly_infected_mask = random_samples < prob_getting_infected
        self._commit_infections(agent_state, target_indices[newly_infected_mask], route=route)

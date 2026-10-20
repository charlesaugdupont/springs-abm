# abm/pathogens/cryptosporidium.py
"""
Cryptosporidium spp. transmission model.

Why this pathogen is structured differently from the other two
---------------------------------------------------------------
Cryptosporidiosis in rural sub-Saharan Africa - including Ghana - is
overwhelmingly ANTHROPONOTIC, which is the opposite of campylobacter in this
model, and it is unusually well evidenced. The route hierarchy this
implementation targets is therefore

    household person-to-person > community person-to-person > waterborne > zoonotic

See CryptosporidiumConfig in config.py for the full citation list behind that
ordering (Krumkamp et al. 2020 is the key study: contact-network tracing in
Gabon, Ghana, Madagascar and Tanzania found elevated infection risk among
household members (RR 3.6) and neighbouring children (RR 2.9) but not among
animal contacts (RR 1.2, n.s.)).

Two features have no analogue in rotavirus or campylobacter and drove the
framework refactors:

* MOST CHILDHOOD INFECTIONS ARE SUBCLINICAL (~2/3), yet asymptomatic carriers
  shed for weeks and transmit normally - the "silent reservoir" that sustains
  community transmission. This is handled by
  PathogenConfig.symptomatic_fraction, consumed by ChildIllnessSystem: an
  infection that loses the symptomatic draw still progresses
  S/R -> E -> I -> R and still infects others, it simply never acquires a
  severity or an illness duration, so it generates no health loss, no episode
  log entry and no care-seeking decision.

* OOCYSTS PERSIST IN WATER FOR WEEKS AND ACCUMULATE. Rotavirus's water model is
  a binary CLEAN/CONTAMINATED flag with a ~5-day memory, which cannot represent
  a slowly-decaying environmental load. Cryptosporidium therefore owns a
  CONTINUOUS reservoir per water source, measured in infectious-agent-days of
  shedding, decaying first-order at `oocyst_decay_rate`, and converted to an
  infection probability through the standard exponential dose-response
  P(inf | dose) = 1 - exp(-r * dose). The water subsystem
  (GridEnvironment water-state layers + abm/systems/environment.py) is
  per-pathogen precisely so these two models can coexist.

The four routes
---------------
1. Household person-to-person (NIGHT phase, once per day)
   Every susceptible member of a household containing an infectious member
   faces a Bernoulli trial at `household_transmission_prob`. The dominant
   route, anchored on measured household secondary attack rates.

2. Community person-to-person (DAY phase only)
   Co-location pressure at the agent's daytime activity location, using the
   same machinery as rotavirus H2H, so schools, places of worship, water
   points and social visits all mix the population.
   Deliberately DAY-only: at night every household occupies its own
   residence cell, so a co-location route evaluated in the night phase would
   be a second, duplicate household route.

3. Waterborne (DAY phase; infectious agents shed at the source they visit,
   susceptible visitors ingest a dose) plus a household channel for water
   CARRIED home, driven by HouseholdSystem.

4. Zoonotic, ruminant-only (both phases)
   Reads the same ruminant density layer campylobacter uses, so both pathogens
   see one consistent livestock landscape. Poultry are excluded: birds are not
   hosts of C. parvum / C. hominis. Expected to remain a minor contributor.

Route attribution is handled generically by the Pathogen base class; an agent
can be infected by at most one route per simulated day.
"""

from typing import Any
import torch

from .pathogen import Pathogen
from abm.state import AgentState
from abm.constants import Compartment, AgentPropertyKeys, Phase
from config import CryptosporidiumConfig, SteeringParamsSVEIR


class Cryptosporidium(Pathogen):
    """Implements the logic for Cryptosporidium spp."""

    ROUTES = ("household", "community", "water", "zoonotic", "imported")
    WATERBORNE = True

    def __init__(
        self,
        config: CryptosporidiumConfig,
        global_params: SteeringParamsSVEIR,
        device: torch.device,
    ):
        super().__init__(config, global_params, device)
        self.config: CryptosporidiumConfig = config
        # Cached boolean mask of cells containing a water source, and the
        # ruminant-density field scaled by its risk weight. Both are constant
        # within a run (the density layers are built once per
        # initialize_model() and the weight is a config value), so they are
        # computed once rather than on every twice-daily transmission call -
        # mirroring Campylobacter._combined_animal_density.
        self._water_mask: torch.Tensor | None = None
        self._ruminant_field_cache: torch.Tensor | None = None
        self._ruminant_field_key: float | None = None

    # ------------------------------------------------------------------
    # Dose-response
    # ------------------------------------------------------------------

    def _infection_prob_from_dose(self, dose: torch.Tensor) -> torch.Tensor:
        """Exponential dose-response: P(inf | dose) = 1 - exp(-r * dose).

        The standard model for Cryptosporidium in quantitative microbial risk
        assessment. `dose` is in model units, not absolute oocysts - the
        absolute scaling lives in `water_dose_scale` / `zoonotic_interaction_
        rate`, exactly as it lives in `human_animal_interaction_rate` for
        campylobacter's Beta-Poisson route. See CryptosporidiumConfig.
        dose_response_r for the challenge-study ID50s behind r.
        """
        return 1.0 - torch.exp(-self.config.dose_response_r * dose)

    # ------------------------------------------------------------------
    # Pathogen interface
    # ------------------------------------------------------------------

    def step_progression(self, agent_state: AgentState):
        """Disease-state progression - called once per day."""
        self.begin_day(agent_state)
        # Recovery is evaluated BEFORE the exposed->infectious transition, for
        # the same reason as the other two pathogens: otherwise an agent that
        # becomes infectious on this call would face a same-day recovery roll
        # before ever reaching a transmission phase, and could skip its
        # infectious period entirely.
        self._increment_exposure_time(agent_state)
        self._infectious_to_recovered(agent_state)
        self._exposed_to_infectious(agent_state)
        self._imported_infections(agent_state)

    def step_transmission(
        self,
        agent_state: AgentState,
        location_ids: torch.Tensor,
        num_locations: int,
        grid: Any,
        phase: Phase,
    ):
        if phase is Phase.DAY:
            self._community_transmission(agent_state, location_ids, num_locations)
            if not self.use_catchment_water:   # legacy point-source water model
                self._shed_into_water(agent_state, grid)
                self._water_to_human(agent_state, grid)
        else:
            self._household_transmission(agent_state)

        # Livestock proximity applies wherever the agent currently is, so this
        # route fires in both phases (as campylobacter's does).
        self._zoonotic_transmission(agent_state, grid)

    # ------------------------------------------------------------------
    # Route 1: household person-to-person
    # ------------------------------------------------------------------

    def _household_transmission(self, agent_state: AgentState):
        """Within-household person-to-person transmission (dominant route).

        Mechanically a per-susceptible-member Bernoulli trial in every
        household holding at least one infectious member. The probability is
        anchored on measured household secondary attack rates rather than being
        a free parameter - see CryptosporidiumConfig.household_transmission_prob
        (Korpe et al. 2018: 8-week SAR 35.8% urban / 7.8% rural, vs 0% in
        control households).
        """
        hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        status_key = AgentPropertyKeys.status(self.name)

        is_infectious = agent_state.ndata[status_key] == Compartment.INFECTIOUS
        if not torch.any(is_infectious):
            return

        infectious_hh = torch.unique(hh_ids[is_infectious])
        target_mask = (
            self.susceptible_like_mask(agent_state, respect_daily_guard=True)
            & torch.isin(hh_ids, infectious_hh)
        )
        if not torch.any(target_mask):
            return

        target_indices = target_mask.nonzero(as_tuple=True)[0]
        prob = self.config.household_transmission_prob * self._relative_susceptibility(
            agent_state, target_indices, route="household"
        )
        self._apply_bernoulli_infections(agent_state, target_mask, prob, route="household")

    # ------------------------------------------------------------------
    # Route 2: community person-to-person
    # ------------------------------------------------------------------

    def _community_transmission(
        self, agent_state: AgentState, location_ids: torch.Tensor, num_locations: int
    ):
        """Co-location transmission at the daytime activity location.

        Carries the neighbouring-children signal in Krumkamp et al. 2020
        (RR 2.9; 23% prevalence among neighbouring children under 5 - the
        highest of any contact group sampled, consistent with a large
        asymptomatic community reservoir).

        Same-household contacts are excluded from the pressure (see
        Pathogen._apply_new_infections' exclude_same_household): children spend
        most of the day at home, so without that exclusion within-household
        exposure would be double-counted against the dedicated household route
        AND mislabelled as community transmission, which would corrupt the
        route-attribution metrics this pathogen is calibrated against. Social
        visitors from other households still count.
        """
        target_mask = self.susceptible_like_mask(agent_state, respect_daily_guard=True)
        self._apply_new_infections(
            agent_state,
            target_nodes_mask=target_mask,
            base_prob=self.config.community_infection_prob,
            location_ids=location_ids,
            num_locations=num_locations,
            route="community",
            exclude_same_household=True,
        )

    # ------------------------------------------------------------------
    # Route 3: waterborne, via a persistent oocyst reservoir
    # ------------------------------------------------------------------

    def _cached_water_mask(self, grid: Any) -> "torch.Tensor | None":
        if self._water_mask is None and grid is not None:
            self._water_mask = grid.water_point_mask()
        return self._water_mask

    def init_water_reservoir(self, grid: Any):
        """Creates an empty oocyst-load field (infectious-agent-days of shedding)."""
        if grid is None:
            return
        mask = grid.water_point_mask()
        if mask is None:
            return
        self._water_mask = mask
        grid.set_water_state(
            self.name, torch.zeros(mask.shape, dtype=torch.float32, device=self.device)
        )

    def decay_water_reservoir(self, grid: Any):
        """First-order oocyst die-off: load *= exp(-k).

        Deterministic, unlike rotavirus's stochastic per-source recovery roll -
        a concentration decays, it does not flip state. Note this consumes NO
        random numbers, which is what keeps rotavirus's RNG stream unchanged
        when both pathogens are active.
        """
        state = grid.get_water_state(self.name)
        if state is None:
            return
        state.mul_(float(torch.exp(torch.tensor(-self.config.oocyst_decay_rate))))

    def apply_water_shock(self, grid: Any):
        """Raises every water source to `shock_oocyst_load` if it is lower.

        The continuous-reservoir analogue of rotavirus's "every clean source
        becomes contaminated": a flood or infrastructure failure loads all
        sources at once. Sources already dirtier than the shock level are left
        alone rather than being reduced.
        """
        state = grid.get_water_state(self.name)
        mask = self._cached_water_mask(grid)
        if state is None or mask is None:
            return
        state[mask] = torch.clamp(state[mask], min=self.config.shock_oocyst_load)

    def _shed_into_water(self, agent_state: AgentState, grid: Any):
        """Infectious agents standing at a water source add to its oocyst load.

        Matches rotavirus's "contamination happens at the source the agent
        visits" convention. A limitation worth stating: real contamination of
        a surface source is dominated by open defecation in the catchment and
        runoff rather than by infectious people being at the water point, and
        this model has no catchment-runoff component.
        """
        if grid is None:
            return
        state = grid.get_water_state(self.name)
        mask = self._cached_water_mask(grid)
        if state is None or mask is None:
            return

        status_key = AgentPropertyKeys.status(self.name)
        infectious = agent_state.ndata[status_key] == Compartment.INFECTIOUS
        if not torch.any(infectious):
            return

        y = agent_state.ndata[AgentPropertyKeys.Y].long()
        x = agent_state.ndata[AgentPropertyKeys.X].long()
        at_water = mask[y, x]
        shedders = infectious & at_water
        if not torch.any(shedders):
            return

        cols = state.shape[1]
        flat_idx = (y[shedders] * cols + x[shedders])
        contribution = torch.full(
            (int(flat_idx.numel()),),
            self.config.oocyst_shedding_rate,
            dtype=state.dtype,
            device=self.device,
        )
        state.view(-1).index_add_(0, flat_idx, contribution)

    def _water_to_human(self, agent_state: AgentState, grid: Any):
        """Susceptible agents at a water source ingest a dose of oocysts."""
        if grid is None:
            return
        state = grid.get_water_state(self.name)
        mask = self._cached_water_mask(grid)
        if state is None or mask is None:
            return

        y = agent_state.ndata[AgentPropertyKeys.Y].long()
        x = agent_state.ndata[AgentPropertyKeys.X].long()
        at_water = mask[y, x]
        target_mask = self.susceptible_like_mask(agent_state, respect_daily_guard=True) & at_water
        if not torch.any(target_mask):
            return

        target_indices = target_mask.nonzero(as_tuple=True)[0]
        load = state[y[target_indices], x[target_indices]]
        dose = load * self.config.water_dose_scale
        prob = self._infection_prob_from_dose(dose) * self._relative_susceptibility(
            agent_state, target_indices, route="water"
        )
        self._apply_bernoulli_infections(agent_state, target_mask, prob, route="water")

    def water_carrying_exposure(
        self,
        agent_state: AgentState,
        grid: Any,
        household_ids: torch.Tensor,
        source_values: torch.Tensor,
    ):
        """Exposes households that carried water home from a loaded source.

        HouseholdSystem decides WHICH households fetched water and supplies the
        oocyst load at each household's own source. Unlike rotavirus - whose
        source state is a binary flag, so every carrying household faces the
        same probability - the dose here scales with how loaded that particular
        source was, so the per-household load is carried all the way through to
        the dose-response.

        `household_ids` is sorted ascending (HouseholdSystem builds it with
        torch.unique), which is what makes the searchsorted lookup below valid.
        """
        loaded = source_values > 0
        if not torch.any(loaded):
            return
        hh_ids = household_ids[loaded].long()
        loads = source_values[loaded]

        agent_hh = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        target_mask = self.susceptible_like_mask(agent_state, respect_daily_guard=True) & torch.isin(
            agent_hh, hh_ids
        )
        if not torch.any(target_mask):
            return

        target_indices = target_mask.nonzero(as_tuple=True)[0]

        # Map each targeted agent to the load of its own household's source.
        pos = torch.searchsorted(hh_ids, agent_hh[target_indices].long())
        pos = pos.clamp(max=hh_ids.numel() - 1)
        agent_load = loads[pos]

        dose = agent_load * self.config.water_dose_scale
        prob = self._infection_prob_from_dose(dose) * self._relative_susceptibility(
            agent_state, target_indices, route="water"
        )
        self._apply_bernoulli_infections(agent_state, target_mask, prob, route="water")

    # --- Catchment water model ---

    @property
    def water_decay_rate(self) -> float:
        return float(self.config.oocyst_decay_rate)

    def water_infection_prob(self, concentration: torch.Tensor) -> torch.Tensor:
        """Exponential dose-response on dose = water_dose_scale * C."""
        return self._infection_prob_from_dose(concentration * self.config.water_dose_scale)

    # ------------------------------------------------------------------
    # Route 4: zoonotic (ruminant only)
    # ------------------------------------------------------------------

    def _ruminant_field(self, grid: Any) -> "torch.Tensor | None":
        """Ruminant density scaled by its risk weight, cached.

        Reads the SAME `ruminant_density` dynamic layer campylobacter uses, so
        both pathogens see one consistent livestock landscape derived from the
        same household ownership draw. The poultry layer is deliberately not
        read: birds are not hosts of C. parvum / C. hominis.
        """
        if grid is None:
            return None
        ruminant = grid.get_dynamic_layer("ruminant_density")
        if ruminant is None:
            return None
        key = float(self.config.ruminant_weight)
        if self._ruminant_field_cache is None or self._ruminant_field_key != key:
            self._ruminant_field_cache = ruminant * key
            self._ruminant_field_key = key
        return self._ruminant_field_cache

    def _zoonotic_transmission(self, agent_state: AgentState, grid: Any):
        """Exponential dose-response infection from local ruminant density.

        Uses the same dose_response_r as the waterborne route - both represent
        ingestion of oocysts, so one dose-response curve is more coherent than
        fitting a second functional form, and it avoids introducing another
        pair of unidentifiable shape parameters.
        """
        field = self._ruminant_field(grid)
        if field is None:
            return

        grid_shape = grid.grid_shape
        y = agent_state.ndata[AgentPropertyKeys.Y].long().clamp(0, grid_shape[0] - 1)
        x = agent_state.ndata[AgentPropertyKeys.X].long().clamp(0, grid_shape[1] - 1)

        target_mask = self.susceptible_like_mask(agent_state, respect_daily_guard=True)
        if not torch.any(target_mask):
            return

        target_indices = target_mask.nonzero(as_tuple=True)[0]
        dose = field[y[target_indices], x[target_indices]] * self.config.zoonotic_interaction_rate
        prob = self._infection_prob_from_dose(dose) * self._relative_susceptibility(
            agent_state, target_indices, route="zoonotic"
        )
        self._apply_bernoulli_infections(agent_state, target_mask, prob, route="zoonotic")

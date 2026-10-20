# abm/pathogens/rotavirus.py
from typing import Any
import torch

from .pathogen import Pathogen
from abm.state import AgentState
from abm.constants import Compartment, AgentPropertyKeys, Phase, WaterStatus
from config import RotavirusConfig, SteeringParamsSVEIR

class Rotavirus(Pathogen):
    """Implements the logic for Rotavirus, a viral, H2H-driven pathogen.

    Water model: a BINARY per-source state (WaterStatus.CLEAN /
    WaterStatus.CONTAMINATED) with a flat daily chance of reverting to clean
    (steering_parameters.water_recovery_prob), i.e. a ~5-day environmental
    memory. This is unchanged from the original implementation and remains the
    calibrated, validated model for rotavirus; it is deliberately NOT unified
    with Cryptosporidium's continuous oocyst reservoir, which represents a
    pathogen whose environmental stage accumulates and decays over weeks. The
    two coexist because the water subsystem is now per-pathogen (see
    GridEnvironment.set_water_state and abm/systems/environment.py).
    """

    ROUTES = ("h2h", "water", "imported")
    WATERBORNE = True

    def __init__(self, config: RotavirusConfig, global_params: SteeringParamsSVEIR, device: torch.device):
        super().__init__(config, global_params, device)
        self.config: RotavirusConfig = config
        self.current_infection_prob = 0.0

    def step_progression(self, agent_state: AgentState):
        """Internal state progression (Once per day)."""
        # 1. Update internal disease progression. Recovery must be evaluated
        # BEFORE the exposed->infectious transition: otherwise agents that
        # become infectious this call would also face a same-day recovery
        # roll, before ever reaching a transmission phase, letting some
        # infections skip the infectious period entirely.
        self._increment_exposure_time(agent_state)
        self._infectious_to_recovered(agent_state)
        self._exposed_to_infectious(agent_state)

        # 2. Update population states (vaccination)
        self._susceptible_to_vaccinated(agent_state)

        # 3. Update stochastic global infection probability for the day
        self._update_infection_probability()

        # 4. Infections acquired outside the modelled population.
        self._imported_infections(agent_state)

    def step_transmission(self, agent_state: AgentState, location_ids: torch.Tensor, num_locations: int, grid: Any, phase: Phase):
        """Transmission logic (Run for Day and Night phases)."""

        # 1. H2H Transmission
        self._susceptible_to_exposed_h2h(agent_state, location_ids, num_locations)
        self._vaccinated_to_exposed_h2h(agent_state, location_ids, num_locations)

        # 2. Waterborne Transmission (legacy point-source model only; the
        # catchment model is driven by EnvironmentSystem/HouseholdSystem).
        # Water transmission only happens if agents are actually AT the water source.
        # In Night phase (Reset to Home), no one is at water, so this naturally results in 0 infections, which is correct.
        if not self.use_catchment_water:
            self._human_to_water(agent_state, grid)
            self._water_to_human(agent_state, grid)

    # ------------------------------------------------------------------
    # Catchment water model
    # ------------------------------------------------------------------

    @property
    def water_decay_rate(self) -> float:
        return float(self.config.water_decay_rate)

    def water_infection_prob(self, concentration: torch.Tensor) -> torch.Tensor:
        """1 - exp(-water_dose_scale * C): the low-dose (linear) limit of the
        Beta-Poisson rotavirus dose-response (Ward et al. 1986), which is the
        regime a daily drinking-water dose is in; the absolute scale is the
        calibrated water_dose_scale."""
        return 1.0 - torch.exp(-self.config.water_dose_scale * concentration)

    def drinking_water_exposure(self, agent_state: AgentState, concentration: torch.Tensor):
        super().drinking_water_exposure(agent_state, concentration)
        # Vaccinated children: breakthrough at (1 - vaccine_efficacy), as on
        # the other rotavirus routes.
        status = agent_state.ndata[AgentPropertyKeys.status(self.name)]
        vac = (status == Compartment.VACCINATED) & (concentration > 0)
        if not torch.any(vac):
            return
        idx = vac.nonzero(as_tuple=True)[0]
        prob = (self.water_infection_prob(concentration[idx])
                * self._relative_susceptibility(agent_state, idx, route="water")
                * (1.0 - self.config.vaccine_efficacy))
        self._apply_bernoulli_infections(agent_state, vac, prob, route="water")

    # ------------------------------------------------------------------
    # Water reservoir (binary CLEAN / CONTAMINATED)
    # ------------------------------------------------------------------

    def init_water_reservoir(self, grid: Any):
        """Seeds rotavirus's water state from the static water-point mask.

        The grid generator marks a water-source cell with 1, which is
        deliberately the same value as WaterStatus.CLEAN, so a copy of that
        layer is exactly the initial state the original code started from
        (0 = no source, 1 = source and clean).
        """
        if grid is None:
            return
        mask = grid.water_point_mask()
        if mask is None:
            return
        state = torch.zeros(mask.shape, dtype=torch.int64, device=self.device)
        state[mask] = int(WaterStatus.CLEAN)
        grid.set_water_state(self.name, state)

    def decay_water_reservoir(self, grid: Any):
        """Models the natural recovery of contaminated water sources."""
        state = grid.get_water_state(self.name)
        if state is None:
            return

        infected_mask = state == WaterStatus.CONTAMINATED
        if not torch.any(infected_mask):
            return

        prob = self.global_params.water_recovery_prob
        chance = torch.rand(torch.sum(infected_mask), device=self.device)
        success = chance < prob

        coords_to_recover = infected_mask.nonzero(as_tuple=True)
        recovered_coords = (coords_to_recover[0][success], coords_to_recover[1][success])

        if len(recovered_coords[0]) > 0:
            state[recovered_coords] = WaterStatus.CLEAN

    def apply_water_shock(self, grid: Any):
        """Contaminates all currently clean water sources."""
        state = grid.get_water_state(self.name)
        if state is None:
            return
        clean_mask = state == WaterStatus.CLEAN
        if not torch.any(clean_mask):
            return
        state[clean_mask] = WaterStatus.CONTAMINATED

    def water_carrying_exposure(
        self,
        agent_state: AgentState,
        grid: Any,
        household_ids: torch.Tensor,
        source_values: torch.Tensor,
    ):
        """Exposes members of households that fetched contaminated water today.

        Called by HouseholdSystem, which supplies one row per water-fetching
        household together with this pathogen's own water-state value at that
        household's source. Rotavirus's state is a binary flag, so every
        household whose source was CONTAMINATED faces the same probability
        (full for S/R, reduced by vaccine efficacy for V).
        """
        contaminated_hh_ids = household_ids[source_values == WaterStatus.CONTAMINATED]
        if contaminated_hh_ids.numel() == 0:
            return

        status_key = AgentPropertyKeys.status(self.name)
        status = agent_state.ndata[status_key]
        is_in_contaminated_hh = torch.isin(
            agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID], contaminated_hh_ids
        )

        # Allow re-infection of recovered agents, consistent with other water routes:
        susceptible_like = (status == Compartment.SUSCEPTIBLE) | (status == Compartment.RECOVERED)
        vaccinated = (status == Compartment.VACCINATED)

        susceptible_mask = is_in_contaminated_hh & susceptible_like
        vaccinated_mask = is_in_contaminated_hh & vaccinated

        if not torch.any(susceptible_mask) and not torch.any(vaccinated_mask):
            return

        base_prob = self.global_params.water_to_human_infection_prob
        pressure = torch.ones(agent_state.num_nodes(), device=self.device)

        # Unvaccinated (S + R): full base probability
        if torch.any(susceptible_mask):
            self._apply_new_infections(
                agent_state,
                target_nodes_mask=susceptible_mask,
                base_prob=base_prob,
                forced_pressure=pressure,
                route="water",
            )

        # Vaccinated: reduced probability via breakthrough multiplier
        if torch.any(vaccinated_mask):
            breakthrough_multiplier = 1.0 - self.config.vaccine_efficacy
            self._apply_new_infections(
                agent_state,
                target_nodes_mask=vaccinated_mask,
                base_prob=base_prob,
                forced_pressure=pressure,
                prob_multiplier=breakthrough_multiplier,
                route="water",
            )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_infection_probability(self):
        self.current_infection_prob = max(
            0.001,
            torch.normal(mean=self.config.infection_prob_mean, std=self.config.infection_prob_std, size=(1,)).item()
        )

    def _susceptible_to_vaccinated(self, agent_state: AgentState):
        # Rotavirus vaccines are administered in infancy - restrict to
        # children, matching real-world vaccination programs rather than a
        # population-wide campaign.
        status_key = AgentPropertyKeys.status(self.name)
        sus_mask = (
            (agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE)
            & agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        )
        if not torch.any(sus_mask):
            return

        chance = torch.rand(torch.sum(sus_mask), device=self.device)
        vaccinated_mask = chance < self.config.vaccination_rate
        agents_to_vaccinate = sus_mask.nonzero(as_tuple=True)[0][vaccinated_mask]
        agent_state.ndata[status_key][agents_to_vaccinate] = Compartment.VACCINATED
        # Keep a record that outlives the V compartment: a breakthrough
        # infection moves the child V -> E -> I, and illness severity (which
        # is set once the child is INFECTIOUS) needs to know they were
        # vaccinated.
        vacc_key = AgentPropertyKeys.ever_vaccinated(self.name)
        if vacc_key in agent_state.ndata:
            agent_state.ndata[vacc_key][agents_to_vaccinate] = True

    def _susceptible_to_exposed_h2h(self, agent_state: AgentState, location_ids: torch.Tensor, num_locations: int):
        status_key = AgentPropertyKeys.status(self.name)
        sus_mask = (
            (agent_state.ndata[status_key] == Compartment.SUSCEPTIBLE) |
            (agent_state.ndata[status_key] == Compartment.RECOVERED)
        )
        self._apply_new_infections(
            agent_state,
            target_nodes_mask=sus_mask,
            base_prob=self.current_infection_prob,
            location_ids=location_ids,
            num_locations=num_locations,
            prob_multiplier=1.0,
            route="h2h",
        )

    def _vaccinated_to_exposed_h2h(self, agent_state: AgentState, location_ids: torch.Tensor, num_locations: int):
        status_key = AgentPropertyKeys.status(self.name)
        vac_mask = agent_state.ndata[status_key] == Compartment.VACCINATED
        breakthrough_multiplier = 1.0 - self.config.vaccine_efficacy
        self._apply_new_infections(
            agent_state,
            target_nodes_mask=vac_mask,
            base_prob=self.current_infection_prob,
            location_ids=location_ids,
            num_locations=num_locations,
            prob_multiplier=breakthrough_multiplier,
            route="h2h",
        )

    def _human_to_water(self, agent_state: AgentState, grid: Any):
        if grid is None:
            return

        state = grid.get_water_state(self.name)
        if state is None:
            return

        status_key = AgentPropertyKeys.status(self.name)
        infectious_mask = agent_state.ndata[status_key] == Compartment.INFECTIOUS

        current_x = agent_state.ndata[AgentPropertyKeys.X].long()
        current_y = agent_state.ndata[AgentPropertyKeys.Y].long()

        # Get water status at current agent locations
        # If agent is at home, the water state is 0 (no source there).
        is_at_water_loc = state[current_y, current_x] > 0

        contaminator_mask = infectious_mask & is_at_water_loc

        if not torch.any(contaminator_mask):
            return

        prob = self.global_params.human_to_water_infection_prob
        chance = torch.rand(torch.sum(contaminator_mask), device=self.device)
        success = chance < prob
        successful_contaminators = contaminator_mask.nonzero(as_tuple=True)[0][success]

        if len(successful_contaminators) > 0:
            # We contaminate the water source at their current location
            locs = torch.stack((current_y[successful_contaminators], current_x[successful_contaminators]), dim=1)
            unique_locs = torch.unique(locs, dim=0)
            if unique_locs.shape[0] > 0:
                state[unique_locs[:, 0], unique_locs[:, 1]] = WaterStatus.CONTAMINATED

    def _water_to_human(self, agent_state: AgentState, grid: Any):
        """Susceptible agents get infected from contaminated water sources."""
        if grid is None:
            return

        water_slice = grid.get_water_state(self.name)
        if water_slice is None:
            return
        if torch.all(water_slice != WaterStatus.CONTAMINATED):
            return

        current_x = agent_state.ndata[AgentPropertyKeys.X].long()
        current_y = agent_state.ndata[AgentPropertyKeys.Y].long()

        agent_water_status = water_slice[current_y, current_x]
        at_contaminated_water_mask = (agent_water_status == WaterStatus.CONTAMINATED)

        if not torch.any(at_contaminated_water_mask):
            return

        status_key = AgentPropertyKeys.status(self.name)
        status = agent_state.ndata[status_key]

        # recovered agents can be re-infected.
        susceptible_like = (status == Compartment.SUSCEPTIBLE) | (status == Compartment.RECOVERED)
        susceptible_mask = susceptible_like & at_contaminated_water_mask
        vaccinated_mask = (status == Compartment.VACCINATED) & at_contaminated_water_mask

        if not torch.any(susceptible_mask) and not torch.any(vaccinated_mask):
            return

        # Everyone experiences one "contact" with contaminated water
        pressure = torch.ones(agent_state.num_nodes(), device=self.device)
        base_prob = self.global_params.water_to_human_infection_prob

        # Unvaccinated (S + R): full base probability
        if torch.any(susceptible_mask):
            self._apply_new_infections(
                agent_state,
                target_nodes_mask=susceptible_mask,
                base_prob=base_prob,
                forced_pressure=pressure,
                route="water",
            )

        # Vaccinated: reduced probability via (1 - vaccine_efficacy)
        if torch.any(vaccinated_mask):
            breakthrough_multiplier = 1.0 - self.config.vaccine_efficacy
            self._apply_new_infections(
                agent_state,
                target_nodes_mask=vaccinated_mask,
                base_prob=base_prob,
                forced_pressure=pressure,
                prob_multiplier=breakthrough_multiplier,
                route="water",
            )

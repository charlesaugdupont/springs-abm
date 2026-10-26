# abm/model/initialize_model.py
"""This module contains the primary SVEIRModel class."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import qmc

from config import SVEIRConfig
from abm.model.step import sveir_step
from abm.constants import AgentPropertyKeys, Compartment
from abm.factories.agent_factory import AgentFactory
from abm.factories.environment_factory import EnvironmentFactory
from abm.pathogens.registry import build_pathogen, is_implemented
from abm.state import AgentState


class Model:
    """Abstract base class for a model."""
    def __init__(self, model_identifier: str, root_path: str = '.'):
        self._model_identifier = model_identifier
        self.root_path = Path(root_path)
        self.model_dir = self.root_path / self._model_identifier
        self.step_count = 0
        self.config: SVEIRConfig | None = None

    def save_model_parameters(self):
        """Saves the model's configuration to a YAML file."""
        if self.config is None:
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cfg_filename = self.model_dir / f"{self._model_identifier}.yaml"
        self.config.to_yaml(cfg_filename)


class SVEIRModel(Model):
    """The main class for the SVEIR agent-based model."""
    def __init__(self, *, model_identifier: str, root_path: str = '.'):
        super().__init__(model_identifier=model_identifier, root_path=root_path)
        self.steering_parameters = None
        self.graph = None
        self.infection_incidence = []
        self.u5_prevalence_history: dict[str, list[float]] = {}
        # Fraction of under-5s with an ACTIVE SYMPTOMATIC EPISODE, as distinct
        # from u5_prevalence_history, which is the fraction INFECTIOUS (i.e.
        # shedding). For rotavirus and campylobacter the two track each other,
        # since every infection in a child produces an episode. For
        # Cryptosporidium they differ by design and by a large factor: ~2/3 of
        # childhood infections are subclinical and shedding runs ~3 weeks, so
        # carriage prevalence is several times symptomatic prevalence. Keeping
        # both means a "prevalence" target can never silently compare a
        # carriage figure to a diarrhoea figure.
        self.u5_symptomatic_prevalence_history: dict[str, list[float]] = {}
        # Age-structured daily incidence records - see
        # _record_age_structured_incidence(). Populated on EVERY step (whether
        # stepped via run() or manually), so callers that drive step()
        # themselves (webapp, orchestrator) need no extra setup.
        self.child_days_by_age_month: list = []
        self.new_infections_by_age_month: dict[str, list] = {}
        # Daily adult and young-child carriage per pathogen - see
        # _record_carriage_by_group(); (re)initialised with the age records.
        self.carriage_by_group: dict[str, list] = {}
        self._prev_num_infections: dict[str, torch.Tensor] = {}
        self._age_records_ready = False
        self.pathogens = []
        self.systems = []
        self.grid_environment = None
        self.agent_personas = None

    def set_model_parameters(self, **kwargs):
        """Sets model parameters from a dictionary, overriding defaults."""
        self.config = SVEIRConfig.from_dict(kwargs)
        self.steering_parameters = self.config.steering_parameters
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_parameters()

    def _load_agent_personas(self):
        """Generates agent behavioral personas using LH sampling."""
        sampler = qmc.LatinHypercube(d=3, seed=self.config.seed)
        samples = sampler.random(n=self.config.num_agent_personas)
        ranges = [self.config.alpha_range, self.config.gamma_range, self.config.lambda_range]
        scaled_samples = qmc.scale(samples, [r[0] for r in ranges], [r[1] for r in ranges])
        self.agent_personas = torch.from_numpy(scaled_samples).float()

    def _build_animal_density_layers(self):
        """
        Builds and attaches the household-ownership-derived poultry/ruminant
        density layers used by the zoonotic routes.

        Both Campylobacter (poultry + ruminants) and Cryptosporidium
        (ruminants only) read these layers, so they are built whenever either
        pathogen is configured - and from ONE ownership draw, so the two
        pathogens always see the same livestock landscape.

        Unlike the water/school/worship layers, these are NOT part of the
        static grid.npz file: they depend on this run's random household
        placement and ownership draw, so they're (cheaply) recomputed once
        per initialize_model() call rather than cached per grid_id.
        """
        if self.grid_environment is None:
            return

        from abm.environment.animal_density import (
            build_animal_density_layers,
            resolve_ownership_params,
        )

        ownership = resolve_ownership_params(self.config.pathogens, getattr(self, "cell_size_m", None))
        if ownership is None:
            return

        household_ids = self.graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        unique_households, inverse = torch.unique(household_ids, return_inverse=True)
        counts = torch.zeros(len(unique_households), device=household_ids.device)
        counts.index_add_(0, inverse, torch.ones_like(household_ids, dtype=torch.float))

        def household_mean(agent_values: torch.Tensor) -> torch.Tensor:
            # All members of a household share identical Y/X/ownership values,
            # so a group mean is just a cheap, fully-vectorised way to grab
            # "one row per household" without a loop.
            summed = torch.zeros(len(unique_households), device=agent_values.device, dtype=torch.float)
            summed.index_add_(0, inverse, agent_values.float())
            return summed / counts

        hh_y = household_mean(self.graph.ndata[AgentPropertyKeys.Y])
        hh_x = household_mean(self.graph.ndata[AgentPropertyKeys.X])
        hh_owns_poultry = household_mean(self.graph.ndata[AgentPropertyKeys.OWNS_POULTRY])
        hh_owns_ruminant = household_mean(self.graph.ndata[AgentPropertyKeys.OWNS_RUMINANT])

        grid_shape = self.grid_environment.grid_shape[:2]
        poultry_density, ruminant_density = build_animal_density_layers(
            household_y=hh_y,
            household_x=hh_x,
            household_owns_poultry=hh_owns_poultry,
            household_owns_ruminant=hh_owns_ruminant,
            grid_shape=grid_shape,
            poultry_sigma=ownership["poultry_roam_sigma"],
            ruminant_sigma=ownership["ruminant_roam_sigma"],
            device=self.config.device,
        )
        self.grid_environment.set_dynamic_layer("poultry_density", poultry_density)
        self.grid_environment.set_dynamic_layer("ruminant_density", ruminant_density)

    def initialize_model(self, verbose: bool = False):
        """Initializes the entire model state, including agents and environment."""
        if self.config is None:
            raise RuntimeError(
                "Model parameters have not been set. Call set_model_parameters() first."
            )

        torch.manual_seed(self.config.seed)
        self._load_agent_personas()

        # 1. Create Agents
        household_ids, is_child = self._calculate_demographics(self.config.number_agents)

        self.graph = AgentState(self.config.number_agents, device=self.config.device)
        self.graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID] = household_ids
        self.graph.ndata[AgentPropertyKeys.IS_CHILD] = is_child

        # 2. Create Environment and Place Agents
        if self.config.spatial:
            env_factory = EnvironmentFactory(self.config.spatial_creation_args, device=self.config.device)
            env_factory.create_grid()
            env_factory.place_agents(self.graph)
            self.grid_environment = env_factory.grid_environment
            self._resolve_metric_parameters()
        else:
            self.grid_environment = None
            self.graph.ndata[AgentPropertyKeys.X] = torch.zeros(
                self.config.number_agents, device=self.config.device
            )
            self.graph.ndata[AgentPropertyKeys.Y] = torch.zeros(
                self.config.number_agents, device=self.config.device
            )

        # 3. Initialize Agent Properties
        agent_factory = AgentFactory(self.config, self.agent_personas)
        agent_factory.initialize_agent_properties(self.graph, self.grid_environment)

        # 4. Finalize Graph (move to compute device)
        self.graph.to(self.config.device)

        # 5. Build household-ownership-derived animal density layers.
        self._build_animal_density_layers()

        # 6. Initialize pathogen and system objects
        self._initialize_pathogens_and_systems()

        # 7. Water. Catchment model (default): build the source network
        # (sources, household assignments, runoff kernels), send water
        # collectors to their household's own source, and give each
        # waterborne pathogen a concentration vector. Legacy model: seed each
        # waterborne pathogen's own point-source state (see
        # GridEnvironment.set_water_state for why that is per-pathogen).
        if self.grid_environment is not None:
            if self.config.water.model == "catchment":
                from abm.environment.water_network import WaterNetwork

                network = WaterNetwork.build(self.grid_environment, self.graph, self.config.water,
                                             device=self.config.device)
                self.grid_environment.water_network = network
                self.graph.ndata[AgentPropertyKeys.WATER_LOCATION] = (
                    network.agent_water_location().to(self.config.device)
                )
                for pathogen in self.pathogens:
                    if getattr(pathogen, "WATERBORNE", False):
                        pathogen.use_catchment_water = True
                        network.init_pathogen(pathogen.name)
            else:
                for pathogen in self.pathogens:
                    if getattr(pathogen, "WATERBORNE", False):
                        pathogen.init_water_reservoir(self.grid_environment)

        # 8. Age-structured incidence bookkeeping (no RNG; pure observation).
        self._init_age_structured_records()

        if verbose:
            print(f'{self.graph.num_nodes()} agents initialized on {self.config.device}')

    def _resolve_metric_parameters(self):
        """Converts every distance given in metres (the "_m" twins of
        cell-based parameters) into cells of the loaded grid, writing the
        cell value into this model's own config. Distances left as None keep
        their cell values, so the original grid is untouched."""
        import math
        bounds = getattr(self.grid_environment, "bounds", None)
        if bounds is None:
            return
        rows = self.grid_environment.grid_tensor.shape[0]
        minx, miny, x_step, y_step = (float(v) for v in bounds)
        lat = miny + rows * y_step / 2
        dx = x_step * 111_320.0 * math.cos(math.radians(lat))
        dy = y_step * 110_574.0
        cell_m = math.sqrt(dx * dy)
        self.cell_size_m = cell_m
        sp = self.config.steering_parameters
        if sp.social_interaction_radius_m is not None:
            sp.social_interaction_radius = sp.social_interaction_radius_m / cell_m
        for p in self.config.pathogens:
            for f in ("poultry_roam_sigma", "ruminant_roam_sigma"):
                metres = getattr(p, f + "_m", None)
                if metres is not None:
                    setattr(p, f, metres / cell_m)
        w = self.config.water
        if w.reach_block_m is not None:
            w.reach_block_cells = max(1, int(round(w.reach_block_m / cell_m)))
        if w.river_min_width_m is not None:
            w.river_min_width_cells = max(2, int(round(w.river_min_width_m / cell_m)))

    def _initialize_pathogens_and_systems(self):
        """Instantiates the pathogen and system objects that drive the simulation."""
        from abm.systems.movement import MovementSystem
        from abm.systems.environment import EnvironmentSystem
        from abm.systems.child_illness import ChildIllnessSystem
        from abm.systems.care_seeking import CareSeekingSystem
        from abm.systems.economics import EconomicSystem
        from abm.systems.household import HouseholdSystem
        from abm.systems.demography import DemographySystem

        for p_config in self.config.pathogens:
            if is_implemented(p_config.name):
                self.pathogens.append(
                    build_pathogen(p_config, self.steering_parameters, self.config.device)
                )
            else:
                print(f"Warning: Pathogen '{p_config.name}' not implemented.")

        self.systems = [
            MovementSystem(self.config),
            ChildIllnessSystem(self.config),   # systems[1] — has episode_log
            CareSeekingSystem(self.config),
            HouseholdSystem(self.config),
            EnvironmentSystem(self.config),
            EconomicSystem(self.config),
            # systems[6]. Not called from sveir_step: it runs from
            # SVEIRModel.step() AFTER the day's age-structured incidence has
            # been recorded, so infections are attributed to the age the
            # child had when infected, before any slot is renewed.
            DemographySystem(self.config),
        ]

    def _calculate_demographics(self, num_agents: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates household IDs and child status for all agents robustly."""
        avg_size = float(self.config.average_household_size)

        sizes_list = []
        total_generated = 0
        while total_generated < num_agents:
            remaining = num_agents - total_generated
            batch_count = max(5, int((remaining / avg_size) * 1.5))
            batch_sizes = torch.poisson(torch.full((batch_count,), avg_size - 1.0)) + 1.0
            sizes_list.append(batch_sizes)
            total_generated += batch_sizes.sum().item()

        sizes = torch.cat(sizes_list)
        full_ids = torch.repeat_interleave(torch.arange(len(sizes)), sizes.long())
        household_ids = full_ids[:num_agents]

        is_child = torch.zeros(num_agents, dtype=torch.bool)
        for hid in torch.unique(household_ids):
            members = (household_ids == hid).nonzero(as_tuple=True)[0]
            if len(members) > 1:
                others = members[1:]
                is_child[others] = (torch.rand(len(others)) < self.config.child_probability)
        return household_ids, is_child

    # ------------------------------------------------------------------
    # Under-5 prevalence tracking helpers
    # ------------------------------------------------------------------
    def _record_u5_prevalence(self, u5_mask: torch.Tensor):
        """
        Appends the current fraction of under-5s who are infectious for each
        pathogen to u5_prevalence_history.  Called at the end of every step.
        """
        n_u5 = u5_mask.sum().item()
        if n_u5 == 0:
            for p in self.pathogens:
                self.u5_prevalence_history[p.name].append(0.0)
                self.u5_symptomatic_prevalence_history[p.name].append(0.0)
            return

        for p in self.pathogens:
            status_key  = AgentPropertyKeys.status(p.name)
            status_u5   = self.graph.ndata[status_key][u5_mask]
            n_infectious = (status_u5 == Compartment.INFECTIOUS).sum().item()
            self.u5_prevalence_history[p.name].append(n_infectious / n_u5)

            severity_u5 = self.graph.ndata[AgentPropertyKeys.symptom_severity(p.name)][u5_mask]
            n_symptomatic = (severity_u5 > 0).sum().item()
            self.u5_symptomatic_prevalence_history[p.name].append(n_symptomatic / n_u5)

    # ------------------------------------------------------------------
    # Age-structured incidence
    # ------------------------------------------------------------------
    #: Children are binned by completed month of age, 0..AGE_RECORD_MONTHS-1.
    #: 60 covers the under-5 population; any older child (none exist today)
    #: would be clamped into the last bin.
    AGE_RECORD_MONTHS = 60

    def _init_age_structured_records(self):
        """(Re)initialises the per-day, per-month-of-age incidence records."""
        self.child_days_by_age_month = []
        self.new_infections_by_age_month = {p.name: [] for p in self.pathogens}
        # The same two records restricted to children with NO prior infection
        # by that pathogen at the start of the day (the "naive" stratum of
        # MAL-ED Supplementary Table 3); the ">=1 prior" stratum is the
        # difference from the totals above.
        self.child_days_naive_by_age_month = {p.name: [] for p in self.pathogens}
        self.new_infections_naive_by_age_month = {p.name: [] for p in self.pathogens}
        # Daily carriage by group - see _record_carriage_by_group().
        self.carriage_by_group = {p.name: [] for p in self.pathogens}
        self._prev_num_infections = {
            p.name: self.graph.ndata[AgentPropertyKeys.num_infections(p.name)].clone()
            for p in self.pathogens
        }
        self._age_records_ready = True

    def _record_age_structured_incidence(self):
        """Appends today's child-days and new child infections by month of age.

        Records, for each simulated day:
          * child_days_by_age_month[d][m]: number of children whose age is in
            completed month m at the end of day d (the person-time denominator);
          * new_infections_by_age_month[p][d][m]: number of NEW infections
            (S/R -> E transitions) with pathogen p among those children on day d.

        Rates over any age window and any day window (e.g. excluding a burn-in
        period) are then exact ratios of sums - see
        experiments.metrics.age_window_rates. Unlike num_infections, which is a
        lifetime total per agent, this lets metrics ignore the seeded initial
        epidemic wave and select the 0-24 month window that the MAL-ED
        targets refer to.

        Also stamps first_infection_day for children born in the simulation
        whose first infection with a pathogen happened today.

        Pure observation: reads state, draws no random numbers.
        """
        if not self._age_records_ready:
            self._init_age_structured_records()
        is_child = self.graph.ndata[AgentPropertyKeys.IS_CHILD].bool()
        n_bins = self.AGE_RECORD_MONTHS
        age_m = self.graph.ndata[AgentPropertyKeys.AGE][is_child]
        month = torch.clamp(age_m.floor().long(), 0, n_bins - 1)
        self.child_days_by_age_month.append(
            torch.bincount(month, minlength=n_bins).cpu().numpy()
        )
        birth_key = AgentPropertyKeys.BIRTH_DAY
        born_in_sim = (
            (self.graph.ndata[birth_key] >= 0) & is_child
            if birth_key in self.graph.ndata else None
        )
        for p in self.pathogens:
            current = self.graph.ndata[AgentPropertyKeys.num_infections(p.name)]
            prev = self._prev_num_infections[p.name]
            if born_in_sim is not None:
                first_key = AgentPropertyKeys.first_infection_day(p.name)
                first = born_in_sim & (prev == 0) & (current > 0)
                if torch.any(first):
                    self.graph.ndata[first_key][first] = self.step_count
            new = (current - prev)[is_child]
            naive = (prev[is_child] == 0)
            self._prev_num_infections[p.name] = current.clone()
            self.new_infections_by_age_month[p.name].append(
                torch.bincount(month, weights=new.double(), minlength=n_bins)
                .round().long().cpu().numpy()
            )
            self.child_days_naive_by_age_month[p.name].append(
                torch.bincount(month[naive], minlength=n_bins).cpu().numpy()
            )
            self.new_infections_naive_by_age_month[p.name].append(
                torch.bincount(month[naive], weights=new[naive].double(), minlength=n_bins)
                .round().long().cpu().numpy()
            )

    #: Columns of each daily carriage_by_group row.
    CARRIAGE_COLUMNS = ("n_adult", "n_adult_infectious", "n_child_12_59m_well",
                        "n_child_12_59m_well_infectious", "weight_adult", "weight_total")

    def _record_carriage_by_group(self):
        """Appends, per pathogen, today's carriage in the two groups that
        carriage surveys compare - adults (every agent aged 5+) and children
        aged 12-59 months not in an illness episode (a community stool survey
        samples people without diarrhoea) - and the infectiousness weight
        (Pathogen.infectiousness_weights) the adults carry. One row of
        CARRIAGE_COLUMNS per day, aligned with child_days_by_age_month.
        Carriage is the INFECTIOUS compartment (true infection, not detection).

        Pure observation: reads state, draws no random numbers.
        """
        nd = self.graph.ndata
        is_child = nd[AgentPropertyKeys.IS_CHILD].bool()
        adult = ~is_child
        older_child = is_child & (nd[AgentPropertyKeys.AGE] >= 12.0)
        n_adult = int(adult.sum())
        for p in self.pathogens:
            infectious = nd[AgentPropertyKeys.status(p.name)] == Compartment.INFECTIOUS
            well = older_child & (nd[AgentPropertyKeys.symptom_severity(p.name)] <= 0)
            w = p.infectiousness_weights(self.graph)
            self.carriage_by_group[p.name].append(np.array([
                n_adult, int((adult & infectious).sum()), int(well.sum()), int((well & infectious).sum()),
                float(w[adult].sum()), float(w.sum()),
            ], dtype=np.float64))

    def _advance_demography(self):
        """Ages everyone by a day and renews children leaving the under-5
        population (abm/systems/demography.py)."""
        from abm.systems.demography import DemographySystem
        from abm.systems.child_illness import ChildIllnessSystem

        demo = next((s for s in self.systems if isinstance(s, DemographySystem)), None)
        if demo is None:
            return
        illness = next((s for s in self.systems if isinstance(s, ChildIllnessSystem)), None)
        reborn = demo.update(self.graph, timestep=self.step_count, illness_system=illness)
        if reborn.numel() > 0:
            # The renewed slots' infection counts were reset to 0; keep the
            # incidence snapshot in step so tomorrow's diff is not negative.
            for name, prev in self._prev_num_infections.items():
                prev[reborn] = 0

    def get_infection_log(self, pathogen_name: str) -> dict:
        """ChildIllnessSystem.get_infection_log for this model (empty arrays
        if there is no illness system)."""
        from abm.systems.child_illness import ChildIllnessSystem
        for system in self.systems:
            if isinstance(system, ChildIllnessSystem):
                return system.get_infection_log(pathogen_name)
        import numpy as np
        empty = np.zeros(0, dtype=np.int64)
        return {"agent_idx": empty, "birth_day": empty, "day": empty, "symptomatic": empty.astype(bool)}

    def get_birth_cohorts(self) -> list[dict]:
        """Every child born in the simulation, with first-infection days.

        Children who have already left (aged out) come from
        DemographySystem.cohort_log; children still present are read from
        agent state, with exit_day None. first_infection_day is -1 for a
        pathogen the child has not (yet) had.
        """
        from abm.systems.demography import DemographySystem

        demo = next((s for s in self.systems if isinstance(s, DemographySystem)), None)
        cohorts = list(demo.cohort_log) if demo is not None else []
        if AgentPropertyKeys.BIRTH_DAY not in self.graph.ndata:
            return cohorts
        birth = self.graph.ndata[AgentPropertyKeys.BIRTH_DAY]
        present = ((birth >= 0) & self.graph.ndata[AgentPropertyKeys.IS_CHILD].bool()).nonzero(as_tuple=True)[0]
        if present.numel() == 0:
            return cohorts
        firsts = {
            p.name: self.graph.ndata[AgentPropertyKeys.first_infection_day(p.name)][present].tolist()
            for p in self.pathogens
        }
        slots = present.tolist()
        for j, b in enumerate(birth[present].tolist()):
            cohorts.append({
                "agent_idx": int(slots[j]), "birth_day": int(b), "exit_day": None,
                "first_infection_day": {name: int(v[j]) for name, v in firsts.items()},
            })
        return cohorts

    # ------------------------------------------------------------------
    # Run / step
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False):
        """Runs the simulation from the current step_count to the step_target."""
        self.infection_incidence.clear()
        # Initialise per-pathogen under-5 prevalence lists
        self.u5_prevalence_history = {p.name: [] for p in self.pathogens}
        self.u5_symptomatic_prevalence_history = {p.name: [] for p in self.pathogens}
        # Pre-compute the under-5 mask once - IS_CHILD never changes mid-run
        # (turnover renews child slots in place; see DemographySystem)
        child_mask = self.graph.ndata[AgentPropertyKeys.IS_CHILD]

        while self.step_count < self.config.step_target:
            self.step(verbose, child_mask=child_mask)

    def step(self, verbose: bool = False, child_mask: torch.Tensor | None = None):
        """Executes one full timestep of the simulation."""
        if verbose:
            print(f'Performing step {self.step_count} of {self.config.step_target}')
        try:
            new_cases_by_pathogen, compartment_counts = sveir_step(
                self.graph, self.step_count, self.config,
                self.grid_environment, self.pathogens, self.systems
            )
            total_new_cases = sum(new_cases_by_pathogen.values())
            self.infection_incidence.append(total_new_cases)

            # Record under-5 prevalence if a mask was supplied (i.e. called from run())
            if child_mask is not None:
                self._record_u5_prevalence(child_mask)
            # Always recorded, regardless of how step() is driven.
            self._record_age_structured_incidence()
            self._record_carriage_by_group()
            self._advance_demography()

        except Exception as e:
            raise RuntimeError(f'Execution of step {self.step_count} failed.') from e
        self.step_count += 1

    # ------------------------------------------------------------------
    # Result-gathering methods
    # ------------------------------------------------------------------

    def get_total_infections(self) -> int:
        return sum(
            torch.sum(self.graph.ndata[AgentPropertyKeys.num_infections(p.name)]).item()
            for p in self.pathogens
        )

    def get_proportion_infected_at_least_once(self) -> float:
        if not self.pathogens:
            return 0.0
        infected_masks = [
            (self.graph.ndata[AgentPropertyKeys.num_infections(p.name)] > 0)
            for p in self.pathogens
        ]
        combined_mask = torch.stack(infected_masks).any(dim=0)
        return torch.sum(combined_mask).item() / self.graph.num_nodes()

    def get_time_series_data(self) -> dict:
        return {
            "incidence": self.infection_incidence,
            "u5_prevalence": self.u5_prevalence_history,
            "u5_symptomatic_prevalence": self.u5_symptomatic_prevalence_history,
        }

    def get_child_episode_log(self) -> list[dict]:
        """
        Returns the full per-episode illness log collected by ChildIllnessSystem.

        Each entry is a dict with keys:
            agent_idx, pathogen, initial_severity, initial_duration, timestep
        """
        from abm.systems.child_illness import ChildIllnessSystem
        for system in self.systems:
            if isinstance(system, ChildIllnessSystem):
                return list(system.episode_log)
        return []

    def get_final_agent_states(self) -> dict:
        return {
            'health': self.graph.ndata[AgentPropertyKeys.HEALTH].cpu().numpy(),
            'wealth': self.graph.ndata[AgentPropertyKeys.WEALTH].cpu().numpy(),
            'household_id': self.graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID].cpu().numpy(),
            'initial_health': self.graph.ndata[AgentPropertyKeys.INITIAL_HEALTH].cpu().numpy(),
            'initial_wealth': self.graph.ndata[AgentPropertyKeys.INITIAL_WEALTH].cpu().numpy(),
            'care_seeking_count': self.graph.ndata[AgentPropertyKeys.CARE_SEEKING_COUNT].cpu().numpy(),
            'is_parent': self.graph.ndata[AgentPropertyKeys.IS_PARENT].cpu().numpy(),
            'is_child': self.graph.ndata[AgentPropertyKeys.IS_CHILD].cpu().numpy(),
            'age': self.graph.ndata[AgentPropertyKeys.AGE].cpu().numpy(),
            'alpha': self.graph.ndata[AgentPropertyKeys.ALPHA].cpu().numpy(),
            'gamma': self.graph.ndata[AgentPropertyKeys.GAMMA].cpu().numpy(),
            'lambda': self.graph.ndata[AgentPropertyKeys.LAMBDA].cpu().numpy(),
        }

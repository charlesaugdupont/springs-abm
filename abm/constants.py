# abm/constants.py
from enum import IntEnum

class Compartment(IntEnum):
    SUSCEPTIBLE = 0
    VACCINATED = 1
    EXPOSED = 2
    INFECTIOUS = 3
    RECOVERED = 4

class Activity(IntEnum):
    HOME = 0
    SCHOOL = 1
    WORSHIP = 2
    WATER = 3
    SOCIAL = 4

class Phase(IntEnum):
    """Which of the two within-day transmission phases is currently executing.

    abm/model/step.py runs transmission twice per simulated day: once after
    agents have moved to their chosen daytime activity location (DAY) and once
    after MovementSystem.reset_to_home has returned everyone to their household
    (NIGHT). The phase is passed explicitly to Pathogen.step_transmission so a
    pathogen with a once-per-day, household-scoped route can identify the right
    call without inferring it from agent positions.

    This replaces Campylobacter._is_night_phase, which inferred the phase by
    checking whether *every* agent happened to be standing on their home cell.
    That test is only probabilistically false during the day phase (it would
    return True on any day where every agent independently drew the HOME
    activity), and it had to be duplicated by every pathogen that needed the
    distinction.
    """
    DAY = 0
    NIGHT = 1

class WaterStatus(IntEnum):
    """
    Represents the contamination status of a water source cell.

    Grid cells use the following encoding:
      0 = no water source present (empty cell)
      1 = water source present, clean      (WaterStatus.CLEAN)
      2 = water source present, contaminated (WaterStatus.CONTAMINATED)

    The grid generator marks water-source cells with 1 on creation, which
    intentionally matches WaterStatus.CLEAN so that the two systems share
    a single encoding rather than relying on a coincidental value match.
    """
    CLEAN = 1
    CONTAMINATED = 2

class GridLayer:
    """Centralized string keys for layers in the spatial grid."""
    RESIDENCES = "residences"
    SCHOOL = "school"
    WORSHIP = "place_of_worship"
    WATER = "water"
    NATURAL_WATER = "natural_water"

class AgentPropertyKeys:
    """Centralized string keys for agent properties in AgentState.ndata."""
    # Core State
    HEALTH = "health"
    WEALTH = "wealth"
    INITIAL_HEALTH = "initial_health"
    INITIAL_WEALTH = "initial_wealth"

    # Demographics
    HOUSEHOLD_ID = "household_id"
    IS_CHILD = "is_child"
    IS_PARENT = "is_parent"
    AGE = "age"   # months
    # Day (timestep) on which the agent slot's current occupant was born
    # inside the simulation; -1 for agents present at t=0. See
    # abm/systems/demography.py: a child reaching the upper age limit leaves
    # the modelled population and the slot is reused for a newborn.
    BIRTH_DAY = "birth_day"
    # Whether this child slot attends school/early-childhood education once
    # old enough (drawn once per slot; see DemographyConfig).
    ATTENDS_SCHOOL = "attends_school"
    # Household-level multiplier (mean 1) on infection risk via environmental
    # routes, shared by every member and pathogen - see
    # SteeringParamsSVEIR.household_exposure_sdlog.
    EXPOSURE_FRAILTY = "exposure_frailty"

    # Behavior / CPT
    PERSONA_ID = "persona_id"
    ALPHA = "alpha"
    GAMMA = "gamma"
    LAMBDA = "lambda"

    # Location & Activity
    X = "x"
    Y = "y"
    TIME_USE = "time_use"
    # Time-use weights as originally drawn, before age-based eligibility
    # (e.g. no school below the entry age) is applied to give TIME_USE.
    TIME_USE_BASE = "time_use_base"
    ACTIVITY_CHOICE = "activity_choice"
    HOME_LOCATION = "home_location"
    SCHOOL_LOCATION = "school_location"
    WORSHIP_LOCATION = "worship_location"
    WATER_LOCATION = "water_location"
    
    # Illness State
    CARE_SEEKING_COUNT = "care_seeking_count"

    # Household Animal Ownership (Campylobacter zoonotic route)
    OWNS_POULTRY = "owns_poultry"
    OWNS_RUMINANT = "owns_ruminant"

    # Household economy in cedis (EconomyConfig.model == "cedi" only; absent
    # otherwise). Household-level values are broadcast to every member, like
    # WEALTH. Money is 2026 GH¢ per day.
    ECON_AGE_GROUP = "econ_age_group"            # 0 under 5, 1 aged 5-14, 2 aged 15-64, 3 aged 65+
    WEALTH_QUINTILE = "wealth_quintile"          # 0 (lowest) .. 4 (highest), per household
    EARNING_POTENTIAL = "earning_potential"      # GH¢/day at full health; 0 for non-earners
    CONSUMPTION_TARGET = "consumption_target"    # household's GH¢/day consumption target
    CASH = "cash"                                # household's cash, GH¢
    INSURED = "insured"                          # household holds NHIS cover
    PROVIDER = "provider"                        # usual provider: 0 public primary, 1 public hospital, 2 private, 3 drug seller
    WORK_LOST = "work_lost"                      # share of today's working time lost (reset daily)

    # Dynamic keys for pathogens
    @staticmethod
    def status(pathogen_name: str) -> str:
        return f"status_{pathogen_name}"

    @staticmethod
    def exposure_time(pathogen_name: str) -> str:
        return f"exposure_time_{pathogen_name}"

    @staticmethod
    def num_infections(pathogen_name: str) -> str:
        return f"num_infections_{pathogen_name}"
    
    @staticmethod
    def symptom_severity(pathogen_name: str) -> str:
        return f"symptom_severity_{pathogen_name}"
    
    @staticmethod
    def illness_duration(pathogen_name: str) -> str:
        return f"illness_duration_{pathogen_name}"

    @staticmethod
    def ever_vaccinated(pathogen_name: str) -> str:
        """Persistent vaccination history. Unlike the V compartment, this
        survives V -> E -> I -> R, so it is still readable when a vaccinated
        child falls ill (see ChildIllnessSystem._initialize_illness)."""
        return f"ever_vaccinated_{pathogen_name}"

    @staticmethod
    def first_infection_day(pathogen_name: str) -> str:
        """Timestep of the slot occupant's first infection with this pathogen
        (-1 = none yet). Only meaningful for children born in the simulation
        (BIRTH_DAY >= 0); used for age-at-first-infection metrics."""
        return f"first_infection_day_{pathogen_name}"

    @staticmethod
    def symptomatic_infection(pathogen_name: str) -> str:
        """True while the agent's CURRENT infection with this pathogen is one
        that produced an illness episode (set when the episode opens, cleared
        at the next exposure). Weights the agent's infectiousness by
        PathogenConfig.asymptomatic_relative_infectiousness."""
        return f"symptomatic_infection_{pathogen_name}"

    @staticmethod
    def last_episode_infection(pathogen_name: str) -> str:
        """num_infections at the start of the child's most recent illness
        episode for this pathogen (-1 = no episode yet). Lets
        ChildIllnessSystem start at most one episode per infection."""
        return f"last_episode_infection_{pathogen_name}"
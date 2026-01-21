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

class AgentPropertyKeys:
    """Centralized string keys for agent properties in AgentGraph.ndata."""
    # Core State
    HEALTH = "health"
    WEALTH = "wealth"
    INITIAL_HEALTH = "initial_health"
    INITIAL_WEALTH = "initial_wealth"

    # Demographics
    HOUSEHOLD_ID = "household_id"
    IS_CHILD = "is_child"

    # Behavior / CPT
    PERSONA_ID = "persona_id"
    ALPHA = "alpha"
    GAMMA = "gamma"
    OMEGA = "omega"
    ETA = "eta"

    # Location & Activity
    X = "x"
    Y = "y"
    TIME_USE = "time_use"
    ACTIVITY_CHOICE = "activity_choice"
    HOME_LOCATION = "home_location"
    SCHOOL_LOCATION = "school_location"
    WORSHIP_LOCATION = "worship_location"
    WATER_LOCATION = "water_location"

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

class EdgePropertyKeys:
    """Centralized string keys for edge properties in AgentGraph.edata."""
    WEIGHT = "weight"
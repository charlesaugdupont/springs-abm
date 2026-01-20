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

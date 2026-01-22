# abm/systems/system.py
from abc import ABC, abstractmethod
from typing import Any
import torch

from abm.state import AgentState
from config import SVEIRConfig

class System(ABC):
    """Abstract Base Class for a core model process."""

    def __init__(self, config: SVEIRConfig):
        self.config = config
        self.device = config.device

    @abstractmethod
    def update(self, agent_state: AgentState, **kwargs: Any):
        """
        Executes the system's logic for a single timestep.
        """
        pass
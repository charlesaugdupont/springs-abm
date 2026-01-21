# abm/systems/child_illness.py
import torch

from .system import System
from abm.state import AgentGraph
from abm.constants import AgentPropertyKeys, Compartment
from abm.agent.illness_mechanics import calculate_illness_severity, calculate_illness_duration

class ChildIllnessSystem(System):
    """
    Manages the state of illness for child agents, including severity and duration.
    """

    def update(self, agent_graph: AgentGraph, **kwargs):
        """
        Updates illness states. This should be called *after* pathogen progression.
        """
        # --- 1. Check for New Illnesses in Children ---
        is_child = agent_graph.ndata[AgentPropertyKeys.IS_CHILD]
        newly_symptomatic_children = self._find_newly_symptomatic(agent_graph, is_child)

        # Process for each pathogen that can cause symptoms
        for pathogen_name in ['rota', 'campy']: # Hardcoded for now
             # This mask should be specific to the pathogen causing the new symptom
            pathogen_status = AgentPropertyKeys.status(pathogen_name)
            p_mask = (agent_graph.ndata[pathogen_status] == Compartment.INFECTIOUS) & newly_symptomatic_children
            if torch.any(p_mask):
                self._initialize_illness(agent_graph, p_mask, pathogen_name)

        # --- 2. Progress Existing Illnesses ---
        is_sick = agent_graph.ndata[AgentPropertyKeys.ILLNESS_DURATION] > 0
        if torch.any(is_sick):
            agent_graph.ndata[AgentPropertyKeys.ILLNESS_DURATION][is_sick] -= 1
            # When duration ends, reset severity
            illness_ended = (agent_graph.ndata[AgentPropertyKeys.ILLNESS_DURATION] == 0)
            agent_graph.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][illness_ended] = 0.0

    def _find_newly_symptomatic(self, agent_graph: AgentGraph, is_child: torch.Tensor) -> torch.Tensor:
        """Identifies child agents who just became infectious and are not already sick."""
        already_sick = agent_graph.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY] > 0
        # A child is newly symptomatic if they are infectious for any reason and not already sick
        # A more robust model would check for *new* transitions to infectious
        is_infectious = torch.zeros_like(is_child)
        for p_config in self.config.pathogens:
             is_infectious |= (agent_graph.ndata[AgentPropertyKeys.status(p_config.name)] == Compartment.INFECTIOUS)

        return is_child & is_infectious & ~already_sick

    def _initialize_illness(self, agent_graph: AgentGraph, mask: torch.Tensor, pathogen_name: str):
        """Calculates and sets the initial severity and duration for an illness."""
        vaccine_status_tensor = None
        if pathogen_name == 'rota':
            status_key = AgentPropertyKeys.status('rota')
            all_statuses = agent_graph.ndata[status_key][mask]
            vaccine_status_tensor = (all_statuses == Compartment.VACCINATED)

        severity = calculate_illness_severity(
            pathogen_name=pathogen_name,
            is_child=agent_graph.ndata[AgentPropertyKeys.IS_CHILD][mask],
            wealth=agent_graph.ndata[AgentPropertyKeys.WEALTH][mask],
            vaccine_status=vaccine_status_tensor,
            num_infections=agent_graph.ndata[AgentPropertyKeys.num_infections(pathogen_name)][mask]
        )
        duration = calculate_illness_duration(severity)

        agent_graph.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][mask] = severity
        agent_graph.ndata[AgentPropertyKeys.ILLNESS_DURATION][mask] = duration
# abm/systems/child_illness.py
import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys, Compartment
from abm.agent.illness_mechanics import calculate_illness_severity, calculate_illness_duration

class ChildIllnessSystem(System):
    """
    Manages the state of illness for child agents, including severity and duration.
    """

    def update(self, agent_state: AgentState, **kwargs):
        """
        Updates illness states. This should be called *after* pathogen progression.
        """
        # --- 1. Check for New Illnesses in Children ---
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD]
        newly_symptomatic_children = self._find_newly_symptomatic(agent_state, is_child)

        # Process for each pathogen that can cause symptoms (now dynamically loaded)
        for pathogen_config in self.config.pathogens:
            pathogen_name = pathogen_config.name
            # This mask should be specific to the pathogen causing the new symptom
            pathogen_status_key = AgentPropertyKeys.status(pathogen_name)
            p_mask = (agent_state.ndata[pathogen_status_key] == Compartment.INFECTIOUS) & newly_symptomatic_children
            if torch.any(p_mask):
                self._initialize_illness(agent_state, p_mask, pathogen_name)

        # --- 2. Progress Existing Illnesses & Apply Health Impact ---
        is_sick = agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION] > 0
        if torch.any(is_sick):
            # Apply health degradation due to being sick
            severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][is_sick]
            health_shock = severity * self.config.steering_parameters.severity_health_impact_factor
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][is_sick]
            agent_state.ndata[AgentPropertyKeys.HEALTH][is_sick] = torch.clamp(current_health - health_shock, min=0.0)

            # Reduce illness duration
            agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION][is_sick] -= 1

            # When duration ends, reset severity
            illness_ended = (agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION] == 0)
            agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][illness_ended] = 0.0

    def _find_newly_symptomatic(self, agent_state: AgentState, is_child: torch.Tensor) -> torch.Tensor:
        """Identifies child agents who just became infectious and are not already sick."""
        already_sick = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY] > 0
        # A child is newly symptomatic if they are infectious for any reason and not already sick
        # A more robust model would check for *new* transitions to infectious
        is_infectious = torch.zeros_like(is_child)
        for p_config in self.config.pathogens:
             is_infectious |= (agent_state.ndata[AgentPropertyKeys.status(p_config.name)] == Compartment.INFECTIOUS)

        return is_child & is_infectious & ~already_sick

    def _initialize_illness(self, agent_state: AgentState, mask: torch.Tensor, pathogen_name: str):
        """Calculates and sets the initial severity and duration, taking the MAXIMUM if already sick."""
        # 1. Calculate the potential new severity and duration for these agents
        vaccine_status_tensor = None
        if pathogen_name == 'rota':
            status_key = AgentPropertyKeys.status('rota')
            vaccine_status_tensor = (agent_state.ndata[status_key][mask] == Compartment.VACCINATED)

        new_severity = calculate_illness_severity(
            pathogen_name=pathogen_name,
            is_child=agent_state.ndata[AgentPropertyKeys.IS_CHILD][mask],
            age=agent_state.ndata[AgentPropertyKeys.AGE][mask],
            wealth=agent_state.ndata[AgentPropertyKeys.WEALTH][mask],
            vaccine_status=vaccine_status_tensor,
            num_infections=agent_state.ndata[AgentPropertyKeys.num_infections(pathogen_name)][mask]
        )
        new_duration = calculate_illness_duration(new_severity)

        # 2. Get current severity
        current_severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][mask]

        # 3. Determine who gets updated (Only if New Severity > Current Severity)
        update_decision_mask = new_severity > current_severity

        # We need to map the update_decision_mask (which is a subset) back to the full agent list
        # We do this by applying the values only where update_decision_mask is True within the subset
        
        # Get indices of the subset (the agents currently being processed)
        subset_indices = mask.nonzero(as_tuple=True)[0]
        
        # Filter for those who actually need an update
        final_update_indices = subset_indices[update_decision_mask]
        
        # 4. Apply Updates
        if len(final_update_indices) > 0:
            # We must index new_severity/new_duration using update_decision_mask to get the matching values
            agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][final_update_indices] = new_severity[update_decision_mask]
            agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION][final_update_indices] = new_duration[update_decision_mask]
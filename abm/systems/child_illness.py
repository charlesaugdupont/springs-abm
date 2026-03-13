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
        Updates illness states. Must be called *after* pathogen progression so
        that compartment statuses reflect the current day's transitions.
        """
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD]

        # ------------------------------------------------------------------
        # 1. Initialise illness for children who just became infectious
        # ------------------------------------------------------------------
        # "Already sick" means they currently have an active episode
        already_sick = agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION] > 0

        # Any infectious status across pathogens
        is_infectious_any = torch.zeros_like(is_child)
        for p_config in self.config.pathogens:
            status_key = AgentPropertyKeys.status(p_config.name)
            is_infectious_any |= (
                agent_state.ndata[status_key] == Compartment.INFECTIOUS
            )

        newly_symptomatic_children = (
            is_child & is_infectious_any & ~already_sick
        )

        # For each pathogen, initialise illness for the subset of newly
        # symptomatic children who are infectious with that pathogen.
        for p_config in self.config.pathogens:
            pathogen_name = p_config.name
            status_key = AgentPropertyKeys.status(pathogen_name)
            p_mask = (
                newly_symptomatic_children
                & (agent_state.ndata[status_key] == Compartment.INFECTIOUS)
            )
            if torch.any(p_mask):
                self._initialize_illness(agent_state, p_mask, pathogen_name)

        # ------------------------------------------------------------------
        # 2. Progress existing illnesses and apply daily health impact
        # ------------------------------------------------------------------
        duration = agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION]
        is_sick = duration > 0

        if torch.any(is_sick):
            severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][is_sick]
            health_shock = (
                severity
                * self.config.steering_parameters.severity_health_impact_factor
            )
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][is_sick]
            agent_state.ndata[AgentPropertyKeys.HEALTH][is_sick] = torch.clamp(
                current_health - health_shock, min=0.0
            )

            # Count down the illness timer (and clamp at 0 to avoid negatives)
            duration_new = duration.clone()
            duration_new[is_sick] -= 1
            duration_new = torch.clamp(duration_new, min=0)
            agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION] = duration_new

        # ------------------------------------------------------------------
        # 3. Enforce invariant: if duration == 0, severity must be 0
        # ------------------------------------------------------------------
        duration = agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION]
        illness_ended = duration == 0
        agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][illness_ended] = 0.0

        # ------------------------------------------------------------------
        # 4. Passive health recovery for agents who are not currently sick
        # ------------------------------------------------------------------
        is_not_sick = illness_ended
        if torch.any(is_not_sick):
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][is_not_sick]
            wealth = agent_state.ndata[AgentPropertyKeys.WEALTH][is_not_sick]

            base_recovery = self.config.steering_parameters.daily_health_recovery_rate
            # Wealthier agents recover up to 2× faster (wealth ∈ [0, 1] → multiplier ∈ [1, 2])
            wealth_multiplier = 1.0 + wealth

            new_health = current_health + (base_recovery * wealth_multiplier)
            agent_state.ndata[AgentPropertyKeys.HEALTH][is_not_sick] = torch.clamp(
                new_health, max=1.0
            )

    def _initialize_illness(
        self, agent_state: AgentState, mask: torch.Tensor, pathogen_name: str
    ):
        """
        Calculates and sets the initial severity and duration for newly ill
        children. If a child is somehow already sick (duration > 0), the new
        episode only overwrites the existing one when the new severity is higher.
        """
        vaccine_status_tensor = None
        if pathogen_name == "rota":
            status_key = AgentPropertyKeys.status("rota")
            vaccine_status_tensor = (
                agent_state.ndata[status_key][mask] == Compartment.VACCINATED
            )

        new_severity = calculate_illness_severity(
            pathogen_name=pathogen_name,
            is_child=agent_state.ndata[AgentPropertyKeys.IS_CHILD][mask],
            age=agent_state.ndata[AgentPropertyKeys.AGE][mask],
            wealth=agent_state.ndata[AgentPropertyKeys.WEALTH][mask],
            vaccine_status=vaccine_status_tensor,
            num_infections=agent_state.ndata[
                AgentPropertyKeys.num_infections(pathogen_name)
            ][mask],
        )
        new_duration = calculate_illness_duration(new_severity)

        current_severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][mask]

        # Only update agents for whom the new episode is more severe
        update_decision_mask = new_severity > current_severity
        subset_indices = mask.nonzero(as_tuple=True)[0]
        final_update_indices = subset_indices[update_decision_mask]

        if len(final_update_indices) > 0:
            agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][final_update_indices] = (
                new_severity[update_decision_mask]
            )
            agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION][final_update_indices] = (
                new_duration[update_decision_mask]
            )
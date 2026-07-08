# abm/systems/child_illness.py

import torch

from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys, Compartment
from abm.agent.illness_mechanics import calculate_illness_severity, calculate_illness_duration


class ChildIllnessSystem(System):
    """
    Manages the state of illness for child agents, including severity and
    duration — tracked independently per pathogen, so a child can be
    concurrently ill with more than one pathogen at once. Each pathogen's
    episode has its own onset, decay, and resolution. The daily health toll
    sums the contribution of every currently active episode.
    
    Per-episode data collection
    ---------------------------
    Each time a new illness episode is initialised for a child (for a given
    pathogen), the system appends a record to ``self.episode_log``:

        {
            'agent_idx':        int,
            'pathogen':         str,
            'initial_severity': float,
            'initial_duration': int,
            'timestep':         int,   # day the episode started
        }

    A child concurrently infected with two pathogens will therefore produce
    two separate log entries. The log is a plain list of dicts so it is
    trivially serialisable (pickle / JSON).
    """

    def __init__(self, config):
        super().__init__(config)
        self.episode_log: list[dict] = []
        self._current_timestep: int = 0

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, agent_state: AgentState, **kwargs):
        """
        Updates illness states. Must be called after pathogen progression so
        that compartment statuses reflect the current day's transitions.
        """
        self._current_timestep = kwargs.get("timestep", self._current_timestep + 1)

        illness_cfg = self.config.illness_mechanics
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD]

        # ------------------------------------------------------------------
        # 1. Initialise illness for children who just became infectious with
        #    a pathogen for which they do NOT already have an active
        #    episode. Each pathogen is evaluated independently, so a child
        #    with an ongoing episode of pathogen A can still start a fresh
        #    episode of pathogen B.
        # ------------------------------------------------------------------
        for p_config in self.config.pathogens:
            pathogen_name = p_config.name
            status_key = AgentPropertyKeys.status(pathogen_name)
            duration_key = AgentPropertyKeys.illness_duration(pathogen_name)

            already_sick_this_pathogen = agent_state.ndata[duration_key] > 0
            is_infectious = agent_state.ndata[status_key] == Compartment.INFECTIOUS

            newly_symptomatic = is_child & is_infectious & ~already_sick_this_pathogen
            if torch.any(newly_symptomatic):
                self._initialize_illness(agent_state, newly_symptomatic, pathogen_name, illness_cfg)

        # ------------------------------------------------------------------
        # 2. Progress every active episode and apply the combined daily
        #    health impact (concurrent infections compound the toll).
        # ------------------------------------------------------------------
        health_impact_factor = self.config.steering_parameters.severity_health_impact_factor
        total_health_shock = torch.zeros(agent_state.num_nodes(), device=agent_state.device)
        any_active_before = torch.zeros_like(is_child)
        any_active_after = torch.zeros_like(is_child)

        for p_config in self.config.pathogens:
            pathogen_name = p_config.name
            duration_key  = AgentPropertyKeys.illness_duration(pathogen_name)
            severity_key  = AgentPropertyKeys.symptom_severity(pathogen_name)

            duration = agent_state.ndata[duration_key]
            is_sick = duration > 0
            any_active_before |= is_sick

            if torch.any(is_sick):
                total_health_shock[is_sick] += (
                    agent_state.ndata[severity_key][is_sick] * health_impact_factor
                )

                duration_new = duration.clone()
                duration_new[is_sick] -= 1
                duration_new = torch.clamp(duration_new, min=0)
                agent_state.ndata[duration_key] = duration_new

            # Enforce invariant: once this pathogen's duration hits 0, its
            # severity resets to 0 (that specific episode has ended).
            illness_ended = agent_state.ndata[duration_key] == 0
            agent_state.ndata[severity_key][illness_ended] = 0.0
            any_active_after |= (agent_state.ndata[duration_key] > 0)

        if torch.any(any_active_before):
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][any_active_before]
            agent_state.ndata[AgentPropertyKeys.HEALTH][any_active_before] = torch.clamp(
                current_health - total_health_shock[any_active_before], min=0.0
            )

        # ------------------------------------------------------------------
        # 3. Passive health recovery for agents with NO active illness
        #    (across any pathogen).
        # ------------------------------------------------------------------
        is_not_sick = ~any_active_after
        if torch.any(is_not_sick):
            current_health = agent_state.ndata[AgentPropertyKeys.HEALTH][is_not_sick]
            wealth = agent_state.ndata[AgentPropertyKeys.WEALTH][is_not_sick]

            base_recovery = self.config.steering_parameters.daily_health_recovery_rate
            wealth_multiplier = 1.0 + wealth

            new_health = current_health + (base_recovery * wealth_multiplier)
            agent_state.ndata[AgentPropertyKeys.HEALTH][is_not_sick] = torch.clamp(
                new_health, max=1.0
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialize_illness(
        self,
        agent_state: AgentState,
        mask: torch.Tensor,
        pathogen_name: str,
        illness_cfg,
    ):
        """
        Calculates and sets the initial severity and duration for a new
        episode of `pathogen_name`, then logs it. `mask` is guaranteed (by
        the caller) to only include children who don't currently have an
        active episode of this specific pathogen.
        """
        vaccine_status_tensor = None
        if pathogen_name == "rota":
            status_key = AgentPropertyKeys.status("rota")
            vaccine_status_tensor = (
                agent_state.ndata[status_key][mask] == Compartment.VACCINATED
            )

        new_severity = calculate_illness_severity(
            pathogen_name  = pathogen_name,
            is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD][mask],
            age = agent_state.ndata[AgentPropertyKeys.AGE][mask],
            vaccine_status = vaccine_status_tensor,
            num_infections = agent_state.ndata[AgentPropertyKeys.num_infections(pathogen_name)][mask] - 1,
            cfg = illness_cfg,
        )
        new_duration = calculate_illness_duration(new_severity, illness_cfg)

        severity_key = AgentPropertyKeys.symptom_severity(pathogen_name)
        duration_key = AgentPropertyKeys.illness_duration(pathogen_name)

        agent_indices = mask.nonzero(as_tuple=True)[0]
        agent_state.ndata[severity_key][agent_indices] = new_severity
        agent_state.ndata[duration_key][agent_indices] = new_duration

        # --- Episode logging ---
        for i, agent_idx in enumerate(agent_indices):
            self.episode_log.append({
                'agent_idx':        agent_idx.item(),
                'pathogen':         pathogen_name,
                'initial_severity': new_severity[i].item(),
                'initial_duration': new_duration[i].item(),
                'timestep':         self._current_timestep,
            })
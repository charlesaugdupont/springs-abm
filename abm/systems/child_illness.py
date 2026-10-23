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

    Each infection produces at most one episode. The episode's duration is
    set from its severity and is independent of the infectious period, so a
    child may remain infectious (shedding without symptoms) after the
    episode ends; they are not given a second episode for that infection.
    
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
            'age_months':       float, # child's age at onset
            'realized_duration': int | None,
        }

    ``realized_duration`` is the number of days the episode was actually
    active - which differs from ``initial_duration`` whenever care-seeking
    resolves an episode early (CareSeekingSystem._apply_treatment_outcome).
    It is filled in on the first day the episode is observed to have ended,
    and stays None for an episode still open when the simulation stops. This
    is the quantity comparable to a cohort study's observed diarrhoea
    duration (e.g. MAL-ED, abm/evidence/maled.py).

    A child concurrently infected with two pathogens will therefore produce
    two separate log entries. The log is a plain list of dicts so it is
    trivially serialisable (pickle / JSON).
    """

    def __init__(self, config):
        super().__init__(config)
        self.episode_log: list[dict] = []
        self._current_timestep: int = 0
        # Per pathogen, the episode_log index of each agent's currently open
        # episode (-1 if none). Lazily allocated on the first update().
        self._open_episode: dict[str, torch.Tensor] = {}
        # Every child infection's one decision point - the day it became
        # infectious and whether it produced an episode - per pathogen, for
        # the MAL-ED observation layer (experiments/observation.py). Chunks of
        # (agent_idx, birth_day, symptomatic) tensors with their day; pure
        # bookkeeping, no RNG. See get_infection_log().
        self._infection_log: dict[str, list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        # Timestep whose episode onsets begin_day() already processed, so that
        # update() does not process them twice.
        self._onsets_done_for: int | None = None

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def begin_day(self, agent_state: AgentState, timestep: int):
        """Episode onsets for the day, BEFORE the day's transmission.

        Used when IllnessMechanicsConfig.onset_before_transmission is True
        (called from sveir_step right after disease progression), so that a
        newly infectious child's infectiousness already reflects whether it
        is ill (PathogenConfig.asymptomatic_relative_infectiousness). update()
        then skips steps 0-1 for this timestep.
        """
        self._current_timestep = timestep
        self._close_ended_episodes(agent_state)
        self._start_new_episodes(agent_state)
        self._onsets_done_for = timestep

    def update(self, agent_state: AgentState, **kwargs):
        """
        Updates illness states. Must be called after pathogen progression so
        that compartment statuses reflect the current day's transitions.
        """
        self._current_timestep = kwargs.get("timestep", self._current_timestep + 1)
        if self._onsets_done_for != self._current_timestep:
            # 0. Close the log entries of episodes that ended since the last
            #    update (naturally, or cured by care-seeking after it). Must
            #    run before step 1, which may open a new episode for the same
            #    agent.
            self._close_ended_episodes(agent_state)
            self._start_new_episodes(agent_state)
        self._progress_episodes(agent_state)

    def _start_new_episodes(self, agent_state: AgentState):
        illness_cfg = self.config.illness_mechanics
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD]

        # ------------------------------------------------------------------
        # 1. Initialise illness for children who are infectious with a
        #    pathogen for which they do NOT already have an active episode
        #    AND whose current infection has not yet produced one. Each
        #    pathogen is evaluated independently, so a child with an ongoing
        #    episode of pathogen A can still start a fresh episode of
        #    pathogen B.
        # ------------------------------------------------------------------
        for p_config in self.config.pathogens:
            pathogen_name = p_config.name
            status_key = AgentPropertyKeys.status(pathogen_name)
            duration_key = AgentPropertyKeys.illness_duration(pathogen_name)

            already_sick_this_pathogen = agent_state.ndata[duration_key] > 0
            is_infectious = agent_state.ndata[status_key] == Compartment.INFECTIOUS

            # One episode per infection. Illness runs on its own clock (a
            # duration set from severity), independent of the infectious
            # period, so a child can still be INFECTIOUS when their episode
            # ends. Without this check a fresh episode (new severity, new
            # duration, new log entry) started the next day for the SAME
            # infection, inflating episode counts, health losses and
            # care-seeking decisions. num_infections increases at every new
            # exposure, so it identifies the current infection.
            infection_count = agent_state.ndata[AgentPropertyKeys.num_infections(pathogen_name)]
            last_key = AgentPropertyKeys.last_episode_infection(pathogen_name)
            if last_key in agent_state.ndata:
                new_infection = infection_count > agent_state.ndata[last_key]
            else:
                new_infection = torch.ones_like(is_infectious)

            episode_candidates = (
                is_child & is_infectious & ~already_sick_this_pathogen & new_infection
            )
            if torch.any(episode_candidates):
                # Not every infection produces symptoms. A subclinical
                # infection still progresses E -> I -> R and still transmits,
                # but acquires no severity, no duration, no episode-log entry
                # and triggers no care-seeking decision. The probability comes
                # from PathogenConfig.symptomatic_fraction, times its age
                # profile when one is set (MAL-ED, see abm/evidence/maled.py). When the probability is exactly 1.0 for
                # everyone the draw is skipped entirely, so no RNG is consumed.
                symptomatic = episode_candidates
                cand_idx = episode_candidates.nonzero(as_tuple=True)[0]
                frac = symptomatic_probability(
                    p_config, agent_state.ndata[AgentPropertyKeys.AGE][cand_idx],
                    # infections before the current one
                    infection_count[cand_idx] - 1,
                )
                if frac is not None:
                    draws = torch.rand(len(cand_idx), device=agent_state.device)
                    keep = torch.zeros_like(episode_candidates)
                    keep[cand_idx] = draws < frac
                    symptomatic = keep

                if torch.any(symptomatic):
                    self._initialize_illness(
                        agent_state, symptomatic, pathogen_name, illness_cfg, p_config
                    )
                self._log_infections(agent_state, pathogen_name, episode_candidates, symptomatic)
                # Mark the infection as having had its one chance of becoming an
                # episode for EVERY candidate, symptomatic or not - otherwise an
                # infection that lost the symptomatic draw would be re-rolled on
                # every subsequent day it stays infectious, which for a
                # three-week shedding period would recover most of the
                # suppressed episodes and silently defeat symptomatic_fraction.
                if last_key in agent_state.ndata:
                    agent_state.ndata[last_key][episode_candidates] = infection_count[episode_candidates]

    def _progress_episodes(self, agent_state: AgentState):
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD]
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
        p_config=None,
    ):
        """
        Calculates and sets the initial severity and duration for a new
        episode of `pathogen_name`, then logs it. `mask` is guaranteed (by
        the caller) to only include children who don't currently have an
        active episode of this specific pathogen.
        """
        # Vaccination history, not the current compartment. Every child in
        # `mask` is INFECTIOUS right now, so a check for status == VACCINATED
        # was always False (a vaccinated child moves V -> E -> I before any
        # episode starts) and immunity_factor_vaccine never applied. The
        # ever-vaccinated flag is set at S -> V and never cleared, so the
        # vaccine's severity reduction applies to every later episode.
        vaccine_status_tensor = None
        vacc_key = AgentPropertyKeys.ever_vaccinated(pathogen_name)
        if vacc_key in agent_state.ndata:
            vaccine_status_tensor = agent_state.ndata[vacc_key][mask].bool()

        new_severity = calculate_illness_severity(
            pathogen_name  = pathogen_name,
            is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD][mask],
            age = agent_state.ndata[AgentPropertyKeys.AGE][mask],
            vaccine_status = vaccine_status_tensor,
            num_infections = agent_state.ndata[AgentPropertyKeys.num_infections(pathogen_name)][mask] - 1,
            cfg = illness_cfg,
        )
        new_duration = calculate_illness_duration(
            new_severity,
            illness_cfg,
            duration_scale=float(getattr(p_config, "duration_scale", 1.0) or 1.0),
        )

        severity_key = AgentPropertyKeys.symptom_severity(pathogen_name)
        duration_key = AgentPropertyKeys.illness_duration(pathogen_name)

        agent_indices = mask.nonzero(as_tuple=True)[0]
        agent_state.ndata[severity_key][agent_indices] = new_severity
        agent_state.ndata[duration_key][agent_indices] = new_duration
        sym_key = AgentPropertyKeys.symptomatic_infection(pathogen_name)
        if sym_key in agent_state.ndata:
            agent_state.ndata[sym_key][agent_indices] = True

        # --- Episode logging ---
        open_idx = self._open_episode_tensor(agent_state, pathogen_name)
        ages = agent_state.ndata[AgentPropertyKeys.AGE][agent_indices].tolist()
        for i, agent_idx in enumerate(agent_indices.tolist()):
            open_idx[agent_idx] = len(self.episode_log)
            self.episode_log.append({
                'agent_idx':         agent_idx,
                'pathogen':          pathogen_name,
                'initial_severity':  new_severity[i].item(),
                'initial_duration':  new_duration[i].item(),
                'timestep':          self._current_timestep,
                'age_months':        float(ages[i]),
                'realized_duration': None,
            })

    def _log_infections(self, agent_state: AgentState, pathogen_name: str,
                        candidates: torch.Tensor, symptomatic: torch.Tensor):
        idx = candidates.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        birth = agent_state.ndata[AgentPropertyKeys.BIRTH_DAY][idx] \
            if AgentPropertyKeys.BIRTH_DAY in agent_state.ndata else torch.full_like(idx, -1)
        self._infection_log.setdefault(pathogen_name, []).append(
            (self._current_timestep, idx.clone(), birth.clone(), symptomatic[idx].clone())
        )

    def get_infection_log(self, pathogen_name: str) -> dict[str, "np.ndarray"]:
        """Every child infection of this pathogen that became infectious:
        arrays agent_idx, birth_day, day (first infectious day) and
        symptomatic (whether it produced an episode). A slot's occupant is
        identified by (agent_idx, birth_day)."""
        import numpy as np
        chunks = self._infection_log.get(pathogen_name, [])
        if not chunks:
            empty = np.zeros(0, dtype=np.int64)
            return {"agent_idx": empty, "birth_day": empty, "day": empty, "symptomatic": empty.astype(bool)}
        return {
            "agent_idx": np.concatenate([c[1].cpu().numpy() for c in chunks]).astype(np.int64),
            "birth_day": np.concatenate([c[2].cpu().numpy() for c in chunks]).astype(np.int64),
            "day": np.concatenate([np.full(len(c[1]), c[0], dtype=np.int64) for c in chunks]),
            "symptomatic": np.concatenate([c[3].cpu().numpy() for c in chunks]).astype(bool),
        }

    def _open_episode_tensor(self, agent_state: AgentState, pathogen_name: str) -> torch.Tensor:
        t = self._open_episode.get(pathogen_name)
        if t is None or t.numel() != agent_state.num_nodes():
            t = torch.full((agent_state.num_nodes(),), -1, dtype=torch.long)
            self._open_episode[pathogen_name] = t
        return t

    def censor_open_episodes(self, agent_indices: torch.Tensor):
        """Closes the given agents' open episodes WITHOUT a realized duration.

        Called by DemographySystem when a child ages out of the modelled
        population mid-episode: the episode's true length is unknown, so it is
        marked censored (realized_duration stays None) rather than being
        recorded as having ended that day.
        """
        idx_list = agent_indices.tolist()
        for open_idx in self._open_episode.values():
            for agent_idx in idx_list:
                log_idx = int(open_idx[agent_idx])
                if log_idx >= 0:
                    self.episode_log[log_idx]['censored'] = True
                    open_idx[agent_idx] = -1

    def _close_ended_episodes(self, agent_state: AgentState):
        """Stamps realized_duration on log entries whose episode has ended.

        An episode started on day t0 is active on day t0 itself (its first
        health toll is applied in the same update), so an episode first seen
        to have ended on day t lasted t - t0 days: 1 day for one cured by
        care-seeking on its first day, d days for one that ran its full
        initial duration d. Pure bookkeeping - no RNG.
        """
        for pathogen_name, open_idx in self._open_episode.items():
            is_open = open_idx >= 0
            if not torch.any(is_open):
                continue
            duration = agent_state.ndata[AgentPropertyKeys.illness_duration(pathogen_name)].cpu()
            ended = is_open & (duration == 0)
            for agent_idx in ended.nonzero(as_tuple=True)[0].tolist():
                entry = self.episode_log[int(open_idx[agent_idx])]
                entry['realized_duration'] = self._current_timestep - entry['timestep']
                open_idx[agent_idx] = -1


def symptomatic_probability(p_config, ages_months: torch.Tensor,
                            prior_infections: torch.Tensor | None = None) -> torch.Tensor | float | None:
    """P(symptomatic episode | infection) for children of the given ages.

    symptomatic_fraction, multiplied by the age profile's multiplier for each
    child's band when PathogenConfig.symptomatic_fraction_age_profile is set,
    and by exp(-symptomatic_reduction_per_prior_infection * prior_infections)
    when that is non-zero, clamped to [0, 1]. Each profile band is
    (upper_age_months, multiplier); a child of age a falls in the first band
    with a <= upper_age_months, and children older than the last upper edge
    take the last band.

    Returns None when the probability is exactly 1 for every child (the caller
    then skips the Bernoulli draw, consuming no RNG), a float when it does not
    depend on age or history, or a per-child tensor otherwise.
    """
    frac = float(getattr(p_config, "symptomatic_fraction", 1.0))
    profile = getattr(p_config, "symptomatic_fraction_age_profile", None)
    lam = float(getattr(p_config, "symptomatic_reduction_per_prior_infection", 0.0) or 0.0)
    if lam > 0.0 and prior_infections is not None:
        base = symptomatic_probability(p_config, ages_months)     # age part (history-free)
        if base is None:
            base = 1.0
        factor = torch.exp(-lam * prior_infections.clamp(min=0).float())
        return (base * factor).clamp(0.0, 1.0)
    if not profile:
        return None if frac >= 1.0 else frac
    probs = torch.tensor(
        [min(1.0, max(0.0, frac * m)) for _, m in profile],
        dtype=torch.float, device=ages_months.device,
    )
    if bool(torch.all(probs >= 1.0)):
        return None
    edges = torch.tensor([a for a, _ in profile], dtype=ages_months.dtype, device=ages_months.device)
    band = torch.bucketize(ages_months, edges).clamp_(max=len(profile) - 1)
    return probs[band]

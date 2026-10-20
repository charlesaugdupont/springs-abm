# abm/systems/demography.py
"""
Ageing, under-5 turnover and age-dependent activity.

Why this exists
---------------
The original model had static ages and a closed population. Children never
aged and nobody was born, so every infection permanently lowered someone's
susceptibility and incidence declined for as long as the model ran (e.g.
Campylobacter fell from ~3.1 to ~1.7 infections per child-year over two
simulated years) - there was no endemic steady state for a calibration to
target, and the 0-24 month window the MAL-ED targets refer to contained the
same children for the whole run.

What it does (once per day, at the end of the day)
--------------------------------------------------
1. Every agent's AGE (months) advances by one day.
2. A child reaching DemographyConfig.child_max_age_months leaves the
   modelled under-5 population. Its agent slot is reused for a newborn in the
   same household: age 0, naive to every pathogen, no illness, health reset to
   the slot's initial draw. The number of under-5s, their (uniform) age
   distribution and the household structure therefore stay fixed. agent_idx
   values in logs refer to SLOTS, which can hold several children in turn;
   BIRTH_DAY distinguishes them.
3. Time use is made age-dependent: a child below school_entry_age_months, or
   one whose slot does not attend, never goes to school (TIME_USE is
   TIME_USE_BASE with the school weight removed).

Draws no random numbers during the run: turnover happens at deterministic
ages, and school attendance is drawn once per slot at initialisation.
"""
from __future__ import annotations

import torch

from .system import System
from abm.state import AgentState
from abm.constants import Activity, AgentPropertyKeys, Compartment

#: Mean days per month (365.25 / 12).
DAYS_PER_MONTH = 365.25 / 12.0


def effective_time_use(
    time_use_base: torch.Tensor,
    is_child: torch.Tensor,
    age_months: torch.Tensor,
    attends_school: torch.Tensor,
    school_entry_age_months: float,
) -> torch.Tensor:
    """TIME_USE from TIME_USE_BASE, removing school for ineligible children.

    A child is eligible if its slot attends and it is at least the entry age.
    For an ineligible child the school weight is set to zero and the row is
    renormalised (for the under-5 base rows, which only have home and school
    weight, that leaves home with probability 1). Adults and eligible children
    keep their base row unchanged - bit for bit.
    """
    tu = time_use_base.clone()
    ineligible = is_child.bool() & ~(attends_school.bool() & (age_months >= school_entry_age_months))
    if torch.any(ineligible):
        rows = tu[ineligible]
        rows[:, Activity.SCHOOL] = 0.0
        rows = rows / (rows.sum(dim=1, keepdim=True) + 1e-9)
        tu[ineligible] = rows
    return tu


class DemographySystem(System):
    """Daily ageing and under-5 turnover. See the module docstring."""

    def __init__(self, config):
        super().__init__(config)
        #: One record per child born IN the simulation who has since left it:
        #: {"birth_day", "exit_day", "first_infection_day": {pathogen: day or -1}}.
        #: Children still present at the end are read directly from agent state.
        self.cohort_log: list[dict] = []

    def update(self, agent_state: AgentState, **kwargs) -> torch.Tensor:
        """Advance one day. Returns the indices of slots renewed as newborns."""
        cfg = self.config.demography
        device = agent_state.device
        empty = torch.zeros(0, dtype=torch.long, device=device)
        if not cfg.ageing:
            return empty
        timestep = int(kwargs.get("timestep", 0))
        illness_system = kwargs.get("illness_system")

        age = agent_state.ndata[AgentPropertyKeys.AGE]
        previous_age = age.clone()
        age += 1.0 / DAYS_PER_MONTH
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD].bool()

        leaving = is_child & (age >= cfg.child_max_age_months)
        reborn = leaving.nonzero(as_tuple=True)[0]
        if reborn.numel() > 0:
            self._archive(agent_state, reborn, timestep)
            self._renew_as_newborns(agent_state, reborn, timestep, illness_system)

        # Children whose school eligibility may have changed today: those who
        # crossed the entry age, plus the newborns.
        entry = cfg.school_entry_age_months
        crossed = is_child & (previous_age < entry) & (age >= entry)
        changed = crossed.clone()
        changed[reborn] = True
        if torch.any(changed):
            idx = changed.nonzero(as_tuple=True)[0]
            agent_state.ndata[AgentPropertyKeys.TIME_USE][idx] = effective_time_use(
                agent_state.ndata[AgentPropertyKeys.TIME_USE_BASE][idx],
                is_child[idx],
                age[idx],
                agent_state.ndata[AgentPropertyKeys.ATTENDS_SCHOOL][idx],
                entry,
            )
        return reborn

    # ------------------------------------------------------------------

    def _archive(self, agent_state: AgentState, idx: torch.Tensor, timestep: int):
        birth = agent_state.ndata[AgentPropertyKeys.BIRTH_DAY][idx]
        born_in_sim = (birth >= 0).nonzero(as_tuple=True)[0]
        if born_in_sim.numel() == 0:
            return
        firsts = {
            p.name: agent_state.ndata[AgentPropertyKeys.first_infection_day(p.name)][idx][born_in_sim].tolist()
            for p in self.config.pathogens
        }
        for j, b in enumerate(birth[born_in_sim].tolist()):
            self.cohort_log.append({
                "birth_day": int(b),
                "exit_day": timestep,
                "first_infection_day": {name: int(v[j]) for name, v in firsts.items()},
            })

    def _renew_as_newborns(self, agent_state: AgentState, idx: torch.Tensor, timestep: int,
                           illness_system=None):
        nd = agent_state.ndata
        cfg = self.config.demography
        nd[AgentPropertyKeys.AGE][idx] = nd[AgentPropertyKeys.AGE][idx] - cfg.child_max_age_months
        nd[AgentPropertyKeys.BIRTH_DAY][idx] = timestep
        nd[AgentPropertyKeys.HEALTH][idx] = nd[AgentPropertyKeys.INITIAL_HEALTH][idx]
        for p in self.config.pathogens:
            name = p.name
            nd[AgentPropertyKeys.status(name)][idx] = Compartment.SUSCEPTIBLE
            nd[AgentPropertyKeys.exposure_time(name)][idx] = 0
            nd[AgentPropertyKeys.num_infections(name)][idx] = 0
            nd[AgentPropertyKeys.symptom_severity(name)][idx] = 0.0
            nd[AgentPropertyKeys.illness_duration(name)][idx] = 0
            nd[AgentPropertyKeys.ever_vaccinated(name)][idx] = False
            nd[AgentPropertyKeys.last_episode_infection(name)][idx] = -1
            nd[AgentPropertyKeys.first_infection_day(name)][idx] = -1
        if illness_system is not None:
            illness_system.censor_open_episodes(idx)

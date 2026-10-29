# abm/economy/cedi.py
"""
The household economy in 2026 Ghana cedis: initialisation and shared helpers.

Used only when EconomyConfig.model == "cedi" (see config.EconomyConfig for the
model and docs/economy_design_proposal.md for the evidence). Nothing here runs,
and no agent property it defines exists, in the legacy economy.

Household-level quantities (cash, consumption target, quintile, insurance) are
stored per agent and broadcast to every member, exactly as WEALTH is, so the
systems that read them need no household table. WEALTH itself holds the
care-seeking wealth term W = clip(cash / (wealth_scale_days * target), 0, 1).
"""
from __future__ import annotations

import numpy as np
import torch

from abm.constants import AgentPropertyKeys as K

#: Economic age groups (AgentPropertyKeys.ECON_AGE_GROUP).
UNDER_5, AGE_5_14, AGE_15_64, AGE_65_PLUS = 0, 1, 2, 3
N_QUINTILES = 5
#: Usual provider for a child's diarrhoea (AgentPropertyKeys.PROVIDER).
PRIMARY, HOSPITAL, PRIVATE, DRUG_SELLER = 0, 1, 2, 3
N_PROVIDERS = 4


def is_cedi(config) -> bool:
    eco = getattr(config, "economy", None)
    return eco is not None and eco.model == "cedi"


def _age_groups(rng: np.random.Generator, is_child: np.ndarray, is_head: np.ndarray,
                shares: tuple[float, float, float]) -> np.ndarray:
    """Economic age group for every agent: under-fives keep group 0; heads are
    15-64 or 65+ in the census ratio; the other agents aged 5+ take whatever
    makes the shares of the whole population aged 5+ equal `shares` in
    expectation."""
    p_5_14, p_15_64, p_65 = shares
    group = np.full(is_child.shape, UNDER_5, dtype=np.int64)
    five_plus = ~is_child
    n5, n_head = int(five_plus.sum()), int(is_head.sum())
    n_other = n5 - n_head
    p_old_15plus = p_65 / (p_15_64 + p_65) if (p_15_64 + p_65) > 0 else 0.0

    u_head = rng.random(n_head)
    group[is_head] = np.where(u_head < p_old_15plus, AGE_65_PLUS, AGE_15_64)

    other = five_plus & ~is_head
    if n_other > 0:
        q_young = min(1.0, p_5_14 * n5 / n_other)
        q_old = float(np.clip((p_65 * n5 - p_old_15plus * n_head) / n_other, 0.0, 1.0 - q_young))
        u = rng.random(n_other)
        group[other] = np.where(u < q_young, AGE_5_14, np.where(u < q_young + q_old, AGE_65_PLUS, AGE_15_64))
    return group


def initialise(agent_state, config) -> dict:
    """Gives every agent and household its economic attributes (cedi mode).

    Call after AgentFactory.initialize_agent_properties. Draws only from its
    own generator (config.seed + economy.rng_seed_offset). Returns a summary
    of the realised economy (per-quintile earning levels etc.) for reporting.
    """
    eco = config.economy
    rng = np.random.default_rng(int(config.seed) + int(eco.rng_seed_offset))
    nd = agent_state.ndata
    device = nd[K.HOUSEHOLD_ID].device

    hh = nd[K.HOUSEHOLD_ID].cpu().numpy()
    is_child = nd[K.IS_CHILD].cpu().numpy().astype(bool)
    _, first_idx, inv = np.unique(hh, return_index=True, return_inverse=True)
    n_hh = len(first_idx)
    is_head = np.zeros(len(hh), dtype=bool)
    is_head[first_idx] = True
    if np.any(is_child[first_idx]):
        raise ValueError("cedi economy: every household's first member must be aged 5+")
    size = np.bincount(inv, minlength=n_hh).astype(np.float64)

    # Roles by age, and strata.
    a_shares = np.asarray(eco.age_group_shares_5plus, dtype=np.float64)
    group = _age_groups(rng, is_child, is_head, tuple(a_shares / a_shares.sum()))
    earner = group >= AGE_15_64
    q_shares = np.asarray(eco.quintile_shares, dtype=np.float64)
    quintile = rng.choice(N_QUINTILES, size=n_hh, p=q_shares / q_shares.sum())

    # Earning level per quintile: the quintile's total full-health income is
    # (1 + s) x GLSS per-capita consumption x its population.
    c = np.asarray(eco.consumption_per_capita_daily, dtype=np.float64)
    s = float(eco.savings_rate)
    earners_hh = np.bincount(inv, weights=earner.astype(np.float64), minlength=n_hh)
    pop_q = np.bincount(quintile, weights=size, minlength=N_QUINTILES)
    earners_q = np.bincount(quintile, weights=earners_hh, minlength=N_QUINTILES)
    ybar = np.where(earners_q > 0, (1.0 + s) * c * pop_q / np.maximum(earners_q, 1.0), 0.0)

    sd = float(eco.earning_spread_sdlog)
    factor = np.exp(sd * rng.standard_normal(n_hh) - 0.5 * sd * sd)
    y = np.where(earner, ybar[quintile[inv]] * factor[inv], 0.0)
    target = np.bincount(inv, weights=y, minlength=n_hh) / (1.0 + s)

    insured = rng.random(n_hh) < float(eco.insured_prob)
    # buffer: days of consumption, growing with consumption per person when the elasticity > 0
    c_ref = float(c[N_QUINTILES // 2])
    per_person = target / size
    buffer_days = float(eco.buffer_days) * (per_person / c_ref) ** float(eco.buffer_wealth_elasticity)
    cap = buffer_days * target
    cash = rng.random(n_hh) * float(eco.initial_buffer_fraction) * cap
    w = wealth_term(cash, target, eco.wealth_scale_days)
    shares = np.asarray(eco.provider_shares_by_quintile, dtype=np.float64)
    shares = shares / shares.sum(axis=1, keepdims=True)
    cum = np.cumsum(shares, axis=1)[quintile]
    provider = (rng.random(n_hh)[:, None] > cum[:, :-1]).sum(axis=1)

    def put(key, values, dtype):
        nd[key] = torch.as_tensor(values, dtype=dtype).to(device)

    put(K.ECON_AGE_GROUP, group, torch.int64)
    put(K.WEALTH_QUINTILE, quintile[inv], torch.int64)
    put(K.EARNING_POTENTIAL, y, torch.float32)
    put(K.CONSUMPTION_TARGET, target[inv], torch.float32)
    put(K.CASH, cash[inv], torch.float32)
    put(K.CASH_CAP, cap[inv], torch.float32)
    put(K.INSURED, insured[inv], torch.bool)
    put(K.PROVIDER, provider[inv], torch.int64)
    put(K.WORK_LOST, np.zeros(len(hh)), torch.float32)
    put(K.WEALTH, w[inv], torch.float32)
    put(K.INITIAL_WEALTH, w[inv], torch.float32)

    return {
        "earning_level_by_quintile": ybar.tolist(),
        "households_by_quintile": np.bincount(quintile, minlength=N_QUINTILES).tolist(),
        "earners_per_household": float(earners_hh.mean()),
        "buffer_days_by_quintile": [float(np.average(buffer_days[quintile == k], weights=None))
                                    if np.any(quintile == k) else float("nan") for k in range(N_QUINTILES)],
        "provider_shares": (np.bincount(provider, minlength=N_PROVIDERS) / n_hh).tolist(),
        "age_group_shares_5plus": [float((group[~is_child] == g).mean()) for g in (AGE_5_14, AGE_15_64, AGE_65_PLUS)],
    }


def wealth_term(cash, target, scale_days: float):
    """W = clip(cash / (scale_days * target), 0, 1); numpy or torch."""
    if isinstance(cash, torch.Tensor):
        return torch.clamp(cash / (float(scale_days) * torch.clamp(target, min=1e-9)), 0.0, 1.0)
    return np.clip(cash / (float(scale_days) * np.maximum(target, 1e-9)), 0.0, 1.0)


def add_household_cash(agent_state, member_idx, delta: float, scale_days: float) -> None:
    """Adds `delta` GH¢ to the cash of `member_idx`'s household (not below 0),
    and updates every member's CASH and WEALTH (cedi counterpart of
    HouseholdSystem.apply_household_wealth_delta)."""
    nd = agent_state.ndata
    same = nd[K.HOUSEHOLD_ID] == nd[K.HOUSEHOLD_ID][member_idx]
    cash = max(0.0, float(nd[K.CASH][member_idx].item()) + float(delta))
    target = float(nd[K.CONSUMPTION_TARGET][member_idx].item())
    nd[K.CASH][same] = cash
    nd[K.WEALTH][same] = float(wealth_term(np.float64(cash), np.float64(target), scale_days))


def visit_cost(nd, member_idx, eco) -> tuple[float, float]:
    """(out-of-pocket GH¢, share of a working day lost) of a visit to the
    household's usual provider."""
    provider = int(nd[K.PROVIDER][member_idx].item())
    if provider in (PRIMARY, HOSPITAL):
        if bool(nd[K.INSURED][member_idx].item()):
            oop = eco.cost_primary_insured if provider == PRIMARY else eco.cost_hospital_insured
        else:
            oop = eco.cost_facility_uninsured
        return float(oop), float(eco.public_visit_day_fraction)
    if provider == PRIVATE:
        return float(eco.cost_private), float(eco.private_visit_day_fraction)
    return float(eco.cost_drug_seller), float(eco.drug_seller_visit_day_fraction)

# experiments/verification/seir_docking.py
"""
Docking (Axtell et al. 1996): the model's person-to-person transmission
against its equivalent equations (paper 1, analysis 6).

The reduced model: rotavirus alone, person to person alone, N agents in one
well-mixed cell, each its own household; no births, deaths, water, imports or
vaccination; complete immunity after one infection. It is run by the model's
own code - abm.pathogens.rotavirus.Rotavirus, its daily progression and its
two transmission phases a day, called as abm.model.step.sveir_step calls them,
with the calibrated contact cap - so what is checked is the code that runs in
every experiment, in the limit where it should reduce to an SEIR model.

In that limit each susceptible escapes infection in a phase with probability
(1 - p) ** (c I / (N - 1)) (p the per-contact probability, c the contacts per
visit, I the number infectious), the latent period is L days and the
infectious period is geometric with daily recovery probability g. Its
mean-field equations (the deterministic limit for large N) are

    I[t]   = (1 - g) I[t-1] + new[t-L]                 (after the morning's progression)
    S[t+1] = S[t] (1 - p) ** (2 c I[t] / (N - 1)),    new[t] = S[t] - S[t+1],

with the seeds infectious before day 0 (they face a recovery draw on its
morning), and the final size Z of the epidemic satisfies

    -ln(1 - Z / S0) = R0 (Z + (1 - g) I0) / (N - 1),    R0 = 2 c (-ln(1 - p)) / g,

which for small seeds is z = 1 - exp(-R0 z), the final size of the
continuous-time SEIR model with the same R0. The script compares the reduced
model's mean epidemic curve and final size over replicates with both, for
R0 from 1.25 to 5.

    python -m experiments.verification.seir_docking            # 5 R0 x 20 replicates = 100 runs, ~2 min
    python -m experiments.verification.seir_docking --reps 2   # quick

Outputs: experiments/outputs/verification/seir_docking/{summary.csv, curves.csv, docking.json}
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from abm.constants import AgentPropertyKeys as K, Compartment, Phase

OUT_DIR = os.path.join("experiments", "outputs", "verification", "seir_docking")
R0_LEVELS = (1.25, 1.5, 2.0, 3.0, 5.0)
N_AGENTS = 5_000
SEEDS_INFECTIOUS = 50
DAYS = 400
BASE_SEED = 71_001


def reduced_rotavirus(p: float):
    """The model's Rotavirus with the calibrated natural history, per-contact
    probability p (no day-to-day noise) and nothing but person-to-person
    transmission; complete immunity after one infection."""
    from config import SVEIRCONFIG
    from abm.pathogens.rotavirus import Rotavirus
    rota = next(q for q in SVEIRCONFIG.pathogens if q.name == "rota")
    cfg = rota.model_copy(update={
        "infection_prob_mean": float(p), "infection_prob_std": 0.0,
        "immunity_floor": None, "adult_relative_susceptibility": None,
        "infant_susceptibility_at_birth": 1.0, "infant_susceptibility_ramp_months": 0.0,
        "post_infection_refractory_days": 0.0, "asymptomatic_relative_infectiousness": 1.0,
        "importation_rate": 0.0, "vaccination_model": "hazard", "vaccination_rate": 0.0,
        # exp(-50) = 2e-22: a recovered agent is never infected again (an SEIR model)
        "immunity_decay_per_infection": 50.0,
    })
    sp = SVEIRCONFIG.steering_parameters.model_copy(update={"household_exposure_sdlog": 0.0})
    path = Rotavirus(cfg, sp, torch.device("cpu"))
    path.use_catchment_water = True          # no legacy point-source water either
    return path


def contacts() -> float:
    from config import SVEIRCONFIG
    c = SVEIRCONFIG.steering_parameters.max_contacts_per_visit
    if c is None:
        raise ValueError("the docking assumes the contact cap (steering_parameters.max_contacts_per_visit)")
    return float(c)


def natural_history() -> tuple[int, float]:
    from config import SVEIRCONFIG
    rota = next(q for q in SVEIRCONFIG.pathogens if q.name == "rota")
    return int(rota.exposure_period), float(rota.recovery_rate)


def p_for_r0(r0: float, c: float, g: float) -> float:
    """The per-contact probability that gives R0 = 2 c (-ln(1 - p)) / g."""
    return 1.0 - math.exp(-r0 * g / (2.0 * c))


def run_reduced(p: float, n: int, i0: int, days: int, seed: int) -> dict:
    """One run of the reduced model: daily S, E, I, R (I as it stands during
    the day's transmission) and new infections."""
    from abm.state import AgentState
    torch.manual_seed(seed)
    path = reduced_rotavirus(p)
    st = AgentState(n)
    nd = st.ndata
    nd[K.status("rota")] = torch.full((n,), int(Compartment.SUSCEPTIBLE), dtype=torch.int)
    nd[K.status("rota")][:i0] = int(Compartment.INFECTIOUS)
    nd[K.exposure_time("rota")] = torch.zeros(n, dtype=torch.int)
    nd[K.num_infections("rota")] = torch.zeros(n, dtype=torch.int)
    nd[K.num_infections("rota")][:i0] = 1
    nd[K.symptomatic_infection("rota")] = torch.zeros(n, dtype=torch.bool)
    nd[K.IS_CHILD] = torch.ones(n, dtype=torch.bool)
    nd[K.AGE] = torch.full((n,), 24.0)
    nd[K.HOUSEHOLD_ID] = torch.arange(n, dtype=torch.long)
    cell = torch.zeros(n, dtype=torch.long)
    status = nd[K.status("rota")]
    rec = {k: np.zeros(days, dtype=np.int64) for k in ("S", "E", "I", "R", "new")}
    for t in range(days):
        path.reset_incidence()
        path.step_progression(st)
        rec["I"][t] = int((status == Compartment.INFECTIOUS).sum())
        rec["S"][t] = int((status == Compartment.SUSCEPTIBLE).sum())
        path.step_transmission(st, cell, 1, None, Phase.DAY)
        path.step_transmission(st, cell, 1, None, Phase.NIGHT)
        rec["new"][t] = path.new_cases_this_step
        rec["E"][t] = int((status == Compartment.EXPOSED).sum())
        rec["R"][t] = int((status == Compartment.RECOVERED).sum())
    return rec


def mean_field(p: float, c: float, latent: int, g: float, n: int, i0: int, days: int) -> dict:
    """The deterministic equations of the reduced model (module docstring)."""
    S = float(n - i0)
    I_prev = float(i0)
    new = np.zeros(days)
    I = np.zeros(days)
    Ss = np.zeros(days)
    q = 1.0 - p
    for t in range(days):
        arrivals = new[t - latent] if t - latent >= 0 else 0.0
        It = (1.0 - g) * I_prev + arrivals
        Ss[t], I[t] = S, It
        S_next = S * q ** (2.0 * c * It / (n - 1))
        new[t] = S - S_next
        S, I_prev = S_next, It
    return {"S": Ss, "I": I, "new": new}


def final_size(r0: float, g: float, n: int, i0: int) -> float:
    """Z / (N - I0) from -ln(1 - Z/S0) = R0 (Z + (1 - g) I0) / (N - 1) (bisection)."""
    s0 = n - i0
    f = lambda z: -math.log(1.0 - z) - r0 * (z * s0 + (1.0 - g) * i0) / (n - 1)
    lo, hi = 1e-12, 1.0 - 1e-15
    if f(lo) > 0:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def seir_ode(r0: float, latent: int, g: float, n: int, i0: int, days: int, stages: int = 8) -> dict:
    """The continuous-time SEIR model with the same R0 (beta = R0 g), a latent
    period of mean L in `stages` Erlang stages and an infectious period of mean
    1 / g, for the figure (its final size is the same relation)."""
    from scipy.integrate import solve_ivp
    beta, sigma = r0 * g, stages / latent

    def rhs(_t, y):
        S, I = y[0], y[-1]
        E = y[1:-1]
        inf = beta * S * I / (n - 1)
        dE = np.empty_like(E)
        dE[0] = inf - sigma * E[0]
        dE[1:] = sigma * (E[:-1] - E[1:])
        return np.concatenate([[-inf], dE, [sigma * E[-1] - g * I]])

    y0 = np.concatenate([[n - i0], np.zeros(stages), [i0]])
    sol = solve_ivp(rhs, (0, days), y0, t_eval=np.arange(days + 1), rtol=1e-8, atol=1e-8)
    S = sol.y[0]
    return {"S": S[:-1], "I": sol.y[-1][:-1], "new": S[:-1] - S[1:]}


def dock(r0_levels=R0_LEVELS, reps: int = 20, n: int = N_AGENTS, i0: int = SEEDS_INFECTIOUS,
         days: int = DAYS, base_seed: int = BASE_SEED) -> tuple[list[dict], list[dict]]:
    c = contacts()
    latent, g = natural_history()
    summary, curves = [], []
    for r0 in r0_levels:
        p = p_for_r0(r0, c, g)
        mf = mean_field(p, c, latent, g, n, i0, days)
        ode = seir_ode(r0, latent, g, n, i0, days)
        runs = [run_reduced(p, n, i0, days, base_seed + k) for k in range(reps)]
        attack = np.array([r["new"].sum() / (n - i0) for r in runs])
        new = np.array([r["new"] for r in runs], dtype=float)
        mean_new = new.mean(axis=0)
        z_eq = final_size(r0, g, n, i0)
        z_mf = float(mf["new"].sum() / (n - i0))
        peak = float(mf["new"].max())
        summary.append({
            "R0": r0, "p": p, "c": c, "latent_days": latent, "g": g, "N": n, "I0": i0, "reps": reps,
            "attack_mean": float(attack.mean()), "attack_sd": float(attack.std(ddof=1)) if reps > 1 else float("nan"),
            "attack_final_size_eq": z_eq, "attack_mean_field": z_mf,
            "attack_ode": float(ode["new"].sum() / (n - i0)),
            "peak_day_model": float(np.mean([np.argmax(r["new"]) for r in runs])),
            "peak_day_mean_field": float(np.argmax(mf["new"])), "peak_day_ode": float(np.argmax(ode["new"])),
            # the peak of the mean curve (a mean of each run's peak is inflated by the runs' noise)
            "peak_new_model": float(mean_new.max()), "peak_new_mean_field": peak,
            # the mean curve's distance from the equations, relative to their peak
            "curve_rmse_rel_mean_field": float(np.sqrt(np.mean((mean_new - mf["new"]) ** 2)) / peak),
            "curve_max_abs_rel_mean_field": float(np.max(np.abs(mean_new - mf["new"])) / peak),
        })
        for t in range(days):
            curves.append({"R0": r0, "day": t, "new_model_mean": float(mean_new[t]),
                           "new_model_sd": float(new[:, t].std(ddof=1)) if reps > 1 else float("nan"),
                           "new_mean_field": float(mf["new"][t]), "new_ode": float(ode["new"][t]),
                           "I_model_mean": float(np.mean([r["I"][t] for r in runs])),
                           "I_mean_field": float(mf["I"][t])})
    return summary, curves


def main(argv=None) -> None:
    import pandas as pd
    ap = argparse.ArgumentParser(description="docking of person-to-person transmission against its equations")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--agents", type=int, default=N_AGENTS)
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    summary, curves = dock(reps=args.reps, n=args.agents, days=args.days)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(args.out, "curves.csv"), index=False)
    with open(os.path.join(args.out, "docking.json"), "w") as f:
        json.dump({"summary": summary}, f, indent=1)
    cols = ["R0", "attack_mean", "attack_sd", "attack_final_size_eq", "attack_mean_field", "attack_ode",
            "peak_day_model", "peak_day_mean_field", "peak_day_ode", "curve_rmse_rel_mean_field"]
    print(pd.DataFrame(summary)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

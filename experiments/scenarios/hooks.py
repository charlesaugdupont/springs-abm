# experiments/scenarios/hooks.py
"""
Day-by-day interventions for the v5 experiments,
applied through the orchestrator's step_callback, BEFORE each day's step.
They change household state or parameters during a run without touching the
calibrated model; a scenario without "hook.*" keys runs exactly as the
default model.

Keys read from the scenario (combo), all optional:

  hook.transfer_quintiles   "1" or "1,2": wealth quintiles (1 = lowest) that
                            receive LEAP-style cash transfers
  hook.transfer_multiple    multiple of LEAP's 2026 grant (default 1.0)
  hook.transfer_start_day   first payment day (default 0); then every 61 days
                            (LEAP pays every two months, six times a year)
  hook.transfer_free_nhis   recipients hold NHIS cover from the first payment
                            (default True, as in LEAP)

  hook.stockout_days        experiment 1B: actual efficacy of care falls to
                            hook.stockout_efficacy (default 0.2) for this many
                            days from hook.stockout_start (default 1,095)
  hook.ramp_days            experiment 1B: actual efficacy falls linearly to
                            hook.ramp_low (default 0.1) over this many days
                            from hook.ramp_start (default 1,095), then rises
                            back over as many days
  hook.campaign_day         experiment 1B: an information campaign resets the
                            beliefs of a share hook.campaign_share (default
                            0.5) of households to the true probabilities
  hook.free_care_start      experiment 1B: care is free for
                            hook.free_care_days (default 91) days
  hook.booster_age_days     experiment 3B: a booster dose at this age (days),
                            for children who had all three doses, with
                            coverage hook.booster_coverage (default 0.9) and
                            take hook.booster_take (default 0.95): it renews
                            protection for a fresh duration (an upper bound)
  hook.wash_quintiles       experiment 3B: "1,2": households in these wealth
                            quintiles get improved sanitation and piped water
                            from day 0
  hook.take_by_wash         experiment 3B, sensitivity D4: vaccine take
                            depends on household water and sanitation as in
                            SHINE (x TAKE_WASH_RATIO with improved WASH), the
                            population mean kept at the calibrated take
  hook.regime               weather-shock regime of experiment 2 (REGIMES:
                            "R0", "C-lo", "C-hi", "C-rare", "F-fast",
                            "F-slow", "D-fast", "D-slow", "C+F", "C+D")
  hook.strategy             risk-management strategy of experiment 2
                            (STRATEGIES: "none", "storage", "exchange",
                            "infrastructure", "adjustment", "mobility")
  hook.env                  experiment 2: True turns on the environmental
                            pathways of illness at their sized values
                            (ENV_SIZED); hook.drought_hygiene,
                            hook.first_rains and hook.flood_contamination set
                            one explicitly (see env_settings)

LEAP's 2026 bi-monthly grant per household (households with a vulnerable
member, which LEAP households are by eligibility): GH¢392 for 1-2 members,
474 for 3-4, 630 for 5 or more (https://leap.mogcsp.gov.gh/faq.php).

Experiment 2 (shocks). Day 0 is taken
as 1 January; the shocks start in year 3 (day 730), after the two warm-up years
(until 2026-11-10: year 4, day 1,095, after one measured year). Channels, all
deterministic (no random draws, so runs keep their common random numbers until
behaviour diverges):
  income          earners' earning potential times a town-wide factor, and a
                  flood-zone factor for households in the flood zone
  prices          every household's consumption target (GH¢) and the poverty
                  lines times a price factor (the same basket costs more)
  water           the daily probability of a contamination pulse
                  (steering_parameters.shock_daily_prob) times a factor
  sanitation      during a flood, latrines in the flood zone release as open
                  defecation
  access          during a flood, facility visits take twice as long
  illness         with hook.env (step 1b of the revision): drought hygiene,
                  the first rains after a drought, and the flood's
                  contamination resized (see ENV_SIZED below)
The flood zone is every household within FLOOD_ZONE_M of a river or canal
cell (from the water network's runoff kernel: its row sum is
exp(-d / runoff_length) for the nearest water cell).
"""
from __future__ import annotations

import numpy as np
import torch

from abm.constants import AgentPropertyKeys as K

LEAP_2026_BIMONTHLY = ((2, 392.0), (4, 474.0), (10**9, 630.0))
LEAP_PAYMENT_INTERVAL_DAYS = 61


def leap_grant(size: np.ndarray) -> np.ndarray:
    """LEAP's 2026 bi-monthly grant (GH¢) for households of the given sizes."""
    out = np.empty(len(size))
    lo = 0
    for upper, amount in LEAP_2026_BIMONTHLY:
        out[(size > lo) & (size <= upper)] = amount
        lo = upper
    return out


def _quintiles(spec) -> list[int]:
    if spec is None or spec == "":
        return []
    return [int(x) - 1 for x in str(spec).split(",")]


def apply_hooks(model, day: int, combo: dict) -> None:
    """The step_callback for the v5 experiments (picklable, module-level)."""
    if "hook.transfer_quintiles" in combo:
        _transfers(model, day, combo)
    if "hook.stockout_days" in combo or "hook.ramp_days" in combo:
        _efficacy(model, day, combo)
    if "hook.campaign_day" in combo and day == int(combo["hook.campaign_day"]):
        _campaign(model, combo)
    if "hook.free_care_start" in combo:
        _free_care(model, day, combo)
    if combo.get("hook.take_by_wash"):              # before any upgrade: its mean take is today's
        _take_by_wash(model, day)
    if "hook.wash_quintiles" in combo and day == 0:
        _wash_upgrade(model, combo["hook.wash_quintiles"])
    if "hook.booster_age_days" in combo:
        _booster(model, day, combo)
    if "hook.regime" in combo or "hook.strategy" in combo:
        # R0 without a strategy rescales nothing (every factor is 1), but tracks the flood zone
        _shocks_and_strategies(model, day, combo.get("hook.regime", "R0"), combo.get("hook.strategy", "none"),
                               env_settings(combo))


def _transfers(model, day: int, combo: dict) -> None:
    start = int(combo.get("hook.transfer_start_day", 0))
    if day < start or (day - start) % LEAP_PAYMENT_INTERVAL_DAYS != 0:
        return
    quintiles = _quintiles(combo.get("hook.transfer_quintiles"))
    if not quintiles:
        return
    nd = model.graph.ndata
    if K.CASH not in nd:
        raise ValueError("hook.transfer_*: needs the cedi economy (economy.model = 'cedi')")
    hh = nd[K.HOUSEHOLD_ID].cpu().numpy()
    _, inv, size = np.unique(hh, return_inverse=True, return_counts=True)
    grant_hh = leap_grant(size) * float(combo.get("hook.transfer_multiple", 1.0))
    q = nd[K.WEALTH_QUINTILE].cpu().numpy()
    target = np.isin(q, quintiles)
    if not target.any():
        return
    add = torch.as_tensor(np.where(target, grant_hh[inv], 0.0), dtype=nd[K.CASH].dtype, device=nd[K.CASH].device)
    nd[K.CASH] = nd[K.CASH] + add          # every member holds the household's cash
    if bool(combo.get("hook.transfer_free_nhis", True)) and day == start:
        t = torch.as_tensor(target, device=nd[K.INSURED].device)
        nd[K.INSURED] = nd[K.INSURED] | t
        from abm.systems.economics import EconomicSystem
        for s in model.systems:            # its cached hospital prices follow NHIS status
            if isinstance(s, EconomicSystem):
                s._facility_cost_cache = None


# ---------------------------------------------------------------------------
# Experiment 2: weather shocks and risk-management strategies
# ---------------------------------------------------------------------------

DAYS_PER_YEAR = 365
#: the shocks (and experiment 1B's stock-outs) start after the two-year warm-up and are followed for
#: one year (the user's decision R1 of 2026-11-10; before it, day 1,095, followed for two years)
SHOCK_START_DAY = 2 * DAYS_PER_YEAR            # 1 January, year 3
FLOOD_START_DAY = SHOCK_START_DAY + 273        # 1 October (the 2023 spillage came in September-October)
FLOOD_DAYS = 28
DROUGHT_START_DAY = SHOCK_START_DAY + 59       # 1 March: the major rainy season fails
DROUGHT_DAYS = 184                             # to the end of August
SLOW_RECOVERY_DAYS = 365
FLOOD_ZONE_M = 150.0
#: rainy-season windows (day of year): the major season's peak (mid-May to
#: mid-July) and the minor season's (mid-September to mid-November)
RAINY_WINDOWS = ((135, 200), (258, 320))
#: day of year of the seasonal earnings peak (mid-December, after the harvests;
#: the trough, the lean season, falls in mid-June)
EARNINGS_PEAK_DOY = 349

#: regime -> (seasonal cycle (earnings amplitude, contamination factor) or None,
#:            years in which the cycle runs (None = every year from year 4),
#:            flood recovery ("fast", "slow" or None), drought recovery (same))
REGIMES = {
    "R0": (None, None, None, None),
    "C-lo": ((0.15, 2.0), None, None, None),
    "C-hi": ((0.40, 5.0), None, None, None),
    "C-rare": ((0.40, 5.0), (SHOCK_START_DAY // DAYS_PER_YEAR,), None, None),  # one bad year in three: the first
    "F-fast": (None, None, "fast", None),
    "F-slow": (None, None, "slow", None),
    "D-fast": (None, None, None, "fast"),
    "D-slow": (None, None, None, "slow"),
    "C+F": ((0.40, 5.0), None, "slow", None),
    "C+D": ((0.40, 5.0), None, None, "slow"),
}
STRATEGIES = ("none", "storage", "exchange", "infrastructure", "adjustment", "mobility")
#: storage (village savings and loan associations): households save this share
#: of their income (consumption target / (1 + rate)) and hold a buffer this
#: much larger
STORAGE_SAVINGS_RATE = 0.05
STORAGE_BUFFER_FACTOR = 1.5
#: economic adjustment: consumption cut while the household's earnings are
#: below normal or prices above
ADJUSTMENT_CUT = 0.20


# Step 1b of the revision (the user's decision R3b of 2026-11-10):
# environmental pathways of illness, in experiment 2 only, as scenario mechanisms sized to
# published effects (experiments.scenarios.env_sizing), not predicted by the calibrated model. Each
# multiplies routes' daily infection probabilities (Pathogen.route_multipliers) while it lasts:
#   drought hygiene  while the drought lasts, the person-to-person routes and Campylobacter's
#                    food-borne exposure x hook.drought_hygiene (less water for washing and food
#                    hygiene; a severe six-month drought raises child diarrhoea by 8%, a mild one
#                    by 5%: Wang et al. 2022)
#   first rains      for FIRST_RAINS_DAYS after the drought, Cryptosporidium's water route and
#                    Campylobacter's environmental (zoonotic) exposure x hook.first_rains (extreme
#                    rain after a dry period, IRR 1.26: Kraay et al. 2020); none for rotavirus,
#                    which is less common in rainy seasons (IRR 0.46)
#   flood            the flood's contamination pulse: hook.flood_contamination times the daily
#                    probability of a pulse (10 before the resizing), sized to the floods' IRR
#                    among studies adjusting for temperature, 1.23 (1.04-1.47; Kraay et al. 2020),
#                    in the flood zone
DROUGHT_HYGIENE_ROUTES = {"rota": ("h2h",), "crypto": ("household", "community"),
                          "campy": ("fecal_oral", "food_borne")}
FIRST_RAINS_ROUTES = {"crypto": ("water",), "campy": ("zoonotic",)}
FIRST_RAINS_DAYS = 28
FLOOD_CONTAMINATION = 10.0
#: the sized values, taken by hook.env = True (experiments.scenarios.env_sizing); None until sized
ENV_SIZED = {"drought_hygiene": 1.036, "first_rains": 1.9, "flood_contamination": 7.6}
#: drought hygiene sized to a mild six-month drought (x1.05; Wang et al. 2022), experiment 2's sensitivity
ENV_MILD_DROUGHT_HYGIENE: float | None = 1.022


def env_settings(combo: dict) -> dict:
    """A scenario's environmental pathways: multipliers (1 = off) and the flood's
    contamination factor (FLOOD_CONTAMINATION as before the resizing).
    hook.env = True takes the sized values; hook.<name> sets one explicitly."""
    out = {"drought_hygiene": 1.0, "first_rains": 1.0, "flood_contamination": FLOOD_CONTAMINATION}
    if combo.get("hook.env"):
        for k, v in ENV_SIZED.items():
            if v is None:
                raise ValueError("hook.env: the environmental pathways have not been sized yet")
            out[k] = float(v)
    for k in out:
        if f"hook.{k}" in combo:
            out[k] = float(combo[f"hook.{k}"])
    return out


def env_multipliers(regime: str, day: int, env: dict) -> dict:
    """{(pathogen, route): multiplier} of the environmental pathways on `day`."""
    drought = REGIMES[regime][3]
    want: dict = {}
    if drought is None or day < SHOCK_START_DAY:
        return want
    end = DROUGHT_START_DAY + DROUGHT_DAYS
    if DROUGHT_START_DAY <= day < end and env["drought_hygiene"] != 1.0:
        for p, routes in DROUGHT_HYGIENE_ROUTES.items():
            for r in routes:
                want[(p, r)] = env["drought_hygiene"]
    if end <= day < end + FIRST_RAINS_DAYS and env["first_rains"] != 1.0:
        for p, routes in FIRST_RAINS_ROUTES.items():
            for r in routes:
                want[(p, r)] = want.get((p, r), 1.0) * env["first_rains"]
    return want


def _set_route_multipliers(model, want: dict) -> None:
    for p in model.pathogens:
        for r in list(p.route_multipliers):
            if (p.name, r) not in want:
                del p.route_multipliers[r]
        for (name, r), m in want.items():
            if name == p.name:
                p.route_multipliers[r] = m


def regime_factors(regime: str, day: int, flood_contamination: float = FLOOD_CONTAMINATION) -> dict:
    """The regime's factors on `day`: town-wide and flood-zone earnings, prices,
    contamination, and whether a flood is under way."""
    cycle, cycle_years, flood, drought = REGIMES[regime]
    f = {"income_town": 1.0, "income_zone": 1.0, "price": 1.0, "contamination": 1.0, "flood": False}
    if day < SHOCK_START_DAY:
        return f
    if cycle is not None and (cycle_years is None or day // DAYS_PER_YEAR in cycle_years):
        amp, contamination = cycle
        doy = day % DAYS_PER_YEAR
        f["income_town"] *= 1.0 + amp * float(np.cos(2.0 * np.pi * (doy - EARNINGS_PEAK_DOY) / DAYS_PER_YEAR))
        if any(lo <= doy < hi for lo, hi in RAINY_WINDOWS):
            f["contamination"] *= contamination
    if flood is not None:
        end = FLOOD_START_DAY + FLOOD_DAYS
        if FLOOD_START_DAY <= day < end:
            f["contamination"] *= flood_contamination
            f["flood"] = True
            f["income_zone"] *= 0.5               # lost crops, nets, stock and working days
        elif flood == "slow" and end <= day < end + SLOW_RECOVERY_DAYS:
            f["income_zone"] *= 0.5 + 0.5 * (day - end) / SLOW_RECOVERY_DAYS
    if drought is not None:
        end = DROUGHT_START_DAY + DROUGHT_DAYS
        if DROUGHT_START_DAY <= day < end:
            f["income_town"] *= 0.6
            f["price"] *= 1.2
        elif drought == "slow" and end <= day < end + SLOW_RECOVERY_DAYS:
            r = (day - end) / SLOW_RECOVERY_DAYS
            f["income_town"] *= 0.6 + 0.4 * r
            f["price"] *= 1.2 - 0.2 * r
    return f


def flood_zone(model, radius_m: float = FLOOD_ZONE_M) -> np.ndarray:
    """Households (water-network order) within radius_m of a river or canal cell."""
    net = model.grid_environment.water_network
    L = float(model.config.water.runoff_length_m)
    if radius_m > 3.0 * L:
        raise ValueError("the runoff kernel only reaches 3 x runoff_length_m")
    rs = net.surface_kernel.sum(dim=1).cpu().numpy()
    with np.errstate(divide="ignore"):
        d = np.where(rs > 0, -L * np.log(np.maximum(rs, 1e-300)), np.inf)
    return d <= radius_m


def _state(model) -> dict:
    """What the hooks restore or scale from, captured on their first call."""
    st = getattr(model, "_exp_state", None)
    if st is None:
        nd = model.graph.ndata
        net = model.grid_environment.water_network
        zone_hh = flood_zone(model)
        agent_hh = net.agent_hh.cpu().numpy()
        st = {
            "potential0": nd[K.EARNING_POTENTIAL].clone(),
            "target0": nd[K.CONSUMPTION_TARGET].clone(),
            "cap0": nd[K.CASH_CAP].clone(),
            "shock_prob0": float(model.config.steering_parameters.shock_daily_prob),
            "poverty0": (float(model.config.economy.poverty_line_daily),
                         float(model.config.economy.extreme_poverty_line_daily)),
            "visit0": (float(model.config.economy.public_visit_day_fraction),
                       float(model.config.economy.private_visit_day_fraction)),
            "zone_hh": zone_hh,
            "zone_agent": torch.as_tensor(zone_hh[agent_hh]),
            "agent_hh": agent_hh,
            "sanitation0": net.hh_sanitation.clone(),
            "source0": net.hh_source.clone(),
            "type0": net.hh_type.clone(),
            "flooded": False, "switched": False,
            "zone_zero_cash": [],
        }
        model._exp_state = st
    return st


def _shocks_and_strategies(model, day: int, regime: str, strategy: str, env: dict | None = None) -> None:
    from abm.environment.water_network import HH_GROUND, HH_PIPED, HH_SURFACE, SAN_IMPROVED_ONSITE, SAN_OPEN
    if regime not in REGIMES or strategy not in STRATEGIES:
        raise ValueError(f"unknown regime {regime!r} or strategy {strategy!r}")
    nd = model.graph.ndata
    st = _state(model)
    net = model.grid_environment.water_network
    eco, steer = model.config.economy, model.config.steering_parameters
    zone = st["zone_agent"].to(nd[K.CASH].device)

    # outcome tracked here: the flood zone's households without cash (yesterday's close)
    first = st.setdefault("zone_first_agent", _first_agents(st["agent_hh"], st["zone_hh"]))
    st["zone_zero_cash"].append(float((nd[K.CASH][first] <= 0.0).float().mean()) if len(first) else float("nan"))

    if day == 0:
        if strategy == "storage":
            nd[K.CASH_CAP] = st["cap0"] * STORAGE_BUFFER_FACTOR
        if strategy == "infrastructure":            # flood-proof improved latrines and piped water
            san = st["sanitation0"].clone()
            zone_idx = torch.as_tensor(np.flatnonzero(st["zone_hh"]))
            san[zone_idx] = SAN_IMPROVED_ONSITE
            net.set_household_sanitation(san, nd[K.IS_CHILD])
            net.hh_source[zone_idx] = net.piped_index
            net.hh_type[zone_idx] = HH_PIPED
            st["sanitation0"], st["source0"], st["type0"] = san, net.hh_source.clone(), net.hh_type.clone()
    if strategy == "exchange":
        _transfers(model, day, {"hook.transfer_quintiles": "1,2", "hook.transfer_multiple": 1.0})

    env = env or env_settings({})
    f = regime_factors(regime, day, env["flood_contamination"])
    _set_route_multipliers(model, env_multipliers(regime, day, env))
    income = torch.full_like(st["potential0"], f["income_town"])
    income[zone] = income[zone] * f["income_zone"]
    nd[K.EARNING_POTENTIAL] = st["potential0"] * income
    target = st["target0"] * f["price"]
    if strategy == "storage":
        target = target / (1.0 + STORAGE_SAVINGS_RATE)
    if strategy == "adjustment":
        stressed = (income < 1.0) | torch.tensor(f["price"] > 1.0)
        target = torch.where(stressed, target * (1.0 - ADJUSTMENT_CUT), target)
    nd[K.CONSUMPTION_TARGET] = target
    eco.poverty_line_daily = st["poverty0"][0] * f["price"]
    eco.extreme_poverty_line_daily = st["poverty0"][1] * f["price"]
    steer.shock_daily_prob = min(1.0, st["shock_prob0"] * f["contamination"])

    # flood: latrines in the zone overflow (unless upgraded), facility visits take longer
    if f["flood"] != st["flooded"]:
        if f["flood"]:
            san = st["sanitation0"].clone()
            if strategy != "infrastructure":
                san[torch.as_tensor(np.flatnonzero(st["zone_hh"]))] = SAN_OPEN
            net.set_household_sanitation(san, nd[K.IS_CHILD])
            eco.public_visit_day_fraction = min(1.0, 2.0 * st["visit0"][0])
            eco.private_visit_day_fraction = min(1.0, 2.0 * st["visit0"][1])
        else:
            net.set_household_sanitation(st["sanitation0"], nd[K.IS_CHILD])
            eco.public_visit_day_fraction, eco.private_visit_day_fraction = st["visit0"]
        st["flooded"] = f["flood"]

    # mobility: surface and groundwater users drink packaged water while contamination is high
    if strategy == "mobility":
        switch = f["contamination"] > 1.0
        if switch != st["switched"]:
            if switch:
                users = (st["type0"] == HH_SURFACE) | (st["type0"] == HH_GROUND)
                net.hh_source[users] = net.packaged_index
            else:
                net.hh_source.copy_(st["source0"])
            st["switched"] = switch


def _first_agents(agent_hh: np.ndarray, zone_hh: np.ndarray) -> torch.Tensor:
    """One agent of every flood-zone household."""
    hh, first = np.unique(agent_hh, return_index=True)
    return torch.as_tensor(first[zone_hh[hh]])


def zone_series(model) -> dict[str, list[float]]:
    """Weekly series for the flood zone (experiment 2 runs only): illness
    episodes among its children, and the share of its households without cash."""
    st = getattr(model, "_exp_state", None)
    if st is None:
        return {}
    from experiments.scenarios.common import _weekly
    days = len(st["zone_zero_cash"])
    zone_agent = st["zone_agent"].cpu().numpy()
    onsets = np.zeros(days)
    for e in model.get_child_episode_log():
        t = int(e["timestep"])
        if 0 <= t < days and zone_agent[int(e["agent_idx"])]:
            onsets[t] += 1
    zero = np.asarray(st["zone_zero_cash"][1:] + [np.nan])   # the value read on day d+1 is day d's close
    return {"wk_fz_episodes": _weekly(onsets), "wk_fz_zero_cash_share": _weekly(zero, "mean"),
            "fz_households": float(st["zone_hh"].sum()), "fz_persons": float(zone_agent.sum())}


# ---------------------------------------------------------------------------
# Experiment 1B: efficacy of care over time, and what restores trust
# ---------------------------------------------------------------------------

PRICE_FIELDS = ("cost_primary_insured", "cost_hospital_insured", "cost_facility_uninsured", "cost_private",
                "cost_drug_seller")
CAMPAIGN_RNG_OFFSET = 7907


def efficacy_on(day: int, combo: dict, base: float) -> float:
    """The actual probability that care works on `day` under a stock-out or a ramp."""
    if "hook.stockout_days" in combo:
        start = int(combo.get("hook.stockout_start", SHOCK_START_DAY))
        if start <= day < start + int(combo["hook.stockout_days"]):
            return float(combo.get("hook.stockout_efficacy", 0.2))
    if "hook.ramp_days" in combo:
        start, n = int(combo.get("hook.ramp_start", SHOCK_START_DAY)), int(combo["hook.ramp_days"])
        low = float(combo.get("hook.ramp_low", 0.1))
        t = day - start
        if 0 <= t < n:
            return base + (low - base) * t / n
        if n <= t < 2 * n:
            return low + (base - low) * (t - n) / n
    return base


def _efficacy(model, day: int, combo: dict) -> None:
    steer = model.config.steering_parameters
    if not hasattr(model, "_exp_efficacy0"):
        model._exp_efficacy0 = float(steer.treatment_success_prob)
    steer.treatment_success_prob = efficacy_on(day, combo, model._exp_efficacy0)


def _care_system(model):
    from abm.systems.care_seeking import CareSeekingSystem
    return next(s for s in model.systems if isinstance(s, CareSeekingSystem))


def _campaign(model, combo: dict) -> None:
    """Resets the beliefs of a random share of households (own generator, so
    the model's random stream is untouched) to the true probabilities: those
    of the calibrated model (care works with probability
    efficacy-before-any-stock-out, waiting worsens with natural_worsening_prob)."""
    cs = _care_system(model)
    b = cs._beliefs
    if b is None:
        raise ValueError("hook.campaign_day needs steering_parameters.belief_learning")
    steer = model.config.steering_parameters
    rng = np.random.default_rng(int(model.config.seed) + CAMPAIGN_RNG_OFFSET)
    chosen = torch.as_tensor(rng.random(len(b["success"])) < float(combo.get("hook.campaign_share", 0.5)))
    b["success"][chosen] = getattr(model, "_exp_efficacy0", float(steer.treatment_success_prob))
    b["worsen"][chosen] = float(steer.natural_worsening_prob)


def _free_care(model, day: int, combo: dict) -> None:
    eco = model.config.economy
    if not hasattr(model, "_exp_prices0"):
        model._exp_prices0 = {f: float(getattr(eco, f)) for f in PRICE_FIELDS}
    start = int(combo["hook.free_care_start"])
    on = start <= day < start + int(combo.get("hook.free_care_days", 91))
    if on != getattr(model, "_exp_free_care_on", False):
        for f, v in model._exp_prices0.items():
            setattr(eco, f, 0.0 if on else v)
        from abm.systems.economics import EconomicSystem
        for s in model.systems:
            if isinstance(s, EconomicSystem):
                s._facility_cost_cache = None
        model._exp_free_care_on = on


# ---------------------------------------------------------------------------
# Experiment 3B: improving the rotavirus vaccine
# ---------------------------------------------------------------------------

#: SHINE (Church et al. 2019): seroconversion 30.3% with improved household WASH, 19.6% without
TAKE_WASH_RATIO = 30.3 / 19.6
BOOSTER_RNG_OFFSET = 7919 + 1
TAKE_RNG_OFFSET = 7919 + 2


def _rota(model):
    return next(p for p in model.pathogens if p.name == "rota")


def _household_quintile(model) -> np.ndarray:
    """Each water-network household's wealth quintile (0 = lowest)."""
    net = model.grid_environment.water_network
    agent_hh = net.agent_hh.cpu().numpy()
    hh, first = np.unique(agent_hh, return_index=True)
    q = np.empty(len(hh), dtype=np.int64)
    q[hh] = model.graph.ndata[K.WEALTH_QUINTILE].cpu().numpy()[first]
    return q


def _wash_upgrade(model, quintiles) -> None:
    """Improved (on-site) sanitation for households without it, and piped
    water, for every household in the given wealth quintiles."""
    from abm.environment.water_network import HH_PIPED, SAN_IMPROVED_ONSITE, SAN_SEWER
    net = model.grid_environment.water_network
    target = torch.as_tensor(np.isin(_household_quintile(model), _quintiles(quintiles)))
    san = net.hh_sanitation.clone()
    san[target & (san != SAN_SEWER)] = SAN_IMPROVED_ONSITE
    net.set_household_sanitation(san, model.graph.ndata[K.IS_CHILD])
    net.hh_source[target] = net.piped_index
    net.hh_type[target] = HH_PIPED


def improved_wash(model) -> torch.Tensor:
    """Per agent: the household has improved sanitation (on-site or sewer) and
    piped or packaged drinking water."""
    from abm.environment.water_network import HH_PACKAGED, HH_PIPED, SAN_IMPROVED_ONSITE, SAN_SEWER
    net = model.grid_environment.water_network
    ok = ((net.hh_sanitation == SAN_IMPROVED_ONSITE) | (net.hh_sanitation == SAN_SEWER)) & \
         ((net.hh_type == HH_PIPED) | (net.hh_type == HH_PACKAGED))
    return ok[net.agent_hh]


def _take_by_wash(model, day: int) -> None:
    """Sensitivity D4. From day 1 (the schedules are drawn at the first step),
    re-draws, with the hook's own generator, whether each dose not yet given
    takes: P = take x TAKE_WASH_RATIO / m with improved household WASH, take /
    m without, m = 1 + (TAKE_WASH_RATIO - 1) x the share of children with
    improved WASH at day 0, so the mean take stays the calibrated one before
    any upgrade. Slots renewed as newborns are re-drawn the day after."""
    rota = _rota(model)
    nd = model.graph.ndata
    st = model.__dict__.setdefault("_exp_take", {})
    if "m" not in st:                                  # day 0, before any WASH upgrade
        is_child = nd[K.IS_CHILD].bool()
        st["m"] = 1.0 + (TAKE_WASH_RATIO - 1.0) * float(improved_wash(model)[is_child].double().mean())
        st["rng"] = np.random.default_rng(int(model.config.seed) + TAKE_RNG_OFFSET)
    v = getattr(rota, "_vax", None)
    if v is None:
        return
    if "birth_day" not in st:
        st["birth_day"] = nd[K.BIRTH_DAY].clone()
        redraw = torch.ones(nd[K.AGE].numel(), dtype=torch.bool)
    else:
        redraw = nd[K.BIRTH_DAY] != st["birth_day"]
        st["birth_day"] = nd[K.BIRTH_DAY].clone()
    if not bool(redraw.any()):
        return
    base = torch.as_tensor(rota.config.vaccine_take_by_dose, dtype=torch.float64)
    factor = torch.where(improved_wash(model), TAKE_WASH_RATIO, 1.0).double() / st["m"]
    p = torch.clamp(factor[:, None] * base[None, :], max=1.0)
    u = torch.as_tensor(st["rng"].random(p.shape))
    k = p.shape[1]
    not_given = torch.arange(k)[None, :] >= v["given"][:, None]
    sel = redraw[:, None] & not_given
    v["takes"][sel] = (u < p)[sel]


def _booster(model, day: int, combo: dict) -> None:
    """A booster dose at hook.booster_age_days for children who had every
    dose: with probability hook.booster_coverage it is given, and with
    hook.booster_take it renews protection for a fresh duration from the
    schedule's distribution (the child enters V from S or R), and protection
    against illness too when its duration is set (the same draw). Own
    generator."""
    from abm.constants import Compartment
    from abm.systems.demography import DAYS_PER_MONTH
    rota = _rota(model)
    v = getattr(rota, "_vax", None)
    if v is None:
        return
    nd = model.graph.ndata
    st = model.__dict__.setdefault("_exp_booster", {})
    n = nd[K.AGE].numel()
    if "done" not in st:
        st.update(done=torch.zeros(n, dtype=torch.bool), birth_day=nd[K.BIRTH_DAY].clone(),
                  rng=np.random.default_rng(int(model.config.seed) + BOOSTER_RNG_OFFSET))
    reborn = nd[K.BIRTH_DAY] != st["birth_day"]
    st["done"][reborn] = False
    st["birth_day"] = nd[K.BIRTH_DAY].clone()
    age_days = nd[K.AGE].double() * DAYS_PER_MONTH
    k = v["ages"].shape[1]
    due = nd[K.IS_CHILD].bool() & ~st["done"] & (v["given"] >= k) & (age_days >= float(combo["hook.booster_age_days"]))
    if not bool(due.any()):
        return
    st["done"][due] = True
    idx = due.nonzero(as_tuple=True)[0]
    rng = st["rng"]
    given = torch.as_tensor(rng.random(len(idx)) < float(combo.get("hook.booster_coverage", 0.9)))
    takes = torch.as_tensor(rng.random(len(idx)) < float(combo.get("hook.booster_take", 0.95)))
    lasts_days = rota.config.vaccine_protection_days
    e = torch.as_tensor(rng.standard_exponential(len(idx)), dtype=torch.float64)
    lasts = e * float(lasts_days) if lasts_days else torch.full((len(idx),), float("inf"), dtype=torch.float64)
    renew = idx[given & takes]
    if renew.numel() == 0:
        return
    today = float(rota._vax_day)
    status = nd[K.status(rota.name)]
    new_wane = today + lasts[given & takes]
    in_v = status[renew] == Compartment.VACCINATED   # still protected: keep the later of the two ends
    v["wane_day"][renew] = torch.where(in_v, torch.maximum(v["wane_day"][renew], new_wane), new_wane)
    v["protected"][renew] = True
    vacc_key = K.ever_vaccinated(rota.name)
    if vacc_key in nd:
        nd[vacc_key][renew] = True
    ill_key = K.vaccine_illness_protected(rota.name)
    t_ill = getattr(rota.config, "vaccine_illness_protection_days", None)
    if ill_key in nd and t_ill:
        # protection against illness renewed too, with the same draw (as the schedule does)
        new_ill = today + e[given & takes] * float(t_ill)
        active = nd[ill_key][renew]
        v["ill_until"][renew] = torch.where(active, torch.maximum(v["ill_until"][renew], new_ill), new_ill)
        nd[ill_key][renew] = True
    enter = (status[renew] == Compartment.SUSCEPTIBLE) | (status[renew] == Compartment.RECOVERED)
    status[renew[enter]] = Compartment.VACCINATED

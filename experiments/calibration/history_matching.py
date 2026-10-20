# experiments/calibration/history_matching.py
"""
Joint calibration of all three pathogens and the care-seeking economics by
history matching (Craig et al. 1997; Vernon, Goldstein & Bower 2010; Andrianakis
et al. 2015 for an epidemic-model application).

Why history matching rather than "minimise a loss"
--------------------------------------------------
The targets are not measurements of Akuse: they are MAL-ED's pooled estimates
for comparable settings, with a between-site spread that is as large as the
signal (Campylobacter infection rates span 0.6-2.1 per child-year across the
high-burden sites). The scientific question is therefore not "which single
parameter set fits best" but "which parameter sets are NOT RULED OUT given
that uncertainty". History matching answers exactly that: it discards the
implausible part of parameter space in waves, using cheap statistical
emulators of the model, and reports the non-implausible region (NROY), from
which a representative default can be chosen and whose extent is the honest
uncertainty to carry into a GSA.

Implausibility of parameter set x for target j (on the target's scale -
log for rates and rate ratios):

    I_j(x) = |E[f_j(x)] - z_j| / sqrt(V_obs,j + V_md,j + V_em,j(x))

V_obs is the target's uncertainty (between-site variance for MAL-ED rates, the
band for route fractions, the survey CI for DHS care-seeking); V_md a model-
discrepancy allowance; V_em the emulator's predictive variance (which includes
the replicate-to-replicate stochasticity). x is non-implausible if
max_j I_j(x) < CUTOFF (3, Pukelsheim's 3-sigma rule).

Usage
-----
    # wave 1: Latin hypercube over the full box
    python -m experiments.calibration.history_matching --wave 1 --n 160 --reps 2
    # wave k: emulate all previous waves, sample the current NROY region
    python -m experiments.calibration.history_matching --wave 2 --n 100 --reps 2
    # report NROY, the best simulated points and write best_params.json
    python -m experiments.calibration.history_matching --report
    # emulator-optimal default candidates, then verify them on independent seeds
    python -m experiments.calibration.history_matching --select
    python -m experiments.calibration.history_matching --verify --reps 8 \
        --candidates-json experiments/outputs/history_matching/emulator_candidates.json --tag verify_final
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import qmc

from abm.evidence import maled
from experiments.calibration import targets as T
from experiments.metrics import calibration_metrics, care_seeking_metrics
from experiments.orchestrator import SweepSpec, run_sweep

GRID_ID = "7d9ce7c720a6"
AGENTS = 4000
STEPS = 1460          # 730-day burn-in (experiments.metrics.DEFAULT_BURN_IN_DAYS) + 2 years measured
OUT_DIR = os.path.join("experiments", "outputs", "history_matching")
CUTOFF = 3.0

#: The first wave run with the current treatment model
#: (treatment_effect_model="supportive"). Wave 1 used the original "cure"
#: model, which changes the care-seeking rate (more decision days per
#: episode) but not the epidemiological outputs, so wave 1 still guides where
#: to sample for the epidemiological targets, while the care-seeking target -
#: and every final judgement - uses only waves from this one on.
FIRST_CURRENT_WAVE = 2
TREATMENT_DEPENDENT = {"episode_care_seeking_rate"}

#: MAL-ED targets describe natural history in UNVACCINATED sites, so the
#: calibration world has no rotavirus vaccination (see targets.py).
BASE_OVERRIDES = {"pathogens[rota].vaccination_rate": 0.0}

# ---------------------------------------------------------------------------
# Parameter space: path -> (low, high, scale)
# ---------------------------------------------------------------------------
PARAMS: dict[str, tuple[float, float, str]] = {
    "pathogens[rota].infection_prob_mean":            (0.003,   0.06,  "log"),
    "pathogens[rota].water_dose_scale":               (0.02,    3.0,   "log"),
    "pathogens[campy].human_animal_interaction_rate": (0.0003,  0.012, "log"),
    "pathogens[campy].fecal_oral_prob":               (0.0005,  0.03,  "log"),
    "pathogens[campy].food_borne_prob":               (0.0001,  0.004, "log"),
    "pathogens[crypto].household_transmission_prob":  (0.003,   0.08,  "log"),
    "pathogens[crypto].community_infection_prob":     (0.00002, 0.004, "log"),
    "pathogens[crypto].water_dose_scale":             (0.02,    5.0,   "log"),
    "pathogens[crypto].zoonotic_interaction_rate":    (0.0002,  0.03,  "log"),
    "steering_parameters.household_exposure_sdlog":   (0.0,     1.2,   "lin"),
    "steering_parameters.cost_of_care":               (0.02,    0.2,   "lin"),
    "steering_parameters.daily_income_rate":          (0.03,    0.08,  "lin"),
    # Added after wave 3 (see WAVE3_FIXED): infections acquired outside the town.
    "pathogens[rota].importation_rate":               (2e-5,    3e-3,  "log"),
}
PATHS = list(PARAMS)

#: Parameters added to PARAMS after some waves had run, with the value they
#: were held at in those waves. Waves 1-3 ran with rotavirus importation fixed
#: at 2e-5/person/day - the lower bound of its range - so they are design
#: points on the lower face of the extended box, and remain valid training
#: data for the 13-parameter emulators.
#:
#: Why it was added: at every wave-3 non-implausible point, rotavirus sat near
#: its persistence threshold in a closed town of 4000 - it faded out in 1-3 of
#: 8 seeds and otherwise ran as stochastic outbreaks, and that temporal
#: clustering inflated its prior-infection rate ratio to 2-12 (MAL-ED 1.008).
#: A scan at two emulator-optimal points (experiments/outputs/history_matching/
#: import_scan) showed importation at 1e-3 removes the fade-out (between-seed
#: CV ~10%), brings the ratio to 0.91-1.0 with imported infections only 3-4%
#: of the total, and leaves the Campylobacter and Cryptosporidium outputs
#: unchanged.
WAVE3_FIXED = {"pathogens[rota].importation_rate": 2e-5}
FIRST_IMPORTATION_WAVE = 4

#: Targets whose outputs the wave-3 scan showed do not depend on the rotavirus
#: block (infection probability, water dose, importation). Wave 4 conditions
#: only on these, evaluated with importation at its wave-3 value - the one
#: value the earlier emulators have seen - and samples the rotavirus block
#: freely, since importation moves where rotavirus's plausible region is.
ROTA_INDEPENDENT_TARGETS = {
    "campy_infections_per_child_year_0_24m", "campy_symptomatic_episodes_per_child_year_0_24m",
    "campy_prior_infection_rate_ratio_12_24m", "crypto_infections_per_child_year_0_24m",
    "crypto_symptomatic_episodes_per_child_year_0_24m", "crypto_prior_infection_rate_ratio_12_24m",
    "campy_zoonotic_fraction_post_burnin", "crypto_household_fraction_post_burnin",
    "crypto_zoonotic_fraction_post_burnin",
}
ROTA_BLOCK = ("pathogens[rota].infection_prob_mean", "pathogens[rota].water_dose_scale",
              "pathogens[rota].importation_rate")


def to_unit(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for j, p in enumerate(PATHS):
        lo, hi, sc = PARAMS[p]
        v = values[:, j]
        out[:, j] = (np.log(v / lo) / np.log(hi / lo)) if sc == "log" else (v - lo) / (hi - lo)
    return out


def from_unit(u: np.ndarray) -> np.ndarray:
    out = np.empty_like(u, dtype=float)
    for j, p in enumerate(PATHS):
        lo, hi, sc = PARAMS[p]
        out[:, j] = lo * (hi / lo) ** u[:, j] if sc == "log" else lo + u[:, j] * (hi - lo)
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

@dataclass
class HMTarget:
    metric: str
    z: float              # observed value, on the transformed scale
    sd_obs: float         # observation / setting uncertainty, transformed scale
    sd_md: float          # model-discrepancy allowance, transformed scale
    transform: str        # "log" or "lin"
    eps: float = 0.0      # added before log, so that a fade-out maps to a finite value

    def tx(self, v):
        v = np.asarray(v, dtype=float)
        return np.log(v + self.eps) if self.transform == "log" else v


def build_targets() -> list[HMTarget]:
    out: list[HMTarget] = []
    for p in ("rota", "campy", "crypto"):
        for kind, eps in (("infections", 0.01), ("symptomatic_episodes", 0.003)):
            m = f"{p}_{kind}_per_child_year_0_24m"
            out.append(HMTarget(m, math.log(T.ANCHORS[m] + eps), T.LOG_SD[m], 0.10, "log", eps))
        # MAL-ED Supplementary Table 3, pooled 12-24 months. sd 0.4 on the log
        # scale (~ +/-50%): the published counts give a sampling SD of ~0.1-0.15,
        # widened for between-site heterogeneity (not reported). The ratio also
        # responds to how a pathogen persists: in a closed town, stochastic
        # outbreaks cluster infections in time and inflate it - which is what
        # identified rotavirus importation (see WAVE3_FIXED).
        rr = maled.prior_infection_rate_ratio(p, 12, 24)
        out.append(HMTarget(f"{p}_prior_infection_rate_ratio_12_24m", math.log(rr), 0.40, 0.10, "log", 0.0))
    for m, (lo, hi) in (("campy_zoonotic_fraction_post_burnin", T.TARGETS["campy_zoonotic_fraction_post_burnin"]),
                        ("crypto_household_fraction_post_burnin", T.CRYPTO_TARGETS["crypto_household_fraction_post_burnin"]),
                        ("crypto_zoonotic_fraction_post_burnin", T.CRYPTO_TARGETS["crypto_zoonotic_fraction_post_burnin"])):
        out.append(HMTarget(m, 0.5 * (lo + hi), (hi - lo) / 4.0, 0.02, "lin"))
    # Ghana DHS childhood-diarrhoea care-seeking: 69.2% (95% CI 65.6-72.8, n=621).
    out.append(HMTarget("episode_care_seeking_rate", 0.692, 0.036 / 1.96, 0.03, "lin"))
    return out


TARGETS_HM = build_targets()


def hm_metrics(model) -> dict:
    out = calibration_metrics(model)
    out.update(care_seeking_metrics(model))
    return out


# ---------------------------------------------------------------------------
# Running waves
# ---------------------------------------------------------------------------

def _spec(name: str, reps: int, base_seed: int | None = None, n_cores: int | None = None) -> SweepSpec:
    return SweepSpec(name=name, grid_id=GRID_ID, params=[], metrics_fn=hm_metrics, reps=reps,
                     steps=STEPS, agents=AGENTS, base_overrides=dict(BASE_OVERRIDES),
                     output_dir=OUT_DIR, n_cores=n_cores, base_seed=base_seed)


def combos_from_unit(u: np.ndarray, wave: int) -> list[dict]:
    vals = from_unit(u)
    return [{"gsa.design_id": f"w{wave}_{i:04d}", **{p: float(v) for p, v in zip(PATHS, row)}}
            for i, row in enumerate(vals)]


def load_all_waves() -> pd.DataFrame:
    frames = []
    for name in sorted(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else []:
        path = os.path.join(OUT_DIR, name, "results.parquet")
        if name.startswith("wave") and os.path.exists(path):
            df = pd.read_parquet(path)
            df["wave"] = int(name[len("wave"):])
            for p, v in WAVE3_FIXED.items():
                if p not in df:
                    df[p] = v
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per design point: transformed replicate means and variances."""
    rows = []
    for design_id, g in df.groupby("gsa.design_id"):
        rec = {"gsa.design_id": design_id, "wave": int(g["wave"].iloc[0]), "n_rep": len(g)}
        for p in PATHS:
            rec[p] = float(g[p].iloc[0])
        for t in TARGETS_HM:
            y = t.tx(g[t.metric].to_numpy(dtype=float)) if t.metric in g else np.array([np.nan])
            y = y[np.isfinite(y)]
            rec[t.metric] = float(y.mean()) if y.size else np.nan
            rec[t.metric + "__var"] = float(y.var(ddof=1)) if y.size > 1 else np.nan
            raw = g[t.metric].to_numpy(dtype=float) if t.metric in g else np.array([np.nan])
            rec[t.metric + "__raw"] = float(np.nanmean(raw)) if np.isfinite(raw).any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def simulated_implausibility(summary: pd.DataFrame) -> pd.DataFrame:
    """Implausibility of each SIMULATED point, using its replicate mean and
    variance (no emulator)."""
    out = summary.copy()
    cols = []
    for t in TARGETS_HM:
        var_ens = out[t.metric + "__var"].fillna(0.0) / out["n_rep"]
        denom = np.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + var_ens)
        col = "I__" + t.metric
        out[col] = (out[t.metric] - t.z).abs() / denom
        out.loc[out[t.metric].isna(), col] = np.inf
        cols.append(col)
    out["I_max"] = out[cols].max(axis=1)
    out["I_second"] = out[cols].apply(lambda r: np.sort(r.to_numpy())[-2], axis=1)
    return out.sort_values("I_max").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Emulation
# ---------------------------------------------------------------------------

def fit_emulators(summary: pd.DataFrame, current_only: set[str] | None = None) -> dict:
    """One GP per target. Targets in `current_only` are fitted only on waves
    >= FIRST_CURRENT_WAVE (skipped if fewer than 10 such points exist)."""
    current_only = TREATMENT_DEPENDENT if current_only is None else current_only
    full = summary
    current = summary[summary["wave"] >= FIRST_CURRENT_WAVE]
    ems = {}
    for t in TARGETS_HM:
        data = current if t.metric in current_only else full
        gp = _fit_one(data, t)
        if gp is not None:
            ems[t.metric] = gp
    return ems


def _fit_one(summary: pd.DataFrame, t: "HMTarget"):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    # Length scales of parameters an output does not depend on run to the
    # upper bound - expected (that is how ARD reports irrelevance), not a
    # fitting problem.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    X = to_unit(summary[PATHS].to_numpy(dtype=float))
    y = summary[t.metric].to_numpy(dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 10:
        return None
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.full(len(PATHS), 0.5), length_scale_bounds=(0.05, 20.0), nu=2.5)
              + WhiteKernel(0.05, (1e-6, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    gp.fit(X[ok], y[ok])
    return gp


def emulated_implausibility(ems: dict, U: np.ndarray, chunk: int = 20000,
                            only: set[str] | None = None) -> np.ndarray:
    """(n, n_targets) emulator implausibility at unit-cube points U. With
    `only`, the other targets' columns are left at 0."""
    I = np.zeros((len(U), len(TARGETS_HM)))
    for j, t in enumerate(TARGETS_HM):
        gp = ems.get(t.metric)
        if gp is None or (only is not None and t.metric not in only):
            continue
        for a in range(0, len(U), chunk):
            mu, sd = gp.predict(U[a:a + chunk], return_std=True)
            I[a:a + chunk, j] = np.abs(mu - t.z) / np.sqrt(t.sd_obs ** 2 + t.sd_md ** 2 + sd ** 2)
    return I


def _criterion(I: np.ndarray, which: str) -> np.ndarray:
    """max ("max") or second-largest ("second") implausibility per row. The
    second-max rule - one target may miss - is the usual choice for early
    waves with many outputs (Vernon et al. 2010), so that one poorly emulated
    or structurally hard output cannot empty the space on its own."""
    if which == "max":
        return I.max(axis=1)
    return np.sort(I, axis=1)[:, -2]


def sample_nroy(ems: dict, n: int, seed: int, n_candidates: int = 200_000,
                criterion: str = "second", conditional_on_rota_independent: bool = False
                ) -> tuple[np.ndarray, float]:
    """n space-filling points from the emulated NROY region, and its volume fraction.

    conditional_on_rota_independent: judge only ROTA_INDEPENDENT_TARGETS, with
    rotavirus importation set to its wave-3 value for the emulator, leaving
    the rotavirus block free (wave 4; see WAVE3_FIXED)."""
    rng = np.random.default_rng(seed)
    U = qmc.LatinHypercube(d=len(PATHS), seed=seed).random(n_candidates)
    if conditional_on_rota_independent:
        U_eval = U.copy()
        for p, v in WAVE3_FIXED.items():
            U_eval[:, PATHS.index(p)] = to_unit(_row_with(p, v))[0, PATHS.index(p)]
        I = emulated_implausibility(ems, U_eval, only=ROTA_INDEPENDENT_TARGETS)
    else:
        I = emulated_implausibility(ems, U)
    score = _criterion(I, criterion)
    keep = score < CUTOFF
    frac = float(keep.mean())
    pool = U[keep]
    if len(pool) == 0:
        # Fall back to the least-implausible candidates so the wave can proceed.
        order = np.argsort(score)
        pool = U[order[: max(n * 20, 1000)]]
    if len(pool) <= n:
        return pool, frac
    # Greedy maximin subsample for space filling.
    chosen = [int(rng.integers(len(pool)))]
    d = np.linalg.norm(pool - pool[chosen[0]], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(pool - pool[k], axis=1))
    return pool[chosen], frac


def _row_with(path: str, value: float) -> np.ndarray:
    """A 1-row parameter array at the box's lower corner with `path` = value."""
    row = from_unit(np.zeros((1, len(PATHS))))
    row[0, PATHS.index(path)] = value
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_wave(wave: int, n: int, reps: int, seed: int, n_cores: int | None, criterion: str = "second"):
    os.makedirs(OUT_DIR, exist_ok=True)
    if wave == 1:
        U = qmc.LatinHypercube(d=len(PATHS), seed=seed).random(n)
        frac = 1.0
    else:
        summary = summarise(load_all_waves())
        ems = fit_emulators(summary)
        U, frac = sample_nroy(ems, n, seed=seed + wave, criterion=criterion,
                              conditional_on_rota_independent=(wave == FIRST_IMPORTATION_WAVE))
    print(f"wave {wave}: emulated NROY fraction of the original box = {frac:.4f}; running {len(U)} points")
    combos = combos_from_unit(U, wave)
    run_sweep(_spec(f"wave{wave}", reps, n_cores=n_cores), combos)
    with open(os.path.join(OUT_DIR, f"wave{wave}_meta.json"), "w") as f:
        json.dump({"wave": wave, "n": len(U), "reps": reps, "criterion": criterion,
                   "nroy_fraction_before": frac}, f, indent=2)


def report(top: int = 10) -> pd.DataFrame:
    """Implausibility of every simulated point of the CURRENT model
    (waves >= FIRST_CURRENT_WAVE)."""
    summary = summarise(load_all_waves())
    summary = summary[summary["wave"] >= FIRST_CURRENT_WAVE]
    imp = simulated_implausibility(summary)
    imp.to_csv(os.path.join(OUT_DIR, "simulated_implausibility.csv"), index=False)
    ok = imp[imp["I_max"] < CUTOFF]
    print(f"{len(imp)} simulated points; {len(ok)} with every target non-implausible (I < {CUTOFF}).")
    show = ["gsa.design_id", "I_max", "I_second"] + [t.metric + "__raw" for t in TARGETS_HM]
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(imp[show].head(top).T)
    if len(ok):
        print("\nNon-implausible simulated points - parameter ranges:")
        for p in PATHS:
            print(f"  {p:52s} {ok[p].min():.5g} .. {ok[p].max():.5g} (median {ok[p].median():.5g})")
    best = imp.iloc[0]
    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump({"design_id": best["gsa.design_id"], "I_max": float(best["I_max"]),
                   "params": {p: float(best[p]) for p in PATHS},
                   "base_overrides": BASE_OVERRIDES}, f, indent=2)
    return imp


#: Upper limit on the share of rotavirus infections that are imported, applied
#: when choosing the default (not a calibration target). The MAL-ED targets do
#: not distinguish a town that sustains rotavirus locally from one that is a
#: sink for infection from outside: in the wave-5 NROY the imported share runs
#: from <5% to >40% at equal fit. The two make opposite predictions for any
#: intervention applied in Akuse, which cannot reduce an external force of
#: infection. Importation was introduced to stop stochastic fade-out (a spark,
#: not a route), so the default is held to mostly-local transmission. This is
#: a modelling judgement - report it, and vary importation_rate in any GSA.
MAX_IMPORTED_FRACTION = 0.15
AUX_IMPORTED = "rota_imported_fraction_post_burnin"


def fit_imported_emulator(df: pd.DataFrame, summary: pd.DataFrame):
    """GP of the replicate-mean rotavirus imported fraction (linear scale)."""
    s = summary.copy()
    s[AUX_IMPORTED] = s["gsa.design_id"].map(df.groupby("gsa.design_id")[AUX_IMPORTED].mean())
    return _fit_one(s, HMTarget(AUX_IMPORTED, 0.0, 1.0, 0.0, "lin"))


def _predict_chunked(gp, U: np.ndarray, chunk: int = 20000) -> np.ndarray:
    # Chunked: an ARD kernel against ~500 training points materialises an
    # (n x 500 x d) array, which for n = 3e5 does not fit in memory.
    return np.concatenate([gp.predict(U[a:a + chunk]) for a in range(0, len(U), chunk)])


def select_candidates(n: int = 3, seed: int = 11, n_candidates: int = 300_000,
                      min_separation: float = 0.15, centrality_weight: float = 0.0,
                      max_imported_fraction: float = MAX_IMPORTED_FRACTION) -> list[dict]:
    """Default candidates: a regularised best fit within the emulated NROY.

    The admissible set is the emulated NROY (max_j I_j < CUTOFF) with the
    emulated rotavirus imported fraction below `max_imported_fraction`. Within
    it, candidates minimise

        sum_j I_j(u)^2  +  centrality_weight * sum_k ((u_k - m_k) / s_k)^2

    over the unit cube, where m_k and s_k are the median and SD of parameter k
    across the admissible set. The first term is the standardised squared
    distance to every target at once. The second (off by default) pulls
    parameters the targets leave flat towards the centre of their admissible
    range. Tried at weight 1, it dominated the objective (12.3 vs a fit term of
    5) because marginal medians ignore the strong correlations within the NROY
    (e.g. rota infection probability vs water dose, r = -0.8), and the chosen
    points fitted worse on fresh seeds (verify_final2: max I 1.7-2.1 vs 1.1-1.3).
    The default is therefore the plain best fit; weakly identified parameters
    are reported as admissible ranges (nroy_marginals.csv), not resolved.

    Refined by two rounds of local random search around the 20 best; `n`
    mutually separated points (unit-cube distance > min_separation) are
    returned and written to emulator_candidates.json for
    `--verify --candidates-json`.
    """
    df = load_all_waves()
    summary = summarise(df)
    ems = fit_emulators(summary)
    gp_imp = fit_imported_emulator(df, summary)
    rng = np.random.default_rng(seed)
    U = qmc.LatinHypercube(d=len(PATHS), seed=seed).random(n_candidates)

    def admissible(X):
        I = emulated_implausibility(ems, X)
        imp = _predict_chunked(gp_imp, X)
        return I, imp, (I.max(axis=1) < CUTOFF) & (imp < max_imported_fraction)

    I, imp, ok = admissible(U)
    print(f"emulated NROY fraction (max I < {CUTOFF}): {(I.max(axis=1) < CUTOFF).mean():.5f}; "
          f"with imported fraction < {max_imported_fraction}: {ok.mean():.5f}")
    m, sd = np.median(U[ok], axis=0), U[ok].std(axis=0)

    def objective(X):
        I_, imp_, ok_ = admissible(X)
        val = (I_ ** 2).sum(axis=1) + centrality_weight * (((X - m) / sd) ** 2).sum(axis=1)
        return I_, imp_, np.where(ok_, val, np.inf)

    _, _, obj = objective(U)
    best = U[np.argsort(obj)[:20]]
    pool = [best]
    for step in (0.05, 0.02):
        c = np.clip(np.repeat(best, 200, axis=0) + rng.normal(0, step, (len(best) * 200, len(PATHS))), 0, 1)
        _, _, obj_c = objective(c)
        best = c[np.argsort(obj_c)[:20]]
        pool.append(best)
    allc = np.vstack(pool)
    I_all, imp_all, obj_all = objective(allc)
    picked: list[int] = []
    for k in np.argsort(obj_all):
        if not np.isfinite(obj_all[k]):
            break
        if all(np.linalg.norm(allc[k] - allc[j]) > min_separation for j in picked):
            picked.append(int(k))
        if len(picked) == n:
            break
    values = from_unit(allc[picked])
    out = []
    for i, (k, row) in enumerate(zip(picked, values)):
        print(f"\ncandidate e{i}: objective {obj_all[k]:.2f} (sum I^2 {(I_all[k] ** 2).sum():.2f}), "
              f"emulated max I {I_all[k].max():.2f}, imported fraction {imp_all[k]:.3f}")
        for t, Ij in zip(TARGETS_HM, I_all[k]):
            mu = ems[t.metric].predict(allc[k:k + 1])[0]
            val = np.exp(mu) - t.eps if t.transform == "log" else mu
            print(f"   {t.metric:52s} emulated {val:7.3f}   I {Ij:.2f}")
        for j, p in enumerate(PATHS):
            print(f"   {p:52s} {row[j]:.4g}   ({(allc[k, j] - m[j]) / sd[j]:+.2f} NROY SD from median)")
        out.append({"gsa.design_id": f"e{i}", **{p: float(v) for p, v in zip(PATHS, row)}})
    with open(os.path.join(OUT_DIR, "emulator_candidates.json"), "w") as f:
        json.dump(out, f, indent=1)
    nroy = pd.DataFrame(from_unit(U[ok]), columns=PATHS).quantile([0.025, 0.25, 0.5, 0.75, 0.975]).T
    nroy.to_csv(os.path.join(OUT_DIR, "nroy_marginals.csv"))
    print("\nadmissible-set marginals (2.5/25/50/75/97.5%):")
    with pd.option_context("display.width", 200):
        print(nroy.to_string(float_format=lambda x: f"{x:.4g}"))
    return out


def verify(reps: int, n_cores: int | None, design_ids: list[str] | None = None, top: int = 3,
           candidates_json: str | None = None, tag: str = "verify"):
    """Re-run chosen points on seeds disjoint from the waves (base_seed 10023).

    Points come either from simulated_implausibility.csv (by design id, or the
    `top` best) or from a JSON list of {"gsa.design_id": ..., <PATHS>: value}
    dicts - e.g. emulator-optimised points that were never simulated."""
    if candidates_json:
        with open(candidates_json) as f:
            combos = [{"gsa.design_id": f"{tag}_" + c["gsa.design_id"], **{p: float(c[p]) for p in PATHS}}
                      for c in json.load(f)]
    else:
        imp = pd.read_csv(os.path.join(OUT_DIR, "simulated_implausibility.csv"))
        sel = imp[imp["gsa.design_id"].isin(design_ids)] if design_ids else imp.head(top)
        combos = [{"gsa.design_id": f"{tag}_" + r["gsa.design_id"], **{p: float(r[p]) for p in PATHS}}
                  for _, r in sel.iterrows()]
    df = run_sweep(_spec(tag, reps, base_seed=10_023, n_cores=n_cores), combos)
    df["wave"] = 99
    v = simulated_implausibility(summarise(df))
    # Between-seed stability: a default that fades out in some seeds is not a
    # usable baseline however good its mean, so report it alongside I.
    stab = df.groupby("gsa.design_id").agg(
        rota_fadeout_reps=("rota_infections_per_child_year_0_24m", lambda x: int((x < 0.05).sum())),
        rota_infections_cv=("rota_infections_per_child_year_0_24m", lambda x: float(x.std() / x.mean())),
        crypto_infections_cv=("crypto_infections_per_child_year_0_24m", lambda x: float(x.std() / x.mean())),
        rota_imported_fraction=(AUX_IMPORTED, "mean"),
    ).reset_index()
    v = v.merge(stab, on="gsa.design_id", how="left")
    v.to_csv(os.path.join(OUT_DIR, f"{tag}_implausibility.csv"), index=False)
    show = (["gsa.design_id", "I_max", "I_second", "rota_fadeout_reps", "rota_infections_cv",
             "crypto_infections_cv", "rota_imported_fraction"] + [t.metric + "__raw" for t in TARGETS_HM])
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(v[show].T)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--wave", type=int)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--select", action="store_true",
                    help="write emulator-optimal default candidates to emulator_candidates.json")
    ap.add_argument("--centrality-weight", type=float, default=0.0, help="see select_candidates")
    ap.add_argument("--design-ids", nargs="*", default=None)
    ap.add_argument("--candidates-json", default=None,
                    help="verify points listed in this JSON file instead of simulated ones")
    ap.add_argument("--tag", default="verify", help="output name for --verify")
    ap.add_argument("--criterion", choices=("max", "second"), default="second")
    args = ap.parse_args()
    if args.wave:
        run_wave(args.wave, args.n, args.reps, args.seed, args.workers, args.criterion)
    if args.report:
        report()
    if args.select:
        select_candidates(centrality_weight=args.centrality_weight)
    if args.verify:
        verify(args.reps, args.workers, args.design_ids, candidates_json=args.candidates_json, tag=args.tag)


if __name__ == "__main__":
    main()

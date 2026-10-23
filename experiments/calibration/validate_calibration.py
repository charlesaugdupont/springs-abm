# experiments/calibration/validate_calibration.py
"""
Out-of-sample checks of the calibrated model against evidence that was NOT
used as a calibration target.

Runs the current config defaults (rotavirus vaccination off, as in
calibration) for `--reps` seeds disjoint from those used in calibration, and
reports, next to the MAL-ED values:

  * the age-specific infection curve, 3-month bands to 24 months (Suppl. Table 2)
  * median age at first infection and fraction infected by 24 months (Table 4)
  * P(diarrhoea | infection) by age (Table 5) - an input, so a consistency check
  * realized and natural diarrhoea duration (Table 11)
  * route mixes, the imported fraction, carriage prevalence, adult share of
    infections and the care-seeking/economic outputs

With --campaign v3 (the default) it runs the settlement grid (11,476 agents,
GHS-POP households) and adds the MAL-ED observation layer, so the detected
age curve, age at first detection and P(attributable | detected infection)
are compared like for like with MAL-ED, next to the true values; --campaign
v2 runs the original grid with experiments.legacy.V2_OVERRIDES (4000 agents).

Usage:
    python -m experiments.calibration.validate_calibration --reps 8 --workers 8
    python -m experiments.calibration.validate_calibration --campaign v2 --reps 8
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from abm.constants import AgentPropertyKeys
from abm.evidence import maled
from experiments.legacy import V2_OVERRIDES
from experiments.metrics import calibration_metrics, care_seeking_metrics, wellbeing_metrics
from experiments.observation import observe
from experiments.orchestrator import SweepSpec, run_sweep

#: campaign -> (grid, agents, base overrides, output folder)
CAMPAIGNS = {
    "v2": ("7d9ce7c720a6", 4000, {**V2_OVERRIDES, "pathogens[rota].vaccination_rate": 0.0},
           os.path.join("experiments", "outputs", "calibration_validation")),
}
GRID_ID, OUT_DIR = CAMPAIGNS["v2"][0], CAMPAIGNS["v2"][3]


def _v3_campaign():
    from experiments.calibration.history_matching import V3_AGENTS, V3_BASE_OVERRIDES, V3_GRID_ID
    return (V3_GRID_ID, V3_AGENTS, dict(V3_BASE_OVERRIDES),
            os.path.join("experiments", "outputs", "calibration_validation_v3"))
BANDS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24)]


def validation_metrics(model) -> dict:
    out = calibration_metrics(model)
    out.update(care_seeking_metrics(model))
    out.update(wellbeing_metrics(model))
    # Share of each pathogen's post-burn-in infections that were in adults.
    is_child = model.graph.ndata[AgentPropertyKeys.IS_CHILD].bool()
    for p in model.pathogens:
        total = sum(sum(v[730:]) for v in p.daily_by_route.values()) if p.daily_by_route else 0
        child = float(np.asarray(model.new_infections_by_age_month[p.name])[730:].sum())
        out[f"{p.name}_adult_share_post_burnin"] = (1 - child / total) if total else float("nan")
    # Natural (initial) duration of episodes with onset at 0-24 months, post burn-in.
    for p in model.pathogens:
        d = [e["initial_duration"] for e in model.get_child_episode_log()
             if e["pathogen"] == p.name and e["timestep"] >= 730 and e["age_months"] < 24]
        out[f"{p.name}_median_natural_duration"] = float(np.median(d)) if d else float("nan")
    out["n_u5_children"] = int(is_child.sum())
    out["n_agents"] = int(is_child.numel())
    return out


def validation_metrics_observed(model) -> dict:
    """validation_metrics plus the MAL-ED observation layer (campaign v3)."""
    out = validation_metrics(model)
    out.update(observe(model))
    return out


def run(reps: int, steps: int, agents: int, n_cores: int | None, campaign: str = "v2") -> pd.DataFrame:
    grid, _, base, out_dir = CAMPAIGNS[campaign] if campaign in CAMPAIGNS else _v3_campaign()
    spec = SweepSpec(name="validation", grid_id=grid, params=[],
                     metrics_fn=validation_metrics if campaign == "v2" else validation_metrics_observed,
                     reps=reps, steps=steps, agents=agents, output_dir=out_dir, n_cores=n_cores,
                     base_seed=50_023, base_overrides=base)
    return run_sweep(spec, [{"gsa.design_id": "defaults"}])


def report(df: pd.DataFrame, out_dir: str = OUT_DIR) -> dict:
    m = df.mean(numeric_only=True)
    sd = df.std(numeric_only=True)
    lines, out = [], {}
    observed = "rota_obs_infections_per_child_year_0_24m" in m
    for p in ("rota", "campy", "crypto"):
        true = [m.get(f"{p}_infections_per_child_year_{lo}_{hi}m", np.nan) for lo, hi in BANDS]
        det = [m.get(f"{p}_obs_infections_per_child_year_{lo}_{hi}m", np.nan) for lo, hi in BANDS]
        mod = det if observed else true
        obs = [v * 0.12 for v in maled.INFECTION_RATE_BY_AGE_PER_100CM[p]]
        lines.append(f"\n{p}  infections per child-year by 3-month band"
                     + (" (model: DETECTED, MAL-ED observation layer)" if observed else ""))
        lines.append("   model : " + " ".join(f"{v:5.2f}" for v in mod))
        if observed:
            lines.append("   (true): " + " ".join(f"{v:5.2f}" for v in true))
        lines.append("   MAL-ED: " + " ".join(f"{v:5.2f}" for v in obs))
        # Shape: correlation of the two curves and ratio of second-year to first-year rates.
        r = float(np.corrcoef(mod, obs)[0, 1]) if np.all(np.isfinite(mod)) else float("nan")
        yr_ratio_mod = np.mean(mod[4:]) / np.mean(mod[:4]) if np.all(np.isfinite(mod)) else float("nan")
        yr_ratio_obs = np.mean(obs[4:]) / np.mean(obs[:4])
        lines.append(f"   shape correlation {r:.2f}; 2nd-year / 1st-year rate: model {yr_ratio_mod:.2f}, MAL-ED {yr_ratio_obs:.2f}")
        med, q1, q3 = maled.AGE_AT_FIRST_INFECTION_DAYS[p]
        lines.append(f"   median age at first infection: model{' (TRUE)' if observed else ''} "
                     f"{m.get(f'{p}_median_age_first_infection_days', np.nan):.0f} d "
                     f"(sd {sd.get(f'{p}_median_age_first_infection_days', np.nan):.0f}); MAL-ED {med:.0f} (IQR {q1:.0f}-{q3:.0f})")
        if observed:
            lines.append(f"   median age at first DETECTED infection: model "
                         f"{m.get(f'{p}_obs_median_age_first_detected_infection_months', np.nan) * 30.44:.0f} d; "
                         f"P(attributable | detected infection): model "
                         f"{m.get(f'{p}_obs_attributable_fraction_0_24m', np.nan):.3f}")
            lines.append(f"   true: infected by 6 m {m.get(f'{p}_true_frac_infected_by_6m', np.nan):.2f}, by 24 m "
                         f"{m.get(f'{p}_true_frac_infected_by_24m', np.nan):.2f}; primary share of infections "
                         f"0-36 m {m.get(f'{p}_true_primary_share_0_36m', np.nan):.2f}")
        lines.append(f"   fraction infected by 24 m: model {m.get(f'{p}_frac_infected_by_24m', np.nan):.2f}")
        lines.append(f"   P(episode|infection) 0-24 m: model{' (TRUE infections; compare the detected value above)' if observed else ''} "
                     f"{m.get(f'{p}_symptomatic_fraction_0_24m', np.nan):.3f}; "
                     f"MAL-ED {maled.overall_symptomatic_fraction(p):.3f}")
        lines.append(f"   median duration: natural {m.get(f'{p}_median_natural_duration', np.nan):.1f} d, realized "
                     f"{m.get(f'{p}_median_realized_duration', np.nan):.1f} d; MAL-ED 2-3 d")
        share = m.get(f"{p}_adult_share_post_burnin", np.nan)
        lines.append(f"   carriage prevalence (u5, mean) {m.get(f'{p}_mean_u5_prevalence_post_burnin', np.nan):.4f}; "
                     f"adult share of infections {share:.2f}")
        # per-person infection rates, adults (incl. children 5+) vs under-fives
        n_u5, n_all = m.get("n_u5_children", np.nan), m.get("n_agents", np.nan)
        u5_rate = m.get(f"{p}_infections_per_child_year_0_60m", np.nan)
        if np.isfinite(n_all) and np.isfinite(share) and share < 1:
            adult_rate = u5_rate * n_u5 * share / (1 - share) / (n_all - n_u5)
            lines.append(f"   infections per person-year: under-5 {u5_rate:.2f}, everyone older {adult_rate:.2f} "
                         f"({adult_rate / u5_rate:.1f}x)")
        routes = {k: m[k] for k in m.index if k.startswith(f"{p}_") and k.endswith("_fraction_post_burnin")}
        lines.append("   routes: " + ", ".join(f"{k[len(p)+1:-len('_fraction_post_burnin')]} {v:.2f}" for k, v in routes.items()))
        out[p] = {"model_curve": mod, "maled_curve": obs, "shape_r": r}
    lines.append(f"\ncare-seeking rate {m['episode_care_seeking_rate']:.3f} (DHS 0.692, 95% CI 0.656-0.728); "
                 f"could not afford {m['could_not_afford_rate']:.3f}; mean household wealth {m['mean_household_wealth']:.3f}")
    text = "\n".join(lines)
    print(text)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "validation_report.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "validation_curves.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1826)
    ap.add_argument("--agents", type=int, default=None, help="default: the campaign's (v2 4000, v3 11,476)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--campaign", choices=("v2", "v3"), default="v3")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    _, agents, _, out_dir = CAMPAIGNS[args.campaign] if args.campaign in CAMPAIGNS else _v3_campaign()
    if args.report_only:
        df = pd.read_parquet(os.path.join(out_dir, "validation", "results.parquet"))
    else:
        df = run(args.reps, args.steps, args.agents or agents, args.workers, args.campaign)
    report(df, out_dir)


if __name__ == "__main__":
    main()

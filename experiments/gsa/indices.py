# experiments/gsa/indices.py
"""
SALib index computation + robust sample<->output alignment + noise diagnostics.

Alignment: each design point (row of the SALib sample matrix) is tagged with a
gsa.design_id before the sweep (run_gsa injects it; orchestrator carries it into
results.parquet and skips it as a config path). We average each design point's
metric over its stochastic replicates and reindex to design-id order - this is
exactly the order SALib's analyze expects, and the integer id makes the join
exact (no fragile float-key matching).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from SALib.analyze import morris as _morris_analyze
from SALib.analyze import sobol as _sobol_analyze

from experiments.metrics import replicate_dispersion

DESIGN_ID_COL = "gsa.design_id"


def align_mean_output(df: pd.DataFrame, n_design: int, metric: str) -> np.ndarray:
    """Per-design-point mean of `metric`, in design-id order (length n_design)."""
    if metric not in df.columns:
        raise KeyError(f"metric '{metric}' not in results columns: {list(df.columns)[:12]}...")
    grouped = df.groupby(DESIGN_ID_COL)[metric].mean()
    Y = grouped.reindex(range(n_design)).to_numpy(dtype=float)
    n_missing = int(np.isnan(Y).sum())
    if n_missing:
        missing = [i for i in range(n_design) if i not in set(grouped.index)]
        raise ValueError(
            f"align_mean_output: {n_missing}/{n_design} design points have no successful "
            f"runs for metric '{metric}' (NaN). First missing design ids: {missing[:10]}"
        )
    return Y


def run_morris(problem: dict, X: np.ndarray, Y: np.ndarray, num_levels: int = 4) -> pd.DataFrame:
    """Morris elementary-effects: mu, mu_star (importance rank), sigma (interaction/nonlinearity)."""
    Si = _morris_analyze.analyze(problem, X, Y, num_levels=num_levels, print_to_console=False)
    return pd.DataFrame({
        "path": problem["names"],
        "mu": Si["mu"],
        "mu_star": Si["mu_star"],
        "sigma": Si["sigma"],
        "mu_star_conf": Si["mu_star_conf"],
    }).sort_values("mu_star", ascending=False).reset_index(drop=True)


def run_sobol(problem: dict, Y: np.ndarray, calc_second_order: bool = False) -> pd.DataFrame:
    """Sobol variance decomposition: S1 (first-order) and ST (total, incl. interactions)."""
    Si = _sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order,
                                print_to_console=False)
    return pd.DataFrame({
        "path": problem["names"],
        "S1": Si["S1"], "S1_conf": Si["S1_conf"],
        "ST": Si["ST"], "ST_conf": Si["ST_conf"],
    }).sort_values("ST", ascending=False).reset_index(drop=True)


def noise_floor(df: pd.DataFrame, metric: str, reps: int) -> dict:
    """Separate parametric variance from the stochastic-noise floor.

    Uses replicate_dispersion per design point. The mean is an adequate summary
    when the standard error of the per-design mean (median rep std / sqrt(reps))
    is much smaller than the parametric spread (std of the per-design means) -
    i.e. snr >> 1. Also flags bimodal design points (tipping-point behavior where
    the mean is a poor summary).
    """
    disp = replicate_dispersion(df, group_cols=[DESIGN_ID_COL], metric_col=metric)
    med_std = float(np.nanmedian(disp["std"].to_numpy()))
    se_mean = med_std / np.sqrt(max(reps, 1))
    parametric_spread = float(np.nanstd(disp["mean"].to_numpy()))
    snr = (parametric_spread / se_mean) if se_mean > 0 else float("inf")
    n_bimodal = int((disp["bimodality_coef"] > 0.555).sum())
    return {
        "metric": metric,
        "median_rep_std": med_std,
        "se_of_mean": se_mean,
        "parametric_spread": parametric_spread,
        "snr": snr,                       # want >> 1
        "n_bimodal_points": n_bimodal,
        "n_design_points": int(len(disp)),
    }

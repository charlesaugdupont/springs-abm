"""
Paper 1's analysis (experiments.scenarios.paper1_analysis): its estimators on
synthetic data whose parts are known, and its bookkeeping of the numbers it
writes for the text.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from experiments.scenarios import paper1_analysis as pa


# ---------------------------------------------------------------------------
# Contrasts and the numbers for the text
# ---------------------------------------------------------------------------

def test_paired_contrast():
    a = pd.Series([2.0, 4.0, 6.0], index=[61001, 61002, 61003])
    b = pd.Series([1.0, 2.0, 3.0, 9.0], index=[61001, 61002, 61003, 61004])     # 61004 has no pair
    p = pa.pair(a, b)
    assert p["n"] == 3 and p["diff"] == pytest.approx(2.0) and p["b"] == pytest.approx(2.0)
    assert p["rel"] == pytest.approx(1.0)
    assert p["se"] == pytest.approx(1.0 / math.sqrt(3.0))
    assert p["rel_se"] == pytest.approx(0.5 / math.sqrt(3.0))
    assert (p["lower"], p["higher"]) == (0, 3)


def test_macros_round_and_refuse_bad_names(monkeypatch):
    monkeypatch.setattr(pa, "VALUES", {})
    pa.macro("Big", 1234.567, 1)
    pa.macro("Negative", -5.4)
    pa.macro("Missing", float("nan"))
    pa.macro("Text", "12--15")
    assert pa.VALUES == {"Big": "1{,}234.6", "Negative": "-5", "Missing": "NaN", "Text": "12--15"}
    with pytest.raises(ValueError):
        pa.macro("not_letters", 1.0)


def test_values_file_keeps_the_numbers_of_parts_not_rerun(monkeypatch, tmp_path):
    (tmp_path / "values_p1.tex").write_text("\\newcommand{\\pvOld}{1}\n\\newcommand{\\pvBoth}{2}\n")
    monkeypatch.setattr(pa, "VALUES", {"Both": "3", "New": "4"})
    pa.write_values(str(tmp_path))
    lines = [ln for ln in (tmp_path / "values_p1.tex").read_text().splitlines() if ln.startswith("\\newcommand")]
    assert lines == ["\\newcommand{\\pvBoth}{3}", "\\newcommand{\\pvNew}{4}", "\\newcommand{\\pvOld}{1}"]


# ---------------------------------------------------------------------------
# The parameter sets: their spread against the seeds' noise
# ---------------------------------------------------------------------------

def test_blocked_spread_recovers_known_parts():
    """Sets x seeds = set effect (SD 1) + a seed effect every set shares (SD 2)
    + interaction (SD 1.5). The blocked estimate finds the sets' variance and the
    noise in a set's mean; the one-way estimate counts the shared seed effect
    as noise and understates the sets' variance."""
    rng = np.random.default_rng(61_001)
    S, R, sp, sc, su = 11, 8, 1.0, 2.0, 1.5
    params, noise, noise_one_way = [], [], []
    for _ in range(400):
        A = rng.normal(0, sp, (S, 1)) + rng.normal(0, sc, (1, R)) + rng.normal(0, su, (S, R))
        r = pa._blocked_spread(pd.DataFrame(A))
        params.append(r["sd_parameters"] ** 2)
        noise.append(r["se_set_mean"] ** 2)
        noise_one_way.append(r["se_set_mean_independent"] ** 2)
    assert np.mean(params) == pytest.approx(sp ** 2, rel=0.10)
    assert np.mean(noise) == pytest.approx(su ** 2 / R, rel=0.05)
    assert np.mean(noise_one_way) == pytest.approx((sc ** 2 + su ** 2) / R, rel=0.05)


def test_blocked_spread_leaves_out_a_set_with_a_missing_run():
    A = pd.DataFrame(np.arange(24, dtype=float).reshape(4, 6) + np.array([[0.0], [3.0], [1.0], [7.0]]))
    A.iloc[2, 3] = np.nan
    r = pa._blocked_spread(A)
    assert r["sets_used"] == 3
    # rows that differ only by a constant: no noise, all the spread from the sets
    assert r["se_set_mean"] == pytest.approx(0.0, abs=1e-12) and r["share_from_parameters"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Morris: the noise floor and the screening
# ---------------------------------------------------------------------------

def _morris_runs(f, c, u, labels):
    """Runs of every point at two seeds: y = f(point) + c(seed) + u(point, seed)."""
    rows = []
    for i, lab in enumerate(labels):
        for r in (0, 1):
            rows.append({pa.LABEL: lab, "rep": r, "seed": 61_001 + r, "y": f[i] + c[r] + u[i, r]})
    return pd.DataFrame(rows)


def test_step_noise_is_the_seeds_own_noise_whatever_the_effects():
    """The floor is sqrt(2/pi) x the interaction SD, and the factors' effects and
    the seed effect every point shares cancel from it exactly."""
    k, T, su = 10, 30, 1.0
    labels = [f"morris_{i:03d}" for i in range(T * (k + 1))]
    rng = np.random.default_rng(61_001)
    u = rng.normal(0, su, (len(labels), 2))
    effects = _morris_runs(rng.normal(0, 10, len(labels)), [0.0, 25.0], u, labels)
    none = _morris_runs(np.zeros(len(labels)), [0.0, 0.0], u, labels)
    floor, steps = pa._step_noise(effects, effects["y"], labels, k)
    floor0, _ = pa._step_noise(none, none["y"], labels, k)
    assert steps == T * k
    assert floor == pytest.approx(floor0, rel=1e-9)
    assert floor == pytest.approx(math.sqrt(2 / math.pi) * su, rel=0.15)


def test_morris_finds_a_real_effect_and_no_false_ones(monkeypatch, tmp_path):
    """On the fixed design, an output that only the first factor moves (by 5 noise
    floors across a step) is flagged for that factor and for no other."""
    from experiments.scenarios.paper1 import MORRIS_FACTORS, load_morris_design
    monkeypatch.setattr(pa, "OUT", str(tmp_path))
    monkeypatch.setattr(pa, "VALUES", {})
    X = load_morris_design()
    labels = [f"morris_{i:03d}" for i in range(len(X))]
    metric = next(m for m in pa.MORRIS_EXTRA if m not in pa.targets())
    mean, su = 10.0, 0.5
    step = 5 * math.sqrt(2 / math.pi) * su                  # the effect of one step of the first factor
    rng = np.random.default_rng(61_001)
    runs = _morris_runs(mean + step / (2 / 3) * X[:, 0], [0.0, 3.0], rng.normal(0, su, (len(X), 2)), labels)
    unit = runs["y"].mean()                                 # an output that is not a target: relative to its mean
    pa.part_morris(runs.rename(columns={"y": metric}))
    tab = pd.read_csv(tmp_path / "p1_morris.csv").set_index("factor")
    first = MORRIS_FACTORS[0][0]
    assert tab.loc[first, "above_noise"] and not tab.drop(index=first)["above_noise"].any()
    assert tab.loc[first, "mu_star"] == pytest.approx(step / unit, rel=0.25)
    noise = pd.read_csv(tmp_path / "p1_morris_noise.csv").iloc[0]
    assert noise["noise_floor"] == pytest.approx(math.sqrt(2 / math.pi) * su / unit, rel=0.15)


def test_morris_refuses_runs_that_do_not_share_their_seeds():
    from experiments.scenarios.paper1 import load_morris_design
    X = load_morris_design()
    labels = [f"morris_{i:03d}" for i in range(len(X))]
    runs = _morris_runs(np.zeros(len(X)), [0.0, 0.0], np.zeros((len(X), 2)), labels)
    runs.loc[runs.index[-1], "seed"] = 99
    with pytest.raises(ValueError, match="share their two seeds"):
        pa.part_morris(runs.rename(columns={"y": pa.MORRIS_EXTRA[0]}))


# ---------------------------------------------------------------------------
# The calibration's numbers for the text
# ---------------------------------------------------------------------------

def test_calibration_numbers_come_from_the_diagnostics(monkeypatch, tmp_path):
    """part_calibration reads diagnostics_v5's tables: the shares not ruled out, the region's width
    per parameter, the sets' spread and fit, and the emulators' check."""
    from experiments.scenarios.param_sets import ADOPTED
    monkeypatch.setattr(pa, "VALUES", {})
    pd.DataFrame({"wave": [1, 2, 3, 4], "points": [100, 100, 80, 100], "runs_per_point": 2,
                  "nroy_share_before": [1.0, 0.01, 0.002, 1e-6], "criterion": "second"}).to_csv(
        tmp_path / "nroy.csv", index=False)
    pd.DataFrame({"wave": [4, 4], "parameter": ["a", "b"], "points": 100, "q025": [0.1, 0.0],
                  "q50": [0.3, 0.5], "q975": [0.6, 1.0]}).to_csv(tmp_path / "design_ranges.csv", index=False)
    sets = [ADOPTED, "x1", "x2"]
    pd.DataFrame({"set": sets * 2, "adopted": [s == ADOPTED for s in sets] * 2, "parameter": ["a"] * 3 + ["b"] * 3,
                  "value": 0.0, "unit": [0.95, 0.97, 0.99, 0.2, 0.4, 0.6]}).to_csv(tmp_path / "sets_box.csv", index=False)
    pd.DataFrame({"set": sets, "adopted": [s == ADOPTED for s in sets], "target": "t", "runs": 8, "model": 0.0,
                  "observed": 0.0, "signed": [-2.5, 1.0, 2.9], "implausibility": [2.5, 1.0, 2.9]}).to_csv(
        tmp_path / "fit_sets.csv", index=False)
    pd.DataFrame({"target": ["t", "u"], "n": 100, "rms_standardised_error": [0.8, 1.2], "share_beyond_2": 0.05,
                  "share_beyond_3": [0.0, 0.02], "mean_standardised_error": 0.0, "r2": [0.9, -0.3],
                  "tolerance_sd": 1.0, "median_emulator_sd": 0.5}).to_csv(tmp_path / "validation.csv", index=False)
    pa.part_calibration(str(tmp_path))
    v = pa.VALUES
    assert (v["CalWaves"], v["CalPoints"], v["CalNroyWaveTwo"], v["CalNroyWaveFour"]) == ("4", "380", "1.00", "0.0001")
    assert v["CalPerDimensionPct"] == "0"                    # 1e-6 spread over 2 parameters: 0.1% each
    assert v["CalWaveFourWidthMedianPct"] == "75"            # widths 0.5 and 1.0
    assert v["CalSetsSpreadMedianPct"] == "22" and v["CalSetsAtBound"] == "1"   # spreads 0.04 and 0.4; a at the top
    assert (v["CalAdoptedImax"], v["CalSetsImaxMax"], v["CalEmuNegativeRTwo"]) == ("2.50", "2.90", "1")
    assert "CalNewSeedsWithin" not in v                      # no fit.csv: the adopted defaults' new seeds not given

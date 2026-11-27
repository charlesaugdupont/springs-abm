"""
Paper 1's figures (experiments.scenarios.paper1_figures) and the calibration
tables behind Fig 2 (experiments.calibration.diagnostics_v5): their labels cover
exactly what the model calibrates and screens, and a figure is written in the
form PLOS asks for.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.scenarios import paper1_figures as pf


@pytest.fixture
def v5():
    from experiments.calibration import history_matching as hm
    old = hm.CAMPAIGN
    hm.use_campaign("v5")
    yield hm
    hm.use_campaign(old)


def _keys(groups):
    return [k for _, items in groups for k, _ in items]


def test_figure_labels_cover_the_calibration_and_the_experiments(v5):
    from experiments.scenarios.paper1 import MORRIS_FACTORS, OBS_DESIGNS, ROUTES
    assert sorted(_keys(pf.PARAM_GROUPS)) == sorted(v5.PATHS)
    assert sorted(_keys(pf.TARGET_GROUPS)) == sorted(t.metric for t in v5.TARGETS_HM)
    assert sorted(_keys(pf.MORRIS_GROUPS)) == sorted(f[0] for f in MORRIS_FACTORS)
    assert set(pf.OBS_LABELS) == set(OBS_DESIGNS)
    assert {r for _, r in ROUTES} <= set(pf.ROUTE_LABELS)
    for groups in (pf.PARAM_GROUPS, pf.TARGET_GROUPS, pf.MORRIS_GROUPS):
        keys = _keys(groups)
        assert len(keys) == len(set(keys))


def test_design_ranges_span_the_box(v5, tmp_path, monkeypatch):
    """A wave whose points sit at the box's bounds spans 0 to 1 on the unit scale, log parameters too."""
    from experiments.calibration import diagnostics_v5 as dg
    lo = {p: v5.PARAMS[p][0] for p in v5.PATHS}
    hi = {p: v5.PARAMS[p][1] for p in v5.PATHS}
    combos = [{"gsa.design_id": f"w1_{i}", **(lo if i % 2 else hi)} for i in range(40)]
    (tmp_path / "wave1_design.json").write_text(json.dumps({"wave": 1, "combos": combos}))
    monkeypatch.setattr(v5, "OUT_DIR", str(tmp_path))
    r = dg.design_ranges_table()
    assert set(r.parameter) == set(v5.PATHS) and (r.points == 40).all()
    assert r.q025.max() == pytest.approx(0.0, abs=1e-9) and r.q975.min() == pytest.approx(1.0, abs=1e-9)


def test_a_figure_is_written_as_plos_asks(tmp_path):
    """Fig 6 from a small decision table: PDF, PNG and a 600 dpi RGB TIFF with LZW, within PLOS's size."""
    from PIL import Image
    variants = ["cpt", *pf.DECISION_LABELS]
    rows = []
    for i, v in enumerate(variants):
        for out in ["episode_care_seeking_rate"] + [f"econ_care_seeking_rate_q{q}" for q in range(1, 6)]:
            value = 0.95 if v == "no_cash_constraint" else 0.6 + 0.01 * i
            rows.append({"variant": v, "output": out, "value": value, "cpt": 0.6, "diff": value - 0.6, "se": 0.01})
    pd.DataFrame(rows).to_csv(tmp_path / "p1_decision.csv", index=False)
    pf.style()
    files = pf.fig_decisions(paper1=str(tmp_path), out=str(tmp_path / "figs"))
    assert [f.rsplit(".", 1)[1] for f in files] == ["pdf", "tif", "png"]
    with Image.open(files[1]) as im:
        assert im.mode == "RGB" and im.info.get("compression") == "tiff_lzw"
        assert im.info["dpi"][0] == pytest.approx(600)
        w_cm, h_cm = (n / 600 * 2.54 for n in im.size)
        assert 6.68 <= w_cm <= 19.05 + 1e-6 and h_cm <= 22.23 + 1e-6


def test_a_figure_larger_than_plos_allows_is_refused(tmp_path):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(pf.MAX_WIDTH + 0.5, 3))
    with pytest.raises(ValueError, match="larger than PLOS allows"):
        pf.save(fig, "too_wide", str(tmp_path))
    plt.close(fig)


def test_blocked_spread_is_what_the_robustness_rule_reads():
    """The parameter sets' rule (experiments.scenarios.param_sets) keeps an effect at 80% of the sets."""
    from experiments.scenarios.param_sets import ROBUST_SHARE
    assert ROBUST_SHARE == pytest.approx(0.8)
    assert int(np.ceil(ROBUST_SHARE * 11)) == 9                   # 9 of the 11 sets, as the Methods say

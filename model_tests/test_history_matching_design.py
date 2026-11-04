"""Rule-constrained wave designs (history_matching.sample_nroy with rules).

v5's waves 1-3 were designed on the targets alone, and the selection rules
then excluded nearly all of their NROY; from wave 4 the design samples only
the region the rules allow."""
import numpy as np
import pytest

from experiments.calibration import history_matching as hm


@pytest.fixture
def v5():
    old = hm.CAMPAIGN
    hm.use_campaign("v5")
    yield hm
    hm.use_campaign(old)


class _LinearGP:
    """Stand-in emulator: mean z + s * slope * (u_j - 0.5), SD 0."""

    def __init__(self, z, s, j, slope):
        self.z, self.s, self.j, self.slope = z, s, j, slope

    def predict(self, U, return_std=False):
        mu = self.z + self.s * self.slope * (U[:, self.j] - 0.5)
        return (mu, np.zeros(len(U))) if return_std else mu


def test_v5_designs_inside_the_selection_rules_from_wave_4():
    assert hm.CAMPAIGNS["v5"].design_rules == (4, hm.V5_SELECTION_RULES)
    assert all(c.design_rules is None for k, c in hm.CAMPAIGNS.items() if k != "v5")
    r = hm.V5_SELECTION_RULES
    assert r["aux_ranges"] == {"rota_water_fraction_post_burnin": (0.0, 0.10)}
    assert r["constraint"] is hm.v3_physical_constraint


def test_rule_constrained_design_meets_the_rules_and_the_cutoff(v5, monkeypatch):
    monkeypatch.setattr(hm, "EXPAND_N", 3000)
    monkeypatch.setattr(hm, "LOCAL_PER_POINT", 40)
    t = hm.TARGETS_HM[0]
    s = float(np.sqrt(t.sd_obs ** 2 + t.sd_md ** 2))
    ems = {t.metric: _LinearGP(t.z, s, 0, 10.0)}         # I < 3 iff |u_0 - 0.5| < 0.3

    def rules(X):                                          # 1e-4 of the box
        return (X[:, 1] < 0.01) & (X[:, 2] < 0.01)

    rng = np.random.default_rng(0)
    seeds = rng.uniform(0, 1, (6, len(hm.PATHS)))
    seeds[:, 1:3] = 0.005
    info: dict = {}
    n = 20
    U, frac = hm.sample_nroy(ems, n, seed=3, n_candidates=5000, criterion="max", rules=rules,
                             local_from=seeds, info=info)
    assert U.shape == (n, len(hm.PATHS))
    assert rules(U).all()
    assert (np.abs(U[:, 0] - 0.5) < 0.3).all()
    assert info["pool"] >= hm.POOL_PER_POINT * n            # the expansion filled the pool
    assert len(info["rounds"]) >= 2 and info["rounds"][-1]["pool"] == info["pool"]
    assert 0.0 <= frac <= 1e-3 and info["lhs_candidates"] == 5000 and info["local_points"] == 6


def test_unconstrained_design_ignores_the_new_options(v5):
    t = hm.TARGETS_HM[0]
    s = float(np.sqrt(t.sd_obs ** 2 + t.sd_md ** 2))
    ems = {t.metric: _LinearGP(t.z, s, 0, 10.0)}
    a, fa = hm.sample_nroy(ems, 15, seed=3, n_candidates=4000, criterion="max", core_fraction=0.5)
    b, fb = hm.sample_nroy(ems, 15, seed=3, n_candidates=4000, criterion="max", core_fraction=0.5, info={})
    assert np.array_equal(a, b) and fa == fb


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, U, return_std=False):
        mu = np.full(len(U), self.v)
        return (mu, np.zeros(len(U))) if return_std else mu


def _fake_selection_inputs(hm, monkeypatch, tmp_path, local):
    """select_candidates on stand-in emulators: every target on its observed
    value except the first, which needs |u_0 - 0.5| < 0.3."""
    import pandas as pd
    summary = pd.DataFrame(hm.from_unit(local), columns=hm.PATHS)
    summary["wave"] = 3
    ems = {t.metric: _Const(t.z) for t in hm.TARGETS_HM}
    t0 = hm.TARGETS_HM[0]
    ems[t0.metric] = _LinearGP(t0.z, float(np.sqrt(t0.sd_obs ** 2 + t0.sd_md ** 2)), 0, 10.0)
    monkeypatch.setattr(hm, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(hm, "load_all_waves", lambda: pd.DataFrame())
    monkeypatch.setattr(hm, "summarise", lambda df: summary)
    monkeypatch.setattr(hm, "fit_emulators", lambda s, n_jobs=None: ems)
    monkeypatch.setattr(hm, "fit_imported_emulator", lambda df, s: _Const(0.0))
    monkeypatch.setattr(hm, "fit_aux_emulator", lambda df, s, m: _Const(0.0))
    monkeypatch.setattr(hm, "EXPAND_N", 3000)
    monkeypatch.setattr(hm, "LOCAL_PER_POINT", 40)
    monkeypatch.setattr(hm, "SELECT_POOL", 100)


def _tiny_region(df):                                    # 4e-6 of the box, on the unit scale
    U = df.to_numpy()
    import experiments.calibration.history_matching as hm
    U = hm.to_unit(U)
    return (U[:, 1] < 0.002) & (U[:, 2] < 0.002)


def test_selection_finds_a_rule_region_the_lhs_misses(v5, monkeypatch, tmp_path):
    import json
    rng = np.random.default_rng(1)
    local = rng.uniform(0.25, 0.75, (6, len(v5.PATHS)))
    local[:, 1:3] = 0.001
    _fake_selection_inputs(v5, monkeypatch, tmp_path, local)
    kw = dict(n=2, n_candidates=5000, refocus_from_wave=2, constraint=_tiny_region, min_separation=0.01)
    with pytest.raises(SystemExit):
        v5.select_candidates(**kw)                       # the LHS alone: nothing admissible
    out = v5.select_candidates(local_from_wave=2, **kw)
    assert [c["gsa.design_id"] for c in out] == ["e0", "e1"]
    U = v5.to_unit(np.array([[c[p] for p in v5.PATHS] for c in out]))
    assert ((U[:, 1] < 0.002) & (U[:, 2] < 0.002)).all() and (np.abs(U[:, 0] - 0.5) < 0.3).all()
    with open(tmp_path / "emulator_candidates.json") as f:
        assert len(json.load(f)) == 2

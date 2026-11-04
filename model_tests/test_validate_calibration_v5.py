"""validate_calibration --campaign v5: the calibrated values it applies."""
import json

import pytest

from experiments.calibration import validate_calibration as vc


def test_values_are_rounded_and_tied_paths_split(tmp_path):
    c = {"gsa.design_id": "e7", "source": "x", "a.b": 49.0897, "c|d": 0.000123456, "e": 0.0, "f": 180.0}
    p = tmp_path / "one.json"
    p.write_text(json.dumps([c]))
    v = vc.calibrated_values(str(p))
    assert v == {"a.b": 49.1, "c": 0.000123, "d": 0.000123, "e": 0.0, "f": 180.0}
    q = tmp_path / "dict.json"
    q.write_text(json.dumps({"a.b": 49.0897}))
    assert vc.calibrated_values(str(q), sig=2) == {"a.b": 49.0}
    r = tmp_path / "two.json"
    r.write_text(json.dumps([c, c]))
    with pytest.raises(SystemExit):
        vc.calibrated_values(str(r))


def test_v5_validation_runs_v5s_base():
    from experiments.calibration.history_matching import V5_BASE_OVERRIDES
    grid, agents, base, out_dir = vc._settlement_campaign("v5")
    assert base == V5_BASE_OVERRIDES and out_dir.endswith("calibration_validation_v5")
    assert all(len(d) == 5 for d in vc.DHS_DIARRHOEA_2WK_BY_QUINTILE.values())

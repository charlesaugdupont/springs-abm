"""
run_sweep's checkpoint/resume (SweepSpec.checkpoint_every, .resume), used by
the long history-matching waves (experiments/calibration/history_matching.py,
campaign v3): an interrupted wave restarts where it stopped, and results from
a different design in the same folder are not mixed in.
"""
from __future__ import annotations

import time

import pandas as pd

from experiments.orchestrator import SweepSpec, run_sweep

GRID_ID = "7d9ce7c720a6"


def stamp_metrics(model) -> dict:
    """Top-level (picklable): when the run finished, to tell re-runs apart."""
    return {"finished_at": time.time()}


def _spec(tmp_path) -> SweepSpec:
    return SweepSpec(name="resume", grid_id=GRID_ID, params=[], metrics_fn=stamp_metrics, reps=1,
                     steps=2, agents=200, output_dir=str(tmp_path), n_cores=1,
                     checkpoint_every=1, resume=True)


def _combo(v):
    return {"gsa.design_id": f"d{v}", "steering_parameters.household_exposure_sdlog": v}


def test_resume_skips_finished_runs_and_drops_rows_of_another_design(tmp_path):
    first = run_sweep(_spec(tmp_path), [_combo(0.1), _combo(0.2)])
    assert len(first) == 2
    again = run_sweep(_spec(tmp_path), [_combo(0.1), _combo(0.2), _combo(0.3)])
    assert len(again) == 3
    old = first.set_index("run_id")["finished_at"]
    new = again.set_index("run_id")["finished_at"]
    for rid, t in old.items():
        assert new[rid] == t, "a finished run was simulated again"
    on_disk = pd.read_parquet(tmp_path / "resume" / "results.parquet")
    assert set(on_disk["run_id"]) == set(again["run_id"])
    # a different design in the same folder: rows that are not part of it go
    only = run_sweep(_spec(tmp_path), [_combo(0.3)])
    assert list(only["gsa.design_id"]) == ["d0.3"]

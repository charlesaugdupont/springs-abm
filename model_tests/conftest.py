"""
Shared fixtures for the model-core tests.

These tests exercise abm/ directly rather than through the webapp API, so they
build small real models (real Akuse grid, few hundred agents, few days) instead
of mocking. A model at this scale initialises and steps in well under a second,
which keeps the suite fast enough to run on every change.
"""
from __future__ import annotations

import pytest
import torch

from config import SVEIRCONFIG, SVEIRConfig
from abm.model.initialize_model import SVEIRModel
from abm.utils.rng import set_global_seed

GRID_ID = "7d9ce7c720a6"


@pytest.fixture(autouse=True)
def _single_threaded():
    """Pin torch to one thread.

    Multi-threaded CPU float reductions are not bit-reproducible, and several
    tests here assert exact equality between two runs. Same rationale as
    experiments/perf/regression_check.py.
    """
    torch.set_num_threads(1)


def build_model(
    *,
    seed: int = 23,
    agents: int = 600,
    steps: int = 30,
    pathogens: list[str] | None = None,
    overrides: dict | None = None,
    run: bool = True,
    tmp_path=None,
) -> SVEIRModel:
    """Builds (and by default runs) a small model.

    `pathogens` restricts to a subset by name; `overrides` is a dict of
    orchestrator-style dot-paths (e.g. "pathogens[crypto].water_dose_scale")
    applied after the subset is taken.
    """
    from experiments.orchestrator import set_param

    set_global_seed(seed)
    cfg: SVEIRConfig = SVEIRCONFIG.model_copy(deep=True)
    cfg.number_agents = agents
    cfg.step_target = steps
    cfg.seed = seed
    cfg.spatial_creation_args.grid_id = GRID_ID

    if pathogens is not None:
        selected = [p for p in cfg.pathogens if p.name in set(pathogens)]
        assert len(selected) == len(set(pathogens)), f"unknown pathogen in {pathogens}"
        cfg.pathogens = selected

    for path, value in (overrides or {}).items():
        set_param(cfg, path, value)

    root = str(tmp_path) if tmp_path is not None else "/tmp/springs_abm_tests"
    model = SVEIRModel(model_identifier=f"_test_{seed}", root_path=root)
    model.set_model_parameters(**cfg.model_dump())
    model.initialize_model(verbose=False)
    if run:
        model.run()
    return model


@pytest.fixture
def make_model(tmp_path):
    """Factory fixture wrapping build_model with a per-test temp output dir."""
    def _make(**kwargs):
        kwargs.setdefault("tmp_path", tmp_path)
        return build_model(**kwargs)
    return _make


@pytest.fixture
def crypto_model(make_model):
    """A short Cryptosporidium-only run."""
    return make_model(pathogens=["crypto"], steps=40)

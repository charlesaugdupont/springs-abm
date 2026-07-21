# experiments/orchestrator.py
"""
Shared sweep-execution orchestrator for SPRINGS-ABM experiments.

Provides
--------
- get_param / set_param : dot-path get/set for SVEIRConfig (handles nested
  objects and "pathogens[name].attr" paths). Generalized from the
  _get_param/_set_param helpers in sensitivity.py so every experiment script
  can reuse the same logic instead of reimplementing it.
- SweepParam / SweepSpec : a declarative description of a parameter sweep.
- run_sweep()            : parallel replicate execution -> tidy long-format
  Parquet output (one row per run x metric), plus an optional companion
  time-series Parquet (one row per run x day) for complex-systems analyses
  that need more than endpoint summary statistics.
- load_results / load_timeseries : convenience readers for the above.

Design intent
-------------
Every experiment script (vaccination, shocks, care-seeking, ...) should be a
short, declarative file: which config parameters to sweep, over what values,
and what `metrics_fn` to compute from a finished SVEIRModel. All
multiprocessing / replicate / seeding / IO plumbing lives here, written once.

IMPORTANT: `metrics_fn` (passed via SweepSpec) is sent to worker processes
via pickling, so it MUST be a plain, importable, module-level function -
not a lambda or a closure defined inside another function.
"""
from __future__ import annotations

import io
import os
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from config import SVEIRCONFIG, SVEIRConfig
from abm.model.initialize_model import SVEIRModel
from abm.utils.rng import set_global_seed


# ---------------------------------------------------------------------------
# Dot-path config access
# ---------------------------------------------------------------------------

def set_param(config: SVEIRConfig, path: str, value: Any) -> None:
    """Sets a (possibly nested / pathogen-scoped) config value in place.

    Supports:
        "steering_parameters.cost_of_care"
        "pathogens[rota].vaccination_rate"
        "number_agents"
    """
    if path.startswith("pathogens["):
        bracket_end = path.index("]")
        p_name = path[len("pathogens["):bracket_end]
        attr = path[bracket_end + 2:]
        for p in config.pathogens:
            if p.name == p_name:
                setattr(p, attr, value)
                return
        raise ValueError(f"Pathogen '{p_name}' not found in config.")
    parts = path.split(".", 1)
    if len(parts) == 2:
        setattr(getattr(config, parts[0]), parts[1], value)
    else:
        setattr(config, path, value)


def get_param(config: SVEIRConfig, path: str) -> Any:
    if path.startswith("pathogens["):
        bracket_end = path.index("]")
        p_name = path[len("pathogens["):bracket_end]
        attr = path[bracket_end + 2:]
        for p in config.pathogens:
            if p.name == p_name:
                return getattr(p, attr)
        raise ValueError(f"Pathogen '{p_name}' not found in config.")
    parts = path.split(".", 1)
    if len(parts) == 2:
        return getattr(getattr(config, parts[0]), parts[1])
    return getattr(config, path)


# ---------------------------------------------------------------------------
# Sweep specification
# ---------------------------------------------------------------------------

@dataclass
class SweepParam:
    """One swept axis: a config dot-path and the values to sweep it over."""
    path: str
    values: Sequence[Any]
    label: Optional[str] = None  # human-readable name for plotting; defaults to `path`

    def display_name(self) -> str:
        return self.label or self.path


@dataclass
class SweepSpec:
    """Full declaration of a sweep. Build one of these and pass it to run_sweep()."""
    name: str
    grid_id: str
    params: List[SweepParam]
    metrics_fn: Callable[[SVEIRModel], Dict[str, Any]]
    reps: int = 20
    steps: int = 250
    agents: int = 4000
    base_overrides: Dict[str, Any] = field(default_factory=dict)
    record_timeseries: bool = False
    output_dir: str = os.path.join("experiments", "outputs")
    n_cores: Optional[int] = None
    base_seed: Optional[int] = None

    def output_path(self) -> Path:
        return Path(self.output_dir) / self.name


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _build_config(spec: SweepSpec, combo: Dict[str, Any], seed: int) -> SVEIRConfig:
    cfg: SVEIRConfig = SVEIRCONFIG.model_copy(deep=True)
    cfg.step_target = spec.steps
    cfg.number_agents = spec.agents
    cfg.seed = seed
    cfg.spatial_creation_args.grid_id = spec.grid_id

    for path, value in spec.base_overrides.items():
        set_param(cfg, path, value)
    for path, value in combo.items():
        set_param(cfg, path, value)
    return cfg


def _run_one(task) -> Optional[Dict[str, Any]]:
    """Top-level (picklable) worker: builds config, runs one replicate, returns a tidy record."""
    spec, combo, rep, run_id = task
    seed = (spec.base_seed if spec.base_seed is not None else SVEIRCONFIG.seed) + rep
    set_global_seed(seed)
    cfg = _build_config(spec, combo, seed)

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SVEIRModel(
                model_identifier=f"_{spec.name}_{run_id}",
                root_path=os.path.join(spec.output_dir, spec.name, "_tmp"),
            )
            model.set_model_parameters(**cfg.model_dump())
            model.initialize_model(verbose=False)
            model.run()

        metrics = spec.metrics_fn(model)
        record = {"run_id": run_id, "rep": rep, "seed": seed, **combo, **metrics}

        timeseries = None
        if spec.record_timeseries:
            timeseries = []
            for pname, series in model.u5_prevalence_history.items():
                for day, val in enumerate(series):
                    timeseries.append({
                        "run_id": run_id, "rep": rep, "pathogen": pname,
                        "day": day, "u5_prevalence": val,
                    })
        return {"record": record, "timeseries": timeseries}

    except Exception:
        print(f"\n--- ERROR in run {run_id} ---")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------

def _combo_id(combo: Dict[str, Any], rep: int) -> str:
    parts = []
    for k, v in combo.items():
        short = k.split(".")[-1].split("]")[-1] or k
        parts.append(f"{short}{v:.4g}" if isinstance(v, float) else f"{short}{v}")
    parts.append(f"rep{rep}")
    return "_".join(parts)


def run_sweep(spec: SweepSpec, combos: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    """Runs the full sweep in parallel, saves a tidy long-format Parquet file,
    and returns the resulting DataFrame.

    By default, combos are built as the full factorial product of
    spec.params (every combination of every swept value) - what you want for
    a phase-map-style grid sweep.

    Pass `combos` explicitly to bypass this and run an arbitrary list of
    parameter-value dicts instead - e.g. Latin-Hypercube-sampled points for
    a calibration search, where a factorial grid isn't the right design.
    In that case spec.params can be left empty; it's only used for the
    factorial default and print header.

    If spec.record_timeseries is True, a second Parquet file
    (<name>/timeseries.parquet) is also written with per-day u5 prevalence.
    """
    out_dir = spec.output_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    explicit_combos = combos is not None
    if combos is None:
        param_names = [p.path for p in spec.params]
        value_lists = [p.values for p in spec.params]
        combos = [dict(zip(param_names, vals)) for vals in product(*value_lists)]

    tasks = []
    for combo in combos:
        for rep in range(spec.reps):
            run_id = _combo_id(combo, rep)
            tasks.append((spec, combo, rep, run_id))

    n_cores = spec.n_cores or max(1, min(6, cpu_count()))
    total = len(tasks)

    print(f"\n--- Sweep: {spec.name} ---")
    if explicit_combos:
        print(f"  Design           : {len(combos)} explicitly-provided parameter combinations "
              f"(non-factorial, e.g. sampled)")
    else:
        print(f"  Axes             : {[p.display_name() for p in spec.params]}")
        for p in spec.params:
            print(f"    {p.display_name():<30}: {list(p.values)}")
        print(f"  Combinations     : {len(combos)}")
    print(f"  Replicates       : {spec.reps}")
    print(f"  Total runs       : {total}")
    print(f"  Workers          : {n_cores}")
    print(f"  Steps / Agents   : {spec.steps} / {spec.agents}\n")

    t0 = time.time()
    records, timeseries_rows = [], []

    with Pool(processes=n_cores) as pool:
        for i, result in enumerate(pool.imap(_run_one, tasks), 1):
            if result is not None:
                records.append(result["record"])
                if result["timeseries"]:
                    timeseries_rows.extend(result["timeseries"])
            elapsed = time.time() - t0
            eta = (elapsed / i) * (total - i)
            print(f"  [{i:>4}/{total}]  elapsed={elapsed:6.0f}s  ETA={eta:6.0f}s", end="\r")

    print(f"\n  Sweep complete in {time.time() - t0:.0f}s  "
          f"({len(records)}/{total} runs succeeded)")

    df = pd.DataFrame.from_records(records)
    results_path = out_dir / "results.parquet"
    df.to_parquet(results_path, index=False)
    print(f"  Results saved -> {results_path}")

    if spec.record_timeseries and timeseries_rows:
        ts_df = pd.DataFrame.from_records(timeseries_rows)
        ts_path = out_dir / "timeseries.parquet"
        ts_df.to_parquet(ts_path, index=False)
        print(f"  Time series saved -> {ts_path}")

    return df


def load_results(spec_name: str, output_dir: str = os.path.join("experiments", "outputs")) -> pd.DataFrame:
    return pd.read_parquet(Path(output_dir) / spec_name / "results.parquet")


def load_timeseries(spec_name: str, output_dir: str = os.path.join("experiments", "outputs")) -> pd.DataFrame:
    return pd.read_parquet(Path(output_dir) / spec_name / "timeseries.parquet")
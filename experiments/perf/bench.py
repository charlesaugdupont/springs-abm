# experiments/perf/bench.py
"""
Timing benchmark for the performance-optimization pass.

Two modes:
  runs  - times N sequential single model runs (per-run cost; captures the
          effect of the in-model optimizations A2-A5).
  sweep - times a parallel sweep through experiments.orchestrator.run_sweep
          (throughput; captures the effect of thread-pinning A1).

Usage
-----
    python -m experiments.perf.bench --mode runs  --n 5   --agents 2000 --steps 100
    python -m experiments.perf.bench --mode sweep --runs 32 --agents 2000 --steps 100 --workers 8

To get "before" numbers, copy this file onto the pre-optimization commit and run
the same command (it imports nothing that the optimizations remove).
"""
from __future__ import annotations

import argparse
import time

import torch

from config import SVEIRCONFIG
from abm.model.initialize_model import SVEIRModel
from abm.utils.rng import set_global_seed
from experiments.orchestrator import SweepSpec, run_sweep
from experiments.metrics import epidemic_metrics

GRID_ID = "7d9ce7c720a6"


def _one_run(seed: int, agents: int, steps: int) -> float:
    set_global_seed(seed)
    cfg = SVEIRCONFIG.model_copy(deep=True)
    cfg.number_agents = agents
    cfg.step_target = steps
    cfg.seed = seed
    cfg.spatial_creation_args.grid_id = GRID_ID
    model = SVEIRModel(model_identifier=f"_bench_{seed}",
                       root_path="experiments/outputs/perf/_tmp")
    model.set_model_parameters(**cfg.model_dump())
    model.initialize_model(verbose=False)
    t0 = time.perf_counter()
    model.run()
    return time.perf_counter() - t0


def bench_runs(n: int, agents: int, steps: int):
    torch.set_num_threads(1)
    times = [_one_run(23 + i, agents, steps) for i in range(n)]
    times.sort()
    print(f"\n--- runs mode: {n} runs, {agents} agents, {steps} days, threads=1 ---")
    print(f"  per-run: min={times[0]:.2f}s  median={times[n // 2]:.2f}s  "
          f"mean={sum(times) / n:.2f}s  total={sum(times):.1f}s")


def bench_sweep(runs: int, agents: int, steps: int, workers: int):
    spec = SweepSpec(
        name="_bench_sweep", grid_id=GRID_ID, params=[],
        metrics_fn=epidemic_metrics, reps=runs, steps=steps, agents=agents,
        output_dir="experiments/outputs/perf", n_cores=workers,
    )
    t0 = time.perf_counter()
    df = run_sweep(spec, combos=[{}])
    elapsed = time.perf_counter() - t0
    print(f"\n--- sweep mode: {len(df)}/{runs} runs, {workers} workers, "
          f"{agents} agents, {steps} days ---")
    print(f"  total={elapsed:.1f}s  throughput={len(df) / elapsed:.2f} runs/s")


def main():
    parser = argparse.ArgumentParser(description="Perf benchmark for the optimization pass.")
    parser.add_argument("--mode", choices=["runs", "sweep"], default="runs")
    parser.add_argument("--n", type=int, default=5, help="runs mode: number of sequential runs")
    parser.add_argument("--runs", type=int, default=32, help="sweep mode: number of runs (reps)")
    parser.add_argument("--agents", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "runs":
        bench_runs(args.n, args.agents, args.steps)
    else:
        bench_sweep(args.runs, args.agents, args.steps, args.workers)


if __name__ == "__main__":
    main()

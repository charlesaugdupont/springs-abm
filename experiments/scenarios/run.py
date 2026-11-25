# experiments/scenarios/run.py
"""
Runs one part of the v5 experiments (experiments.scenarios.designs), resumable:
an interrupted run picks up where it stopped (results.parquet is checkpointed
every 4 runs).

    python -m experiments.scenarios.run 3a --cores 2
    python -m experiments.scenarios.run 1a --dry-run          # list the scenarios
    python -m experiments.scenarios.run 1a --cores 8 --reps 8

Outputs: experiments/outputs/experiments_v5/<design name>/results.parquet
"""
from __future__ import annotations

import argparse

from experiments.orchestrator import run_sweep
from experiments.scenarios.common import experiment_spec
from experiments.scenarios.designs import DESIGNS, LABEL
from experiments.scenarios.hooks import apply_hooks


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("part", choices=sorted(DESIGNS))
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None, help="override the design's run length (tests)")
    ap.add_argument("--dry-run", action="store_true", help="list the scenarios and exit")
    ap.add_argument("--checkpoint-every", type=int, default=4,
                    help="save results.parquet after this many runs (1: an interrupted part loses only runs in progress)")
    args = ap.parse_args(argv)

    design = DESIGNS[args.part]
    combos = design.combos()
    labels = [c[LABEL] for c in combos]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{args.part}: duplicate scenario labels")
    print(f"{args.part}: {design.name} - {design.description}")
    from experiments.orchestrator import REPS_KEY
    n_runs = sum(int(c.get(REPS_KEY, args.reps)) for c in combos)
    seeds = sorted({int(c.get(REPS_KEY, args.reps)) for c in combos})
    print(f"  {len(combos)} scenarios x {'/'.join(map(str, seeds))} seeds = {n_runs} runs, "
          f"{args.steps or design.steps} days")
    if args.dry_run:
        for c in combos:
            rest = {k: v for k, v in c.items() if k not in (LABEL, REPS_KEY)}
            print(f"  {c[LABEL]:<40} x{c.get(REPS_KEY, args.reps):<3} {rest}")
        return
    callback = design.step_callback or (apply_hooks if design.hooks else None)
    spec = experiment_spec(design.name, reps=args.reps, steps=args.steps or design.steps, n_cores=args.cores,
                           step_callback=callback, metrics_fn=design.metrics_fn,
                           checkpoint_every=args.checkpoint_every)
    run_sweep(spec, combos)


if __name__ == "__main__":
    main()

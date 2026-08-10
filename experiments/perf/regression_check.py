# experiments/perf/regression_check.py
"""
Bit-identical regression harness for the performance-optimization pass.
=======================================================================

Why this exists
---------------
The perf pass optimizes the model's hot path under a strict constraint: the
outputs must stay *byte-for-byte identical*. The model runs off a single global
RNG stream (abm/utils/rng.set_global_seed seeds torch/numpy/random together), so
any change that alters the order, size, or count of random draws diverges the
whole trajectory. This harness is the guard rail: it captures a full fingerprint
of a run over a fixed set of seeds and asserts exact equality against a saved
golden reference, so any optimization that accidentally perturbs behavior is
caught immediately.

Determinism
-----------
Torch threads are pinned to 1 (torch.set_num_threads(1)). Multi-threaded CPU
float reductions are not guaranteed to be bit-reproducible, so pinning is a
prerequisite for a meaningful byte-for-byte comparison (and is itself one of the
optimizations in this pass, applied to the sweep workers in
experiments/orchestrator.py). Everything here always runs single-threaded.

Representative config
---------------------
A modest scale (2000 agents / 100 days) that still exercises every code path:
both pathogens enabled, rotavirus vaccination on, water-contamination shocks and
the waterborne route active, and the social-movement path. If a path isn't
exercised, a change that breaks it wouldn't show up in the fingerprint.

Usage
-----
    # Capture the reference from the PRE-optimization code (do this first):
    python -m experiments.perf.regression_check --save-golden

    # After each optimization commit, assert nothing changed:
    python -m experiments.perf.regression_check

    # Prove bit-identity is even achievable (run each seed twice, compare):
    python -m experiments.perf.regression_check --determinism
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Pin threads BEFORE any model work so runs are deterministic and comparable.
torch.set_num_threads(1)

from config import SVEIRCONFIG
from abm.constants import AgentPropertyKeys
from abm.model.initialize_model import SVEIRModel
from abm.pathogens.campylobacter import Campylobacter
from abm.systems.care_seeking import CareSeekingSystem
from abm.utils.rng import set_global_seed

DEFAULT_GRID_ID = "7d9ce7c720a6"
DEFAULT_SEEDS = [23, 24, 25]
DEFAULT_AGENTS = 2000
DEFAULT_STEPS = 100
GOLDEN_PATH = os.path.join("experiments", "outputs", "perf", "golden.pt")
_TMP_ROOT = os.path.join("experiments", "outputs", "perf", "_tmp")


# ---------------------------------------------------------------------------
# Running + fingerprinting
# ---------------------------------------------------------------------------

def run_model(seed: int, agents: int, steps: int, grid_id: str) -> SVEIRModel:
    """Builds and runs one model exactly the way the orchestrator does."""
    set_global_seed(seed)
    cfg = SVEIRCONFIG.model_copy(deep=True)
    cfg.number_agents = agents
    cfg.step_target = steps
    cfg.seed = seed
    cfg.spatial_creation_args.grid_id = grid_id

    model = SVEIRModel(model_identifier=f"_perfcheck_{seed}", root_path=_TMP_ROOT)
    model.set_model_parameters(**cfg.model_dump())
    model.initialize_model(verbose=False)
    model.run()
    return model


def fingerprint(model: SVEIRModel) -> dict:
    """A comprehensive, comparable snapshot of a finished run.

    Captures the full daily time series, every final agent-state array, the
    per-pathogen final compartment + reinfection counts, and the care-seeking
    and campylobacter-route counters. If any RNG draw shifts, at least one of
    these diverges.
    """
    fp: dict = {}

    # Daily time series
    fp["u5_prevalence_history"] = {k: list(v) for k, v in model.u5_prevalence_history.items()}
    fp["infection_incidence"] = list(model.infection_incidence)

    # Final per-agent state arrays
    for k, arr in model.get_final_agent_states().items():
        fp[f"state_{k}"] = np.asarray(arr).copy()

    # Per-pathogen final status + cumulative reinfection counts
    for p in model.pathogens:
        fp[f"status_{p.name}"] = model.graph.ndata[AgentPropertyKeys.status(p.name)].cpu().clone()
        fp[f"num_infections_{p.name}"] = (
            model.graph.ndata[AgentPropertyKeys.num_infections(p.name)].cpu().clone()
        )

    # Care-seeking counters
    for s in model.systems:
        if isinstance(s, CareSeekingSystem):
            fp["cs_decisions_faced"] = s.decisions_faced
            fp["cs_care_sought"] = s.care_sought
            fp["cs_could_not_afford"] = s.could_not_afford
            fp["cs_total_episodes"] = s.total_episodes
            fp["cs_episodes_with_care_sought"] = s.episodes_with_care_sought

    # Campylobacter lifetime route counters
    for p in model.pathogens:
        if isinstance(p, Campylobacter):
            fp["campy_total_zoonotic"] = p.total_zoonotic
            fp["campy_total_fecal_oral"] = p.total_fecal_oral
            fp["campy_total_food_borne"] = p.total_food_borne

    return fp


def build_fingerprints(seeds, agents, steps, grid_id) -> dict:
    return {seed: fingerprint(run_model(seed, agents, steps, grid_id)) for seed in seeds}


# ---------------------------------------------------------------------------
# Exact comparison
# ---------------------------------------------------------------------------

def _exact_equal(a, b, path: str, mismatches: list):
    """Recursively assert exact (byte-for-byte) equality; log every mismatch."""
    if type(a) is not type(b):
        # tolerate int/float only when the values are exactly equal below;
        # otherwise a type change is itself a divergence worth flagging.
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            mismatches.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
            return

    if isinstance(a, dict):
        if a.keys() != b.keys():
            mismatches.append(f"{path}: keys {sorted(a.keys())} != {sorted(b.keys())}")
            return
        for k in a:
            _exact_equal(a[k], b[k], f"{path}.{k}", mismatches)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            mismatches.append(f"{path}: len {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _exact_equal(x, y, f"{path}[{i}]", mismatches)
    elif isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            n = int((a != b).sum().item())
            mismatches.append(f"{path}: tensor differs in {n} element(s)")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            n = int((a != b).sum())
            mismatches.append(f"{path}: ndarray differs in {n} element(s)")
    else:
        if a != b:
            mismatches.append(f"{path}: {a!r} != {b!r}")


def compare(golden: dict, current: dict) -> list:
    mismatches: list = []
    if golden.keys() != current.keys():
        mismatches.append(f"seed sets differ: {sorted(golden)} != {sorted(current)}")
        return mismatches
    for seed in golden:
        _exact_equal(golden[seed], current[seed], f"seed{seed}", mismatches)
    return mismatches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--save-golden", action="store_true",
                        help="Run and save the reference fingerprint to --golden. "
                             "Do this on the pre-optimization code.")
    parser.add_argument("--determinism", action="store_true",
                        help="Run each seed twice and assert the two are identical "
                             "(proves single-threaded bit-reproducibility).")
    parser.add_argument("--golden", default=GOLDEN_PATH, help=f"Golden file path (default: {GOLDEN_PATH})")
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--grid-id", default=DEFAULT_GRID_ID)
    args = parser.parse_args()

    print(f"Config: {len(args.seeds)} seeds {args.seeds}, {args.agents} agents, "
          f"{args.steps} days, grid {args.grid_id}, torch_threads={torch.get_num_threads()}")

    if args.determinism:
        print("\n--- Determinism check (each seed run twice) ---")
        a = build_fingerprints(args.seeds, args.agents, args.steps, args.grid_id)
        b = build_fingerprints(args.seeds, args.agents, args.steps, args.grid_id)
        mismatches = compare(a, b)
        if mismatches:
            print(f"NON-DETERMINISTIC: {len(mismatches)} mismatch(es):")
            for m in mismatches[:20]:
                print(f"  {m}")
            sys.exit(1)
        print("OK - runs are bit-reproducible with threads pinned.")
        return

    if args.save_golden:
        os.makedirs(os.path.dirname(args.golden), exist_ok=True)
        golden = build_fingerprints(args.seeds, args.agents, args.steps, args.grid_id)
        torch.save({"meta": vars(args), "fingerprints": golden}, args.golden)
        print(f"\nGolden saved -> {args.golden}")
        return

    # Default: compare current code against the saved golden.
    if not os.path.exists(args.golden):
        print(f"\nNo golden at {args.golden}. Run with --save-golden first "
              f"(on the pre-optimization code).")
        sys.exit(2)
    saved = torch.load(args.golden, weights_only=False)
    golden = saved["fingerprints"]
    current = build_fingerprints(args.seeds, args.agents, args.steps, args.grid_id)
    mismatches = compare(golden, current)
    if mismatches:
        print(f"\nMISMATCH vs golden: {len(mismatches)} difference(s):")
        for m in mismatches[:30]:
            print(f"  {m}")
        sys.exit(1)
    print("\nOK - byte-for-byte identical to golden.")


if __name__ == "__main__":
    main()

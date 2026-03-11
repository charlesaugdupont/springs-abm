# abm/utils/rng.py
from __future__ import annotations

import random
import numpy as np
import torch

GRID_GENERATION_SEED = 42

_np_rng: np.random.Generator | None = None
_current_seed: int | None = None

def set_global_seed(seed: int) -> None:
    """Set a single global seed for Python, NumPy, and PyTorch RNGs."""
    global _np_rng, _current_seed
    seed = int(seed)
    _current_seed = seed

    random.seed(seed)
    np.random.seed(seed)
    _np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

def get_np_rng() -> np.random.Generator:
    """Return the global NumPy Generator, initialising with a default seed if needed."""
    global _np_rng
    if _np_rng is None:
        set_global_seed(GRID_GENERATION_SEED)
    return _np_rng

def get_current_seed() -> int | None:
    """Return the last seed passed to set_global_seed, or None if unset."""
    return _current_seed
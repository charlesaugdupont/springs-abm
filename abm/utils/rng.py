# abm/utils/rng.py
from __future__ import annotations

import random
import numpy as np
import torch

_np_rng: np.random.Generator | None = None
_current_seed: int | None = None


def set_global_seed(seed: int) -> None:
    """
    Set a single global seed for Python's random, NumPy, and PyTorch, and
    initialise a NumPy Generator instance used throughout the project.
    """
    global _np_rng, _current_seed
    _current_seed = int(seed)

    random.seed(_current_seed)
    np.random.seed(_current_seed)  # for any legacy np.random.* calls
    torch.manual_seed(_current_seed)

    _np_rng = np.random.default_rng(_current_seed)


def get_np_rng() -> np.random.Generator:
    """
    Get the global NumPy Generator. If none exists yet, initialise it with a
    default seed (42).
    """
    global _np_rng
    if _np_rng is None:
        set_global_seed(42)
    return _np_rng


def get_current_seed() -> int | None:
    """Return the last seed passed to set_global_seed, or None if unset."""
    return _current_seed
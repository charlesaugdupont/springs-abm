# abm/pathogens/registry.py
"""
Pathogen registry: the single place that maps a pathogen name to the classes
implementing it.

Before this module, the same name -> class knowledge was hardcoded in three
places that had to be kept in agreement by hand (config.SVEIRConfig's
`pathogens` validator, and two dicts in
SVEIRModel._initialize_pathogens_and_systems). Adding a pathogen meant editing
all three and hoping none was missed.

The config half of the mapping lives in config.PATHOGEN_CONFIG_TYPES rather
than here, because config.py cannot import from abm.pathogens (the pathogen
modules import config, so it would be circular). This module composes the two
halves and is what model construction uses.
"""
from __future__ import annotations

from typing import Type

from config import PATHOGEN_CONFIG_TYPES, PathogenConfig

from .pathogen import Pathogen
from .rotavirus import Rotavirus
from .campylobacter import Campylobacter
from .cryptosporidium import Cryptosporidium

#: Pathogen name -> implementing Pathogen subclass.
PATHOGEN_CLASSES: dict[str, Type[Pathogen]] = {
    "rota": Rotavirus,
    "campy": Campylobacter,
    "crypto": Cryptosporidium,
}


def pathogen_names() -> tuple[str, ...]:
    """Every pathogen name the model can instantiate."""
    return tuple(PATHOGEN_CLASSES)


def is_implemented(name: str) -> bool:
    return name in PATHOGEN_CLASSES


def pathogen_class(name: str) -> Type[Pathogen]:
    if name not in PATHOGEN_CLASSES:
        raise KeyError(
            f"Pathogen '{name}' has no implementation. Known: {sorted(PATHOGEN_CLASSES)}"
        )
    return PATHOGEN_CLASSES[name]


def pathogen_config_class(name: str) -> Type[PathogenConfig]:
    if name not in PATHOGEN_CONFIG_TYPES:
        raise KeyError(
            f"Pathogen '{name}' has no config class. Known: {sorted(PATHOGEN_CONFIG_TYPES)}"
        )
    return PATHOGEN_CONFIG_TYPES[name]


def build_pathogen(p_config: PathogenConfig, global_params, device) -> Pathogen:
    """Instantiates the pathogen for `p_config`, re-typing the config first.

    The re-typing (dumping and reconstructing through the registered config
    class) reproduces the behaviour of the original
    _initialize_pathogens_and_systems: a config that arrived as the Union's
    first matching member is coerced to the class the pathogen expects.
    """
    config_cls = pathogen_config_class(p_config.name)
    typed_config = config_cls(**p_config.model_dump())
    return pathogen_class(p_config.name)(typed_config, global_params, device)


#: Consistency guard: every implemented pathogen must have a config class and
#: vice versa. Catches a half-finished pathogen addition at import time rather
#: than with a KeyError deep inside model construction.
_missing_config = set(PATHOGEN_CLASSES) - set(PATHOGEN_CONFIG_TYPES)
_missing_class = set(PATHOGEN_CONFIG_TYPES) - set(PATHOGEN_CLASSES)
if _missing_config or _missing_class:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "Pathogen registry is inconsistent. "
        f"Implemented but unconfigured: {sorted(_missing_config)}; "
        f"configured but unimplemented: {sorted(_missing_class)}."
    )

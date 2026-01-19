"""Documentation about .."""
import logging

from . import config

#from .model import initialize_model, step
from .model.initialize_model import SVEIRModel

__all__ = ['config', 'SVEIRModel']

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Team Atlas"
__email__ = "m.grootes@esciencecenter.nl"
__version__ = "0.1.0"

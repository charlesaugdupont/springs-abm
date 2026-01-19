"""Documentation about dgl_ptm."""
import logging

from dgl_ptm import config

#from dgl_ptm.model import initialize_model, step
from dgl_ptm.model.initialize_model import SVEIRModel

__all__ = ['config', 'SVEIRModel']

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Team Atlas"
__email__ = "m.grootes@esciencecenter.nl"
__version__ = "0.1.0"

"""

Functions for training.

"""
from . import controllers
from . import evaluator
from . import metric_tracker
from . import assembly
from . import serialization
from . import routines

from .assembly import assemble_for_training
from .routines import setup_and_train, setup_training, train_model, test_model, SetupParams
from .serialization import load_checkpoint, load_checkpoint_from_cwd, load_model_from_cwd

__all__ = ["assemble_for_training", "setup_and_train", "setup_training", "train_model", "test_model", "SetupParams",
           "load_checkpoint", "load_checkpoint_from_cwd", "load_model_from_cwd"]

try:
    from .lightning_trainer import HippynnLightningModule
    __all__ += ["HippynnLightningModule"]
except ImportError:
    pass

"""

The hippynn python package.

.. autodata:: settings
   :no-value:


"""

from . import _version
__version__ = _version.get_versions()['version']

# Configuration settings
from ._settings_setup import settings, reload_settings

# Pytorch modules
from . import layers
from . import networks

# Graph abstractions
from . import graphs
from .graphs import nodes, IdxType, GraphModule, Predictor

# Database loading
from . import databases
from .databases import Database, NPZDatabase, DirectoryDatabase

# Training/testing routines
from . import experiment
from .experiment import setup_and_train, train_model, setup_training,\
    test_model, load_model_from_cwd, load_checkpoint, load_checkpoint_from_cwd

# Other subpackages
from . import molecular_dynamics
from . import optimizer

# Custom Kernels
from . import custom_kernels
from .custom_kernels import set_custom_kernels

from . import pretraining
from .pretraining import hierarchical_energy_initialization

from . import tools
from .tools import active_directory, log_terminal

# The order is adjusted to put functions after objects in the documentation.
_dir = dir()
_lowerdir = [x for x in _dir if x[0].lower() == x[0]]
_upperdir = [x for x in _dir if x[0].upper() == x[0]]
__all__ = _lowerdir + _upperdir
del _dir, _lowerdir, _upperdir

__all__ = [x for x in __all__ if not x.startswith("_")]

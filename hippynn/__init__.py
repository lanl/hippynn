"""

The hippynn python package.

"""

from . import _version
__version__ = _version.get_versions()['version']

# Configuration settings
from ._settings_setup import settings

# Pytorch modules
from . import layers
from . import networks # wait this one is different from the other one.

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

"""

The hippynn python package.

"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Configurational settings
from ._settings_setup import settings


# Pytorch modules
from . import layers
from . import networks

# Graph abstractions
from . import graphs

# Database loading
from . import databases

# Training/testing routines
from . import experiment
from .experiment import setup_and_train

# Custom Kernels
from . import custom_kernels

from . import pretraining

from . import tools

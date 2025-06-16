"""

The hippynn python package.

.. autodata:: settings
   :no-value:


"""
# Dev note: imports of submodules reflect dependency of submodules.

from . import _version
__version__ = _version.get_versions()['version']

# Configuration settings
from ._settings_setup import settings, reload_settings

# Tools should not have any dependencies on internal packages besides settings.
from . import tools
from .tools import active_directory, log_terminal

# Custom Kernels
from . import custom_kernels
from .custom_kernels import set_custom_kernels

# Pytorch modules
from . import layers
from . import networks

# Graph abstractions
from . import graphs
from .graphs import nodes, IdxType, GraphModule, Predictor, make_ensemble

# Kinds of nodes
from .graphs.nodes import inputs, targets, loss, pairs, physics, indexers, pairs
from .graphs.nodes import networks as network_nodes

from . import pretraining
from .pretraining import hierarchical_energy_initialization

# Database loading
from . import databases
from .databases import Database, NPZDatabase, DirectoryDatabase

# Training/testing routines
from . import experiment
from .experiment import setup_and_train, train_model, setup_training,\
    test_model, load_model_from_cwd, load_checkpoint, load_checkpoint_from_cwd

# Optional imports are dealt with in submodule
from . import molecular_dynamics

try:
    from . import plotting
except ImportError:
    pass  # Don't have matplotlib.

# Submodules that require ase
try:
    import ase
except ImportError:
    pass
else:
    del ase
    from . import optimizer
    from .interfaces import ase_interface

# Submodules that require pyseqm
try:
    import seqm
except ImportError:
    pass
else:
    del seqm
    from .interfaces import pyseqm_interface

# Submodules that require lammps
try:
    import lammps
except ImportError:
    pass
else:
    del lammps
    try:
        from .interfaces import lammps_interface
    except Exception as eee:
        import warnings
        warnings.warn(f"Lammps interface was not importable due to exception: :{eee}")
        del eee, warnings

# The order is adjusted to put functions after objects in the documentation.
_dir = dir()
_lowerdir = [x for x in _dir if x[0].lower() == x[0]]
_upperdir = [x for x in _dir if x[0].upper() == x[0]]
__all__ = _lowerdir + _upperdir
del _dir, _lowerdir, _upperdir

__all__ = [x for x in __all__ if not x.startswith("_")]

"""
A range of tools for working with MD data and running MD using hippynn. Includes tools for processing MD data to use 
for training hippynn models and analyzing MD trajectories, a custom molecular dynamics driver that allows for the 
design of user-created algorithms, and a suite of functions for applying coarse-graining mappings to all-atom data. 
"""

from .coarse_grain import coarse_grain_all, cg_one_center_of_mass_pbc, cg_one_center_of_geometry_pbc, cg_one_mass_weighted_average, cg_one_average, cg_one_sum
from .misc import SpeciesLookup
from .rdf import calculate_rdf, calculate_adf
from .writers import write_extxyz

__all__ = [
    "coarse_grain_all",
    "cg_one_center_of_mass_pbc",
    "cg_one_center_of_geometry_pbc",
    "cg_one_mass_weighted_average",
    "cg_one_average",
    "cg_one_sum",
    "SpeciesLookup",
    "calculate_rdf",
    "calculate_adf",
    "write_extxyz",
]

try:
    import MDAnalysis
except ImportError:
    pass
else:
    del MDAnalysis
    from . import readers
    from .readers import extract_trajectory_data

    __all__.extend(["extract_trajectory_data"])


try:
    import ase
except ImportError:
    pass
else:
    del ase
    from . import md
    from .md import MolecularDynamics, Variable, NullUpdater, VelocityVerlet, LangevinDynamics, VariableUpdater

    __all__.extend(["MolecularDynamics", "Variable", "NullUpdater", "VelocityVerlet", "LangevinDynamics", "VariableUpdater"])

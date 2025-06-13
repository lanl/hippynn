"""
A range of tools for working with MD data and running MD using hippynn. Includes tools for processing MD data to use 
for training hippynn models and analyzing MD trajectories, a custom molecular dynamics driver that allows for the 
design of user-created algorithms, and a suite of functions for applying coarse-graining mappings to all-atom data. 

This module is only available if the `ase` package is installed.

"""

from .coarse_grain import coarse_grain_all, cg_one_center_of_mass_pbc, cg_one_center_of_geometry_pbc, cg_one_mass_weighted_average, cg_one_average, cg_one_sum
from .md import MolecularDynamics, Variable, NullUpdater, VelocityVerlet, LangevinDynamics, VariableUpdater
from .misc import SpeciesLookup
from .rdf import calculate_rdf, calculate_adf
from .readers import extract_trajectory_data
from .writers import write_extxyz

__all__ = [
    "coarse_grain_all",
    "cg_one_center_of_mass_pbc",
    "cg_one_center_of_geometry_pbc",
    "cg_one_mass_weighted_average",
    "cg_one_average",
    "cg_one_sum",
    "MolecularDynamics",
    "Variable",
    "NullUpdater",
    "VelocityVerlet",
    "LangevinDynamics",
    "VariableUpdater",
    "SpeciesLookup",
    "calculate_rdf",
    "calculate_adf",
    "extract_trajectory_data",
    "write_extxyz",
]


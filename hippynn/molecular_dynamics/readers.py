'''
Read outputs from LAMMPS, Gromacs.

This module is only available if the `MDAnalysis` package is installed.
'''
# Portions of this code were written with assistance from an LLM.


import os

import numpy as np
import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError
try:
    from MDAnalysis.guesser.tables import SYMB2Z, Z2SYMB
except ImportError:
    from MDAnalysis.topology.tables import SYMB2Z, Z2SYMB


def get_types(universe):
    types = np.array(universe.atoms.types)
    try:
        types = [int(typ) for typ in types]
        species_numbers = types
        species_symbols = [Z2SYMB[num] for num in species_numbers]
        print(f"Found atom types {np.unique(types)}, which were infered to be atomic numbers. If this is not correct, please rerun the function and pass a `type_correction_dict`. The keys should be of type {type(types[0])}.")
    except ValueError:
        species_symbols = types
        species_numbers = [SYMB2Z[sym] for sym in species_symbols]
        print(f"Found atom types {np.unique(types)}, which were infered to be chemical symbols. If this is not correct, please rerun the function and pass a `type_correction_dict`. The keys should be of type {type(types[0])}.")
    return np.array(species_numbers), np.array(species_symbols)

def extract_trajectory_data(topology, trajectory, start=0, stop=None, stride=1, type_correction_dict=None, guess_masses=True, mda_universe_kwargs={}):
    """
    Extracts trajectory data using MDAnalysis.

    E.g., for Gromacs: extract_trajectory_data(topology="output.gro", trajectory="output.trr")
    E.g., for LAMMPS: extract_trajectory_data(topology="system.data", trajectory="output.lammpstrj")

    :param topology: Path to topology file (e.g., 'init.data', 'md.gro').
    :type topology: str
    :param trajectory: Path to trajectory file (e.g., 'traj.lammpstrj', 'md.trr').
    :type trajectory: str
    :param start: Starting frame index (inclusive). Default is 0.
    :type start: int
    :param stop: Ending frame index (exclusive). If None, reads until the end.
    :type stop: int
    :param stride: Step size between frames. Default is 1.
    :type stride: int
    :param type_correction_dict: If types inferred from files are not atomic numbers or symbols, pass {force_field_type: atomic_number} 
                                 to correct this. The keys' types must Default is None. 
    :type type_correction_dict: dict(str, int)
    :param guess_masses: If true, will try to infer masses based on types. Default is True.
    :type guess_masses: bool
    :param mda_universe_kwargs: Keywords to feed to MDAnalysis.Universe. Default is {}. 
    :type mda_universe_kwargs: dict
    :returns: Dictionary with keys:
        - positions: ndarray (n_frames, n_atoms, 3)
        - velocities: ndarray or None (n_frames, n_atoms, 3)
        - forces: ndarray or None (n_frames, n_atoms, 3)
        - cells: ndarray (n_frames, 3, 3)
        - masses: ndarray or None (n_atoms,)
        - species: ndarray (n_atoms,)
        - mol_ids: ndarray (n_atoms,)
    :rtype: dict
    """

    # Give it help if trajectory file is a LAMMPS dump file
    _, extension = os.path.splitext(trajectory)
    if extension == ".lammpstrj":
        ext_kwargs = {"format": "LAMMPSDUMP"}
        ext_kwargs.update(mda_universe_kwargs) # allow it to still be overridden by user
        mda_universe_kwargs = ext_kwargs

    if 'to_guess' not in mda_universe_kwargs.keys():
        if not guess_masses:
            mda_universe_kwargs['to_guess'] = ('types',)
        else:
            mda_universe_kwargs['to_guess'] = ('types', 'masses')

    u = mda.Universe(topology, trajectory, **mda_universe_kwargs)

    if type_correction_dict is not None:
        u.atoms.types = [type_correction_dict[typ] if typ in type_correction_dict.keys() else typ for typ in u.atoms.types]
        if guess_masses: u.guess_TopologyAttrs(force_guess=('masses',))

    total_frames = len(u.trajectory)
    if stop is None or stop > total_frames:
        stop = total_frames
    selected_frames = range(start, stop, stride)
    n_frames = len(selected_frames)
    n_atoms = len(u.atoms)

    # Preallocate trajectory arrays
    positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    velocities = np.full((n_frames, n_atoms, 3), np.nan, dtype=np.float32)
    forces = np.full((n_frames, n_atoms, 3), np.nan, dtype=np.float32)
    cells = np.zeros((n_frames, 3, 3), dtype=np.float32)

    # Static atom info
    try:
        masses = u.atoms.masses.astype(np.float32)
    except NoDataError:
        masses = None
    mol_ids = u.atoms.resindices.astype(np.int32)

    species_numbers, species_symbols = get_types(u)

    # Add frame axis to static data
    masses = (np.tile(masses, (n_frames, 1)) if masses is not None else None)
    mol_ids = np.tile(mol_ids, (n_frames, 1))
    species_numbers = np.tile(species_numbers, (n_frames, 1))
    species_symbols = np.tile(species_symbols, (n_frames, 1))

    # Extract per-frame data
    for i, ts in enumerate(u.trajectory[start:stop:stride]):
        positions[i] = ts.positions
        cells[i] = ts.dimensions[:3] * np.eye(3)

        if ts.velocities is not None:
            velocities[i] = ts.velocities
        if ts.forces is not None:
            forces[i] = ts.forces

    return {
        'positions': positions,
        'velocities': velocities if not np.isnan(velocities).all() else None,
        'forces': forces if not np.isnan(forces).all() else None,
        'cells': cells,
        'masses': masses,
        'species_numbers': species_numbers,
        'species_symbols': species_symbols,
        'mol_ids': mol_ids,
    }
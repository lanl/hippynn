# Read outputs from LAMMPS, Gromacs
import os

import MDAnalysis as mda
import numpy as np

def correct_atom_types(universe, name_to_type_dict):
    for atom in universe.atoms:
        if atom.name in name_to_type_dict:
            atom.type = name_to_type_dict[atom.name]

def extract_trajectory_data(topology, trajectory, start=0, stop=None, stride=1, name_to_type_dict=None, mda_universe_kwargs={}):
    """
    Extracts trajectory data using MDAnalysis.

    E.g., for Gromacs: extract_trajectory_data(topology="output.gro", trajectory="output.trr")
    E.g., for LAMMPS: extract_trajectory_data(topology="system.data", trajectory="output.lammpstrj")

    Written with help from ChatGPT.

    .. warning:: If ``types`` cannot be read from the files but ``names`` is available, the types will be 
     guessed from the names. You can ensure this is done correctly by providing ``name_to_type_dict``. The 
     masses will then be guessed based on the types. 

    :param str topology: Path to topology file (e.g., 'init.data', 'md.gro').
    :param str trajectory: Path to trajectory file (e.g., 'traj.lammpstrj', 'md.trr').
    :param int start: Starting frame index (inclusive). Default is 0.
    :param int stop: Ending frame index (exclusive). If None, reads until the end.
    :param int stride: Step size between frames. Default is 1.
    :param dict name_to_type_dict: Can be provided to assist in the guessing of ``types`` from ``names`` if ``types`` is not available
    in the provided files. Default is None. 
    :param dict mda_universe_kwargs: Keywords to feed to MDAnalysis.Universe. Default is {}. 
    :returns: Dictionary with keys:
        - positions: ndarray (n_frames, n_atoms, 3)
        - velocities: ndarray or None (n_frames, n_atoms, 3)
        - forces: ndarray or None (n_frames, n_atoms, 3)
        - cells: ndarray (n_frames, 3, 3)
        - masses: ndarray (n_frames, n_atoms)
        - species: ndarray (n_frames, n_atoms)
        - mol_ids: ndarray (n_frames, n_atoms)
    :rtype: dict
    """

    # Give it help if trajectory file is a LAMMPS dump file
    _, extension = os.path.splitext(trajectory)
    if extension == ".lammpstrj" :
        ext_kwargs = {"format": "LAMMPSDUMP"}
        ext_kwargs.update(mda_universe_kwargs) # allow it to still be overridden by user
        mda_universe_kwargs = ext_kwargs
    u = mda.Universe(topology, trajectory, **mda_universe_kwargs, to_guess=('types', 'masses'))

    if name_to_type_dict is not None:
        correct_atom_types(u, name_to_type_dict)
        u.guess_TopologyAttrs(force_guess=('masses',))

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
    base_masses = u.atoms.masses.astype(np.float32)
    base_species = np.array(u.atoms.types)
    base_mol_ids = u.atoms.resids.astype(np.int32)

    # Broadcast static info across frames
    masses = np.tile(base_masses, (n_frames, 1))
    species = np.tile(base_species, (n_frames, 1))
    mol_ids = np.tile(base_mol_ids, (n_frames, 1))

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
        'species': species,
        'mol_ids': mol_ids,
    }
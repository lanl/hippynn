# Read outputs from LAMMPS, Gromacs

import MDAnalysis as mda
import numpy as np

def extract_lammps_trajectory(datafile, trajfile, start=0, stop=None, stride=1):
    """
    Extracts trajectory data from a LAMMPS simulation using MDAnalysis.

    Written with help from ChatGPT.

    :param str datafile: Path to the LAMMPS data file (e.g., 'init.data').
    :param str trajfile: Path to the LAMMPS trajectory file (e.g., 'traj.lammpstrj').
    :param int start: Starting frame index (inclusive). Default is 0.
    :param int stop: Ending frame index (exclusive). If None, reads until the end.
    :param int stride: Step size between frames. Default is 1.

    :returns: Dictionary with keys:
        - positions: ndarray (n_frames, n_atoms, 3)
        - velocities: ndarray or None (n_frames, n_atoms, 3)
        - forces: ndarray or None (n_frames, n_atoms, 3)
        - cells: ndarray (n_frames, 3, 3)
        - masses: ndarray (n_frames, n_atoms)
        - species: ndarray (n_frames, n_atoms) — atom types from topology
        - mol_ids: ndarray (n_frames, n_atoms)
    :rtype: dict
    """
    u = mda.Universe(datafile, trajfile, format="LAMMPSDUMP")

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
    try:
        base_mol_ids = u.atoms.resids.astype(np.int32)
    except AttributeError:
        base_mol_ids = np.full((n_atoms,), -1, dtype=np.int32)

    # Broadcast static info across frames
    masses = np.tile(base_masses, (n_frames, 1))
    species = np.tile(base_species, (n_frames, 1))
    mol_ids = np.tile(base_mol_ids, (n_frames, 1))

    # Extract per-frame data
    for i, ts in enumerate(u.trajectory[start:stop:stride]):
        positions[i] = ts.positions
        cells[i] = ts.triclinic_dimensions if ts.triclinic else ts.dimensions[:3] * np.eye(3)

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

def extract_gromacs_trajectory(topology_file, trajectory_file, start=0, stop=None, stride=1):
    """
    Extracts trajectory data from a GROMACS TRR file using MDAnalysis.

    Written with help from ChatGPT.

    :param str topology_file: Path to the GROMACS topology file (e.g., '.gro', '.pdb').
    :param str trajectory_file: Path to the GROMACS TRR trajectory file.
    :param int start: Starting frame index (inclusive). Default is 0.
    :param int stop: Ending frame index (exclusive). If None, reads until the end.
    :param int stride: Step size between frames. Default is 1.

    :returns: Dictionary with keys:
        - positions: ndarray (n_frames, n_atoms, 3)
        - velocities: ndarray or None (n_frames, n_atoms, 3)
        - forces: ndarray or None (n_frames, n_atoms, 3)
        - cells: ndarray (n_frames, 3, 3)
        - masses: ndarray (n_frames, n_atoms)
        - species: ndarray (n_frames, n_atoms) — atom types (strings)
        - mol_ids: ndarray (n_frames, n_atoms)
    :rtype: dict
    """
    u = mda.Universe(topology_file, trajectory_file)

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
    base_species = np.array(u.atoms.types)  # Use atom.type (strings like "CH3", "OW", etc.)
    base_mol_ids = u.atoms.resids.astype(np.int32)

    # Broadcast across frames
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
# ChatGPT was used in creating these functions

import numpy as np

from .pbc_tools import find_mic, validate_diagonal_cell

def compute_bead_indices(atom_to_bead):
    """
    Compute bead indices from an atom-to-bead mapping.

    :param atom_to_bead: Array of shape (n_atoms,) mapping each atom to a bead.
    :return: Tuple (beads, bead_indices) where:
             - beads is a sorted array of unique bead labels.
             - bead_indices is a dict mapping each bead label to its atom indices.
    """
    beads = np.unique(atom_to_bead)
    bead_indices = {bead: np.where(atom_to_bead == bead)[0] for bead in beads}
    return beads, bead_indices

def coarse_grain_all(field, atom_to_bead, coarse_grain_one, masses=None, cells=None, single_frame=False):
    """
    Apply a user-defined coarse_grain_one function to all beads across all frames.

    The user-supplied function, coarse_grain_one, should have one of the following signatures:
    
        coarse_grain_one(bead_field)
        coarse_grain_one(bead_field, bead_masses=bead_masses)
        coarse_grain_one(bead_field, cell=cell)
        coarse_grain_one(bead_field, bead_masses=bead_masses, cell=cell)

    This function loops over each frame (if single_frame=False) and over each bead (as defined by atom_to_bead) and applies
    coarse_grain_one to each bead. Returns the resulting values in an array of shape (n_frames, n_beads, ...) or (n_beads, ...) 
    where the trailing dimensions match the output of coarse_grain_one.

    :param field: Array of data to coarse-grain. Shape (n_frames, n_atoms, d) if single_frame=False or 
                  (n_atoms, d) if single_frame=True.
    :param atom_to_bead: Array of shape (n_atoms,) mapping each atom to a bead.
    :param coarse_grain_one: Function to coarse-grain one bead's data.
    :param masses: (Optional) Array of shape (n_atoms,) of masses. Passed to coarse_grain_one if provided.
    :param cells: (Optional) Array of shape (n_frames, 3, 3) if single_frame=False or (3, 3) if single_frame=True 
                  of cell matrices. The cell for the current frame is passed to coarse_grain_one if provided.
    :param single_frame: (Optional) Use to specify if data arrays contain a frame axis. Defaults to False. 
    :return: Array of coarse-grained values. Shape (n_frames, n_beads, ...) if single_frame=False or 
                  (n_beads, ...) if single_frame=True.
    """
    beads, bead_indices = compute_bead_indices(atom_to_bead)
    n_frames = field.shape[0] if not single_frame else 1
    result = []
    for i in range(n_frames):
        frame_field = field[i] if not single_frame else field  # shape (n_atoms, d)
        frame_cell = (cells[i] if not single_frame else cells) if cells is not None else None
        frame_result = []
        for bead in beads:
            indices = bead_indices[bead]
            bead_field = frame_field[indices]
            kwargs = {}
            if masses is not None:
                kwargs['mass'] = masses[indices]
            if frame_cell is not None:
                kwargs['cell'] = frame_cell
            value = coarse_grain_one(bead_field, **kwargs)
            frame_result.append(value)
        if single_frame:
            return np.array(frame_result)
        else:
            result.append(frame_result)
    return np.array(result)

# --- Mapping functions ---

def cg_one_center_of_mass_pbc(bead_pos, mass, cell):
    """Position center of mass using PBC"""
    validate_diagonal_cell(cell=cell)
    mic_dists = find_mic(bead_pos[1:] - bead_pos[0], cell=cell)
    return np.sum(mic_dists * mass[1:, None], axis=0) / mass.sum() + bead_pos[0]

def cg_one_mass_weighted_average(bead_field, mass):
    """General mass-weighted average (eg. for velocities when using COM position mapping)"""
    return np.sum(bead_field * mass[:, None], axis=0) / mass.sum()

def cg_one_average(bead_field):
    """Unweighted average"""
    return np.mean(bead_field, axis=0)

def cg_one_sum(bead_field):
    """Sum of values"""
    return np.sum(bead_field, axis=0)
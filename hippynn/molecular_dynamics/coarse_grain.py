# Portions of this code were written with assistance from an LLM

import numpy as np

from .pbc_tools import find_mic, validate_diagonal_cell

def validate_nframes_natoms(value, value_name, n_frames, n_atoms):
    value = value.squeeze()
    if value.shape == (n_atoms, ):
        value = np.tile(value, (n_frames, 1))
    elif value.shape != (n_frames, n_atoms):
        raise ValueError(f"Unexpected shape found for '{value_name}'. Expected ({n_frames}, {n_atoms}) or ({n_atoms},). Found {value.shape}.")
    return value

def validate_cells(cells, n_frames):
    cells = cells.squeeze()
    if cells.shape == (3, 3):
        cells = np.tile(cells, (n_frames, 1, 1))
    elif cells.shape != (n_frames, 3, 3):
        raise ValueError(f"Unexpected shape found for 'cells'. Expected ({n_frames}, 3, 3) or (3, 3). Found {cells.shape}.")
    return cells

def validate_bead_species(bead_species, atom_to_bead, n_frames):
    bead_species = bead_species.squeeze()
    max_bead_idx = atom_to_bead.max()
    if bead_species.shape == (max_bead_idx+1,):
        bead_species = np.tile(bead_species, (n_frames, 1))
    elif bead_species.shape != (n_frames, max_bead_idx+1):
        raise ValueError(f"Unexpected shape found for 'bead_species'. Expected ({n_frames}, {max_bead_idx+1}) or ({max_bead_idx+1},). Found {bead_species.shape}.")
    return bead_species

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

def padded_ndarray_from_list(list_of_lists):
    max_len = max(len(lst) for lst in list_of_lists)
    padded = [lst + [0]*(max_len - len(lst)) for lst in list_of_lists]
    return np.array(padded)

def coarse_grain_all(values, atom_to_bead, coarse_grain_one, masses=None, cells=None, atom_species=None, bead_species=None, single_frame=False):
    """
    Apply a user-defined ``coarse_grain_one`` function to all beads across all frames.

    The user-supplied function, ``coarse_grain_one``, should take ``bead_values`` as an argument, which will correspond the the values
    of ``values`` for one bead in one frame. It can optionally take ``bead_masses``, ``frame_cell``, ``bead_atom_species``, and ``bead_type``
    as additional arguments. It should return the coarse_grained version of ``values`` for the bead given the data provided. 

    This function loops over each frame (if single_frame=False) and over each bead (as defined by atom_to_bead) and applies
    ``coarse_grain_one`` to each bead. Returns the resulting values in an array of shape (n_frames, n_beads, ...) or (n_beads, ...) 
    where the trailing dimensions match the output of ``coarse_grain_one``.

    See examples of pre-defined ``coarse_grain_one`` options in hippynn.molecular_dynamics.coarse_grain.

    :param values: Array of data to coarse-grain. Shape (n_frames, n_atoms, d) if single_frame=False or 
                  (n_atoms, d) if single_frame=True.
    :param atom_to_bead: Integer array of shape (n_frames, n_atoms,) or (n_atoms,) mapping each atom to a bead.
    :param coarse_grain_one: Function to coarse-grain one bead's data.
    :param masses: (Optional) Array of shape (n_frames, n_atoms,) or (n_atoms,) of masses. Values for the current frame/bead passed
                  to ``coarse_grain_one`` if provided.
    :param cells: (Optional) Array of shape (n_frames, 3, 3) or (3, 3) of cell matrices. The cell for the current frame
                    is passed to ``coarse_grain_one`` if provided.
    :param atom_species: (Optional) Array of shape (n_frames, n_atoms,) or (n_atoms,) of atom species, defined in any manner 
                        you choose. Values for the current frame passed to ``coarse_grain_one`` if provided.
    :param bead_species: (Optional) Array of shape (n_frames, max_bead_idx+1,) or (max_bead_idx+1,) where ``max_bead_idx``
                        is the maximum value in ``atom_to_bead``. Values for the current frame/bead passed to ``coarse_grain_one`` 
                        if provided.
    :param single_frame: (Optional) Use to specify if data arrays contain a frame axis. Defaults to False. 
    :return: Array of coarse-grained values. Shape (n_frames, n_beads, ...) if single_frame=False or 
                  (n_beads, ...) if single_frame=True.
    """
    if single_frame:
        n_frames = 1
        n_atoms = values.shape[0]
    else:
        n_frames, n_atoms = values.shape[:2]
    
    # Validation
    atom_to_bead = validate_nframes_natoms(atom_to_bead, "atom_to_bead", n_frames, n_atoms)
    if masses is not None: masses = validate_nframes_natoms(masses, "masses", n_frames, n_atoms)
    if cells is not None: cells = validate_cells(cells, n_frames)
    if atom_species is not None: atom_species = validate_nframes_natoms(atom_species, "atom_species", n_frames, n_atoms)
    if bead_species is not None: bead_species = validate_bead_species(bead_species, atom_to_bead, n_frames)

    # Begin computation
    result = []
    for i in range(n_frames):
        beads, bead_indices = compute_bead_indices(atom_to_bead[i])
        frame_result = []
        for bead in beads:
            indices = bead_indices[bead]
            kwargs = {'bead_values': values[i, indices]}
            if masses is not None:
                kwargs['bead_masses'] = masses[i, indices]
            if cells is not None:
                kwargs['frame_cell'] = cells[i]
            if atom_species is not None:
                kwargs['bead_atom_species'] = atom_species[i, indices]
            if bead_species is not None:
                kwargs['bead_type'] = bead_species[i, bead]
            value = coarse_grain_one(**kwargs)
            frame_result.append(value)
        if single_frame:
            return np.array(frame_result)
        else:
            result.append(frame_result)

    return padded_ndarray_from_list(result)

# --- Mapping functions ---

def cg_one_center_of_mass_pbc(bead_values, bead_masses, frame_cell, wrap_into_cell=True):
    """Position center of mass using PBC"""
    validate_diagonal_cell(cell=frame_cell)
    mic_dists = find_mic(bead_values[1:] - bead_values[0], cell=frame_cell)
    com = np.sum(mic_dists * bead_masses[1:, None], axis=0) / bead_masses.sum() + bead_values[0]
    if wrap_into_cell:
        return com % np.diag(frame_cell)
    else:
        return com

def cg_one_center_of_geometry_pbc(bead_values, frame_cell, wrap_into_cell=True):
    """Position center of geometry using PBC"""
    validate_diagonal_cell(cell=frame_cell)
    mic_dists = find_mic(bead_values[1:] - bead_values[0], cell=frame_cell)
    cog = np.sum(mic_dists, axis=0) / bead_values.shape[0] + bead_values[0]
    if wrap_into_cell:
        return cog % np.diag(frame_cell)
    else:
        return cog

def cg_one_mass_weighted_average(bead_values, bead_masses):
    """General mass-weighted average (eg. for velocities when using COM position mapping)"""
    return np.sum(bead_values * bead_masses[:, None], axis=0) / bead_masses.sum()

def cg_one_average(bead_values):
    """Unweighted average"""
    return np.mean(bead_values, axis=0)

def cg_one_sum(bead_values):
    """Sum of values"""
    return np.sum(bead_values, axis=0)
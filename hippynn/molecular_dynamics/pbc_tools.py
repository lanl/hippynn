# Portions of this code were written with assistance from an LLM

import numpy as np

def is_diagonal_cell(cell, rtol=1e-05, atol=1e-08):
    """use when cell shape is (3, 3)"""
    if cell.shape != (3, 3):
        raise ValueError(f"Cell must be shape (3, 3). Found {cell.shape} instead.")
    return np.allclose(cell[~np.eye(3, dtype=bool)], 0, atol=atol, rtol=rtol)

def is_diagonal_multicell(multicell, rtol=1e-05, atol=1e-08):
    """use when cell shape is (n_frames, 3, 3)"""
    if multicell.ndim != 3 or multicell.shape[1:] != (3, 3):
        raise ValueError(f"Cells array should be of shape (n_frames, 3, 3). Found {multicell.shape} instead.")
    return np.allclose(multicell[:, ~np.eye(3, dtype=bool)], 0, atol=atol, rtol=rtol)

def validate_diagonal_cell(cell=None, multicell=None):
    if (cell is not None and not is_diagonal_cell(cell)) or (multicell is not None and not is_diagonal_multicell(multicell)):
        raise ValueError("This function only works for orthorhombic cells with lattice vectors on the Cartesian axes, i.e. cells which can be written as a diagonal matrix.")
    
def extract_cell_diagonal(cell, check_diagonal=True):
    """use when cell shape is (3, 3)"""
    if check_diagonal:
        validate_diagonal_cell(cell=cell)
    return cell[[0,1,2],[0,1,2]]

def extract_multicell_diagonal(multicell, check_diagonal=True):
    """use when cell shape is (n_frames, 3, 3)"""
    if check_diagonal:
        validate_diagonal_cell(multicell=multicell)
    return multicell[:,[0,1,2],[0,1,2]]

def find_mic(vecs, cell):
    """
    Compute the minimum image displacement of vectors under periodic boundary conditions. Only works
    for orthorhombic cells with lattice vectors on the Cartesian axes, i.e. cells which can be written 
    as a diagonal matrix.

    :param vecs: Displacement vectors.
    :type vecs: array_like, shape (n_particles, 3)

    :param cells: Cell matrix.
    :type cells: array_like, shape (3, 3)

    :returns: The minimum image displacement vectors.
    :rtype: ndarray, shape (n_particles, 3)
    """

    cell = extract_cell_diagonal(cell)
    return vecs - cell * np.round(vecs / cell)

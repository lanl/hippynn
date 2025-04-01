import numpy as np
from numba import jit

def extract_cell_diagonals(arr: np.ndarray):
    """
    Given an array of shape (..., 3, 3) where each 3x3 matrix is expected
    to be diagonal, returns an array of shape (..., 3) containing the diagonal
    elements. Raises a ValueError if any off-diagonal element is non-zero.
    """
    if arr.ndim < 2 or arr.shape[-2:] != (3, 3):
        raise ValueError("Input array must have shape (..., 3, 3)")
    return _extract_diagonals(arr)

@jit
def _extract_diagonals(arr):
    """Written with help from ChatGPT o3-mini-high"""
    
    # Compute the flattened shape for the batch of matrices.
    flat_shape = 1
    for s in arr.shape[:-2]:
        flat_shape *= s
    arr_flat = arr.reshape(flat_shape, 3, 3)
    diag_flat = np.empty((flat_shape, 3), dtype=arr.dtype)
    
    for i in range(flat_shape):
        # Extract the diagonal elements.
        diag_flat[i, 0] = arr_flat[i, 0, 0]
        diag_flat[i, 1] = arr_flat[i, 1, 1]
        diag_flat[i, 2] = arr_flat[i, 2, 2]
        # Check that off-diagonals are zero.
        if (arr_flat[i, 0, 1] != 0 or arr_flat[i, 0, 2] != 0 or
            arr_flat[i, 1, 0] != 0 or arr_flat[i, 1, 2] != 0 or
            arr_flat[i, 2, 0] != 0 or arr_flat[i, 2, 1] != 0):
            raise ValueError("Matrix is not diagonal")
    
    # Reshape the result back to the original leading dimensions with shape (..., 3).
    out_shape = arr.shape[:-2] + (3,)
    return diag_flat.reshape(out_shape)

def find_mic(vec, cell):
    """
    Compute the minimum image displacement of vectors under periodic boundary conditions.

    Parameters:
      vec   : array_like, shape (..., 3,)
             The displacement vector.
      cell : array_like, shape (3,)
             The cell (or lattice) vectors as rows. #CHANGE THIS

    Returns:
      vec_mic : ndarray, shape (..., 3,)
               The minimum image displacement vector.
    """

    vec_mic = vec.copy()
    vec_mic -= cell * np.round(vec / cell)
    return vec_mic

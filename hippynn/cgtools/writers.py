# ChatGPT was used in creating these functions

import numpy as np

def write_extxyz(filename, positions, species, cells=None):
    """
    Write a extended XYZ (.extxyz) file with species, positions, and optional cells.

    Parameters:
        filename (str): Output .extxyz file path.
        positions (np.ndarray): Shape (n_frames, n_atoms, 3).
        species (np.ndarray): Shape (n_atoms,).
        cells (np.ndarray, optional): Shape (n_frames, 3, 3). 
    """
    # --- Validation ---
    if not isinstance(species, np.ndarray) or species.ndim != 1:
        raise ValueError("species must be a 1D NumPy array of species")

    if not isinstance(positions, np.ndarray) or positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must be a 3D NumPy array with shape (n_frames, n_atoms, 3)")

    n_frames, n_atoms, _ = positions.shape
    if len(species) != n_atoms:
        raise ValueError(f"species length ({len(species)}) must match positions.shape[1] ({n_atoms})")

    if cells is not None:
        if not isinstance(cells, np.ndarray) or cells.shape != (n_frames, 3, 3):
            raise ValueError("cells must be a NumPy array of shape (n_frames, 3, 3)")

    # --- Write ---
    with open(filename, "w") as f:
        for t in range(n_frames):
            f.write(f"{n_atoms}\n")

            # Header with Lattice and Properties
            header_parts = []
            if cells is not None:
                lattice_flat = " ".join(f"{x:.6f}" for x in cells[t].flatten())
                header_parts.append(f'Lattice="{lattice_flat}"')

            header_parts.append('Properties=species:S:1:pos:R:3')
            f.write(" ".join(header_parts) + "\n")

            # Atom lines
            for i in range(n_atoms):
                x, y, z = positions[t, i]
                f.write(f"{species[i]} {x:.6f} {y:.6f} {z:.6f}\n")

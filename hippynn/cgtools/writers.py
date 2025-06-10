# Portions of this code were written with assistance from an LLM

import os

import numpy as np

from ..tools import progress_bar

def write_extxyz(filename, positions, species=None, cells=None, velocities=None, forces=None):
    """
    Write a extended XYZ (.extxyz) file.

    Parameters:
        filename (str): Output .extxyz file path.
        positions (np.ndarray): Shape (n_frames, n_particles, 3).
        species (np.ndarray, optional): Shape (n_frames, n_particles,).
        cells (np.ndarray, optional): Shape (n_frames, 3, 3). 
        velocities (np.ndarray, optional): Shape (n_frames, n_particles, 3). 
        forces (np.ndarray, optional): Shape (n_frames, n_particles, 3). 
    """
    # --- Validation ---
    if not isinstance(positions, np.ndarray) or positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must be a 3D NumPy array with shape (n_frames, n_particles, 3)")

    n_frames, n_particles, _ = positions.shape

    if species is not None:
        if not isinstance(species, np.ndarray) or species.squeeze().shape != (n_frames, n_particles):
            raise ValueError("species must be a NumPy array of shape (n_frames, n_particles)")

    if cells is not None:
        if not isinstance(cells, np.ndarray) or cells.shape != (n_frames, 3, 3):
            raise ValueError("cells must be a NumPy array of shape (n_frames, 3, 3)")
        
    if velocities is not None:
        if not isinstance(velocities, np.ndarray) or velocities.ndim != 3 or velocities.shape != positions.shape:
            raise ValueError("velocities must be a 3D NumPy array with shape (n_frames, n_particles, 3)")
        
    if forces is not None:
        if not isinstance(forces, np.ndarray) or forces.ndim != 3 or forces.shape != positions.shape:
            raise ValueError("forces must be a 3D NumPy array with shape (n_frames, n_particles, 3)")

    # --- Write ---
    if not os.path.splitext(filename)[1]:  # checks if there's an extension
        filename += '.extxyz'

    with open(filename, "w") as f:
        for t in progress_bar(range(n_frames)):
            f.write(f"{n_particles}\n")

            # Header with Lattice and Properties
            header_parts = []
            if cells is not None:
                lattice_flat = " ".join(f"{x:.6f}" for x in cells[t].flatten())
                header_parts.append(f'Lattice="{lattice_flat}" pbc="T T T" ')

            header_parts.append('Properties=species:S:1:pos:R:3')
            if velocities is not None:
                header_parts.append(':velo:R:3')
            if forces is not None:
                header_parts.append(':force:R:3')
            f.write("".join(header_parts) + "\n")

            # Atom lines
            for i in range(n_particles):
                atom_data = []
                x, y, z = positions[t, i]
                if species is not None:
                    atom_data.append(f"{species[t, i]}")
                else:
                    atom_data.append(f"H")
                atom_data.append(f"{x:.6f} {y:.6f} {z:.6f}")
                if velocities is not None:
                    vx, vy, vz = velocities[t, i]
                    atom_data.append(f"{vx:.6f} {vy:.6f} {vz:.6f}") 
                if forces is not None:
                    fx, fy, fz = forces[t, i]
                    atom_data.append(f"{fx:.6f} {fy:.6f} {fz:.6f}") 
                f.write(" ".join(atom_data) + "\n")


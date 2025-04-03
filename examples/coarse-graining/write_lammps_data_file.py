import os
import numpy as np


def write_lammps_data_from_npz(npz_data_file, tgt_file="system.data", frame=-1):
    """
    :param npz_data_file: Location of .npz file containing trajectory data.
    :type npz_data_file: str
    :param tgt_file: Location at which to write LAMMPS data file. Defaults to "system.data".
    :type tgt_file: str, optional
    :param frame: If multiple timesteps are included in the .npz data file, select which frame 
    to write. Set to `None` if .npz file contains data from a single frame only with no frame axis. 
    Defaults to -1.
    :type frame: int or None, optional
    """
    with np.load(npz_data_file) as f:
        cells = f['cells']
        masses = f['masses']
        positions = f['positions']
        species = f['species']
        try:
            velocities = f['velocities']
        except KeyError:
            velocities = None
            print('NOTE: No velocities specified in input file. LAMMPS will assign velocities of zero by default. Other velocity intializations can also be requested in the LAMMPS input script.')
            

    if frame is not None:
        cell = cells[frame]
        masses = masses[frame]
        positions = positions[frame]
        species = species[frame]
        if velocities is not None:
            velocities = velocities[frame]

    if cell.shape == (3,3):
        if np.allclose(cell, np.diag(np.diag(cell))): # if cell is diagonal matrix
            cell = np.diag(cell)
        else:
            raise ValueError("Cell must be specified as a (3,) matrix or as a (3,3) diagonal matrix.")
    elif cell.shape != (3,):
        raise ValueError("Cell must be specified as a (3,) matrix or as a (3,3) diagonal matrix.")
    
    if velocities is not None:
        print("Velocity data has been converted from ps/Ang to fs/Ang.")
        velocities = velocities / 1000 # the velocities are in ps/A but we'll need them in fs/A. Change as needed
         
    n_beads = positions.shape[0]

    unique_species = np.unique(species)

    with open(tgt_file, "w") as t:
        t.write(f"LAMMPS data file\n\n")

        t.write(f"{n_beads} atoms\n")
        t.write(f"{len(unique_species)} atom types\n\n")

        t.write(f"0.0 {cell[0]} xlo xhi\n")
        t.write(f"0.0 {cell[1]} ylo yhi\n")
        t.write(f"0.0 {cell[2]} zlo zhi\n\n")

        t.write("Masses\n\n")

        for id, mass in sorted(list(set(zip(species.reshape(-1), masses.reshape(-1))))):
            t.write(f"{id} {mass}\n")

        t.write("\nAtoms\n\n")

        for i, (type, coords) in enumerate(zip(species.reshape(-1), positions.reshape(-1, 3))):
            t.write(f"{i+1} {type} {coords[0]} {coords[1]} {coords[2]} \n")

        if velocities is not None:
            t.write("\nVelocities\n\n")

            for i, (type, coords) in enumerate(zip(species.reshape(-1), velocities.reshape(-1, 3))):
                t.write(f"{i+1} {coords[0]} {coords[1]} {coords[2]} \n")

    print(f"LAMMPS data input file written to {os.path.abspath(tgt_file)}.")
    return

if __name__ == "__main__":
    npz_data_file = os.path.join("..", "..", "..", "datasets", "cg_methanol_trajectory.npz")
    tgt_file = os.path.join("lammps_md_inputs", "system.data")
    write_lammps_data_from_npz(npz_data_file=npz_data_file, tgt_file=tgt_file)
"""
1. Download the file methane.extxyz.gz from https://archive.materialscloud.org/records/kz78r-6nx43
2. Unzip the file: $ gunzip methane.extxyz.gz
3. Place the resulting file in a folder called datasets/ at the same level as hippynn/
4. Run this script. 

NOTE: Extracting the entire file in this manner takes something like 30 minutes 
(at least on my machine). For this example, we will only extract the first 100,000 
configurations. This takes about 2 minutes on my machine. You can change the number
of configurations loaded by adjusting the value of `sample_size` below.

Portions of this code were written with assistance from an LLM.
"""

from pathlib import Path
import random
import shutil

import numpy as np
from ase.io import iread

from hippynn.tools import progress_bar

# ----- User parameters -----

sample_size = 100_000 # number of configurations to extract
# sample_size = -1 # extract all configurations

extxyz_path = Path('../../datasets/methane.extxyz') # source
npz_path = Path('../../datasets/methane.npz') # target

# ----- Determine frames to extract -----
all_frames = 7_732_488

if sample_size == -1:
    sample_size = all_frames
    use_all = True
    sampled_indices = None
else:
    use_all = False
    sampled_indices = set(random.sample(range(all_frames), sample_size))

# ----- Pre-allocate arrays -----
n_atoms = 5

species = np.empty((sample_size, n_atoms), dtype=np.int32)
positions = np.empty((sample_size, n_atoms, 3), dtype=np.float32)
forces = np.empty((sample_size, n_atoms, 3), dtype=np.float32)
energies = np.empty((sample_size,), dtype=np.float32)

# ----- Read data -----
print("Reading data header (slow)", flush=True)

reader = iread(extxyz_path, format="extxyz")
frame = next(reader) # Reading in the first frame will take considerably longer than the rest

print("Reading data", flush=True)
file_idx = 0
for i in progress_bar(range(sample_size)):
    while not use_all and file_idx not in sampled_indices:
        file_idx += 1
        frame = next(reader)
    species[i] = frame.get_atomic_numbers()
    positions[i] = frame.get_positions()
    forces[i] = frame.get_forces()
    energies[i] = frame.get_total_energy()


# positions are in Ang
forces = forces * 51.422086 * 23.060541  # Hartrees/Bohr --> eV/Ang --> kcal/mol/Ang
energies = energies * 627.50960  # Hartrees --> kcal/mol

# ----- Save data -----
print("Saving data", flush=True)

np.savez(
    npz_path,
    species = species,
    positions = positions,
    forces = forces,
    energies = energies,
)
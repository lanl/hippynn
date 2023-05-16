import os
import numpy as np

import ase.build
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

from hippynn.databases import AseDatabase


#### Setup 50 Steps of EMT evaluation on Aluminum bulk properties ####
# Doing this to generate ASE trajectory for loading as database #
calc = EMT() # Very cheap calculator suitable for Aluminum calculations

nrep = 5

# Build the atoms object
atoms = ase.build.bulk("Al", crystalstructure="fcc", a=4.05)
reps = nrep * np.eye(3, dtype=int)
atoms = ase.build.make_supercell(atoms, reps, wrap=True)
atoms.calc = calc


# Note - works with all ase-style formats: .traj, .db, .json, .xyz ....
trajectory_file = 'al_emt.traj'
print("Number of atoms:", len(atoms))

atoms.rattle(0.1)
MaxwellBoltzmannDistribution(atoms, temperature_K=500)
dyn = VelocityVerlet(atoms, 1 * units.fs, trajectory=trajectory_file)

# Takes ~45 seconds on 8-core intel Mac laptop
dyn.run(steps=50)

# Load in the database for Hippynn training:
dataParams = {'name': trajectory_file,
 'directory': os.path.abspath('.') + '/',
 'seed': 42,
 'test_size': 0.1,
 'valid_size': 0.1,
 'inputs': ['numbers', 'positions'],
 'targets': ['energy', 'forces'],
 'allow_unfound': True}

db = AseDatabase(**dataParams)


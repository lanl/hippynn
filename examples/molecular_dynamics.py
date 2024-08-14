"""
This script demonstrates how to use the custom MD module. 
It is intended to mirror the `ase_example.py` example,
using the custom MD module rather than ASE.

Before running this script, you must run 
`ani_aluminum_example.py` to train a model.

If a GPU is available, this script
will use it, and run a somewhat bigger system.
"""

import numpy as np
import torch
import ase
import time
from tqdm import trange

import ase.build
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from hippynn.graphs import physics, replace_node
from hippynn.graphs.predictor import Predictor
from hippynn.graphs.nodes.pairs import KDTreePairsMemory
from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory
from hippynn.molecular_dynamics.md import (
    Variable,
    NullUpdater,
    VelocityVerlet,
    MolecularDynamics,
)

# Adjust size of system depending on device
if torch.cuda.is_available():
    nrep = 25
    device = torch.device("cuda")
else:
    nrep = 10
    device = torch.device("cpu")

# Load the pre-trained model
try:
    with active_directory("TEST_ALUMINUM_MODEL", create=False):
        bundle = load_checkpoint_from_cwd(map_location="cpu")
except FileNotFoundError:
    raise FileNotFoundError("Model not found, run ani_aluminum_example.py first!")

# Adjust sign on force node (the HippynnCalculator does this automatically)
model = bundle["training_modules"].model
positions_node = model.node_from_name("coordinates")
energy_node = model.node_from_name("energy")
force_node = physics.GradientNode("force", (energy_node, positions_node), sign=-1)

# Replace pair-finder with more efficient one so that system can fit on GPU
old_pairs_node = model.node_from_name("PairIndexer")
species_node = model.node_from_name("species")
cell_node = model.node_from_name("cell")
new_pairs_node = KDTreePairsMemory("PairIndexer", parents=(positions_node, species_node, cell_node), skin=1.0, dist_hard_max=7.5)
replace_node(old_pairs_node, new_pairs_node)

model = Predictor(inputs=model.input_nodes, outputs=[force_node])
model.to(device)
model.to(torch.float64)

# Use ASE to generate initial positions and velocities
atoms = ase.build.bulk("Al", crystalstructure="fcc", a=4.05, orthorhombic=True)
reps = nrep * np.eye(3, dtype=int)
atoms = ase.build.make_supercell(atoms, reps, wrap=True)

print("Number of atoms:", len(atoms))

rng = np.random.default_rng(seed=0)
atoms.rattle(0.1, rng=rng)
MaxwellBoltzmannDistribution(atoms, temperature_K=500, rng=rng)

# Initialize MD variables
# NOTE: Setting the initial acceleration is only necessary to exactly match the results
# in `ase_example.py.` In general, it can be set to zero without impacting the statistics
# of the trajectory.
coordinates = torch.as_tensor(np.array(atoms.get_positions()), device=device).unsqueeze_(0)  # add batch axis
init_velocity = torch.as_tensor(np.array(atoms.get_velocities())).unsqueeze_(0)
cell = torch.as_tensor(np.array(atoms.get_cell()), device=device).unsqueeze_(0)
species = torch.as_tensor(np.array(atoms.get_atomic_numbers()), device=device).unsqueeze_(0)
mass = torch.as_tensor(atoms.get_masses()).unsqueeze_(0).unsqueeze_(-1)  # add a batch axis and a feature axis 
init_force = model(
    coordinates=coordinates,
    cell=cell,
    species=species,
)["force"]
init_force = torch.as_tensor(init_force)
init_acceleration = init_force / mass

# Define a position "Variable" and set updater to "VelocityVerlet"
position_variable = Variable(
    name="position",
    data={
        "position": coordinates,
        "velocity": init_velocity,
        "acceleration": init_acceleration,
        "mass": mass,
        "cell": cell,  # Optional. If added, coordinates will be wrapped in each step of the VelocityVerlet updater. Otherwise, they will be temporarily wrapped for model evaluation only and stored in their unwrapped form
    },
    model_input_map={
        "coordinates": "position",
    },
    device=device,
    updater=VelocityVerlet(force_db_name="force"),
)

# Define species and cell Variables
species_variable = Variable(
    name="species",
    data={"species": species},
    model_input_map={"species": "species"},
    device=device,
    updater=NullUpdater(),
)

cell_variable = Variable(
    name="cell",
    data={"cell": cell},
    model_input_map={"cell": "cell"},
    device=device,
    updater=NullUpdater(),
)

# Set up MD driver
emdee = MolecularDynamics(
    variables=[position_variable, species_variable, cell_variable],
    model=model,
)

# This Tracker imitates the Tracker from ase_example.py and is optional to use
class Tracker:
    def __init__(self):
        self.last_call_time = time.time()

    def update(self, diff_steps, data):
        now = time.time()
        diff = now - self.last_call_time
        self.n_atoms = data["position_position"].shape[-2]
        time_per_atom_step = diff / (self.n_atoms * diff_steps)
        self.last_call_time = now
        return time_per_atom_step

    def print(self, diff_steps=None, data=None):
        time_per_atom_step = self.update(diff_steps, data)
        """Function to print the potential, kinetic and total energy"""
        atoms.set_positions(np.array(data["position_position"][-1]))
        atoms.set_velocities(np.array(data["position_velocity"][-1]))
        print(
            "Performance:",
            round(1e6 * time_per_atom_step, 1),
            " microseconds/(atom-step)",
        )
        # epot = self.atoms.get_potential_energy() / len(self.atoms)
        ekin = atoms.get_kinetic_energy() / self.n_atoms
        # stress = self.atoms.get_stress()
        print("Energy per atom: Ekin = %.7feV (T=%3.0fK)" % (ekin, ekin / (1.5 * units.kB)))

# Run MD!
tracker = Tracker()
for i in trange(100):  # Run 2 ps
    n_steps = 20
    emdee.run(dt=1 * units.fs, n_steps=n_steps, record_every=n_steps)  # Run 20 fs
    tracker.print(n_steps, emdee.get_data())

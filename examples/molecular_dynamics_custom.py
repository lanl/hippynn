"""
This script demonstrates how to design your own
MD algorithm using the custom MD module. 

Before running this script, you must run 
`ani_aluminum_example.py` to train a model.
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
    StaticVariable,
    DynamicVariable,
    LangevinDynamics,
    VelocityVerlet,
    DynamicVariableUpdater,
    MolecularDynamics,
)

if torch.cuda.is_available():
    nrep = 10
    device = "cuda"
else:
    nrep = 10
    device = "cpu"

# Load the model
try:
    with active_directory("TEST_ALUMINUM_MODEL", create=False):
        bundle = load_checkpoint_from_cwd(map_location="cpu", restore_db=False)
except FileNotFoundError:
    raise FileNotFoundError("Model not found, run ani_aluminum_example.py first!")

# Adjust sign on force node (the HippynnCalculator does this automatically)
model = bundle["training_modules"].model
positions_node = model.node_from_name("coordinates")
energy_node = model.node_from_name("energy")
force_node = physics.GradientNode("force", (energy_node, positions_node), sign=-1)

# # Replace pair-finder with more efficient one (the HippynnCalculator also does this)
# old_pairs_node = model.node_from_name("PairIndexer")
# species_node = model.node_from_name("species")
# cell_node = model.node_from_name("cell")
# model.print_structure()
# # PositionsNode, Encoder, PaddingIndexer, CellNode
# new_pairs_node = KDTreePairsMemory("PairIndexer", parents=(positions_node, species_node, cell_node), skin=2, dist_hard_max=7.5)
# hippynn_node = model.node_from_name("HIPNN")
# print(hippynn_node.parents)
# replace_node(old_pairs_node, new_pairs_node)

model = Predictor(inputs=model.input_nodes, outputs=[force_node])
model.to(device)
model.to(torch.float64)

# Use ASE to generate initial positions and velocities
atoms = ase.build.bulk("Al", crystalstructure="fcc", a=4.05)
reps = nrep * np.eye(3, dtype=int)
atoms = ase.build.make_supercell(atoms, reps, wrap=True)

print("Number of atoms:", len(atoms))

rng = np.random.default_rng(seed=0)
atoms.rattle(0.1, rng=rng)
MaxwellBoltzmannDistribution(atoms, temperature_K=500, rng=rng)

# Initialize MD variables

coordinates=torch.tensor(np.array([atoms.get_positions()]), device=device)
cell=torch.tensor(np.array([atoms.get_cell()]), device=device)
species=torch.tensor(np.array([atoms.get_atomic_numbers()]), device=device)

position_variable = DynamicVariable(
    name="position",
    starting_values={
        "position": atoms.get_positions(),
        "velocity": atoms.get_velocities(),
        "acceleration": np.zeros_like(atoms.get_velocities()),
        "mass": atoms.get_masses(),
    },
    model_input_map={
        "coordinates": "position",
    },
    device=device,
)

### Design your own variable updater ###
class VelocityVerlet2(DynamicVariableUpdater):
    def __init__(self, force_key, param2):
        self.force_key = force_key
        self.param2 = param2

    def pre_step(self, dt):
        self.variable.data["velocity"] = (
            self.variable.data["velocity"]
            + 0.5 * dt * self.variable.data["acceleration"]
        )
        self.variable.data["position"] = (
            self.variable.data["position"] + self.variable.data["velocity"] * dt
        )

    def post_step(self, dt, model_outputs):
        self.variable.data["force"] = model_outputs[self.force_key].to(self.variable.device)
        if len(self.variable.data["force"].shape) == len(
            self.variable.data["mass"].shape
        ):
            self.variable.data["acceleration"] = (
                self.variable.data["force"].detach()
                / self.variable.data["mass"]
                * self.force_factor
            )
        else:
            self.variable.data["acceleration"] = (
                self.variable.data["force"].detach()
                / self.variable.data["mass"][..., None]
                * self.force_factor
            )
        self.variable.data["velocity"] = (
            self.variable.data["velocity"]
            + 0.5 * dt * self.variable.data["acceleration"]
        )

position_updater = VelocityVerlet(force_key="force")
position_variable.set_updater(position_updater)

species_variable = StaticVariable(
    name="species",
    values={"values": atoms.get_atomic_numbers()},
    model_input_map={"species": "values"},
    device=device,
)

cell_variable = StaticVariable(
    name="cell",
    values={"values": np.array(atoms.get_cell())},
    model_input_map={"cell": "values"},
    device=device,
)

# Set up and run MD
emdee = MolecularDynamics(
    dynamic_variables=[position_variable],
    static_variables=[species_variable, cell_variable],
    model=model,
)


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
        atoms.set_positions(data["position_position"][-1])
        atoms.set_velocities(data["position_velocity"][-1])
        print(
            "Performance:",
            round(1e6 * time_per_atom_step, 1),
            " microseconds/(atom-step)",
        )
        # epot = self.atoms.get_potential_energy() / len(self.atoms)
        ekin = atoms.get_kinetic_energy() / self.n_atoms
        # stress = self.atoms.get_stress()
        print(
            "Energy per atom: Ekin = %.7feV (T=%3.0fK)"
            % (ekin, ekin / (1.5 * units.kB))
        )

tracker = Tracker()
for i in trange(100):  # Run 2 ps
    n_steps = 20
    emdee.run(dt=1 * units.fs, n_steps=n_steps, record_every=n_steps)  # Run 20 fs
    tracker.print(n_steps, emdee.get_data())
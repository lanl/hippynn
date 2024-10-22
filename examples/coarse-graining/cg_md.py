import os

import numpy as np
import torch

from ase import units

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.graphs.predictor import Predictor
from hippynn.molecular_dynamics.md import (
    Variable,
    NullUpdater,
    LangevinDynamics,
    MolecularDynamics,
)
from hippynn.tools import active_directory

default_dtype=torch.float  
torch.set_default_dtype(default_dtype)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load initial conditions
training_data_file = os.path.join(os.pardir,os.pardir,os.pardir,"datasets","cg_methanol_trajectory.npz")

with np.load(training_data_file) as data:
    cell = torch.as_tensor(data["cells"][-1], dtype=default_dtype, device=device)[None,...]
    masses = torch.as_tensor(data["masses"][-1], dtype=default_dtype, device=device)[None,...]
    positions = torch.as_tensor(data["positions"][-1], dtype=default_dtype, device=device)[None,...]
    velocities = torch.as_tensor(data["velocities"][-1], dtype=default_dtype, device=device)[None,...]
    species = torch.as_tensor(data["species"][-1], dtype=torch.int, device=device)[None,...]
    
positions_variable = Variable(
    name="positions",
    data={
        "position": positions,
        "velocity": velocities,
        "mass": masses,
        "acceleration": torch.zeros_like(velocities),
        "cell": cell,
    },
    model_input_map={"positions": "position"},
    device=device,
)

position_updater = LangevinDynamics(
    force_db_name="forces",
    temperature=700,
    frix=6,
    force_units=units.kcal / units.mol / units.Ang,
    position_units=units.Ang,
    time_units=units.fs,
    seed=1993,
)
positions_variable.updater = position_updater

cell_variable = Variable(
    name="cell",
    data={"cell": cell},
    model_input_map={"cells": "cell"},
    device=device,
    updater=NullUpdater(),
)

species_variable = Variable(
    name="species",
    data={"species": species},
    model_input_map={"species": "species"},
    device=device,
    updater=NullUpdater(),
)

# Load model
with active_directory("model"):
    check = load_checkpoint_from_cwd(model_device=device, restart_db=False)

repulse = check["training_modules"].model.node_from_name("repulse")
energy = check["training_modules"].model.node_from_name("sys_energy")

model = Predictor.from_graph(
    check["training_modules"].model,
    additional_outputs=[
        repulse.mol_energies,
        energy,
    ],
)

model = Predictor.from_graph(check["training_modules"].model)

model.to(default_dtype)
model.to(device)

pairs = model.graph.node_from_name("pairs")
pairs.skin = 3 # see hippynn.graphs.nodes.pairs.KDTreePairsMemory documentation

# Run MD
with active_directory("md_results"):
    emdee = MolecularDynamics(
        variables=[positions_variable, species_variable, cell_variable],
        model=model,
    )

    emdee.run(dt=0.001, n_steps=20000)
    emdee.run(dt=0.001, n_steps=50000, record_every=50)

    data = emdee.get_data()
    np.savez("hippynn_cg_trajectory.npz",
        positions = data["positions_position"],
        velocities = data["positions_velocity"],
        masses = data["positions_mass"],
        accelerations = data["positions_acceleration"],
        cells = data["positions_cell"],
        unwrapped_positions = data["positions_unwrapped_position"],
        forces = data["positions_force"],
        species = data["species_species"],
    )
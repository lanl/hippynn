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
from hippynn.tools import active_directory, log_terminal
from hippynn.molecular_dynamics.rdf import calculate_rdf
from hippynn.molecular_dynamics.writers import write_extxyz


default_dtype=torch.float  
torch.set_default_dtype(default_dtype)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ase.units does not have a unit for picoseconds, so we'll create one
ps = units.fs * 1000

# Load initial conditions
training_data_file = os.path.join(os.pardir,os.pardir,os.pardir,"datasets","cg_methanol_trajectory.npz")

with np.load(training_data_file) as data:
    cell = torch.as_tensor(data["cells"][-1], dtype=default_dtype, device=device)[None,...]
    masses = torch.as_tensor(data["masses"][-1], dtype=default_dtype, device=device)[None,...]
    positions = torch.as_tensor(data["positions"][-1], dtype=default_dtype, device=device)[None,...]
    velocities = torch.as_tensor(data["velocities"][-1], dtype=default_dtype, device=device)[None,...]
    species = torch.as_tensor(data["species"][-1], dtype=torch.int, device=device)[None,...]
    aa_rdf_bins = data["rdf_bins"]
    aa_rdf_values = data["rdf_values"]
    
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
    temperature_K=700,
    frix=6, # this should be in 1/ps since the time unit we specify below is ps
    force_units=units.kcal / units.mol / units.Ang, # this needs to match the training data we used for our HIPNN model
    position_units=units.Ang, # this needs to match the training data we used for our HIPNN model
    time_units=ps, # we'll use this because the velocity data is in A/ps. Otherwise, it wouldn't matter as long as we are consistent
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
        repulse.mol_energy,
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
    with log_terminal("md_log.txt", "wt"):
        emdee = MolecularDynamics(
            variables=[positions_variable, species_variable, cell_variable],
            model=model,
        )

        try:
            emdee.run(dt=0.001, n_steps=200000) # time units should be whatever was specified in Variable Updater(s). For us that is ps
            emdee.run(dt=0.001, n_steps=500000, record_every=500)
        except KeyboardInterrupt:
            print(f"Keyboard interrupt. Saving results...")

        # Get results
        data = emdee.get_data()
        positions_result = data["positions_position"].detach().cpu().numpy().squeeze()
        velocities_result = data["positions_velocity"].detach().cpu().numpy().squeeze()
        masses_result = data["positions_mass"].detach().cpu().numpy().squeeze()
        accelerations_result = data["positions_acceleration"].detach().cpu().numpy().squeeze()
        cells_result = data["positions_cell"].detach().cpu().numpy().squeeze()
        unwrapped_positions_result = data["positions_unwrapped_position"].detach().cpu().numpy().squeeze()
        forces_result = data["positions_force"].detach().cpu().numpy().squeeze()
        species_result = data["species_species"].detach().cpu().numpy().squeeze()

        # Save results
        print(f"Saving results at {os.getcwd()}.", flush=True)
        npz_filename = "hippynn_cg_trajectory.npz"
        np.savez(npz_filename,
            positions = positions_result,
            velocities = velocities_result,
            masses = masses_result,
            accelerations = accelerations_result,
            cells = cells_result,
            unwrapped_positions = unwrapped_positions_result,
            forces = forces_result,
            species = species_result,
        )
        print(f"Data saved as .npz file.", flush=True)

        # Calculate CG RDF
        print("Calculating CG RDF...", flush=True)
        cg_rdf_bins, cg_rdf_values = calculate_rdf(positions=positions_result, cutoff=15, cells=cells_result)

        np.savez(npz_filename,
            positions = positions_result,
            velocities = velocities_result,
            masses = masses_result,
            accelerations = accelerations_result,
            cells = cells_result,
            unwrapped_positions = unwrapped_positions_result,
            forces = forces_result,
            species = species_result,
            rdf_bins = cg_rdf_bins,
            rdf_values = cg_rdf_values,
        )
        print(f"Updated data saved as .npz file.", flush=True)


        # Write extxyz file
        write_extxyz(
            "hippynn_cg_trajectory.extxyz",
            positions = positions_result,
            velocities = velocities_result,
            cells = cells_result,
            forces = forces_result,
            species = species_result,
        )
        print(f"Data saved as .extxyz file.", flush=True)

        # Plot CG vs AA RDFs
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(aa_rdf_bins, aa_rdf_values, label="AA", alpha=0.8)
        ax.plot(cg_rdf_bins, cg_rdf_values, label="CG", alpha=0.8)
        ax.set_xlabel('x ($\\AA$)')
        ax.set_ylabel('g(x) (-)')
        ax.set_xlim((0,15))
        ax.legend()
        fig.savefig("rdf_comparison_plot.pdf", bbox_inches='tight')
        plt.close(fig)
        print(f"RDF comparison plot saved.")

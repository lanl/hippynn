"""
This script is designed to accompany
Allen, A. E. A., Shinkle, E., Bujack, R., & Lubbers, N. (2025). Optimal
invariant bases for atomistic machine learning. arXiv preprint arXiv:2503.23515.
https://arxiv.org/abs/2503.23515

In the above paper, a methane dataset of ~7M configurations is used to test the expressive
capacity of different HIP-NN variants on different sizes of data. We find that for small 
dataset sizes, different HIP-NN architecture variants produce similar performance. As more 
data becomes available, HIP-HOP-NN is able to learn far more detail about geometries 
in the environment, significantly surpassing HIP-NN-TS and HIP-NN. (See Figure 4.)

BEFORE RUNNING:
1. Download the file methane.extxyz.gz from https://archive.materialscloud.org/records/kz78r-6nx43
2. Unzip the file: $ gunzip methane.extxyz.gz
3. Place the resulting file in a folder called datasets/ at the same level as hippynn/
   or change ``data_src`` below

NOTE: The methane.extxyz file will be very slow to read, so this script only uses 100,000
configurations. You can adjust this with the ``data_size`` variable. If you want to
read the methane.extxyz file repeatedly, I strongly suggest to first convert it into
another format (eg., .npz) that will be faster to read.
"""

import shutil
import os
import sys
from time import time
from math import log10
from itertools import product
from pathlib import Path

import ase
import torch
import numpy as np
import matplotlib.pyplot as plt

import hippynn
from hippynn.graphs import inputs, networks, targets, physics
from hippynn.graphs.nodes.networks import Hipnn, HipnnVec, HipnnQuad, HipHopnn
from hippynn.experiment import setup_training, train_model
from hippynn.graphs import loss
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
from hippynn.plotting import PlotMaker, Hist2D, SensitivityPlot
from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.pretraining import set_e0_values
from hippynn.tools import active_directory


# ----- User parameters -----
seed = 2025
data_src = Path(__file__).parents[2] / "datasets" / "methane.extxyz"
model_save_folder = Path(__file__).parents[1] / Path("TEST_METHANE_MODEL")

n_epochs = 10_000  # reduce to dececrease the run time of the script

# network_class = Hipnn # Original HIP-NN
# network_class = HipnnVec # HIP-NN-TS, l=1
# network_class = HipnnQuad # HIP-NN-TS, l=2
network_class = HipHopnn  # HIP-HOP model with defaults with n = 4 and l = 3

data_size = 100_000

# ----- Construct model -----
torch.random.manual_seed(seed)

species = inputs.SpeciesNode(name="species", db_name="numbers")
positions = inputs.PositionsNode(name="positions", db_name="positions")

network_params = {
    "possible_species": [0, 1, 6],
    "n_features": 32,
    "n_sensitivities": 20,
    "dist_soft_min": 0.4,
    "dist_soft_max": 9.0,
    "dist_hard_max": 10.3,  # diagonal of 6x6x6 cube
    "n_interaction_layers": 1,
    "n_atom_layers": 3,
}

network = network_class("network", (species, positions), module_kwargs=network_params)
henergy = targets.HEnergyNode(
    "HEnergy", network, db_name="energy", first_is_interacting=True
)

force = physics.GradientNode("forces", (henergy, positions), sign=-1, db_name="forces")

# define loss quantities
mse_force = loss.MSELoss.of_node(force)
rmse_force = mse_force ** (1 / 2)
mae_force = loss.MAELoss.of_node(force)
rsq_force = loss.Rsq.of_node(force)

rmse_energy = loss.MSELoss.of_node(henergy) ** (1 / 2)
mae_energy = loss.MAELoss.of_node(henergy)
rsq_energy = loss.Rsq.of_node(henergy)

mol_hier = loss.Mean.of_node(henergy.mol_hier)
atom_hier = loss.Mean.of_node(henergy.atom_hier)
old_hier = loss.Mean.of_node(henergy.hierarchicality)
rbar = henergy.batch_hier.pred  # loss.Mean.of_node(hierarchicality)

loss_energy = rmse_energy + mae_energy
loss_force = rmse_force + mae_force
loss_error = loss_energy + loss_force
l2_reg = 1e-6 * loss.l2reg(network)

loss_reg = l2_reg + 10 * rbar
total_loss = loss_error + loss_reg

validation_losses = {
    "T-RMSE": rmse_energy,
    "T-MAE": mae_energy,
    "T-RSQ": rsq_energy,
    "F-RMSE": rmse_force,
    "F-MAE": mae_force,
    "F-RSQ": rsq_force,
    "BHier": rbar,
    "MHier": mol_hier,
    "AHier": atom_hier,
    "OHier": old_hier,
    "Error Loss": loss_error,
    "L2": l2_reg,
    "Reg Loss": loss_reg,
    "Loss": total_loss,
}

plotters = [
    Hist2D.compare(henergy, saved="energy", shown=False),
    Hist2D.compare(force, saved="force", shown=False),
    SensitivityPlot(
        network.torch_module.sensitivity_layers[0],
        saved="sensitivity",
        shown=False,
    ),
]

plot_maker = PlotMaker(
    *plotters,
    plot_every=10,
)

training_modules, db_info = hippynn.experiment.assemble_for_training(
    total_loss, validation_losses, plot_maker=plot_maker
)

optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=2.5e-3)
scheduler = RaiseBatchSizeOnPlateau(
    optimizer=optimizer,
    max_batch_size=2048,
    patience=150,
    factor=0.5,
)

controller = PatienceController(
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=256,
    eval_batch_size=2048,
    max_epochs=n_epochs,
    stopping_key="T-MAE",
    termination_patience=300,
    fraction_train_eval=1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_params = hippynn.experiment.SetupParams(
    controller=controller,
    device=device,
)

training_modules, controller, metric_tracker = setup_training(
    training_modules=training_modules,
    setup_params=experiment_params,
)

# ----- Load data -----
iterable = ase.io.read(data_src, index=slice(0, data_size))
database = hippynn.databases.AseDatabaseIterable(
    iterable=iterable,
    seed=seed,
    pin_memory=False,
    test_size=0.1,
    valid_size=0.1,
    **db_info,
)

database.send_to_device(device)

set_e0_values(henergy, database, trainable_after=False)

# ----- Train model -----
with active_directory(model_save_folder):

    metric_tracker = train_model(
        training_modules=training_modules,
        database=database,
        controller=controller,
        metric_tracker=metric_tracker,
        callbacks=None,
        batch_callbacks=None,
        store_all_better=False,
        store_best=True,
        store_every=0,
        quiet=False,
    )

import os

import numpy as np
import torch

from hippynn.databases import NPZDatabase
from hippynn.experiment import SetupParams, setup_and_train
from hippynn.experiment.assembly import assemble_for_training
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
from hippynn.graphs import IdxType
from hippynn.graphs.nodes import loss
from hippynn.graphs.nodes.base.algebra import AddNode
from hippynn.graphs.nodes.indexers import acquire_encoding_padding
from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
from hippynn.graphs.nodes.networks import HipnnQuad
from hippynn.graphs.nodes.pairs import KDTreePairsMemory
from hippynn.graphs.nodes.physics import MultiGradientNode
from hippynn.graphs.nodes.targets import HEnergyNode
from hippynn.plotting import PlotMaker, Hist2D, SensitivityPlot
from hippynn.tools import active_directory

from repulsive_potential import RepulsivePotentialNode

training_data_file = os.path.join(os.pardir,os.pardir,os.pardir,"datasets","cg_methanol_trajectory.npz")

with np.load(training_data_file) as data:
    idx = np.where(data["rdf_values"] > 0.01)[0][0]
    repulsive_potential_taper_point = data["rdf_bins"][idx]
    repulsive_potential_strength = np.abs(data["forces"]).mean()

## Initialize needed nodes for network
# Network input nodes
species = SpeciesNode(name="species", db_name="species")
positions = PositionsNode(name="positions", db_name="positions")
cells = CellNode(name="cells", db_name="cells")

# Network hyperparameters
network_params = {
    "possible_species": [0,1],
    "n_features": 128,
    "n_sensitivities": 20,
    "dist_soft_min": 2.0,
    "dist_soft_max": 13.0,
    "dist_hard_max": 15.0,
    "n_interaction_layers": 1,
    "n_atom_layers": 3,
    "sensitivity_type": "inverse",
    "resnet": True,
}

# Species encoder
enc, pdx = acquire_encoding_padding([species], species_set=[0,1])

# Pair finder
pair_finder = KDTreePairsMemory(
    "pairs",
    (positions, enc, pdx, cells),
    dist_hard_max=network_params["dist_hard_max"],
    skin=0,
)

# HIP-NN-TS node with l=2
network = HipnnQuad(
    "HIPNN", (pdx, pair_finder), module_kwargs=network_params, periodic=True
)

# Network energy prediction
henergy = HEnergyNode("HEnergy", parents=(network,))

# Repulsive potential
repulse = RepulsivePotentialNode(
    "repulse", 
    (pair_finder, pdx), 
    taper_point=repulsive_potential_taper_point,
    strength=repulsive_potential_strength,
    dr=0.15,
    perc=0.05,
)

# Combined energy prediction
energy = AddNode(henergy.main_output, repulse.mol_energies)
energy.name = "energies"
energy._index_state = IdxType.Molecules

sys_energy = energy.main_output
sys_energy.name = "sys_energy"

# Force node
grad = MultiGradientNode("forces", energy, (positions,), signs=-1)
force = grad.children[0]
force.db_name = "forces"

## Define losses
force_rsq = loss.Rsq.of_node(force)
force_rmse = loss.MSELoss.of_node(force) ** (1 / 2)
force_mae = loss.MAELoss.of_node(force)
total_loss = force_rmse + force_mae

validation_losses = {
    "ForceRMSE": force_rmse,
    "ForceMAE": force_mae,
    "ForceRsq": force_rsq,
    "TotalLoss": total_loss,
}

plotters = [
    Hist2D.compare(force, saved="forces", shown=False),
    SensitivityPlot(
        network.torch_module.sensitivity_layers[0], saved="sensitivity", shown=False
    ),
]

plot_maker = PlotMaker(
    *plotters,
    plot_every=10,
)

## Build network
training_modules, db_info = assemble_for_training(
    total_loss, validation_losses, plot_maker=plot_maker
)

## Load training data
database = NPZDatabase(
    training_data_file, 
    seed=0, 
    **db_info, 
    valid_size=0.1, 
    test_size=0.1,
)

## Set up optimizer
optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)

scheduler = RaiseBatchSizeOnPlateau(
    optimizer=optimizer,
    max_batch_size=64,
    patience=10,
    factor=0.5,
)

controller = PatienceController(
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=1,
    fraction_train_eval=0.2,
    eval_batch_size=1,
    max_epochs=200,
    termination_patience=20,
    stopping_key="TotalLoss",
)

experiment_params = SetupParams(controller=controller)

## Train!
with active_directory("model"):
    metric_tracker = setup_and_train(
        training_modules=training_modules,
        database=database,
        setup_params=experiment_params,
    )
        
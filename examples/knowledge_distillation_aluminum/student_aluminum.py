""" 
    Example of Student Model for Knowledge Distillation Workflow used in
    Teacher-student training improves accuracy and efficiency of machine learning interatomic potentials
    https://arxiv.org/abs/2502.05379
    
    Before running this script, you must run 
    `ani_aluminum_example.py` to train the corresponding teacher model,
    and then run `gen_AE.py` to generate the augmented dataset. 
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("tag", type=str, help='Model Name')
parser.add_argument("seed", type=int, help='Seed for random number generator')
parser.add_argument("data_loc", type=str, help='Location of Augmented dataset. See `gen_AE.py` for further details.')
args = parser.parse_args()

import torch
# setting device on GPU if available, else resort to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed) # for reproducibility.

# Set Correct Back-end for matplotlib. 
import matplotlib
matplotlib.use('Agg')

import hippynn
hippynn.settings.PROGRESS=None
hippynn.settings.WARN_LOW_DISTANCES=False




# Set up Network. 
network_params = {
    "possible_species": [0, 13],
    "n_features": 24,
    "n_sensitivities": 10,
    "dist_soft_min": 1.5,
    "dist_soft_max": 7,
    "dist_hard_max": 7.5,
    "n_interaction_layers": 1,
    "n_atom_layers": 2,
    "sensitivity_type": "inverse",
    "resnet": True,
    "cusp_reg": 1e-8    
}

from hippynn.graphs import inputs, networks, targets, physics
species = inputs.SpeciesNode(db_name="Z")
positions = inputs.PositionsNode(db_name="R")
cell = inputs.CellNode(db_name="cell")
network = networks.HipnnVec("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)
henergy = targets.HEnergyNode("HEnergy", network, db_name="T")
hierarchicality = henergy.hierarchicality
sys_energy = henergy.mol_energy

# Auxiliary targets to train to. 
atom_energy = henergy.atom_energies
atom_energy.db_name = "AE"

force = physics.GradientNode("F", (henergy, positions), sign=1)
force.db_name = "F"

# Define Loss metrics 
from hippynn.graphs import loss

wE = 1
wF = 30
wAE = 100

mse_force = loss.MSELoss.of_node(force)
rmse_force = mse_force ** (1 / 2)
mae_force = loss.MAELoss.of_node(force)
rsq_force =  loss.Rsq.of_node(force)

rmse_energy = loss.MSELoss.of_node(henergy)**(1/2)
mae_energy = loss.MAELoss.of_node(henergy)
rsq_energy =  loss.Rsq.of_node(henergy)
rbar = loss.Mean.of_node(hierarchicality)

rmse_AE = loss.MSELoss.of_node(atom_energy)**(1/2)
mae_AE = loss.MAELoss.of_node(atom_energy)

loss_energy = rmse_energy + mae_energy
loss_force = rmse_force + mae_force
loss_AE = rmse_AE + mae_AE

loss_error = wE*loss_energy + wF*loss_force + wAE*loss_AE
l2_reg = 1e-6 * loss.l2reg(network)
loss_net = loss_error + rbar*0.01 + l2_reg

validation_losses = {
    "T-RMSE"      : rmse_energy,
    "T-MAE"       : mae_energy,
    "T-RSQ"       : rsq_energy,
    "F-RMSE"      : rmse_force,
    "F-MAE"       : mae_force,
    "F-RSQ"       : rsq_force,
    "AE-RMSE"     : rmse_AE,
    "AE-MAE"      : mae_AE,
    "T-Hier"      : rbar,
    "L2-Reg"      : l2_reg,
    "Error-Loss"  : loss_error,
    "Loss"        : loss_net,
}

# Plotting
from hippynn import plotting
plots_to_make = (
    plotting.Hist2D(henergy.main_output.true, henergy.main_output.pred, saved="energy.pdf", xlabel="True_Energy",ylabel="Pred_Energy"),
    plotting.Hist2D(force.true, force.pred, saved="force.pdf", xlabel="True_Force",ylabel="Pred_Force"),
    plotting.HierarchicalityPlot(hierarchicality.pred, sys_energy.pred - sys_energy.true, saved="HierPlot.pdf"),
    plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="Sensitivity0.pdf",shown=False),
)
if network_params["n_interaction_layers"] > 1:
    plots_to_make = plots_to_make + (
        plotting.SensitivityPlot(network.torch_module.sensitivity_layers[1], saved="Sensitivity1.pdf",shown=False),
    )
plot_maker = plotting.PlotMaker(*plots_to_make, plot_every=100)

# Assemble model 
training_modules, db_info = hippynn.experiment.assemble_for_training(loss_net, validation_losses, plot_maker=plot_maker)

# Database 
data_loc = args.data_loc
database = hippynn.databases.DirectoryDatabase(
    name='data-from-teacher_Al_',  # Prefix for arrays in the directory
    directory=data_loc,
    test_size=0.1, 
    valid_size=0.1, 
    seed=2024,      # Random seed for spliting data
    num_workers=4,
    ** db_info      # Adds the inputs and targets db_namesnames from the model as things to load
)
database.send_to_device()

# Fit the non-interacting energies by examining the database.
from hippynn.pretraining import hierarchical_energy_initialization
hierarchical_energy_initialization(henergy, database, peratom=False, energy_name="T", decay_factor=1e-2)

# Parameters describing the training procedure.
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController
optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)
batch_size = 64 
scheduler =  RaiseBatchSizeOnPlateau(
    optimizer=optimizer,
    max_batch_size=batch_size,
    patience=30,
    factor=0.5,
)
controller = PatienceController(
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=batch_size,
    eval_batch_size=batch_size,
    max_epochs=500,
    stopping_key='Loss',
    termination_patience=50,
)
experiment_params = hippynn.experiment.SetupParams(
    controller=controller,
    device=0,
    stopping_key='Loss',
)

# Training the network!
from hippynn.experiment import setup_and_train
dirname = f"{args.tag}"
with hippynn.tools.active_directory(dirname):
    with hippynn.tools.log_terminal("training_log.txt", 'wt'):
            setup_and_train(
                training_modules=training_modules,
                database=database,
                setup_params=experiment_params,
            )
"""

Example training script to predicted excited-states energies, transition dipoles, and
non-adiabatic coupling vectors (NACR)

The dataset used in this example can be found at https://doi.org/10.5281/zenodo.7076420.

This script is set up to assume the "release" folder from the zenodo record
 is placed in ../../datasets/azomethane/ relative to this script.

For more information on the modeling techniques, please see the paper:
Machine Learning Framework for Modeling Exciton-Polaritons in Molecular Materials
Li, et al. (2023)
https://arxiv.org/abs/2306.02523

"""
import json

import matplotlib
import numpy as np
import torch

import hippynn
from hippynn import plotting
from hippynn.experiment import setup_training, train_model
from hippynn.experiment.controllers import PatienceController, RaiseBatchSizeOnPlateau
from hippynn.graphs import inputs, loss, networks, physics, targets, excited

matplotlib.use("Agg")
# default types for torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)

hippynn.settings.WARN_LOW_DISTANCES = False
hippynn.settings.TRANSPARENT_PLOT = True

n_atoms = 10
n_states = 3
plot_frequency = 100
dipole_weight = 4
nacr_weight = 2
l2_weight = 2e-5

# Hyperparameters for the network
# Note: These hyperparameters were generated via
# a tuning algorithm, hence their somewhat arbitrary nature.
network_params = {
    "possible_species": [0, 1, 6, 7],
    "n_features": 30,
    "n_sensitivities": 28,
    "dist_soft_min": 0.7665723566179274,
    "dist_soft_max": 3.4134447177301515,
    "dist_hard_max": 4.6860240434651805,
    "n_interaction_layers": 3,
    "n_atom_layers": 3,
}
# dump parameters to the log file
print("Network parameters\n\n", json.dumps(network_params, indent=4))

with hippynn.tools.active_directory("TEST_AZOMETHANE_MODEL"):
    with hippynn.tools.log_terminal("training_log.txt", "wt"):
        # build network
        species = inputs.SpeciesNode(db_name="Z")
        positions = inputs.PositionsNode(db_name="R")
        network = networks.Hipnn("hipnn_model", (species, positions), module_kwargs=network_params)
        # add energy
        energy = targets.HEnergyNode("E", network, module_kwargs={"n_target": n_states + 1})
        mol_energy = energy.mol_energy
        mol_energy.db_name = "E"
        # add dipole
        charge = targets.HChargeNode("Q", network, module_kwargs={"n_target": n_states})
        dipole = physics.DipoleNode("D", (charge, positions), db_name="D")
        # add NACR
        nacr = excited.NACRMultiStateNode(
            "ScaledNACR",
            (charge, positions, energy),
            db_name="ScaledNACR",
            module_kwargs={"n_target": n_states},
        )
        # set up plotter
        plotter = []
        for node in [mol_energy, dipole, nacr]:
            plotter.append(plotting.Hist2D.compare(node, saved=True, shown=False))
        for i in range(network_params["n_interaction_layers"]):
            plotter.append(
                plotting.SensitivityPlot(
                    network.torch_module.sensitivity_layers[i],
                    saved=f"Sensitivity_{i}.pdf",
                    shown=False,
                )
            )
        plotter = plotting.PlotMaker(*plotter, plot_every=plot_frequency)
        # build the loss function
        validation_losses = {}
        # energy
        energy_rmse = loss.MSELoss.of_node(energy) ** 0.5
        validation_losses["E-RMSE"] = energy_rmse
        energy_mae = loss.MAELoss.of_node(energy)
        validation_losses["E-MAE"] = energy_mae
        energy_loss = energy_rmse + energy_mae
        validation_losses["E-Loss"] = energy_loss
        total_loss = energy_loss
        # dipole
        dipole_rmse = excited.MSEPhaseLoss.of_node(dipole) ** 0.5
        validation_losses["D-RMSE"] = dipole_rmse
        dipole_mae = excited.MAEPhaseLoss.of_node(dipole)
        validation_losses["D-MAE"] = dipole_mae
        dipole_loss = dipole_rmse / np.sqrt(3) + dipole_mae
        validation_losses["D-Loss"] = dipole_loss
        total_loss += dipole_weight * dipole_loss
        # nacr
        nacr_rmse = excited.MSEPhaseLoss.of_node(nacr) ** 0.5
        validation_losses["NACR-RMSE"] = nacr_rmse
        nacr_mae = excited.MAEPhaseLoss.of_node(nacr)
        validation_losses["NACR-MAE"] = nacr_mae
        nacr_loss = nacr_rmse / np.sqrt(3 * n_atoms) + nacr_mae
        validation_losses["NACR-Loss"] = nacr_loss
        total_loss += nacr_weight * nacr_loss
        # l2 regularization
        l2_reg = loss.l2reg(network)
        validation_losses["L2"] = l2_reg
        loss_regularization = l2_weight * l2_reg
        # add total loss to the dictionary
        validation_losses["Loss_wo_L2"] = total_loss
        validation_losses["Loss"] = total_loss + loss_regularization

        # set up experiment
        training_modules, db_info = hippynn.experiment.assemble_for_training(
            validation_losses["Loss"],
            validation_losses,
            plot_maker=plotter,
        )
        # set up the optimizer
        optimizer = torch.optim.AdamW(training_modules.model.parameters(), lr=1e-3)
        # use higher patience for production runs
        scheduler = RaiseBatchSizeOnPlateau(optimizer=optimizer, max_batch_size=2048, patience=10, factor=0.5)
        controller = PatienceController(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=32,
            eval_batch_size=2048,
            # use higher max_epochs for production runs
            max_epochs=100,
            stopping_key="Loss",
            fraction_train_eval=0.1,
            # use higher termination_patience for production runs
            termination_patience=10,
        )
        experiment_params = hippynn.experiment.SetupParams(controller=controller)

        # load database
        database = hippynn.databases.DirectoryDatabase(
            name="azo_",  # Prefix for arrays in the directory
            directory="../../../datasets/azomethane/release/training/",
            seed=114514,  # Random seed for splitting data
            **db_info,  # Adds the inputs and targets db_names from the model as things to load
        )
        # use 10% of the dataset just for quick testing purpose
        database.make_random_split("train", 0.07)
        database.make_random_split("valid", 0.02)
        database.make_random_split("test", 0.01)
        database.splitting_completed = True
        # split the whole dataset into train, valid, test in the ratio of 7:2:1
        # database.make_trainvalidtest_split(0.1, 0.2)

        # set up training
        training_modules, controller, metric_tracker = setup_training(
            training_modules=training_modules,
            setup_params=experiment_params,
        )
        # train model
        metric_tracker = train_model(
            training_modules,
            database,
            controller,
            metric_tracker,
            callbacks=None,
            batch_callbacks=None,
        )

    del network_params["possible_species"]
    network_params["metric"] = metric_tracker.best_metric_values
    network_params["avg_epoch_time"] = np.average(metric_tracker.epoch_times)
    network_params["Loss"] = metric_tracker.best_metric_values["valid"]["Loss"]

    with open("training_summary.json", "w") as out:
        json.dump(network_params, out, indent=4)

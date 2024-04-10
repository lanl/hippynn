"""

Example training to the ANI-aluminum dataset.

This script was designed for an external dataset available at
https://github.com/atomistic-ml/ani-al

Note: It is necessary to untar the h5 data files in ani-al/data/
before running this script.

For information on the dataset, see publication:
Smith, J.S., Nebgen, B., Mathew, N. et al.
Automated discovery of a robust interatomic potential for aluminum.
Nat Commun 12, 1257 (2021).
https://doi.org/10.1038/s41467-021-21376-0

Approx. CPU performance: 1 min/epoch at 10% of data for training. (Intel Core i9, 12 threads)
Approx. GPU performance: 10s/epoch at 90% of data for training. (NVIDIA GTX 1080Ti)
Note: This model is small and shallow, so the GPU performance is a bit better
with custom kernels -off-.

"""

import sys

sys.path.append("../../datasets/ani-al/readers/lib/")
import pyanitools  # Check if pyanitools is found early

import torch

torch.set_default_dtype(torch.float32)

import hippynn

netname = "TEST_ALUMINUM_MODEL"

if torch.cuda.is_available():
    # If GPU is available, we train to 80/10/10 split.
    test_size = 0.1
    valid_size = 0.1
    plot_every = 100
    # Approximate timing: 1 min/epoch
else:
    # If GPU is not available, we train to 0.09/0.01/0.9 split.
    test_size = 0.9
    valid_size = 0.01
    plot_every = 20
    # Approximate timing 10s/epoch

with hippynn.tools.active_directory(netname):
    with hippynn.tools.log_terminal("training_log.txt", "wt"):

        # Hyperparameters for the network
        network_params = {
            "possible_species": [0, 13],
            "n_features": 10,
            "n_sensitivities": 20,
            "dist_soft_min": 1.25,
            "dist_soft_max": 7,
            "dist_hard_max": 7.5,
            "n_interaction_layers": 1,
            "n_atom_layers": 3,
            "sensitivity_type": "inverse",
            "resnet": True,
        }
        print("Network hyperparameters:")
        print(network_params)

        from hippynn.graphs import inputs, networks, targets, physics

        species = inputs.SpeciesNode(db_name="species")
        positions = inputs.PositionsNode(db_name="coordinates")
        cell = inputs.CellNode(db_name="cell")

        network = networks.Hipnn("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)

        henergy = targets.HEnergyNode("HEnergy", network)
        sys_energy = henergy.mol_energy
        sys_energy.db_name = "energy"
        hierarchicality = henergy.hierarchicality
        hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
        force = physics.GradientNode("force", (sys_energy, positions), sign=1)
        force.db_name = "force"

        en_peratom = physics.PerAtom("T/Atom", sys_energy)
        en_peratom.db_name = "energyperatom"

        from hippynn.graphs import loss

        rmse_energy = loss.MSELoss(en_peratom.pred, en_peratom.true) ** (1 / 2)
        mae_energy = loss.MAELoss(en_peratom.pred, en_peratom.true)
        rsq_energy = loss.Rsq(en_peratom.pred, en_peratom.true)
        force_rsq = loss.Rsq(force.pred, force.true)
        force_mse = loss.MSELoss(force.pred, force.true)
        force_mae = loss.MAELoss(force.pred, force.true)
        force_rmse = force_mse ** (1 / 2)
        rbar = loss.Mean(hierarchicality.pred)
        l2_reg = loss.l2reg(network)

        loss_error = 1e2 * (rmse_energy + mae_energy) + (force_mae + force_rmse)
        loss_regularization = 1e-6 * l2_reg + rbar
        train_loss = loss_error + loss_regularization

        validation_losses = {
            "TpA-RMSE": rmse_energy,
            "TpA-MAE": mae_energy,
            "TpA-RSQ": rsq_energy,
            "ForceRMSE": force_rmse,
            "ForceMAE": force_mae,
            "ForceRsq": force_rsq,
            "T-Hier": rbar,
            "L2Reg": l2_reg,
            "Loss-Err": loss_error,
            "Loss-Reg": loss_regularization,
            "Loss": train_loss,
        }
        early_stopping_key = "Loss-Err"

        from hippynn import plotting

        plot_maker = plotting.PlotMaker(
            plotting.Hist2D.compare(sys_energy, saved=True),
            plotting.Hist2D.compare(en_peratom, saved=True),
            plotting.Hist2D.compare(force, saved=True),
            plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="sense_0.pdf"),
            plot_every=plot_every,
        )

        from hippynn.experiment.assembly import assemble_for_training

        training_modules, db_info = assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)

        model, loss_module, model_evaluator = training_modules

        #### Pre-processing of numpy arrays
        # Should be done with care, e.g. to avoid unit mismatches

        from hippynn.databases.h5_pyanitools import PyAniDirectoryDB

        torch.set_default_dtype(torch.float64)  # Temporary for data pre-processing
        database = PyAniDirectoryDB(
            directory="../../../datasets/ani-al/data/",
            seed=1001,  # Random seed for splitting data
            quiet=False,
            allow_unfound=True,  # allows post-loading preprocessing of arrays
            inputs=None,
            targets=None,
        )

        import ase

        energy_shift = 72  # eV
        arrays = database.arr_dict
        import numpy as np

        n_atoms = arrays["species"].astype(bool).astype(int).sum(axis=1)
        arrays["force"] = arrays["force"] * (ase.units.Hartree / ase.units.eV)
        arrays["energy"] = arrays["energy"] * (ase.units.Hartree / ase.units.eV)
        arrays["energy"] = arrays["energy"] + energy_shift * n_atoms
        arrays["energyperatom"] = arrays["energy"] / n_atoms

        # Adds the inputs and targets db_names from the model as things to load
        database.inputs = db_info["inputs"]
        database.targets = db_info["targets"]

        # Set dtypes back
        torch.set_default_dtype(torch.float32)
        for k, v in arrays.items():
            if v.dtype == np.float64:
                arrays[k] = v.astype(np.float32)

        database.make_trainvalidtest_split(test_size=test_size, valid_size=valid_size)

        # For datasets that fit in device memory, we can store
        # it on the device to avoid worry about the time to move data
        # to the GPU at each batch.
        database.send_to_device()
        ### End pre-processing

        # Now that we have a database and a model, we can
        # Fit the non-interacting energies by examining the database.
        from hippynn.pretraining import hierarchical_energy_initialization

        hierarchical_energy_initialization(henergy, database, peratom=True, energy_name="energyperatom", decay_factor=1e-2)

        from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController

        optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)

        scheduler = RaiseBatchSizeOnPlateau(
            optimizer=optimizer,
            max_batch_size=64,
            patience=25,
            factor=0.5,
        )

        controller = PatienceController(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=16,
            eval_batch_size=64,
            max_epochs=500,
            termination_patience=50,
            stopping_key=early_stopping_key,
        )

        from hippynn.experiment import SetupParams, setup_and_train

        experiment_params = SetupParams(controller=controller)

        print("Experiment Params:")
        print(experiment_params)

        # Parameters describing the training procedure.

        setup_and_train(
            training_modules=training_modules,
            database=database,
            setup_params=experiment_params,
        )

"""
Example training to the SNAP database for Tantalum.

This script was designed for an external dataset available at
https://github.com/FitSNAP/FitSNAP

For info on the dataset, see the following publication:
Thompson, A. P., Swiler, L. P., Trott, C. R., Foiles, S. M., & Tucker, G. J. (2015).
Spectral neighbor analysis method for automated generation of
quantum-accurate interatomic potentials.
Journal of Computational Physics, 285, 316-330
https://doi.org/10.1016/j.jcp.2014.12.018

"""

import torch

torch.set_default_dtype(torch.float32)

import hippynn.tools

netname = "TEST_TA_MODEL"

with hippynn.tools.active_directory(netname):
    with hippynn.tools.log_terminal("training_log.txt", "wt"):

        # Hyperparameters for the network
        network_params = {
            "possible_species": [0, 73],
            "n_features": 10,
            "n_sensitivities": 20,
            "dist_soft_min": 1.8,
            "dist_soft_max": 6.0,
            "dist_hard_max": 7.0,
            "n_interaction_layers": 1,
            "n_atom_layers": 1,
            "sensitivity_type": "inverse",
            "resnet": True,
        }
        print(network_params)

        from hippynn.graphs import inputs, networks, targets, physics
        from hippynn.graphs.nodes import pairs, indexers

        species = inputs.SpeciesNode(db_name="Species")
        positions = inputs.PositionsNode(db_name="Positions")
        cell = inputs.CellNode(db_name="Lattice")

        network = networks.Hipnn("HIPNN", (species, positions, cell), periodic=True, module_kwargs=network_params)
        henergy = targets.HEnergyNode("HEnergy", network)
        sys_energy = henergy.mol_energy
        sys_energy.db_name = "Energy"
        hierarchicality = henergy.hierarchicality
        hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
        force = physics.GradientNode("force", (sys_energy, positions), sign=-1)
        force.db_name = "Forces"

        en_peratom = physics.PerAtom("TperAtom", sys_energy)
        en_peratom.db_name = "EnergyPerAtom"

        from hippynn.graphs import loss

        def quantity_losses(quantity):
            return {
                f"rmse": loss.MSELoss.of_node(quantity) ** (1 / 2),
                f"mae": loss.MAELoss.of_node(quantity),
                f"Rsq": loss.Rsq.of_node(quantity),
            }

        losses = {
            "E": quantity_losses(sys_energy),
            "e": quantity_losses(en_peratom),
            "F": quantity_losses(force),
        }

        en_f_ratio = 1.0
        loss_error = en_f_ratio * (losses["e"]["mae"] + losses["e"]["rmse"]) + (
            losses["F"]["mae"] + losses["F"]["rmse"]
        )

        rbar = loss.Mean.of_node(hierarchicality)
        l2_reg = loss.l2reg(network)
        loss_regularization = 1e-4 * l2_reg + rbar  # L2 regularization and hierarchicality regularization

        train_loss = loss_error + loss_regularization

        validation_losses = {
            **{f"{q}-{ltype}": l for q, qloss in losses.items() for ltype, l in qloss.items()},
            "e-Hier": rbar,
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
            plotting.HierarchicalityPlot(hierarchicality.pred, sys_energy.pred - sys_energy.true, saved="HierPlot.pdf"),
            plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="Sensitivity.pdf"),
            plot_every=100,
        )

        from hippynn.experiment.assembly import assemble_for_training

        training_modules, db_info = assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)

        #### Pre-processing of numpy arrays
        # Should be done with care, e.g. to avoid unit mismatches

        from hippynn.databases.SNAPJson import SNAPDirectoryDatabase

        torch.set_default_dtype(torch.float64)  # Temporary for data pre-processing
        database = SNAPDirectoryDatabase(
            directory="../../../datasets/Ta_Linear_JCP2014/JSON/",
            seed=0,  # Random seed for splitting data
            quiet=False,
            allow_unfound=True,  # allows post-loading preprocessing of arrays
            inputs=None,
            targets=None,
        )

        import numpy as np

        arrays = database.arr_dict
        n_atoms = arrays["Species"].astype(bool).astype(int).sum(axis=1)
        arrays["EnergyPerAtom"] = arrays["Energy"] / n_atoms

        # Adds the inputs and targets from the model for the database to laod
        database.inputs = db_info["inputs"]
        database.targets = db_info["targets"]

        # Set dtypes back
        torch.set_default_dtype(torch.float32)
        for k, v in arrays.copy().items():
            if v.dtype == np.float64:
                arrays[k] = v.astype(np.float32)
            if v.dtype not in [np.float64, np.float32, np.int64]:
                del arrays[k]

        database.make_trainvalidtest_split(test_size=0.2, valid_size=0.4)
        ### End pre-processing

        # Now that we have a database and a model, we can
        # Fit the non-interacting energies by examining the database.
        from hippynn.pretraining import hierarchical_energy_initialization

        hierarchical_energy_initialization(henergy, database, peratom=True, energy_name="EnergyPerAtom", decay_factor=1e-2)
        # Freeze sensitivity layers
        for sense_layer in network.torch_module.sensitivity_layers:
            sense_layer.mu.requires_grad_(False)
            sense_layer.sigma.requires_grad_(False)
        del sense_layer

        from hippynn.experiment.assembly import precompute_pairs

        precompute_pairs(training_modules.model, database, n_images=4)
        training_modules, db_info = assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)
        database.inputs = db_info["inputs"]
        database.send_to_device()

        from hippynn.experiment.controllers import PatienceController
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-2)

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            patience=5,
            factor=0.5,
        )

        controller = PatienceController(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=4,
            eval_batch_size=64,
            max_epochs=500,
            termination_patience=20,
            fraction_train_eval=1.0,
            stopping_key=early_stopping_key,
        )

        experiment_params = hippynn.experiment.SetupParams(
            controller=controller,
        )
        print(experiment_params)

        # Parameters describing the training procedure.
        from hippynn.experiment import setup_and_train

        setup_and_train(
            training_modules=training_modules,
            database=database,
            setup_params=experiment_params,
        )

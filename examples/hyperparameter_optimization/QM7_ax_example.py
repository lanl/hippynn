"""
To obtain the data files needed for this example, use the script process_QM7_data.py,
also located in the parent folder. The script contains further instructions for use.
"""

PERFORM_PLOTTING = True  # Make sure you have matplotlib if you want to set this to TRUE

import os

#### Setup pytorch things
import torch

torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Don't try this if you want CPU training!

import hippynn

hippynn.settings.PROGRESS = None


def training(dist_soft_min, dist_soft_max, dist_hard_max):
    # Log the output of python to `training_log.txt`
    with hippynn.tools.log_terminal("training_log.txt", "wt"):

        # Hyperparameters for the network

        network_params = {
            "possible_species": [0, 1, 6, 7, 8, 16],  # Z values of the elements
            "n_features": 10,  # Number of neurons at each layer
            "n_sensitivities": 10,  # Number of sensitivity functions in an interaction layer
            "dist_soft_min": dist_soft_min,  #
            "dist_soft_max": dist_soft_max,
            "dist_hard_max": dist_hard_max,
            "n_interaction_layers": 1,  # Number of interaction blocks
            "n_atom_layers": 1,  # Number of atom layers in an interaction block
        }

        # Define a model

        from hippynn.graphs import inputs, networks, physics, targets

        # model inputs
        species = inputs.SpeciesNode(db_name="Z")
        positions = inputs.PositionsNode(db_name="R")

        # Model computations
        network = networks.Hipnn(
            "HIPNN", (species, positions), module_kwargs=network_params
        )
        henergy = targets.HEnergyNode("HEnergy", network)
        molecule_energy = henergy.mol_energy
        molecule_energy.db_name = "T"
        hierarchicality = henergy.hierarchicality

        # define loss quantities
        from hippynn.graphs import loss

        rmse_energy = loss.MSELoss.of_node(molecule_energy) ** (1 / 2)
        mae_energy = loss.MAELoss.of_node(molecule_energy)
        rsq_energy = loss.Rsq.of_node(molecule_energy)

        ### More advanced usage of loss graph

        pred_per_atom = physics.PerAtom(
            "PeratomPredicted", (molecule_energy, species)
        ).pred
        true_per_atom = physics.PerAtom(
            "PeratomTrue", (molecule_energy.true, species.true)
        )
        mae_per_atom = loss.MAELoss(pred_per_atom, true_per_atom)

        ### End more advanced usage of loss graph

        loss_error = rmse_energy + mae_energy

        rbar = loss.Mean.of_node(hierarchicality)
        l2_reg = loss.l2reg(network)
        loss_regularization = (
            1e-6 * l2_reg + rbar
        )  # L2 regularization and hierarchicality regularization

        train_loss = loss_error + loss_regularization

        # Validation losses are what we check on the data between epochs -- we can only train to
        # a single loss, but we can check other metrics too to better understand how the model is training.
        # There will also be plots of these things over time when training completes.
        validation_losses = {
            "T-RMSE": rmse_energy,
            "T-MAE": mae_energy,
            "T-RSQ": rsq_energy,
            "TperAtom MAE": mae_per_atom,
            "T-Hier": rbar,
            "L2Reg": l2_reg,
            "Loss-Err": loss_error,
            "Loss-Reg": loss_regularization,
            "Loss": train_loss,
        }
        early_stopping_key = "Loss-Err"

        if PERFORM_PLOTTING:

            from hippynn import plotting

            plot_maker = plotting.PlotMaker(
                # Simple plots which compare the network to the database
                plotting.Hist2D.compare(molecule_energy, saved=True),
                # Slightly more advanced control of plotting!
                plotting.Hist2D(
                    true_per_atom,
                    pred_per_atom,
                    xlabel="True Energy/Atom",
                    ylabel="Predicted Energy/Atom",
                    saved="PerAtomEn.pdf",
                ),
                plotting.HierarchicalityPlot(
                    hierarchicality.pred,
                    molecule_energy.pred - molecule_energy.true,
                    saved="HierPlot.pdf",
                ),
                plot_every=10,  # How often to make plots -- here, epoch 0, 10, 20...
            )
        else:
            plot_maker = None

        from hippynn.experiment import assemble_for_training

        # This piece of code glues the stuff together as a pytorch model,
        # dropping things that are irrelevant for the losses defined.
        training_modules, db_info = assemble_for_training(
            train_loss, validation_losses, plot_maker=plot_maker
        )
        training_modules[0].print_structure()

        max_batch_size = 12
        database_params = {
            "name": "qm7",  # Prefix for arrays in folder
            # for Ax, a relative path can be used
            # with Ray, an absolute path must be used
            "directory": "/path/to/dataset",
            "quiet": False,
            "test_size": 0.1,
            "valid_size": 0.1,
            "seed": 2001,
            # How many samples from the training set to use during evaluation
            **db_info,  # Adds the inputs and targets names from the model as things to load
        }

        from hippynn.databases import DirectoryDatabase

        database = DirectoryDatabase(**database_params)

        # Now that we have a database and a model, we can
        # Fit the non-interacting energies by examining the database.

        from hippynn.pretraining import hierarchical_energy_initialization

        hierarchical_energy_initialization(henergy, database, trainable_after=False)

        min_epochs = 50
        max_epochs = 800
        patience_epochs = 20

        from hippynn.experiment.controllers import (
            PatienceController,
            RaiseBatchSizeOnPlateau,
        )

        optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)

        scheduler = RaiseBatchSizeOnPlateau(
            optimizer=optimizer,
            max_batch_size=2048,
            patience=5,
        )

        controller = PatienceController(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=512,
            eval_batch_size=512,
            max_epochs=10,
            termination_patience=20,
            fraction_train_eval=0.1,
            stopping_key=early_stopping_key,
        )

        experiment_params = hippynn.experiment.SetupParams(
            controller=controller,
        )
        print(experiment_params)

        # Parameters describing the training procedure.
        from hippynn.experiment import setup_and_train

        metric_tracker = setup_and_train(
            training_modules=training_modules,
            database=database,
            setup_params=experiment_params,
        )

        return metric_tracker.best_metric_values

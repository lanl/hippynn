import torch

# Setup pytorch things
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Don't try this if you want CPU training!

import argparse


parser = argparse.ArgumentParser(prog="singlet_triplet_model")

parser.add_argument("-d", "--directory", action="store", type=str)
parser.add_argument("-n", "--name", action="store", type=str)

args = parser.parse_args()
db_path = args.directory
db_name = args.name

import matplotlib

matplotlib.use("agg")

import hippynn

netname = "TEST_singlet_triplet_model"

with hippynn.tools.active_directory(netname, create=True):
    with hippynn.tools.log_terminal("training_log.txt", "wt"):
        # Hyperparameters for the network
        network_params = {
            "possible_species": [0, 1, 6, 7, 8, 16, 17],
            "n_features": 96,  # was 60
            "n_sensitivities": 20,  # was 20
            "dist_soft_min": 0.8,  # qm7 1.7  qm9 .85  AL100 .85
            "dist_soft_max": 5.0,  # qm7 10.  qm9 5.   AL100 5.
            "dist_hard_max": 5.5,  # qm7 15.  qm9 7.5  AL100 7.5
            "n_interaction_layers": 2,  # was 2
            "n_atom_layers": 3,  # was 3
            "sensitivity_type": "inverse",
            "resnet": True,
        }

        # define a model with links to the database names
        from hippynn.graphs import inputs, networks, targets, physics

        species = inputs.SpeciesNode(db_name="Z")
        positions = inputs.PositionsNode(db_name="R")

        network = networks.HipnnVec("HIPNN", (species, positions), module_kwargs=network_params)

        henergy = targets.HEnergyNode("HEnergyS", network)
        singlet_energy = henergy.mol_energy
        hierarchicality = henergy.hierarchicality

        dscf = targets.LocalEnergyNode("DeltaSCF", network, first_is_interacting=False)
        excitation_energy = dscf.mol_energy
        excitation_energy.db_name = "excitation_energy"

        triplet_energy = excitation_energy + singlet_energy

        singlet_energy.set_dbname("singlet_T")
        singlet_energy.name = "singlet energy"

        triplet_energy = triplet_energy
        triplet_energy.set_dbname("triplet_T")
        triplet_energy.name = "triplet energy"

        forces_singlet = physics.GradientNode("gradientsS", (singlet_energy, positions), sign=-1)
        forces_singlet.set_dbname("singlet_Grad")

        forces_excitation = physics.GradientNode("gradientsDSCF", (excitation_energy, positions), sign=-1)
        forces_excitation.set_dbname("excitation_Grad")

        forces_triplet = forces_singlet + forces_excitation
        forces_triplet.name = "gradientsT"
        forces_triplet.set_dbname("triplet_Grad")

        from hippynn.graphs import loss

        rmse_singlet = loss.MSELoss.of_node(singlet_energy) ** (1 / 2)
        mae_singlet = loss.MAELoss.of_node(singlet_energy)
        rsq_singlet = loss.Rsq.of_node(singlet_energy)

        rmse_excitation = loss.MSELoss.of_node(excitation_energy) ** (1 / 2)
        mae_excitation = loss.MAELoss.of_node(excitation_energy)
        rsq_excitation = loss.Rsq.of_node(excitation_energy)

        rmse_triplet = loss.MSELoss.of_node(triplet_energy) ** (1 / 2)
        mae_triplet = loss.MAELoss.of_node(triplet_energy)
        rsq_triplet = loss.Rsq.of_node(triplet_energy)

        rmse_force_singlet = loss.MSELoss.of_node(forces_singlet) ** (1 / 2)
        mae_force_singlet = loss.MAELoss.of_node(forces_singlet)
        rsq_force_singlet = loss.Rsq.of_node(forces_singlet)

        rmse_force_excitation = loss.MSELoss.of_node(forces_excitation) ** (1 / 2)
        mae_force_excitation = loss.MAELoss.of_node(forces_excitation)
        rsq_force_excitation = loss.Rsq.of_node(forces_excitation)

        rmse_force_triplet = loss.MSELoss.of_node(forces_triplet) ** (1 / 2)
        mae_force_triplet = loss.MAELoss.of_node(forces_triplet)
        rsq_force_triplet = loss.Rsq.of_node(forces_triplet)

        rbar1 = loss.Mean.of_node(hierarchicality)
        l2_reg = loss.l2reg(network)

        force1_error = rmse_force_singlet + mae_force_singlet
        force2_error = rmse_force_excitation + mae_force_excitation
        energy1_error = rmse_singlet + mae_singlet
        energy2_error = rmse_excitation + mae_excitation

        loss_error = (energy1_error + energy2_error) / 10 + (force1_error + force2_error) / 30
        loss_regularization = 1e-6 * l2_reg + 0.1 * rbar1
        train_loss = loss_error + loss_regularization

        validation_losses = {
            "S-E-RMSE": rmse_singlet,
            "S-E-MAE": mae_singlet,
            "S-E-RSQ": rsq_singlet,
            "S-E-Hier": rbar1,
            "T-E-RMSE": rmse_triplet,
            "T-E-MAE": mae_triplet,
            "T-E-RSQ": rsq_triplet,
            "DSCF-E-RMSE": rmse_excitation,
            "DSCF-E-MAE": mae_excitation,
            "DSCF-E-RSQ": rsq_excitation,
            "S-GradRMSE": rmse_force_singlet,
            "S-GradMAE": mae_force_singlet,
            "S-GradRsq": rsq_force_singlet,
            "T-GradRMSE": rmse_force_triplet,
            "T-GradMAE": mae_force_triplet,
            "T-GradRsq": rsq_force_triplet,
            "DSCF-GradRMSE": rmse_force_excitation,
            "DSCF-GradMAE": mae_force_excitation,
            "DSCF-GradRsq": rsq_force_excitation,
            "L2Reg": l2_reg,
            "Loss-Err": loss_error,
            "Loss-Reg": loss_regularization,
            "Loss": train_loss,
        }
        early_stopping_key = "Loss-Err"

        # define plots to be made
        from hippynn import plotting

        def maghist(vec_prediction):
            true = physics.VecMag(vec_prediction.name + "mag-true", vec_prediction.true)
            pred = physics.VecMag(vec_prediction.name + "mag-pred", vec_prediction.pred)
            return plotting.Hist2D(true, pred, xlabel=true.name, ylabel=pred.name, saved=vec_prediction.name + ".pdf")

        makehist = lambda var: plotting.Hist2D.compare(var, saved=True)
        make_sense = lambda i, layer: plotting.SensitivityPlot(layer, saved="sense_{}.pdf".format(i))
        import itertools

        plot_maker = plotting.PlotMaker(
            *map(makehist, [singlet_energy, forces_singlet, triplet_energy, forces_excitation]),
            *itertools.starmap(make_sense, enumerate(network.torch_module.sensitivity_layers)),
            *map(maghist, [forces_singlet, forces_excitation]),
            plotting.HierarchicalityPlot(
                hierarchicality.pred, singlet_energy.pred - singlet_energy.true, saved="HierPlotS.pdf"
            ),
            plot_every=20,
        )

        from hippynn.experiment.assembly import assemble_for_training

        training_modules, db_variable_info = assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)
        model, loss_module, model_evaluator = training_modules

        db_inputs = {
            "name": db_name,
            "directory": db_path,
            "quiet": False,
            "split_seed": 8000,
            "test_size": 0.1,
            "valid_size": 0.1,
            **db_variable_info,  # adds inputs and targets into database
        }

        from hippynn.databases import DirectoryDatabase

        database = DirectoryDatabase(**db_inputs)

        # Now that we have a database and a model, we can
        # Fit the non-interacting energies by examining the database.
        from hippynn.pretraining import hierarchical_energy_initialization

        hierarchical_energy_initialization(henergy, database, energy_name="singlet_T", trainable_after=True)

        patience = 10
        batch_size = 512

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
        from hippynn.experiment.controllers import PatienceController

        controller = PatienceController(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=batch_size,
            eval_batch_size=batch_size,
            max_epochs=500,
            termination_patience=patience * 2,
            fraction_train_eval=0.1,
            stopping_key=early_stopping_key,
        )

        from hippynn.experiment import setup_and_train, SetupParams

        experiment_params = SetupParams(
            controller=controller, device=torch.device("cuda"), stopping_key=early_stopping_key
        )

        experiment_state = setup_and_train(
            training_modules=training_modules,
            database=database,
            setup_params=experiment_params,
        )

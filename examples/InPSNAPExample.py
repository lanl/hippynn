import numpy as np
import torch
torch.set_default_dtype(torch.float32)

import hippynn

netname = "TEST_INP_MODEL"

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
            "possible_species": [0, 49, 15],    # order here matters when mapping to elements in lammps
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

        species = inputs.SpeciesNode(db_name="Species")
        positions = inputs.PositionsNode(db_name="Positions")
        cell = inputs.CellNode(db_name="Lattice")

        network = networks.Hipnn("HIPNN", (species,positions,cell),module_kwargs=network_params,periodic=True)

        henergy = targets.HEnergyNode("HEnergy", network)
        sys_energy = henergy.mol_energy
        sys_energy.db_name = "Energy"
        hierarchicality = henergy.hierarchicality
        hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
        force = physics.GradientNode("force", (sys_energy, positions), sign=1)
        force.db_name = "Forces"

        en_peratom = physics.PerAtom("T/Atom", sys_energy)
        en_peratom.db_name = "EnergyPerAtom"

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
            "TpA-RMSE"    : rmse_energy,
            "TpA-MAE"     : mae_energy,
            "TpA-RSQ"     : rsq_energy,
            "ForceRMSE"   : force_rmse,
            "ForceMAE"    : force_mae,
            "ForceRsq"    : force_rsq,
            "T-Hier"      : rbar,
            "L2Reg"       : l2_reg,
            "Loss-Err"    : loss_error,
            "Loss-Reg"    : loss_regularization,
            "Loss"        : train_loss,
        }
        early_stopping_key = "Loss-Err"

        from hippynn import plotting

        plot_maker = plotting.PlotMaker(
            plotting.Hist2D.compare(sys_energy, saved=True),
            plotting.Hist2D.compare(en_peratom, saved=True),
            plotting.Hist2D.compare(force,saved=True),
            plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="sense_0.pdf"),
            plot_every=plot_every,
        )

        from hippynn.experiment.assembly import assemble_for_training

        training_modules, db_info = \
            assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)

        model, loss_module, model_evaluator = training_modules

        from hippynn.databases.SNAPJson import SNAPDirectoryDatabase
        torch.set_default_dtype(torch.float64)  # Temporary for data pre-processing
        database = SNAPDirectoryDatabase(
            directory="../../../datasets/InP_JPCA2020/JSON/",
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
        for k,v in arrays.copy().items():
            if v.dtype == np.float64:
                arrays[k] = v.astype(np.float32)
            if v.dtype not in [np.float64,np.float32,np.int64]:
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
        # precompute_pairs(training_modules.model, database,n_images=4)
        # training_modules, db_info = assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)
        # database.inputs = db_info["inputs"]
        database.send_to_device()

        from hippynn.experiment.controllers import PatienceController
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        optimizer = torch.optim.Adam(training_modules.model.parameters(),lr=1e-2)

        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      patience=5,
                                      factor=0.5,)

        controller = PatienceController(optimizer=optimizer,
                                        scheduler=scheduler,
                                        batch_size=4,
                                        eval_batch_size=64,
                                        max_epochs=500,
                                        termination_patience=20,
                                        fraction_train_eval=1.,
                                        stopping_key=early_stopping_key,
                                        )

        experiment_params = hippynn.experiment.SetupParams(
            controller=controller,
        )
        print(experiment_params)

        # Parameters describing the training procedure.
        from hippynn.experiment import setup_and_train

        setup_and_train(training_modules=training_modules,
                        database=database,
                        setup_params=experiment_params,
                        )

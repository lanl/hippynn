"""

This is a test script based on /examples/QM7_example.py which uses pytorch lightning to train.

"""

PERFORM_PLOTTING = True  # Make sure you have matplotlib if you want to set this to TRUE

#### Setup pytorch things
import torch

torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Don't try this if you want CPU training!

import hippynn


def main():
    hippynn.settings.WARN_LOW_DISTANCES = False

    # Note: these settings may need to be adjusted depending on the platform where
    # this code is run.
    n_devices = 2
    num_workers = 0
    multiprocessing_context = "fork"

    # Hyperparameters for the network
    netname = "TEST_LIGHTNING_MODEL"
    network_params = {
        "possible_species": [0, 1, 6, 7, 8, 16],  # Z values of the elements
        "n_features": 20,  # Number of neurons at each layer
        "n_sensitivities": 20,  # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 1.6,  #
        "dist_soft_max": 10.0,
        "dist_hard_max": 12.5,
        "n_interaction_layers": 2,  # Number of interaction blocks
        "n_atom_layers": 3,  # Number of atom layers in an interaction block
    }

    # Define a model

    from hippynn.graphs import inputs, networks, targets, physics

    # model inputs
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")

    # Model computations
    network = networks.HipnnVec("HIPNN", (species, positions), module_kwargs=network_params)
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

    pred_per_atom = physics.PerAtom("PeratomPredicted", (molecule_energy, species)).pred
    true_per_atom = physics.PerAtom("PeratomTrue", (molecule_energy.true, species.true))
    mae_per_atom = loss.MAELoss(pred_per_atom, true_per_atom)

    ### End more advanced usage of loss graph

    loss_error = rmse_energy + mae_energy

    rbar = loss.Mean.of_node(hierarchicality)
    l2_reg = loss.l2reg(network)
    loss_regularization = 1e-6 * l2_reg + rbar  # L2 regularization and hierarchicality regularization

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
            plotting.HierarchicalityPlot(hierarchicality.pred, molecule_energy.pred - molecule_energy.true, saved="HierPlot.pdf"),
            plot_every=10,  # How often to make plots -- here, epoch 0, 10, 20...
        )
    else:
        plot_maker = None

    from hippynn.experiment import assemble_for_training

    # This piece of code glues the stuff together as a pytorch model,
    # dropping things that are irrelevant for the losses defined.
    training_modules, db_info = assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)
    training_modules[0].print_structure()

    if num_workers > 0:
        dataloader_kwargs = dict(multiprocessing_context=multiprocessing_context, persistent_workers=True)
    else:
        dataloader_kwargs = None
    database_params = {
        "name": "qm7",  # Prefix for arrays in folder
        "directory": "../../datasets/qm7_processed",
        "quiet": False,
        "test_size": 0.1,
        "valid_size": 0.1,
        "seed": 2001,
        # How many samples from the training set to use during evaluation
        **db_info,  # Adds the inputs and targets names from the model as things to load
        "dataloader_kwargs": dataloader_kwargs,
        "num_workers": num_workers,
    }

    from hippynn.databases import DirectoryDatabase

    database = DirectoryDatabase(**database_params)

    # Now that we have a database and a model, we can
    # Fit the non-interacting energies by examining the database.

    from hippynn.pretraining import hierarchical_energy_initialization

    hierarchical_energy_initialization(henergy, database, trainable_after=False)

    from hippynn.experiment.controllers import PatienceController
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.5,
        patience=1,
    )

    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=16,  # start batch size
        eval_batch_size=16,
        max_epochs=3,
        termination_patience=10,
        fraction_train_eval=0.1,
        stopping_key=early_stopping_key,
    )

    experiment_params = hippynn.experiment.SetupParams(
        controller=controller,
    )

    from hippynn.experiment import HippynnLightningModule

    lightmod, datamodule = HippynnLightningModule.from_experiment_setup(training_modules, database, experiment_params)
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger

    logger = CSVLogger(save_dir=".", name=netname, flush_logs_every_n_steps=100)
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpointer = ModelCheckpoint(
        monitor=f"valid_{early_stopping_key}",
        save_last=True,
        save_top_k=5,
        every_n_epochs=1,
        every_n_train_steps=None,
    )

    from hippynn.experiment.lightning_trainer import LightingPrintStagesCallback

    cb = LightingPrintStagesCallback()  # include this callback if you aren't sure what stage of lightning is broken.

    # The default accelerator, 'auto' detects MPS on mac. hippynn doesn't work on MPS (yet).
    # So we set cpu here.
    trainer = pl.Trainer(
        accelerator="cpu",
        logger=logger,
        num_nodes=1,
        devices=n_devices,
        callbacks=[checkpointer],
        log_every_n_steps=1,
        max_epochs=-1,  # This is set this way because the hippynn controller should terminate training.
    )

    trainer.fit(
        model=lightmod,
        datamodule=datamodule,
    )
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()

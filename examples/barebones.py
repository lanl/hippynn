'''
To obtain the data files needed for this example, use the script process_QM7_data.py, 
also located in this folder. The script contains further instructions for use.
'''

import torch

# Setup pytorch things
torch.set_default_dtype(torch.float32)

import hippynn

netname = "TEST_BAREBONES_SCRIPT"

# Hyperparameters for the network
# These are set deliberately small so that you can easily run the example on a laptop or similar.
network_params = {
    "possible_species": [0, 1, 6, 7, 8, 16],  # Z values of the elements in QM7
    "n_features": 20,  # Number of neurons at each layer
    "n_sensitivities": 20,  # Number of sensitivity functions in an interaction layer
    "dist_soft_min": 1.6,  # qm7 is in Bohr!
    "dist_soft_max": 10.0,
    "dist_hard_max": 12.5,
    "n_interaction_layers": 2,  # Number of interaction blocks
    "n_atom_layers": 3,  # Number of atom layers in an interaction block
}

# Define a model
from hippynn.graphs import inputs, networks, targets, physics

species = inputs.SpeciesNode(db_name="Z")
positions = inputs.PositionsNode(db_name="R")

network = networks.Hipnn("hipnn_model", (species, positions), module_kwargs=network_params)
henergy = targets.HEnergyNode("HEnergy", network, db_name="T")
# hierarchicality = henergy.hierarchicality

# define loss quantities
from hippynn.graphs import loss

mse_energy = loss.MSELoss.of_node(henergy)
mae_energy = loss.MAELoss.of_node(henergy)
rmse_energy = mse_energy ** (1 / 2)

# Validation losses are what we check on the data between epochs -- we can only train to
# a single loss, but we can check other metrics too to better understand how the model is training.
# There will also be plots of these things over time when training completes.
validation_losses = {
    "RMSE": rmse_energy,
    "MAE": mae_energy,
    "MSE": mse_energy,
}

# This piece of code glues the stuff together as a pytorch model,
# dropping things that are irrelevant for the losses defined.
training_modules, db_info = hippynn.experiment.assemble_for_training(mse_energy, validation_losses)

# Go to a directory for the model.
# hippynn will save training files in the current working directory.
with hippynn.tools.active_directory(netname):
    # Log the output of python to `training_log.txt`
    with hippynn.tools.log_terminal("training_log.txt", "wt"):
        database = hippynn.databases.DirectoryDatabase(
            name="data-qm7",  # Prefix for arrays in the directory
            directory="../../../datasets/qm7_processed",
            test_size=0.1,  # Fraction or number of samples to test on
            valid_size=0.1,  # Fraction or number of samples to validate on
            seed=2001,  # Random seed for splitting data
            **db_info,  # Adds the inputs and targets db_names from the model as things to load
        )

        # Now that we have a database and a model, we can
        # Fit the non-interacting energies by examining the database.
        # This tends to stabilize training a lot.
        from hippynn.pretraining import hierarchical_energy_initialization

        hierarchical_energy_initialization(henergy, database, trainable_after=False)

        # Parameters describing the training procedure.
        from hippynn.experiment import setup_and_train

        experiment_params = hippynn.experiment.SetupParams(
            stopping_key="MSE",  # The name in the validation_losses dictionary.
            batch_size=12,
            optimizer=torch.optim.Adam,
            max_epochs=100,
            learning_rate=0.001,
        )
        setup_and_train(
            training_modules=training_modules,
            database=database,
            setup_params=experiment_params,
        )

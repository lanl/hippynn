"""
allegro_ag_example.py

Script trains a hippynn model based on Ag MD data.

This example uses the AseDatabase Loader and thus requires the `ase` package.

The database file is:
 - Ag_warm_nospin.xyz - https://archive.materialscloud.org/record/file?filename=Ag_warm_nospin.xyz&record_id=1387

This file was released in conjunction with
"Learning local equivariant representations for large-scale atomistic dynamics"
Musaelian et al. 2023 Nat. Comm.
https://doi.org/10.1038/s41467-023-36329-y

Timing:
    ~60 s/epoch@batch_size=128 on 4-core intel MacbookPro laptop.
    ~4s/epoch@batch_size=128 on M1Max MacbookPro laptop.

"""
import os
import torch
import ase.io
import time

import hippynn
from hippynn.experiment import SetupParams, setup_and_train, test_model
from hippynn.databases import AseDatabase

torch.set_default_dtype(torch.float32)
hippynn.settings.WARN_LOW_DISTANCES = False

max_epochs=500

network_params = {
    "possible_species": [0, 47],
    "n_features": 8,
    "n_sensitivities": 16,
    "dist_soft_min": 2.3,
    "dist_soft_max": 3.75,
    "dist_hard_max": 4,
    "n_interaction_layers": 1,
    "n_atom_layers": 3,
    "sensitivity_type": "inverse",
    "resnet": True,
}

early_stopping_key = "Loss-Err"
test_size = 0.1
valid_size = 0.1
dbname = '../../datasets/Ag_warm_nospin.xyz'
training_path = os.path.abspath('.') + '/'

def setup_network(network_params):

    # Hyperparameters for the network
    print("Network hyperparameters:")
    print(network_params)

    from hippynn.graphs import inputs, networks, targets, physics

    species = inputs.SpeciesNode(db_name="numbers")
    positions = inputs.PositionsNode(db_name="positions")
    cell = inputs.CellNode(db_name="cell")

    network = networks.HipnnQuad("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)

    henergy = targets.HEnergyNode("HEnergy", network)
    sys_energy = henergy.mol_energy
    sys_energy.db_name = "energy"
    hierarchicality = henergy.hierarchicality
    hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
    force = physics.GradientNode("forces", (sys_energy, positions), sign=-1)
    force.db_name = "forces"

    en_peratom = physics.PerAtom("T/Atom", sys_energy)
    en_peratom.db_name = "energy_per_atom"

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

    loss_error = 10 * (rmse_energy + mae_energy) + (force_mae + force_rmse)
    loss_regularization = 1e-6 * l2_reg + rbar
    train_loss = loss_error + loss_regularization

    # Factors of 1e3 for meV
    validation_losses = {
        "EpA-RMSE": 1e3*rmse_energy,
        "EpA-MAE": 1e3*mae_energy,
        "EpA-RSQ": rsq_energy,
        "ForceRMSE": 1e3*force_rmse,
        "ForceMAE": 1e3*force_mae,
        "ForceRsq": force_rsq,
        "T-Hier": rbar,
        "L2Reg": l2_reg,
        "Loss-Err": loss_error,
        "Loss-Reg": loss_regularization,
        "Loss": train_loss,
    }

    from hippynn.experiment.assembly import assemble_for_training
    training_modules, db_info = assemble_for_training(train_loss, validation_losses)
    
    return henergy, training_modules, db_info

def fit_model(training_modules,database):

    model, loss_module, model_evaluator = training_modules

    from hippynn.pretraining import hierarchical_energy_initialization
    hierarchical_energy_initialization(henergy, database, peratom=True, energy_name="energy_per_atom", decay_factor=1e-2)

    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController

    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)

    scheduler = RaiseBatchSizeOnPlateau(
        optimizer=optimizer,
        max_batch_size=32,
        patience=25,
        factor=0.5,
    )

    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=4,
        eval_batch_size=128,
        max_epochs=max_epochs,
        termination_patience=50,
        stopping_key=early_stopping_key,
    )


    experiment_params = SetupParams(controller=controller)

    print("Experiment Params:")
    print(experiment_params)

    # Parameters describing the training procedure.
    print(controller.current_epoch)
    sTime = time.time()

    setup_and_train(
        training_modules=training_modules,
        database=database,
        setup_params=experiment_params,
    )

    elTime = time.time()-sTime
    EpTime = elTime/controller.current_epoch


    with hippynn.tools.log_terminal("model_results.txt",'wt'):
        test_model(database, training_modules.evaluator, 128, "Final Training")
        print("FOM Average Epoch time: {:12.8f}".format(EpTime))
    

if __name__=="__main__":
    print("Setting up model.")
    henergy, training_modules, db_info = setup_network(network_params)
    
    print(db_info)
    print("Preparing dataset.")
    database = AseDatabase(directory=training_path,
        name=dbname,
        seed=1001,  # Random seed for splitting data
        quiet=False,
        pin_memory=False,
        test_size=test_size,
        valid_size=valid_size,
        **db_info)
    
    database.send_to_device() # Send to GPU if available

    print("Training model")
    with hippynn.tools.active_directory("model_files"):
        fit_model(training_modules,database)
    
    print("Writing test results")
    with hippynn.tools.log_terminal("model_results.txt",'wt'):
        test_model(database, training_modules.evaluator, 128, "Final Training")
    
    ## Possible to export lammps MLIPInterface for model if Lammps with MLIP Installed!
    # print("Exporting lammps interface")
    # first_frame = ase.io.read(dbname) # Reads in first frame only for saving box
    # ase.io.write('ag_box.data', first_frame, format='lammps-data')
    # from hippynn.interfaces.lammps_interface import MLIAPInterface
    # unified = MLIAPInterface(henergy, ["Ag"], model_device=torch.cuda.current_device())
    # torch.save(unified, "hippynn_lammps_model.pt")    
    print("All done.")

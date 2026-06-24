"""
Example script for training HIP-NN with charge-equilibration (ChEQ) model directly from the ANI2x_datasets h5 file.

This script was designed for an external dataset available at
https://zenodo.org/records/10108942
See convert_ani2x.py to prepare ANI2x_datasets h5 file for training

For info on the HIP-NN with ChEQ model, see the following publication:
Li, C.-H., Kaymak, M.C., Kulichenko, M. et al.
Shadow Molecular Dynamics with a Machine Learned Flexible Charge Potential
J. Chem. Theory Comput. 2025, 21, 7, 3658-3675
https://pubs.acs.org/doi/10.1021/acs.jctc.5c00062

"""
import os
import sys
import torch

from hippynn.graphs import find_unique_relative

import matplotlib
matplotlib.use('agg')

import hippynn

hippynn.custom_kernels.set_custom_kernels (False)

dataset_name = 'data-ANI-2x-B973c-def2mTZVP_all_D3_included_'    # Prefix for arrays in folder
dataset_path = os.path.join(os.path.dirname(__file__), "../../data/final_h5")

netname = 'TEST_all'
dirname = netname
if not os.path.exists(dirname):
    os.mkdir(dirname)
else:
    raise ValueError("Directory {} already exists!".format(dirname))
os.chdir(dirname)

dtype=torch.float32
torch.set_default_dtype(dtype)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with hippynn.tools.log_terminal("training_log.txt",'wt'):

    # Hyperparameters for the network
    # HIP-NN for charge-dependent part
    network_params = {
        "possible_species": [0,  1,  6,  7,  8,  9, 16, 17],   # Z values of the elements
        'n_features': 40,                                      # Number of neurons at each layer
        "n_sensitivities": 10,                                 # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.30,  
        "dist_soft_max": 3.0,  
        "dist_hard_max": 5.0, 
        "n_interaction_layers": 2,                             # Number of interaction blocks
        "n_atom_layers": 3,                                    # Number of atom layers in an interaction block
    }
    # HIP-NN for short-range, charge-independent part 
    network1_params = {
        "possible_species": [0,  1,  6,  7,  8,  9, 16, 17],   # Z values of the elements
        'n_features': 80,                                      # Number of neurons at each layer
        "n_sensitivities": 10,                                 # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.30,  
        "dist_soft_max": 3.0,  
        "dist_hard_max": 5.0, 
        "n_interaction_layers": 2,                             # Number of interaction blocks
        "n_atom_layers": 4,                                    # Number of atom layers in an interaction block
    }

    # Define a model

    from hippynn.graphs import inputs, networks, targets, physics

    species = inputs.SpeciesNode(db_name="species")

    positions = inputs.PositionsNode(db_name="coordinates")

    network = networks.Hipnn("HIPNN", (species, positions), module_kwargs = network_params)
    
    cheq = physics.ChEQNode("ChEQ", (network,), units={'energy':'kcal/mol', 'length':"Angstrom"}, lower_bound=0.01)

    network1 = networks.Hipnn("HIPNN2", network.parents, module_kwargs = network1_params)
    henergy = targets.HEnergyNode("HEnergy",network1)
    
    dipole = cheq.dipole

    molecule_energy = cheq.coul_energy + henergy.mol_energy 
    gradient = physics.GradientNode("Gradient", (molecule_energy, positions), sign=+1)

    molecule_energy.db_name="energies"
    gradient.db_name = "Grad"
    dipole.db_name = "dipole"

    hierarchicality = henergy.hierarchicality

    # define loss quantities
    from hippynn.graphs import loss

    rmse_energy = loss.MSELoss.of_node(molecule_energy) ** (1 / 2)
    rmse_grad = loss.MSELoss.of_node(gradient) ** (1 / 2)
    rmse_dipole = loss.MSELoss.of_node(dipole) ** (1 / 2)

    mae_energy = loss.MAELoss.of_node(molecule_energy)
    mae_grad = loss.MAELoss.of_node(gradient)
    mae_dipole = loss.MAELoss.of_node(dipole)

    rsq_energy = loss.Rsq.of_node(molecule_energy)
    rsq_grad = loss.Rsq.of_node(gradient)
    rsq_dipole = loss.Rsq.of_node(dipole)

    loss_error = 1.0 * (rmse_energy + mae_energy) + 100.0 * (rmse_grad + mae_grad) + \
                10000.0 * (rmse_dipole + mae_dipole) 

    rbar = loss.Mean.of_node(hierarchicality)
    l2_reg = loss.l2reg(network1) + 10.0 * loss.l2reg(network) 
    loss_regularization = 1e-1 * loss.Mean(l2_reg) + rbar #+ 10.0 * ((U-8.0) ** 2)   # L2 regularization and hierarchicality regularization

    train_loss = loss_error + loss_regularization

    # Validation losses are what we check on the data between epochs -- we can only train to
    # a single loss, but we can check other metrics too to better understand how the model is training.
    # There will also be plots of these things over time when training completes.
    validation_losses = {
        "T-RMSE"      : rmse_energy,
        "T-MAE"       : mae_energy,
        "T-RSQ"       : rsq_energy,
        "F-RMSE"      : rmse_grad,
        "F-MAE"       : mae_grad,
        "F-RSQ"       : rsq_grad,
        "Dipole-RMSE"      : rmse_dipole,
        "Dipole-MAE"       : mae_dipole,
        "Dipole-RSQ"       : rsq_dipole,
        "T-Hier"      : rbar,
        "L2Reg"       : l2_reg,
        "Loss-Err"    : loss_error,
        "Loss-Reg"    : loss_regularization,
        "Loss"        : train_loss,
    }
    early_stopping_key = "Loss-Err"

    from hippynn import plotting

    plot_maker = plotting.PlotMaker(
        # Simple plots which compare the network to the database

        plotting.Hist2D(molecule_energy.true, molecule_energy.pred,
                        xlabel="True En",ylabel="Predicted En",
                        saved="En.pdf"),
        
        plotting.Hist2D(gradient.true, gradient.pred,
                        xlabel="True gradient",ylabel="Predicted gradient",
                        saved="gradient.pdf"),
        
        plotting.Hist2D(dipole.true, dipole.pred,
                        xlabel="True dipole",ylabel="Predicted dipole",
                        saved="dipole.pdf"),

        plotting.HierarchicalityPlot(hierarchicality.pred,
                                        molecule_energy.pred - molecule_energy.true,
                                        saved="HierPlot.pdf"),
        plot_every=10,   # How often to make plots -- here, epoch 0, 10, 20...
    )

    from hippynn.experiment.assembly import assemble_for_training

    training_modules, db_info = \
        assemble_for_training(train_loss,validation_losses,plot_maker=plot_maker)
    training_modules[0].print_structure()


    database_params = {
        'name': dataset_name,       # Prefix for arrays in folder
        'directory': dataset_path,
        'quiet': False,             # Quiet==True: suppress info about loading database
        'seed': 8000,               # Random seed for data splitting
        **db_info                   # Adds the inputs and targets names from the model as things to load
    }

    from hippynn.databases import DirectoryDatabase
    database = DirectoryDatabase(**database_params)
    database.make_trainvalidtest_split(test_size=0.1,valid_size=0.1)

    from hippynn.pretraining import set_e0_values
    set_e0_values(henergy, database, energy_name="energies",trainable_after=False)

    init_lr =  1.0 * 1e-3
    optimizer = torch.optim.Adam(training_modules.model.parameters(),lr=init_lr)

    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController

    scheduler =  RaiseBatchSizeOnPlateau(optimizer=optimizer,
                                        max_batch_size=4096,
                                        patience=5,
                                        )

    controller = PatienceController(optimizer=optimizer,
                                    scheduler=scheduler,
                                    batch_size=4096,
                                    eval_batch_size=1024,
                                    max_epochs=1000,
                                    termination_patience=10,
                                    fraction_train_eval=0.1,
                                    stopping_key=early_stopping_key,
                                    )

    experiment_params = hippynn.experiment.SetupParams(
        controller = controller,
        device=('cuda'),
    )
    print(experiment_params)

    # Parameters describing the training procedure.
    from hippynn.experiment import setup_training

    training_modules, controller, metric_tracker  = setup_training(training_modules=training_modules,
                                                    setup_params=experiment_params)

    from hippynn.experiment import train_model
    store_all_better=False
    store_best=True
    store_every=1
    
    train_model(training_modules=training_modules,
                database=database,
                controller=controller,
                metric_tracker=metric_tracker,
                callbacks=None,
                batch_callbacks=None,
                store_all_better=store_all_better,
                store_best=store_best,
                store_every=store_every)


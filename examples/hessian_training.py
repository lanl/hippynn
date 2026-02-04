"""
Example script for training HIP-NN to energies, forces, and/or Hessian data from the RTP dataset h5 file.
Hessian-vector products (HVPs) can be used instead of full Hessians for faster training.

This script was designed for an external dataset available at
https://doi.org/10.6084/m9.figshare.29189858

For info on the dataset, see the following publication:
Rodriguez, A., Smith, J. S. & Mendoza‑Cortes, J. L.
Does Hessian Data Improve the Performance of Machine Learning Potentials? 
J. Chem. Theory Comput. (2025), pp. 6698–6710.
https://doi.org/10.1021/acs.jctc.5c00402

"""

import torch
import hippynn
import ase.units
from hippynn.graphs import inputs, networks, targets, physics
from hippynn.graphs.nodes.base import InputNode
from hippynn.graphs.nodes.loss import MSELoss, WeightedMSELoss
import os

def load_data(file, quiet=False):

    # Ensure total energies, forces, and Hessians loaded in float32/float64.
    torch.set_default_dtype(torch.float64)

    
    from hippynn.databases.h5_pyanitools import PyAniFileDB
    database = PyAniFileDB(
        #file="../../datasets/gau-files-12k.h5", # Change this to your h5 file dataset location
        file=file,
        species_key="species",  # or "Z", etc
        allow_unfound=False, # Note: file has extraneous "iform" variable which can't beloaded with hippynn; this skips that.
        seed=2025,
        quiet=quiet,
        pin_memory=torch.cuda.is_available(),
        **db_info
    )

    # compute (approximate) atomization energy by subtracting self energies

    # wb97x-6-31g*, G16. Doesn't need to be exact for most models, except atomization consistent.
    # # # Old values with singlet/triplet multiplicity only
    # # SELF_ENERGY_APPROX = {'C': -37.764142, 'H': -0.4993212, 'N': -54.4628753, 'O': -74.940046}
    # Recalculated with appropriate vacuum multiplicity
    SELF_ENERGY_APPROX = {"C": -37.8338334397, "H": -0.499321232710, "N": -54.5732824628, "O": -75.0424519384}
    SELF_ENERGY_APPROX = {k: SELF_ENERGY_APPROX[v] for k, v in zip([6, 1, 7, 8], "CHNO")}
    SELF_ENERGY_APPROX[0] = 0
    # Build a lookup tensor for self energies
    max_z = max(SELF_ENERGY_APPROX.keys()) + 1  # +1 in case max Z is the last index
    lookup_table = torch.zeros(max_z, dtype=torch.float32)
    for z, energy in SELF_ENERGY_APPROX.items():
        lookup_table[z] = energy

    database.arr_dict["species"] = database.arr_dict["species"].long()

    self_energy = lookup_table[database.arr_dict["species"]]
    self_energy = self_energy.sum(dim=1)
    database.arr_dict['energies'] = database.arr_dict["energies"] - self_energy


    # Convert from Hartree to kcal/mol
    kcalpmol = ase.units.kcal / ase.units.mol
    conversion = ase.units.Ha / kcalpmol
    for k in ["energies", "forces", "hessian"]:
        database.arr_dict[k] = database.arr_dict[k] * conversion

    for k, v in database.arr_dict.items():
        if v.dtype == torch.float64:
            database.arr_dict[k] = v.to(torch.float32)

    torch.set_default_dtype(torch.float32) # until self-energy subtracted
    return database




if __name__ == "__main__":
    # Active directory
    active_directory = "TEST_hvp_training_run"
    data_location = "../../datasets/OpenREACT-CHON-EFH/"

    # hippynn.custom_kernels.set_custom_kernels("triton")
    hippynn.settings.WARN_LOW_DISTANCES=False
    torch.set_default_dtype(torch.float32)

    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")


    # === HIPNN Model Parameters ===
    network_params = {
        "possible_species": [0, 1, 6, 7, 8],  # Z values of the elements in RTP dataset
        "n_features": 32,     # example value! production run might use more.
        "n_sensitivities": 16,
        "dist_soft_min": 0.8,
        "dist_soft_max": 5.5,
        "dist_hard_max": 6.0,
        "n_interaction_layers": 1, # production might use 2 interactions
        "n_atom_layers": 4,
        "l_max": 2,
        "n_max": 3,
    }

    # === Build the HIPNN Model ===
    
    # usual setup / targets
    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    hipnn_model = networks.HipHopnn("hipnn_model", (species, positions), module_kwargs=network_params)
    energy = targets.HEnergyNode("energy", hipnn_model, db_name="energies")
    force = physics.GradientNode("forces", (energy, positions), sign=-1, db_name="forces")

    
    hessian = physics.HessianNode("hessian", (energy,)) # hessian of model
    true_hessian = InputNode("hessian", db_name="hessian", index_state=physics.IdxType.Molecules) # actual hessian
    HVPVector = physics.HVPVectorNode("hvp_vector", (positions,), vector_type="random") # probing vector
    HVP = physics.HVPNode("hvp", (force, positions, HVPVector)) # efficient HVP product
    TrueHVP = physics.TrueHVPNode("true_hvp", (true_hessian, HVPVector)) # product with true hessian

    # === Losses ===
    # Coefficients derived from work on ANI, see manuscript.
    force_coefficient = 0.30
    hessian_coefficient = 0.09
    losses = {
        "E-RMSE": MSELoss.of_node(energy) ** (1 / 2),
        "F-RMSE": MSELoss.of_node(force) ** (1 / 2),
        # Uncomment (comment) the next line if you want (don't want) the network to evaluate the full hessian loss -- this is very slow.
        # "H-RMSE": WeightedMSELoss(hessian.hessian.pred, true_hessian.true, hessian.mask.pred) ** (1 / 2),
        "HVP-RMSE": WeightedMSELoss(HVP.hvp.pred, TrueHVP.pred, HVP.mask.pred) ** (1 / 2)
    }

    losses["LossTotal"] = losses["E-RMSE"] + force_coefficient * losses["F-RMSE"] + hessian_coefficient * losses["HVP-RMSE"]

    # This piece of code glues the stuff together as a pytorch model,
    # dropping things that are irrelevant for the losses defined.
    training_modules, db_info = hippynn.experiment.assemble_for_training(losses['LossTotal'], validation_losses=losses)

    database = load_data(os.path.join(data_location, "molecules-RTP.h5"))
    # Split the data into train, validation, and test sets
    database.make_trainvalidtest_split(test_size=0.1, valid_size=0.1)

    # Parameters describing the training procedure.
    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController
    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)
    batch_size = 64
    scheduler =  RaiseBatchSizeOnPlateau(
        optimizer=optimizer,
        max_batch_size=batch_size,
        patience=5, # might be longer for production run
        factor=0.5,
    )
    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        max_epochs=100, # might be longer for production run
        stopping_key="E-RMSE",
        termination_patience=20
    )
    experiment_params = hippynn.experiment.SetupParams(
        controller=controller,
    )

    with hippynn.tools.active_directory(active_directory):
        with hippynn.tools.log_terminal("training_log.txt", 'wt'):
            print("Data Loaded and Network set up! Just need to train... ")
            from hippynn.experiment import setup_and_train

            try:
                setup_and_train(
                    training_modules=training_modules,
                    database=database,
                    setup_params=experiment_params,
                )

            except KeyboardInterrupt:
                print("Now testing model, interrupt again if needed!")

            for data_file in ["molecules-IRC.h5", "molecules-NMS.h5"]:
                print("Testing:", data_file)
                path = os.path.join("..",data_location, data_file) # One more parent since we are in the model's directory now.
                database = load_data(path, quiet=True)
                database.split_the_rest("all data")
                metrics = hippynn.experiment.test_model(database,training_modules.evaluator,batch_size=64, when="Final Test")
                torch.save(metrics, "finalmetrics-" + data_file + ".pt")





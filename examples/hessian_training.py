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
from hippynn.graphs import GraphModule

# Active directory
active_directory = "hvp_training_run"

# hippynn.custom_kernels.set_custom_kernels("triton")
hippynn.settings.WARN_LOW_DISTANCES=False
torch.set_default_dtype(torch.float32)
torch.cuda.set_device("cuda:0")

# === Input Nodes ===
species = inputs.SpeciesNode(db_name="species")
positions = inputs.PositionsNode(db_name="coordinates")

# === HIPNN Model Parameters ===
network_params = {
    "possible_species": [0, 1, 6, 7, 8],  # Z values of the elements in RTP dataset
    "n_features": 16,
    "n_sensitivities": 16,
    "dist_soft_min": 0.8,
    "dist_soft_max": 4.5,
    "dist_hard_max": 6.0,
    "n_interaction_layers": 2,
    "n_atom_layers": 2,
}

# === Build the HIPNN Model ===
hipnn_model = networks.Hipnn("hipnn_model", (species, positions), module_kwargs=network_params)

# === Energy Node ===
energy = targets.HEnergyNode("energy", hipnn_model, db_name="energies")

## === Force Node ===
# gradients = physics.GradientNode("gradients", (energy, positions), sign=1, db_name="forces")
force = physics.GradientNode("forces", (energy, positions), sign=-1, db_name="forces")

# === Hessian Node ===
hessian = physics.HessianNode("hessian", (energy,))

# === True Hessian Node ===
true_hessian = InputNode("hessian", db_name="hessian", index_state=physics.IdxType.Molecules)

# === HVP Vector Node ===
HVPVector = physics.HVPVectorNode("hvp_vector", (positions,), vector_type="random")

# === HVP Node ===
HVP = physics.HVPNode("hvp", (force, positions, HVPVector))

# === True HVP Node ===
TrueHVP = physics.TrueHVPNode("true_hvp", (true_hessian, HVPVector))

# === Losses ===
force_coefficient = 0.30
hessian_coefficient = 0.09
losses = {
    "E-RMSE": MSELoss.of_node(energy) ** (1 / 2),
    "F-RMSE": MSELoss.of_node(force) ** (1 / 2),
    # "H-RMSE": WeightedMSELoss(hessian.hessian.pred, true_hessian.true, hessian.mask.pred) ** (1 / 2),
    "HVP-RMSE": WeightedMSELoss(HVP.hvp.pred, TrueHVP.pred, HVP.mask.pred) ** (1 / 2)
}

losses["LossTotal"] = losses["E-RMSE"] + force_coefficient * losses["F-RMSE"] + hessian_coefficient * losses["HVP-RMSE"]

validation_losses = {
    "E-RMSE": MSELoss.of_node(energy) ** (1 / 2),
    "F-RMSE": MSELoss.of_node(force) ** (1 / 2),
    # "H-RMSE": WeightedMSELoss(hessian.hessian.pred, true_hessian.true, hessian.mask.pred) ** (1 / 2),
    "HVP-RMSE": WeightedMSELoss(HVP.hvp.pred, TrueHVP.pred, HVP.mask.pred) ** (1 / 2)
}

# === Assemble Graph ===
graph = GraphModule((species, positions), [energy, force, HVP])

# This piece of code glues the stuff together as a pytorch model,
# dropping things that are irrelevant for the losses defined.
training_modules, db_info = hippynn.experiment.assemble_for_training(losses['LossTotal'], validation_losses)

# Ensure total energies, forces, and Hessians loaded in float32/float64.
torch.set_default_dtype(torch.float32)

from hippynn.databases.h5_pyanitools import PyAniFileDB
database = PyAniFileDB(
    file="../datasets/gau-files-12k.h5", # Change this to your h5 file dataset location
    species_key="species",  # or "Z"
    allow_unfound=True,
    seed=2025,
    **db_info
)

# wb97x-6-31g*, G16. Doesn't need to be exact for most models, except atomization consistent.
# # # Old values with singlet/triplet multiplicity only
# # SELF_ENERGY_APPROX = {'C': -37.764142, 'H': -0.4993212, 'N': -54.4628753, 'O': -74.940046}
# Recalculated with appropriate vacuum multiplicity
SELF_ENERGY_APPROX = {"C": -37.8338334397, "H": -0.499321232710, "N": -54.5732824628, "O": -75.0424519384}
SELF_ENERGY_APPROX = {k: SELF_ENERGY_APPROX[v] for k, v in zip([6, 1, 7, 8], "CHNO")}
SELF_ENERGY_APPROX[0] = 0

# compute (approximate) atomization energy by subtracting self energies

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
database.arr_dict["energies"] = database.arr_dict["energies"].float() * conversion
database.arr_dict["forces"] = database.arr_dict["forces"].float() * conversion
database.arr_dict["hessian"] = database.arr_dict["hessian"].float() * conversion

# Split the data into train, validation, and test sets
database.make_trainvalidtest_split(test_size=0.1, valid_size=0.1)

# Parameters describing the training procedure.
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController
optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-4)
batch_size = 300
scheduler =  RaiseBatchSizeOnPlateau(
    optimizer=optimizer,
    max_batch_size=batch_size,
    patience=8,
    factor=0.5,
)
controller = PatienceController(
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=batch_size,
    eval_batch_size=batch_size,
    max_epochs=1000,
    stopping_key="E-RMSE",
    termination_patience=20
)
experiment_params = hippynn.experiment.SetupParams(
    controller=controller,
    device="cuda:0"
)

with hippynn.tools.active_directory(active_directory):
    with hippynn.tools.log_terminal("training_log.txt", 'wt'):
        print("Data Loaded and Network set up! Just need to train... ")
        from hippynn.experiment import setup_and_train
        setup_and_train(
            training_modules=training_modules,
            database=database,
            setup_params=experiment_params,
        )
"""
This script is designed to accompany
Allen, A. E. A., Shinkle, E., Bujack, R., & Lubbers, N. (2025). 
Optimal invariant bases for atomistic machine learning. 
arXiv preprint arXiv:2503.23515. https://arxiv.org/abs/2503.23515

In the above paper, a methane dataset of ~7M configurations is used to test the expressive
capacity of different HIP-NN variants on different sizes of data. We find that for small 
dataset sizes, different HIP-NN architecture variants produce similar performance. As more 
data becomes available, HIP-HOP-NN is able to learn far more detail about geometries 
in the environment, significantly surpassing HIP-NN-TS and HIP-NN. (See Figure 4.)

BEFORE RUNNING:
1. Download the file methane.extxyz.gz from https://archive.materialscloud.org/records/kz78r-6nx43
2. Unzip the file: $ gunzip methane.extxyz.gz
3. Place the resulting file in a folder called datasets/ at the same level as hippynn/
   or change ``data_src`` below

NOTE: The methane.extxyz file will be very slow to read, so this script only uses first 1000
configurations for training and subsequent 80,000 for testing. You can adjust this with 
the ``data_size`` variable and by setting ``random_subset = True`` below. If you want to read
the methane.extxyz file repeatedly, I strongly suggest to first convert it into another format
(eg., .traj, .npz) that will be faster to read. You can do this by setting ``random_subset = True``
below, which will create a methane.traj file automatically for future use. But this conversion
may take a while (~1hr)."
"""

import os
import ase
import torch
import numpy as np
from pathlib import Path

import hippynn
from hippynn.graphs import inputs, targets, physics
from hippynn.graphs.nodes.networks import HipHopnn, Hipnn, HipnnVec, HipnnQuad
from hippynn.experiment import setup_training, train_model, test_model
from hippynn.graphs import loss
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
from hippynn.plotting import PlotMaker, Hist2D, SensitivityPlot
from hippynn.pretraining import set_e0_values
from hippynn.tools import active_directory

# ----- Constants -----
TOTAL_NUM_SAMPLES = 7_732_488 
TEST_SET_SIZE = 80_000
ENERGY_MEAN = -25042.327220945674

# ----- User parameters -----
seed = 2025
data_root = Path(__file__).parents[2] / "datasets"
data_src = data_root / "methane.extxyz"
processed_src = data_root / "methane.traj"
model_save_folder = Path(__file__).parents[1] / Path("TEST_METHANE_MODEL")
n_epochs = 10_000  # reduce to decrease the run time of the script
data_size = 1000
random_subset = False  # whether to use a random subset of data or the first data_size sample
# network_class = Hipnn # Original HIP-NN
# network_class = HipnnVec # HIP-NN-TS, l=1
# network_class = HipnnQuad # HIP-NN-TS, l=2
network_class = HipHopnn  # HIP-HOP
hiphop_l_max = 3 # these will not be used if network_class != HipHopnn
hiphop_n_max = 4 # these will not be used if network_class != HipHopnn

network_params = {
    "possible_species": [0, 1, 6],
    "n_features": 32,
    "n_sensitivities": 20,
    "dist_soft_min": 0.4,
    "dist_soft_max": 9.0,
    "dist_hard_max": 10.3,  # diagonal of 6x6x6 cube
    "n_interaction_layers": 1,
    "n_atom_layers": 3,
}

if network_class == HipHopnn:
    network_params.update(
        {
            "l_max": hiphop_l_max,
            "n_max": hiphop_n_max,
        }
    )

# ----- Prepare data -----
def prepare_data(data_src, train_size, test_size, random_subset=random_subset): 
    assert os.path.exists(data_src), f"Data source {data_src} does not exist! Please download methane.extxyz from https://archive.materialscloud.org/records/kz78r-6nx43 !"
    train_dict = {
        "numbers": [],
        "positions": [],
        "forces": [],
        "energy": [],
    }
    test_dict = {
        "numbers": [],
        "positions": [],
        "forces": [],
        "energy": [],
    }
    if not random_subset:
        generator = ase.io.iread(data_src)
    else: 
        if os.path.exists(processed_src):
            data_src = processed_src  # use the .traj file if it exists
        else:
            print(f"Converting {data_src} to {processed_src} for faster reading next time, this may take a while (~1hr)...")
            frames = ase.io.read(data_src, index=':')
            ase.io.write(processed_src, frames)
            data_src = processed_src
            del frames
        indices = np.arange(TOTAL_NUM_SAMPLES)
        np.random.seed(seed)
        np.random.shuffle(indices)
        with ase.io.trajectory.Trajectory(processed_src) as raw_data:
            generator = [raw_data[i] for i in indices[:data_size + TEST_SET_SIZE]]

    for idx, frame in enumerate(generator):
        species = frame.get_atomic_numbers()
        positions = frame.get_positions()
        forces = frame.get_forces()
        energy = frame.get_total_energy()
        # Change units 
        forces = forces * 51.42208619083232 * 23.060541945329334  # Hartrees/Bohr --> eV/Ang --> kcal/mol/Ang
        energy = energy * 627.5096080305927  # Hartrees --> kcal/mol
        # Shift energy mean 
        energy -= ENERGY_MEAN
        if idx < train_size:
            train_dict["numbers"].append(species)
            train_dict["positions"].append(positions)
            train_dict["forces"].append(forces)
            train_dict["energy"].append(energy)
        elif idx < train_size + test_size:
            test_dict["numbers"].append(species)
            test_dict["positions"].append(positions)
            test_dict["forces"].append(forces)
            test_dict["energy"].append(energy)
        else:
            break
    # Convert to arrays
    for key in train_dict:
        if key != "numbers": 
            train_dict[key] = np.array(train_dict[key], dtype=np.float32)
            test_dict[key] = np.array(test_dict[key], dtype=np.float32)
    return train_dict, test_dict

# ----- Construct model -----
torch.random.manual_seed(seed)

species = inputs.SpeciesNode(name="species", db_name="numbers")
positions = inputs.PositionsNode(name="positions", db_name="positions")

network = network_class("network", (species, positions), module_kwargs=network_params)
henergy = targets.HEnergyNode(
    "HEnergy", network, db_name="energy", first_is_interacting=True
)

force = physics.GradientNode("forces", (henergy, positions), sign=-1, db_name="forces")

# define loss quantities
mse_force = loss.MSELoss.of_node(force)
rmse_force = mse_force ** (1 / 2)
mae_force = loss.MAELoss.of_node(force)
rsq_force = loss.Rsq.of_node(force)

rmse_energy = loss.MSELoss.of_node(henergy) ** (1 / 2)
mae_energy = loss.MAELoss.of_node(henergy)
rsq_energy = loss.Rsq.of_node(henergy)

loss_energy = rmse_energy + mae_energy
loss_force = rmse_force + mae_force
loss_error = loss_energy + loss_force
l2_reg = 1e-6 * loss.l2reg(network)

total_loss = loss_error + l2_reg

validation_losses = {
    "T-RMSE": rmse_energy,
    "T-MAE": mae_energy,
    "T-RSQ": rsq_energy,
    "F-RMSE": rmse_force,
    "F-MAE": mae_force,
    "F-RSQ": rsq_force,
    "Error Loss": loss_error,
    "L2": l2_reg,
    "Loss": total_loss,
}

plotters = [
    Hist2D.compare(henergy, saved="energy", shown=False),
    Hist2D.compare(force, saved="force", shown=False),
    SensitivityPlot(
        network.torch_module.sensitivity_layers[0],
        saved="sensitivity",
        shown=False,
    ),
]

plot_maker = PlotMaker(
    *plotters,
    plot_every=10,
)

training_modules, db_info = hippynn.experiment.assemble_for_training(
    total_loss, validation_losses, plot_maker=plot_maker
)

optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=2.5e-3)
scheduler = RaiseBatchSizeOnPlateau(
    optimizer=optimizer,
    max_batch_size=2048,
    patience=150,
    factor=0.5,
)

controller = PatienceController(
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=256,
    eval_batch_size=2048,
    max_epochs=n_epochs,
    stopping_key="T-MAE",
    termination_patience=300,
    fraction_train_eval=1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_params = hippynn.experiment.SetupParams(
    controller=controller,
    device=device,
)

training_modules, controller, metric_tracker = setup_training(
    training_modules=training_modules,
    setup_params=experiment_params,
)


# ----- Load data -----
train_dict, test_dict = prepare_data(data_src, 
                                     data_size,
                                     TEST_SET_SIZE)

train_database = hippynn.databases.Database(
    arr_dict=train_dict,
    seed=seed,
    pin_memory=True,
    test_size=0.1,
    valid_size=0.1,
    **db_info,
)
train_database.send_to_device(device)

test_database = hippynn.databases.Database(
    arr_dict=test_dict,
    seed=seed + 1,
    pin_memory=True,
    **db_info,
)
test_database.split_the_rest("test")
test_database.send_to_device(device)

set_e0_values(henergy, train_database, trainable_after=False)

# ----- Train model -----
with active_directory(model_save_folder):

    metric_tracker = train_model(
        training_modules=training_modules,
        database=train_database,
        controller=controller,
        metric_tracker=metric_tracker,
        callbacks=None,
        batch_callbacks=None,
        store_all_better=False,
        store_best=True,
        store_every=0,
        quiet=False,
    )

    # ----- Evaluate model -----
    evaluator = training_modules.evaluator
    best_model = metric_tracker.best_model
    if best_model:
        evaluator.model.load_state_dict(best_model)

    print("Testing model...")
    torch.cuda.empty_cache()
    test_model(
        test_database,
        evaluator,
        when="FinalTraining",
        batch_size=controller.eval_batch_size,
        metric_tracker=metric_tracker,
    )

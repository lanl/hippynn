"""
This script is designed to accompany
Allen, A. E. A., Shinkle, E., Bujack, R., & Lubbers, N. (2025). Optimal
invariant bases for atomistic machine learning. arXiv preprint arXiv:2503.23515.
https://arxiv.org/abs/2503.23515

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

NOTE: The methane.extxyz file will be very slow to read, so this script only uses 100,000
configurations. You can adjust this with the ``data_size`` variable. If you want to
read the methane.extxyz file repeatedly, I strongly suggest to first convert it into
another format (eg., .npz) that will be faster to read.
"""

import os
import wandb

import torch

import hippynn
from hippynn.graphs import inputs, targets, physics
from hippynn.graphs.nodes.networks import HipHopnn
from hippynn.experiment import setup_training, train_model
from hippynn.graphs import loss
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
from hippynn.plotting import PlotMaker, Hist2D, SensitivityPlot
from hippynn.pretraining import set_e0_values
from hippynn.tools import active_directory
import argparse


class TomasWandbLogger():
    def __init__(self, metric_tracker, wandb_run):
        self.metric_tracker = metric_tracker
        self.run = wandb_run

    def __call__(self, epoch, better_model):
        tracker = self.metric_tracker
        assert epoch == (tracker.current_epoch - 1)
        current_metrics = tracker.epoch_metric_values[epoch]
        for split, metrics in current_metrics.items():
            for key, value in metrics.items():
                self.run.log({f"{split}-{key}": value}, step=epoch)
        for key, value in tracker.best_metric_values.items():
            self.run.summary[f"Best-{key}"] = value

# ----- User parameters -----
def get_parameters():
    """
    Parse command line arguments for the methane configuration experiment.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Methane configuration experiment with HIP-NN variants")
    
    parser.add_argument('--seed', type=int, default=2025, 
                       help='Random seed for reproducibility')
    parser.add_argument('-t', '--train-file-name', type=str, default='./datasets/train_emily',
                       help='Path to training data file (defaults to root/train_data_emily.npz)')
    parser.add_argument('--data-split', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--model-save-folder', type=str, default='./test_methane_model',
                       help='Directory to save model (defaults to root/TEST_METHANE_MODEL)')
    parser.add_argument('--n-epochs', type=int, default=10000,
                       help='Maximum number of training epochs')
    parser.add_argument('--data-size', type=int, default=1000,
                       help='Number of configurations to use from dataset')
    parser.add_argument('--emily-test-data', type=str, default="/Users/karella/Projects/hippynn/test_data_emily.npz",)
    parser.add_argument('--my-test-data', type=str, default="/Users/karella/Projects/hippynn/my_test_data_emily.npz",)
    args = parser.parse_args()
    
    args.train_file = f"{args.train_file_name}_{args.num_samples}_{args.data_split}.npz"
    assert os.path.exists(args.train_file), FileNotFoundError(f"Training file {args.train_file} does not exist.")
    if args.model_save_folder is None:
        args.model_save_folder = os.path.join(args.root, "TEST_METHANE_MODEL")
    return args

# Get parameters from command line
params = get_parameters()

# Extract individual parameters
seed = params.seed
train_data = f"{params.train_file_name}_{params.num_samples}_{params.data_split}.npz"
model_save_folder = params.model_save_folder
n_epochs = params.n_epochs
network_class = HipHopnn  # HIP-HOP model with defaults with n = 4 and l = 3
data_size = params.data_size
test_data = params.test_data

# ----- Construct model -----
torch.random.manual_seed(seed)
wandb_settings = wandb.Settings(_disable_stats=True, save_code=False, _disable_meta=True,x_save_requirements=False)

with wandb.init(project="methane-hiphop", settings=wandb_settings, entity="karella", config=params.__dict__) as wandb_run:

    species = inputs.SpeciesNode(name="species", db_name="numbers")
    positions = inputs.PositionsNode(name="positions", db_name="positions")

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
    for key, val in network_params.items():
        wandb_run.config[key] = val

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

    mol_hier = loss.Mean.of_node(henergy.mol_hier)
    atom_hier = loss.Mean.of_node(henergy.atom_hier)
    old_hier = loss.Mean.of_node(henergy.hierarchicality)
    rbar = henergy.batch_hier.pred  # loss.Mean.of_node(hierarchicality)

    loss_energy = rmse_energy + mae_energy
    loss_force = rmse_force + mae_force
    loss_error = loss_energy + loss_force
    l2_reg = 1e-6 * loss.l2reg(network)

    loss_reg = l2_reg + 10 * rbar
    total_loss = loss_error + loss_reg

    validation_losses = {
        "T-RMSE": rmse_energy,
        "T-MAE": mae_energy,
        "T-RSQ": rsq_energy,
        "F-RMSE": rmse_force,
        "F-MAE": mae_force,
        "F-RSQ": rsq_force,
        "BHier": rbar,
        "MHier": mol_hier,
        "AHier": atom_hier,
        "OHier": old_hier,
        "Error Loss": loss_error,
        "L2": l2_reg,
        "Reg Loss": loss_reg,
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

    # Wandb logger
    wandb_callback = TomasWandbLogger(metric_tracker, wandb_run=wandb_run)


    database = hippynn.databases.NPZDatabase(
        file=train_data,
        test_size=0.1,  # Fraction or number of samples to test on -> this should be reduced closed to zero
        valid_size=0.1,  # Fraction or number of samples to validate on
        seed=seed,  # Random seed for spliting data
        # num_workers=2,
        pin_memory=False,
        **db_info,  # Adds the inputs and targets db_namesnames from the model as things to load
    )
    database.send_to_device(device)

    set_e0_values(henergy, database, trainable_after=False)

    # ----- Train model -----
    with active_directory(model_save_folder):

        metric_tracker = train_model(
            training_modules=training_modules,
            database=database,
            controller=controller,
            metric_tracker=metric_tracker,
            callbacks=[wandb_callback],
            batch_callbacks=None,
            store_all_better=False,
            store_best=True,
            store_every=0,
            quiet=False,
        )
    # Load a test file
    evaluator = training_modules.evaluator
    best_model = metric_tracker.best_model
    if best_model:
        print("Reverting to best model found.")
        evaluator.model.load_state_dict(best_model)
    evaluator.model.eval()
    
    test_database = hippynn.databases.NPZDatabase(
    file=test_data,
    seed=seed,
    pin_memory=True,
    **evaluator.db_info,
    )
    
    test_database.split_the_rest("all")
    data_generator = test_database.make_generator("all", "eval",
                                              batch_size=controller.eval_batch_size)

    metrics = evaluator.evaluate(data_generator, eval_type="all")
    
    # ---- Log Wandb test metrics ----
    print("Emily test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}" )
        wandb_run.summary[f"Test-{key}"] = value
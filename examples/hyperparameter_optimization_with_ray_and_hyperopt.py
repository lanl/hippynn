''' 
We use the Ray Tune python package to perform hyperparameter optimization on a HIPNN model for QM7 
using the Tree-structured Parzen Estimators (TPE) method (see <https://arxiv.org/abs/2304.11127>
for a good overview of the algorithm).

More information about Ray can be found at <https://docs.ray.io/>. We will only utilize Ray Tune. 
Ray can be downloaded via pip (pip install "ray[tune]") or via conda through the conda-forge channel 
(conda install conda-forge::ray-tune). 

We will use utilize the HyperOpt python package via a Ray wrapper. More information about HyperOpt can 
be found at <https://hyperopt.github.io/hyperopt/>. HyperOpt implements the TPE algorithm. HyperOpt 
must be installed via pip (pip install -U hyperopt).

To obtain the data files needed for this example, use the script process_QM7_data.py, also located in 
this folder. The script contains further instructions for use.
'''

import os

import ray
from ray import tune, air
from ray.tune.search.hyperopt import HyperOptSearch

from tqdm.autonotebook import trange
import numpy as np
import torch

import hippynn
from hippynn.graphs import inputs, networks, targets, physics
from hippynn.experiment import setup_training, train_model
from hippynn.graphs import loss
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
from hippynn.databases.ondisk import DirectoryDatabase
from hippynn.experiment import assemble_for_training
from hippynn.pretraining import hierarchical_energy_initialization

if torch.cuda.is_available(): # we'll run more and bigger trials if GPU resources are available
    USE_GPU = True
    n_splits_per_test = 5 # create this many models with the same hyperparameters and return the average value of their losses to the optimizer
    n_trials = 30 # number of hyperparameter combinations to test
else:
    USE_GPU = False
    n_splits_per_test = 2 # create this many models with the same hyperparameters and return the average value of their losses to the optimizer
    n_trials = 5 # number of hyperparameter combinations to test

hippynn.settings.PROGRESS = None # writing progress bars to file takes time and memory
hippynn.settings.WARN_LOW_DISTANCES = False

# First we create a function that we will hand to Ray for the optimization. The function will take a single input which
# is a dictionary containing names and values for the hyperparameters we want to optimize. The function will create and 
# train a HIPNN network (or networks) using those hyperparameters. The output of the function should be a dictionary 
# which contains as one of its entries the value of the loss function we want Ray to minimize for the trained network. 

abs_path_to_data = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "datasets", "qm7_processed"))

results_dir = os.path.join(os.path.abspath(""), "TEST_QM7_HYPER_OPT")

def build_and_train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_results = []

    for i in trange(n_splits_per_test):
        torch.random.manual_seed(i)
        network_params = {
            "possible_species": [0, 1, 6, 7, 8, 16],
            "n_features": config["n_features"],
            "n_sensitivities": config["n_sensitivities"],
            "dist_soft_min": 1.6,  #
            "dist_soft_max": 10.0,
            "dist_hard_max": 12.5,
            "n_interaction_layers": config["n_int_layers"],
            "n_atom_layers": config["n_atom_layers"],
        }
        max_epochs = 2000

        # model inputs
        species = inputs.SpeciesNode(db_name="Z")
        positions = inputs.PositionsNode(db_name="R")
        network = networks.HipnnQuad("HIPNN", (species, positions), module_kwargs=network_params)
        henergy = targets.HEnergyNode("HEnergy", network)
        molecule_energy = henergy.mol_energy
        molecule_energy.db_name = "T"
        hierarchicality = henergy.hierarchicality
        rmse_energy = loss.MSELoss.of_node(molecule_energy) ** (1 / 2)
        mae_energy = loss.MAELoss.of_node(molecule_energy)
        rsq_energy = loss.Rsq.of_node(molecule_energy)

        pred_per_atom = physics.PerAtom("PeratomPredicted", (molecule_energy, species)).pred
        true_per_atom = physics.PerAtom("PeratomTrue", (molecule_energy.true, species.true))
        mae_per_atom = loss.MAELoss(pred_per_atom, true_per_atom)

        lambda_error = 1/230 # I do not remember where these values came from
        lambda_l2 = 1e-6
        lambda_R = 1e-2

        loss_error = rmse_energy + mae_energy
        rbar = loss.Mean.of_node(hierarchicality)
        l2_reg = loss.l2reg(network)

        train_loss = lambda_error * loss_error + lambda_l2 * l2_reg + lambda_R * rbar

        validation_losses = {
            "T-RMSE": rmse_energy,
            "T-MAE": mae_energy,
            "T-RSQ": rsq_energy,
            "TperAtom MAE": mae_per_atom,
            "T-Hier": rbar,
            "L2Reg": l2_reg,
            "Loss-Err": loss_error,
            "Loss": train_loss,
        }
        early_stopping_key = "T-MAE"

        # This piece of code glues the stuff together as a pytorch model,
        # dropping things that are irrelevant for the losses defined.
        training_modules, db_info = assemble_for_training(train_loss, validation_losses)

        database_params = {
            "name": "qm7",
            "directory": abs_path_to_data,
            "quiet": True,
            "test_size": 0.2,
            "valid_size": 0.1,
            "seed": i,
            **db_info,
        }

        database = DirectoryDatabase(**database_params)
        database.send_to_device(device)

        hierarchical_energy_initialization(henergy, database, trainable_after=False)

        optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=config["lr"])
        scheduler = RaiseBatchSizeOnPlateau(
            optimizer=optimizer,
            max_batch_size=config["batch_size"] * 4,
            patience=config["patience"],
        )

        controller = PatienceController(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=config["batch_size"],
            eval_batch_size=4 * config["batch_size"],
            max_epochs=max_epochs,
            stopping_key=early_stopping_key,
            termination_patience=2 * config["patience"],
            fraction_train_eval=1,
        )

        experiment_params = hippynn.experiment.SetupParams(
            controller=controller,
            device=device,
        )

        training_modules, controller, metric_tracker = setup_training(
            training_modules=training_modules,
            setup_params=experiment_params,
        )

        metric_tracker = train_model(
            training_modules=training_modules,
            database=database,
            controller=controller,
            metric_tracker=metric_tracker,
            callbacks=None,
            batch_callbacks=None,
            store_all_better=False,
            store_best=True,
            store_every=0,
        )

        metrics_results.append(metric_tracker.other_metric_values['FinalTraining'])

    results_summary = dict()

    for dataset_type in metrics_results[0].keys(): # 'train', 'valid', 'test'
        results_summary[dataset_type] = dict()
        for metric in metrics_results[0][dataset_type]:
            results_summary[dataset_type][metric] = np.mean([result[dataset_type][metric] for result in metrics_results])

    results_summary["all"] = metrics_results

    return results_summary

# Now we will set up Ray to do the hyperparameter tuning
# Define hyperparameter ranges
param_space = {
    "batch_size": tune.randint(20, 101), # sample integer uniformly from 20 (inclusive) to 101 (exclusive)
    "n_atom_layers": tune.randint(1,6),
    "n_int_layers": tune.randint(1,3),
    "n_features": tune.randint(10,181),
    "n_sensitivities": tune.randint(10,31),
    "lr": tune.loguniform(1e-5, 1e-1),
    "patience": tune.randint(10, 201),
}

ray.init(
    log_to_driver=False # if set to 'True', all model training output will be printed to screen
    )

# Initialize search algorithm
hyperopt = HyperOptSearch(metric="valid/Loss", mode="min")

if USE_GPU:
    # We specify the number of GPUs per trial. If sufficient resources are present, 
    # multiple trials will be run in parallel
    trainable = tune.with_resources(build_and_train_model, resources={"gpu": 1})
else:
    trainable = build_and_train_model

run_config = air.RunConfig(
        storage_path=results_dir,
        log_to_file=True,
    )

tune_config = tune.TuneConfig(search_alg=hyperopt, num_samples=n_trials)

# Initialize tuner
tuner = tune.Tuner(
    trainable=trainable,
    tune_config=tune_config,
    param_space=param_space,
    run_config=run_config,
)

# Tune!
results = tuner.fit()

# Process, save and print results
print("Getting results..")
df = results.get_dataframe()
df.to_csv(os.path.join(tuner.get_results().experiment_path, 'results.csv'))

print(f"\nBest config:")
for key, value in results.get_best_result(metric='valid/T-MAE', mode='min').config.items():
    print(f"  {key}: {value}")
print(f"\nFinal results on test set (averaged over {n_splits_per_test} models):")
metrics = results.get_best_result(metric="valid/T-MAE", mode="min").metrics
for metric_name in ['T-RMSE', 'T-MAE', 'T-RSQ']:
    metric_values = [metric_dict['test'][metric_name] for metric_dict in metrics['all']]
    print(f"{metric_name}: {np.mean(metric_values):.2f} " + u"\u00B1" + f" {np.std(metric_values):.2f}")
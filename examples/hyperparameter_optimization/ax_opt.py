#!/usr/bin/env python3
# fmt: off
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=all
#SBATCH -J hyperopt
#SBATCH --qos=long
#SBATCH -o run.log
import os
import sys
import warnings

# SLURM copies the script to a tmp folder
# so to find the local package `QM7_ax_example` we need add cwd to path
# per https://stackoverflow.com/a/39574373/7066315
sys.path.append(os.getcwd())
# fmt: on
"""
    B-Opt tuning for HIPNN using AX.

    Originally developed by Sakib Matin (LANL) and modified by Xinyang Li (LANL).
"""

import json

import numpy as np
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from QM7_ax_example import training

import hippynn

warnings.warn(
    "\nMake sure to modify the dataset path in QM7_ax_example.py before running this example.\n"
    "For this test (Ax sequential optimization), a relative path can be used.\n"
    "The work directory for each trial will be ./test_ax/index.\n"
    "If the dataset is in ./dataset, the relative path should be ../../dataset.\n"
    "However, an absolute path is strongly recommended."
)


def evaluate(parameters: dict, trial_index: int):
    """Evaluate a trial for QM7

    Args:
        parameter (dict): Python dictionary for trial values of hipnn-hyper params.
        trial_index (int): Index of the current trial. A new folder will be created
            based on this index to host the files associated with this trial.

    Returns:
        dictionary : Loss metrics to be minimized.
    """

    # train model
    with hippynn.tools.active_directory(str(trial_index)):
        out = training(**parameters)
    # get the loss
    return {"Metric": out["valid"]["Loss"]}


# Initalize the Client and experiment.
restart = False
# load parameters from a json file
if len(sys.argv) == 2:
    with open(sys.argv[1], "r") as param:
        parameters = json.load(param)
# or directly provide them here
else:
    parameters = [
        {
            "name": "dist_soft_min",
            "type": "range",
            "value_type": "float",
            "bounds": [0.5, 1.5],
        },
        {
            "name": "dist_soft_max",
            "type": "range",
            "value_type": "float",
            "bounds": [3.0, 20.0],
        },
        {
            "name": "dist_hard_max",
            "type": "range",
            "value_type": "float",
            "bounds": [5.0, 40.0],
        },
    ]

# create or reload the Ax experiment
if restart:
    ax_client = AxClient.load_from_json_file(filepath="hyperopt.json")
    # update existing experiment
    # `immutable_search_space_and_opt_config` has to be False
    # when the experiment was created
    ax_client.set_search_space(parameters)
else:
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name="QM7_opt",
        parameters=parameters,
        objectives={
            "Metric": ObjectiveProperties(minimize=True),
        },
        overwrite_existing_experiment=True,
        immutable_search_space_and_opt_config=False,
        is_test=False,
        parameter_constraints=[
            "dist_soft_min <= dist_soft_max",
            "dist_soft_max <= dist_hard_max",
        ],
    )

if not os.path.exists("test_ax"):
    os.mkdir("test_ax")
os.chdir("test_ax")
# Run the Optimization Loop.
for k in range(8):
    parameter, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=evaluate(parameter, trial_index)
    )

    # Save experiment to file as JSON file
    ax_client.save_to_json_file(filepath="hyperopt.json")
    # Save a human-friendly summary as csv file
    data_frame = ax_client.get_trials_data_frame().sort_values("Metric")
    data_frame.to_csv("hyperopt.csv", header=True)

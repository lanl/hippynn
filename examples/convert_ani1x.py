"""
Example script for converting from ANI format to numpy format for training with hippynn.


This script was designed for an exteranl dataset available at
https://doi.org/10.6084/m9.figshare.c.4712477
pyanitools reader available at
https://github.com/aiqm/ANI1x_datasets

For info on the dataset, see the following publication:
Smith, J.S., Zubatyuk, R., Nebgen, B. et al.
The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional
theory properties for molecules. Sci Data 7, 134 (2020).
https://doi.org/10.1038/s41597-020-0473-z


"""
import argparse
import os
import numpy as np

from ANI1x_datasets import dataloader as dl  # ANI1x_datsets from https://github.com/aiqm/ANI1x_datasets

parser = argparse.ArgumentParser(prog="convert_ani1x_data")

parser.add_argument(
    "-i",
    "--input_file",
    action="store",
    default="./ani1x-release.h5",
    help="Location of h5 file for ani1x. Defaults to ./ani1x-release.h5",
    type=str,
)
parser.add_argument(
    "-o",
    "--output_directory",
    action="store",
    type=str,
    default=".",
    help="Directory to put the arrays in. Defaults to cwd",
)

args = parser.parse_args()

# Path to the ANI-1x data set
path_to_h5file = os.path.abspath(args.input_file)

######## Constants

# List of keys to point to requested data
# Hardcoded for original ani1x level of theory
data_keys = ["wb97x_dz.energy", "wb97x_dz.forces"]

max_atoms = 63  # hardcoded for ANI1x

# From CCCBDB, for transforming from raw energies to approximately the atomization energy.
self_energy_byspecies = {"C": -37.830234, "H": -0.500608, "N": -54.568004, "O": -75.036223}

number_map = dict(zip([6, 1, 7, 8], "CHNO"))

number_selfenergy = {k: self_energy_byspecies[v] for k, v in number_map.items()}

hartree_in_kcal = 627.5094740631


####### Functions


def pad_atoms_to(arr, total):
    widths = [[0, 0] for _ in arr.shape]
    widths[1][1] = total - arr.shape[1]
    return np.pad(arr, widths, constant_values=0, mode="constant")


def repeat_species(arr, n_conf):
    return arr[np.newaxis].repeat(n_conf, axis=0)


def compute_self_en(conformation_array):
    return sum((arr == k).sum() * v for k, v in number_selfenergy.items())


####### Perform the extraction
sets = {k: [] for k in ["Z", "R", "Forces", "E_total", "T"]}

name_map = {
    "atomic_numbers": "Z",
    "coordinates": "R",
    "wb97x_dz.forces": "Forces",
    "wb97x_dz.energy": "E_total",
}

print("Reading arrays...", end="", flush=True)
for data in dl.iter_data_buckets(path_to_h5file, keys=data_keys):

    n_conformations = len(data["coordinates"])

    for k, arr in data.items():
        if k == "atomic_numbers":
            self_en = compute_self_en(arr)
            arr = repeat_species(arr, n_conformations)

        if k != "wb97x_dz.energy":
            arr = pad_atoms_to(arr, max_atoms)

        sets[name_map[k]].append(arr)

    sets["T"].append(data["wb97x_dz.energy"] - self_en)
print("Done!")

print("Post-processing.")

for k in sets:
    sets[k] = np.concatenate(sets[k], axis=0)

for k in ["Forces", "T"]:
    sets[k] = sets[k] * hartree_in_kcal

sets["Z"] = sets["Z"].astype(int)
sets["T"] = sets["T"].astype("float32")[:, np.newaxis]
sets["Grad"] = -sets["Forces"]
del sets["Forces"]
del sets["E_total"]

print("Saving arrays...", end="", flush=True)

for k, v in sets.items():
    name = f"data-Ani1x_dz{k}.npy"
    name = os.path.join(args.output_directory, name)
    np.save(name, v)

print("Done!")

"""
Example script for converting from ANI-2X dataset to numpy format for training with hippynn.

This script was designed for an external dataset available at
https://zenodo.org/records/10108942

For info on the dataset, see the following publication:
Devereux, C., Smith, J.S., Huddleston, K.K. et al.
Extending the Applicability of the ANI Deep Learning Molecular Potential to Sulfur and Halogens
J. Chem. Theory Comput. 2020, 16, 7, 4192-4202
https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00121


"""
import argparse
import os
import numpy as np
import h5py

parser = argparse.ArgumentParser(prog="convert_ani2x_data")

parser.add_argument(
    "-i",
    "--input_file",
    action="store",
    default="./ANI-2x-B973c-def2mTZVP.h5",
    help="Location of h5 file for ani2x. Defaults to ./ANI-2x-B973c-def2mTZVP.h5",
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
data_keys = ['D3.energy-corrections', 'D3.force-corrections', 'coordinates', 'dipole', 'energies', 'forces', 'species'] 

max_atoms = 63  # hardcoded for ANI2x

hartree_in_kcal = 627.5094740631

####### Functions

def iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
    """ Iterate over buckets of data in ANI HDF5 file. 
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('species')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            # if grp.name == '/007':
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['species'] = grp['species'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d 

def repeat_species(arr, n_conf):
    return arr[np.newaxis].repeat(n_conf, axis=0)

def pad_atoms_to(arr, total):
    widths = [[0, 0] for _ in arr.shape]
    widths[1][1] = total - arr.shape[1]
    return np.pad(arr, widths, constant_values=0, mode="constant")

####### Perform the extraction
sets = {k: [] for k in data_keys}

print("Reading arrays...", end="", flush=True)
for data in iter_data_buckets(path_to_h5file, keys=data_keys):

    # n_conformations = len(data["coordinates"])

    for k, arr in data.items():
        if k not in ["energies", 'D3.energy-corrections', 'dipole']:
            arr = pad_atoms_to(arr, max_atoms)

        sets[k].append(arr)
print("Done!")

print("Post-processing.")

for k in sets:
    sets[k] = np.concatenate(sets[k], axis=0)

for k in ['D3.energy-corrections', 'D3.force-corrections', 'energies', 'forces']:
    sets[k] = sets[k] * hartree_in_kcal

sets["energies"] = sets["energies"].astype("float32")[:, np.newaxis]
sets["Grad"] = -sets["forces"]
del sets["forces"]

print("Saving arrays...", end="", flush=True)

for k, v in sets.items():
    name = f"data-ANI-2x-B973c-def2mTZVP_all_D3_included_{k}.npy"
    name = os.path.join(args.output_directory, name)
    np.save(name, v)

print("Done!")

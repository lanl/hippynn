"""
Example script for training HIP-NN directly from the ANI1x_datasets h5 file.

This script was designed for an external dataset available at
https://doi.org/10.6084/m9.figshare.c.4712477

For info on the dataset, see the following publication:
Smith, J.S., Zubatyuk, R., Nebgen, B. et al.
The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional
theory properties for molecules. Sci Data 7, 134 (2020).
https://doi.org/10.1038/s41597-020-0473-z

"""

import argparse
import torch
import hippynn
import ase.units

# wb97x-6-31g*, G16. Doesn't need to be exact for most models, except atomization consistent.
# # # Old values with singlet/triplet multiplicity only
# # SELF_ENERGY_APPROX = {'C': -37.764142, 'H': -0.4993212, 'N': -54.4628753, 'O': -74.940046}
# Recalculated with appropriate vacuum multiplicity
SELF_ENERGY_APPROX = {"C": -37.8338334397, "H": -0.499321232710, "N": -54.5732824628, "O": -75.0424519384}
SELF_ENERGY_APPROX = {k: SELF_ENERGY_APPROX[v] for k, v in zip([6, 1, 7, 8], "CHNO")}
SELF_ENERGY_APPROX[0] = 0


def load_db(en_name, force_name, seed, anidata_location, use_ccx_subset):
    """
    Load the database.
    """

    from hippynn.databases.h5_pyanitools import PyAniFileDB

    # Ensure total energies loaded in float64.
    torch.set_default_dtype(torch.float64)

    # Load DB, ensuring CCX energy info is available if that subset is selected.
    CCX_EN_NAME = "ccsd(t)_cbs.energy"
    #if use_ccx_subset and en_name != CCX_EN_NAME:
    #    db_info["targets"].append(CCX_EN_NAME) # note, this is in-place and affects the evaluator.
    database = PyAniFileDB(file=anidata_location, allow_unfound=True, species_key="atomic_numbers", seed=seed, num_workers=0,inputs=None,targets=None)
    if use_ccx_subset and en_name != CCX_EN_NAME:
        database.targets.remove(CCX_EN_NAME) # undo in-place addition

    # compute (approximate) atomization energy by subtracting self energies

    # Build a lookup tensor for self energies
    max_z = max(SELF_ENERGY_APPROX.keys()) + 1  # +1 in case max Z is the last index
    lookup_table = torch.zeros(max_z, dtype=torch.float32)
    for z, energy in SELF_ENERGY_APPROX.items():
        lookup_table[z] = energy

    database.arr_dict["atomic_numbers"] = database.arr_dict["atomic_numbers"].long()

    self_energy = lookup_table[database.arr_dict["atomic_numbers"]]
    self_energy = self_energy.sum(dim=1)
    database.arr_dict[en_name] = database.arr_dict[en_name] - self_energy
    kcalpmol = ase.units.kcal / ase.units.mol
    conversion = ase.units.Ha / kcalpmol
    database.arr_dict[en_name] = database.arr_dict[en_name].float() * conversion

    if force_name in database.arr_dict:
        database.arr_dict[force_name] = database.arr_dict[force_name] * conversion
    torch.set_default_dtype(torch.float32)

    # Drop indices where computed energy not retrieved.
    if use_ccx_subset:
        filter_name = CCX_EN_NAME
    else:
        filter_name = en_name
    found_indices = ~torch.isnan(database.arr_dict[filter_name])
    database.arr_dict = {k: v[found_indices] for k, v in database.arr_dict.items()}

    return database

ANI1X_DSETS_KEYS = [
    "hf_tz.energy",
    "coordinates",
    "tpno_ccsd(t)_dz.corr_energy",
    "wb97x_dz.hirshfeld_charges",
    "wb97x_tz.mbis_charges",
    "wb97x_tz.forces",
    "mp2_tz.corr_energy",
    "npno_ccsd(t)_tz.corr_energy",
    "wb97x_tz.mbis_volumes",
    "wb97x_tz.energy",
    "wb97x_tz.dipole",
    "wb97x_tz.mbis_octupoles",
    "wb97x_tz.mbis_quadrupoles",
    "mp2_qz.corr_energy",
    "wb97x_tz.mbis_dipoles",
    "wb97x_dz.cm5_charges",
    "path",
    "atomic_numbers",
    "hf_qz.energy",
    "mp2_dz.corr_energy",
    "wb97x_dz.dipole",
    "npno_ccsd(t)_dz.corr_energy",
    "wb97x_dz.energy",
    "hf_dz.energy",
    "wb97x_dz.quadrupole",
    "ccsd(t)_cbs.energy",
    "wb97x_dz.forces",
]

AVAIL_METHODS = ["hf", "wb97x", "ccsd(t)", "mp2"]
AVAIL_BASIS = ["dz", "tz", "qz", "cbs"]


def get_data_names(qm_method, basis_set):
    assert qm_method in AVAIL_METHODS, f"Method not found: {qm_method}"
    assert basis_set in AVAIL_BASIS, f"Basis set not found: {basis_set}"
    data_spec = f"{qm_method}_{args.basis_set}"
    en_name = f"{data_spec}.energy"
    force_name = f"{data_spec}.forces"
    assert en_name in ANI1X_DSETS_KEYS, f"Method-basis combination not available: {data_spec}"
    #assert f"{data_spec}.forces" in ANI1X_DSETS_KEYS, f"Force training not available for data spec: {data_spec}"
    return en_name, force_name


def make_database_splits(database, ranks, seed, compress):

    import numpy as np
    n_examples = len(database.arr_dict["atomic_numbers"])
    indices = np.arange(n_examples)
    np.random.shuffle(indices)
    split_partitions = np.array_split(indices, ranks)
    print("split partitions",split_partitions)
    from tqdm.auto import tqdm
    all_restarters = []
    for i, split_indices in enumerate(tqdm(split_partitions, unit="splits")):
        name = f"rank_{i}"
        split_indices = torch.as_tensor(split_indices)
        print(split_indices)
        database.make_explicit_split(name, split_indices)

        split_db= hippynn.databases.Database(arr_dict=database.splits[name],
                                allow_unfound=True,
                                seed=seed,inputs=None,targets=None,quiet=True)
        split_db.make_trainvalidtest_split(test_size=0.1, valid_size=0.1)
        print("writing cache...")
        split_db.quiet = False # so it displays the arrays when it loads
        split_db =split_db.make_database_cache(file=name+'.npz', overwrite=True, compress=compress)
        restarter = split_db.restarter
        all_restarters.append(restarter)

    torch.save(all_restarters, "restarters.pt")
    
        
    return 


def main(args):

    en_name, force_name = get_data_names(args.qm_method, args.basis_set)
    
    database = load_db(
        en_name,
        force_name,
        seed=args.seed,
        anidata_location=args.anidata_location,
        use_ccx_subset=args.use_ccx_subset,
    )
    with hippynn.tools.active_directory("./data_ani1x_split"):
        make_database_splits(database, args.ranks, seed=args.seed, compress=args.compress)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ranks", type=int, default=4, help="How many ranks to split the database for")
    parser.add_argument(
        "--use_ccx_subset",
        type=bool,
        default=False,
        help="Train only to configurations from the ANI-1ccx subset."
        " Note that this will still use the energies using the `qm_method` argument."
        " *Note!* This argument will multiply the patience by a factor of 4.",
    )
    from argparse import BooleanOptionalAction
    parser.add_argument("--seed", type=int, default=0, help="random seed for init and split")

    parser.add_argument("--anidata_location", type=str, default="../../../datasets/ani1x_release/ani1x-release.h5")
    parser.add_argument("--qm_method", type=str, default="wb97x")
    parser.add_argument("--basis_set", type=str, default="dz")
    parser.add_argument("--compress", type=argparse.BooleanOptionalAction, default=False)

  
    args = parser.parse_args()

    main(args)

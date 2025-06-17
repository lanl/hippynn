"""

Read Databases in the pyanitools H5 format.

"""

import os
import collections
from pathlib import Path

import h5py  # If dependency not available then just fail here.

import numpy as np
import torch

from ase.data import atomic_numbers

from . import Database
from .restarter import Restartable

from ..tools import progress_bar, np_of_torchdefaultdtype
from ._ani_reader import AniDataLoader, DataPacker

numpy_map_elements = np.vectorize(atomic_numbers.__getitem__)


class PyAniMethods:
    _IGNORE_KEYS = ("path", "Jnames")

    # Note: assumes that all data files have 'coordinates'.
    def extract_full_file(self, file, species_key="species"):
        n_atoms_max = 0
        batches = []
        x = AniDataLoader(file, driver=self.driver)  # Engine=core reads the entire file at once.
        sys_counter = collections.Counter()

        for c in progress_bar(x, desc="Data Groups", unit="group", total=x.group_size()):
            batch_dict = {}
            if species_key not in c:
                raise ValueError(f"Species key '{species_key}' not found' in file {file}!\n" f"\tFound keys: {set(c.keys())}")
            for k, v in c.items():
                # Filter things we don't need
                if k in self._IGNORE_KEYS:
                    continue

                # Convert to numpy
                v = v if np.isscalar(v) else np.asarray(v)

                # Special logic for species
                if k == species_key:
                    # Groups have the same species, broadcast out the batch axis
                    v = np.expand_dims(v, 0)
                    # If given as strings, map to atomic elements
                    if (not isinstance(v.dtype, type)) and issubclass(v.dtype.type, np.str_):
                        v = numpy_map_elements(v)

                    n_atoms_max = max(n_atoms_max, v.shape[1])

                sys_counter[k] += v.shape[0]
                batch_dict[k] = v

            batches.append(batch_dict)

        if len(sys_counter)==0:
            sys_count=0
        else:
            sys_count = max(sys_counter.values())  # some variables are batch-wise, but most of these should be the same
        return batches, n_atoms_max, sys_count

    def determine_key_structure(self, batch_list, sys_count, n_atoms_max, species_key="species"):
        """Determine what arrays to pad"""
        batch = batch_list[0]
        n_atoms = batch[species_key].shape[1]

        # algorithm will fail if n_atoms is confusable with other dimensions,
        # so just recurse to the next batch if this is detected.
        # The number 7 is relatively arbitrary, but if we use, for example, 3 then
        # the algorithm would treat the coordinates as needing padding.
        if n_atoms < 7:
            try:
                return self.determine_key_structure(batch_list[1:], species_key=species_key)
            except RecursionError as re:
                msg = "Automatic detection of arrays is only compatible with datasets of at least 6 atoms -- this is not supported."
                raise ValueError(msg) from re

        # dict of which axes need to be padded;
        # pad if the array size is equal to the number of atoms along a given axis
        padding_scheme = {k: [] for k in batch.keys()}
        shape_scheme = {}
        bsize = 0
        bkey = None
        for k, v in batch.items():
            for i, l in enumerate(v.shape):
                if i == 0:
                    continue  # Don't pad the batch index
                if l == n_atoms:
                    padding_scheme[k].append(i)
                    # Use the largest 0th-axis shape that has an atom index
                    # as the indicator key for the batch size
                    this_bsize = v.shape[0]
                    if this_bsize > bsize:
                        bsize = this_bsize
                        bkey = k
            shape_scheme[k] = list(v.shape)
            for axis in padding_scheme[k]:
                shape_scheme[k][axis] = n_atoms_max
            shape_scheme[k][0] = sys_count

        padding_scheme["sys_number"] = []
        return padding_scheme, shape_scheme, bkey

    def process_batches(self, batches, n_atoms_max, sys_count, species_key="species"):

        # Get padding abd shape info and batch size key
        padding_scheme, shape_scheme, size_key = self.determine_key_structure(batches, sys_count, n_atoms_max, species_key=species_key)

        # add system numbers to the final arrays
        shape_scheme["sys_number"] = [
            sys_count,
        ]
        batches[0]["sys_number"] = np.asarray([0], dtype=np.int64)

        arr_dict = {}
        for k, shape in shape_scheme.items():
            dtype = batches[0][k].dtype
            arr_dict[k] = np.zeros(shape, dtype=dtype)

        sys_start = 0
        for i, b in enumerate(progress_bar(batches, desc="Processing Batches", unit="batch")):
            # Get batch metadata
            n_sys = b[size_key].shape[0]
            b["sys_number"] = np.asarray([i], dtype=np.int64)
            sys_end = sys_start + n_sys
            # n_atoms_batch = b[species_key].shape[1]  # don't need this!

            for k, arr in b.items():

                if k == species_key:
                    arr = np.repeat(arr, n_sys, axis=0)

                # set up slicing for non-batch axes
                where = tuple(slice(0, s) for s in arr.shape[1:])
                # add batch slicing
                where = (slice(sys_start, sys_end), *where)

                # store array!
                arr_dict[k][where] = arr

            sys_start += n_sys

        if sys_start != sys_count:
            # Just in case someone tries to change this code later,
            # Here is a consistency check.
            raise RuntimeError(f"Number of systems was inconsistent: {sys_start} vs. {sys_count}")

        return arr_dict

    def filter_arrays(self, arr_dict, allow_unfound=False, quiet=False):
        if not quiet:
            print("Arrays found: ", list(arr_dict.keys()))

        floatX = np_of_torchdefaultdtype()
        for k, v in arr_dict.copy().items():
            if not allow_unfound:
                if k not in self.inputs and k not in self.targets:
                    del arr_dict[k]
                    continue
            if v.dtype == "float64":
                arr_dict[k] = v.astype(floatX)

        if not quiet:
            print("Data types:")
            print({k: v.dtype for k, v in arr_dict.items()})

        return arr_dict


class PyAniFileDB(Database, PyAniMethods, Restartable):
    def __init__(self, file, inputs, targets, *args, allow_unfound=False, species_key="species", quiet=False, driver="core", **kwargs):
        """

        :param file:
        :param inputs:
        :param targets:
        :param args:
        :param allow_unfound:
        :param species_key:
        :param quiet:
        :param driver: h5 file driver.
        :param kwargs:
        """

        self.file = file
        self.inputs = inputs
        self.targets = targets
        self.species_key = species_key
        self.driver = driver

        arr_dict = self.load_arrays(quiet=quiet, allow_unfound=allow_unfound)

        super().__init__(arr_dict, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound)
        self.restarter = self.make_restarter(
            file,
            inputs,
            targets,
            *args,
            **kwargs,
            driver=driver,
            quiet=quiet,
            allow_unfound=allow_unfound,
            species_key=species_key,
        )

    def load_arrays(self, allow_unfound=False, quiet=False):
        if not quiet:
            print("Loading arrays from", self.file)
        batches, n_atoms_max, sys_count = self.extract_full_file(self.file, species_key=self.species_key)
        arr_dict = self.process_batches(batches, n_atoms_max, sys_count, species_key=self.species_key)
        arr_dict = self.filter_arrays(arr_dict, quiet=quiet, allow_unfound=allow_unfound)
        return arr_dict


class PyAniDirectoryDB(Database, PyAniMethods, Restartable):
    def __init__(
        self,
        directory,
        inputs,
        targets,
        *args,
        files=None,
        allow_unfound=False,
        species_key="species",
        quiet=False,
        driver="core",
        **kwargs,
    ):

        self.directory = directory
        self.files = files
        self.inputs = inputs
        self.targets = targets
        self.species_key = species_key
        self.driver = driver

        arr_dict = self.load_arrays(allow_unfound=allow_unfound, quiet=quiet)

        super().__init__(arr_dict, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound)
        self.restarter = self.make_restarter(directory, inputs, targets, *args, files=files, quiet=quiet, species_key=species_key, **kwargs)

    def load_arrays(self, allow_unfound=False, quiet=False):

        if self.files:
            files = [os.path.join(self.directory, f) for f in self.files]
        else:
            files = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.endswith(".h5")]
        files.sort()

        if len(files) == 0:
            raise FileNotFoundError(f"No '.h5' files found in directory {self.directory}")

        if not quiet:
            print("Gathering data from files:\n\t", end="")
            print(*files, sep="\n\t")

        file_batches = []
        for f in progress_bar(files, desc="Data Files", unit="file"):
            file_batches.append(self.extract_full_file(f, species_key=self.species_key))

        data, max_atoms_list, sys_count = zip(*file_batches)

        n_atoms_max = max(max_atoms_list)
        batches = [item for fb in data for item in fb]
        sys_count = sum(sys_count)

        arr_dict = self.process_batches(batches, n_atoms_max, sys_count, species_key=self.species_key)
        arr_dict = self.filter_arrays(arr_dict, quiet=quiet, allow_unfound=allow_unfound)
        return arr_dict


def write_h5(
    database: Database,
    split: str = None,
    file: Path = None,
    species_key: str = "species",
    overwrite: bool = False,
) -> dict:
    """
    :param database: Database to use
    :param split:  str, None, or True; selects data split to save.
     If None, contents of arr_dict are used.
     If True, save all splits and save split masks as well.
    :param file: where to save the database. if None, does not save the file.
    :param species_key:  the key used for system contents (padding and chemical formulas)
    :param overwrite:  boolean; enables over-writing of h5 file.
    :return: dictionary of pyanitools-format systems.
    """

    if split is True:
        database = database.write_npz("", record_split_masks=True, return_only=True)
    elif split in database.splits:
        database = database.splits[split]
        database = {k: v.to("cpu").numpy() for k, v in database.items()}
    elif split is None:
        database = database.arr_dict
    else:
        raise Exception(f"Unknown split variable supplied (must be True, None, or str): {split:s}")

    if file is not None:
        if Path(file).exists():
            if overwrite:
                print("Overwriting h5 file:", file)
                Path(file).unlink()
            else:
                raise FileExistsError(f"h5path {file:s} exists.")
        print("Saving h5 file:", file)
        packer = DataPacker(file)
    else:
        packer = None

    db_species = database[species_key]
    total_systems = db_species.shape[0]
    n_atoms_max = db_species.shape[1]

    # determine which keys have second shape of N atoms
    is_atom_var = {k: (len(k_arr.shape) > 1) and (k_arr.shape[1] == n_atoms_max) for k, k_arr in database.items()}
    del is_atom_var[species_key]  # species handled separately

    # Create the data dictionary
    # Maps hashes of system chemical formulas to dictionaries of system information.
    data = {}
    for i, db_mol_species in enumerate(db_species):
        # We can append the system data to an existing set of system data
        mol_n_atom = np.count_nonzero(db_mol_species)
        if np.count_nonzero(db_mol_species[mol_n_atom:]) > 0:
            raise ValueError(f"Malformed species row with non-standard padding: {db_mol_species}")
        db_mol_species = db_mol_species[:mol_n_atom]
        mhash = hash(np.array(db_mol_species).tobytes())  # molecule hash

        if mhash not in data:
            # need to make a new mol entry in the data for this chemical formula
            # the mol dictionary maps a data key to the array for that mol.
            # The species key has one value, but the other keys can store batch of values.
            mol = {species_key: db_mol_species}
            for k, k_is_atom_based in is_atom_var.items():
                db_arr = database[k]
                store_arr = db_arr[i, :mol_n_atom] if k_is_atom_based else db_arr[i]
                mol[k] = [store_arr]
            data[mhash] = mol
        else:
            # If there is already an entry for this chemical formula, append it to the current one
            mol = data[mhash]
            mol_species = mol[species_key]
            # First sanity check that the mhash hash we are using is unique, or else BAD.
            if (db_mol_species.shape != mol_species.shape) or not (db_mol_species == mol_species).all():
                raise ValueError("Error. Hash not unique. You should never see this.")

            # Now append the system to the set of systems with this chemical formula.
            for k, k_is_atom_based in is_atom_var.items():
                db_arr = database[k]
                store_arr = db_arr[i, :mol_n_atom] if k_is_atom_based else db_arr[i]
                mol[k].append(store_arr)

    # post-process atom variables into arrays and handle strings.
    for mhash, mol in data.items():
        for k in is_atom_var.keys():
            mol[k] = np.asarray(mol[k])

            if np.issubdtype(mol[k].dtype, np.unicode_):
                mol[k] = [el.encode("utf-8") for el in list(mol[k])]
                mol[k] = np.array(mol[k])
    # Store data
    if packer is not None:
        for key in data:
            packer.store_data(str(key), **data[key])
        packer.cleanup()

    return data

"""
Read Databases in the ANI H5 format.

Note: You will need `pyanitools.py` to be importable to import this module.
"""

import pyanitools

import os

import numpy as np
import torch

from ..tools import progress_bar, np_of_torchdefaultdtype
from ..tools import pad_np_array_to_length_with_zeros as pad_atoms

from . import Database
from .restarter import Restartable

from ase.data import atomic_numbers

numpy_map_elements = np.vectorize(atomic_numbers.__getitem__)


class PyAniMethods:
    _IGNORE_KEYS = ("path", "Jnames")

    # Note: assumes that all data files have 'coordinates'.
    def extract_full_file(self, file, species_key="species"):
        n_atoms_max = 0
        batches = []
        x = pyanitools.anidataloader(file)

        for c in progress_bar(x, desc="Data Groups", unit="group", total=x.group_size()):
            batch_dict = {}
            if species_key not in c:
                raise ValueError(f"Species key '{species_key}' not found' in file {file}!\n"
                                 f"\tFound keys: {set(c.keys())}")
            for k, v in c.items():
                # Filter things we don't need
                if k in self._IGNORE_KEYS:
                    continue

                # Convert to numpy
                v = v if np.isscalar(v) else np.asarray(v)

                # Special logic for species
                if k == species_key:
                    # Groups have the same species, broad-cast out the batch axis
                    v = np.expand_dims(v, 0)
                    # If given as strings, map to atomic elements
                    if (not isinstance(v.dtype, type)) and issubclass(v.dtype.type, np.str_):
                        v = numpy_map_elements(v)

                    n_atoms_max = max(n_atoms_max, v.shape[1])

                batch_dict[k] = v

            batches.append(batch_dict)

        return batches, n_atoms_max

    def determine_key_structure(self, batch_list, species_key="species"):
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

        padding_scheme['sys_number'] = []
        return padding_scheme, bkey

    def process_batches(self, batches, n_atoms_max, species_key="species"):

        # Get padding info and batch size key
        padding_scheme, size_key = self.determine_key_structure(batches, species_key=species_key)

        # Pad the arrays
        padded_batches = []
        for i, b in enumerate(progress_bar(batches, desc="Processing Batches", unit="batch")):
            pb = {}
            b['sys_number'] = np.asarray([i], dtype=np.int64)
            for k, v in b.items():
                bsize = len(b[size_key])
                # Expand species array to fit batch size
                if k == species_key:

                    v = np.repeat(v, bsize, axis=0)

                # Perform padding as needed
                for axis in padding_scheme[k]:
                    v = pad_atoms(v, n_atoms_max, axis=axis)
                if 0 not in padding_scheme[k] and v.shape[0] == 1:
                    v = np.broadcast_to(v, (bsize, *v.shape[1:]))

                pb[k] = v

            padded_batches.append(pb)

        arr_dict = {}

        for k in b.keys():
            try:
                arr_dict[k] = np.concatenate([pb[k] for pb in padded_batches])
            except ValueError as ve:
                print("Error occured:",ve)
                print("Skipping key:",k)
                continue

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
    def __init__(self, file, inputs, targets, *args, allow_unfound=False,species_key="species", quiet=False, **kwargs):

        self.file = file
        self.inputs = inputs
        self.targets = targets
        self.species_key = species_key

        arr_dict = self.load_arrays(quiet=quiet, allow_unfound=allow_unfound)

        super().__init__(arr_dict, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound)
        self.restarter = self.make_restarter(
            file, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound,
            species_key=species_key,
        )

    def load_arrays(self, allow_unfound=False, quiet=False):
        if not quiet:
            print("Loading arrays from", self.file)
        batches, n_atoms_max = self.extract_full_file(self.file,species_key=self.species_key)
        arr_dict = self.process_batches(batches, n_atoms_max,species_key=self.species_key)
        arr_dict = self.filter_arrays(arr_dict, quiet=quiet, allow_unfound=allow_unfound)
        return arr_dict


class PyAniDirectoryDB(Database, PyAniMethods, Restartable):
    def __init__(self, directory, inputs, targets, *args, files=None, allow_unfound=False,species_key="species",
                 quiet=False,**kwargs):

        self.directory = directory
        self.files = files
        self.inputs = inputs
        self.targets = targets
        self.species_key = species_key
        arr_dict = self.load_arrays(allow_unfound=allow_unfound,quiet=quiet)

        super().__init__(arr_dict, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound)
        self.restarter = self.make_restarter(directory, inputs, targets, *args, files=files, quiet=quiet,
                                             species_key=species_key, **kwargs)

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
            file_batches.append(self.extract_full_file(f,species_key=self.species_key))

        data, max_atoms_list = zip(*file_batches)

        n_atoms_max = max(max_atoms_list)
        batches = [item for fb in data for item in fb]

        arr_dict = self.process_batches(batches, n_atoms_max,species_key=self.species_key)
        arr_dict = self.filter_arrays(arr_dict, quiet=quiet, allow_unfound=allow_unfound)
        return arr_dict

"""
Dataset stored as NPY files in directory or
as NPZ dictionary.
"""
import os

import numpy as np
import torch

from ..tools import np_of_torchdefaultdtype
from .database import Database
from .restarter import Restartable


class DirectoryDatabase(Database, Restartable):
    """
    Database stored as NPY files in a directory.

    :param directory: directory path where the files are stored
    :param name: prefix for the arrays.

    This function loads arrays of the format f"{name}{db_name}.npy" for each variable db_name in inputs and targets.

    Other arguments: See ``Database``.

    .. Note::
       This database loader does not support the ``allow_unfound`` setting in the base ``Database``. The
       variables to load must be set explicitly in the inputs and targets.
    """

    def __init__(self, directory, name, inputs, targets, *args, quiet=False, allow_unfound=False, **kwargs):
        #if allow_unfound:
        #    raise ValueError("DirectoryDatabase class does not support allow_unfound argument.")

        arr_dict = self.load_arrays(directory, name, inputs, targets, quiet=quiet)
        super().__init__(arr_dict, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound)

        self.restarter = self.make_restarter(
            directory,
            name,
            inputs,
            targets,
            *args,
            **kwargs,
            quiet=quiet,
            allow_unfound=allow_unfound,
        )

    def get_file_dict(self, directory, prefix):
        try:
            file_list = os.listdir(directory)
        except FileNotFoundError as fee:
            raise FileNotFoundError(
                "ERROR: Couldn't find directory {} containing files."
                'A solution is to explicitly specify "path" in database_params '.format(directory)
            ) from fee

        data_labels = {
            file[len(prefix) : -4]: file for file in file_list if file.startswith(prefix) and file.endswith(".npy")
        }

        # Make sure we actually found some files
        if not data_labels:
            raise FileNotFoundError(
                "No files found at {} .".format(directory) + "for database prefix {}".format(prefix)
            )
        return data_labels

    def load_arrays(self, directory, name, inputs, targets, quiet=False, allow_unfound=False):

        var_list = inputs + targets
        # Make sure the path actually exists

        try:
            # Backward compatibility.
            data_labels = self.get_file_dict(directory, prefix="data-" + name)
        except FileNotFoundError:
            data_labels = self.get_file_dict(directory, prefix=name)

        if not quiet:
            print("Arrays found: ", data_labels)

        # Load files
        arr_dict = {
            label: np.load(os.path.join(directory, file))
            for label, file in data_labels.items()
            if allow_unfound or (label in var_list)
        }

        # Put float64 data in pytorch default dtype
        floatX = np_of_torchdefaultdtype()
        for k, v in arr_dict.items():
            if v.dtype == "float64":
                arr_dict[k] = v.astype(floatX)

        if not quiet:
            print("Data types:")
            print({k: v.dtype for k, v in arr_dict.items()})

        return arr_dict


class NPZDatabase(Database, Restartable):
    def __init__(self, file, inputs, targets, *args, allow_unfound=False, quiet=False, **kwargs):
        arr_dict = self.load_arrays(file, inputs, targets, quiet=quiet, allow_unfound=allow_unfound)
        super().__init__(arr_dict, inputs, targets, *args, **kwargs, quiet=quiet,allow_unfound=allow_unfound)
        self.restarter = self.make_restarter(
            file, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound
        )

    def load_arrays(self, file, inputs, targets, allow_unfound=False, quiet=False):

        arr_dict = np.load(file)
        # Make sure the path actually exists
        if not quiet:
            print("Arrays found: ", list(arr_dict.keys()))

        # Load files
        if not allow_unfound:
            var_list = inputs + targets
            arr_dict = {k: v for k, v in arr_dict.items() if k in var_list}
        else:
            arr_dict = {k: v for k,v, in arr_dict.items()}

        # Put float64 data in pytorch default dtype
        floatX = np_of_torchdefaultdtype()
        for k, v in arr_dict.items():
            if v.dtype == "float64":
                arr_dict[k] = v.astype(floatX)

        if not quiet:
            print("Data types:")
            print({k: v.dtype for k, v in arr_dict.items()})

        return arr_dict

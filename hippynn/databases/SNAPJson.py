"""
Load database from SNAP format.
"""
import json, glob, os
import torch
import numpy as np

from .database import Database
from .restarter import Restartable
from ..tools import pad_np_array_to_length_with_zeros as padax
from ..tools import progress_bar, np_of_torchdefaultdtype

from ase.data import chemical_symbols


class SNAPDirectoryDatabase(Database, Restartable):
    def __init__(
        self,
        directory,
        inputs,
        targets,
        *args,
        files=None,
        depth=1,
        transpose_cell=True,
        allow_unfound=False,
        quiet=False,
        n_comments=1,
        **kwargs,
    ):

        self.directory = directory
        self.files = files
        self.inputs = inputs
        self.targets = targets
        self.transpose_cell = transpose_cell
        self.depth = depth
        self.n_comments = n_comments
        arr_dict = self.load_arrays(quiet=quiet, allow_unfound=allow_unfound)

        super().__init__(arr_dict, inputs, targets, *args, **kwargs, allow_unfound=allow_unfound, quiet=quiet)

        self.restarter = self.make_restarter(
            directory,
            inputs,
            targets,
            *args,
            transpose_cell=transpose_cell,
            files=files,
            allow_unfound=allow_unfound,
            n_comments=n_comments,
            **kwargs,
            quiet=quiet,
        )

    def load_arrays(self, allow_unfound=False, quiet=False):

        if not self.files:
            if not quiet:
                print("Acquiring file list")
            glob_str = f"{self.directory}" + ("/*" * self.depth) + "/*.json"
            files = glob.glob(glob_str)
            if len(files) == 0:
                raise FileNotFoundError(f"No '.json' files found in directory {self.directory}")
        else:
            files = [os.path.join(self.directory, f) for f in self.files]

        files.sort()

        config_unprocessed = []
        for f in progress_bar(files, desc="Data Files", unit="file"):
            config_unprocessed.append(self.extract_snap_file(f))

        config_unprocessed = [c for batch in config_unprocessed for c in batch]  # Flattening groups

        n_atoms_max = max(d["NumAtoms"] for d in config_unprocessed)

        arr_dict = self.process_configs(config_unprocessed, n_atoms_max)
        arr_dict = self.filter_arrays(arr_dict, allow_unfound=allow_unfound, quiet=quiet)
        return arr_dict

    def filter_arrays(self, arr_dict, allow_unfound=False, quiet=False):
        if not quiet:
            print("Arrays found: ", list(arr_dict.keys()))

        floatX = np_of_torchdefaultdtype()
        for k, v in arr_dict.copy().items():
            if not allow_unfound:
                if k not in self.inputs and k not in self.targets:
                    del arr_dict[k]
            if v.dtype == "float64":
                arr_dict[k] = v.astype(floatX)

        if not quiet:
            print("Data types:")
            print({k: v.dtype for k, v in arr_dict.items()})

        return arr_dict

    def extract_snap_file(self, file):
        with open(file, "rt") as jf:
            for i in range(self.n_comments):
                comment = jf.readline()
            content = jf.read()
        parsed = json.loads(content)
        dataset = parsed["Dataset"]
        data_items = dataset["Data"]

        for i, d in enumerate(data_items):
            group_path, config_name = os.path.split(file)
            base_path, group_name = os.path.split(file)
            data_items[i]["FileName"] = config_name
            data_items[i]["Group"] = group_name
            data_items[i]["SubConfig"] = i

        return data_items

    def process_configs(self, configs, n_atoms_max):

        arr_dict = {}
        all_keys = "AtomTypes", "Energy", "Forces", "Lattice", "Positions", "Group", "FileName", "SubConfig"
        pad_keys = "AtomTypes", "Forces", "Positions"
        for key in all_keys:
            value_list = [c[key] for c in configs]
            if key in pad_keys:
                value_list = [padax(np.asarray(v), n_atoms_max) for v in value_list]
            arr_dict[key] = np.stack(value_list)
        arr_dict["AtomTypes"][arr_dict["AtomTypes"] == "0"] = "X"  # ASE calls blank atoms 'X'
        z_array = [[chemical_symbols.index(s) for s in sym] for sym in arr_dict["AtomTypes"]]
        arr_dict["Species"] = np.asarray(z_array).astype(int)
        arr_dict["AtomCount"] = arr_dict["Species"].astype(bool).sum(axis=1)
        arr_dict["EnergyPerAtom"] = arr_dict["Energy"] / arr_dict["AtomCount"]

        if self.transpose_cell:
            arr_dict["Lattice"] = arr_dict["Lattice"].transpose((0, 2, 1))

        return arr_dict

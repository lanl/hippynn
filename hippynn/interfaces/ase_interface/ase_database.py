"""
Dataset stored as ase database file (ideally either .json or .db, or in .xyz, .extxyz formats)

See: https://databases.fysik.dtu.dk/ase/ase/db/db.html for documentation
on typical columns present in ase database

Typically used column names in ase db/xyz format:

positions : x,y,z cartesian positions (Angstrom)
forces : x,y,z carteisian forces (ev/Angstrom)
energy : energy (eV)
energy_per_atom : energy per atom (eV/atom)
cell : x,y,z of cell (3x3) (Angstrom)
charges : atom-specific charges
stress: (6,) atomic stresses
initial_charges : atom-specific initial charges 
inital_magmoms :atom-specific initial magnetic moments
numbers : atom Zs (integer)
pbc : periodic boundary conditions (bool)
ctime : computer time (float)
mtime : time (float)
dipole : molecular dipole (3) vector
"""
import os

import numpy as np
from ase.io import read, iread

from ...tools import np_of_torchdefaultdtype, progress_bar
from ...databases.database import Database
from ...databases.restarter import Restartable
from typing import Union
from typing import List
import hippynn.tools

class AseDatabase(Database, Restartable):
    """
    Database stored as ase database file(s) in a directory.

    :param directory: directory path where the ase database(s) is stored
    :param name: name or list of names for files for ase databases.

    This function loads an ase database(s) ({name}.json/.db) OR ({{name}.extxyz,.xyz})
    variable db_name including all inputs and targets.

    filenames should end with .json, .db, .extxyz, and .xyz, etc; Anything parsable by ase.io.load


    Other arguments: See ``Database``.

    See: https://databases.fysik.dtu.dk/ase/ase/db/db.html for documentation
        on typical columns present in ase database
    """

    def __init__(self, directory: str, name: Union[str, List[str]], inputs, targets, *args, quiet=False, allow_unfound=False, **kwargs):

        arr_dict = self.load_arrays(directory, name, inputs, targets, quiet=quiet, allow_unfound=allow_unfound)
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

    def load_arrays(self, directory, filename, inputs, targets, quiet=False, allow_unfound=False):
        """
        load arrays load ase database into hippynn database arrays

        :param directory: directory where database is stored
        :param filename: file or path to file from directory
        :param inputs:
        :param targets:
        :param quiet:
        :param allow_unfound:
        :return:
        """

        var_list = inputs + targets
        try:
            if isinstance(filename, str):
                db = list(progress_bar(iread(directory+filename,index=":"), desc='configs'))#read(directory + filename, index=":")
            elif isinstance(filename, (list, np.ndarray)):
                db = []
                for name in progress_bar(filename, desc='files'):
                    temp_db = list(progress_bar(iread(directory + name, index=":"), desc='configs'))
                    db += temp_db
        except FileNotFoundError as fee:
            raise FileNotFoundError(
                "ERROR: Couldn't find {} ase xyz database."
                'A solution is to explicitly specify "path" in database_params '.format(directory + filename)
            ) from fee
        if not quiet:
            print("ASE Database found")
        record_list = []
        max_n_atom = 0
        max_atoms_record = None
        for row in db:
            result_dict = row.__dict__
            for k, v in result_dict["arrays"].items():
                result_dict[k] = v
            for k, v in result_dict["info"].items():
                result_dict[k] = v
            del result_dict["arrays"]
            del result_dict["info"]
            result_dict["cell"] = result_dict["_cellobj"][:]
            del result_dict["_cellobj"]
            if result_dict.get("_calc", None) is not None:
                calc_dict = result_dict.get("_calc").__dict__
                # Overwrite atoms-stored objects with calculator objects.
                # Helpful for .json/.db files where energy is not stored under 'info' section
                for k, v in calc_dict["results"].items():
                    result_dict[k] = v
                del result_dict["_calc"]
            if not allow_unfound:
                if "energy_per_atom" in var_list:
                    var_list += ["energy"]
                result_dict = {k: v for k, v in result_dict.items() if (k in var_list)}
            record_list.append(result_dict)
            if len(result_dict["positions"]) > max_n_atom:
                max_n_atom = len(result_dict["positions"])
                max_atoms_record = result_dict
        n_record = len(record_list)
        array_dict = dict()
        delete_cols = []
        for key, val in max_atoms_record.items():
            if isinstance(val, np.ndarray):
                array_dict[key] = np.zeros([n_record] + [x for x in val.shape])
            else:
                if isinstance(val, (float, int)):
                    array_dict[key] = np.zeros([n_record])
                    if (key == "energy") and (("energy_per_atom" in var_list) or (allow_unfound)):
                        array_dict["energy_per_atom"] = np.zeros([n_record])
                else:
                    array_dict[key] = None  # Do Not Save
                    delete_cols.append(key)
        # Save the record uid from the ase db object names.
        record_list.sort(key=lambda rec: len(rec["numbers"]))
        for i, record in enumerate(record_list):
            natom = len(record["numbers"])
            for k, v in record.items():
                if isinstance(v, np.ndarray):
                    if array_dict.get(k,None) is not None:
                        shape = array_dict[k].shape
                    else:
                        shape=[0]
                    # Note this assumes the maximum number of atoms greater than the length of property of interest
                    # E.g. 3 for dipole (make sure your training set has something with more than 3 atoms)
                    # Or 6 for stress tensor (make sure your training set has something with more than 6 atoms)
                    if (len(shape) == 2) and (shape[1] < max_n_atom):  # 1D array, e.g. dipole, stress tensor
                        array_dict[k][i, :] = v
                    elif (len(shape) == 2) and (shape[1] == max_n_atom):  # 1D array including maximum number of atoms: e.g. charges
                        array_dict[k][i, :natom] = v
                    elif (len(shape) == 3) and (shape[1] < max_n_atom):  # 2D array, e.g. cells
                        array_dict[k][i, :, :] = v
                    elif (len(shape) == 3) and (shape[1] == max_n_atom):  # 2D array, e.g. positions, forces
                        array_dict[k][i, :natom, :] = v
                    elif (len(shape) == 1):
                        print('Skipping {}'.format(k))
                    else:
                        raise ValueError("Shape of Numpy array for key: {} unknown.".format(k))
                elif isinstance(array_dict.get(k,None), np.ndarray):  # Energy, float or integers only
                    array_dict[k][i] = v
                    if (k == "energy") and (("energy_per_atom" in var_list) or (allow_unfound)):  # Add in per-atom-energy
                        array_dict["energy_per_atom"][i] = v / natom
                else:  # Everything else either list of strings or something else.
                    pass

        # Make sure Zs are integers.
        array_dict["numbers"] = array_dict["numbers"].astype("int")

        arr_dict = dict()

        # Put float64 data in pytorch default dtype
        floatX = np_of_torchdefaultdtype()
        for k, v in array_dict.items():
            if k not in delete_cols:
                if v.dtype == "float64":
                    arr_dict[k] = v.astype(floatX)
                else:
                    arr_dict[k] = v

        if not quiet:
            print("Data types:")
            print({k: v.dtype for k, v in arr_dict.items()})

        return arr_dict

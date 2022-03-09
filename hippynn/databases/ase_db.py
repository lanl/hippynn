"""
Dataset stored as ase database file (ideally either .json or .db)

Reference: https://wiki.fysik.dtu.dk/ase/ase/db/db.html 
"""
import os

import numpy as np
import torch
from ase.db import connect

from ..tools import np_of_torchdefaultdtype
from .database import Database
from .restarter import Restartable


class AseDatabase(Database, Restartable):
    """
    Database stored as ase database file in a directory.

    :param directory: directory path where the ase database is stored
    :param name: name of the ase database to load

    This function loads arrays of the format f"{name}{db_name}.npy" for each variable db_name in inputs and targets.

    Other arguments: See ``Database``.

    .. Note::
       This database loader does not support the ``allow_unfound`` setting in the base ``Database``. The
       variables to load must be set explicitly in the inputs and targets.
    """

    def __init__(self, directory, name, inputs, targets, *args, quiet=False, allow_unfound=False, **kwargs):
        if allow_unfound:
            raise ValueError("AseDatabase class does not support allow_unfound argument.")

        arr_dict = self.load_arrays(directory, name, inputs, targets, quiet=quiet)
        super().__init__(arr_dict, inputs, targets, *args, **kwargs, quiet=quiet)

        self.restarter = self.make_restarter(
            directory,
            name,
            inputs,
            targets,
            *args,
            **kwargs,
            quiet=quiet,
        )

    def load_arrays(self, directory, filename, inputs, targets, quiet=False):
        """load arrays load ase database into hippynn database arrays

        Parameters
        ----------
        filename : str
            filename or path of database to convert
        prefix : str, optional
            prefix for output numpy arrays, by default None
        return_data : bool, optional
            whether or not to return the data or write to files, by default False
        """
        var_list = inputs + targets
        try:
            db = connect(directory + filename)
        except FileNotFoundError as fee:
                    raise FileNotFoundError(
                        "ERROR: Couldn't find {} ase database."
                        'A solution is to explicitly specify "path" in database_params '.format(directory+filename)
                    ) from fee
        if not quiet:
            print("ASE Database found")
        record_list = []
        max_n_atom = 0
        any_pbc = False
        for row in db.select():
            is_pbc = False
            if np.any(row.pbc):
                is_pbc = True
                any_pbc = True
            result_dict = {
                'atoms':row.numbers,
                'xyz': row.positions,
                'cell': row.cell,
                'is_pbc':is_pbc,
                'force': row.forces,
                'energy': row.energy,
                'uid': row.unique_id
            }
            record_list.append(result_dict)
            if row.natoms > max_n_atom:
                max_n_atom = row.natoms
        n_record = len(record_list)
        # Sort the list base on number of atoms
        record_list.sort(key=lambda rec: len(rec['atoms']))
        # Save the record uid from the ase db object names.
        xyz_array = np.zeros([n_record, max_n_atom, 3])
        force_array = np.zeros([n_record, max_n_atom, 3])
        atom_z_array = np.zeros([n_record, max_n_atom])
        if any_pbc:
            cell_array = np.zeros([n_record, 3, 3])
            pbc = True
        else:
            cell_array = np.zeros([n_record, 3, 3])
            pbc = False
        energy_array = np.array([record['energy'] for record in record_list])
        for i, record in enumerate(record_list):
            natom = len(record['atoms'])
            xyz_array[i, :natom, :] = record['xyz']
            force_array[i, :natom, :] = record['force']
            atom_z_array[i,:natom] = record['atoms']
            if pbc:
                cell_array[i, :, :] = record['cell']
        arr_dict = dict()
        for label in var_list:
            if label == 'energy':
                arr_dict.update({label:energy_array})
            elif label == 'R':
                arr_dict.update({label:xyz_array})
            elif label == 'force':
                arr_dict.update({label:force_array})
            elif label == 'Z':
                arr_dict.update({label:atom_z_array.astype('int')})
            elif label == 'cell':
                arr_dict.update({label:cell_array})

        # Put float64 data in pytorch default dtype
        floatX = np_of_torchdefaultdtype()
        for k, v in arr_dict.items():
            if v.dtype == "float64":
                arr_dict[k] = v.astype(floatX)

        if not quiet:
            print("Data types:")
            print({k: v.dtype for k, v in arr_dict.items()})
        
        return arr_dict

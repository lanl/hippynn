"""
Generic, reusable database helpers that are not tied to a specific
:class:`~hippynn.databases.database.Database` subclass.

Includes tools for loading and exporting databases in EXTXYZ format, and for
auto-detecting standard database key names (e.g. species, coordinates, energy,
forces, cell) from a dictionary of arrays.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write

from .ondisk import NPZDatabase
from .h5_pyanitools import PyAniFileDB


# Key name sets for auto-detection
SPECIES_KEYSET = ['species', 'atomic_numbers', 'z', 'atom_types', 'atomic_number']
COORDINATES_KEYSET = ['coordinates', 'positions', 'pos', 'coords', 'r']
ENERGIES_KEYSET = ['energy', 'energies', 'e', 'total_energy']
FORCES_KEYSET = ['forces', 'force', 'f']
CELL_KEYSET = ['cell', 'lattice', 'box', 'unit_cell', 'c']


def auto_detect_key(keys, keyset, key_name, required=True):
    """
    Auto-detect a database key from a set of possible names using case-insensitive matching.

    This function searches for keys in a case-insensitive manner and returns the first unique match.
    If multiple matches are found, a ValueError is raised to avoid ambiguity. If no matches are found
    and the key is required, a ValueError is raised with available keys listed.

    **Common Keysets:**

    - **SPECIES_KEYSET**: 'species', 'atomic_numbers', 'z', 'atom_types', 'atomic_number'
    - **COORDINATES_KEYSET**: 'coordinates', 'positions', 'pos', 'coords', 'r'
    - **ENERGIES_KEYSET**: 'energy', 'energies', 'e', 'total_energy'
    - **FORCES_KEYSET**: 'forces', 'force', 'f'
    - **CELL_KEYSET**: 'cell', 'lattice', 'box', 'unit_cell'

    :param keys: available keys in the array dictionary
    :param keyset: list of possible key name patterns to match
    :param key_name: descriptive name for error messages (e.g., 'species_key')
    :param required: whether this key is required (if False, returns None with warning if not found)
    :return: detected key name or None
    :raises ValueError: if ambiguous (multiple matches) or missing required key

    Examples
    --------
    >>> from hippynn.databases.utils import auto_detect_key, SPECIES_KEYSET
    >>> keys = ['Species', 'coordinates', 'energy']
    >>> auto_detect_key(keys, SPECIES_KEYSET, 'species_key')
    'Species'

    >>> keys_ambiguous = ['species', 'atomic_numbers', 'coordinates']
    >>> auto_detect_key(keys_ambiguous, SPECIES_KEYSET, 'species_key')  # doctest: +SKIP
    ValueError: Multiple candidates found
    """
    # Normalize keys for case-insensitive matching
    normalized_keyset = [alias.casefold() for alias in keyset]

    # Find matches
    matches = [k for k in keys if k.casefold() in normalized_keyset]

    if len(matches) == 0:
        if required:
            raise ValueError(
                f"Could not auto-detect {key_name}. No matches found for aliases: {keyset}.\n"
                f"Available keys: {list(keys)}\n"
                f"Please specify {key_name} explicitly."
            )
        else:
            warnings.warn(f"Optional key {key_name} not found in arr_dict. Proceeding without it.")
            return None
    elif len(matches) == 1:
        return matches[0]
    else:
        raise ValueError(
            f"Could not auto-detect {key_name}. Multiple candidates found: {matches}.\n"
            f"Please specify {key_name} explicitly to resolve ambiguity."
        )


def load_base_database(
    data_file: Union[str, os.PathLike],
    seed: int = 101,
    num_workers: int = 2,
):
    """
    Load either an NPZDatabase (.npz) or PyAniFileDB (.h5/.hdf5) with a consistent interface.
    Returns (database, energies_key), where energies_key is 'energy' (npz) or 'energies' (h5).
    """
    data_file = os.path.expanduser(str(data_file))
    ext = Path(data_file).suffix.lower()

    # Define base_database by the file extension
    if ext == ".npz":
        inputs  = ['coordinates', 'species']
        targets = ['energy', 'forces']
        energies_key = 'energy'
        db = NPZDatabase(
            file=data_file,
            seed=seed,
            allow_unfound=True,
            inputs=inputs,
            targets=targets,
            quiet=False,
        )

    elif ext in (".h5", ".hdf5"):
        inputs  = ['coordinates', 'species']
        targets = ['energies', 'forces']
        energies_key = 'energies'
        db = PyAniFileDB(
            file=data_file,
            species_key="species",
            seed=seed,
            num_workers=num_workers,
            allow_unfound=True,
            inputs=inputs,
            targets=targets,
        )

    else:
        raise ValueError(f"Unrecognized dataset file extension: {ext}. Supported file extensions are: .h5, .hdf5, .npz.")

    return db, energies_key


def write_extxyz(
    database,
    filename: Union[str, os.PathLike],
    overwrite: bool = False,
    pbc: Union[bool, Tuple[bool, bool, bool]] = False,
    split: Optional[str] = None,
):
    """
    Write a hippynn Database to an EXTXYZ file using ASE.

    .. seealso::
       :func:`hippynn.molecular_dynamics.writers.write_extxyz` for exporting MD trajectories
       instead of a :class:`~hippynn.databases.database.Database`.

    Expected keys in database.arr_dict:
      coordinates: (n, max_atoms, 3)
      species:     (n, max_atoms) int, padded with <= 0
      forces:      (n, max_atoms, 3)
      atomenergies:(n, max_atoms, 1) or (n, max_atoms)
      energy or energies: (n,)
      cell:        (n, 3, 3)
      stress:      (n, 3, 3) or (n, 9)

    pbc can be:
      - False (default, non-periodic)
      - True (periodic in all directions)
      - tuple(bool, bool, bool) for per-axis periodicity
    """
    out_path = Path(str(filename))
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Path exists: {out_path}")
        out_path.unlink()
    print(f"Saving EXTXYZ file: {out_path}")

    # Select arrays to write based on split option
    if split is True:
        # Write the entire dataset as it would be written to NPZ, but in-memory
        arr = database.write_npz("", record_split_masks=True, return_only=True)
    elif isinstance(split, str) and hasattr(database, "splits") and split in database.splits:
        # Specific split subset
        subset = database.splits[split]
        arr = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in subset.items()}
    elif split is None:
        arr = database.arr_dict
    else:
        raise ValueError("split must be True, None, or a valid split name (str).")

    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    A = {k: to_np(v) for k, v in arr.items()}

    # Resolve energy key if available
    energy_key = "energy" if "energy" in A else ("energies" if "energies" in A else None)

    # Basic presence checks
    required = ["coordinates", "species"]
    for rk in required:
        if rk not in A:
            raise KeyError(f"Required key '{rk}' not found in database arrays.")

    n_frames = A["species"].shape[0]

    # Normalize pbc
    if isinstance(pbc, bool):
        pbc_tuple = (pbc, pbc, pbc)
    else:
        if not (isinstance(pbc, (tuple, list)) and len(pbc) == 3):
            raise ValueError("pbc must be a bool or a tuple/list of 3 bools.")
        pbc_tuple = tuple(bool(b) for b in pbc)

    for i in range(n_frames):
        sp = A["species"][i]                      # (max_atoms,)
        mask = sp > 0                             # valid atoms
        if not np.any(mask):
            continue
        Z = sp[mask].astype(int)
        R = A["coordinates"][i][mask].astype(float)  # (nat, 3)

        atoms = Atoms(positions=R, numbers=Z)

        # cell and periodic flags
        if "cell" in A:
            atoms.set_cell(A["cell"][i], scale_atoms=False)
            atoms.set_pbc(pbc_tuple)
        else:
            atoms.set_pbc(pbc_tuple)

        # per-atom arrays
        if "forces" in A:
            atoms.new_array("forces", A["forces"][i][mask])
        if "atomenergies" in A:
            ae = A["atomenergies"][i]
            if ae.ndim == 3 and ae.shape[-1] == 1:
                ae = ae[..., 0]
            atoms.new_array("atomenergies", ae[mask])

        # frame scalars
        if energy_key is not None:
            atoms.info["energy"] = float(A[energy_key][i])
        if "stress" in A:
            st = A["stress"][i]
            st = st.reshape(-1)
            # write up to 9 components if present
            atoms.info["stress"] = st[:9]

        ase_write(str(out_path), atoms, format="extxyz", append=True)


def database_to_extxyz(
    data_file: Union[str, os.PathLike],
    output_file: Optional[Union[str, os.PathLike]] = None,
    overwrite: bool = True,
    pbc: Union[bool, Tuple[bool, bool, bool]] = (False, False, False),
):
    """
    Convenience wrapper: load a database from file and write it to EXTXYZ.
    """
    db, _energies_key = load_base_database(data_file)
    out = (
        Path(str(output_file))
        if output_file is not None
        else Path(os.path.splitext(os.path.basename(str(data_file)))[0] + ".extxyz")
    )
    write_extxyz(db, out, overwrite=overwrite, pbc=pbc)

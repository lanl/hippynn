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


#: Built-in key name sets for auto-detection, searched when auto_detect_key is given a hint instead of a keyset.
BUILTIN_AUTO_KEYSETS = {
    'SPECIES_KEYSET': ['species', 'atomic_numbers', 'z', 'atom_types', 'atomic_number'],
    'COORDINATES_KEYSET': ['coordinates', 'positions', 'pos', 'coords', 'r'],
    'ENERGIES_KEYSET': ['energy', 'energies', 'e', 'total_energy'],
    'FORCES_KEYSET': ['forces', 'force', 'f'],
    'CELL_KEYSET': ['cell', 'lattice', 'box', 'unit_cell', 'c'],
    'CHARGES_KEYSET': ['charges', 'charge', 'partial_charges', 'q'],
    'DIPOLE_KEYSET': ['dipole', 'dipoles', 'dipole_moment', 'mu'],
    'QUADRUPOLE_KEYSET': ['quadrupole', 'quadrupoles'],
    'STRESS_KEYSET': ['stress', 'stresses', 'virial'],
    'HESSIAN_KEYSET': ['hessian', 'hessians'],
}


def auto_detect_key(keys, keyset_or_hint: Union[list[str], str], required=True):
    """
    Auto-detect a database key from a set of possible names using case-insensitive matching.

    This function searches among keys returns the first unique match.
    If multiple matches are found, a ValueError is raised to avoid ambiguity. If no matches are found
    and the key is required, a ValueError is raised with available keys listed.

    See BUILTIN_AUTO_KEYSETS for valid hints.

    :param keys: available keys in the array dictionary
    :param keyset_or_hint: list of possible key name patterns to match, or a single hint string. If a hint
        string is given, it is matched (case-insensitively) against the aliases in the built-in
        keysets above, and the matching keyset is used in its place.
    :param required: whether this key is required (if False, returns None with warning if not found)
    :return: detected key name or None
    :raises ValueError: if a hint matches zero or more than one built-in keyset, if ambiguous
        (multiple matches), or if missing a required key

    Examples
    --------
    >>> from hippynn.databases.utils import auto_detect_key, 
    >>> auto_detect_key(keys, 'atomic_numbers')  # hint resolves to SPECIES_KEYSET
    'Species'

    """

    # Process a keyset
    if isinstance(keyset_or_hint, str):
        folded_hint = keyset_or_hint.casefold()
        candidates = [ks for ks in BUILTIN_AUTO_KEYSETS.values() if any(alias.casefold() == folded_hint for alias in ks)]
        if len(candidates) == 0:
            raise ValueError(
                f"Could not match hint {keyset_or_hint!r} against any built-in keyset.\n"
                f"Built-in keysets: {', '.join(BUILTIN_AUTO_KEYSETS)}.\n"
                f"Please pass an explicit keyset (list of aliases) instead."
            )
        elif len(candidates) > 1:
            raise ValueError(f"Hint {keyset_or_hint!r} matches multiple built-in keysets; please pass an explicit keyset instead.")
        keyset = candidates[0]
    else:
        keyset = keyset_or_hint

    # Normalize keys for case-insensitive matching
    normalized_keyset = [alias.casefold() for alias in keyset]

    # Find matches
    matches = [k for k in keys if k.casefold() in normalized_keyset]

    if len(matches) == 0:
        if required:
            raise ValueError(
                f"Could not auto-detect key. No matches found for possible keys: {keyset}.\n"
                f"Available keys: {list(keys)}\n"
                f"Please specify the key explicitly."
            )
        else:
            warnings.warn(f"Optional key not found for possible keys: {keyset}. Proceeding without it.", stacklevel=2)
            return None
    elif len(matches) == 1:
        return matches[0]
    else:  # len(matches) > 1, ambiguous
        raise ValueError(
            f"Could not auto-detect key. Multiple candidates found: {matches}.\n"
            f"Please specify the key explicitly to resolve ambiguity."
        )


def load_database(
    data_file: Union[str, os.PathLike],
    seed: int = 101,
    num_workers: int = 2,
    species_key: str = "species",
    coordinates_key: str = "coordinates",
    energies_key: str = "energy",
    forces_key: str = "forces",
    name: Optional[str] = None,
    files: Optional[list] = None,
):
    """
    Load a database with a consistent interface, dispatching on ``data_file``:

    - ``.npz`` file -> NPZDatabase
    - ``.h5``/``.hdf5`` file -> PyAniFileDB
    - directory containing ``.h5``/``.hdf5`` files -> PyAniDirectoryDB
    - directory containing ``.npy`` files -> DirectoryDatabase (requires ``name``)

    :param data_file: path to the dataset file or directory
    :param seed: random seed for the database split
    :param num_workers: number of dataloader workers (see :class:`~hippynn.databases.database.Database`)
    :param species_key: key name for species/atomic numbers in the dataset
    :param coordinates_key: key name for atomic coordinates in the dataset
    :param energies_key: key name for energies in the dataset
    :param forces_key: key name for forces in the dataset
    :param name: filename prefix for a directory of ``.npy`` files; required only in that case
    :param files: explicit list of ``.h5`` filenames to load from a directory; if None, all ``.h5`` files in the directory are used
    :return: database
    """
    from .ondisk import NPZDatabase, DirectoryDatabase
    from .h5_pyanitools import PyAniFileDB, PyAniDirectoryDB

    # Backend database class for each supported file extension
    _BASE_DATABASE_BACKENDS = {
        ".npz": NPZDatabase,
        ".h5": PyAniFileDB,
        ".hdf5": PyAniFileDB,
    }

    data_file = os.path.expanduser(str(data_file))
    inputs = [coordinates_key, species_key]
    targets = [energies_key, forces_key]

    if os.path.isdir(data_file):
        has_h5_files = any(f.lower().endswith((".h5", ".hdf5")) for f in os.listdir(data_file))
        if has_h5_files:
            db = PyAniDirectoryDB(
                directory=data_file,
                inputs=inputs,
                targets=targets,
                files=files,
                species_key=species_key,
                seed=seed,
                num_workers=num_workers,
                allow_unfound=True,
            )
        else:
            if name is None:
                raise ValueError("Loading a directory of .npy files requires `name` (the filename prefix) to be specified.")
            db = DirectoryDatabase(
                directory=data_file,
                name=name,
                inputs=inputs,
                targets=targets,
                seed=seed,
                num_workers=num_workers,
                allow_unfound=True,
                quiet=False,
            )
        return db

    ext = Path(data_file).suffix.lower()

    try:
        db_class = _BASE_DATABASE_BACKENDS[ext]
    except KeyError:
        raise ValueError(f"Unrecognized dataset file extension: {ext}. Supported file extensions are: .h5, .hdf5, .npz.")

    if db_class is PyAniFileDB:
        db = db_class(
            file=data_file,
            species_key=species_key,
            seed=seed,
            num_workers=num_workers,
            allow_unfound=True,
            inputs=inputs,
            targets=targets,
        )
    else:
        db = db_class(
            file=data_file,
            seed=seed,
            num_workers=num_workers,
            allow_unfound=True,
            inputs=inputs,
            targets=targets,
            quiet=False,
        )

    return db


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

    Expected keys in ``database.arr_dict``, all optional except ``coordinates`` and ``species``:
    ``coordinates`` (n, max_atoms, 3), ``species`` (n, max_atoms) int padded with <= 0,
    ``forces`` (n, max_atoms, 3), ``atomenergies`` (n, max_atoms, 1) or (n, max_atoms),
    ``energy``/``energies`` (n,), ``cell`` (n, 3, 3), ``stress`` (n, 3, 3) or (n, 9).

    :param database: hippynn Database (or any object exposing ``arr_dict`` and, for ``split``,
     ``splits``/``write_npz``) to export
    :param filename: output path for the EXTXYZ file
    :param overwrite: if False, raise ``FileExistsError`` when ``filename`` already exists
    :param pbc: ``False`` for non-periodic (default), ``True`` for periodic in all directions,
     or a tuple/list of three bools for per-axis periodicity
    :param split: if a split name, write only that split; if ``True``, write the full dataset
     (as it would be written to NPZ); if ``None`` (default), write ``database.arr_dict`` directly
    """
    from ase import Atoms
    from ase.io import write as ase_write

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

    atoms_list = []
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

        atoms_list.append(atoms)

    if not atoms_list:
        warnings.warn("No frames with valid atoms found; writing an empty EXTXYZ file.", stacklevel=2)
    ase_write(str(out_path), atoms_list, format="extxyz")

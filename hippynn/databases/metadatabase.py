"""
MetaDatabase

Parses a `Database` object to extract species, positions, forces, and other relevant data,
organizing them into structured metadata. Calculates metrics such as force magnitudes,
pairwise atomic distances, and simulation box densities to facilitate data searching
and visualization.

Designed for easy extension with additional metadata calculations and visualization methods.
"""
from .database import Database

from ase.data import atomic_masses, chemical_symbols
from ase.units import _amu
import json
import numpy as np
import torch
from collections import defaultdict, Counter
from itertools import islice 
import re
import matplotlib.pyplot as plt  
from hippynn.pretraining import compute_hipnn_e0
from hippynn.layers.indexers import OneHotSpecies

class MetaDatabase(Database):
    """
    MetaDatabase

    A class to parse a database object and generate a metadata representation.
    This metadata facilitates searching, filtering, and visualization of the underlying representations.

    Attributes
    ----------
    db : Database
        The original database object to be parsed.
    metadata : dict
        A dictionary containing extracted metadata from the database.
    densities : list of float
        Calculated density values for each entry in the dataset.
    atom_counts : dict of {int: int}
        A mapping from atomic numbers to their respective atom counts in the dataset.

    Methods
    -------
    calculate_atom_counts()
        Computes the number of atoms for each unique atomic number present in the dataset.
    
    calculate_densities()
        Calculates the density for each entry based on mass and volume.
    
    search(criteria)
        Searches the metadata based on specified criteria and returns matching entries.
                search_entries_by_distance_range(distance_range):
                search_entries_by_max_force(force_range):
                search_entries_by_species(target_species):
    
    plot_density_distribution()
        Generates a histogram plot of the density distribution across the dataset.
    
    plot_atom_counts()
        Creates a bar chart representing the count of each atomic number in the dataset.
    
    plot_coordinates(entry_id)
        Plots the spatial coordinates of atoms for a given entry in 3D space.
    
    Examples
    --------
    >>> # Initialize MetaDatabase with an existing database object
    >>> from hippynn.databases import metadatabase
    >>> meta_db = metadatabase.MetaDatabase(
    >>> arr_dict = db.arr_dict,
    >>> inputs=inputs,
    >>> targets=targets,
    >>> seed=12345,
    >>> num_workers=1,
    >>> pin_memory=True,
    >>> allow_unfound=True,
    >>> quiet=True,        
    >>> species_key='species',
    >>> coordinates_key='coordinates',
    >>> energies_key='energy',
    >>> forces_key='forces',
    >>> cell_key="cell",
    >>> pair_dist_hard_max = 4.0,
    >>> metadata={ 
    >>>    "Energy_unit" : 'eV',
    >>>    "Mass_unit" : 'grams/mol', 
    >>>    "Distance_unit" : 'Angstroms',
    >>>    "Electronic_Structure_Package" : 'VASP',
    >>>    "Electronic_Structure_Package_Version" : '6.4.3',
    >>>    "Computer_System" : 'LANL',
    >>>    "Input_Procedure" : ''
    >>> },
    >>> populate_metadata=True,
    >>> write_metadata_to_json=True,
    >>> json_filename='metadata.json',
    >>> distribution_plots=False,
    >>> )
    
    >>> # Calculate atom counts and densities
    >>> meta_db.calculate_atom_counts()
    >>> meta_db.calculate_densities()
    
    >>> # Search for entries with density between 1.0 and 5.0
    >>> results = meta_db.search({'density': {'min': 1.0, 'max': 5.0}})
    
    >>> # Plot the Force Magnitude Distribution, Density Distribution and Pairwise Distance Distribution
    >>> meta_db.plot_distributions(
    >>> density_range=(0.1, 1.5), 
    >>>     max_force_range=(0, 1),   
    >>>     min_distance_range=(0, 5), 
    >>>     bins=100,                  
    >>>     alpha=0.5       
    >>>     )
    
    >>> # Update metadata with a single "Comments" key
    >>> meta_db.update_metadata({"Comments": '' })

    >>> # Remove "Input_Proceedure" key from metadata
    >>> meta_db.remove_metadata("Input_Proceedure")

    >>> # Print metadata
    >>> meta_db.print_metadata()

    >>> # Search for indicies out of all entries containing atleast Carbon 
    >>> meta_db.search_entries_by_species(['C'], exact_match=False)
    
    >>> # Search for indicies out of all databaseentries containing exactly Hydrogen, Carbon and Oxygen
    >>> meta_db.search_entries_by_species(['CHO'], exact_match=True)
    
    >>> # Search for indicies out of all database entries with a calculated maximum atomic force in the range of [0,0.1]
    >>> meta_db.search_entries_by_max_force([0.0,0.1])

    >>> # Search for indicies out of all database entries with a calculated maximum pairwise atomic distance in the range of [0,0.9]
    >>> meta_db.search_entries_by_distance_range([0.0,0.9])
    
    >>> # Plot Distributions


    **Key Functionalities:**

    1. **Parsing and Metadata Extraction:**
       - Extracts species and coordinate information from the database.
       - Computes unique atomic numbers and counts of each atom type.
       - Calculates physical properties like mass and density based on extracted data.

    2. **Searching Capabilities:**
       - Enables complex queries based on multiple criteria (e.g., density ranges,
         specific atomic compositions).
       - Supports logical operations (AND, OR, NOT) to refine search results.
       - Returns entries that match the specified search parameters.

    3. **Plotting and Visualization:**
       - Provides methods to visualize database distributions (e.g., pairwise atomic distances, density, force, histograms).
       - Generates plots for atom counts to understand elemental compositions of database.


    """

"""
MetaDatabase

Parses a `Database` object to extract species, positions, forces, and other relevant data,
organizing them into structured metadata. Calculates metrics such as force magnitudes,
pairwise atomic distances, and simulation box densities to facilitate data searching
and visualization.

Designed for easy extension with additional metadata calculations and visualization methods.
"""
import json
import re
from collections import defaultdict, Counter
from itertools import islice

import numpy as np
import torch
import matplotlib.pyplot as plt
from ase.data import atomic_masses, chemical_symbols
from ase.units import _amu
from scipy.spatial.distance import cdist

from hippynn.databases.database import Database
from hippynn.pretraining import compute_hipnn_e0
from hippynn.layers.indexers import OneHotSpecies


class MetaDatabase(Database):
    """
    MetaDatabase

    A class to parse a database object and generate a metadata representation.
    This metadata facilitates searching, filtering, and visualization of the underlying representations.
    """

    def __init__(
        self,
        arr_dict,
        inputs,
        targets,
        species_key='species',
        coordinates_key='coordinates',
        energies_key='energies',
        forces_key='forces',
        cell_key=None,
        metadata: dict[str, object] = None,
        entry_metadata: dict[int, dict[str, object]] = None,
        populate_metadata=True,
        pair_dist_hard_max=5.0,
        write_metadata_to_json=True,
        json_filename='metadata.json',
        write_metadata_to_csv=True,
        csv_filename='metadata.csv',
        distribution_plots=False,
        density_range=None,
        max_force_range=None,
        min_distance_range=None,
        energies_range=None,
        bins=50,
        alpha=0.7,
        peratom=False,
        **kwargs
    ):
        # Global and per-entry metadata
        self.metadata = metadata.copy() if metadata else {}
        self.entry_metadata = entry_metadata.copy() if entry_metadata else {}

        # Initialize base Database
        super().__init__(
            arr_dict=arr_dict,
            inputs=inputs,
            targets=targets,
            **kwargs
        )

        # Keys
        self.species_key = species_key
        self.coordinates_key = coordinates_key
        self.energies_key = energies_key
        self.forces_key = forces_key
        self.cell_key = cell_key

        # Settings
        self.pair_dist_hard_max = pair_dist_hard_max
        self.write_metadata_to_json = write_metadata_to_json
        self.json_filename = json_filename
        self.write_metadata_to_csv = write_metadata_to_csv
        self.csv_filename = csv_filename

        # Distribution / filtering settings
        self.distribution_plots = distribution_plots
        self.density_range = density_range
        self.max_force_range = max_force_range
        self.min_distance_range = min_distance_range
        self.energies_range = energies_range
        self.bins = bins
        self.alpha = alpha
        self.peratom = peratom

        # Computed caches
        self.atomic_numbers_in_dataset = None
        self.element_combinations = None
        self.atom_counts = None
        self.entry_species_index = None
        self.densities = None
        self.max_force = None
        self.min_force = None
        self.min_distance = None
        self.E0_regression = None

        # Auto-populate metadata
        if populate_metadata:
            self.populate_metadata(update=True, quiet=False)

        # Optional plots
        if self.distribution_plots:
            self.plot_distributions()

    # ─── Utility ────────────────────────────────────────────────────────────────

    def _to_numpy(self, data):
        """Convert a torch.Tensor to numpy.ndarray, leave numpy arrays unchanged."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    # ─── Entry-level Metadata ────────────────────────────────────────────────────

    def set_entry_metadata(self, index: int, metadata: dict[str, object]):
        self.entry_metadata[index] = metadata

    def get_entry_metadata(self, index: int) -> dict[str, object]:
        return self.entry_metadata.get(index, {})

    def update_entry_metadata(self, index: int, metadata: dict[str, object]):
        if index not in self.entry_metadata:
            self.entry_metadata[index] = {}
        self.entry_metadata[index].update(metadata)

    def remove_entry_metadata(self, index: int):
        self.entry_metadata.pop(index, None)

    def print_all_entry_metadata(self):
        print("Entry Metadata:")
        for idx, md in self.entry_metadata.items():
            print(f"  Entry {idx}: {md}")

    def metadata_generator(self):
        """
        Yield dicts {'species': tensor/array, 'coordinates': tensor/array} per entry.
        Converts to numpy for downstream parsing.
        """
        species_arr = self._to_numpy(self.arr_dict[self.species_key])
        coords_arr = self._to_numpy(self.arr_dict[self.coordinates_key])
        for sp, cr in zip(species_arr, coords_arr):
            yield {self.species_key: sp, 'coordinates': cr}

    # ─── Global Metadata ────────────────────────────────────────────────────────

    def set_metadata(self, key: str, value: object):
        self.metadata[key] = value

    def get_metadata(self, key: str):
        return self.metadata.get(key)

    def update_metadata(self, new_md: dict[str, object]):
        self.metadata.update(new_md)

    def remove_metadata(self, key: str):
        self.metadata.pop(key, None)

    def print_metadata(self):
        print("Metadata:")
        for k, v in self.metadata.items():
            print(f"  {k}: {v}")

    # ─── Atom/Mass Mapping ───────────────────────────────────────────────────────

    def convert_atomic_number_to_symbol(self):
        return {i: sym for i, sym in enumerate(chemical_symbols) if sym}

    def convert_symbol_to_atomic_number(self):
        return {sym: i for i, sym in enumerate(chemical_symbols) if sym}

    def atomic_masses(self):
        unit = self.metadata.get("Mass_unit", "grams/mol")
        conv = {
            "grams/mol": 1.0,
            "amu": 1.0 / 1.66053906660e-24,
            "kg": _amu
        }
        if unit not in conv:
            raise ValueError(f"Unsupported Mass_unit: {unit}")
        factor = conv[unit]
        return {sym: atomic_masses[i] * factor
                for i, sym in enumerate(chemical_symbols) if sym}

    def get_mass_from_species(self, species):
        if species == 0:
            return 0.0
        masses = self.atomic_masses()
        num2sym = self.convert_atomic_number_to_symbol()
        sym = num2sym.get(species)
        return masses.get(sym, 0.0)

    # ─── Parsing & Extraction ────────────────────────────────────────────────────

    def extract_element_combinations_large(self, chunk_size=100_000):
        combos = Counter()
        for entry in self.metadata_generator():
            sp = entry[self.species_key]
            unique = tuple(sorted(set(sp[sp != 0])))
            combos[unique] += 1
        self.element_combinations = dict(combos)
        return self.element_combinations

    def extract_unique_numbers_large(self):
        self.extract_element_combinations_large()
        nums = {n for combo in self.element_combinations for n in combo}
        self.atomic_numbers_in_dataset = sorted(nums)
        return self.atomic_numbers_in_dataset

    
    # ─── Geometric & Physical Calculations ─────────────────────────────────────

    def calculate_volume(self, coordinates, cell=None):
        coords = self._to_numpy(coordinates)
        res = {"bounding_box_volume": None, "cell_volume": None}
        if coords.size > 0:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            res["bounding_box_volume"] = np.prod(maxs - mins)
        if cell is not None:
            cell_np = self._to_numpy(cell)
            if cell_np.shape != (3, 3):
                raise ValueError("Cell must be 3×3")
            res["cell_volume"] = abs(np.linalg.det(cell_np))
        return res
    
    def find_pairs(self, coordinates, species, cell=None, periodic=True):
        coords = self._to_numpy(coordinates)
        if coords.shape[0] < 2:
            return {"pair_dist": np.array([]),
                    "pair_first": np.array([]),
                    "pair_second": np.array([]),
                    "pair_coord": np.array([]),
                    "cell_offsets": np.array([]) if periodic else None}
        if periodic and cell is not None:
            cell_np = self._to_numpy(cell)
            frac = (coords @ np.linalg.inv(cell_np)) % 1.0
            cart = frac @ cell_np
            dists = cdist(cart, cart)
        else:
            dists = cdist(coords, coords)

        np.fill_diagonal(dists, np.inf)
        mask = dists < self.pair_dist_hard_max
        idx = np.argwhere(mask)
        pd = dists[mask]
        first, second = idx[:, 0], idx[:, 1]
        rel = coords[second] - coords[first]
        out = {"pair_dist": pd,
               "pair_first": first,
               "pair_second": second,
               "pair_coord": rel}
        if periodic and cell is not None:
            offs = frac[second] - frac[first]
            offs -= np.round(offs)
            out["cell_offsets"] = offs @ cell_np
        return out

    def calculate_min_distance(self, periodic=True):
        mins = []
        species_arr = self._to_numpy(self.arr_dict[self.species_key])
        coords_arr = self._to_numpy(self.arr_dict[self.coordinates_key])
        cell_arr = self.arr_dict.get(self.cell_key)
        for i, sp in enumerate(species_arr):
            mask = sp != 0
            valid = mask.sum() >= 2
            if not valid:
                mins.append(np.inf)
                continue
            coords = coords_arr[i][mask]
            cell_i = cell_arr[i] if cell_arr is not None else None
            pr = self.find_pairs(coords, None, cell=cell_i, periodic=periodic)
            if pr["pair_dist"].size:
                mins.append(pr["pair_dist"].min())
            else:
                mins.append(np.inf)
        self.min_distance = np.array(mins)
        return self.min_distance

    def calculate_max_force(self):
        f = self.arr_dict[self.forces_key]
        if isinstance(f, torch.Tensor):
            mags = torch.norm(f, dim=2)
            self.max_force = mags.max(dim=1).values.cpu().numpy()
        else:
            arr = self._to_numpy(f)
            self.max_force = np.linalg.norm(arr, axis=2).max(axis=1)
        return self.max_force

    def calculate_min_force(self):
        f = self.arr_dict[self.forces_key]
        if isinstance(f, torch.Tensor):
            mags = torch.norm(f, dim=2)
            self.min_force = mags.min(dim=1).values.cpu().numpy()
        else:
            arr = self._to_numpy(f)
            self.min_force = np.linalg.norm(arr, axis=2).min(axis=1)
        return self.min_force

    def calculate_densities(self):
        densities = []
        species_arr = self._to_numpy(self.arr_dict[self.species_key])
        coords_arr = self._to_numpy(self.arr_dict[self.coordinates_key])
        cell_arr = self.arr_dict.get(self.cell_key)
        unique_sp = np.unique(species_arr)
        sp_masses = {sp: self.get_mass_from_species(int(sp)) for sp in unique_sp}
        for i, sp in enumerate(species_arr):
            mask = sp != 0
            mass = sum(sp_masses[int(x)] for x in sp[mask])
            vol = self.calculate_volume(coords_arr[i], 
                        cell=cell_arr[i] if cell_arr is not None else None)
            V = vol["cell_volume"] or vol["bounding_box_volume"]
            densities.append((mass / V) if (mass > 0 and V and V > 0) else None)
        self.densities = np.array(densities, dtype=float)
        return self.densities

    def calculate_E0_regression(self):
        nums = self.atomic_numbers_in_dataset or self.extract_unique_numbers_large()
        if 0 not in nums:
            nums = [0] + nums
        encoder = OneHotSpecies(nums)
        species_t = torch.tensor(self._to_numpy(self.arr_dict[self.species_key]), dtype=torch.long)
        energies_t = torch.tensor(self._to_numpy(self.arr_dict[self.energies_key]), dtype=torch.float32)
        e0 = compute_hipnn_e0(encoder, species_t, energies_t, peratom=self.peratom)
        self.E0_regression = e0.cpu().numpy()
        return self.E0_regression

    # ─── Atom Counts & Combinations ────────────────────────────────────────────

    def count_atoms_by_type(self, atomic_number):
        arr = self._to_numpy(self.arr_dict[self.species_key])
        return int((arr == atomic_number).sum())

    def calculate_atom_counts(self):
        self.extract_unique_numbers_large()
        self.atom_counts = {n: self.count_atoms_by_type(n) for n in self.atomic_numbers_in_dataset}
        return self.atom_counts

    def build_entry_species_index(self):
        self.entry_species_index = defaultdict(list)
        arr = self._to_numpy(self.arr_dict[self.species_key])
        for i, sp in enumerate(arr):
            combo = tuple(sorted(set(sp[sp != 0])))
            self.entry_species_index[combo].append(i)
        return self.entry_species_index

    # ─── Search ────────────────────────────────────────────────────────────────

    def search_entries_by_species(self, target_species, exact_match=True, use_symbols=True):
        if self.entry_species_index is None:
            self.build_entry_species_index()

        if use_symbols:
            s2n = self.convert_symbol_to_atomic_number()
            valid_nums = set(self._to_numpy(self.arr_dict[self.species_key]).flatten())
            valid_syms = {s for s, n in s2n.items() if n in valid_nums}
            expanded = []
            for it in target_species:
                if isinstance(it, str):
                    parts = re.findall(r"[A-Z][a-z]?", it)
                    for p in parts:
                        if p not in valid_syms:
                            raise ValueError(f"Invalid symbol {p}")
                        expanded.append(p)
                else:
                    expanded.append(it)
            target_nums = [s2n[p] for p in expanded]
        else:
            target_nums = list(target_species)

        tgt = set(target_nums)
        matches = []
        for combo, idxs in self.entry_species_index.items():
            s = set(combo)
            if (exact_match and s == tgt) or (not exact_match and tgt.issubset(s)):
                matches += idxs
        return matches

    def search_entries_by_max_force(self, force_range):
        if self.max_force is None:
            self.calculate_max_force()
        lo, hi = force_range
        if lo > hi:
            raise ValueError("min > max")
        arr = np.array(self.max_force)
        return np.where((arr >= lo) & (arr <= hi))[0].tolist()

    def search_entries_by_distance_range(self, distance_range):
        if self.min_distance is None:
            self.calculate_min_distance()
        lo, hi = distance_range
        if lo > hi:
            raise ValueError("min > max")
        mn = np.array(self.min_distance)
        # no max_distance stored; only min-distance available
        return np.where((mn >= lo) & (mn <= hi))[0].tolist()

    # ─── Getters & Statistics ──────────────────────────────────────────────────

    def get_element_combinations(self):
        if self.element_combinations is None:
            raise RuntimeError("Run extract_element_combinations_large() first")
        num2sym = self.convert_atomic_number_to_symbol()
        return {"".join(num2sym[n] for n in combo): cnt
                for combo, cnt in self.element_combinations.items()}

    def get_atom_counts_by_symbol(self):
        if self.atom_counts is None:
            raise RuntimeError("Run calculate_atom_counts() first")
        num2sym = self.convert_atomic_number_to_symbol()
        return {num2sym[n]: cnt for n, cnt in self.atom_counts.items()}

    def calculate_range(self, data, manual_range):
        if manual_range:
            return manual_range
        if len(data):
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            return (q1 - 2.5*iqr, q3 + 2.5*iqr)
        return (None, None)

    def get_density_statistics(self):
        if self.densities is None:
            self.calculate_densities()
        vals = self.densities[np.isfinite(self.densities)]
        lo, hi = self.calculate_range(vals, self.density_range)
        filt = vals[(vals >= lo) & (vals <= hi)]
        if filt.size:
            return {
                "min": filt.min(),
                "max": filt.max(),
                "mean": filt.mean(),
                "median": np.median(filt),
                "std": filt.std(),
                "outliers": len(self.densities) - len(filt)
            }
        return {k: None for k in ("min", "max", "mean", "median", "std", "outliers")}

    def get_min_distance_statistics(self):
        if self.min_distance is None:
            self.calculate_min_distance()
        vals = np.array(self.min_distance)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            return {
                "min": vals.min(),
                "max": vals.max(),
                "mean": vals.mean(),
                "median": np.median(vals),
                "std": vals.std()
            }
        return {k: None for k in ("min", "max", "mean", "median", "std")}

    def get_max_force_statistics(self):
        if self.max_force is None:
            self.calculate_max_force()
        vals = np.array(self.max_force)
        vals = vals[np.isfinite(vals)]
        lo, hi = self.calculate_range(vals, self.max_force_range)
        filt = vals[(vals >= lo) & (vals <= hi)]
        if filt.size:
            return {
                "min": filt.min(),
                "max": filt.max(),
                "mean": filt.mean(),
                "median": np.median(filt),
                "std": filt.std(),
                "outliers": len(vals) - len(filt),
            }
        return {k: None for k in ("min", "max", "mean", "median", "std", "outliers")}

    def get_energy_statistics(self):
        if self.E0_regression is None:
            self.calculate_E0_regression()
        nums = self.atomic_numbers_in_dataset or self.extract_unique_numbers_large()
        if 0 not in nums:
            nums = [0] + nums
        enc = OneHotSpecies(nums)
        sp_t = torch.tensor(self._to_numpy(self.arr_dict[self.species_key]), dtype=torch.long)
        enc_sp = enc(sp_t)[0].to(torch.float64)
        lin_e = torch.tensordot(enc_sp,
                                torch.tensor(self.E0_regression, dtype=torch.float64),
                                dims=([2], [0])
                                ).sum(dim=1)
        defects = torch.tensor(self._to_numpy(self.arr_dict[self.energies_key]), dtype=torch.float64) - lin_e
        arr = defects.cpu().numpy()
        lo, hi = self.calculate_range(arr, self.energies_range)
        filt = arr[(arr >= lo) & (arr <= hi)]
        if filt.size:
            return {
                "min": filt.min(),
                "max": filt.max(),
                "mean": filt.mean(),
                "median": np.median(filt),
                "std": filt.std(),
                "outliers": len(arr) - len(filt),
            }
        return {k: None for k in ("min", "max", "mean", "median", "std", "outliers")}

    # ─── Populate & Save ──────────────────────────────────────────────────────

    def make_json_serializable(self):
        def convert(v):
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: convert(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [convert(vv) for vv in v]
            return v
        return {k: convert(v) for k, v in self.metadata.items()}

    def save_metadata_to_json(self):
        with open(self.json_filename, "w") as f:
            json.dump(self.make_json_serializable(), f, indent=4)

    def save_metadata_to_csv(self):
        flat = {}
        def _flat(d, prefix=None):
            for k, v in d.items():
                key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, dict):
                    _flat(v, key)
                else:
                    flat[key] = str(v)
        _flat(self.make_json_serializable())
        with open(self.csv_filename, "w") as f:
            for k, v in flat.items():
                f.write(f"{k},{v}\n")

    def populate_metadata(self, update=True, quiet=False):
        md_upd = {}
        try:
            self.calculate_densities()
            md_upd["density_statistics"] = self.get_density_statistics()
        except Exception as e:
            if not quiet: print("Error density:", e)

        try:
            self.calculate_max_force()
            md_upd["max_force_statistics"] = self.get_max_force_statistics()
        except Exception as e:
            if not quiet: print("Error max force:", e)

        try:
            self.calculate_E0_regression()
            md_upd["energy_statistics"] = self.get_energy_statistics()
        except Exception as e:
            if not quiet: print("Error energy stats:", e)

        try:
            self.calculate_min_force()
        except Exception as e:
            if not quiet: print("Error min force:", e)

        try:
            self.calculate_min_distance()
            md_upd["min_distance_statistics"] = self.get_min_distance_statistics()
        except Exception as e:
            if not quiet: print("Error min distance:", e)

        try:
            self.calculate_atom_counts()
            md_upd["atom_count"] = self.get_atom_counts_by_symbol()
        except Exception as e:
            if not quiet: print("Error atom counts:", e)

        try:
            self.extract_unique_numbers_large()
            md_upd["element_combinations"] = self.get_element_combinations()
        except Exception as e:
            if not quiet: print("Error combinations:", e)

        if update:
            self.update_metadata(md_upd)

        if not quiet:
            print("Metadata populated:")
            for k, v in md_upd.items():
                print(f"  {k}: {v}")

        if self.write_metadata_to_json:
            try: self.save_metadata_to_json()
            except Exception as e:
                if not quiet: print("Error saving JSON:", e)

        if self.write_metadata_to_csv:
            try: self.save_metadata_to_csv()
            except Exception as e:
                if not quiet: print("Error saving CSV:", e)

    # ─── Plotting ──────────────────────────────────────────────────────────────

    def plot_distributions(
        self,
        density_range=None,
        max_force_range=None,
        min_distance_range=None,
        bins=None,
        alpha=None
    ):
        import matplotlib.pyplot as plt
        import numpy as np
    
        def safe_array(arr, label, max_len=10000):
            """Convert to NumPy, filter NaNs and Infs, and truncate if too long."""
            arr = np.array(arr, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                print(f"[Warning] No finite values found for {label}. Skipping plot.")
            if arr.size > max_len:
                arr = arr[:max_len]
            return arr
    
        # Ensure data is computed
        if self.densities is None:
            self.calculate_densities()
        if self.max_force is None:
            self.calculate_max_force()
        if self.min_distance is None:
            self.calculate_min_distance()
    
        # Sanitize arrays
        dvals = safe_array(self.densities, "densities")
        fvals = safe_array(self.max_force, "max_force")
        mvals = safe_array(self.min_distance, "min_distance")
    
        # Compute plot ranges
        dr = self.calculate_range(dvals, density_range or self.density_range)
        fr = self.calculate_range(fvals, max_force_range or self.max_force_range)
        mr = self.calculate_range(mvals, min_distance_range or self.min_distance_range)
    
        # Use defaults if not provided
        b = bins or self.bins
        a = alpha or self.alpha
    
        # Start plotting
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
        if dvals.size:
            axs[0, 0].hist(dvals, bins=b, alpha=a)
            axs[0, 0].set(title="Density Distribution", xlabel="Density", ylabel="Count")
            axs[0, 0].set_xlim(dr)
        else:
            axs[0, 0].set(title="Density: No Data")
    
        if fvals.size:
            axs[0, 1].hist(fvals, bins=b, alpha=a)
            axs[0, 1].set(title="Max Force Distribution", xlabel="Force", ylabel="Count")
            axs[0, 1].set_xlim(fr)
        else:
            axs[0, 1].set(title="Max Force: No Data")
    
        if mvals.size:
            axs[1, 0].hist(mvals, bins=b, alpha=a)
            axs[1, 0].set(title="Min Distance Distribution", xlabel="Distance", ylabel="Count")
            axs[1, 0].set_xlim(mr)
        else:
            axs[1, 0].set(title="Min Distance: No Data")
    
        # Atom counts
        counts = self.get_atom_counts_by_symbol()
        if counts:
            syms, cnts = zip(*counts.items())
            axs[1, 1].bar(syms, cnts, alpha=a)
            axs[1, 1].set(title="Atom Counts by Symbol", xlabel="Element", ylabel="Count")
        else:
            axs[1, 1].set(title="Atom Counts: No Data")
    
        plt.tight_layout()
        plt.show()
    
    


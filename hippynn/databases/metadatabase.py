"""MetaDatabase

Parses a dictionary of arrays (e.g. from a :class:`~hippynn.databases.database.Database`)
to extract species, positions, forces, and other relevant data, organizing them into
structured metadata. Calculates metrics such as force magnitudes, pairwise atomic
distances, and simulation box densities to facilitate data searching and visualization.
"""

from ase.data import atomic_masses, chemical_symbols

from ase.units import _amu
import json
import copy
import torch
from collections import defaultdict
import re

from ..layers.indexers import OneHotSpecies
from ..pretraining import calculate_min_dists, compute_hipnn_e0
from ..tools import progress_bar
from .utils import auto_detect_key


class MetaDatabase:
    """
    Parse a dictionary of arrays and generate a metadata representation.

    This metadata facilitates searching, filtering, and visualization of molecular database contents.

    Database keys (species_key, coordinates_key, etc.) can be explicitly provided or auto-detected.
    See :func:`auto_detect_key` for details on auto-detection behavior and supported key names.

    Examples
    >>> # Initialize MetaDatabase with an existing database object
    >>> from hippynn.databases import metadatabase
    >>> meta_db = metadatabase.MetaDatabase(
    >>> arr_dict = db.arr_dict,
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
    >>> )
    >>>
    >>> # Save metadata to files
    >>> meta_db.save_metadata_to_json('metadata.json')
    >>> meta_db.save_metadata_to_csv('metadata.csv')
    >>>
    >>> # Generate plots
    >>> meta_db.plot_distributions(density_range=(0.1, 1.5), bins=100, alpha=0.5)

    >>> # Calculate atom counts and densities
    >>> meta_db.species_counts
    >>> meta_db.density

    >>> # Plot the Force Magnitude Distribution, Density Distribution and Pairwise Distance Distribution
    >>> meta_db.plot_distributions(
    >>> density_range=(0.1, 1.5),
    >>>     max_force_range=(0, 1),
    >>>     min_distance_range=(0, 5),
    >>>     bins=100,
    >>>     alpha=0.5
    >>>     )

    >>> # Update metadata with a single "Comments" key
    >>> meta_db.metadata["Comments"] = ''

    >>> # Remove "Input_Procedure" key from metadata
    >>> meta_db.metadata.pop("Input_Procedure", None)

    >>> # Search for indicies out of all entries containing atleast Carbon
    >>> meta_db.search_entries_by_species(['C'], exact_match=False)

    >>> # Search for indicies out of all databaseentries containing exactly Hydrogen, Carbon and Oxygen
    >>> meta_db.search_entries_by_species(['CHO'], exact_match=True)

    >>> # Search for indicies out of all database entries with a calculated maximum atomic force in the range of [0,0.1]
    >>> meta_db.search_entries_by_max_force([0.0,0.1])

    >>> # Search for indicies out of all database entries with a calculated maximum pairwise atomic distance in the range of [0,0.9]
    >>> meta_db.search_entries_by_min_distance([0.0,0.9])

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

    # ---------------------------------------------------------------------
    # Class‑level constants and cached mappings
    # ---------------------------------------------------------------------
    PAIR_DIST_HARD_MAX_DEFAULT = 5.0
    DEFAULT_BINS = 50
    DEFAULT_ALPHA = 0.7

    # Cached atomic number ↔ symbol dictionaries to avoid recomputation
    ATOMIC_NUMBER_TO_SYMBOL = {i: sym for i, sym in enumerate(chemical_symbols) if sym}
    SYMBOL_TO_ATOMIC_NUMBER = {sym: i for i, sym in enumerate(chemical_symbols) if sym}

    class MetaDatabaseError(Exception):
        """Custom exception type for MetaDatabase specific errors."""

    def __init__(
        self,
        arr_dict,
        species_key=None,
        coordinates_key=None,
        energies_key=None,
        forces_key=None,
        cell_key=None,
        metadata: dict[str, object] = None,
        entry_metadata: dict[int, dict[str, object]] = None,
        populate_metadata=True,
        pair_dist_hard_max=PAIR_DIST_HARD_MAX_DEFAULT,
        peratom=False,
    ):
        # Global and per-entry metadata (deep copy to avoid shared mutable state)
        self.metadata = copy.deepcopy(metadata) if metadata else {}
        self.entry_metadata = copy.deepcopy(entry_metadata) if entry_metadata else {}

        self.arr_dict = {}
        for k, v in arr_dict.items():
            try:
                self.arr_dict[k] = torch.as_tensor(v)
            except Exception:
                # Skip non-tensor-convertible values
                pass

        arr_dict_keys = self.arr_dict.keys()
        self.species_key = species_key or auto_detect_key(arr_dict_keys, "species", required=True)
        self.coordinates_key = coordinates_key or auto_detect_key(arr_dict_keys, "coordinates", required=True)
        self.energies_key = energies_key or auto_detect_key(arr_dict_keys, "energy", required=False)
        self.forces_key = forces_key or auto_detect_key(arr_dict_keys, "forces", required=False)
        self.cell_key = cell_key or auto_detect_key(arr_dict_keys, "cell", required=False)

        # Cached validated tensors for required and optional arrays.
        # Initialize these explicitly so later methods do not depend on
        # attribute errors being swallowed in ``populate_metadata``.
        self._species_tensor = None
        self._coordinates_tensor = None
        self._energies_tensor = None
        self._forces_tensor = None
        self._cell_tensor = None

        self._validate_inputs()
        self.has_energies = self._energies_tensor is not None
        self.has_forces = self._forces_tensor is not None
        self.has_cell = self._cell_tensor is not None

        self.pair_dist_hard_max = pair_dist_hard_max
        self.peratom = peratom

        # Computed caches (backing storage for the lazy properties below)
        self._unique_species = None
        self._species_combination_counts = None
        self._symbols_combination_counts = None
        self._species_combination_index = None
        self._species_counts = None
        self._symbols_counts = None
        self._density = None
        self._volume = None
        self._max_force = None
        self._min_force = None
        self._min_distance = None
        self._E0_regression = None

        if populate_metadata:
            self.populate_metadata(quiet=False)

    # ─── Utility ────────────────────────────────────────────────────────────────

    def _to_tensor(self, data):

        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        return torch.as_tensor(data)

    def _validated_tensor(
        self,
        key,
        name,
        expected_ndim,
        dtype_check,
        *,
        required=False,
        shape=None,
        squeeze_trailing_column=False,
    ):
        if key is None or key not in self.arr_dict:
            if required:
                raise ValueError(f"Missing required key '{key}' in arr_dict")
            return None

        tensor = self._to_tensor(self.arr_dict[key])
        if squeeze_trailing_column and tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)

        if not isinstance(tensor, torch.Tensor):
            raise self.MetaDatabaseError(f"{name} must be a torch.Tensor")
        if tensor.ndim != expected_ndim:
            raise self.MetaDatabaseError(f"{name} must have {expected_ndim} dimensions, got {tensor.ndim}")
        if not dtype_check(tensor.dtype):
            raise self.MetaDatabaseError(f"{name} has incorrect dtype {tensor.dtype}")
        if shape is not None and tuple(tensor.shape) != tuple(shape):
            raise self.MetaDatabaseError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _force_magnitudes(self):
        if not self.has_forces:
            return None
        return torch.norm(self._forces_tensor, dim=2)

    def _calculate_force_extrema(self):
        mags = self._force_magnitudes()
        if mags is None:
            self._max_force = None
            self._min_force = None
            return None
        self._max_force = mags.max(dim=1).values
        self._min_force = mags.min(dim=1).values
        return mags

    def _search_entries_by_range(self, values, value_range):
        lo, hi = value_range
        if lo > hi:
            raise ValueError("min > max")
        mask = (values >= lo) & (values <= hi)
        return torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()

    # ─── Validation ───────────────────────────────────────────────────────────────

    def _validate_inputs(self):
        """
        Validate that required database entries have compatible shapes and dtypes.
        This method is called during initialization and raises a clear ``ValueError``
        if any inconsistency is detected.

        Expected shapes (where N = number of entries, M = number of atoms per entry):
        - ``species``: (N, M) integer type
        - ``coordinates``: (N, M, 3) floating point type
        - ``forces`` (optional): (N, M, 3) floating point type
        - ``energies`` (optional): (N,) floating point type
        - ``cell`` (optional): (N, 3, 3) floating point type
        """
        self._species_tensor = self._validated_tensor(
            self.species_key,
            "species",
            2,
            lambda dt: dt in (torch.int64, torch.int32, torch.int16, torch.uint8),
            required=True,
        )
        species = self._species_tensor
        n_entries, n_atoms = species.shape

        self._coordinates_tensor = self._validated_tensor(
            self.coordinates_key,
            "coordinates",
            3,
            lambda dt: dt.is_floating_point,
            required=True,
            shape=(n_entries, n_atoms, 3),
        )
        self._energies_tensor = self._validated_tensor(
            self.energies_key,
            "energies",
            1,
            lambda dt: dt.is_floating_point,
            shape=(n_entries,),
            squeeze_trailing_column=True,
        )
        self._forces_tensor = self._validated_tensor(
            self.forces_key,
            "forces",
            3,
            lambda dt: dt.is_floating_point,
            shape=self._coordinates_tensor.shape,
        )
        self._cell_tensor = self._validated_tensor(
            self.cell_key,
            "cell",
            3,
            lambda dt: dt.is_floating_point,
            shape=(n_entries, 3, 3),
        )

    # ─── Atom/Mass Mapping ───────────────────────────────────────────────────────

    def atomic_masses(self):

        unit = self.metadata.get("Mass_unit", "grams/mol")
        conv = {
            "grams/mol": 1.0,
            "amu": 1.0 / 1.66053906660e-24,
            "kg": _amu,
        }
        if unit not in conv:
            raise ValueError(f"Unsupported Mass_unit: {unit}")
        factor = conv[unit]
        return {sym: atomic_masses[i] * factor for i, sym in enumerate(chemical_symbols) if sym}

    def get_mass_by_species(self, species):
        if species == 0:
            return 0.0
        masses = self.atomic_masses()
        sym = self.ATOMIC_NUMBER_TO_SYMBOL.get(species)
        return masses.get(sym, 0.0)

    # ─── Species math ────────────────────────────────────────────────────

    def _extract_species_combination_data(self):
        index = defaultdict(list)
        rows = self._species_tensor.tolist()
        for i, row in enumerate(progress_bar(rows, desc="Species combinations", unit="entry")):
            s = set(row)
            s.discard(0)
            index[tuple(sorted(s))].append(i)
        self._species_combination_index = dict(index)
        self._species_combination_counts = {combo: len(idxs) for combo, idxs in self._species_combination_index.items()}

    @property
    def species_combination_counts(self):
        if self._species_combination_counts is None:
            self._extract_species_combination_data()
        return self._species_combination_counts

    @property
    def species_combination_index(self):
        if self._species_combination_index is None:
            self._extract_species_combination_data()
        return self._species_combination_index

    @property
    def unique_species(self):
        if self._unique_species is None:
            species = self._species_tensor[self._species_tensor != 0]
            self._unique_species = sorted(species.unique().tolist())
        return self._unique_species

    # ─── Geometric & Physical Calculations ─────────────────────────────────────

    def calculate_min_distance(self, periodic=True, batch_size=50):
        """
        Calculate minimum pairwise atomic distances for each entry.

        :param periodic: whether to use periodic boundary conditions if cell is available
        :param batch_size: batch size for distance calculation (also enables progress bar)
        :return: tensor of minimum distances for each entry
        """

        array_dict = {
            self.species_key: self._species_tensor.to(dtype=torch.int64),
            self.coordinates_key: self._coordinates_tensor,
        }

        cell_name = False
        if periodic and self.has_cell:
            array_dict[self.cell_key] = self._cell_tensor
            cell_name = self.cell_key

        self._min_distance = calculate_min_dists(
            array_dict=array_dict,
            species_name=self.species_key,
            positions_name=self.coordinates_key,
            dist_hard_max=self.pair_dist_hard_max,
            cell_name=cell_name,
            device=self._coordinates_tensor.device,
            batch_size=batch_size,
        )
        return self._min_distance

    @property
    def min_distance(self):
        """Minimum pairwise atomic distance per entry, shape ``(N,)``."""
        if self._min_distance is None:
            self.calculate_min_distance()
        return self._min_distance

    @property
    def max_force(self):
        """Maximum force magnitude per entry, shape ``(N,)``."""
        if self._max_force is None:
            self._calculate_force_extrema()
        return self._max_force

    @property
    def min_force(self):
        """Minimum force magnitude per entry, shape ``(N,)``."""
        if self._min_force is None:
            self._calculate_force_extrema()
        return self._min_force

    def calculate_volume(self, coordinates, cell=None):
        """
        Compute the bounding-box volume, and cell volume if a cell is given, for a single entry.

        :param coordinates: atomic positions, shape ``(n_atoms, 3)``
        :param cell: optional cell matrix, shape ``(3, 3)``
        :return: dict with keys ``bounding_box_volume`` and ``cell_volume`` (``None`` if ``cell`` not given)
        """
        coords = self._to_tensor(coordinates).to(dtype=torch.float64)
        result = {"bounding_box_volume": None, "cell_volume": None}
        if coords.numel() > 0:
            mins = coords.min(dim=0).values
            maxs = coords.max(dim=0).values
            result["bounding_box_volume"] = (maxs - mins).prod().item()
        if cell is not None:
            cell_t = self._to_tensor(cell).to(dtype=torch.float64)
            if tuple(cell_t.shape) != (3, 3):
                raise ValueError("Cell must be 3x3")
            result["cell_volume"] = torch.abs(torch.linalg.det(cell_t)).item()
        return result

    @property
    def density(self):
        """
        Density for each entry, computed lazily using vectorized operations.

        Computes mass from atomic species and volume from either periodic cells
        or bounding boxes. Caches both density and volumes for later use.
        """
        if self._density is not None:
            return self._density

        species_arr = self._species_tensor
        coords_arr = self._coordinates_tensor
        cell_arr = self._cell_tensor

        masses = self.atomic_masses()
        unique_sp = torch.unique(species_arr[species_arr != 0]).tolist()

        max_z = max(unique_sp) if unique_sp else 1
        mass_lookup = torch.zeros(max_z + 1, dtype=torch.float64)
        for sp in unique_sp:
            sym = self.ATOMIC_NUMBER_TO_SYMBOL.get(int(sp))
            mass_lookup[int(sp)] = masses.get(sym, 0.0)

        # Clamp to avoid indexing errors from any stray out-of-range species values
        species_clamped = torch.clamp(species_arr, 0, max_z).long()
        entry_masses = mass_lookup[species_clamped].sum(dim=1)  # (n_entries,)

        if cell_arr is not None:
            volumes = torch.abs(torch.linalg.det(cell_arr.to(dtype=torch.float64)))  # (n_entries,)
        else:
            mins = coords_arr.min(dim=1).values  # (n_entries, 3)
            maxs = coords_arr.max(dim=1).values  # (n_entries, 3)
            volumes = (maxs - mins).prod(dim=1)  # (n_entries,)

        self._volume = volumes

        densities = entry_masses / volumes
        densities = torch.where(
            (entry_masses > 0) & (volumes > 0) & torch.isfinite(volumes), densities, torch.tensor(float("nan"), dtype=torch.float64)
        )

        self._density = densities
        return self._density

    @property
    def volume(self):
        """Volume for each entry, computed lazily via the :attr:`density` property."""
        if self._volume is None:
            self.density
        return self._volume

    @property
    def E0_regression(self):
        if not self.has_energies:
            return None
        if self._E0_regression is None:
            nums = self.unique_species
            if 0 not in nums:
                nums = [0] + nums
            encoder = OneHotSpecies(nums)
            sp_t = self._species_tensor.long()
            energies_t = self._energies_tensor.float()
            e0 = compute_hipnn_e0(encoder, sp_t, energies_t, peratom=self.peratom)
            # Store as a torch tensor to avoid NumPy conversion
            self._E0_regression = e0.detach()
        return self._E0_regression

    # ─── Atom Counts & Combinations ────────────────────────────────────────────

    def count_atoms_by_species(self, species):
        return int((self._species_tensor == species).sum())

    @property
    def species_counts(self):
        """Atom counts keyed by atomic number."""
        if self._species_counts is None:
            self._species_counts = {n: self.count_atoms_by_species(n) for n in self.unique_species}
        return self._species_counts

    # ─── Search ────────────────────────────────────────────────────────────────

    def search_entries_by_species(self, target_species, exact_match=True, use_symbols=True):
        """
        Search for entries containing specified atomic species.

        :param target_species: list of element symbols or atomic numbers to search for
        :param exact_match: if True, entry must contain exactly these species; if False, at least these species
        :param use_symbols: if True, interpret input as element symbols; if False, as atomic numbers
        :return: list of matching entry indices
        """
        if use_symbols:
            valid_nums = set(self._species_tensor.flatten().tolist())
            valid_syms = {s for s, n in self.SYMBOL_TO_ATOMIC_NUMBER.items() if n in valid_nums}

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
            target_nums = [self.SYMBOL_TO_ATOMIC_NUMBER[p] for p in expanded]

        else:
            target_nums = list(target_species)

        tgt = set(target_nums)
        matches = []
        for combo, idxs in self.species_combination_index.items():
            s = set(combo)
            if (exact_match and s == tgt) or (not exact_match and tgt.issubset(s)):
                matches += idxs
        return matches

    def search_entries_by_max_force(self, force_range):
        return self._search_entries_by_range(self.max_force, force_range)

    def search_entries_by_min_distance(self, distance_range):
        return self._search_entries_by_range(self.min_distance, distance_range)

    # ─── Getters & Statistics ──────────────────────────────────────────────────

    @property
    def symbols_combination_counts(self):
        """Species-combination counts keyed by concatenated element symbols, e.g. ``"CHO"``."""
        if self._symbols_combination_counts is None:
            self._symbols_combination_counts = {
                "".join(self.ATOMIC_NUMBER_TO_SYMBOL[n] for n in combo): cnt for combo, cnt in self.species_combination_counts.items()
            }
        return self._symbols_combination_counts

    @property
    def symbols_counts(self):
        """Atom counts keyed by element symbol, e.g. ``"C"``."""
        if self._symbols_counts is None:
            self._symbols_counts = {self.ATOMIC_NUMBER_TO_SYMBOL[n]: cnt for n, cnt in self.species_counts.items()}
        return self._symbols_counts

    def calculate_range(self, data, manual_range):
        if manual_range is not None:
            return manual_range
        if data.numel() > 0:
            q1 = torch.quantile(data, 0.25).item()
            q3 = torch.quantile(data, 0.75).item()
            iqr = q3 - q1
            return (q1 - 2.5 * iqr, q3 + 2.5 * iqr)
        return (None, None)

    def _finite_values(self, values):
        values = self._to_tensor(values).to(dtype=torch.float64)
        return values[torch.isfinite(values)]

    def _empty_statistics(self, include_outliers=False):
        keys = ["min", "max", "mean", "median", "std"]
        if include_outliers:
            keys.append("outliers")
        return {k: None for k in keys}

    def _filtered_statistics(self, values, manual_range=None, include_outliers=False, total_count=None):
        values = self._finite_values(values)
        if values.numel() == 0:
            return self._empty_statistics(include_outliers=include_outliers)

        lo, hi = self.calculate_range(values, manual_range)
        if lo is not None and hi is not None:
            values = values[(values >= lo) & (values <= hi)]

        if values.numel() == 0:
            return self._empty_statistics(include_outliers=include_outliers)

        stats = {
            "min": values.min().item(),
            "max": values.max().item(),
            "mean": values.mean().item(),
            "median": torch.median(values).item(),
            "std": values.std().item(),
        }
        if include_outliers:
            if total_count is None:
                total_count = len(values)
            stats["outliers"] = total_count - len(values)
        return stats

    def get_density_statistics(self):
        return self._filtered_statistics(
            self.density,
            None,
            include_outliers=True,
            total_count=len(self.density),
        )

    def get_min_distance_statistics(self):
        return self._filtered_statistics(self.min_distance)

    def get_max_force_statistics(self):
        if not self.has_forces:
            return self._empty_statistics(include_outliers=True)
        return self._filtered_statistics(
            self.max_force,
            None,
            include_outliers=True,
            total_count=len(self.max_force),
        )

    def get_energy_statistics(self):
        if not self.has_energies:
            return self._empty_statistics(include_outliers=True)
        nums = self.unique_species
        if 0 not in nums:
            nums = [0] + nums
        enc = OneHotSpecies(nums)
        sp_t = self._species_tensor.long()
        enc_sp = enc(sp_t)[0].to(torch.float64)
        lin_e = torch.tensordot(enc_sp, self.E0_regression.to(dtype=torch.float64), dims=([2], [0])).sum(dim=1)
        defects = self._energies_tensor.to(dtype=torch.float64) - lin_e
        return self._filtered_statistics(
            defects,
            None,
            include_outliers=True,
            total_count=len(defects),
        )

    # ─── Populate & Save ──────────────────────────────────────────────────────

    def make_json_serializable(self):
        def convert(v):
            if isinstance(v, torch.Tensor):
                return v.tolist()
            if isinstance(v, dict):
                return {k: convert(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [convert(vv) for vv in v]
            return v

        return {k: convert(v) for k, v in self.metadata.items()}

    def save_metadata_to_json(self, filename="metadata.json"):
        """Save metadata to a JSON file.

        :param filename: Output JSON filename
        """
        with open(filename, "w") as f:
            json.dump(self.make_json_serializable(), f, indent=4)

    def save_metadata_to_csv(self, filename="metadata.csv"):
        """Save metadata to a CSV file.

        :param filename: Output CSV filename
        """
        flat = {}

        def _flat(d, prefix=None):
            for k, v in d.items():
                key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, dict):
                    _flat(v, key)
                else:
                    flat[key] = str(v)

        _flat(self.make_json_serializable())
        with open(filename, "w") as f:
            for k, v in flat.items():
                f.write(f"{k},{v}\n")

    def populate_metadata(self, quiet=False):
        self._density = None
        self.metadata["density_statistics"] = self.get_density_statistics()

        if self.has_forces:
            self._max_force = None
            self._min_force = None
            self.metadata["max_force_statistics"] = self.get_max_force_statistics()

        if self.has_energies:
            self._E0_regression = None
            self.metadata["energy_statistics"] = self.get_energy_statistics()

        self.calculate_min_distance()
        self.metadata["min_distance_statistics"] = self.get_min_distance_statistics()

        self._unique_species = None
        self._species_counts = None
        self._symbols_counts = None
        self._species_combination_counts = None
        self._species_combination_index = None
        self._symbols_combination_counts = None
        self.metadata["symbols_counts"] = self.symbols_counts
        self.metadata["symbols_combination_counts"] = self.symbols_combination_counts

        if not quiet:
            print("Metadata populated:")
            for k, v in self.metadata.items():
                print(f"  {k}: {v}")

    # ─── Plotting ──────────────────────────────────────────────────────────────

    def plot_distributions(self, density_range=None, max_force_range=None, min_distance_range=None, bins=None, alpha=None, figsize=(12, 9)):
        """
        Plot distribution histograms for density, max force, min distance, and atom counts.

        :param density_range: manual range for density filtering
        :param max_force_range: manual range for max force filtering
        :param min_distance_range: manual range for min distance filtering
        :param bins: number of bins for histograms
        :param alpha: transparency for plots
        :param figsize: figure size as (width, height) in inches
        """
        import matplotlib.pyplot as plt

        max_len = 10000

        dvals = self._finite_values(self.density)[:max_len]
        if dvals.numel() == 0:
            print("[Warning] No finite values found for densities. Skipping plot.")

        if self.has_forces and self.max_force is not None:
            fvals = self._finite_values(self.max_force)[:max_len]
            if fvals.numel() == 0:
                print("[Warning] No finite values found for max_force. Skipping plot.")
        else:
            fvals = torch.tensor([], dtype=torch.float64)

        mvals = self._finite_values(self.min_distance)[:max_len]
        if mvals.numel() == 0:
            print("[Warning] No finite values found for min_distance. Skipping plot.")

        dr = self.calculate_range(dvals, density_range)
        fr = self.calculate_range(fvals, max_force_range)
        mr = self.calculate_range(mvals, min_distance_range)

        b = bins or self.DEFAULT_BINS
        a = alpha or self.DEFAULT_ALPHA

        fig, axs = plt.subplots(2, 2, figsize=figsize)

        histogram_specs = [
            (axs[0, 0], dvals, "Density Distribution", "Density", dr, "Density: No Data"),
            (axs[0, 1], fvals, "Max Force Distribution", "Force", fr, "Max Force: No Data"),
            (axs[1, 0], mvals, "Min Distance Distribution", "Distance", mr, "Min Distance: No Data"),
        ]
        for ax, values, title, xlabel, axis_range, empty_title in histogram_specs:
            if values.numel():
                ax.hist(values, bins=b, alpha=a)
                ax.set(title=title, xlabel=xlabel, ylabel="Count")
                ax.set_xlim(axis_range)
            else:
                ax.set(title=empty_title)

        counts = self.symbols_counts
        if counts:
            syms, cnts = zip(*counts.items())
            axs[1, 1].bar(syms, cnts, alpha=a)
            axs[1, 1].set(title="Atom Counts by Symbol", xlabel="Element", ylabel="Count")
        else:
            axs[1, 1].set(title="Atom Counts: No Data")

        plt.tight_layout()
        backend = plt.get_backend().lower()
        if "agg" in backend:
            return fig, axs
        plt.show()
        return fig, axs

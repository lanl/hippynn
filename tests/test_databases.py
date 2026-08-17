import pytest
import torch

from hippynn.databases.database import Database
from hippynn.databases import auto_detect_key, database_to_extxyz, load_database, write_extxyz

import os
from pathlib import Path
import pytest

from conftest import ignore_optional_key_warning


@pytest.fixture
def dummy_db() -> Database:
    arr_dict = {"a": torch.randn(5, 1), "b": torch.randn(5, 1)}
    return Database(
        arr_dict=arr_dict,
        inputs=["a"],
        targets=["b"],
        seed=0,
        quiet=True,
    )



def test_align_database(dummy_db: Database) -> None:
    """Test database alignment succeeds with matching sets and fails with mismatches."""
    # Test success case
    dummy_db.align(["a"], ["b"])
    assert dummy_db.inputs == ["a"]
    assert dummy_db.targets == ["b"]

    # Test failure cases
    with pytest.raises(ValueError):
        dummy_db.align(["c"], ["b"])

    with pytest.raises(ValueError):
        dummy_db.align(["a"], ["c"])

# ---------------------------------------------------------------------------
# Skip‑able test for building a MetaDatabase from the ANI‑Aluminum dataset
# ---------------------------------------------------------------------------

# Determine the expected location of the ANI‑Aluminum dataset relative to the
# repository root. The example script uses "../../../datasets/ani-al/data/".
_ANI_AL_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "ani-al" / "data"

@pytest.mark.skipif(
    not _ANI_AL_DATASET.is_dir(),
    reason="ANI‑Aluminum dataset not found; test skipped",
)
def test_metadatabase_ani_aluminum() -> None:
    """Build a :class:`MetaDatabase` from the ANI‑Aluminum dataset.

    The test verifies that the MetaDatabase can be instantiated without
    automatically populating metadata and that basic metadata‑generation methods
    execute without error. It is skipped when the dataset directory is absent.
    """
    from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
    from hippynn.databases.metadatabase import MetaDatabase

    # Load the raw ANI‑Aluminum data. ``allow_unfound=True`` mirrors the example
    # script and permits post‑loading preprocessing.
    raw_db = PyAniDirectoryDB(
        directory=str(_ANI_AL_DATASET),
        seed=123,
        quiet=True,
        allow_unfound=True,
        inputs=None,
        targets=None,
    )

    # Keep only a small random fraction of the data to keep this test fast.
    raw_db.make_random_split("delete", 0.95)
    del raw_db.splits["delete"]

    # Construct the MetaDatabase without auto‑populating metadata to keep the
    # test lightweight.
    # Note: the raw ANI‑Aluminum dataset uses the key "force" (singular) for forces.
    # Provide the correct forces_key to ensure max‑force calculations succeed.
    meta = MetaDatabase(
        arr_dict=raw_db.arr_dict,
        forces_key="force",
        populate_metadata=False,
    )


    # Run a subset of metadata calculations.
    meta.calculate_atom_counts()
    meta.calculate_densities()

    # Basic sanity check – metadata dictionary should exist (may be empty).
    assert isinstance(meta.metadata, dict)

    # ---------------------------------------------------------------------------
    # Additional checks: populate metadata and ensure expected keys are present
    # ---------------------------------------------------------------------------
    # Populate metadata (quiet to avoid noisy prints)
    meta.populate_metadata(update=True, quiet=True)
    
    # Validate density statistics contain reasonable values
    density_stats = meta.metadata["density_statistics"]
    assert density_stats["min"] > 0, "Minimum density should be positive"
    assert density_stats["max"] > density_stats["min"], "Max density should exceed min"
    assert density_stats["mean"] > 0, "Mean density should be positive"
    assert density_stats["median"] > 0, "Median density should be positive"
    
    # Validate force statistics contain reasonable values
    force_stats = meta.metadata["max_force_statistics"]
    assert force_stats["min"] >= 0, "Minimum force magnitude should be non-negative"
    assert force_stats["max"] >= force_stats["min"], "Max force should be >= min"
    assert isinstance(force_stats["mean"], float), "Mean force should be a float"
    
    # Validate atom count is a dictionary with positive values
    atom_count = meta.metadata["atom_count"]
    assert isinstance(atom_count, dict), "Atom count should be a dictionary"
    assert all(count > 0 for count in atom_count.values()), "All atom counts should be positive"
    
    # Validate element combinations
    element_combinations = meta.metadata["element_combinations"]
    assert isinstance(element_combinations, dict), "Element combinations should be a dictionary"
    assert len(element_combinations) > 0, "Should have at least one element combination"

    # Verify that densities were computed and have the correct length
    assert meta.densities is not None
    assert len(meta.densities) == len(raw_db.arr_dict[meta.species_key])

    # ---------------------------------------------------------------------------
    # Plotting: ensure plot_distributions runs without error (using Agg backend)
    # ---------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")  # non‑interactive backend for testing
    # The method should complete without raising an exception
    meta.plot_distributions()
    # Print out some key metadata for inspection
    print("--- MetaDatabase Metadata ---")
    for key, val in meta.metadata.items():
        print(f"{key}: {val}")

# ---------------------------------------------------------------------------
# Additional integrity inspection test
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_metadb():
    """Shared synthetic MetaDatabase for common test scenarios."""
    from hippynn.databases.metadatabase import MetaDatabase

    species = torch.tensor(
        [
            [1, 6, 0],
            [1, 8, 0],
            [6, 8, 0],
        ],
        dtype=torch.int64,
    )
    coordinates = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    forces = torch.tensor(
        [
            [[0.0, 0.0, 0.1], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.2, 0.0], [0.1, 0.0, 0.4], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    energies = torch.tensor([1.0, 1.5, 2.5], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1) * 5.0

    return MetaDatabase(
        arr_dict={
            "species": species,
            "coordinates": coordinates,
            "force": forces,
            "energies": energies,
            "cell": cell,
        },
        forces_key="force",
        cell_key="cell",
        populate_metadata=False,
    )


@ignore_optional_key_warning
def test_metadatabase_core_functionality(synthetic_metadb) -> None:
    """Test statistics, plotting, search, species helpers, and caching behavior."""
    import warnings
    import matplotlib
    import matplotlib.pyplot as plt
    from hippynn.databases.metadatabase import MetaDatabase

    matplotlib.use("Agg")

    # Test all statistics methods return proper float values
    density_stats = synthetic_metadb.get_density_statistics()
    force_stats = synthetic_metadb.get_max_force_statistics()
    distance_stats = synthetic_metadb.get_min_distance_statistics()
    energy_stats = synthetic_metadb.get_energy_statistics()

    for stats in (density_stats, force_stats, distance_stats, energy_stats):
        assert isinstance(stats["median"], float)
        assert isinstance(stats["mean"], float)
        assert isinstance(stats["min"], float)
        assert isinstance(stats["max"], float)

    # Test metadata population preserves computed statistics
    synthetic_metadb.populate_metadata(update=True, quiet=True)

    assert synthetic_metadb.metadata["density_statistics"] == density_stats
    assert synthetic_metadb.metadata["max_force_statistics"] == force_stats
    assert synthetic_metadb.metadata["min_distance_statistics"] == distance_stats
    assert synthetic_metadb.metadata["energy_statistics"] == energy_stats

    # Test plotting works without warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, _ = synthetic_metadb.plot_distributions()
    plt.close(fig)

    assert not any("FigureCanvasAgg is non-interactive" in str(w.message) for w in caught)

    # Test species helpers are independent
    assert synthetic_metadb.extract_unique_numbers_large() == [1, 6, 8]
    assert synthetic_metadb.extract_element_combinations_large(chunk_size=1) == {
        (1, 6): 1,
        (1, 8): 1,
        (6, 8): 1,
    }

    # Test search functionality
    assert synthetic_metadb.search_entries_by_species(["H"], exact_match=False) == [0, 1]
    assert synthetic_metadb.search_entries_by_species(["HO"], exact_match=True) == [1]
    assert synthetic_metadb.search_entries_by_max_force((0.15, 0.35)) == [0, 1]
    assert synthetic_metadb.search_entries_by_distance_range((0.75, 1.0)) == [0, 1]

    # Test validation of search ranges
    with pytest.raises(ValueError, match="min > max"):
        synthetic_metadb.search_entries_by_max_force((1.0, 0.0))

    with pytest.raises(ValueError, match="min > max"):
        synthetic_metadb.search_entries_by_distance_range((1.0, 0.0))

    # Test density calculations with cell volume
    species_dens = torch.tensor([[1, 1]], dtype=torch.int64)
    coordinates_dens = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 2.0

    meta_dens = MetaDatabase(
        arr_dict={
            "species": species_dens,
            "coordinates": coordinates_dens,
            "cell": cell,
        },
        cell_key="cell",
        populate_metadata=False,
    )

    volumes = meta_dens.calculate_volume(coordinates_dens[0], cell=cell[0])
    expected_density = 2.0 * meta_dens.get_mass_from_species(1) / volumes["cell_volume"]

    assert volumes["bounding_box_volume"] == pytest.approx(0.125, abs=1e-8)
    assert volumes["cell_volume"] == pytest.approx(8.0, abs=1e-8)
    assert meta_dens.calculate_densities()[0].item() == pytest.approx(expected_density, rel=1e-6)




@ignore_optional_key_warning
def test_metadatabase_min_distance_behavior() -> None:
    """Test periodic boundaries and fallback behavior for minimum distance calculations."""
    from hippynn.databases.metadatabase import MetaDatabase

    # Test periodic vs open boundary conditions
    species_periodic = torch.tensor([[1, 1]], dtype=torch.int64)
    coordinates_periodic = torch.tensor(
        [[[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 5.0

    meta_periodic = MetaDatabase(
        arr_dict={
            "species": species_periodic,
            "coordinates": coordinates_periodic,
            "cell": cell,
        },
        cell_key="cell",
        pair_dist_hard_max=0.5,
        populate_metadata=False,
    )

    periodic_min_distance = meta_periodic.calculate_min_distance(periodic=True)
    assert periodic_min_distance[0].item() == pytest.approx(0.2, abs=1e-6)

    open_min_distance = meta_periodic.calculate_min_distance(periodic=False)
    assert open_min_distance[0].item() == pytest.approx(0.0, abs=1e-6)

    # Test fallback behavior when cutoff misses some systems
    species_fallback = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=torch.int64,
    )
    coordinates_fallback = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    meta_fallback = MetaDatabase(
        arr_dict={
            "species": species_fallback,
            "coordinates": coordinates_fallback,
        },
        pair_dist_hard_max=1.5,
        populate_metadata=False,
    )

    min_distance = meta_fallback.calculate_min_distance(periodic=False)
    assert min_distance.tolist() == pytest.approx([0.8, 0.8], abs=1e-6)

    assert meta_fallback.search_entries_by_distance_range((0.0, 1.0)) == [0, 1]

    distance_stats = meta_fallback.get_min_distance_statistics()
    assert distance_stats["min"] == pytest.approx(0.8, abs=1e-6)
    assert distance_stats["max"] == pytest.approx(0.8, abs=1e-6)



@ignore_optional_key_warning
def test_metadatabase_validation_and_optional_inputs() -> None:
    """Test shape validation, missing optional inputs, and metadata access patterns."""
    from hippynn.databases.metadatabase import MetaDatabase

    species = torch.tensor(
        [
            [1, 6, 0],
            [1, 8, 0],
        ],
        dtype=torch.int64,
    )
    coordinates = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.2, 0.0], [1.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    # Test malformed forces validation
    bad_forces = torch.zeros((2, 2, 3), dtype=torch.float32)
    with pytest.raises(MetaDatabase.MetaDatabaseError, match="forces must have shape"):
        MetaDatabase(
            arr_dict={
                "species": species,
                "coordinates": coordinates,
                "force": bad_forces,
            },
            forces_key="force",
            populate_metadata=False,
        )

    # Test metadata population with missing optional inputs
    meta = MetaDatabase(
        arr_dict={
            "species": species,
            "coordinates": coordinates,
        },
        metadata={"Comments": "initial"},
        entry_metadata={0: {"split": "train"}},
        populate_metadata=False,
    )

    assert not meta.has_forces
    assert not meta.has_energies
    assert not meta.has_cell

    # Test direct dictionary manipulation
    meta.metadata["Comments"] = "updated"
    meta.metadata["Source"] = "synthetic"
    meta.metadata.pop("Comments")

    meta.entry_metadata[0]["split"] = "valid"
    meta.entry_metadata[1] = {"split": "test"}
    meta.entry_metadata.pop(1)

    assert meta.metadata == {"Source": "synthetic"}
    assert meta.entry_metadata == {0: {"split": "valid"}}

    # Populate and verify optional statistics are skipped
    meta.populate_metadata(update=True, quiet=True)

    assert meta.metadata["Source"] == "synthetic"
    assert "density_statistics" in meta.metadata
    assert "min_distance_statistics" in meta.metadata
    assert "atom_count" in meta.metadata
    assert "element_combinations" in meta.metadata
    assert "max_force_statistics" not in meta.metadata
    assert "energy_statistics" not in meta.metadata

    # Verify missing optional statistics return None
    assert meta.get_max_force_statistics() == {
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "outliers": None,
    }
    assert meta.get_energy_statistics() == {
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "outliers": None,
    }


def test_auto_detect_key():
    """Test auto-detection: standard names, case-insensitive, alternatives, and errors."""
    from hippynn.databases.utils import BUILTIN_AUTO_KEYSETS

    species_keyset = BUILTIN_AUTO_KEYSETS['SPECIES_KEYSET']
    coordinates_keyset = BUILTIN_AUTO_KEYSETS['COORDINATES_KEYSET']

    keys = ['SPECIES', 'coordinates', 'energy']
    assert auto_detect_key(keys, species_keyset) == 'SPECIES'
    assert auto_detect_key(keys, coordinates_keyset) == 'coordinates'

    # Ambiguous keys raise ValueError
    keys_ambig = ['species', 'atomic_numbers', 'coordinates']
    with pytest.raises(ValueError, match="Multiple candidates"):
        auto_detect_key(keys_ambig, species_keyset)

    # Missing required keys raise ValueError
    with pytest.raises(ValueError, match="Could not auto-detect"):
        auto_detect_key(['coordinates'], species_keyset, required=True)

    # Missing optional keys return None with warning
    with pytest.warns(UserWarning, match="not found"):
        result = auto_detect_key(['species'], coordinates_keyset, required=False)
    assert result is None


def test_auto_detect_key_hint():
    # soft matching

    keys = ['atomic_numbers', 'coordinates', 'energy']
    assert auto_detect_key(keys, 'atomic_numbers') == 'atomic_numbers'
    assert auto_detect_key(keys, 'Z') == 'atomic_numbers'
    assert auto_detect_key(keys, 'pos') == 'coordinates'

    # Test that it breaks
    with pytest.raises(ValueError, match="Could not match hint"):
        auto_detect_key(keys, 'not_a_good_hint')


def test_builtin_keysets_have_no_overlap():
    from hippynn.databases.utils import BUILTIN_AUTO_KEYSETS

    seen = {}
    for name, keyset in BUILTIN_AUTO_KEYSETS.items():
        for alias in keyset:
            folded = alias.casefold()
            assert folded not in seen, f"Alias {alias!r} appears in both {seen.get(folded)} and {name}"
            seen[folded] = name


@ignore_optional_key_warning
def test_metadatabase_auto_detection():
    """Test MetaDatabase with auto-detected keys."""
    import numpy as np
    from hippynn.databases.metadatabase import MetaDatabase
    
    arr_dict = {
        'atomic_numbers': np.array([[1, 6, 0], [8, 1, 0]], dtype=np.int32),
        'positions': np.random.rand(2, 3, 3).astype(np.float32),
        'energy': np.array([1.0, 2.0], dtype=np.float32),
    }
    
    # Should auto-detect all keys
    meta = MetaDatabase(arr_dict, populate_metadata=False)
    assert meta.species_key == 'atomic_numbers'
    assert meta.coordinates_key == 'positions'
    assert meta.energies_key == 'energy'
    assert meta.forces_key is None
    assert meta.cell_key is None


@pytest.fixture
def xyz_db() -> Database:
    """A small synthetic Database with coordinates/species/forces/energy, each system with >= 7 real atoms.

    Each system needs at least 7 non-padding atoms because PyAniFileDB's key-structure
    auto-detection (hippynn.databases.h5_pyanitools.PyAniMethods.determine_key_structure)
    requires n_atoms >= 7 to disambiguate padded axes, and write_h5 trims each system's
    species down to its actual (non-padded) atom count before storing.
    """
    species = torch.tensor([[1, 6, 8, 1, 1, 6, 6], [8, 1, 1, 6, 6, 7, 7]], dtype=torch.int64)
    coordinates = torch.zeros((2, 7, 3), dtype=torch.float64)
    forces = torch.zeros((2, 7, 3), dtype=torch.float64)
    energy = torch.tensor([1.0, 2.0], dtype=torch.float64)
    return Database(
        arr_dict={"species": species, "coordinates": coordinates, "forces": forces, "energy": energy},
        inputs=["coordinates", "species"],
        targets=["energy", "forces"],
        seed=0,
        quiet=True,
    )


def test_remove_high_property_auto_detects_species_key(xyz_db: Database) -> None:
    """species_key=None should auto-detect 'species' when atomwise/norm_per_atom needs it."""
    xyz_db.remove_high_property("energy", atomwise=False, norm_per_atom=True, std_factor=5)


def test_calculate_min_dists_auto_detects_keys(xyz_db: Database) -> None:
    from hippynn.pretraining import calculate_min_dists

    calculate_min_dists(xyz_db.arr_dict, dist_hard_max=5.0)


def test_calculate_min_dists_warns_on_unused_cell_key(xyz_db: Database) -> None:
    from hippynn.pretraining import calculate_min_dists

    arr_dict = dict(xyz_db.arr_dict)
    arr_dict["cell"] = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    with pytest.warns(UserWarning, match="cell_name"):
        calculate_min_dists(arr_dict, dist_hard_max=5.0)


def test_write_extxyz(xyz_db: Database, temporary_directory) -> None:
    """Covers basic writing/round-trip, split selection, overwrite guard, and error paths."""
    pytest.importorskip("ase")
    from ase.io import read as ase_read

    out_path = Path(temporary_directory) / "out.extxyz"
    write_extxyz(xyz_db, out_path)
    frames = ase_read(str(out_path), index=":")
    assert len(frames) == 2
    assert len(frames[0]) == 7
    assert len(frames[1]) == 7
    assert frames[0].get_potential_energy() == pytest.approx(1.0)

    # overwrite guard
    with pytest.raises(FileExistsError):
        write_extxyz(xyz_db, out_path, overwrite=False)
    write_extxyz(xyz_db, out_path, overwrite=True)  # succeeds

    # named-split selection
    xyz_db.make_explicit_split("only", xyz_db.arr_dict["indices"][:1])
    out_split = Path(temporary_directory) / "split.extxyz"
    write_extxyz(xyz_db, out_split, split="only")
    assert len(ase_read(str(out_split), index=":")) == 1

    # invalid split raises
    with pytest.raises(ValueError, match="split must be"):
        write_extxyz(xyz_db, Path(temporary_directory) / "bad.extxyz", split="nope")

    # invalid pbc raises
    with pytest.raises(ValueError, match="pbc must be"):
        write_extxyz(xyz_db, Path(temporary_directory) / "badpbc.extxyz", pbc=(True, False))


def test_load_database(xyz_db: Database, temporary_directory) -> None:
    """Backend is chosen based on `data_file`: by file extension for a file, by directory contents for a
    directory. Unknown/unsupported inputs raise an informative error."""
    import numpy as np

    from hippynn.databases.ondisk import NPZDatabase, DirectoryDatabase

    xyz_db.split_the_rest("all")

    def check(db, energies_key, expected_class, label):
        assert isinstance(db, expected_class), f"{label}: expected {expected_class.__name__}, got {type(db).__name__}"
        assert energies_key == "energy", f"{label}: expected energies_key 'energy', got {energies_key!r}"
        assert db.arr_dict["energy"].shape == (2,), f"{label}: unexpected arr_dict['energy'] shape {db.arr_dict['energy'].shape}"

    # .npz file -> NPZDatabase
    npz_path = Path(temporary_directory) / "data.npz"
    xyz_db.write_npz(str(npz_path), record_split_masks=False)
    check(*load_database(npz_path), NPZDatabase, "npz file")

    # .h5 file -> PyAniFileDB
    h5py = pytest.importorskip("h5py")
    from hippynn.databases.h5_pyanitools import PyAniFileDB, PyAniDirectoryDB

    h5_path = Path(temporary_directory) / "data.h5"
    xyz_db.write_h5(split=True, h5path=str(h5_path), overwrite=True)
    check(*load_database(h5_path), PyAniFileDB, "h5 file")

    # directory of .h5 files -> PyAniDirectoryDB
    h5_dir = Path(temporary_directory) / "h5_dir"
    h5_dir.mkdir()
    xyz_db.write_h5(split=True, h5path=str(h5_dir / "data.h5"), overwrite=True)
    check(*load_database(h5_dir), PyAniDirectoryDB, "h5 directory")

    # directory of .npy files -> DirectoryDatabase (requires `name`)
    npy_dir = Path(temporary_directory) / "npy_dir"
    npy_dir.mkdir()
    for key, arr in xyz_db.splits["all"].items():
        np.save(npy_dir / f"prefix_{key}.npy", arr.detach().cpu().numpy() if hasattr(arr, "detach") else arr)

    with pytest.raises(ValueError, match="requires `name`"):
        load_database(npy_dir)
    check(*load_database(npy_dir, name="prefix_"), DirectoryDatabase, "npy directory")

    # unrecognized file extension raises an informative error
    bad_path = Path(temporary_directory) / "data.txt"
    bad_path.write_text("not a database")
    with pytest.raises(ValueError, match="Unrecognized dataset file extension"):
        load_database(bad_path)


def test_load_database_custom_keys(temporary_directory) -> None:
    """Custom species/coordinates/energies/forces key names are honored, not just the defaults."""
    from hippynn.databases.ondisk import NPZDatabase

    species = torch.tensor([[1, 6, 8, 1, 1, 6, 6], [8, 1, 1, 6, 6, 7, 7]], dtype=torch.int64)
    coordinates = torch.zeros((2, 7, 3), dtype=torch.float64)
    forces = torch.zeros((2, 7, 3), dtype=torch.float64)
    energy = torch.tensor([1.0, 2.0], dtype=torch.float64)
    custom_db = Database(
        arr_dict={
            "atomic_numbers": species,
            "positions": coordinates,
            "custom_force": forces,
            "custom_energy": energy,
        },
        inputs=["positions", "atomic_numbers"],
        targets=["custom_energy", "custom_force"],
        seed=0,
        quiet=True,
    )
    custom_db.split_the_rest("all")

    npz_path = Path(temporary_directory) / "custom.npz"
    custom_db.write_npz(str(npz_path), record_split_masks=False)
    db, energies_key = load_database(
        npz_path,
        species_key="atomic_numbers",
        coordinates_key="positions",
        energies_key="custom_energy",
        forces_key="custom_force",
    )
    assert isinstance(db, NPZDatabase)
    assert energies_key == "custom_energy"
    assert set(db.inputs) == {"positions", "atomic_numbers"}
    assert set(db.targets) == {"custom_energy", "custom_force"}
    assert db.arr_dict["custom_energy"].shape == (2,)


def test_database_to_extxyz(xyz_db: Database, temporary_directory) -> None:
    """End-to-end wrapper: load a database from file and write it to EXTXYZ."""
    pytest.importorskip("ase")
    from ase.io import read as ase_read

    xyz_db.split_the_rest("all")

    npz_path = Path(temporary_directory) / "data.npz"
    xyz_db.write_npz(str(npz_path), record_split_masks=False)

    database_to_extxyz(npz_path)  # default output filename derived from input basename
    out_path = Path(temporary_directory) / "data.extxyz"
    assert len(ase_read(str(out_path), index=":")) == 2


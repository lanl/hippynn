import os
import argparse
import torch

# Dataset loaders
from hippynn.databases.h5_pyanitools import PyAniFileDB
from hippynn.databases import NPZDatabase
from hippynn.databases.metadatabase import MetaDatabase

# Read dataset filename and database keys from command line
parser = argparse.ArgumentParser(
    description='Load a dataset and compute metadata statistics using MetaDatabase.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
default_dataset_path = os.path.join(
    os.path.dirname(__file__), '../../datasets/ani1x_release/ani1x-release.h5'
)
parser.add_argument(
    'dataset_path',
    nargs='?',
    default=default_dataset_path,
    help='Path to the dataset file (.h5, .hdf5, or .npz)'
)
parser.add_argument(
    '--species-key',
    default='atomic_numbers',
    help='Key name for species/atomic numbers in the dataset'
)
parser.add_argument(
    '--coordinates-key',
    default='coordinates',
    help='Key name for atomic coordinates in the dataset'
)
parser.add_argument(
    '--energy-key',
    default='wb97x_dz.energy',
    help='Key name for energies in the dataset'
)
parser.add_argument(
    '--forces-key',
    default='wb97x_dz.forces',
    help='Key name for forces in the dataset'
)
args = parser.parse_args()

DATA_FILE = os.path.expanduser(args.dataset_path)
filetype = os.path.splitext(DATA_FILE)[1].lower()

# ANI specific helpers
# ---------------------------------------------------------------------------

AVAIL_METHODS = ['hf', 'wb97x', 'ccsd(t)', 'mp2']
AVAIL_BASIS   = ['dz', 'tz', 'qz', 'cbs']
ANI1X_DSETS_KEYS = [
    'hf_tz.energy', 'coordinates', 'tpno_ccsd(t)_dz.corr_energy', 'wb97x_dz.hirshfeld_charges', 
    'wb97x_tz.mbis_charges', 'wb97x_tz.forces', 'mp2_tz.corr_energy', 'npno_ccsd(t)_tz.corr_energy', 
    'wb97x_tz.mbis_volumes', 'wb97x_tz.energy', 'wb97x_tz.dipole', 'wb97x_tz.mbis_octupoles', 
    'wb97x_tz.mbis_quadrupoles', 'mp2_qz.corr_energy', 'wb97x_tz.mbis_dipoles', 'wb97x_dz.cm5_charges', 
    'path', 'atomic_numbers', 'hf_qz.energy', 'mp2_dz.corr_energy', 'wb97x_dz.dipole', 
    'npno_ccsd(t)_dz.corr_energy', 'wb97x_dz.energy', 'hf_dz.energy', 'wb97x_dz.quadrupole', 
    'ccsd(t)_cbs.energy', 'wb97x_dz.forces'
]

def load_db(db_info, en_name, force_name, seed, location, n_workers, species_key):
    torch.set_default_dtype(torch.float64)
    return PyAniFileDB(
        file=location, species_key=species_key, seed=seed, num_workers=n_workers, 
        allow_unfound=True, 
        **db_info
    )

def get_data_names(qm_method, basis_set, force_training=False):
    assert qm_method in AVAIL_METHODS, f"Method not found: {qm_method}"
    assert basis_set in AVAIL_BASIS, f"Basis set not found: {basis_set}"
    spec = f"{qm_method}_{basis_set}"
    en_name = f"{spec}.energy"
    assert en_name in ANI1X_DSETS_KEYS, f"Data spec not available: {spec}"
    if force_training:
        assert f"{spec}.forces" in ANI1X_DSETS_KEYS, f"No force training for: {spec}"
    return en_name, f"{spec}.forces"

force_training = False
qm_method, basis_set = 'wb97x', 'dz'
en_name, force_name = get_data_names(qm_method, basis_set, force_training)

# ---------------------------------------------------------------------------

# Dataset-specific loaders
# ---------------------------------------------------------------------------

# Define base_database by the file extension
if filetype == ".npz":
    inputs  = [args.coordinates_key, args.species_key]
    targets = [args.energy_key, args.forces_key]
    base_database = NPZDatabase(
        file=DATA_FILE,
        seed=101,
        allow_unfound=True,
        inputs=inputs,
        targets=targets,
        quiet=False
    )

elif filetype in (".h5", ".hdf5"):
    inputs  = [args.coordinates_key, args.species_key]
    targets = [args.energy_key, args.forces_key]
    db_info = {"inputs": inputs , "targets": targets}
    base_database = load_db(
        db_info,
        en_name,
        force_name,
        seed=101,
        location=DATA_FILE,
        n_workers=2,
        species_key=args.species_key
    )

else:
    raise ValueError(f"Unrecognized dataset file extension: {filetype}. Supported file extensions are: .h5, .hdf5, .npz.")

# ---------------------------------------------------------------------------
print("Database loaded. Constructing MetaDatabase.")

# Build MetaDatabase using the selected base_database; compute metadata statistics
meta_database = MetaDatabase(
    arr_dict=base_database.arr_dict,
    species_key=args.species_key,
    coordinates_key=args.coordinates_key,
    energies_key=args.energy_key,
    forces_key=args.forces_key,
    metadata={ 
        "Energy_unit" : 'eV',
        "Mass_unit" : 'grams/mol', 
        "Distance_unit" : 'Angstroms',
        "Electronic_Structure_Package" : '',
        "Electronic_Structure_Package_Version" : '',
        "Computer_System" : '',
        "Input_Proceedure" : '' 
    },
    populate_metadata=True,
)

# Save metadata to files
meta_database.save_metadata_to_json('TEST_metadata.json')
meta_database.save_metadata_to_csv('TEST_metadata.csv')

# Plot metadata statistics
meta_database.plot_distributions()

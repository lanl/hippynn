import os
import argparse
import torch

# Dataset loaders
from hippynn.databases.utils import load_base_database
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

torch.set_default_dtype(torch.float64)
base_database, _energies_key = load_base_database(
    DATA_FILE,
    seed=101,
    num_workers=2,
    species_key=args.species_key,
    coordinates_key=args.coordinates_key,
    energies_key=args.energy_key,
    forces_key=args.forces_key,
)

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

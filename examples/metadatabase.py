import os
import sys
import torch

# Dataset loaders
from hippynn.databases.h5_pyanitools import PyAniFileDB
from hippynn.databases import NPZDatabase
from hippynn.databases.metadatabase import MetaDatabase

# Read dataset filename from command line; determine file type
if len(sys.argv) < 2:
    print(f"Usage: python {os.path.basename(__file__)} [dataset_file_name]")
    raise SystemExit(1)

DATA_FILE = os.path.expanduser(sys.argv[1])
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

def load_db(db_info, en_name, force_name, seed, location, n_workers):
    torch.set_default_dtype(torch.float64)
    return PyAniFileDB(
        file=location, species_key='species', seed=seed, num_workers=n_workers, 
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
    inputs  = ['coordinates', 'species']
    targets = ['energy', 'forces']
    energies_key_sel = 'energy'
    base_database = NPZDatabase(
        file=DATA_FILE,
        seed=101,
        allow_unfound=True,
        inputs=inputs,
        targets=targets,
        quiet=False
    )

elif filetype in (".h5", ".hdf5"):
    inputs  = ['coordinates', 'species']
    targets = ['energies', 'forces']
    db_info = {"inputs": inputs , "targets": targets}
    energies_key_sel = 'energies'
    base_database = load_db(
        db_info,
        en_name,
        force_name,
        seed=101,
        location=DATA_FILE,
        n_workers=2
    )

else:
    raise ValueError(f"Unrecognized dataset file extension: {filetype}. Supported file extensions are: .h5, .hdf5, .npz.")

# ---------------------------------------------------------------------------

# Build MetaDatabase using the selected base_database; plot metadata statistics
meta_database = MetaDatabase(
    arr_dict=base_database.arr_dict,
    inputs=inputs,
    targets=targets,
    seed=12345,
    num_workers=1,
    pin_memory=True,
    allow_unfound=True,
    quiet=True,        
    species_key='species',
    coordinates_key='coordinates',
    energies_key=energies_key_sel,
    forces_key='forces',
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
    write_metadata_to_json=True,
    json_filename='metadata.json',
    distribution_plots=True,
)

# Plot metadata statistics
#meta_database.plot_distributions()

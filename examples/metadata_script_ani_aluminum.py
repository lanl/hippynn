"""
Metadata Script for ANI-Aluminum Dataset
=========================================

This script demonstrates how to use :class:`hippynn.databases.metadatabase.MetaDatabase`
to compute and display metadata for the ANI-Aluminum dataset. It accepts an optional
command line argument for the dataset path.

Usage
-----
``python examples/metadata_script_ani_aluminum.py [dataset_path]``

If no path is provided, the script will attempt to use the default ANI-Aluminum
dataset location: ``../../../datasets/ani-al/data/``

The script will write ``TEST_metadata.json`` and ``TEST_metadata.csv`` files containing 
the collected metadata and will generate Matplotlib figures showing the distributions 
of energies, forces, densities, and atom counts.
"""

import sys
import argparse
import torch
from pathlib import Path

# Dataset loaders
from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
from hippynn.databases.metadatabase import MetaDatabase


def main():
    # Determine the default location of the ANI-Aluminum dataset
    # The example uses: ../../../datasets/ani-al/data/
    default_dataset_path = (
        Path(__file__).resolve().parents[2] / "datasets" / "ani-al" / "data"
    )

    parser = argparse.ArgumentParser(
        description="Compute and display metadata for the ANI-Aluminum dataset using MetaDatabase."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=str(default_dataset_path) if default_dataset_path.exists() else None,
        type=str,
        help=(
            "Path to the ANI-Aluminum dataset directory containing .h5 files. "
            f"If omitted, will attempt to use: {default_dataset_path}"
        ),
    )
    args = parser.parse_args()

    if args.dataset_path is None:
        print(f"Error: No dataset path provided and default path not found: {default_dataset_path}")
        print("\nUsage: python metadata_script_ani_aluminum.py [dataset_path]")
        sys.exit(1)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_dir():
        print(f"Error: Dataset path does not exist or is not a directory: {dataset_path}")
        sys.exit(1)

    print(f"Loading ANI-Aluminum dataset from: {dataset_path}")

    # Load the raw ANI-Aluminum data using PyAniDirectoryDB.
    base_db = PyAniDirectoryDB(
        directory=str(dataset_path),
        seed=101,
        allow_unfound=True,
        inputs=None,
        targets=None,
    )

    print(f"Loaded {len(base_db.arr_dict[base_db.species_key])} structures")

    # Build MetaDatabase – this will compute statistics automatically
    # Note: keys are auto-detected from arr_dict if not explicitly provided
    meta_db = MetaDatabase(
        arr_dict=base_db.arr_dict,
        metadata={
            "Energy_unit": "eV",
            "Mass_unit": "grams/mol",
            "Distance_unit": "Angstrom",
            "Dataset": "ANI-Aluminum",
        },
        populate_metadata=True,
    )

    # Save metadata to files
    print("\nSaving metadata to files...")
    meta_db.save_metadata_to_json("TEST_metadata.json")
    meta_db.save_metadata_to_csv("TEST_metadata.csv")
    print("  - TEST_metadata.json")
    print("  - TEST_metadata.csv")

    # Generate distribution plots
    print("\nGenerating distribution plots...")
    meta_db.plot_distributions()
    print("  - Distribution plots displayed/saved")

    # Print the collected metadata to stdout for quick inspection
    print("\n" + "="*60)
    print("COMPUTED METADATA")
    print("="*60)
    for key, value in meta_db.metadata.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("="*60)


if __name__ == "__main__":
    # Ensure a deterministic dtype for the base database loading
    torch.set_default_dtype(torch.float32)
    main()

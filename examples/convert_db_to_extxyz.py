import os
import sys
from pathlib import Path

from hippynn.databases import load_database, write_extxyz

def main():
    if len(sys.argv) < 2:
        script = os.path.basename(__file__)
        print(f"Usage: python {script} [database file name] [optional: output file name]")
        sys.exit(1)

    data_file = os.path.expanduser(sys.argv[1])
    output_xyz = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else Path(os.path.splitext(os.path.basename(data_file))[0] + ".extxyz")
    )

    db, _energies_key = load_database(data_file)
    write_extxyz(db, output_xyz, overwrite=True, pbc=(False, False, False))

    # Alternative one-liner:
    # from hippynn.databases import database_to_extxyz
    # database_to_extxyz(data_file, output_xyz, overwrite=True, pbc=(False, False, False))

if __name__ == "__main__":
    main()
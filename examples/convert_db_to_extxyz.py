"""
Convert a Database to EXTXYZ
=============================

Loads a database supported by :func:`hippynn.databases.load_database`
(``.npz`` file, ``.h5``/``.hdf5`` file, or a directory of ``.h5``/``.npy`` files)
and writes it out as an EXTXYZ file using :func:`hippynn.databases.write_extxyz`.
"""

import argparse
from pathlib import Path

from hippynn.databases import load_database, write_extxyz


def main(args):
    data_file = Path(args.data_file).expanduser()
    if not data_file.exists():
        raise FileNotFoundError(f"Input database not found: {data_file}")

    output_file = Path(args.output_file) if args.output_file else data_file.with_suffix(".extxyz")

    db = load_database(data_file)
    write_extxyz(db, output_file, overwrite=args.overwrite, pbc=args.pbc, split=args.split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a hippynn database to EXTXYZ format.")
    parser.add_argument("data_file", type=str, help="Path to input database (.npz, .h5/.hdf5, or a directory).")
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default=None,
        help="Output .extxyz path. Defaults to the input filename with its suffix replaced by '.extxyz'.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--pbc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether structures are periodic in all three directions.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default=None,
        help="Restrict output to one data split. Omit to write the full database as-is.",
    )
    args = parser.parse_args()

    main(args)
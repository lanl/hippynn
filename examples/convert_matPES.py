"""
This script uses the MatPES dataset introduced in:
"A Foundational Potential Energy Surface Dataset for Materials"
(arXiv:2503.04070, https://arxiv.org/abs/2503.04070).

For more details and raw files see https://matpes.ai/ .

"""
import os.path
import numpy as np
import gzip
import json
from ase.data import atomic_numbers
import argparse
from tqdm.auto import tqdm


def unpack(records, split):
    N = len(records)  # number of structures
    counts = [len(rec['structure']['sites']) for rec in records] # atoms per structure
    M = max(counts)  # max number of atoms

    # allocate data arrays
    Z = np.zeros((N, M), dtype=np.int64)
    R = np.zeros((N, M, 3), dtype=np.float64)
    F = np.zeros((N, M, 3), dtype=np.float64)
    E = np.zeros((N, 1), dtype=np.float64)
    stress = np.zeros((N, 3, 3), dtype=np.float64)
    C = np.zeros((N, 3, 3), dtype=np.float64)
    t_peratom = np.full((N, 1), np.nan, dtype=np.float64)
    T = np.full((N, 1), np.nan, dtype=np.float64)
    magmoms = np.zeros((N, M), dtype=np.float64)
    total_charge = np.zeros((N, 1), dtype=np.float64)


    # boolean masks for splits
    train_mask = np.zeros(N, dtype=bool)
    valid_mask = np.zeros(N, dtype=bool)
    test_mask  = np.zeros(N, dtype=bool)
    for idx in split.get('train', []):
        train_mask[idx] = True
    for idx in split.get('valid', []):
        valid_mask[idx] = True
    for idx in split.get('test', []):
        test_mask[idx] = True


    for i, rec in enumerate(tqdm(records, unit="configs")):
        # cell
        C[i] = np.array(rec['structure']['lattice']['matrix'], dtype=np.float64)

        # stress: Voigt → tensor
        s = rec['stress']
        stress[i] = np.array([
            [s[0], s[5], s[4]],
            [s[5], s[1], s[3]],
            [s[4], s[3], s[2]]
        ], dtype=np.float64)

        # energies & charge
        E[i, 0] = rec['energy']
        n_atoms = rec['nsites']
        ce = rec.get('cohesive_energy_per_atom')
        ce_extensive = ce*n_atoms
        if ce is not None:
            t_peratom[i, 0] = ce
            T[i, 0] = ce_extensive
        total_charge[i, 0] = rec['structure'].get('charge', 0.0)

        # sites
        for j, site in enumerate(rec['structure']['sites']):
            sym = site['species'][0]['element']
            Z[i, j] = atomic_numbers[sym]
            R[i, j] = site['xyz']
            F[i, j] = rec['forces'][j]
            mag = site.get('properties', {}).get('magmom')
            if mag is not None:
                magmoms[i, j] = mag

    return {
        'Z': Z,
        'R': R,
        'E': E,
        'F': F,
        'stress': stress, 'C': C,
        't_peratom': t_peratom,
        'T': T,
        'magmoms': magmoms,
        'total_charge': total_charge,
        'train': train_mask,
        'valid': valid_mask,
        'test': test_mask
    }

def main(args):

    # First validate filenames.
    split_fname = os.path.abspath(args.split_fname)
    data_fname = os.path.abspath(args.data_fname)
    if not os.path.exists(split_fname):
        raise FileNotFoundError(f"Did not find matPES split file at {split_fname}.")
    if not os.path.exists(data_fname):
        raise FileNotFoundError(f"Did not find matPES config file at {data_fname}.")

    output_fname = args.output_fname
    output_fname = os.path.abspath(output_fname)
    if os.path.exists(output_fname) and not args.overwrite:
      raise FileExistsError(f"File {output_fname} exists; pass --overwrite to overwrite.")

    # Now process data.
    print("Loading data split")
    with gzip.open(args.split_fname, 'rt', encoding='utf-8') as f:
        split_data = json.load(f)

    all_ind = list(sorted(split_data['valid'] + split_data['train'] + split_data['test']))
    all_indices = not (all_ind - np.arange(len(all_ind))).max()
    print("All indices found in a some split:", all_indices)

    print("Loading data file.")
    with gzip.open(args.data_fname, 'rt', encoding='utf-8') as f:
        config_data = json.load(f)

    print("Assembling arrays.")
    array_data = unpack(config_data, split_data)

    if os.path.exists(output_fname) and not args.overwrite:
        raise FileExistsError(f"File {output_fname} exists; pass --overwrite to overwrite.")

    print("Writing data.")
    np.savez_compressed("matPES_R2SCAN.npz", **array_data)
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from argparse import BooleanOptionalAction

    parser.add_argument("--data_fname", type=str, default="../../datasets/matPES/MatPES-R2SCAN-2025.1.json.gz")
    parser.add_argument("--split_fname", type=str, default="../../datasets/matPES/MatPES-R2SCAN-split.json.gz")
    parser.add_argument("--output_fname", type=str, default="../../datasets/matPES/matPES_R2SCAN.npz")
    parser.add_argument("--overwrite", action=BooleanOptionalAction, default=False,
                        help="Whether to overwrite an existing copy of the exported database.")

    args = parser.parse_args()

    main(args)


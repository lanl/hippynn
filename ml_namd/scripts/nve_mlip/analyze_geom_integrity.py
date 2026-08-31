#!/usr/bin/env python3
"""Scan a multi-frame .xyz MD trajectory for fragmentation / collision / bond distortion.

For each frame: pairwise distances -> report the smallest (collision), the largest
nearest-neighbor distance (an atom detaching = fragmentation), and the molecular
extent. Also tracks per-frame nearest-neighbor bond stats vs frame 0 so rearrangement
(a bond that stretches far or a new close contact) shows up. numpy only.
"""
import sys
import numpy as np

# acn_Z order: O O C C C C C H H H H H H H H
COV = {"H": 0.31, "C": 0.76, "O": 0.66}  # covalent radii (A); bond if d < 1.3*(ri+rj)


def read_xyz_frames(path):
    L = open(path).read().splitlines()
    frames, syms, i = [], None, 0
    while i < len(L) and L[i].strip():
        n = int(L[i].split()[0]); i += 2
        blk = [L[i + j].split() for j in range(n)]
        if syms is None:
            syms = [b[0] for b in blk]
        frames.append(np.array([[float(x) for x in b[1:4]] for b in blk]))
        i += n
    return syms, np.array(frames)


def main():
    path = sys.argv[1]
    syms, F = read_xyz_frames(path)
    nf, n, _ = F.shape
    # frame-0 bond list (covalent-radius criterion)
    d0 = np.linalg.norm(F[0][:, None] - F[0][None], axis=-1)
    bonds = [(a, b) for a in range(n) for b in range(a + 1, n)
             if d0[a, b] < 1.3 * (COV[syms[a]] + COV[syms[b]])]
    mind, maxnn, extent = [], [], []
    bond_max = np.zeros(len(bonds)); bond_min = np.full(len(bonds), 1e9)
    for fr in F:
        d = np.linalg.norm(fr[:, None] - fr[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        mind.append(d.min())
        maxnn.append(d.min(axis=1).max())
        dd = d.copy(); dd[np.isinf(dd)] = 0.0
        extent.append(dd.max())
        for k, (a, b) in enumerate(bonds):
            v = np.linalg.norm(fr[a] - fr[b])
            bond_max[k] = max(bond_max[k], v); bond_min[k] = min(bond_min[k], v)
    mind = np.array(mind); maxnn = np.array(maxnn); extent = np.array(extent)
    print(f"== {path}  ({nf} frames, {n} atoms, {len(bonds)} frame-0 bonds) ==")
    print(f"  min pairwise dist   : {mind.min():.3f} A  (collision if <~0.8)   worst frame {int(mind.argmin())}")
    print(f"  max nearest-neighbor: {maxnn.max():.3f} A  (fragment if >~1.9)   worst frame {int(maxnn.argmax())}")
    print(f"  molecular extent    : start {extent[0]:.2f} -> max {extent.max():.2f} A")
    # biggest bond stretch relative to frame 0
    worst = np.argsort(bond_max - d0[tuple(np.array(bonds).T)])[::-1][:3] if bonds else []
    for k in worst:
        a, b = bonds[k]
        print(f"  bond {syms[a]}{a}-{syms[b]}{b}: d0={d0[a,b]:.3f}  range[{bond_min[k]:.3f},{bond_max[k]:.3f}] A")
    frag = maxnn.max() > 1.9; coll = mind.min() < 0.8
    print(f"  VERDICT: {'FRAGMENTED/DETACHED' if frag else 'connected'}; "
          f"{'COLLISION' if coll else 'no collision'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 5: guarded harvest of the 5-seed ensemble -> mean +/- sigma_ML populations.

Runs the fragmentation-guarded harvester (harvest_topo.py) on each seed's trajectories, then reports the
final survivor-conditioned classical populations as mean +/- sigma_ML (sample std, ddof=1, across the 5
split seeds), plus the original-N normalization (survivors / launched). Compare against the FSSH@PYSEQM
reference harvested the same way.

Usage: python 05_harvest.py [--head both] [--maxdist 15.0] [--fragfac 1.8]
"""
import argparse
import subprocess
import sys

import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="both", choices=C.HEADS)
    ap.add_argument("--maxdist", default="15.0")
    ap.add_argument("--fragfac", default="1.8")
    a = ap.parse_args()

    harv_dir = C.OUT / "harvest"
    harv_dir.mkdir(parents=True, exist_ok=True)
    try:
        n_launched = int(np.load(C.ICS / f"{C.IC_PREFIX}_R0.npy", mmap_mode="r").shape[0])
    except OSError:
        n_launched = 0

    npzs = []
    for s in C.SEEDS:
        traj_glob = str(C.OUT / f"{C.TAUT}_{a.head}_seed{s}" / "*.h5")
        npz = harv_dir / f"topo_{C.TAUT}_{a.head}_seed{s}.npz"
        subprocess.run([sys.executable, str(C.ANALYSIS / "harvest_topo.py"),
                        traj_glob, f"{C.TAUT}_{a.head}_seed{s}", str(npz),
                        a.maxdist, a.fragfac], check=True)
        if npz.exists():
            npzs.append(npz)

    if not npzs:
        sys.exit("[05] no harvested .npz found -- run step 4 first")

    surv = {s: [] for s in (1, 2, 3)}
    orig_s1 = []
    for npz in npzs:
        d = np.load(npz)
        cd = d["clas_abcd"]
        n_abcd = int(d["N_abcd"])
        for s in (1, 2, 3):
            surv[s].append(float(cd[-1, s]))
        if n_launched:
            orig_s1.append(float(cd[-1, 1]) * n_abcd / n_launched)

    print(f"\n==== {C.TAUT} head={a.head}: {len(npzs)}-seed ML-TSH populations ====")
    print("  survivor-conditioned (authoritative):")
    for s in (1, 2, 3):
        v = np.array(surv[s])
        per = ", ".join(f"{x:.3f}" for x in v)
        print(f"    S{s}: mean {v.mean():.3f} +/- {v.std(ddof=1):.3f}   (per-seed: {per})")
    if orig_s1:
        o = np.array(orig_s1)
        print(f"  original-N (survivors/launched={n_launched}) S1: mean {o.mean():.3f} +/- {o.std(ddof=1):.3f}")
    print("  sigma_ML = sample std (ddof=1) across the 5 split seeds.")
    print("  Compare with the FSSH@PYSEQM reference harvested the same way (see ../../docs/fssh_pyseqm_guide.md).")


if __name__ == "__main__":
    main()

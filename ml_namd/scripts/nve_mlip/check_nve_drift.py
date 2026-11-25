#!/usr/bin/env python3
"""
Report NVE energy conservation for a PySEQM MD run driven by HIPPYNN.

Reads the HDF5 output (`{prefix}.0.h5`), NOT the screen log. The screen thermo
line is printed with "%e" -- 7 significant figures -- so around -1.39e3 eV it
quantizes Etot to ~1 meV, which is the same order as the 15 meV pass threshold.
The HDF5 stores `data/thermo/{Ep,Ek,T}` as float64.

    Etot(t) = Ep(t) + Ek(t)      Ep == molecule.Etot == sE_n + energy_offset

PASS (Phase 4): max |Etot(t) - Etot(0)| < 1 meV/atom (= 15 meV for this molecule),
with no monotonic trend. Drift rate also reported in meV/ps.

Developer Information
---------------------
  __developer__: "Ajay Khanna"
  __place__: "LANL/Sergei Lab"
  __date__: "Sep-01-2026"
  __project__: "ml_namd"
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import argparse
import sys

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5file")
    ap.add_argument("--natoms", type=int, default=15)
    ap.add_argument("--timestep", type=float, default=0.5, help="fs")
    ap.add_argument("--tol-mev-per-atom", type=float, default=1.0)
    ap.add_argument("--expect-ep0", type=float, default=None,
                    help="ground-truth Ep at step 0 (eV); reports the deviation")
    a = ap.parse_args()

    with h5py.File(a.h5file, "r") as h:
        steps = np.asarray(h["data/steps"])
        Ep = np.asarray(h["data/thermo/Ep"])
        Ek = np.asarray(h["data/thermo/Ek"])
        T = np.asarray(h["data/thermo/T"])

    if Ep.ndim > 1:      # (nsteps, nmol) -> molecule 0
        Ep, Ek, T = Ep[:, 0], Ek[:, 0], T[:, 0]
    if len(steps) < 5:
        sys.exit(f"FATAL: only {len(steps)} thermo points in {a.h5file}")

    Etot = Ep + Ek
    dev_mev = (Etot - Etot[0]) * 1e3
    max_abs = float(np.abs(dev_mev).max())
    per_atom = max_abs / a.natoms

    t_ps = (steps - steps[0]) * a.timestep * 1e-3
    slope = float(np.polyfit(t_ps, dev_mev, 1)[0]) if t_ps[-1] > 0 else float("nan")

    print(f"file            : {a.h5file}")
    print(f"thermo points   : {len(steps)}  (step {steps[0]} -> {steps[-1]})")
    print(f"simulated time  : {t_ps[-1]*1e3:.1f} fs ({t_ps[-1]:.4f} ps)")
    print(f"T mean/min/max  : {T.mean():.2f} / {T.min():.2f} / {T.max():.2f} K")
    print(f"Ep(0)           : {Ep[0]:.9f} eV")
    print(f"Ek(0)           : {Ek[0]:.9f} eV")
    print(f"Etot(0)         : {Etot[0]:.9f} eV")
    if a.expect_ep0 is not None:
        d = (Ep[0] - a.expect_ep0) * 1e3
        print(f"Ep(0) vs ground truth {a.expect_ep0:.9f} eV : {d:+.4f} meV")
    print()
    print(f"max |dEtot|     : {max_abs:.4f} meV   ({per_atom:.5f} meV/atom)")
    print(f"drift rate      : {slope:+.4f} meV/ps  ({slope/a.natoms:+.5f} meV/ps/atom)")
    print(f"std(dEtot)      : {dev_mev.std():.4f} meV")
    print(f"final dEtot     : {dev_mev[-1]:+.4f} meV")
    print()
    ok = per_atom < a.tol_mev_per_atom
    print(f"PASS criterion  : max |dEtot| < {a.tol_mev_per_atom} meV/atom "
          f"(= {a.tol_mev_per_atom*a.natoms:.1f} meV)")
    print(f"RESULT          : {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

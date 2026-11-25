#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SURVIVORS-ONLY ensemble average with a COMMON fixed N = min survivor count across sources.
Survivor = alive at the final frame (never terminated). No half-survived trajectories, no S0 recovery.
Pass 1: count survivors per source. N = min. Pass 2: use the FIRST N survivors of each; denominator = N (constant,
all frames alive). Classical = active-surface fraction; quantum = mean excited |c|^2. S0 column kept but not used in plots.
Usage: python survivors_ensemble.py <pspf> <glob1> <out1> <glob2> <out2> ...

Inputs (positional argv): picoseconds-per-frame, then (glob, out.npz) pairs, one per source.
Output: one survivors-only ensemble .npz per source (classical/quantum populations, N used).
Kept as a top-level script (no main()) for md5 stability; this pass is documentation only.
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import sys, glob, numpy as np, h5py
PSPF = float(sys.argv[1]); rest = sys.argv[2:]
pairs = [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
QTOL = 0.1

def survivor_files(globpat):
    """Return the h5 files under ``globpat`` whose final frame is still alive (survivors)."""
    fs = sorted(f for f in glob.glob(globpat) if f.endswith(".h5"))
    surv = []
    for f in fs:
        try:
            with h5py.File(f, "r") as h:
                se = h["data/excitation/state_energies"][:]
        except Exception:
            continue
        if np.any(se[-1] != 0.0):     # alive at last frame -> survivor
            surv.append(f)
    return surv

lists = [(g, o, survivor_files(g)) for g, o in pairs]
print("survivors per source:", [(o.split("/")[-1], len(s)) for _, o, s in lists])
for g, o, slist in lists:
    use = slist                          # ALL survivors of THIS source (own N)
    N = len(use)
    with h5py.File(use[0], "r") as h:
        NS = h["data/excitation/state_energies"].shape[1]; NF = h["data/excitation/state_energies"].shape[0]
    NEXC = NS - 1
    clas_count = np.zeros((NF, NS)); quant_sum = np.zeros((NF, NS))
    for f in use:
        with h5py.File(f, "r") as h:
            act = h["data/nonadiabatic/active_surface"][:].ravel()
            amp = h["data/nonadiabatic/electronic_amplitudes"][:]
        a = np.clip(act[:NF], 0, NS - 1)
        np.add.at(clas_count, (np.arange(NF), a), 1)
        with np.errstate(over="ignore", invalid="ignore"):
            p = amp[:NF, :, 0] ** 2 + amp[:NF, :, 1] ** 2; norm = p.sum(1)
        good = np.isfinite(norm) & (np.abs(norm - 1.0) < QTOL)
        pe = np.zeros((NF, NEXC)); pe[good] = p[good] / norm[good][:, None]
        quant_sum[:, 1:] += pe
    clas = clas_count / N
    quant = quant_sum / N                     # excited only; S0 column stays 0 (not used)
    np.savez_compressed(o, clas=clas, quant=quant, NF=NF, NS=NS, N=N, pspf=PSPF, nsurv_avail=len(slist))
    print("WROTE", o.split("/")[-1], "N", N, "end clas:", " ".join(f"S{s}:{clas[-1, s]:.3f}" for s in range(NS)),
          "| end quant:", " ".join(f"S{s}:{quant[-1, s]:.3f}" for s in range(1, NS)))

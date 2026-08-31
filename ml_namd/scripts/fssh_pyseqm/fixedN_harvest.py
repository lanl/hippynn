#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FIXED-N population harvest (total conserved, ground-state recovery visible).
Uses the FIRST N trajectories. Each trajectory contributes at EVERY frame:
  - while alive: its active surface (classical) / excited |c|^2 (quantum)
  - after it terminates (reaches the S1-S0 funnel): counted as S0 (recovered)
Denominator is FIXED at N, so populations sum to 1 at every frame and S0 rises as trajectories recover.
Usage: python fixedN_harvest.py <h5_glob> <out_npz> <N> <ps_per_frame>
"""
import sys, glob, numpy as np, h5py
GLOB, OUT, NFIX, PSPF = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
QTOL = 0.1
fs = sorted(f for f in glob.glob(GLOB) if f.endswith(".h5"))[:NFIX]
assert fs, f"no h5 in {GLOB}"
NF = NS = None
for f in fs:
    try:
        with h5py.File(f, "r") as h:
            NS = h["data/excitation/state_energies"].shape[1]
            NF = h["data/excitation/state_energies"].shape[0]
        break
    except Exception:
        continue
NEXC = NS - 1
clas_count = np.zeros((NF, NS)); quant_sum = np.zeros((NF, NS)); nalive = np.zeros(NF)
nused = 0; bad = 0
for f in fs:
    try:
        with h5py.File(f, "r") as h:
            se = np.asarray(h["data/excitation/state_energies"][:])
            act = np.asarray(h["data/nonadiabatic/active_surface"][:]).ravel()
            amp = np.asarray(h["data/nonadiabatic/electronic_amplitudes"][:])
    except Exception:
        bad += 1; continue
    n = min(NF, se.shape[0])
    alive = np.any(se[:n] != 0.0, axis=1)
    lastv = np.where(alive)[0]; death = (lastv[-1] + 1) if len(lastv) else 0
    aliveF = np.arange(NF) < death
    # classical assigned surface: alive -> active; dead/recovered -> S0 (0)
    assigned = np.zeros(NF, dtype=np.int64)
    assigned[:n] = np.where(alive, np.clip(act[:n], 0, NS - 1), 0)
    np.add.at(clas_count, (np.arange(NF), assigned), 1)
    nalive += aliveF
    # quantum (fixed-N, EXCITED ONLY, S0 never used): alive+good-norm -> excited |c|^2;
    # terminated / bad-norm -> contribute 0 (left the excited manifold). So excited pops DEPLETE with recovery.
    qrow = np.zeros((NF, NS))                   # default 0 -> terminated frames add nothing
    with np.errstate(over="ignore", invalid="ignore"):
        p = amp[:n, 0] ** 2 + amp[:n, 1] ** 2 if amp.ndim == 2 else amp[:n, :, 0] ** 2 + amp[:n, :, 1] ** 2
        norm = p.sum(1)
    good = np.zeros(NF, bool)
    good[:n] = alive & np.isfinite(norm) & (np.abs(norm - 1.0) < QTOL)
    pe = np.zeros((NF, NEXC)); pe[:n][good[:n]] = p[good[:n]] / norm[good[:n]][:, None]
    qrow[good, 1:] = pe[good]                   # S0 column stays 0 (never used)
    quant_sum += qrow
    nused += 1
clas = clas_count / nused
quant = quant_sum / nused
np.savez_compressed(OUT, clas=clas, quant=quant, NF=NF, NS=NS, N=nused, pspf=PSPF,
                    nalive=nalive, frac_alive=nalive / nused, bad=bad)
print(f"WROTE {OUT}  N={nused} NF={NF} NS={NS} bad={bad}")
print("  end clas: " + " ".join(f"S{s}:{clas[-1, s]:.3f}" for s in range(NS)))
print("  end quant:" + " ".join(f"S{s}:{quant[-1, s]:.3f}" for s in range(NS)))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full FSSH ensemble harvest for the 6 like-for-like plots (reference OR MLIP).
Reads a dir of per-trajectory h5, computes ensemble quantities with LIVE-ENSEMBLE
(survivor-conditional) normalization, saves a compact npz. Schema-agnostic in n_states.

Usage: python harvest_fssh_full.py <h5_glob> <out_npz> <ps_per_frame>
  e.g. python harvest_fssh_full.py "out/enol_init3/*.h5" enol_ref_init3_full.npz 0.002
Conventions (from plot_enol1ps.py / L-045 / L-047):
  * alive = any(state_energies != 0) per frame (driver zero-fills after termination)
  * classical pop = fraction on each surface among ALIVE traj
  * quantum |c|^2: excited-only amplitudes (col j -> S(j+1)); screen finite & |norm-1|<QTOL,
    renormalize; S0 = 1 - sum(excited) (norm deficit)
  * transition E_j - E_0 (eV), mean over alive
  * adjacent gaps E_{i+1}-E_i (eV): median + IQR over alive
  * aCC = per-frame MAX over backbone C-C bonds (2,3),(2,4),(3,5),(4,6) [0-based]; median over survivors

Inputs (positional argv): h5 glob, output .npz path, picoseconds-per-frame (e.g. 0.002).
Output: a compact .npz with ensemble arrays (clas, quant, transE, gaps, aCC, frac_alive, death, ...).
This is the CANONICAL like-for-like harvester; kept as a top-level script (no main()) so its md5
stays stable across the codebase. Behavior is unchanged by this documentation pass.
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import sys, glob, os
import numpy as np, h5py

GLOB = sys.argv[1]; OUT = sys.argv[2]; PSPF = float(sys.argv[3]); QTOL = 0.1
ACC_BONDS = [(2, 3), (2, 4), (3, 5), (4, 6)]   # 0-based backbone C-C (keto==enol)
fs = sorted(glob.glob(GLOB)); fs = [f for f in fs if f.endswith(".h5")]
assert fs, f"no h5 in {GLOB}"

# discover NF, nstates from first readable
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
N = len(fs)
print(f"{GLOB}: N={N} NF={NF} NS={NS} (S0..S{NS-1})")

# per-traj stores
ACT = np.full((N, NF), -1, np.int16)
SE = np.zeros((N, NF, NS))                 # state energies (eV, absolute)
C2 = np.full((N, NF, NEXC), np.nan)        # |c|^2 excited
CCX = np.full((N, NF), np.nan)             # per-frame max backbone C-C
DEATH = np.full(N, NF, np.int32)
bad = 0
for i, f in enumerate(fs):
    try:
        with h5py.File(f, "r") as h:
            se = np.asarray(h["data/excitation/state_energies"][:])
            act = np.asarray(h["data/nonadiabatic/active_surface"][:]).ravel()
            amp = np.asarray(h["data/nonadiabatic/electronic_amplitudes"][:])
            co = np.asarray(h["coordinates/values"][:])
    except Exception:
        bad += 1; continue
    n = min(NF, se.shape[0]); SE[i, :n] = se[:n]
    alive = np.any(se[:n] != 0.0, axis=1)
    lastv = np.where(alive)[0]
    DEATH[i] = (lastv[-1] + 1) if len(lastv) else 0
    ACT[i, :n][alive] = np.clip(act[:n][alive], 0, NS - 1)
    with np.errstate(over="ignore", invalid="ignore"):
        p = amp[:n, :, 0] ** 2 + amp[:n, :, 1] ** 2
        norm = p.sum(1)
    ok = alive & np.isfinite(norm) & (np.abs(norm - 1.0) < QTOL)
    pn = np.zeros_like(p); pn[ok] = p[ok] / norm[ok][:, None]
    C2[i, :n][ok] = pn[ok]
    # aCC max over backbone bonds, alive frames only (coords aligned to same stride)
    m = min(n, co.shape[0])
    d = np.stack([np.linalg.norm(co[:m, a] - co[:m, b], axis=1) for a, b in ACC_BONDS], 0).max(0)
    CCX[i, :m][alive[:m]] = d[alive[:m]]

# alive mask (N,NF)
alive = np.arange(NF)[None, :] < DEATH[:, None]
nalive = alive.sum(0)
den = np.where(nalive > 0, nalive, np.nan)

# 1. classical populations (live-ensemble)
clas = np.stack([np.where(nalive > 0, (alive & (ACT == s)).sum(0) / den, np.nan) for s in range(NS)], 1)
# 2. quantum populations: mean excited |c|^2 over frames with data; S0 = norm deficit
nq = np.sum(~np.isnan(C2[:, :, 0]), 0)
qden = np.where(nq > 0, nq, np.nan)
qexc = np.stack([np.nansum(C2[:, :, j], 0) / qden for j in range(NEXC)], 1)   # (NF,NEXC) -> S1..
quant = np.zeros((NF, NS)); quant[:, 1:] = qexc
quant[:, 0] = np.clip(1.0 - np.nansum(qexc, 1), 0.0, None)
qcov = nq / max(N, 1)
# 3. fraction alive (last-live >= t)
comp = DEATH >= NF
last = np.where(comp, NF - 1, DEATH - 1)
frac_alive = np.array([np.mean(last >= t) for t in range(NF)])
# 4. transition energies E_j - E_0 (mean over alive, eV)
dE = SE - SE[:, :, :1]                         # (N,NF,NS)
transE = np.stack([np.where(nalive > 0, np.nansum(np.where(alive, dE[:, :, s], np.nan), 0) / den, np.nan)
                   for s in range(NS)], 1)
# 5. adjacent gaps E_{i+1}-E_i: median + IQR over alive (>=5 alive)
gaps = SE[:, :, 1:] - SE[:, :, :-1]            # (N,NF,NS-1)
gA = np.where(alive[:, :, None], gaps, np.nan)
gmed = np.nanmedian(gA, 0); gq1 = np.nanpercentile(gA, 25, 0); gq3 = np.nanpercentile(gA, 75, 0)
gmask = nalive >= 5
# 6. aCC elongation: median + mean over survivors
ccmed = np.nanmedian(np.where(alive, CCX, np.nan), 0)
ccmean = np.where(nalive > 0, np.nansum(np.where(alive, CCX, np.nan), 0) / den, np.nan)

np.savez_compressed(OUT, NF=NF, NS=NS, N=N, bad=bad, pspf=PSPF, nalive=nalive,
                    clas=clas, quant=quant, qcov=qcov, frac_alive=frac_alive, transE=transE,
                    gmed=gmed, gq1=gq1, gq3=gq3, gmask=gmask, ccmed=ccmed, ccmean=ccmean,
                    death=DEATH)
print(f"WROTE {OUT} (bad={bad}); survivors(reach end)={int((DEATH>=NF).sum())}")
print(f"  end classical: " + " ".join(f"S{s}:{np.nan_to_num(clas)[-1,s]:.3f}" for s in range(NS)))
print(f"  aCC median frame0/end: {ccmed[0]:.3f}/{np.nan_to_num(ccmed)[-1]:.3f} A")

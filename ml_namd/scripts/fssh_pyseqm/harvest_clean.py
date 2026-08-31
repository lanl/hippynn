#!/usr/bin/env python
"""STRICT clean-survivor ensemble: use ONLY trajectories that (a) survive the full
length (state_energies != 0 at final frame) AND (b) contain NO NaN in the electronic
amplitude norm at any alive frame. Reports per-source counts + every excluded NaN traj.
Classical = active-surface fraction; quantum = mean excited |c|^2 (renormalised).
Usage: harvest_clean.py <pspf> <label> <glob> <out.npz>
"""
import sys, glob, numpy as np, h5py
PSPF = float(sys.argv[1]); LABEL = sys.argv[2]; GLOB = sys.argv[3]; OUT = sys.argv[4]
fs = sorted(f for f in glob.glob(GLOB) if f.endswith(".h5"))
ntot = len(fs); n_term = 0; n_unread = 0
nan_surv = []; clean = []
for f in fs:
    try:
        with h5py.File(f, "r") as h:
            se = h["data/excitation/state_energies"][:]
            amp = h["data/nonadiabatic/electronic_amplitudes"][:]
    except Exception:
        n_unread += 1; continue
    alive = np.any(se != 0, 1)
    survivor = bool(np.any(se[-1] != 0.0))
    p = amp[:, :, 0] ** 2 + amp[:, :, 1] ** 2; norm = p.sum(1)
    has_nan = bool(np.any(~np.isfinite(norm) & alive)) or bool(np.any(~np.isfinite(se[alive])))
    if not survivor:
        n_term += 1
        continue
    if has_nan:
        bf = np.where(~np.isfinite(norm) & alive)[0]
        nan_surv.append((f.split("/")[-1], int(bf[0]) if bf.size else -1))
        continue
    clean.append(f)
N = len(clean)
with h5py.File(clean[0], "r") as h:
    NF = h["data/excitation/state_energies"].shape[0]; NS = h["data/excitation/state_energies"].shape[1]
clas = np.zeros((NF, NS)); quant = np.zeros((NF, NS))
for f in clean:
    with h5py.File(f, "r") as h:
        act = h["data/nonadiabatic/active_surface"][:].ravel()
        amp = h["data/nonadiabatic/electronic_amplitudes"][:]
    a = np.clip(act[:NF], 0, NS - 1)
    np.add.at(clas, (np.arange(NF), a), 1)
    p = amp[:NF, :, 0] ** 2 + amp[:NF, :, 1] ** 2; norm = p.sum(1)
    pe = p / norm[:, None]        # (NF, NEXC); amp col j -> state S(j+1)
    quant[:, 1:] += pe
clas /= N; quant /= N
np.savez_compressed(OUT, clas=clas, quant=quant, NF=NF, NS=NS, N=N, pspf=PSPF,
                    ntot=ntot, n_terminated=n_term, n_nan_excluded=len(nan_surv))
endc = " ".join("S%d:%.3f" % (s, clas[-1, s]) for s in range(NS))
endq = " ".join("S%d:%.3f" % (s, quant[-1, s]) for s in range(1, NS))
print("===== %s =====" % LABEL)
print("  total h5=%d | unreadable=%d | terminated(non-survivor)=%d | survivors=%d"
      % (ntot, n_unread, n_term, ntot - n_term - n_unread))
print("  NaN survivors EXCLUDED = %d  %s" % (len(nan_surv), nan_surv if nan_surv else ""))
print("  >>> CLEAN N used in ensemble = %d" % N)
print("  end clas:  %s" % endc)
print("  end quant: %s" % endq)

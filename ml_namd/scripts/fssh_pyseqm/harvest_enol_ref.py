#!/usr/bin/env python
"""Harvest classical (active_surface) + quantum (|c|^2) populations from
updated-PySEQM enol reference pilots (init3 / init5, 1 ps, dt=0.05, 100 IC each).

CONVENTIONS (verified against init state, memory fssh-pyseqm-no-decoherence-overcoherence):
  * active_surface is S0-indexed (0=S0). CLASSICAL pop = fraction of traj on each surface.
  * electronic_amplitudes has width W = n_excited; column j maps to EXCITED state S(j+1).
    No S0 amplitude is stored, so quantum S0 pop = 1 - sum_j |c_j|^2 (over-coherence deficit).
Writes a compact npz + prints a sanity summary. Data stays on Darwin.

Inputs (positional argv): the base dir holding ``enol_init3/`` and ``enol_init5/`` h5 subdirs.
Output: ``<base>/enol_ref_pops.npz`` (per-init time axis + classical/quantum populations).
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import sys, glob, os
import numpy as np
import h5py

def harvest(d, init):
    """Harvest classical + quantum populations from the enol reference h5 in dir ``d``.

    ``init`` is the initial surface label (3 or 5), used only for reporting. Returns a dict with
    the time axis ``t`` [fs] and ``(T, n_states)`` classical/quantum population arrays.
    """
    files = sorted(glob.glob(os.path.join(d, "*.h5")))
    assert files, f"no h5 in {d}"
    t = None; clas = []; qexc = []; bad = 0
    for fp in files:
        try:
            with h5py.File(fp, "r") as f:
                asf = f["data/nonadiabatic/active_surface"][:]            # (T,) S0-indexed
                amp = f["data/nonadiabatic/electronic_amplitudes"][:]     # (T, W, 2) excited only
                steps = f["data/nonadiabatic/steps"][:]
        except Exception:
            bad += 1; continue
        c2 = amp[..., 0]**2 + amp[..., 1]**2                             # (T, W) |c|^2 excited
        if t is None:
            t = steps.astype(float) * 0.05                               # fs
        clas.append(asf); qexc.append(c2)
    clas = np.stack(clas)                # (N, T)
    qexc = np.stack(qexc)                # (N, T, W)
    N, T = clas.shape
    W = qexc.shape[2]
    ntot = W + 1                         # S0 .. SW
    # CLASSICAL: fraction of trajectories on each surface (S0-indexed)
    clas_pop = np.zeros((T, ntot))
    for k in range(ntot):
        clas_pop[:, k] = (clas == k).mean(axis=0)
    # QUANTUM: excited col j -> S(j+1); S0 = 1 - sum(excited) (clamped >=0)
    qexc_mean = qexc.mean(axis=0)        # (T, W) -> S1..SW
    quant_pop = np.zeros((T, ntot))
    quant_pop[:, 1:] = qexc_mean
    quant_pop[:, 0] = np.clip(1.0 - qexc_mean.sum(axis=1), 0.0, None)
    print(f"\n=== enol init={init}  N={N} traj (bad={bad}), T={T} frames, states=S0..S{W} ===")
    print(f"  t: {t[0]:.1f}..{t[-1]:.1f} fs   (initial active_surface unique: {np.unique(clas[:,0])})")
    def line(tag, pop):
        print(f"  {tag}: " + "  ".join(f"S{k}:{pop[k]:.3f}" for k in range(ntot)))
    line("CLASSICAL @  0fs", clas_pop[0]); line("CLASSICAL @1ps ", clas_pop[-1])
    line("QUANTUM   @  0fs", quant_pop[0]); line("QUANTUM   @1ps ", quant_pop[-1])
    s1 = clas_pop[:, 1]; idx = np.where(s1 > 0.5)[0]
    print(f"  classical S1>0.5 first at: {t[idx[0]]:.0f} fs" if len(idx) else "  classical S1 never >0.5")
    # S0 arrival (classical) — internal conversion onset to ground state
    s0 = clas_pop[:, 0]; idx0 = np.where(s0 > 0.1)[0]
    print(f"  classical S0>0.1 first at: {t[idx0[0]]:.0f} fs" if len(idx0) else "  classical S0 stays <0.1")
    return dict(t=t, clas_pop=clas_pop, quant_pop=quant_pop, N=N, ntot=ntot, init=init)

if __name__ == "__main__":
    base = sys.argv[1]
    r3 = harvest(os.path.join(base, "enol_init3"), 3)
    r5 = harvest(os.path.join(base, "enol_init5"), 5)
    np.savez(os.path.join(base, "enol_ref_pops.npz"),
             t3=r3["t"], clas3=r3["clas_pop"], quant3=r3["quant_pop"], N3=r3["N"], ntot3=r3["ntot"],
             t5=r5["t"], clas5=r5["clas_pop"], quant5=r5["quant_pop"], N5=r5["N"], ntot5=r5["ntot"])
    print(f"\nWROTE {os.path.join(base, 'enol_ref_pops.npz')}")

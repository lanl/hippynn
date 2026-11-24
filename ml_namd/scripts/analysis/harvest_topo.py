#!/usr/bin/env python
"""Fragmentation-guarded clean-survivor harvest. Conditions a,b,c exactly as harvest_clean.py
(survive to end / no NaN / <=15A peak-dist) + (d) NO FRAGMENTATION: no canonical acac bond stretched
past FRAGFAC x covalent-radii at ANY alive frame. Calibrated reference-clean at 1.8x (flags only the
pre-existing explosion). Reports OLD clean (a,b,c) vs NEW frag-clean (a..d): N + final classical and
quantum populations; saves full population curves.
Usage: harvest_topo.py <glob> <label> <out.npz> [maxdist=15.0] [fragfac=1.8] [limit=0]
"""
import sys, glob, numpy as np, h5py
GLOB = sys.argv[1]; LABEL = sys.argv[2]; OUT = sys.argv[3]
MAXDIST = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0
FRAGFAC = float(sys.argv[5]) if len(sys.argv) > 5 else 1.8
LIMIT = int(sys.argv[6]) if len(sys.argv) > 6 else 0
RCOV = {1: 0.31, 6: 0.76, 8: 0.66}


def peak_pairdist(coa):
    d = coa[:, :, None, :] - coa[:, None, :, :]
    return float(np.sqrt((d * d).sum(-1)).max())


fs = sorted(f for f in glob.glob(GLOB) if f.endswith(".h5"))
if LIMIT:
    fs = fs[:LIMIT]
Z = None; cu = None; GENp = None; iu = None
ntot = 0; n_unread = 0; n_term = 0; nan_surv = []; expl = []; clean_abc = []; frag_bad = []; clean_abcd = []
for f in fs:
    ntot += 1
    try:
        with h5py.File(f, "r") as h:
            se = h["data/excitation/state_energies"][:]; amp = h["data/nonadiabatic/electronic_amplitudes"][:]
            co = h["coordinates/values"][:]
            if Z is None:
                Z = np.asarray(h["atoms"][:]).ravel().astype(int)
    except Exception:
        n_unread += 1; continue
    alive = np.any(se != 0, 1); survivor = bool(np.any(se[-1] != 0.0))
    with np.errstate(over="ignore", invalid="ignore"):
        p = amp[:, :, 0] ** 2 + amp[:, :, 1] ** 2; norm = p.sum(1)
    has_nan = bool(np.any(~np.isfinite(norm) & alive)) or bool(np.any(~np.isfinite(se[alive])))
    if not survivor:
        n_term += 1; continue
    if has_nan:
        nan_surv.append(f); continue
    coa = co[alive]
    if peak_pairdist(coa) > MAXDIST:
        expl.append(f); continue
    clean_abc.append(f)
    dd = np.sqrt(((coa[:, :, None, :] - coa[:, None, :, :]) ** 2).sum(-1))
    if cu is None:
        rc = np.array([RCOV[int(z)] for z in Z]); TIGHT = 1.3 * (rc[:, None] + rc[None, :]); GEN = FRAGFAC * (rc[:, None] + rc[None, :])
        a0 = dd[0] < TIGHT; np.fill_diagonal(a0, False); iu = np.triu_indices(len(Z), 1); cu = a0[iu]; GENp = GEN[iu]
        print("  [canon] bonds=%d fragfac=%.2f" % (int(cu.sum()), FRAGFAC), flush=True)
    dp = dd[:, iu[0], iu[1]]
    broken = ((dp > GENp) & cu).any(1)
    if broken.any():
        frag_bad.append((f.split("/")[-1], int(np.where(broken)[0][0]))); continue
    clean_abcd.append(f)


def pops(files):
    with h5py.File(files[0], "r") as h:
        NF, NS = h["data/excitation/state_energies"].shape
    clas = np.zeros((NF, NS)); quant = np.zeros((NF, NS))
    for f in files:
        with h5py.File(f, "r") as h:
            act = h["data/nonadiabatic/active_surface"][:].ravel(); amp = h["data/nonadiabatic/electronic_amplitudes"][:]
        np.add.at(clas, (np.arange(NF), np.clip(act[:NF], 0, NS - 1)), 1)
        with np.errstate(over="ignore", invalid="ignore"):
            p = amp[:NF, :, 0] ** 2 + amp[:NF, :, 1] ** 2; norm = p.sum(1); quant[:, 1:] += (p / norm[:, None])
    return clas / len(files), quant / len(files), NF, NS


ca, qa, NF, NS = pops(clean_abc)
cd_, qd_ = (pops(clean_abcd)[:2] if clean_abcd else (None, None))
drop = len(clean_abc) - len(clean_abcd)


def fmt(v, lo=0):
    return " ".join("S%d:%.3f" % (s, v[-1, s]) for s in range(lo, NS))


print("===== %s =====" % LABEL)
print("  total=%d unread=%d term=%d nan=%d exploded(>%.0fA)=%d" % (ntot, n_unread, n_term, len(nan_surv), MAXDIST, len(expl)))
print("  OLD clean(a,b,c)      N=%d  clas[%s]  quant[%s]" % (len(clean_abc), fmt(ca), fmt(qa, 1)))
print("  FRAGMENTED excluded   = %d  (%.1f%% of old clean)" % (drop, 100 * drop / max(len(clean_abc), 1)))
if cd_ is not None:
    print("  NEW frag-clean(a..d)  N=%d  clas[%s]  quant[%s]" % (len(clean_abcd), fmt(cd_), fmt(qd_, 1)))
np.savez_compressed(OUT, clas_abc=ca, quant_abc=qa, clas_abcd=(cd_ if cd_ is not None else np.zeros_like(ca)),
                    quant_abcd=(qd_ if qd_ is not None else np.zeros_like(qa)), N_abc=len(clean_abc),
                    N_abcd=len(clean_abcd), NF=NF, NS=NS, fragfac=FRAGFAC)
print("  DONE", flush=True)

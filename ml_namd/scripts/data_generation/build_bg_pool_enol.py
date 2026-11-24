#!/usr/bin/env python3
"""Build the enol S0-S3 Model-B training pool (pool_s0s3) for the BALF and GALF sets.

Copies the S0-S3 energies/forces + dENACR{12,13,23} targets from each set's ``extracted/50K`` into a
``pool_s0s3`` dir, then derives the Model-B inputs: GDV{ij} = F_i - F_j (excited-force difference,
flattened to (N,45) float32) and GAP{ij} = sE_j - sE_i (float64). Prints a GAP12 near-CI summary.
Top-level script (no main()); paths are the confirmed the cluster data_gen locations.
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import numpy as np, os, shutil
L="<ML_NAMD_ROOT>/data_gen/enol_balf_galf"
PAIRS={"12":(1,2),"13":(1,3),"23":(2,3)}
COPY=["acn_R","acn_Z"]+[f"acn_sE{s}" for s in range(4)]+[f"acn_F{s}" for s in range(4)]+[f"acn_dENACR{p}" for p in PAIRS]
for t in ["BALF","GALF"]:
    src=f"{L}/{t}/extracted/50K"; dst=f"{L}/{t}/pool_s0s3"; os.makedirs(dst,exist_ok=True)
    for k in COPY: shutil.copy(f"{src}/{k}.npy",f"{dst}/{k}.npy")
    F={s:np.load(f"{src}/acn_F{s}.npy") for s in range(4)}
    sE={s:np.load(f"{src}/acn_sE{s}.npy") for s in range(4)}
    N=F[0].shape[0]
    for p,(i,j) in PAIRS.items():
        np.save(f"{dst}/acn_GDV{p}.npy",(F[i]-F[j]).reshape(N,45).astype(np.float32))
        np.save(f"{dst}/acn_GAP{p}.npy",(sE[j]-sE[i]).astype(np.float64))
    g=np.load(f"{dst}/acn_GAP12.npy")
    print(f"{t}: pool_s0s3 N={N}  files={len(os.listdir(dst))}  GAP12 median={np.median(g):.3f} min={g.min():.4f} pct_lt0.1eV={100*(g<0.1).mean():.1f}")

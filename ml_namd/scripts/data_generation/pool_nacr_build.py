#!/usr/bin/env python3
"""Build the NACR-inclusive pooled S0-S3 dataset (200K per tautomer): E+F + dENACR{12,13,23}
targets + GDV{12,13,23} inputs. Same source order/slicing as pool_s0s3.py so frames align.

Pools the GS (S0) + S1/S2/S3-dynamics extracted sources per tautomer, writes a source_surface provenance
array, and derives GDV{ij} = F_i - F_j (float32, (N,45)). Top-level script; paths are the Chicoma locations.
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import numpy as np, os
GS = "/users/akhanna2/data/ml_namd_project/data/{t}/50K"
EX = "/lustre/scratch5/akhanna2/ml_namd_project/data_gen/prod/{t}_s{s}/extracted/50K"
OUT = "/lustre/scratch5/akhanna2/ml_namd_project/data_gen/pooled/{t}_s0s3_200k_nacr"
EF = ["acn_R","acn_Z","acn_sE0","acn_sE1","acn_sE2","acn_sE3","acn_F0","acn_F1","acn_F2","acn_F3"]
DEN = ["acn_dENACR12","acn_dENACR13","acn_dENACR23"]

for t in ("enol","keto"):
    sources = [GS.format(t=t)] + [EX.format(t=t,s=s) for s in (1,2,3)]
    out = OUT.format(t=t); os.makedirs(out, exist_ok=True)
    print(f"\n===== {t} =====")
    for f in EF + DEN:
        arrs = [np.load(os.path.join(s, f+".npy")) for s in sources]
        if f in DEN:
            arrs = [a.reshape(a.shape[0], -1) for a in arrs]   # normalize to (N,45)
            widths = {a.shape[1] for a in arrs}
            assert widths == {45}, f"{f} width mismatch across sources: {widths}"
        pooled = np.concatenate(arrs, axis=0)
        np.save(os.path.join(out, f+".npy"), pooled)
    prov = np.concatenate([np.full(np.load(os.path.join(s,"acn_sE0.npy"),mmap_mode="r").shape[0], i, dtype=np.int8)
                           for i,s in enumerate(sources)])
    np.save(os.path.join(out,"source_surface.npy"), prov)
    # GDV_ij = F_i - F_j (excited forces, 1-indexed): subtract in f64 -> f32, flatten (N,45)
    F1 = np.load(os.path.join(out,"acn_F1.npy")).astype(np.float64)
    F2 = np.load(os.path.join(out,"acn_F2.npy")).astype(np.float64)
    F3 = np.load(os.path.join(out,"acn_F3.npy")).astype(np.float64)
    N = F1.shape[0]
    for pair, (a, b) in {"12": (F1,F2), "13": (F1,F3), "23": (F2,F3)}.items():
        gdv = (a - b).reshape(N, -1).astype(np.float32)
        assert gdv.shape == (N, 45), gdv.shape
        np.save(os.path.join(out, f"acn_GDV{pair}.npy"), gdv)
    de = np.load(os.path.join(out,"acn_dENACR12.npy"))
    print(f"  POOLED NACR -> {out}\n  N={N}  S0/S1/S2/S3={np.bincount(prov).tolist()}  "
          f"dENACR12{de.shape}  GDV(N,45)  finite(dENACR12)={np.isfinite(de).all()}")
print("\nDONE")

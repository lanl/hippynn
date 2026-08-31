#!/usr/bin/env python3
"""Generate NVE start geometries (enol/keto) from the GS S0 acn_R frame, in acn_Z order.

Reads frame ``FRAME`` of each tautomer's S0 ``acn_R``/``acn_Z`` and writes ``geom/<t>_s0_start.xyz``.
Paths are the confirmed Chicoma locations. Top-level script (no main()); documentation-only refactor.
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import numpy as np, os
SYM = {1: "H", 6: "C", 8: "O"}
GEOM = "/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md/geom"
FRAME = 25000  # mid-trajectory, well-sampled, physical
os.makedirs(GEOM, exist_ok=True)
for t in ("enol", "keto"):
    d = f"/users/akhanna2/data/ml_namd_project/data/{t}/50K"
    R = np.load(f"{d}/acn_R.npy")           # (N,15,3)
    Z = np.load(f"{d}/acn_Z.npy")           # (N,15) or (15,)
    z = Z[FRAME] if Z.ndim == 2 else Z
    r = R[FRAME]
    out = f"{GEOM}/{t}_s0_start.xyz"
    with open(out, "w") as f:
        f.write(f"{len(z)}\n{t} GS S0 frame {FRAME}\n")
        for zi, ri in zip(z, r):
            f.write(f"{SYM[int(zi)]} {ri[0]:.10f} {ri[1]:.10f} {ri[2]:.10f}\n")
    print(f"wrote {out}  R{r.shape}  Z={[int(x) for x in z]}")

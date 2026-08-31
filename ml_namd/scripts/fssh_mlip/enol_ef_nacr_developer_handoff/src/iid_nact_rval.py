#!/usr/bin/env python3
"""IN-DISTRIBUTION (i.i.d.) NACT|r| = corr(|NACT_pred|, |NACT_true|) for JOINT vs Model B,
the held-out-test control for the OOD NACT|r| (ood_nact_bvj.py). SAME recipe, but the eval set
is the model's EXACT held-out test split of the 400K pool (hippynn RandomState(114514) split,
reproduced in numpy), restricted to the S1-S3 dynamics segments (the S0 GS-MD segment has no
aligned velocity and is off-CI; excited surfaces also match the OOD's setting).

pool row layout (build_pool_400k.py, 100K each, fixed order): [S0 gs100k | S1 | S2 | S3].
velocity for S1-S3 test rows comes from nact_work/acn_V_{t}_s{k}.npy[row - k*100000] (aligned,
run_nact_all-validated). true NACT = (dENACR_oracle/GAP_oracle . V); pred NACT = (dENACR_pred/gap_pred . V).
The C=2.41888425e-5 scale cancels in the correlation, so it is omitted.
"""
import os, sys, argparse
os.environ["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"
import numpy as np, torch
torch.set_default_dtype(torch.float64)
TRAIN = "/users/akhanna2/data/ml_namd_project/scripts/train"
HSRC = "/users/akhanna2/data/ml_namd_project/fssh_mlip/HANDOFF/src"
sys.path.insert(0, TRAIN); sys.path.insert(0, HSRC)
import unified_training, direct_nacr_head_v3, direct_nacr_head_v2  # noqa
import __main__ as _M
for _n in dir(unified_training):
    if not _n.startswith("__"):
        setattr(_M, _n, getattr(unified_training, _n))
from hiphop_predictor_v3 import HiphopPredictorV3
from modelb_composed_predictor import ModelBComposedPredictor

STATES = [0, 1, 2, 3]; PAIRS = ["12", "13", "23"]
L = "/lustre/scratch5/akhanna2/ml_namd_project"; EXP = L + "/experiments"
N_TOTAL = 400000; N_PER = 100000; SEED = 114514

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=["enol", "keto"])
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--device", default="cuda")
ap.add_argument("--joint-dir", default=f"{EXP}/joint_pool400k_nf172", help="JOINT model root holding {t}/model")
ap.add_argument("--modelA-dir", default=f"{EXP}/modelA_pool400k", help="frozen E+F Model A root")
ap.add_argument("--modelB-dir", default=f"{EXP}/modelB_pool400k_v3", help="decoupled NACR Model B root")
a = ap.parse_args()
t = a.system

# ---- reproduce the model's EXACT held-out test split (hippynn DirectoryDatabase, seed 114514) ----
rs = np.random.RandomState(SEED)
test_idx = np.sort(rs.choice(np.arange(N_TOTAL, dtype=int), size=int(0.1 * N_TOTAL), replace=False))
seg = test_idx // N_PER                                       # 0,1,2,3
N = len(test_idx)
print(f"### {t} i.i.d. HELD-OUT TEST: {N} frames (FULL S0-S3 held-out split). per-surface: "
      f"S0={int((seg==0).sum())} S1={int((seg==1).sum())} S2={int((seg==2).sum())} S3={int((seg==3).sum())}")

# ---- pool arrays at the test rows ----
pool = f"{L}/data_gen/pooled/{t}_s0s3_400k"
R = np.ascontiguousarray(np.load(f"{pool}/acn_R.npy", mmap_mode="r")[test_idx])           # (N,15,3)
Z = np.ascontiguousarray(np.load(f"{pool}/acn_Z.npy", mmap_mode="r")[test_idx].astype(np.int64))
den_true = {p: np.asarray(np.load(f"{pool}/acn_dENACR{p}.npy", mmap_mode="r")[test_idx], np.float64) for p in PAIRS}
gap_true = {p: np.asarray(np.load(f"{pool}/acn_GAP{p}.npy", mmap_mode="r")[test_idx], np.float64) for p in PAIRS}

# ---- velocity for all S0-S3 test rows from aligned sources, (N,45) ----
# S0 from the D: ground-truth s0_dyn/100k velocity (transferred to HSRC; acn_R sha == pool gs100k, verified);
# S1-S3 from the run_nact_all-validated staged velocities. All row-aligned to the pool segments.
VPATH = {0: f"{HSRC}/acn_V_{t}_s0.npy",
         1: f"{L}/data_gen/nact_work/acn_V_{t}_s1.npy",
         2: f"{L}/data_gen/nact_work/acn_V_{t}_s2.npy",
         3: f"{L}/data_gen/nact_work/acn_V_{t}_s3.npy"}
Vf = np.empty((N, 45), dtype=np.float64)
Vsrc = {s: np.load(VPATH[s], mmap_mode="r") for s in (0, 1, 2, 3)}
for s in (0, 1, 2, 3):
    m = seg == s
    if not m.any():
        continue
    rows = test_idx[m] - s * N_PER
    Vf[m] = np.asarray(Vsrc[s][rows], np.float64).reshape(int(m.sum()), -1)

# alignment spot-check: pool R at S1 test rows == S1 extracted R at the same local rows
_chk = np.load(f"{L}/data_gen/prod/{t}_s1/extracted/100K/acn_R.npy", mmap_mode="r")
_m = seg == 1
if _m.any():
    _lr = test_idx[_m][:50] - N_PER
    _ok = np.allclose(R[_m][:50], np.asarray(_chk[_lr]))
    print(f"  [align-check] pool R == S1-source R at test rows: {_ok}")

def true_nact(p):
    d = den_true[p] / gap_true[p][:, None]
    return (d * Vf).sum(1)

NT = {p: true_nact(p) for p in PAIRS}

def predict_chunked(pred, **kw):
    acc = []
    for i in range(0, N, a.chunk):
        acc.append(pred.predict(Z[i:i + a.chunk], R[i:i + a.chunk], **kw))
    out = {}
    for key in ("sE", "dENACR", "gap_pair"):
        if key in acc[0] and isinstance(acc[0][key], dict) and acc[0][key]:
            out[key] = {k: np.concatenate([o[key][k] for o in acc]) for k in acc[0][key]}
    return out

def report(name, out, gapfn):
    print(f"  === {name} ===")
    for p in PAIRS:
        gap = np.asarray(gapfn(out, p), dtype=np.float64)
        d = out["dENACR"][p] / gap[:, None]
        nact = (d * Vf).sum(1)
        rr = np.corrcoef(np.abs(nact), np.abs(NT[p]))[0, 1]
        print(f"    NACT{p}: |r|={abs(rr):.3f}")

print("\n=== JOINT (converged 400K, v3 self-gap) ===")
jp = HiphopPredictorV3(f"{a.joint_dir}/{t}/model", TRAIN, states=STATES,
                       pairs=tuple(PAIRS), device=a.device, dtype="float64")
jo = predict_chunked(jp, gdv_source="forces", gap_source="self", check_species=False)
report("JOINT", jo, lambda o, p: o["sE"][int(p[1])] - o["sE"][int(p[0])])

print("\n=== MODEL B (decoupled, A->B composition) ===")
cp = ModelBComposedPredictor(f"{a.modelA_dir}/{t}/model", f"{a.modelB_dir}/{t}/model",
                             TRAIN, pairs=tuple(PAIRS), device=a.device, dtype="float64")
co = predict_chunked(cp, check_species=False)
report("MODEL B", co, lambda o, p: o["gap_pair"][p])
print("\nOOD reference (ab-initio geoms): enol JOINT 0.599/0.102/0.009 B 0.680/0.163/0.227 ;"
      " keto JOINT 0.783/0.736/0.800 B 0.626/0.711/0.780")

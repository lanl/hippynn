#!/usr/bin/env python3
"""OOD NACT|r|: decoupled Model B (A->B) vs the converged JOINT, on ab-initio S3-FSSH geoms.

Ground truth = PySEQM/AM1-CIS values in the FSSH h5 (E, active-F, true NACT via tdc=overlap).
Recipe: dENACR -> d_ij = dENACR_ij/gap_ij -> nact = (d.V); |r| = corr(|nact_mlip|, |nact_true|).
JOINT (v3) supplies its own gap (gap_source='self'); Model B rides frozen A's gap (composed A->B).
CHUNKED predict (forces build a create_graph autodiff graph -> OOM if all geoms at once).
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"
import os, sys, glob, argparse
os.environ["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"
import numpy as np, torch, h5py
torch.set_default_dtype(torch.float64)
TRAIN = "<ML_NAMD_ROOT>/scripts/train"
HSRC = "<ML_NAMD_ROOT>/fssh_mlip/HANDOFF/src"
sys.path.insert(0, TRAIN); sys.path.insert(0, HSRC)
import unified_training, direct_nacr_head_v3, direct_nacr_head_v2  # noqa
import __main__ as _M
for _n in dir(unified_training):
    if not _n.startswith("__"):
        setattr(_M, _n, getattr(unified_training, _n))
from hiphop_predictor_v3 import HiphopPredictorV3
from modelb_composed_predictor import ModelBComposedPredictor

STATES = [0, 1, 2, 3]; PAIRS = ["12", "13", "23"]; IDX = {"12": (0, 1), "13": (0, 2), "23": (1, 2)}
EXP = "<ML_NAMD_ROOT>/experiments"
EXPECTED = [8, 8, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=["enol", "keto"])
ap.add_argument("--h5glob", required=True)
ap.add_argument("--max-traj", type=int, default=0)
ap.add_argument("--chunk", type=int, default=100)
ap.add_argument("--device", default="cpu")
ap.add_argument("--joint-dir", default=f"{EXP}/joint_pool400k_nf172", help="JOINT model root holding {system}/model")
ap.add_argument("--modelA-dir", default=f"{EXP}/modelA_pool400k", help="frozen E+F Model A root")
ap.add_argument("--modelB-dir", default=f"{EXP}/modelB_pool400k_v3", help="decoupled NACR Model B root")
a = ap.parse_args()

paths = sorted(glob.glob(a.h5glob))
if a.max_traj: paths = paths[:a.max_traj]
R=[]; V=[]; NT=[]; AC=[]; ET=[]; FT=[]; atoms=None
for p in paths:
    with h5py.File(p, "r") as h:
        atoms = h["atoms"][:]
        R.append(h["coordinates/values"][:]); V.append(h["velocities/values"][:])
        NT.append(h["data/nonadiabatic/NACT"][:]); AC.append(h["data/nonadiabatic/active_surface"][:])
        ET.append(h["data/excitation/state_energies"][:]); FT.append(h["forces/values"][:])
R=np.concatenate(R); V=np.concatenate(V); NT=np.concatenate(NT); AC=np.concatenate(AC); ET=np.concatenate(ET); FT=np.concatenate(FT)
N=len(R); Z=np.tile(atoms.astype(np.int64),(N,1)); Vf=V.reshape(N,-1)
print(f"### {a.system.upper()} OOD: {len(paths)} traj -> N={N} geoms | atoms match EXPECTED = {list(atoms)==EXPECTED} | NACT{NT.shape} | chunk={a.chunk}")

def predict_chunked(pred, **kw):
    acc=[]
    for i in range(0, N, a.chunk):
        acc.append(pred.predict(Z[i:i+a.chunk], R[i:i+a.chunk], **kw))
    m={}
    for key in ("sE","E_abs","F","dENACR","gap_pair"):
        if key in acc[0] and isinstance(acc[0][key], dict) and acc[0][key]:
            m[key]={k: np.concatenate([o[key][k] for o in acc]) for k in acc[0][key]}
    return m

def report(out, gapfn, e_key, f_key):
    if e_key in out:
        emae=np.mean([np.abs(out[e_key][s]-ET[:,s]).mean() for s in STATES])*1000
        Fact=np.stack([out[f_key][int(AC[i])][i] for i in range(N)])
        print(f"  E S0-S3 MAE={emae:.1f} meV | active-F MAE={np.abs(Fact-FT).mean()*1000:.1f} meV/A")
    for p in PAIRS:
        gap=np.asarray(gapfn(out,p),dtype=np.float64)
        d=out["dENACR"][p]/gap[:,None]; nact=(d*Vf).sum(1)
        r,c=IDX[p]; rr=np.corrcoef(np.abs(nact),np.abs(NT[:,r,c]))[0,1]
        print(f"  NACT{p}: |r|={abs(rr):.3f}")

print("\n=== JOINT (converged 400K, v3 self-gap) ===")
jp=HiphopPredictorV3(f"{a.joint_dir}/{a.system}/model", TRAIN, states=STATES, pairs=tuple(PAIRS), device=a.device, dtype="float64")
jo=predict_chunked(jp, gdv_source="forces", gap_source="self", check_species=False)
report(jo, lambda o,p: o["sE"][int(p[1])]-o["sE"][int(p[0])], "E_abs", "F")

print("\n=== MODEL B (decoupled, A->B composition) ===")
cp=ModelBComposedPredictor(f"{a.modelA_dir}/{a.system}/model", f"{a.modelB_dir}/{a.system}/model", TRAIN, pairs=tuple(PAIRS), device=a.device, dtype="float64")
co=predict_chunked(cp, check_species=False)
report(co, lambda o,p: o["gap_pair"][p], "E_abs", "F")
print("\nbaselines: GS f310c8da |r| ~0.04/-0.13/0.19 ; direct_vector_v2/200K plateau S1S2 0.60 / S2S3 0.30 / S1S3 0.09")

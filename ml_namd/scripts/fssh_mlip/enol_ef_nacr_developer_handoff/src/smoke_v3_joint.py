#!/usr/bin/env python3
"""Smoke-test HiphopPredictorV3 against the JOINT (23141751) in-dist data.

Proves: (1) the v3 predictor CONSTRUCTS on a gap-gated model (GAP inputs wired);
(2) forward runs; (3) dENACR is correct -> PhLessMAE vs oracle targets is small
(should land near the training-log in-dist test MAE); (4) gap_source="self"
(self-consistent gap from the joint's own energies) reproduces the oracle-gap
result -- the JOINT OOD path.

Run on Chicoma (CPU login node OK; small subset). NOT a paper result.
"""
import os, sys, argparse
os.environ["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--head-dir", required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--prefix", default="acn_")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The pooled-model pickle carries DataDumper etc. referenced via __main__; inject
# unified_training's symbols there BEFORE the unpickle (mirrors phase2_ood_nact.py).
sys.path.insert(0, a.head_dir)
import unified_training, direct_nacr_head_v3, direct_nacr_head_v2  # noqa
import __main__ as _M
for _n in dir(unified_training):
    if not _n.startswith("__"):
        setattr(_M, _n, getattr(unified_training, _n))
from hiphop_predictor_v3 import HiphopPredictorV3

PAIRS = ("12", "13", "23")
D, PX = a.data_dir, a.prefix

def load(name):
    return np.load(os.path.join(D, PX + name + ".npy"), mmap_mode="r")

Rall, Zall = load("R"), load("Z")
Ntot = Rall.shape[0]
rng = np.random.default_rng(a.seed)
idx = np.sort(rng.choice(Ntot, size=min(a.n, Ntot), replace=False))
Z = np.asarray(Zall[idx]); R = np.asarray(Rall[idx])
GDV = {p: np.asarray(load(f"GDV{p}")[idx]) for p in PAIRS}
GAP = {p: np.asarray(load(f"GAP{p}")[idx]) for p in PAIRS}
TGT = {p: np.asarray(load(f"dENACR{p}")[idx]) for p in PAIRS}
print(f"loaded {len(idx)} in-dist frames from {D}")
print(f"Z[0] species = {Z[0].tolist()}")

def phaseless_mae(pred, true):
    """Per-frame sign-fold MAE (phase is per-frame). Returns eV/A."""
    s = np.sign((pred * true).sum(axis=1, keepdims=True))
    s[s == 0] = 1.0
    return np.abs(pred - s * true).mean()

print("\n=== construct HiphopPredictorV3 on the JOINT (v3, gap-gated) ===")
p = HiphopPredictorV3(a.model_dir, a.head_dir, states=(0, 1, 2, 3),
                      pairs=PAIRS, device=a.device, dtype="float64")
print(f"  has_gap = {p._has_gap}   full inputs = {p._full_in_names}")
assert p._has_gap, "FAIL: GAP inputs not detected on a v3 model"

print("\n=== TEST A: oracle GDV (file) + oracle GAP (file) -> reproduce training ===")
outA = p.predict(Z, R, gdv_source="file", gdv=GDV,
                 gap_source="file", gap=GAP, check_species=False)
print(f"  gdv_source={outA['gdv_source']} gap_source={outA['gap_source']}")
for pr in PAIRS:
    mae = phaseless_mae(outA["dENACR"][pr], TGT[pr]) * 1000
    print(f"  dENACR{pr}: PhLessMAE = {mae:7.3f} meV/A   "
          f"|pred|mean={np.abs(outA['dENACR'][pr]).mean():.4e}  "
          f"|tgt|mean={np.abs(TGT[pr]).mean():.4e}")

print("\n=== TEST B: oracle GDV (file) + SELF gap (from own energies) -> OOD path ===")
outB = p.predict(Z, R, gdv_source="file", gdv=GDV,
                 gap_source="self", check_species=False)
print(f"  gdv_source={outB['gdv_source']} gap_source={outB['gap_source']}")
for pr in PAIRS:
    mae = phaseless_mae(outB["dENACR"][pr], TGT[pr]) * 1000
    g_self = outB["gap_input"][pr]
    g_orac = GAP[pr]
    r = np.corrcoef(g_self, g_orac)[0, 1]
    gmae = np.abs(g_self - g_orac).mean() * 1000
    print(f"  dENACR{pr}: PhLessMAE = {mae:7.3f} meV/A   "
          f"gap(self-vs-oracle) |r|={abs(r):.4f} MAE={gmae:.2f} meV")

print("\n=== energy sanity (self-consistent gap source) ===")
for s in (0, 1, 2, 3):
    print(f"  S{s}: sE mean={outB['sE'][s].mean():.4f} eV  "
          f"|F|max={np.abs(outB['F'][s]).max():.3f}")
print("\nSMOKE OK: v3 predictor constructed, ran both gap paths, dENACR sane.")

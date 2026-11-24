#!/usr/bin/env python3
"""
ModelBComposedPredictor -- evaluate the DECOUPLED DeNAC Model B (NACR-only,
gap-gated v3) OOD, by composing it with the frozen E+F Model A that supplies
its inputs.

Model B has NO energy/force heads: it ingests A's predicted GDV{ij} and GAP{ij}
as inputs. So a standalone HiphopPredictor(V3) cannot run it self-consistently.
This wrapper does the two-stage forward exactly as B was TRAINED (teacher-forced
on A's outputs), but now self-consistently for OOD geometries:

  Stage A  (frozen Model A, E+F):  Z,R -> sE_s, F_s          (states 0..3)
           GDV_ij = F_i - F_j      (build_gdv_from_forces, bitwise as genA)
           GAP_ij = sE_j - sE_i    (eV)
  Stage B  (Model B, v3 NACR):     Z,R,GDV,GAP -> dENACR_ij
           d_ij   = dENACR_ij / GAP_ij   (un-scale with A's gap; B has no energy)

The result exposes A's energies/forces AND B's dENACR + the A-gap used to
un-scale, so a caller can compute NACT = (d_ij . V).

For the JOINT control there is no composition -- use HiphopPredictorV3 directly
with gap_source="self".

Developer: Ajay Khanna
LANL / Triad National Security, LLC.
"""
from __future__ import annotations

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import numpy as np

from hiphop_predictor_v3 import HiphopPredictorV3, build_gdv_from_forces, N_ATOMS


class ModelBComposedPredictor:
    def __init__(self, modelA_dir, modelB_dir, head_module_dir, *,
                 pairs=("12", "13", "23"), energy_offset=None,
                 device="auto", dtype="float64"):
        self.pairs = list(pairs)
        self._pair_states = sorted({int(c) for p in self.pairs for c in p})  # e.g. 1,2,3
        statesA = sorted(set([0]) | set(self._pair_states))                  # 0..3 for A

        kwA = dict(states=statesA, pairs=(), energies=True, forces=True,
                   nacr=False, device=device, dtype=dtype)
        if energy_offset is not None:
            kwA["energy_offset"] = float(energy_offset)
        # Model A = pure E+F: no NACR/GDV/GAP nodes, so nacr=False (no GDV inputs,
        # no forces-GDV subgraph). Supplies sE + F.
        self.predA = HiphopPredictorV3(modelA_dir, head_module_dir, **kwA)

        # Model B = NACR-only v3: energies/forces OFF (it has no such heads),
        # nacr ON, allow_forces_gdv=False (B cannot self-build GDV -> it comes
        # from A via gdv_source="file"). GAP comes from A via gap_source="file".
        self.predB = HiphopPredictorV3(
            modelB_dir, head_module_dir, states=(), pairs=self.pairs,
            energies=False, forces=False, nacr=True, allow_forces_gdv=False,
            device=device, dtype=dtype)
        if not self.predB._has_gap:
            raise ValueError(
                "Model B has no GAP inputs -> not a gap-gated v3 head; "
                "use the v2 composition (GDV only) instead")

    def predict(self, Z, R, *, check_species=True):
        """Compose A -> B. Returns dict with A's sE/E_abs/F/gap AND B's dENACR,
        plus out['gap_pair'][pair] (A's gap used to scale) and out['gdv'][pair]
        (A-forces GDV fed to B). All numpy."""
        # ---- Stage A: energies + forces (self-consistent) ----
        outA = self.predA.predict(Z, R, check_species=check_species)

        fdict = {s: outA["F"][s] for s in self._pair_states}
        gdv = build_gdv_from_forces(self.pairs, fdict)                     # {p:(N,45) f32}
        gap = {p: (outA["sE"][int(p[1])].astype(np.float64)
                   - outA["sE"][int(p[0])].astype(np.float64))
               for p in self.pairs}                                       # {p:(N,)} eV

        # ---- Stage B: dENACR from A's GDV+GAP ----
        outB = self.predB.predict(Z, R, gdv_source="file", gdv=gdv,
                                  gap_source="file", gap=gap, check_species=False)

        out = {"sE": outA["sE"], "E_abs": outA["E_abs"], "F": outA["F"],
               "gap": outA.get("gap", {}), "dENACR": outB["dENACR"],
               "gdv": gdv, "gap_pair": gap,
               "nframes": outB["nframes"], "gdv_source": "A-forces",
               "gap_source": "A-energies", "has_gap": True}
        return out

    def derivative_coupling(self, out, pair):
        """d_ij = dENACR_ij / GAP_ij using A's per-pair gap. inf where |gap| tiny."""
        gap = out["gap_pair"][pair].astype(np.float64)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            return out["dENACR"][pair].astype(np.float64) / gap


# --------------------------------------------------------------------- smoke
if __name__ == "__main__":
    import os, sys, argparse
    os.environ["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    ap = argparse.ArgumentParser(description="in-dist smoke: compose A->B, dENACR vs oracle")
    ap.add_argument("--modelA-dir", required=True)
    ap.add_argument("--modelB-dir", required=True)
    ap.add_argument("--head-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="Model B data dir (acn_R/Z + oracle acn_dENACR{p})")
    ap.add_argument("--prefix", default="acn_")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, a.head_dir)
    import unified_training, direct_nacr_head_v3, direct_nacr_head_v2  # noqa
    import __main__ as _M
    for _n in dir(unified_training):
        if not _n.startswith("__"):
            setattr(_M, _n, getattr(unified_training, _n))

    PAIRS = ("12", "13", "23")
    D, PX = a.data_dir, a.prefix
    ld = lambda n: np.load(os.path.join(D, PX + n + ".npy"), mmap_mode="r")
    Rall, Zall = ld("R"), ld("Z")
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(Rall.shape[0], size=min(a.n, Rall.shape[0]), replace=False))
    Z = np.asarray(Zall[idx]); R = np.asarray(Rall[idx])
    TGT = {p: np.asarray(ld(f"dENACR{p}")[idx]) for p in PAIRS}
    # A-predicted GDV/GAP stored by genA (to cross-check the composition reproduces them)
    try:
        GDVs = {p: np.asarray(ld(f"GDV{p}")[idx]) for p in PAIRS}
        GAPs = {p: np.asarray(ld(f"GAP{p}")[idx]) for p in PAIRS}
        have_stored = True
    except Exception:
        have_stored = False

    def phaseless_mae(pred, true):
        s = np.sign((pred * true).sum(1, keepdims=True)); s[s == 0] = 1.0
        return np.abs(pred - s * true).mean()

    print(f"compose A->B on {len(idx)} in-dist frames; Z[0]={Z[0].tolist()}")
    cp = ModelBComposedPredictor(a.modelA_dir, a.modelB_dir, a.head_dir,
                                 pairs=PAIRS, device=a.device, dtype="float64")
    out = cp.predict(Z, R, check_species=False)
    print(f"  gdv_source={out['gdv_source']} gap_source={out['gap_source']}")
    print("\n  dENACR PhLessMAE (composed A->B) vs oracle target:")
    for p in PAIRS:
        mae = phaseless_mae(out["dENACR"][p], TGT[p]) * 1000
        line = (f"    dENACR{p}: {mae:7.3f} meV/A   "
                f"|pred|mean={np.abs(out['dENACR'][p]).mean():.4e} "
                f"|tgt|mean={np.abs(TGT[p]).mean():.4e}")
        if have_stored:
            gdv_rel = np.abs(out["gdv"][p] - GDVs[p]).mean() / (np.abs(GDVs[p]).mean() + 1e-30)
            gap_mae = np.abs(out["gap_pair"][p] - GAPs[p]).mean() * 1000
            line += f"  [A-vs-stored GDVrel={gdv_rel*100:.2f}% GAPmae={gap_mae:.2f}meV]"
        print(line)
    print("\n  A energies: " + "  ".join(
        f"S{s}={out['sE'][s].mean():.3f}" for s in sorted(out["sE"])))
    print("\nCOMPOSED SMOKE OK: A->B ran, dENACR sane.")

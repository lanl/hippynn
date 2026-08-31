#!/usr/bin/env python3
"""
HiphopPredictor -- load the enol-AcAc E+F+NACR HIPPYNN model ONCE, then predict
E / F / dENACR per geometry (or per MD step).

This is the interface a dynamics engine (e.g. PySEQM FSSH) should use. The CLI
scripts (`multistate_ef*_inference_hiphop.py`) reload the 736 MB model on every
invocation and must NOT be called per step. This class loads once and keeps the
pruned graphs resident.

No PySEQM dependency: `predict()` takes Z and R arrays directly, so it can be
imported into any driver. (The CLI scripts import `seqm.api.read_xyz`; this does
not.)

Design
------
`states` and `pairs` are fixed for the lifetime of the object (they are fixed for
a trajectory), so the pruned graphs are built once in `__init__`:
  * `_graph_full` : [Z, R, GDV_ij...] -> [sE_n..., F_n..., dENACR_ij...]
  * `_graph_force`: [Z, R] -> [F_s...]  (only the states GDV needs; built only
                    when gdv_source can be "forces")

`predict()` then just runs them. Two GDV sources:
  * "file"   : you pass `gdv={pair: (N,45) array}` (oracle / reference GDV)
  * "forces" : the model's OWN predicted forces are used to build
               GDV_ij = F_i - F_j. This is a TRAIN/INFERENCE MISMATCH (the head
               was teacher-forced on oracle GDV) and is NOT validated -- it is
               the hook for closing the loop inside dynamics.

Conventions (see the package README for the full discussion)
------------------------------------------------------------
  * sE_n = E_n - E0_min          (one shared shift; `energy_offset` = -1391.422970121047 eV)
    E_abs = sE_n + offset,  gap = sE_n - sE_0
  * GDV_ij = F_i - F_j           (physical forces, no /2, no norm; f64 subtract then f32 cast)
  * pairs are 1-indexed EXCITED states: "12" = S1-S2, needs F1..F5, never F0
  * dENACR is the native scaled coupling (E_j-E_i)*d_ij; its SIGN is arbitrary
    per frame (PHASESIGN_CORRECTED_SCALED_NACR=False). To recover the bare
    derivative coupling: d_ij = dENACR_ij / (E_j - E_i), using the model's own gap.

Example (per-step, self-consistent GDV -- the FSSH use case)
------------------------------------------------------------
    p = HiphopPredictor(model_dir="model", head_module_dir="src",
                        states=(1,2,3), pairs=("12","13","23"), device="cuda")
    out = p.predict(Z, R, gdv_source="forces")
    #   out["sE"][3]        shifted energy of S3        (nframes,)
    #   out["E_abs"][3]     absolute energy of S3       (nframes,)  float64
    #   out["F"][3]         force on S3                 (nframes, 15, 3)
    #   out["dENACR"]["13"] scaled NACR S1-S3           (nframes, 45)

Developer: Ajay Khanna <akhanna2@ucmerced.edu>
LANL / Triad National Security, LLC.  LA-UR-XX-XXXXX
"""

from __future__ import annotations

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import os
from pathlib import Path

import numpy as np

DEFAULT_OFFSET = -1391.422970121047  # eV, enol AcAc
ALL_PAIRS = ("12", "13", "14", "15", "23", "24", "25", "34", "35", "45")
EXPECTED_SPECIES = (8, 8, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1)
N_ATOMS = 15
MAX_STATE = 5


def build_gdv_from_forces(pairs, forces):
    """GDV_ij = F_i - F_j for 1-indexed excited-state pairs.

    forces: {state -> (N, N_ATOMS, 3) or (N, 3*N_ATOMS)} for the needed states.
    Subtracts in float64 then casts to float32 -- BITWISE-identical to
    `generate_gdv_data.py`. Returns {pair: (N, 45) float32}.
    """
    need = sorted({int(c) for p in pairs for c in p})
    if need and need[0] < 1:
        raise ValueError("GDV pairs are 1-indexed excited states; state 0 is never used")
    F = {s: np.asarray(forces[s], np.float64).reshape(-1, 3 * N_ATOMS) for s in need}
    return {p: (F[int(p[0])] - F[int(p[1])]).astype(np.float32, copy=False) for p in pairs}


class HiphopPredictor:
    def __init__(self, model_dir, head_module_dir, *, states=range(MAX_STATE + 1),
                 pairs=ALL_PAIRS, energies=True, forces=True, nacr=True,
                 allow_forces_gdv=True, energy_offset=DEFAULT_OFFSET,
                 device="auto", dtype="float64"):
        self.states = sorted({int(s) for s in states})
        self.pairs = list(pairs)
        self.energies, self.forces, self.nacr = bool(energies), bool(forces), bool(nacr)
        self.offset = float(energy_offset)
        for s in self.states:
            if not 0 <= s <= MAX_STATE:
                raise ValueError(f"state {s} out of range 0..{MAX_STATE}")
        for p in self.pairs:
            if p not in ALL_PAIRS:
                raise ValueError(f"unknown pair {p!r}; valid: {ALL_PAIRS}")

        # hippynn reads this at import; triton has no float64 path.
        if dtype == "float64":
            os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")

        # The custom head class must be importable as a TOP-LEVEL module BEFORE
        # the unpickle (experiment_structure.pt references it by module path).
        import sys
        sys.path.insert(0, str(Path(head_module_dir).resolve()))
        import direct_nacr_head_v2  # noqa: F401

        import torch
        from hippynn.experiment.serialization import load_model_from_cwd
        from hippynn.graphs import GraphModule
        from hippynn.tools import active_directory
        self._torch = torch
        self._dtype = torch.float64 if dtype == "float64" else torch.float32
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        with active_directory(str(model_dir)):
            self._model = load_model_from_cwd(map_location="cpu")
        self._model.eval()

        self._in = self._model.input_nodes
        self._out = self._model.nodes_to_compute

        # Build the graphs once.
        outs, self._full_keys = [], []
        for s in self.states:
            if self.energies:
                outs.append(self._node(self._out, f"sE{s}")); self._full_keys.append(f"sE{s}")
        for s in self.states:
            if self.forces:
                outs.append(self._node(self._out, f"F{s}")); self._full_keys.append(f"F{s}")
        gdv_needed = self.nacr and len(self.pairs) > 0
        full_inputs = [self._node(self._in, "Z"), self._node(self._in, "R")]
        self._full_in_names = ["Z", "R"]
        if gdv_needed:
            for p in self.pairs:
                full_inputs.append(self._node(self._in, f"GDV{p}"))
                self._full_in_names.append(f"GDV{p}")
            for p in self.pairs:
                outs.append(self._node(self._out, f"dENACR{p}")); self._full_keys.append(f"dENACR{p}")
        if not outs:
            raise ValueError("nothing to compute (energies, forces and nacr all off)")

        # GraphModule raises if an output needs an unspecified input -> the prune
        # is self-validating.
        self._graph_full = GraphModule(full_inputs, outs).to(self.device, self._dtype)
        self._graph_full.eval()

        # Forces-only graph for self-consistent GDV (F for the states GDV needs).
        self._graph_force = None
        self._force_states = []
        if gdv_needed and allow_forces_gdv:
            self._force_states = sorted({int(c) for p in self.pairs for c in p})  # 1..5
            fouts = [self._node(self._out, f"F{s}") for s in self._force_states]
            self._graph_force = GraphModule(
                [self._node(self._in, "Z"), self._node(self._in, "R")], fouts
            ).to(self.device, self._dtype)
            self._graph_force.eval()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _node(nodes, db_name):
        hits = [n for n in nodes if getattr(n, "db_name", None) == db_name]
        if len(hits) != 1:
            raise ValueError(f"expected 1 node db_name={db_name!r}, found {len(hits)}")
        return hits[0]

    def _tensors(self, Z, R, nframes):
        t = self._torch
        Zt = t.as_tensor(np.asarray(Z).reshape(nframes, N_ATOMS), dtype=t.int64, device=self.device)
        # fresh tensor: GraphModule.forward calls requires_grad_() IN PLACE
        Rt = t.as_tensor(np.asarray(R).reshape(nframes, N_ATOMS, 3),
                         dtype=self._dtype, device=self.device).clone().requires_grad_(True)
        return Zt, Rt

    # ------------------------------------------------------------------- predict
    def predict(self, Z, R, *, gdv_source="forces", gdv=None, check_species=True):
        """Predict for one or more geometries.

        Z : (n_atoms,) or (N, n_atoms)   atomic numbers
        R : (n_atoms, 3) or (N, n_atoms, 3)   coordinates, Angstrom
        gdv_source : "forces" (self-consistent, NOT validated) or "file" (oracle)
        gdv : {pair: (N, 45)} required when gdv_source == "file"

        Returns dict:
          out["sE"][s], out["E_abs"][s] (float64), out["gap"][s], out["F"][s],
          out["dENACR"][pair]
        All numpy; energies (N,), forces (N, n_atoms, 3), dENACR (N, 45).
        """
        t = self._torch
        Z = np.asarray(Z)
        R = np.asarray(R)
        if Z.ndim == 1:
            Z = Z[None, :]
            R = R[None, :, :]
        nframes = Z.shape[0]
        if check_species:
            for row in Z:
                got = [int(z) for z in row if int(z) > 0]
                if got != list(EXPECTED_SPECIES):
                    raise ValueError(f"species order {got} != training order {list(EXPECTED_SPECIES)}")

        # ---- GDV ----
        gdv_arrays = {}
        if self.nacr and self.pairs:
            if gdv_source == "file":
                if gdv is None:
                    raise ValueError('gdv_source="file" needs gdv={pair:(N,45)}')
                gdv_arrays = {p: np.asarray(gdv[p]).reshape(nframes, 3 * N_ATOMS) for p in self.pairs}
            elif gdv_source == "forces":
                if self._graph_force is None:
                    raise ValueError("this predictor was built with allow_forces_gdv=False")
                Zt, Rt = self._tensors(Z, R, nframes)
                with t.enable_grad():
                    fouts = self._graph_force(Zt, Rt)
                fdict = {s: o.detach().to("cpu").numpy().reshape(nframes, N_ATOMS, 3)
                         for s, o in zip(self._force_states, fouts)}
                gdv_arrays = build_gdv_from_forces(self.pairs, fdict)
            else:
                raise ValueError(f"gdv_source must be 'file' or 'forces', got {gdv_source!r}")

        # ---- full graph ----
        Zt, Rt = self._tensors(Z, R, nframes)
        pool = {"Z": Zt, "R": Rt}
        for p in self.pairs if (self.nacr and self.pairs) else []:
            pool[f"GDV{p}"] = t.as_tensor(gdv_arrays[p], dtype=self._dtype, device=self.device)
        with t.enable_grad():   # NEVER torch.no_grad(): F_n is a GradientNode
            outs = self._graph_full(*(pool[n] for n in self._full_in_names))
        raw = {k: o.detach().to("cpu").numpy() for k, o in zip(self._full_keys, outs)}

        # ---- assemble ----
        out = {"sE": {}, "E_abs": {}, "gap": {}, "F": {}, "dENACR": {}}
        if self.energies:
            for s in self.states:
                se = raw[f"sE{s}"].reshape(nframes)
                out["sE"][s] = se
                out["E_abs"][s] = se.astype(np.float64) + self.offset   # float64 accumulate
            if 0 in self.states:
                for s in self.states:
                    out["gap"][s] = out["sE"][s].astype(np.float64) - out["sE"][0].astype(np.float64)
        if self.forces:
            for s in self.states:
                out["F"][s] = raw[f"F{s}"].reshape(nframes, N_ATOMS, 3)
        if self.nacr:
            for p in self.pairs:
                out["dENACR"][p] = raw[f"dENACR{p}"].reshape(nframes, 3 * N_ATOMS)
        out["nframes"] = nframes
        out["gdv_source"] = gdv_source if (self.nacr and self.pairs) else None
        return out

    def derivative_coupling(self, out, pair):
        """Recover d_ij = dENACR_ij / (E_j - E_i) from a predict() result.

        Requires the pair's two states in `self.states` and energies on. Warns
        (returns inf where |gap| is tiny) near conical intersections.
        """
        i, j = int(pair[0]), int(pair[1])
        if i not in self.states or j not in self.states:
            raise ValueError(f"need states {i} and {j} in self.states={self.states}")
        gap = (out["sE"][j].astype(np.float64) - out["sE"][i].astype(np.float64))[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            return out["dENACR"][pair].astype(np.float64) / gap


# --------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="smoke-test the predictor on a golden .npz of Z,R")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--head-module-dir", required=True)
    ap.add_argument("--npz", help="npz with 'Z' (N,15) and 'R' (N,15,3); else a random geometry")
    ap.add_argument("--gdv-source", default="forces", choices=("forces", "file"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    p = HiphopPredictor(a.model_dir, a.head_module_dir, states=(0, 1, 2, 3),
                        pairs=("12", "13", "23"), device=a.device)
    if a.npz:
        d = np.load(a.npz)
        Z, R = d["Z"], d["R"]
    else:
        Z = np.array(EXPECTED_SPECIES)
        R = np.random.default_rng(0).normal(0, 1.5, (N_ATOMS, 3))
    out = p.predict(Z, R, gdv_source=a.gdv_source)
    print("nframes:", out["nframes"], "gdv_source:", out["gdv_source"])
    for s in (0, 1, 2, 3):
        print(f"  S{s}: sE={out['sE'][s][0]:.6f}  E_abs={out['E_abs'][s][0]:.6f}  "
              f"|F|max={np.abs(out['F'][s][0]).max():.4f}")
    for pr in ("12", "13", "23"):
        print(f"  dENACR{pr}: |.|max={np.abs(out['dENACR'][pr][0]).max():.4f}")
    print("OK")

#!/usr/bin/env python3
"""
HiphopPredictorV3 -- v3-capable extension of HiphopPredictor.

Adds support for GAP-GATED scaled-NACR heads (`direct_vector_v3`): the v3 head
carries a SECOND per-pair InputNode `GAP{ij}` (= sE_j - sE_i, eV) alongside the
`GDV{ij}` input. The v2 predictor's `_graph_full` wires only [Z, R, GDV{p}...],
so building it against a v3 model raises ("output needs an unspecified input").
This class detects the GAP inputs and wires them, and supplies their values by
one of:

  gap_source="self"  : gap_ij = sE_j - sE_i from the model's OWN predicted
                        energies (self-consistent; the JOINT OOD path).
  gap_source="file"  : you pass gap={pair: (N,)} (e.g. frozen Model A's gaps ->
                        the DECOUPLED Model B composition path, or oracle gaps).

Backward compatible: a v2 model (no GAP inputs) behaves EXACTLY as the original
HiphopPredictor -- gap_source is ignored and no GAP graph is built.

Everything else (GDV file/forces, energy/force/dENACR outputs, conventions) is
inherited verbatim from the v2 design; see hiphop_predictor.py.

Developer: Ajay Khanna <akhanna2@ucmerced.edu>
LANL / Triad National Security, LLC.
"""

from __future__ import annotations

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


class HiphopPredictorV3:
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

        # The custom head classes must be importable as TOP-LEVEL modules BEFORE
        # the unpickle (experiment_structure.pt references them by module path).
        # v3 pickles reference direct_nacr_head_v3 (which itself uses v2); import
        # all three that exist so both v2 and v3 checkpoints load.
        import sys
        sys.path.insert(0, str(Path(head_module_dir).resolve()))
        for _mod in ("direct_nacr_head", "direct_nacr_head_v2", "direct_nacr_head_v3"):
            try:
                __import__(_mod)
            except Exception:
                pass  # v1/v3 may be absent for pure-v2 checkpoints; that's fine

        import torch
        from hippynn.experiment.serialization import load_model_from_cwd
        from hippynn.graphs import GraphModule
        from hippynn.tools import active_directory
        self._torch = torch
        self._GraphModule = GraphModule
        self._dtype = torch.float64 if dtype == "float64" else torch.float32
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        with active_directory(str(model_dir)):
            self._model = load_model_from_cwd(map_location="cpu")
        self._model.eval()

        self._in = self._model.input_nodes
        self._out = self._model.nodes_to_compute

        gdv_needed = self.nacr and len(self.pairs) > 0

        # ---- detect GAP inputs (v3 gap-gated head) ----
        # A v3 model exposes one GAP{ij} InputNode per NACR pair. If present, the
        # dENACR outputs depend on it and it MUST be wired into _graph_full.
        gap_hits = [p for p in self.pairs if self._find(self._in, f"GAP{p}") is not None]
        if gap_needed := (gdv_needed and len(gap_hits) > 0):
            if len(gap_hits) != len(self.pairs):
                missing = [p for p in self.pairs if p not in gap_hits]
                raise ValueError(
                    f"model has GAP inputs for {gap_hits} but not {missing}; "
                    "inconsistent v3 head (expected a GAP input per NACR pair)")
        self._has_gap = bool(gap_needed)

        # ---- build the full graph ----
        outs, self._full_keys = [], []
        for s in self.states:
            if self.energies:
                outs.append(self._node(self._out, f"sE{s}")); self._full_keys.append(f"sE{s}")
        for s in self.states:
            if self.forces:
                outs.append(self._node(self._out, f"F{s}")); self._full_keys.append(f"F{s}")
        full_inputs = [self._node(self._in, "Z"), self._node(self._in, "R")]
        self._full_in_names = ["Z", "R"]
        if gdv_needed:
            for p in self.pairs:
                full_inputs.append(self._node(self._in, f"GDV{p}"))
                self._full_in_names.append(f"GDV{p}")
            if self._has_gap:
                for p in self.pairs:
                    full_inputs.append(self._node(self._in, f"GAP{p}"))
                    self._full_in_names.append(f"GAP{p}")
            for p in self.pairs:
                outs.append(self._node(self._out, f"dENACR{p}")); self._full_keys.append(f"dENACR{p}")
        if not outs:
            raise ValueError("nothing to compute (energies, forces and nacr all off)")

        # GraphModule raises if an output needs an unspecified input -> the prune
        # is self-validating (this is what caught the missing GAP input for v3).
        self._graph_full = GraphModule(full_inputs, outs).to(self.device, self._dtype)
        self._graph_full.eval()

        # ---- pair states (used by both forces-GDV and self-gap subgraphs) ----
        self._pair_states = sorted({int(c) for p in self.pairs for c in p}) if gdv_needed else []

        # Forces-only graph for self-consistent GDV (F for the states GDV needs).
        self._graph_force = None
        self._force_states = []
        if gdv_needed and allow_forces_gdv:
            self._force_states = list(self._pair_states)  # 1..5
            fouts = [self._node(self._out, f"F{s}") for s in self._force_states]
            self._graph_force = GraphModule(
                [self._node(self._in, "Z"), self._node(self._in, "R")], fouts
            ).to(self.device, self._dtype)
            self._graph_force.eval()

        # Energy-only graph for self-consistent GAP (sE for the pair states). Only
        # buildable when the model HAS energy heads (joint); Model B (NACR-only)
        # has no sE -> gap_source="self" is unavailable there (use "file").
        self._graph_energy = None
        self._energy_states = []
        if self._has_gap and self.energies:
            self._energy_states = list(self._pair_states)
            eouts = [self._node(self._out, f"sE{s}") for s in self._energy_states]
            self._graph_energy = GraphModule(
                [self._node(self._in, "Z"), self._node(self._in, "R")], eouts
            ).to(self.device, self._dtype)
            self._graph_energy.eval()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _find(nodes, db_name):
        hits = [n for n in nodes if getattr(n, "db_name", None) == db_name]
        if len(hits) > 1:
            raise ValueError(f"expected <=1 node db_name={db_name!r}, found {len(hits)}")
        return hits[0] if hits else None

    def _node(self, nodes, db_name):
        n = self._find(nodes, db_name)
        if n is None:
            raise ValueError(f"expected 1 node db_name={db_name!r}, found 0")
        return n

    def _tensors(self, Z, R, nframes):
        t = self._torch
        Zt = t.as_tensor(np.asarray(Z).reshape(nframes, N_ATOMS), dtype=t.int64, device=self.device)
        # fresh tensor: GraphModule.forward calls requires_grad_() IN PLACE
        Rt = t.as_tensor(np.asarray(R).reshape(nframes, N_ATOMS, 3),
                         dtype=self._dtype, device=self.device).clone().requires_grad_(True)
        return Zt, Rt

    # ------------------------------------------------------------------- predict
    def predict(self, Z, R, *, gdv_source="forces", gdv=None,
                gap_source="self", gap=None, check_species=True):
        """Predict for one or more geometries.

        gdv_source : "forces" (self-consistent) or "file" (gdv={pair:(N,45)}).
        gap_source : (v3 models only) "self" (gap from own predicted energies;
                     needs energy heads) or "file" (gap={pair:(N,)}). Ignored for
                     v2 models.

        Returns dict: out["sE"][s], out["E_abs"][s], out["gap"][s], out["F"][s],
        out["dENACR"][pair].  All numpy.
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

        # ---- GAP (v3 only) ----
        gap_arrays = {}
        if self._has_gap:
            if gap_source == "file":
                if gap is None:
                    raise ValueError('gap_source="file" needs gap={pair:(N,)}')
                gap_arrays = {p: np.asarray(gap[p], np.float64).reshape(nframes) for p in self.pairs}
            elif gap_source == "self":
                if self._graph_energy is None:
                    raise ValueError(
                        'gap_source="self" needs energy heads; this model has none '
                        '(e.g. decoupled Model B) -> supply gap_source="file"')
                Zt, Rt = self._tensors(Z, R, nframes)
                with t.enable_grad():
                    eouts = self._graph_energy(Zt, Rt)
                seg = {s: o.detach().to("cpu").numpy().reshape(nframes).astype(np.float64)
                       for s, o in zip(self._energy_states, eouts)}
                gap_arrays = {p: seg[int(p[1])] - seg[int(p[0])] for p in self.pairs}
            else:
                raise ValueError(f"gap_source must be 'self' or 'file', got {gap_source!r}")

        # ---- full graph ----
        Zt, Rt = self._tensors(Z, R, nframes)
        pool = {"Z": Zt, "R": Rt}
        if self.nacr and self.pairs:
            for p in self.pairs:
                pool[f"GDV{p}"] = t.as_tensor(gdv_arrays[p], dtype=self._dtype, device=self.device)
            if self._has_gap:
                for p in self.pairs:
                    pool[f"GAP{p}"] = t.as_tensor(gap_arrays[p], dtype=self._dtype, device=self.device)
        with t.enable_grad():   # NEVER torch.no_grad(): F_n is a GradientNode
            outs = self._graph_full(*(pool[n] for n in self._full_in_names))
        raw = {k: o.detach().to("cpu").numpy() for k, o in zip(self._full_keys, outs)}

        # ---- assemble ----
        out = {"sE": {}, "E_abs": {}, "gap": {}, "F": {}, "dENACR": {}}
        if self.energies:
            for s in self.states:
                se = raw[f"sE{s}"].reshape(nframes)
                out["sE"][s] = se
                out["E_abs"][s] = se.astype(np.float64) + self.offset
            if 0 in self.states:
                for s in self.states:
                    out["gap"][s] = out["sE"][s].astype(np.float64) - out["sE"][0].astype(np.float64)
        if self.forces:
            for s in self.states:
                out["F"][s] = raw[f"F{s}"].reshape(nframes, N_ATOMS, 3)
        if self.nacr:
            for p in self.pairs:
                out["dENACR"][p] = raw[f"dENACR{p}"].reshape(nframes, 3 * N_ATOMS)
            if gap_arrays:
                out["gap_input"] = gap_arrays
        out["nframes"] = nframes
        out["gdv_source"] = gdv_source if (self.nacr and self.pairs) else None
        out["gap_source"] = (gap_source if self._has_gap else None)
        out["has_gap"] = self._has_gap
        return out

    def derivative_coupling(self, out, pair):
        """Recover d_ij = dENACR_ij / (E_j - E_i) using the model's own gap."""
        i, j = int(pair[0]), int(pair[1])
        if i not in self.states or j not in self.states:
            raise ValueError(f"need states {i} and {j} in self.states={self.states}")
        gap = (out["sE"][j].astype(np.float64) - out["sE"][i].astype(np.float64))[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            return out["dENACR"][pair].astype(np.float64) / gap

"""Composed A+B (frozen E+F Model A + decoupled NACR Model B) adapter for PySEQM FSSH.

Drives FSSH with the DECOUPLED DeNAC stack instead of the joint model:
  Stage A (frozen Model A): Z,R -> sE, F        (energies + forces for propagation)
  Stage B (Model B, v3):    Z,R,GDV(=Fi-Fj),GAP(=sEj-sEi) -> dENACR   (coupling)
  NAC vector d_ij = dENACR_ij / GAP_ij  (A's gap), then NACT = d.v  (hop driver)

`ModelBComposedPredictor.predict()` returns the SAME dict keys the joint predictor
does (sE, E_abs, F, dENACR), so `HiphopSurfaceHopping._compute_electronic_structure`
is inherited UNCHANGED -- we only swap which predictor object it calls.

Model B is NACR-only (no E/F heads), so we must NOT let the parent build a
HiphopPredictorV3 on it. This subclass replicates the parent __init__ verbatim
except the final predictor line.

Developer: Ajay Khanna <akhanna2@ucmerced.edu>
LANL / Triad National Security, LLC.
"""
from __future__ import annotations

from hippynn_fssh import HiphopSurfaceHopping, HANDOFF, pairs
from seqm.api import SurfaceHoppingDynamics
from modelb_composed_predictor import ModelBComposedPredictor


class _ComposedAdapter:
    """Wrap ModelBComposedPredictor to the HiphopPredictorV3.predict signature used
    in _compute_electronic_structure (it passes gdv_source='forces'). Model A
    self-builds GDV from its own forces internally, so gdv_source/gap_source are
    irrelevant here and are accepted-and-ignored."""

    def __init__(self, modelA_dir, modelB_dir, head_dir, *, pairs, device, dtype="float64"):
        self._cp = ModelBComposedPredictor(
            modelA_dir, modelB_dir, head_dir, pairs=tuple(pairs), device=device, dtype=dtype
        )

    def predict(self, Z, R, gdv_source=None, gap_source=None, check_species=True, **_kw):
        return self._cp.predict(Z, R, check_species=check_species)


class ComposedSurfaceHopping(HiphopSurfaceHopping):
    """FSSH driven by composed A->B. Same electronic-structure / hop machinery as
    HiphopSurfaceHopping; only the predictor differs."""

    def __init__(
        self,
        modelA_dir,
        modelB_dir,
        labels,
        initial_state,
        timestep,
        temperature,
        output,
        device,
        friction=None,
        nac_gap_floor=1e-8,
    ):
        # --- replicate HiphopSurfaceHopping.__init__ EXCEPT the predictor line ---
        self.labels = tuple(x for x in labels if x)
        self.model_states, self.model_pairs = tuple(labels), pairs(labels)
        if initial_state not in self.labels:
            raise ValueError(f"initial state S{initial_state} is outside {self.labels}")
        self.gap_floor, self.previous_nac = float(nac_gap_floor), None
        params = {
            "method": "AM1",
            "elements": [1, 6, 8],
            "scf_eps": 1e-8,
            "scf_converger": [1],
            "analytical_gradient": [True],
            "excited_states": {"n_states": len(self.labels), "method": "cis"},
            "nonadiabatic": {"tdc_method": "nac_dot_v", "decohere_on_hop": True, "detect_crossings": False},
        }
        SurfaceHoppingDynamics.__init__(
            self,
            params,
            timestep=timestep,
            Temp=temperature,
            initial_state=self.labels.index(initial_state) + 1,
            damp=None if friction is None else 1 / friction,
            output=output,
        )
        # --- the one substitution: composed A->B predictor instead of the joint ---
        self.predictor = _ComposedAdapter(
            modelA_dir,
            modelB_dir,
            HANDOFF / "src",
            pairs=self.model_pairs,
            device=device,
            dtype="float64",
        )

"""Composed A+B (frozen E+F Model A + decoupled NACR Model B) adapter for PySEQM FSSH.

Scientific purpose
------------------
Drive FSSH with the DECOUPLED "DeNAC" stack instead of a single joint model:
  Stage A (frozen Model A): ``Z, R -> sE, F``        (energies + forces for propagation)
  Stage B (Model B, v3):    ``Z, R, GDV(=Fi-Fj), GAP(=sEj-sEi) -> dENACR``   (coupling)
  NAC vector ``d_ij = dENACR_ij / GAP_ij`` (A's gap), then ``NACT = d . v`` (the hop driver).

``ModelBComposedPredictor.predict()`` returns the SAME dict keys the joint predictor does
(``sE``, ``E_abs``, ``F``, ``dENACR``), so :meth:`HiphopSurfaceHopping._compute_electronic_structure`
is inherited UNCHANGED; only the predictor object is swapped. Because Model B is NACR-only (no E/F
heads), the parent must NOT build a ``HiphopPredictorV3`` on it, so :class:`ComposedSurfaceHopping`
replicates the parent ``__init__`` verbatim except for the final predictor line.

Dependencies
------------
Requires ``modelb_composed_predictor.ModelBComposedPredictor`` on ``sys.path`` (the path setup and
the ``HANDOFF`` src directory are inherited from :mod:`hippynn_fssh`, imported here).
"""

from __future__ import annotations

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

from hippynn_fssh import HiphopSurfaceHopping, HANDOFF, pairs
from seqm.api import SurfaceHoppingDynamics
from modelb_composed_predictor import ModelBComposedPredictor


class _ComposedAdapter:
    """Adapt ``ModelBComposedPredictor`` to the ``HiphopPredictorV3.predict`` signature.

    :meth:`HiphopSurfaceHopping._compute_electronic_structure` calls ``predict(..., gdv_source='forces')``.
    Model A self-builds the GDV from its own forces internally, so ``gdv_source``/``gap_source`` are
    irrelevant here and are accepted-and-ignored.
    """

    def __init__(self, modelA_dir, modelB_dir, head_dir, *, pairs, device, dtype="float64"):
        self._cp = ModelBComposedPredictor(
            modelA_dir, modelB_dir, head_dir, pairs=tuple(pairs), device=device, dtype=dtype
        )

    def predict(self, Z, R, gdv_source=None, gap_source=None, check_species=True, **_kw):
        """Return the composed A+B electronic-structure dict (``gdv_source``/``gap_source`` ignored)."""
        return self._cp.predict(Z, R, check_species=check_species)


class ComposedSurfaceHopping(HiphopSurfaceHopping):
    """FSSH driven by the composed A->B stack.

    Identical electronic-structure / hop machinery to :class:`HiphopSurfaceHopping`; only the
    predictor differs. The ``__init__`` intentionally mirrors the parent's body (so the PySEQM
    CIS/nonadiabatic parameters stay in lockstep) except for the final predictor assignment.
    """

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
        """Configure PySEQM exactly as the parent, then attach the composed A->B predictor."""
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

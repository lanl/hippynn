"""FSSH@MLIP engine: a HIPPYNN-driven electronic-structure adapter for PySEQM surface hopping.

Scientific purpose
------------------
Run fewest-switches surface hopping (FSSH) where the electronic-structure payload (per-state
energies, forces, and nonadiabatic coupling vectors) comes from a trained HIPPYNN MLIP instead of
a semiempirical CIS calculation. :class:`HiphopSurfaceHopping` subclasses PySEQM's
``SurfaceHoppingDynamics`` and overrides the electronic-structure hooks to query a HiphopNN
predictor each step, so the classical propagation, hopping, and decoherence machinery of PySEQM is
reused unchanged.

Key physics conventions (do not change; they are load-bearing and were hard-won):
- NAC vectors are ``nac_ij = dENACR_ij / (E_j - E_i)`` using the RAW gap, with NO floor and NO clamp
  (PySEQM ``seqm_functions/nac.py``). Flooring the gap (the old ``max(gap, 1e-8)``) inverted small/
  negative MLIP gaps, exploded the coupling, and froze hopping via amplitude NaNs.
- Time-derivative coupling uses ``tdc_method = "nac_dot_v"`` with ``decohere_on_hop = True`` (the
  MLIP has no CIS amplitudes, so ``hamiltonian_fd`` is infeasible for the MLIP path).
- ``_phase_match`` fixes the arbitrary per-state sign of the coupling vectors against the previous
  step, GPU-vectorized and unit-verified bit-identical to the former per-molecule loop (n=3,4,5).

Dependencies and layout assumptions (IMPORTANT)
-----------------------------------------------
This module is meant to run from the FSSH driver directory on the cluster, which must also contain:
- ``enol_ef_nacr_developer_handoff/src/`` with ``hiphop_predictor_v3.py`` (the predictor) and the
  training modules (``unified_training``, ``direct_nacr_head_v3``, ``direct_nacr_head_v2``);
- a repository ROOT two levels up (``parents[2]``) that is added to ``sys.path``.
Both are prepended to ``sys.path`` at import time (see below). The joint/pooled model pickle
references training symbols (e.g. ``DataDumper``) via ``__main__``, so ``unified_training``'s public
names are copied into ``__main__`` BEFORE any model is unpickled. When relocating this file, keep
that directory next to it (or edit the two path assumptions here and document the change).
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

# Path assumptions (documented above): repo ROOT two levels up, and the developer-handoff src dir.
ROOT = Path(__file__).resolve().parents[2]
HANDOFF = Path(__file__).resolve().parent / "enol_ef_nacr_developer_handoff"
sys.path[:0] = [str(ROOT), str(HANDOFF / "src")]

# The pooled/joint-model pickle carries DataDumper etc. referenced via __main__; inject
# unified_training's symbols there BEFORE the unpickle (mirrors phase2_ood_nact.py / smoke_v3_joint.py).
import unified_training, direct_nacr_head_v3, direct_nacr_head_v2  # noqa
import __main__ as _M
for _n in dir(unified_training):
    if not _n.startswith("__"):
        setattr(_M, _n, getattr(unified_training, _n))

from hiphop_predictor_v3 import HiphopPredictorV3  # v3-capable superset (auto-detects GAP node; runs v2 too)

from seqm.api import Constants, Molecule, SurfaceHoppingDynamics, read_xyz


def states(spec):
    """Parse a ``--states`` spec (e.g. ``"0,1,2,3"``) into a sorted tuple of state labels.

    Requires at least two EXCITED labels drawn from 1..5 (S0 may be included). Raises
    ``ValueError`` otherwise.
    """
    out = tuple(sorted({int(x) for x in str(spec).replace(" ", "").split(",") if x}))
    if any(x < 0 or x > 5 for x in out) or len([x for x in out if x]) < 2:
        raise ValueError("--states must contain at least two excited labels from 1..5 (optionally S0).")
    return out


def pairs(labels):
    """Return the ordered two-digit pair keys (e.g. ``"12"``) for every excited-state combination."""
    return tuple(f"{i}{j}" for i, j in itertools.combinations([x for x in labels if x], 2))


def molecule(xyz, ntraj, device):
    """Build a PySEQM ``Molecule`` replicated ``ntraj`` times from an .xyz file (AM1 constants)."""
    Z, R = read_xyz([str(Path(xyz).expanduser())], sort=False)
    Z = torch.as_tensor(Z, dtype=torch.long, device=device).repeat(ntraj, 1)
    R = torch.as_tensor(R, dtype=torch.float64, device=device).repeat(ntraj, 1, 1)
    return Molecule(
        Constants().to(device), {"method": "AM1", "scf_eps": 1.0, "scf_converger": [-1]}, R, Z
    ).to(device)


class HiphopSurfaceHopping(SurfaceHoppingDynamics):
    """FSSH that obtains E, F, and NAC vectors from a ``HiphopPredictor`` (single-model path).

    ``labels`` is the tuple of state labels handled by the model (e.g. ``(0, 1, 2, 3)``);
    ``self.labels`` drops S0 to give the excited states, while ``self.model_states`` keeps all
    labels and ``self.model_pairs`` are the two-digit pair keys. The class name and the overridden
    PySEQM hook signatures are load-bearing.
    """

    def __init__(
        self,
        model_dir,
        labels,
        initial_state,
        timestep,
        temperature,
        output,
        device,
        friction=None,
        nac_gap_floor=1e-8,
    ):
        """Configure the PySEQM CIS/nonadiabatic parameters and attach the HiphopNN predictor.

        ``nac_gap_floor`` is retained for API compatibility but the coupling uses the RAW gap
        (see module docstring); ``friction`` (if given) selects Langevin damping ``1/friction``.
        """
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
        super().__init__(
            params,
            timestep=timestep,
            Temp=temperature,
            initial_state=self.labels.index(initial_state) + 1,
            damp=None if friction is None else 1 / friction,
            output=output,
        )
        self.predictor = HiphopPredictorV3(
            model_dir,
            HANDOFF / "src",
            states=self.model_states,
            pairs=self.model_pairs,
            device=device,
            dtype="float64",
        )  # gap_source defaults to "self" (joint self-supplies gaps at inference)

    def _setup_states(self, mol):
        """Cache the state/pair bookkeeping (counts, 0-based and 1-based pair keys) for ``mol``."""
        self._nstates = len(self.labels)
        self._full_nac_pair_keys = list(itertools.combinations(range(self._nstates), 2))
        self._full_nac_pairs_1based = [(i + 1, j + 1) for i, j in self._full_nac_pair_keys]
        self._full_nac_state_i, self._full_nac_state_j = zip(*self._full_nac_pair_keys)
        self._ensure_active_states(mol.species.shape[0], mol.coordinates.device)

    def initialize(self, mol, remove_com=None, learned_parameters=None, *args, **kwargs):
        """PySEQM hook: set up states/coeffs/DOF, seed velocities (nac_dot_v needs v at t=0), then defer."""
        self._setup_states(mol)
        self._init_coeffs(mol)
        self.set_dof(mol)
        self.initialize_velocity(mol)  # nac_dot_v needs v at t=0
        self._compute_electronic_structure(mol, {})
        return super().initialize(
            mol, remove_com=remove_com, learned_parameters=learned_parameters, *args, **kwargs
        )

    def initialize_velocity(self, mol, vel_com=True):
        """Keep pre-supplied velocities if present; otherwise draw them via the base class."""
        if torch.is_tensor(mol.velocities):
            return mol.velocities
        return super().initialize_velocity(mol, vel_com=vel_com)

    def _phase_match(self, nac):
        """Fix the arbitrary per-state sign of the coupling vectors against the previous step.

        GPU-VECTORIZED (bit-identical to the former per-molecule Python argmax loop; unit-verified
        n=3,4,5, 2026-08-29). Per state ``s`` a sign ``q_s in {+/-1}`` (``q_0`` fixed +1); per molecule
        pick the sign set maximizing ``sum_{i<j} q_i q_j <nac_ij(t), nac_ij(t-1)>`` and apply
        ``q_i q_j`` to each pair. ``nac`` maps 0-based pair keys to (M, n_atoms, 3) tensors; on the
        first call the current couplings are cached and returned unchanged.
        """
        if self.previous_nac is None:
            self.previous_nac = {k: v.detach().clone() for k, v in nac.items()}
            return nac
        pk = self._full_nac_pair_keys
        ref = nac[pk[0]]
        dev, dtype, M = ref.device, ref.dtype, ref.shape[0]  # M = number of molecules (trajectories)
        # O[m, p] = <nac_p(t), nac_p(t-1)> per molecule m and pair p (flatten atom*3, sum).
        O = torch.stack(
            [(nac[i, j] * self.previous_nac[i, j]).reshape(M, -1).sum(1) for (i, j) in pk], dim=1
        )  # (M, P) per-pair overlaps <nac(t), nac(t-1)>
        # S[g, p] = q_i q_j for candidate gauge g (2^{nstates-1} gauges, q_0 fixed +1); cached per nstates.
        if getattr(self, "_phase_S", None) is None or self._phase_S_n != self._nstates:
            cand = [(1,) + x for x in itertools.product((-1, 1), repeat=self._nstates - 1)]
            q = torch.tensor(cand, dtype=dtype, device=dev)  # (G, nstates)
            self._phase_S = torch.stack([q[:, i] * q[:, j] for (i, j) in pk], dim=1)  # (G, P) = q_i q_j
            self._phase_S_n = self._nstates
        S = self._phase_S.to(device=dev, dtype=dtype)
        best = (O @ S.t()).argmax(dim=1)  # (M,) best gauge per molecule (first-max == loop tie-break)
        signs = S[best]                   # (M, P)
        for pidx, (i, j) in enumerate(pk):
            v = nac[i, j]
            v.mul_(signs[:, pidx].reshape(M, *([1] * (v.dim() - 1))))
        self.previous_nac = {k: v.detach().clone() for k, v in nac.items()}
        return nac

    def _compute_electronic_structure(self, mol, learned_parameters, **kwargs):
        """Query the MLIP for E/F/dENACR, populate ``mol``, and return the CIS (gap-to-S0) energies.

        Shapes: ``n`` molecules, ``a`` atoms. Fills per-state energies/forces, the active-state force,
        total energies, and the phase-matched NAC vectors + their time-derivative coupling
        (``mol.nac_dot``). SCF/CIS convergence flags are forced all-converged because the MLIP never
        runs SCF/CIS (so the master's gap_stop still fires on the S0-S1 gap).
        """
        del learned_parameters, kwargs
        self._setup_states(mol)
        pred = self.predictor.predict(
            mol.species.cpu().numpy(), mol.coordinates.detach().cpu().numpy(), gdv_source="forces"
        )
        dev, dtype, (n, a) = mol.coordinates.device, mol.coordinates.dtype, mol.coordinates.shape[:2]
        # Option B (matched termination): MLIP never runs SCF/CIS, so give master batch-shaped
        # all-converged flags -> failed is all-False (batch), gap_stop fires on the S0-S1 gap.
        mol.cis_converged = torch.ones(n, dtype=torch.bool, device=dev)
        self.esdriver.notconverged = torch.zeros(n, dtype=torch.bool, device=dev)
        E = {s: torch.as_tensor(pred["E_abs"][s], dtype=dtype, device=dev) for s in self.model_states}
        e0 = E.get(0, E[self.labels[0]])  # ground-state energy (S0 if present, else the lowest label)
        cis = torch.stack([E[s] - e0 for s in self.labels], 1)  # excitation energies (n, n_excited)
        mol.cis_energies = cis
        mol.all_forces = torch.zeros((n, self._nstates + 1, a, 3), dtype=dtype, device=dev)
        for i, s in enumerate(self.labels):
            mol.all_forces[:, i + 1] = torch.as_tensor(pred["F"][s], dtype=dtype, device=dev)
        active, idx = self._ensure_active_states(n, dev), self._get_arange(n, device=dev)
        mol.active_state, mol.force = active + 1, mol.all_forces[idx, active + 1]
        mol.Etot = torch.stack([E[s] for s in self.labels], 1)[idx, active]
        mol.Hf = mol.Eelec = mol.Etot.clone()
        mol.Enuc = mol.Eiso = torch.zeros_like(mol.Etot)
        mol.e_gap, mol.dipole, mol.dm, mol.cis_amplitudes = (
            cis[idx, active],
            torch.zeros((n, 3), dtype=dtype, device=dev),
            None,
            None,
        )
        nac = {}
        for i, j in self._full_nac_pair_keys:
            si, sj, name = self.labels[i], self.labels[j], f"{self.labels[i]}{self.labels[j]}"
            # PySEQM convention (seqm_functions/nac.py:452-453): nac = coupling / (E_j - E_i),
            # with NO floor and NO clamp. Training dENACR = NAC * (E_j - E_i); divide by the RAW gap.
            # (The old `np.maximum(gap, 1e-8)` clipped inverted/negative gaps to 1e-8, exploding NAC->~1e8
            #  and NACT->~1e6 -> amplitude NaN -> frozen hopping. The raw gap magnitude is ~0.01 eV, finite.)
            gap = np.asarray(pred["sE"][sj]) - np.asarray(pred["sE"][si])
            nac[i, j] = torch.as_tensor(
                (np.asarray(pred["dENACR"][name]) / gap[:, None]).reshape(n, a, 3), dtype=dtype, device=dev
            )
        mol.nac = self._phase_match(nac)
        mol.nac_dot = self._nac_dot_from_vectors(mol, mol.nac)
        self._cache_new = {"energies": cis, "ground_energy": e0, "nac_vec": mol.nac, "nac_dot": mol.nac_dot}
        return cis

    def _compute_NACR_for_hop(self, mol, nac_pairs):
        """Return the ML NAC vectors for the requested hop pairs (raises if any are missing)."""
        out = self._select_nac_pairs(mol.nac, nac_pairs)
        if out is None:
            raise RuntimeError("missing ML NAC vector required for hop rescaling")
        return out

    def _recompute_active_force(self, mol):
        """Refresh ``mol.active_state`` and ``mol.force`` from the cached per-state forces."""
        active, idx = (
            self._active_states,
            self._get_arange(len(self._active_states), device=mol.coordinates.device),
        )
        mol.active_state, mol.force = active + 1, mol.all_forces[idx, active + 1]

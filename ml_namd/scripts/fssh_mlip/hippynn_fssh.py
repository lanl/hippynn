"""Small HiphopNN electronic-structure adapter for PySEQM FSSH."""

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

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
    out = tuple(sorted({int(x) for x in str(spec).replace(" ", "").split(",") if x}))
    if any(x < 0 or x > 5 for x in out) or len([x for x in out if x]) < 2:
        raise ValueError("--states must contain at least two excited labels from 1..5 (optionally S0).")
    return out


def pairs(labels):
    return tuple(f"{i}{j}" for i, j in itertools.combinations([x for x in labels if x], 2))


def molecule(xyz, ntraj, device):
    Z, R = read_xyz([str(Path(xyz).expanduser())], sort=False)
    Z = torch.as_tensor(Z, dtype=torch.long, device=device).repeat(ntraj, 1)
    R = torch.as_tensor(R, dtype=torch.float64, device=device).repeat(ntraj, 1, 1)
    return Molecule(
        Constants().to(device), {"method": "AM1", "scf_eps": 1.0, "scf_converger": [-1]}, R, Z
    ).to(device)


class HiphopSurfaceHopping(SurfaceHoppingDynamics):
    """FSSH that obtains E, F, and NAC vectors from ``HiphopPredictor``."""

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
        self._nstates = len(self.labels)
        self._full_nac_pair_keys = list(itertools.combinations(range(self._nstates), 2))
        self._full_nac_pairs_1based = [(i + 1, j + 1) for i, j in self._full_nac_pair_keys]
        self._full_nac_state_i, self._full_nac_state_j = zip(*self._full_nac_pair_keys)
        self._ensure_active_states(mol.species.shape[0], mol.coordinates.device)

    def initialize(self, mol, remove_com=None, learned_parameters=None, *args, **kwargs):
        self._setup_states(mol)
        self._init_coeffs(mol)
        self.set_dof(mol)
        self.initialize_velocity(mol)  # nac_dot_v needs v at t=0
        self._compute_electronic_structure(mol, {})
        return super().initialize(
            mol, remove_com=remove_com, learned_parameters=learned_parameters, *args, **kwargs
        )

    def initialize_velocity(self, mol, vel_com=True):
        if torch.is_tensor(mol.velocities):
            return mol.velocities
        return super().initialize_velocity(mol, vel_com=vel_com)

    def _phase_match(self, nac):
        # GPU-VECTORIZED (bit-identical to the former per-molecule Python argmax loop; unit-verified
        # n=3,4,5, 2026-08-29). Per state s a sign q_s in {+/-1} (q_0 fixed +1); per molecule pick the
        # sign set maximizing sum_{i<j} q_i q_j <nac_ij(t), nac_ij(t-1)>; apply q_i q_j to each pair.
        if self.previous_nac is None:
            self.previous_nac = {k: v.detach().clone() for k, v in nac.items()}
            return nac
        pk = self._full_nac_pair_keys
        ref = nac[pk[0]]
        dev, dtype, M = ref.device, ref.dtype, ref.shape[0]
        O = torch.stack(
            [(nac[i, j] * self.previous_nac[i, j]).reshape(M, -1).sum(1) for (i, j) in pk], dim=1
        )  # (M, P) per-pair overlaps <nac(t), nac(t-1)>
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
        e0 = E.get(0, E[self.labels[0]])
        cis = torch.stack([E[s] - e0 for s in self.labels], 1)
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
        out = self._select_nac_pairs(mol.nac, nac_pairs)
        if out is None:
            raise RuntimeError("missing ML NAC vector required for hop rescaling")
        return out

    def _recompute_active_force(self, mol):
        active, idx = (
            self._active_states,
            self._get_arange(len(self._active_states), device=mol.coordinates.device),
        )
        mol.active_state, mol.force = active + 1, mol.all_forces[idx, active + 1]

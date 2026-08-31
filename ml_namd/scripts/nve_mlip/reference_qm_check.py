#!/usr/bin/env python3
"""Reference-QM (AM1/CIS) deviation check for an MLIP NVE trajectory — Best-Practices §8.

For sampled frames of an NVE trajectory, recompute the GROUND-TRUTH PySEQM AM1/CIS
E_i and F_i (the same electronic-structure method that generated the training data)
and compare against the MLIP's E_i/F_i on the identical geometries. This separates
"the model conserves its OWN ML energy" from "the model is faithful to the true PES"
(a model can do the former while badly failing the latter — guide §8).

    delta_E_i(t) = E_i^ML(R(t)) - E_i^ref(R(t))       [meV]
    delta_F_i(t) = F_i^ML(R(t)) - F_i^ref(R(t))       [meV/A]

Reference settings are copied VERBATIM from PYSEQM
examples/hippynn_namd/compare_hippynn_pyseqm.py (validated by the user to match the
training-data generation: AM1 / CIS). ML side reuses HIPPYNN_ESDriver from
pyseqm_hippynn_md.py, so the compared E_i/F_i are exactly what the NVE run propagated.

Energy alignment: HIPPYNN_ESDriver returns molecule.Etot = acn_sE_i + energy_offset
= (E_Si - E_gs_opt) + E_gs_opt = E_Si (absolute) — directly comparable to the AM1/CIS
absolute E_Si (= mol.Etot + mol.cis_energies[:, i-1]).

Usage:
  python reference_qm_check.py --model-dir <...> --state-index 3 \
     --energy-offset -1391.422970121047 --traj <run>.0.xyz --n-frames 30 [--n-states-cis 10]
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# reuse the exact ML wiring + species convention from the NVE runner (same dir)
from pyseqm_hippynn_md import HIPPYNN_ESDriver, normalize_device, EXPECTED_SPECIES  # noqa: E402
from seqm.api import Constants, Electronic_Structure, Molecule  # noqa: E402

SYM2Z = {"H": 1, "C": 6, "O": 8}


def read_xyz_frames(path):
    """Parse a multi-frame .xyz -> Z (natoms,), R (nframes, natoms, 3)."""
    L = Path(path).read_text().splitlines()
    frames, syms, i = [], None, 0
    while i < len(L) and L[i].strip():
        n = int(L[i].split()[0]); i += 2
        blk = [L[i + j].split() for j in range(n)]
        if syms is None:
            syms = [b[0] for b in blk]
        frames.append([[float(x) for x in b[1:4]] for b in blk]); i += n
    Z = np.array([SYM2Z[s] for s in syms], dtype=np.int64)
    return Z, np.asarray(frames, dtype=np.float64)


def pyseqm_reference(Z, R, states, method, n_states_cis, dev):
    """AM1/CIS reference E_i (eV, absolute) and F_i (eV/A) for a batch of frames."""
    params = {
        "method": method,
        "elements": [1, 6, 8],
        "scf_eps": 1e-8,
        "scf_converger": [1],
        "active_state": 0,
        "analytical_gradient": [True],
        "do_all_forces": True,
        "excited_states": {"n_states": int(n_states_cis), "method": "cis"},
    }
    nf = R.shape[0]
    Zt = torch.as_tensor(np.tile(Z, (nf, 1)), dtype=torch.long, device=dev)
    Rt = torch.as_tensor(R, dtype=torch.float64, device=dev)
    mol = Molecule(Constants().to(dev), params, Rt, Zt).to(dev)
    mol.verbose = False
    Electronic_Structure(params).to(dev)(mol, do_force=True)
    E, F = {}, {}
    for s in states:
        E[s] = (mol.Etot if s == 0 else mol.Etot + mol.cis_energies[:, s - 1]).detach().cpu().numpy()
        F[s] = mol.all_forces[:, s].detach().cpu().numpy().reshape(nf, -1, 3)
    return E, F


def ml_predict(model_dir, state_index, energy_offset, Z, R, dev):
    """MLIP E_i (eV, absolute) and F_i (eV/A) via the NVE runner's ES driver."""
    esd = HIPPYNN_ESDriver(model_dir, state_index, energy_offset, str(dev))
    nf = R.shape[0]
    dummy = {"method": "AM1", "scf_eps": 123.456, "scf_converger": [-1]}
    Zt = torch.as_tensor(np.tile(Z, (nf, 1)), dtype=torch.long, device=dev)
    Rt = torch.as_tensor(R, dtype=torch.float64, device=dev)
    mol = Molecule(Constants().to(dev), dummy, Rt, Zt).to(dev)
    mol.dm = None
    esd(mol)
    E = mol.Etot.detach().cpu().numpy()
    F = mol.force.detach().cpu().numpy().reshape(nf, -1, 3)
    return E, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--state-index", type=int, required=True, help="target adiabatic state i (0..5)")
    ap.add_argument("--energy-offset", type=float, required=True, help="E_gs_opt (eV) used by the NVE run")
    ap.add_argument("--traj", required=True, help="trajectory .xyz (multi-frame, acn_Z order)")
    ap.add_argument("--n-frames", type=int, default=30, help="evenly-sampled frames to check")
    ap.add_argument("--n-states-cis", type=int, default=10, help="CIS states to solve (match generation)")
    ap.add_argument("--chunk", type=int, default=15, help="frames per reference batch")
    ap.add_argument("--method", default="AM1")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = normalize_device(("cuda" if torch.cuda.is_available() else "cpu")
                           if a.device == "auto" else a.device)
    i = a.state_index
    states = list(range(0, i + 1))  # need S0..Si; S0 anchors the absolute energy

    Z, R_all = read_xyz_frames(a.traj)
    got = [int(z) for z in Z]
    if got != EXPECTED_SPECIES:
        raise SystemExit(f"atom order != acn_Z; got {got}")
    N = R_all.shape[0]
    idx = np.linspace(0, N - 1, min(a.n_frames, N)).round().astype(int)
    R = R_all[idx]
    print(f"traj={a.traj}  frames_total={N}  sampled={len(idx)}  state=S{i}  "
          f"method={a.method}/CIS(n={a.n_states_cis})  dev={dev}", flush=True)

    # ML on all sampled frames (cheap)
    E_ml, F_ml = ml_predict(a.model_dir, i, a.energy_offset, Z, R, dev)

    # Reference in chunks (CIS is heavier)
    E_ref = np.empty(len(idx)); F_ref = np.empty((len(idx), len(Z), 3))
    for c in range(0, len(idx), a.chunk):
        e = min(c + a.chunk, len(idx))
        Eref_c, Fref_c = pyseqm_reference(Z, R[c:e], states, a.method, a.n_states_cis, dev)
        E_ref[c:e] = Eref_c[i]; F_ref[c:e] = Fref_c[i]
        print(f"  ref frames {c}-{e} done", flush=True)

    dE = (E_ml - E_ref) * 1e3                       # meV
    dF = (F_ml - F_ref) * 1e3                        # meV/A
    fmae = np.abs(dF).reshape(len(idx), -1).mean(1)  # per-frame force MAE

    print("\n=== REFERENCE-QM (AM1/CIS) DEVIATION, state S%d ===" % i)
    print(f"  dE   : MAE {np.abs(dE).mean():8.3f}  RMSE {np.sqrt((dE**2).mean()):8.3f} meV   "
          f"(min {dE.min():+.2f}, max {dE.max():+.2f})")
    print(f"  dF   : MAE {np.abs(dF).mean():8.3f}  RMSE {np.sqrt((dF**2).mean()):8.3f} meV/A")
    # growth with trajectory time (extrapolation signature)
    tfrac = idx / max(N - 1, 1)
    if len(idx) > 3:
        slopeE = np.polyfit(tfrac, np.abs(dE), 1)[0]
        slopeF = np.polyfit(tfrac, fmae, 1)[0]
        print(f"  trend: |dE| slope {slopeE:+.2f} meV / |dF| slope {slopeF:+.2f} meV/A over the run "
              f"(positive ⇒ drifting off the training manifold)")
    print(f"  first-frame |dF| {fmae[0]:.2f} -> last-frame |dF| {fmae[-1]:.2f} meV/A")


if __name__ == "__main__":
    main()

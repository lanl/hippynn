#!/usr/bin/env python3
"""FSSH@PYSEQM reference trajectory with a detect-crossings A/B knob (variant of run_fssh_ref.py).

Scientific purpose
------------------
Same as :mod:`run_fssh_ref` (AM1/CIS, ``tdc_method = hamiltonian_fd``, decoherence ON, microcanonical,
Vishikh velocities, one h5 per trajectory, CPU one-thread-per-traj, skip-not-raise) but adds a
``--detect-crossings`` switch to A/B-test PySEQM's trivial-crossing detector. ``None`` (the default)
leaves the PySEQM default (True); ``true``/``false`` force it.

Inputs / outputs
----------------
Identical to :mod:`run_fssh_ref` plus ``--detect-crossings {true,false}``. Output: ``<out-dir>/<prefix>*.h5``
+ ``<prefix>.summary.json``; exit 0 on success, 1 on failure.

Usage: python run_fssh_ref_dc.py --coords coords.xyz --velocities velocity.out --idx 0 \
    --n-states 4 --initial-state 3 --nsteps 200000 --timestep 0.05 --detect-crossings false \
    --out-dir out --prefix keto_s3_i0
"""

__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import argparse, json, sys, traceback
from pathlib import Path
import numpy as np
import torch
from seqm.api import Constants, Molecule, SurfaceHoppingDynamics
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1); torch.set_num_interop_threads(1)  # 1 thread/traj -> true one-per-core (avoid oversubscription)

VEL_DIV = 1000.0  # Vishikh velocity.out -> PySEQM velocity units
SYM2Z = {"H": 1, "C": 6, "O": 8, "N": 7}  # element symbol -> atomic number


def read_xyz_frame(path, idx):
    """Parse the idx-th frame of a multi-frame .xyz -> (Z (natoms,), R (natoms,3)); raises IndexError if absent."""
    lines = open(path).read().splitlines()
    natoms = int(lines[0].split()[0])
    per = natoms + 2
    s = idx * per
    if s + per > len(lines):
        raise IndexError(f"coord frame {idx} not found ({len(lines)//per} frames)")
    Z = []; R = []
    for k in range(natoms):
        tok = lines[s + 2 + k].split()
        Z.append(SYM2Z[tok[0]]); R.append(list(map(float, tok[1:4])))
    return np.asarray(Z, dtype=np.int64), np.asarray(R, dtype=np.float64)


def build_params(n_states, tdc, detect_crossings=None):
    """PySEQM parameter dict (AM1/CIS + TDC + decoherence); optionally set the trivial-crossing detector.

    ``detect_crossings`` None -> leave the PySEQM default (True); otherwise force the given bool.
    """
    nad = {"tdc_method": tdc, "decohere_on_hop": True}
    if detect_crossings is not None:
        nad["detect_crossings"] = detect_crossings   # A/B test knob; default None -> PySEQM default (True)
    return {"method": "AM1", "elements": [1, 6, 8], "scf_eps": 1e-8, "scf_converger": [1],
            "excited_states": {"n_states": n_states, "method": "cis", "tolerance": 1e-6},
            "analytical_gradient": [True], "nonadiabatic": nad}


def read_velocity_frame(path, idx, natoms):
    """Parse a NEXMD-style velocity.out; return the idx-th frame's (natoms,3) velocities (raises IndexError if absent).

    Frames are delimited by a ``$VELOC`` marker; the ``natoms`` lines after it are the velocities.
    """
    lines = open(path).read().splitlines()
    starts = [i for i, l in enumerate(lines) if "$VELOC" in l]
    if idx >= len(starts):
        raise IndexError(f"velocity frame {idx} not found ({len(starts)} frames)")
    s = starts[idx] + 1
    vel = [list(map(float, lines[s + k].split()[:3])) for k in range(natoms)]
    return np.asarray(vel, dtype=np.float64)


def main():
    """Read the IC frame, build the PySEQM molecule + dynamics (with the detect-crossings knob), run, and record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True); ap.add_argument("--velocities", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--n-states", type=int, default=4); ap.add_argument("--initial-state", type=int, default=3)
    ap.add_argument("--nsteps", type=int, default=200000); ap.add_argument("--timestep", type=float, default=0.05)
    ap.add_argument("--interval", type=int, default=40); ap.add_argument("--tdc", default="hamiltonian_fd")  # project rule (2026-08-18): FSSH uses hamiltonian_fd
    ap.add_argument("--detect-crossings", choices=["true", "false"], default=None)  # A/B knob; None = PySEQM default (True)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--out-dir", default="./out"); ap.add_argument("--prefix", default="ref")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    rec = {"idx": a.idx, "prefix": a.prefix, "status": "ok", "error": None, "hops": 0,
           "n_states": a.n_states, "init": a.initial_state, "nsteps": a.nsteps, "dt": a.timestep}
    try:
        z, r = read_xyz_frame(a.coords, a.idx)          # idx-th frame of the multi-frame xyz
        natoms = len(z)
        vel = read_velocity_frame(a.velocities, a.idx, natoms) / VEL_DIV
        # master requires species non-increasing per row: sort descending Z, permute r & vel with it
        order = np.argsort(-z, kind="stable")
        z, r, vel = z[order], r[order], vel[order]

        species = torch.as_tensor(z, dtype=torch.long, device=dev).reshape(1, natoms)      # batched (1,N)
        coords = torch.as_tensor(r, dtype=torch.float64, device=dev).reshape(1, natoms, 3) # batched (1,N,3)
        const = Constants().to(dev)
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        output = {"molid": [0], "prefix": str(out / a.prefix), "print every": a.interval,
                  "xyz": a.interval, "checkpoint every": 0,
                  "h5": {"data": a.interval, "velocities": a.interval, "coordinates": a.interval,
                         "forces": a.interval, "nonadiabatic": a.interval}}
        torch.manual_seed(422 + a.idx)
        dc = None if a.detect_crossings is None else (a.detect_crossings == "true")
        params = build_params(a.n_states, a.tdc, dc)
        mol = Molecule(const, params, coords, species).to(dev)
        mol.velocities = torch.as_tensor(vel, dtype=torch.float64, device=dev).reshape(1, natoms, 3)
        dyn = SurfaceHoppingDynamics(params, timestep=a.timestep, Temp=a.temperature,
                                     initial_state=a.initial_state, output=output).to(dev)
        dyn.run(mol, a.nsteps, reuse_P=True, remove_com=("angular", 1))
        rec["hops"] = len(getattr(dyn, "hop_log", None) or [])
        print(f"[ref] idx={a.idx} n_states={a.n_states} S{a.initial_state} "
              f"{a.nsteps}x{a.timestep}fs tdc={a.tdc} detect_crossings={dc} dev={dev}  REF_DONE hops={rec['hops']}", flush=True)
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"[ref] idx={a.idx} REF_FAILED {rec['error']}", flush=True)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / f"{a.prefix}.summary.json").write_text(json.dumps(rec, indent=1))
    sys.exit(0 if rec["status"] == "ok" else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batched FSSH@MLIP (COMPOSED A+B) over N distinct ICs. IC count = --count (any N);
e.g. enol init3 production = 1000 Vishikh ICs in 8 chunks of 125.
Same as namd_batch.py but driven by ComposedSurfaceHopping (frozen E+F Model A + NACR Model B).
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from hippynn_fssh import pairs, states
from hippynn_fssh_composed import ComposedSurfaceHopping
from seqm.api import Constants, Molecule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelA-dir", required=True)
    p.add_argument("--modelB-dir", required=True)
    p.add_argument("--coords-npy", required=True)
    p.add_argument("--vel-npy", required=True)
    p.add_argument("--z-npy", required=True)
    p.add_argument("--states", default="0,1,2,3")
    p.add_argument("--initial-state", type=int, default=3)
    p.add_argument("--nacr-pairs", default="12,13,23")
    p.add_argument("--nsteps", type=int, default=5000)
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--friction", type=float)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--seed", type=int, default=422)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--output-prefix", default="d1c")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--trajectory-interval", type=int, default=10)
    a = p.parse_args()

    label = states(a.states); need = pairs(label)
    if a.nacr_pairs and set(a.nacr_pairs.split(",")) != set(need):
        raise ValueError(f"--nacr-pairs must be {','.join(need)}")
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")

    R = np.load(a.coords_npy); V = np.load(a.vel_npy); Z1 = np.load(a.z_npy)
    s = a.start; e = (s + a.count) if a.count > 0 else R.shape[0]
    R = R[s:e]; V = V[s:e]
    N, A = R.shape[0], R.shape[1]
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    output = {
        "molid": list(range(N)), "prefix": str(out / a.output_prefix),
        "print every": a.log_interval, "xyz": a.trajectory_interval, "checkpoint every": 0,
        "h5": {"data": a.log_interval, "coordinates": a.trajectory_interval,
               "velocities": a.trajectory_interval, "forces": a.trajectory_interval,
               "nonadiabatic": a.log_interval},
    }
    torch.manual_seed(a.seed)
    Zt = torch.as_tensor(Z1, dtype=torch.long, device=dev).unsqueeze(0).repeat(N, 1)
    Rt = torch.as_tensor(R, dtype=torch.float64, device=dev)
    mol = Molecule(Constants().to(dev),
                   {"method": "AM1", "scf_eps": 1.0, "scf_converger": [-1]}, Rt, Zt).to(dev)
    mol.velocities = torch.as_tensor(V, dtype=mol.coordinates.dtype, device=dev).clone()
    dyn = ComposedSurfaceHopping(a.modelA_dir, a.modelB_dir, label, a.initial_state,
                                 a.timestep, a.temperature, output, str(dev), a.friction).to(dev)
    print(f"D1 COMPOSED A+B batched FSSH@MLIP: N={N} (start={s}) states={label} "
          f"nsteps={a.nsteps} dt={a.timestep} dev={dev}", flush=True)
    dyn.run(mol, a.nsteps, reuse_P=False, remove_com=None, seed=a.seed)
    try:
        dyn._print_hop_log()   # dump accepted/frustrated hop events per molecule (hop-type analysis)
    except Exception as _e:
        print(f"[hop_log dump failed: {_e}]", flush=True)
    print(f"D1_COMPOSED_DONE N={N} start={s}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batched FSSH@MLIP (COMPOSED A+B) over N distinct ICs. IC count = --count (any N);
e.g. enol init3 production = 1000 reference ICs in 8 chunks of 125.
Same as namd_batch.py but driven by ComposedSurfaceHopping (frozen E+F Model A + NACR Model B).

PER-IC 422+idx VARIANT: each IC's FSSH hop RNG is seeded base + global_idx and drawn once per
step -- like-for-like with the PySEQM reference run_fssh_ref.py (torch.manual_seed(422+idx),
one draw/step). The stock driver seeds the GLOBAL torch RNG once per batch, so the vectorised
torch.rand(nmol) gives IC j the j-th of nmol draws from a SINGLE stream -> only `count` distinct
hop realisations, reused across chunks (IC j == IC count*k+j). This variant replaces that single
draw site in SurfaceHoppingDynamics._attempt_hop (master_src/seqm, line 1372) with a per-IC
generator seeded base+(start+i), via a driver-side monkeypatch that leaves shared master_src/seqm
untouched. Device/engine differ from the CPU PySEQM reference by necessity (MLIP is GPU); the
SEEDING SCHEME (each of the 1000 ICs seeded base+idx, one draw/step) is replicated exactly.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from hippynn_fssh import pairs, states
from hippynn_fssh_composed import ComposedSurfaceHopping
from seqm.api import Constants, Molecule

# ---- per-IC hop seeding monkeypatch (see module docstring) ----
from seqm.NonadiabaticDynamics import SurfaceHoppingDynamics as _SHD
_ORIG_ATTEMPT_HOP = _SHD._attempt_hop


def _attempt_hop_perIC(self):
    gens = getattr(self, "_hop_gens", None)
    if gens is None:
        return _ORIG_ATTEMPT_HOP(self)          # unpatched fallback (no per-IC seeds attached)
    nmol = int(self._active_states.shape[0])
    dev = self._active_states.device
    _real_rand = torch.rand
    _state = {"done": False}

    def _patched_rand(*args, **kw):
        # intercept exactly the single hop draw: torch.rand(nmol, device=device)
        if (not _state["done"]) and len(args) == 1 and isinstance(args[0], int) and args[0] == nmol:
            _state["done"] = True
            out = _real_rand(nmol, device=dev)                       # preserve original dtype/layout
            for _i in range(nmol):
                out[_i] = _real_rand(1, generator=gens[_i], device=dev)[0]
            _log = os.environ.get("HOP_RLOG")
            if _log:
                self._hop_step = getattr(self, "_hop_step", 0) + 1
                gidx = self._hop_global_idx
                with open(_log, "a") as _f:
                    for _i in range(nmol):
                        _f.write(f"{self._hop_step} {gidx[_i]} {float(out[_i]):.17g}\n")
            return out
        return _real_rand(*args, **kw)

    torch.rand = _patched_rand
    try:
        return _ORIG_ATTEMPT_HOP(self)
    finally:
        torch.rand = _real_rand


_SHD._attempt_hop = _attempt_hop_perIC


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
    # per-IC 422+idx hop seeds: IC at batch position i has global index s+i, seeded base+(s+i).
    dyn._hop_global_idx = [s + i for i in range(N)]
    dyn._hop_gens = [torch.Generator(device=dev).manual_seed(int(a.seed) + s + i) for i in range(N)]
    print(f"PER-IC SEED ACTIVE: base={a.seed} global_idx={s}..{s + N - 1} "
          f"(hop_seed[i]=base+global_idx[i]; IC{s}->{int(a.seed) + s})", flush=True)
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

#!/usr/bin/env python3
"""Step 4: composed ML-TSH (frozen Model A + Model B) for one seed + head.

First asserts train-A == deploy-A (fssh_provenance): the deploy Model A must be the same checkpoint whose
genA produced Model B's training pool. Then runs the per-IC-seeded composed driver (each initial condition
seeded base + global_index, one draw per step -- like-for-like with the FSSH@PYSEQM reference).

Usage: python 04_run_ml_tsh.py --seed 42 [--head both] [--device cuda] [--count 125] [--start 0]
"""
import argparse
import subprocess
import sys

import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--head", default="both", choices=C.HEADS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--count", type=int, default=125, help="ICs per chunk (<= 125 on a 40 GB GPU)")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    env = C.base_env()

    # train-A == deploy-A gate: refuse to launch if the deploy Model A != the genA-source Model A
    subprocess.run([sys.executable, str(C.FSSH / "fssh_provenance.py"), "assert",
                    "--modelA-dir", str(C.modelA_dir(a.seed)),
                    "--pool-dir", str(C.genA_pool(a.seed))], env=env, check=True)

    outdir = C.OUT / f"{C.TAUT}_{a.head}_seed{a.seed}"
    subprocess.run([sys.executable, str(C.FSSH / "namd_composed_batch_perIC.py"),
                    "--modelA-dir", str(C.modelA_dir(a.seed)),
                    "--modelB-dir", str(C.modelB_dir(a.seed, a.head)),
                    "--coords-npy", str(C.ICS / f"{C.IC_PREFIX}_R0.npy"),
                    "--vel-npy", str(C.ICS / f"{C.IC_PREFIX}_V0.npy"),
                    "--z-npy", str(C.ICS / f"{C.IC_PREFIX}_Z.npy"),
                    "--states", "0,1,2,3", "--initial-state", "3", "--nacr-pairs", "12,13,23",
                    "--nsteps", str(C.NSTEPS), "--timestep", "0.05",
                    "--count", str(a.count), "--start", str(a.start),
                    "--seed", "422", "--device", a.device,
                    "--output-dir", str(outdir), "--output-prefix", "traj"], env=env, check=True)

    print(f"[04] DONE seed={a.seed} head={a.head}: trajectories in {outdir}", flush=True)


if __name__ == "__main__":
    main()

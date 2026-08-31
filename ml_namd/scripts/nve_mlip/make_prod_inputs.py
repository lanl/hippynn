#!/usr/bin/env python3
"""Generate self-contained per-run .input files for the 100 ps PRODUCTION NVE grid.

Per the per-run-inp-config-policy: every NVE (and its excited-state optimization) gets
its OWN .input with ALL variables defined; the slurm launches via `-i <file>` with NO
CLI overrides -> exact reproducibility.

Grid: S0-S3 x {enol,keto} x {multi, single}, per-PES models. 100 ps = 1e6 steps @ dt=0.1 fs,
seed 42. S0 -> NVE from the frame-25000 start (no opt); S1-S3 -> optimize onto Sk then NVE.

Writes into  <nve_md>/inputs_prod/  :  {t}_s{k}_{m}_opt.input (excited) + {t}_s{k}_{m}_nve.input.
"""
import os

EXP = "/lustre/scratch5/akhanna2/ml_namd_project/experiments"
ND = f"{EXP}/nve_md"
GEOM = f"{ND}/geom"
OUTPUTS = f"{ND}/prod"
OUT_INPUTS = f"{ND}/inputs_prod"
E_OPT = {"enol": -1391.422970121047, "keto": -1391.451136550848}

# NVE production constants (dt-convergence-validated; §8 grid passed all cells)
NVE = dict(ensemble="nve", temperature=300.0, timestep=0.1, nsteps=1000000,
           random_seed=42, trajectory_interval=1000, log_interval=500)
OPT_MAXSTEPS = 500


def model_dir(t, k, m):
    """Per-PES: NVE on Sk uses the Sk-dynamics model of the given type."""
    if m == "multi":
        return (f"{EXP}/multistate_unified/{t}_s0s5/model" if k == 0
                else f"{EXP}/excited_nf145_test/{t}_s{k}/model")
    return (f"{EXP}/single_state/{t}_s0/model" if k == 0
            else f"{EXP}/single_state_s{k}dyn/{t}_s{k}/model")


def write_opt(t, k, m):
    pfx = f"{t}_s{k}_{m}"
    lines = [
        f"# PROD opt: relax {t} onto S{k} with the {m} S{k}-dynamics model (geomeTRIC), then NVE.",
        f"model_dir = {model_dir(t, k, m)}",
        f"xyz_file = {GEOM}/{t}_s0_start.xyz",
        f"state_index = {k}",
        f"energy_offset = {E_OPT[t]:.12f}",
        f"output_dir = {OUTPUTS}",
        f"output_prefix = {pfx}",
        f"opt_maxsteps = {OPT_MAXSTEPS}",
        "device = auto",
    ]
    open(f"{OUT_INPUTS}/{pfx}_opt.input", "w").write("\n".join(lines) + "\n")


def write_nve(t, k, m):
    pfx = f"{t}_s{k}_{m}"
    xyz = f"{GEOM}/{t}_s0_start.xyz" if k == 0 else f"{OUTPUTS}/{pfx}_opt.xyz"
    lines = [
        f"# PROD NVE 100 ps on S{k}, {t}, {m} per-PES model ({model_dir(t,k,m).split('/experiments/')[1]}).",
        f"model_dir = {model_dir(t, k, m)}",
        f"xyz_file = {xyz}",
        f"state_index = {k}",
        f"energy_offset = {E_OPT[t]:.12f}",
        f"ensemble = {NVE['ensemble']}",
        f"temperature = {NVE['temperature']}",
        f"timestep = {NVE['timestep']}",
        f"nsteps = {NVE['nsteps']}",
        f"random_seed = {NVE['random_seed']}",
        f"output_prefix = {pfx}_nve",
        f"output_dir = {OUTPUTS}",
        f"trajectory_interval = {NVE['trajectory_interval']}",
        f"log_interval = {NVE['log_interval']}",
        "device = auto",
    ]
    open(f"{OUT_INPUTS}/{pfx}_nve.input", "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_INPUTS, exist_ok=True)
    n = 0
    for t in ("enol", "keto"):
        for k in (0, 1, 2, 3):
            for m in ("multi", "single"):
                if k != 0:
                    write_opt(t, k, m); n += 1
                write_nve(t, k, m); n += 1
    print(f"wrote {n} .input files to {OUT_INPUTS}")


if __name__ == "__main__":
    main()

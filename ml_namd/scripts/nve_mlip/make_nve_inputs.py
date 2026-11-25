#!/usr/bin/env python3
"""Generate the S0-S3 optimize + NVE .input files for both tautomers, using the
PER-DYNAMICS MULTISTATE model mapping (confirmed 2026-07-19).

MODEL MAP (each excited state -> its OWN S_k-dynamics-trained S0-S5 model; NOT one
shared model):
    S0 -> experiments/multistate_unified/{t}_s0s5   (GS/S0-dyn),   state_index 0
    S1 -> experiments/excited_nf145_test/{t}_s1      (S1-dyn),      state_index 1
    S2 -> experiments/excited_nf145_test/{t}_s2      (S2-dyn),      state_index 2
    S3 -> experiments/excited_nf145_test/{t}_s3      (S3-dyn),      state_index 3

WORKFLOW:
    S0  : geom/{t}_s0_start.xyz (frame 25000) --NVE--> outputs/{t}_s0_nve  (no opt)
    S1-3: geom/{t}_s0_start.xyz --geomeTRIC opt on S_k--> geom/{t}_s{k}_opt.xyz
          --NVE on S_k--> outputs/{t}_s{k}_nve

NVE settings match the proven S0 run: 200000 x 0.5 fs = 100 ps, MB@300K init then
microcanonical, traj every 200 steps, log every 100, seed 42.

Writes into  <nve_md>/inputs_ms/  (non-destructive; leaves the old single_state
*_nve.input files untouched).
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import os

EXP = "<ML_NAMD_ROOT>/experiments"
ND = f"{EXP}/nve_md"                      # /users/.../experiments is a symlink to this
GEOM = f"{ND}/geom"
OUTPUTS = f"{ND}/outputs"                 # <SCRATCH> outputs (large trajectories)
OUT_INPUTS = f"{ND}/inputs_ms"

E_OPT = {"enol": -1391.422970121047, "keto": -1391.451136550848}

# state_index -> model experiment dir (relative to EXP), per-dynamics matched
MODEL = {
    0: "multistate_unified/{t}_s0s5",
    1: "excited_nf145_test/{t}_s1",
    2: "excited_nf145_test/{t}_s2",
    3: "excited_nf145_test/{t}_s3",
}

# NVE production constants. dt=0.1 fs (2026-07-19 dt-convergence finding: the S3-dyn
# multistate drift at 0.5 fs was integrator error; 0.1 fs conserves). 100 ps = 1e6 steps,
# same seed 42 across all runs, other MD params = PySEQM defaults.
NVE = dict(ensemble="nve", temperature=300.0, timestep=0.1, nsteps=1000000,
           random_seed=42, trajectory_interval=1000, log_interval=500, device="auto")
OPT_MAXSTEPS = 500


def model_dir(t, k):
    return f"{EXP}/{MODEL[k].format(t=t)}/model"


def write_opt_input(t, k):
    """Excited-state geometry optimization onto S_k (S1-S3 only)."""
    path = f"{OUT_INPUTS}/{t}_s{k}_opt.input"
    lines = [
        f"# geomeTRIC optimization of {t} onto S{k} (per-dynamics model), then NVE on S{k}.",
        f"model_dir = {model_dir(t, k)}",
        f"xyz_file = {GEOM}/{t}_s0_start.xyz",
        f"state_index = {k}",                       # optimizer reads this as active_state
        f"energy_offset = {E_OPT[t]:.12f}",
        f"output_dir = {GEOM}",                     # writes {t}_s{k}_opt.xyz next to the start geoms
        f"output_prefix = {t}_s{k}",
        f"opt_maxsteps = {OPT_MAXSTEPS}",
        "device = auto",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_nve_input(t, k):
    """NVE 100 ps on S_k with the per-dynamics multistate model."""
    xyz = f"{GEOM}/{t}_s0_start.xyz" if k == 0 else f"{GEOM}/{t}_s{k}_opt.xyz"
    path = f"{OUT_INPUTS}/{t}_s{k}_nve.input"
    lines = [
        f"# {t} NVE 100 ps on S{k}, per-dynamics multistate model "
        f"({MODEL[k].format(t=t)}).",
        f"model_dir = {model_dir(t, k)}",
        f"xyz_file = {xyz}",
        f"state_index = {k}",
        f"energy_offset = {E_OPT[t]:.12f}",
        f"ensemble = {NVE['ensemble']}",
        f"temperature = {NVE['temperature']}",
        f"timestep = {NVE['timestep']}",
        f"nsteps = {NVE['nsteps']}",
        f"random_seed = {NVE['random_seed']}",
        f"output_prefix = {t}_s{k}_nve",
        f"output_dir = {OUTPUTS}",
        f"trajectory_interval = {NVE['trajectory_interval']}",
        f"log_interval = {NVE['log_interval']}",
        f"device = {NVE['device']}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    os.makedirs(OUT_INPUTS, exist_ok=True)
    written = []
    for t in ("enol", "keto"):
        for k in (0, 1, 2, 3):
            if k != 0:
                written.append(write_opt_input(t, k))   # S1-S3 need optimization
            written.append(write_nve_input(t, k))
    print(f"wrote {len(written)} input files to {OUT_INPUTS}:")
    for p in written:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()

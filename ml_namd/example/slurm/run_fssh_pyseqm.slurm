#!/bin/bash
# ============================ USER-EDIT (site) ================================
#SBATCH --job-name=ex_fssh_ref
#SBATCH --account=CHANGE_ME
#SBATCH --partition=general           # <-- a CPU partition (one FSSH trajectory per physical core)
#SBATCH --nodes=1
#SBATCH --ntasks=50                   # cores used; each core round-robins its share of the 100 ICs
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=ex_fssh_ref_%j.out
#SBATCH --error=ex_fssh_ref_%j.out
# =============================================================================
# PURPOSE : (EXAMPLE) FSSH@PYSEQM reference (AM1/CIS) over the 100-IC enol subset. n_states=4 (S0-S3),
#           init S3, hamiltonian_fd TDC + decohere_on_hop, 1 ps @ 0.05 fs (store every 40 steps).
#           CPU, ONE trajectory per physical core (the A100 batched eigh is fault-prone near CIs).
# INPUTS  : ics/fssh_pyseqm/{coords_100.xyz, velocity_100.out} (100 frames each, same indices as the MLIP set).
# OUTPUTS : per-trajectory h5 + summary.json under out/fssh_pyseqm/. Harvest with the SAME analysis
#           harvester as the MLIP run for the apple-to-apple comparison.
# DEPENDS : scripts/fssh_pyseqm/run_fssh_ref.py + venv_master (master PySEQM).
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -eo pipefail
PY="${PY:-/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/venv_master/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EX="$(cd "$HERE/.." && pwd)"
DRV="$EX/scripts/fssh_pyseqm/run_fssh_ref.py"
IC="$EX/ics/fssh_pyseqm"
OUT="$EX/out/fssh_pyseqm"; LOG="$OUT/logs"; mkdir -p "$OUT" "$LOG"
NIC=100
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
echo "host=$(hostname) NIC=$NIC ntasks=${SLURM_NTASKS:-1}"
# one srun task per physical core; each core round-robins idx = PROCID, PROCID+NTASKS, ...
srun --cpu-bind=cores bash -c '
  P=${SLURM_PROCID}; NT=${SLURM_NTASKS}
  for ((idx=P; idx<'"$NIC"'; idx+=NT)); do
    pfx=enol_ref_i$(printf "%04d" $idx)
    grep -q REF_DONE "'"$LOG"'/${pfx}.log" 2>/dev/null && continue
    "'"$PY"'" "'"$DRV"'" --coords "'"$IC"'/coords_100.xyz" --velocities "'"$IC"'/velocity_100.out"       --idx $idx --n-states 4 --initial-state 3 --nsteps 20000 --timestep 0.05 --interval 40       --tdc hamiltonian_fd --out-dir "'"$OUT"'" --prefix "$pfx" --device cpu > "'"$LOG"'/${pfx}.log" 2>&1
  done'
echo "EX_FSSH_PYSEQM_DONE rc=$? $(date)"

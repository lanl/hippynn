#!/bin/bash
#SBATCH --job-name=nve_ms
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md/nve_ms_%j.out
#SBATCH --error=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md/nve_ms_%j.err
# Per-dynamics MULTISTATE S0-S3 NVE 100 ps, both tautomers, on Chicoma.
#   Phase 1: geomeTRIC-optimize S1/S2/S3 onto their PES (6 quick jobs) -> geom/{t}_s{k}_opt.xyz
#   Phase 2: NVE 100 ps for S0-S3 x {enol,keto} (8 jobs), 4 GPUs, 4-at-a-time.
# PREREQ: geomeTRIC installed in mlip_md_dyn (pip install geometric).
set -o pipefail
CONDA_SH=/usr/projects/hpcsoft/common/x86_64/anaconda/2023.03-python-3.10/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate /usr/projects/ml4chem/akhanna2/conda_envs/mlip_md_dyn
export PYTHONUTF8=1 LC_ALL=C.UTF-8 LANG=C.UTF-8 HIPPYNN_USE_CUSTOM_KERNELS=False
echo "host=$(hostname) date=$(date)"; nvidia-smi -L
ND=/users/akhanna2/data/ml_namd_project/experiments/nve_md
IN=$ND/inputs_ms
LOG=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md/outputs
mkdir -p "$LOG"

# ---- Phase 1: excited-state optimization (S1,S2,S3 x enol,keto) ----
echo "===== PHASE 1: geomeTRIC optimize $(date +%H:%M:%S) ====="
opt=("enol 1" "enol 2" "enol 3" "keto 1" "keto 2" "keto 3")
gpu=0
for r in "${opt[@]}"; do
  set -- $r; T=$1; K=$2
  echo "[$(date +%H:%M:%S)] opt $T S$K on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${gpu} \
    python "$ND/pyseqm_hippynn_optimize.py" -i "$IN/${T}_s${K}_opt.input" \
    > "$LOG/${T}_s${K}_opt_${SLURM_JOB_ID}.log" 2>&1 &
  gpu=$(( (gpu+1) % 4 ))
  if [ "$gpu" -eq 0 ]; then wait || true; fi
done
wait || true
echo "===== PHASE 1 DONE $(date +%H:%M:%S) ====="

# ---- Phase 2: NVE 100 ps (S0-S3 x enol,keto) ----
echo "===== PHASE 2: NVE 100 ps $(date +%H:%M:%S) ====="
nve=("enol 0" "enol 1" "enol 2" "enol 3" "keto 0" "keto 1" "keto 2" "keto 3")
gpu=0
for r in "${nve[@]}"; do
  set -- $r; T=$1; K=$2
  echo "[$(date +%H:%M:%S)] NVE $T S$K on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_nve_${gpu} \
    python "$ND/pyseqm_hippynn_md.py" -i "$IN/${T}_s${K}_nve.input" \
    > "$LOG/${T}_s${K}_nve_${SLURM_JOB_ID}.log" 2>&1 &
  gpu=$(( (gpu+1) % 4 ))
  if [ "$gpu" -eq 0 ]; then wait || true; fi
done
wait || true
echo "NVE_MS_DONE $(date)"

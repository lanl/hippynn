#!/bin/bash
#SBATCH --job-name=train_B_enol
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=train_B_enol_%j.out
#SBATCH --error=train_B_enol_%j.out
# =============================================================================
# PURPOSE : train Model B (scaled NACR / dENACR, v3 head, pairs 12/13/23) on the ENOL S0-S3 pool.
# INPUTS  : configs/model_b_enol.inp (self-contained; edit data_dir/output_dir there). The dataset
#           must carry acn_dENACR{ij} targets + acn_GDV{ij} + acn_GAP{ij} inputs for pairs 12/13/23.
#           Optional env: SMOKE=1 -> 2-epoch smoke.
# OUTPUTS : trained Model B under the .inp output_dir (experiment_structure.pt + best_model.pt +
#           training_summary.json with per-pair dENACR metrics); SLURM log via --output/--error.
# CONFIG  : #SBATCH block (Chicoma gpu/gpu40, 1 GPU); HIPPYNN training conda env activated below.
#           For the sign-consistent joint loss, set joint_state_nacr_loss = true in the .inp (its node
#           class must then be importable at load time).
# DEPENDS : ./unified_training.py + ./direct_nacr_head_v3.py (+ ./joint_state_nacr_loss.py if joint).
# LAUNCH  : sbatch train_model_b_enol.slurm        (full run)
#           SMOKE=1 sbatch train_model_b_enol.slurm (2-epoch smoke)
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -eo pipefail

# --- USER-CONFIGURABLE: the HIPPYNN training environment (edit to your install) ---
CONDA_SH="${CONDA_SH:-/usr/projects/hpcsoft/common/x86_64/anaconda/2023.03-python-3.10/etc/profile.d/conda.sh}"
ENV="${ENV:-/usr/projects/ml4chem/akhanna2/conda_envs/hiphop_env}"
source "$CONDA_SH"
conda activate "$ENV"
export PYTHONUTF8=1 LC_ALL=C.UTF-8 LANG=C.UTF-8
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-local}"

# --- self-locate: the trainer + config live in this script's directory ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
TRAIN="$HERE/unified_training.py"
CFG="$HERE/configs/model_b_enol.inp"
[ -f "$TRAIN" ] || { echo "ERROR: trainer not found: $TRAIN" >&2; exit 2; }
[ -f "$CFG" ]   || { echo "ERROR: config not found: $CFG"   >&2; exit 2; }

SMOKE_ARG=""
[ "${SMOKE:-0}" = "1" ] && SMOKE_ARG="--max_epochs 2"

echo "host=$(hostname)  date=$(date)  job=${SLURM_JOB_ID:-local}  SMOKE=${SMOKE:-0}"
nvidia-smi -L || true
echo "TRAIN=$TRAIN"
echo "CFG=$CFG"

python "$TRAIN" --config_file "$CFG" $SMOKE_ARG
echo "TRAIN_B_ENOL_DONE rc=$? $(date)"

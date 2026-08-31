#!/bin/bash
#SBATCH --job-name=train_A_enol
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=train_A_enol_%j.out
#SBATCH --error=train_A_enol_%j.out
# =============================================================================
# PURPOSE : train Model A (E + F, S0-S5) on the ENOL S0-dynamics dataset.
# INPUTS  : configs/model_a_enol.inp (self-contained; edit data_dir/output_dir there).
#           Optional env: SMOKE=1 runs a 2-epoch smoke instead of the full 300-epoch run.
# OUTPUTS : trained model under the .inp output_dir (experiment_structure.pt + best_model.pt +
#           training_summary.json + parity CSVs); SLURM log via --output/--error above.
# CONFIG  : #SBATCH block (Chicoma gpu/gpu40, 1 GPU); the HIPPYNN training conda env is activated below.
# DEPENDS : ./unified_training.py (this dir) + its sibling head/loss modules; the HIPPYNN env.
# LAUNCH  : sbatch train_model_a_enol.slurm        (full run)
#           SMOKE=1 sbatch train_model_a_enol.slurm (2-epoch smoke)
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
CFG="$HERE/configs/model_a_enol.inp"
[ -f "$TRAIN" ] || { echo "ERROR: trainer not found: $TRAIN" >&2; exit 2; }
[ -f "$CFG" ]   || { echo "ERROR: config not found: $CFG"   >&2; exit 2; }

SMOKE_ARG=""
[ "${SMOKE:-0}" = "1" ] && SMOKE_ARG="--max_epochs 2"

echo "host=$(hostname)  date=$(date)  job=${SLURM_JOB_ID:-local}  SMOKE=${SMOKE:-0}"
nvidia-smi -L || true
echo "TRAIN=$TRAIN"
echo "CFG=$CFG"

python "$TRAIN" --config_file "$CFG" $SMOKE_ARG
echo "TRAIN_A_ENOL_DONE rc=$? $(date)"

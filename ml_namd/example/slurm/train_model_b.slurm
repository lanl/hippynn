#!/bin/bash
# ============================ USER-EDIT (site) ================================
#SBATCH --job-name=ex_train_B
#SBATCH --account=CHANGE_ME
#SBATCH --partition=shared-gpu-ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=ex_train_B_%j.out
#SBATCH --error=ex_train_B_%j.out
# =============================================================================
# PURPOSE : (EXAMPLE) train Model B (dENACR, v3 head, pairs 12/13/23) on the enol BALF or GALF pool.
#           ORACLE GDV+GAP are read from the pool at training; Model A supplies them at inference.
# INPUTS  : DATASET={BALF|GALF} (env, default BALF) -> configs/model_b_enol_<DATASET>.inp; SMOKE=1 -> 2 epochs.
# OUTPUTS : trained model under output_dir (example/runs/model_b_<DATASET>/model).
# DEPENDS : scripts/training/unified_training.py + direct_nacr_head_v3.py (symlinked repo dir) + HIPPYNN env.
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -eo pipefail
PY="${PY:-/usr/projects/cint/anaconda3/hiphopnn_env/bin/python}"
DATASET="${DATASET:-BALF}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EX="$(cd "$HERE/.." && pwd)"
TRAIN="$EX/scripts/training/unified_training.py"
CFG="$EX/configs/model_b_enol_${DATASET}.inp"
[ -f "$TRAIN" ] || { echo "ERROR: trainer not found: $TRAIN" >&2; exit 2; }
[ -f "$CFG" ]   || { echo "ERROR: config not found: $CFG" >&2; exit 2; }
export HIPPYNN_USE_CUSTOM_KERNELS=False PYTHONUTF8=1
SMOKE_ARG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_ARG="--max_epochs 2"
echo "host=$(hostname) DATASET=$DATASET CFG=$CFG"; nvidia-smi -L || true
"$PY" "$TRAIN" --config_file "$CFG" $SMOKE_ARG
echo "EX_TRAIN_B_DONE rc=$? $(date)"

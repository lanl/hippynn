#!/bin/bash
# ============================ USER-EDIT (site) ================================
#SBATCH --job-name=ex_train_A
#SBATCH --account=CHANGE_ME            # <-- set your Darwin allocation
#SBATCH --partition=shared-gpu-ampere  # <-- a GPU partition you can use
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=ex_train_A_%j.out
#SBATCH --error=ex_train_A_%j.out
# =============================================================================
# PURPOSE : (EXAMPLE) train Model A (E+F, S0-S5) on the enol BALF or GALF pool.
# INPUTS  : DATASET={BALF|GALF} (env, default BALF) -> picks configs/model_a_enol_<DATASET>.inp.
#           SMOKE=1 -> 2-epoch smoke instead of the full run.
# OUTPUTS : trained model under the config's output_dir (example/runs/model_a_<DATASET>/model).
# DEPENDS : scripts/training/unified_training.py (symlink -> repo) + the HIPPYNN training env.
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -eo pipefail
PY="${PY:-/usr/projects/cint/anaconda3/hiphopnn_env/bin/python}"                 # HIPPYNN training interpreter (edit if needed)
DATASET="${DATASET:-BALF}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EX="$(cd "$HERE/.." && pwd)"
TRAIN="$EX/scripts/training/unified_training.py"
CFG="$EX/configs/model_a_enol_${DATASET}.inp"
[ -f "$TRAIN" ] || { echo "ERROR: trainer not found: $TRAIN (is the scripts/ symlink present?)" >&2; exit 2; }
[ -f "$CFG" ]   || { echo "ERROR: config not found: $CFG" >&2; exit 2; }
export HIPPYNN_USE_CUSTOM_KERNELS=False PYTHONUTF8=1
SMOKE_ARG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_ARG="--max_epochs 2"
echo "host=$(hostname) DATASET=$DATASET CFG=$CFG"; nvidia-smi -L || true
"$PY" "$TRAIN" --config_file "$CFG" $SMOKE_ARG
echo "EX_TRAIN_A_DONE rc=$? $(date)"

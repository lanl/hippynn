#!/bin/bash
# ============================ USER-EDIT (site) ================================
#SBATCH --job-name=ex_fssh_mlip
#SBATCH --account=CHANGE_ME
#SBATCH --partition=shared-gpu-ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=ex_fssh_mlip_%j.out
#SBATCH --error=ex_fssh_mlip_%j.out
# =============================================================================
# PURPOSE : (EXAMPLE) FSSH@MLIP -- composed decoupled Model A + Model B -- over the 100-IC enol subset.
#           States S0-S3, coupling pairs 12/13/23, init S3, 1 ps @ 0.05 fs (20000 steps), store every 40.
# INPUTS  : DATASET={BALF|GALF} (env, default BALF) selects the model cell; the 100 ICs are in
#           ics/fssh_mlip/{R0_100.npy, V0_100.npy, Z.npy}  (Z is the (15,) species vector).
# OUTPUTS : per-trajectory h5 under out/fssh_mlip_<DATASET>/. Harvest with scripts/analysis/harvest_fssh_full.py.
# DEPENDS : scripts/fssh_mlip/namd_composed_batch.py (+ its predictor bundle) + venv_master; the shipped
#           decoupled models models/model{A,B}_enol_nf145/<DATASET>/.
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -eo pipefail
PY="${PY:-/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/venv_master/bin/python}"
DATASET="${DATASET:-BALF}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EX="$(cd "$HERE/.." && pwd)"
DRV="$EX/scripts/fssh_mlip/namd_composed_batch.py"
MA="$EX/models/modelA_enol_nf145/${DATASET}"
MB="$EX/models/modelB_enol_nf145/${DATASET}"
IC="$EX/ics/fssh_mlip"
OUT="$EX/out/fssh_mlip_${DATASET}"; mkdir -p "$OUT"
export HIPPYNN_USE_CUSTOM_KERNELS=False PYTHONUTF8=1
echo "host=$(hostname) DATASET=$DATASET"; nvidia-smi -L || true
"$PY" "$DRV"   --modelA-dir "$MA" --modelB-dir "$MB"   --coords-npy "$IC/R0_100.npy" --vel-npy "$IC/V0_100.npy" --z-npy "$IC/Z.npy"   --states 0,1,2,3 --nacr-pairs 12,13,23 --initial-state 3   --nsteps 20000 --timestep 0.05 --count 100 --start 0   --trajectory-interval 40 --log-interval 40   --output-dir "$OUT" --output-prefix "enol_${DATASET}"
echo "EX_FSSH_MLIP_DONE rc=$? $(date)"

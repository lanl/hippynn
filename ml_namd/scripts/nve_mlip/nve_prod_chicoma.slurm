#!/bin/bash
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md/prod/prod_%j.out
#SBATCH --error=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md/prod/prod_%j.err
# 100 ps PRODUCTION NVE grid, one tautomer (TAUT env), S0-S3 x {multi, single}.
# INPUT-FILE DRIVEN (per-run-inp-config-policy): every step is `-i <file>` with NO CLI overrides,
# for exact reproducibility. Per cell: [optimize (excited)] -> NVE 100 ps -> drift + integrity.
# 4 GPUs, 4 cells at a time. Submit twice (TAUT=enol, keto). Inputs from make_prod_inputs.py.
set -o pipefail
CONDA_SH=/usr/projects/hpcsoft/common/x86_64/anaconda/2023.03-python-3.10/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate /usr/projects/ml4chem/akhanna2/conda_envs/mlip_md_dyn
export PYTHONUTF8=1 LC_ALL=C.UTF-8 LANG=C.UTF-8 HIPPYNN_USE_CUSTOM_KERNELS=False
echo "host=$(hostname) date=$(date) TAUT=${TAUT}"; nvidia-smi -L
NDW=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md
IN=$NDW/inputs_prod; OUT=$NDW/prod; mkdir -p "$OUT"
TAUT=${TAUT:-enol}

run_cell() {  # k m gpu
  local k=$1 m=$2 gpu=$3 PFX=${TAUT}_s${1}_${2}
  {
    echo "### CELL $PFX (state S$k, $m) GPU $gpu ###"
    export CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${gpu}
    if [ "$k" != "0" ]; then
      python "$NDW/pyseqm_hippynn_optimize.py" -i "$IN/${PFX}_opt.input" 2>&1 | tail -3
    fi
    python "$NDW/pyseqm_hippynn_md.py" -i "$IN/${PFX}_nve.input" 2>&1 | tail -2
    echo "-- drift --";     python "$NDW/check_nve_drift.py" "$OUT/${PFX}_nve.0.h5" --timestep 0.1 2>&1 | grep -E "drift rate|RESULT|max .dEtot|T mean" || true
    echo "-- integrity --"; python "$NDW/analyze_geom_integrity.py" "$OUT/${PFX}_nve.0.xyz" 2>&1 | grep -E "VERDICT|max nearest|min pairwise" || true
    echo "### DONE $PFX ###"
  } > "$OUT/${PFX}.log" 2>&1
}

gpu=0
for k in 0 1 2 3; do for m in multi single; do
  echo "[$(date +%H:%M:%S)] launch ${TAUT}_s${k}_${m} on GPU $gpu"
  run_cell "$k" "$m" "$gpu" &
  gpu=$(( (gpu + 1) % 4 ))
  [ "$gpu" -eq 0 ] && wait
done; done
wait

echo "==================== PRODUCTION SUMMARY ($TAUT, 100 ps) ===================="
for k in 0 1 2 3; do for m in multi single; do
  PFX=${TAUT}_s${k}_${m}; echo "---- $PFX ----"
  grep -E "RESULT|VERDICT|drift rate" "$OUT/${PFX}.log" 2>/dev/null || echo "  (no log)"
done; done
echo "PROD_${TAUT}_DONE $(date)"

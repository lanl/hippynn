#!/bin/bash
#SBATCH --job-name=nve_dt01
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md/smoke/smoke_dt01_%j.out
#SBATCH --error=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md/smoke/smoke_dt01_%j.err
# dt-TEST: does timestep 0.1 fs (vs 0.5) remove the multistate energy leak? NVE 10 ps from the
# SAME S3-optimized minima (reused), both models. If multistate drift -> ~0 it was integrator
# error; if it stays, it is genuine PES roughness.
# =============================================================================
# PURPOSE : dt convergence smoke for NVE@MLIP. 10 ps NVE at dt=0.1 fs from the SAME S3-optimized
#           minima, both models, to decide whether the multistate energy leak is integrator error.
# INPUTS  : the S3-optimized start .xyz + trained models under $ND (Chicoma nve_md dir).
# OUTPUTS : per-run h5/xyz + drift check under the smoke dir; logs in the #SBATCH output/error paths.
# CONFIG  : #SBATCH block above (Chicoma gpu/gpu40, 1 GPU); conda env mlip_md_dyn activated below.
# DEPENDS : pyseqm_hippynn_md.py + check_nve_drift.py (this dir); paths are the confirmed Chicoma env.
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -o pipefail
CONDA_SH=/usr/projects/hpcsoft/common/x86_64/anaconda/2023.03-python-3.10/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate /usr/projects/ml4chem/akhanna2/conda_envs/mlip_md_dyn
export PYTHONUTF8=1 LC_ALL=C.UTF-8 LANG=C.UTF-8 HIPPYNN_USE_CUSTOM_KERNELS=False
echo "host=$(hostname) date=$(date)"; nvidia-smi -L
ND=/lustre/scratch5/akhanna2/ml_namd_project/experiments/nve_md
S=$ND/smoke; cd "$S"

echo "===== NVE 10 ps @ dt=0.1 fs: MULTISTATE ====="; python "$ND/pyseqm_hippynn_md.py" -i "$S/enol_s3_msmin_dt01.input" 2>&1 | tail -4
echo "===== NVE 10 ps @ dt=0.1 fs: SINGLE-STATE ====="; python "$ND/pyseqm_hippynn_md.py" -i "$S/enol_s3_ssmin_dt01.input" 2>&1 | tail -4
echo "===== DRIFT: MULTISTATE dt=0.1 ====="; python "$ND/check_nve_drift.py" "$S/enol_s3_msmin_dt01.0.h5" --timestep 0.1 || true
echo "===== DRIFT: SINGLE-STATE dt=0.1 ====="; python "$ND/check_nve_drift.py" "$S/enol_s3_ssmin_dt01.0.h5" --timestep 0.1 || true
echo "SMOKE_DT01_DONE $(date)"

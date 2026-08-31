#!/bin/bash
#SBATCH --job-name=bggen_enol
#SBATCH --reservation=gpu_debug
#SBATCH --partition=gpu_debug
#SBATCH --account=s17_cint_g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=96GB
#SBATCH --time=02:00:00
#SBATCH -o /lustre/scratch5/akhanna2/ml_namd_project/data_gen/enol_balf_galf/bggen_%j.out
#SBATCH -e /lustre/scratch5/akhanna2/ml_namd_project/data_gen/enol_balf_galf/bggen_%j.err
# Full BALF + GALF re-labeling gen (keto): AM1/CIS n_states=11, batch 5000, CG max_iter=400. Outputs -> /lustre.
set -o pipefail
CONDA_SH=/usr/projects/hpcsoft/common/x86_64/anaconda/2023.03-python-3.10/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate /usr/projects/ml4chem/akhanna2/conda_envs/mlip_md_dyn
export PYTHONUTF8=1 LC_ALL=C.UTF-8 LANG=C.UTF-8
export PYTHONPATH=/users/akhanna2/data/ml_namd_project/data_gen/PYSEQM-ml_namd:$PYTHONPATH
BASE=/users/akhanna2/data/ml_namd_project/data_gen
L=/lustre/scratch5/akhanna2/ml_namd_project/data_gen/enol_balf_galf
nvidia-smi -L
for t in BALF GALF; do
  cd "$L/$t"
  echo "===== $t gen START $(date) ====="
  python "$BASE/run_pyseqm_electronic_properties_calcs.py" \
      --coords acn_R.npy --species acn_Z.npy --gpus 0 1 2 3 --batch 5000 --n_states 11
  echo "===== $t gen rc=$?  DONE $(date) ====="
  ls -la nacrs_all_states.npy energies_all_states.npy forces_all_states.npy 2>/dev/null
done
echo BG_GEN_DONE

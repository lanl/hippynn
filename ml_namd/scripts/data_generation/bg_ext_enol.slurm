#!/bin/bash
#SBATCH --job-name=bgext_enol
#SBATCH --reservation=gpu_debug
#SBATCH --partition=gpu_debug
#SBATCH --account=s17_cint_g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96GB
#SBATCH --time=01:00:00
#SBATCH -o /lustre/scratch5/akhanna2/ml_namd_project/data_gen/enol_balf_galf/bgext_%j.out
#SBATCH -e /lustre/scratch5/akhanna2/ml_namd_project/data_gen/enol_balf_galf/bgext_%j.err
# Extract BALF + GALF re-labeled gen -> acn_ splits (sE0-10, F, 45 dENACR pairs incl 12/13/23), enol E0.
set -o pipefail
CONDA_SH=/usr/projects/hpcsoft/common/x86_64/anaconda/2023.03-python-3.10/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate /usr/projects/ml4chem/akhanna2/conda_envs/mlip_md_dyn
BASE=/users/akhanna2/data/ml_namd_project/data_gen
SCRIPT=$BASE/process_acn_data_parallel_REPURPOSED.py
L=/lustre/scratch5/akhanna2/ml_namd_project/data_gen/enol_balf_galf
E0=-1391.422970121047
for t in BALF GALF; do
  echo "===== extract $t START $(date) ====="
  python "$SCRIPT" \
      --input-dir "$L/$t" --output-dir "$L/$t/extracted" \
      --n-states 10 --split-nacr --tautomer enol --e0-opt="$E0" \
      --coords-r acn_R.npy --coords-z acn_Z.npy \
      --chop-start 50000 --chop-step 50000 --chop-end 50001 --n-workers 32
  echo "===== extract $t rc=$?  DONE $(date) ====="
done
echo BG_EXT_DONE
ls -d "$L"/BALF/extracted/* "$L"/GALF/extracted/* 2>/dev/null

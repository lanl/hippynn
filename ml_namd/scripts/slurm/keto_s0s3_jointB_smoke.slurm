#!/bin/bash
#SBATCH --job-name=ks0s3jsmk
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu_debug
#SBATCH --qos=debug
#SBATCH --reservation=gpu_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/ks0s3jsmk_%j.out
#SBATCH -e /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/ks0s3jsmk_%j.out
# SMOKE: S0-S3 3-pair composed FSSH, TIER-1 JOINT Model B (joint per-state phase loss).
# Verify the joint Model B loads clean (experiment_structure.pt references joint loss node ->
# must NOT break inference load), no KeyError/NaN, hops fire. COUNT=5, 500 steps, init3.
set -o pipefail
RUN=/lustre/scratch5/akhanna2/keto_mlip_10ps
DRV=$RUN/driver; IC=$RUN/ics; PY=/usr/projects/ml4chem/akhanna2/venv_master/bin/python
EXP=/users/akhanna2/data/ml_namd_project/experiments
MA=$EXP/bg10p/nf145/modelA/BALF/keto/model
MB=$EXP/keto_s0s5A_3pairB_joint/nf145/modelB/BALF/keto/model
O=$RUN/smoke_s0s3_jointB; mkdir -p $O $RUN/logs
export PYTHONPATH=$DRV OMP_NUM_THREADS=4 PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
cd $DRV; echo "SMOKE keto-s0s3 jointB $(date)"; nvidia-smi -L
$PY namd_composed_batch.py --modelA-dir $MA --modelB-dir $MB \
  --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
  --start 0 --count 5 --nsteps 500 --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
  --output-dir $O/comp_BALF --output-prefix ks0s3j_smk --log-interval 20 --trajectory-interval 20 > $O/comp_BALF.log 2>&1
echo "rc=$?"
L=$O/comp_BALF.log
echo "DONE=$(grep -c _DONE $L) h5=$(find $O/comp_BALF -name '*.h5'|wc -l) NaN=$(grep -ic nan $L)"
grep -iE "error|traceback|KeyError|found 0|must be a subset|Can't get" $L | head -5 | sed 's/^/ ERR: /'
tail -6 $L | sed 's/^/  /'
echo "KS0S3J_SMOKE_DONE $(date)"

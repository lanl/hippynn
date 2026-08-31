#!/bin/bash
#SBATCH --job-name=ks0s5Asmk
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu_debug
#SBATCH --qos=debug
#SBATCH --reservation=gpu_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/ks0s5Asmk_%j.out
#SBATCH -e /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/ks0s5Asmk_%j.out
# SMOKE: validate the NEW-model inference wiring into the S0-S3 FSSH driver, BEFORE the big run.
#   composed: S0-S5 Model A (bg10p) + 3-pair Model B (keto_s0s5A_3pairB) -> does predictor select S0-S3?
#   joint:    S0-S5 E/F + 3-pair NACR joint -> same selection check.
# COUNT=5 IC, 500 steps, init3, states 0,1,2,3, pairs 12,13,23. Pass = rc0 + h5 + NaN=0.
set -o pipefail
RUN=/lustre/scratch5/akhanna2/keto_mlip_10ps
DRV=$RUN/driver; IC=$RUN/ics; PY=/usr/projects/ml4chem/akhanna2/venv_master/bin/python
EXP=/users/akhanna2/data/ml_namd_project/experiments
MA=$EXP/bg10p/nf145/modelA/BALF/keto/model
MB=$EXP/keto_s0s5A_3pairB/nf145/modelB/BALF/keto/model
JT=$EXP/keto_s0s5A_3pairB/nf145/joint/BALF/keto/model
O=$RUN/smoke_s0s5A; mkdir -p $O $RUN/logs
export PYTHONPATH=$DRV OMP_NUM_THREADS=4 PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
cd $DRV
echo "SMOKE keto-s0s5A $(date)"; nvidia-smi -L
$PY -c "import seqm,inspect,torch; print('seqm:',inspect.getfile(seqm)); print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "=== JOINT (S0-S5 E/F + 3-pair NACR) BALF ==="
$PY namd_batch.py --model-dir $JT \
  --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
  --start 0 --count 5 --nsteps 500 --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
  --output-dir $O/joint_BALF --output-prefix ks_joint_smk --log-interval 20 --trajectory-interval 20 \
  > $O/joint_BALF.log 2>&1; echo "joint rc=$?"
echo "=== COMPOSED (S0-S5 Model A + 3-pair Model B) BALF ==="
$PY namd_composed_batch.py --modelA-dir $MA --modelB-dir $MB \
  --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
  --start 0 --count 5 --nsteps 500 --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
  --output-dir $O/comp_BALF --output-prefix ks_comp_smk --log-interval 20 --trajectory-interval 20 \
  > $O/comp_BALF.log 2>&1; echo "comp rc=$?"
echo "===================== SMOKE SUMMARY ====================="
for m in joint comp; do
  L=$O/${m}_BALF.log; nan=$(grep -ic "nan" $L 2>/dev/null); done=$(grep -c "_DONE" $L 2>/dev/null)
  h5=$(find $O/${m}_BALF -name "*.h5" 2>/dev/null | wc -l); hops=$(grep -c "Hop events" $L 2>/dev/null)
  echo "$m: DONE=$done h5=$h5 NaN_lines=$nan hop_log_blocks=$hops"
  grep -iE "error|traceback|expected 1 node|found 0" $L 2>/dev/null | head -3 | sed 's/^/   ERR: /'
  echo "  --- tail ---"; tail -4 $L 2>/dev/null | sed 's/^/    /'
done
echo "KS0S5A_SMOKE_DONE $(date)"

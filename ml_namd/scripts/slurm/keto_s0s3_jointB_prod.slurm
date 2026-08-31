#!/bin/bash
#SBATCH --job-name=ks0s3jfssh
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH -o /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/ks0s3j_%x_%j.out
#SBATCH -e /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/ks0s3j_%x_%j.out
# TIER-1 keto FSSH@MLIP, S0-S3 (states 0,1,2,3; pairs 12,13,23), init3, 10 ps.
# COMPOSED: Model A = S0-S5 (bg10p), Model B = 3-pair JOINT per-state phase loss
# (keto_s0s5A_3pairB_joint). Decisive test: does training-time sign-consistency close
# keto over-relaxation (S1 0.70 -> toward ref 0.52)? Baseline = phaseless Model B.
# venv_master e273f6d; nac_dot_v + decohere; original 3-pair `driver`.
set -o pipefail
: ${CELL:?}
COUNT=${COUNT:-125}; NIC_TOTAL=${NIC_TOTAL:-1000}
if [ -z "${STARTS:-}" ]; then STARTS=""; s=0; while [ $s -lt $NIC_TOTAL ]; do STARTS="${STARTS}${STARTS:+:}$s"; s=$((s+COUNT)); done; fi
RUN=/lustre/scratch5/akhanna2/keto_mlip_10ps
DRV=$RUN/driver; IC=$RUN/ics
PY=/usr/projects/ml4chem/akhanna2/venv_master/bin/python
EXP=/users/akhanna2/data/ml_namd_project/experiments
MA=$EXP/bg10p/nf145/modelA/$CELL/keto/model
MB=$EXP/keto_s0s5A_3pairB_joint/nf145/modelB/$CELL/keto/model
O=$RUN/prod_s0s3_jointB/keto_comp_init3; mkdir -p $O $RUN/logs
NSTEPS=200000; INTERVAL=40
export PYTHONPATH=$DRV OMP_NUM_THREADS=4 PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
cd $DRV
echo "START keto-s0s3 jointB comp $CELL init3 COUNT=$COUNT $(date)"; nvidia-smi -L
mapfile -t U < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d " "); NG=${#U[@]}; echo "NGPU=$NG"
run_task(){ local gpu=$1 c=$2; local t0=$SECONDS
  CUDA_VISIBLE_DEVICES=$gpu $PY namd_composed_batch.py --modelA-dir $MA --modelB-dir $MB \
    --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
    --start $c --count $COUNT --nsteps $NSTEPS --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
    --output-dir $O/${CELL}/c${c} --output-prefix keto_comp_${CELL}_s0s3j --log-interval $INTERVAL --trajectory-interval $INTERVAL \
    > $O/${CELL}_c${c}.log 2>&1
  local rc=$?; local dt=$((SECONDS-t0))
  awk -v t=$dt -v n=$NSTEPS -v g=$gpu -v tag="keto comp $CELL c$c" -v rc=$rc "BEGIN{printf \"TIMING %s gpu=%s: %ds wall, %.4f s/step, rc=%d\n\",tag,g,t,t/n,rc}"
}
tasks=(${STARTS//[,:]/ })
POOL=$(mktemp -u); mkfifo "$POOL"; exec 9<>"$POOL"; rm -f "$POOL"
for u in "${U[@]}"; do printf "%s\n" "$u" >&9; done
for c in "${tasks[@]}"; do read -u 9 gpu; { run_task "$gpu" "$c"; printf "%s\n" "$gpu" >&9; } & done
wait; exec 9>&-
echo "n h5 in $CELL: $(find $O/$CELL -name '*.h5' 2>/dev/null|wc -l)"; echo "KS0S3J_DONE comp $CELL $(date)"

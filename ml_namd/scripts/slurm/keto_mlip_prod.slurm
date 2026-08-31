#!/bin/bash
#SBATCH --job-name=kmlip10
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH -o /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/kmlip10_%x_%j.out
#SBATCH -e /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/kmlip10_%x_%j.out
# KETO FSSH@MLIP, 10 ps (200000 steps), init3, S0-S3 (states 0,1,2,3; pairs 12/13/23), dt=0.05.
# Chicoma gpu (4x A100-40GB/node), venv_master = master PySEQM e273f6d + torch cu118 + hippynn.
# FIXED driver (L-082/083/084): nac = dENACR/(E_j-E_i) RAW gap NO floor; tdc_method=nac_dot_v; decohere_on_hop=True.
# Params via --export: MODE={joint|comp}, CELL={BALF|GALF}, COUNT (default 125 = OOM-safe max on 40GB),
#   NIC_TOTAL (default 1000), STARTS (optional colon-list; auto from COUNT/NIC_TOTAL).
# GPU dispatch: DYNAMIC FIFO work-queue over the node's GPUs (uuid-addressed), self-balancing.
set -o pipefail
: ${MODE:?}; : ${CELL:?}
COUNT=${COUNT:-125}; NIC_TOTAL=${NIC_TOTAL:-1000}   # 125 = OOM-safe max (joint OOMs at 250 on 40GB)
if [ -z "${STARTS:-}" ]; then
  STARTS=""; s=0
  while [ $s -lt $NIC_TOTAL ]; do STARTS="${STARTS}${STARTS:+:}$s"; s=$((s+COUNT)); done
fi
RUN=/lustre/scratch5/akhanna2/keto_mlip_10ps
DRV=$RUN/driver
PY=/usr/projects/ml4chem/akhanna2/venv_master/bin/python
BG=$RUN/bg_ladder_keto/nf145
IC=$RUN/ics
O=$RUN/prod/keto_${MODE}_init3; mkdir -p $O $RUN/logs
NSTEPS=200000; INTERVAL=40
export PYTHONPATH=$DRV OMP_NUM_THREADS=4 PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
cd $DRV
echo "START keto $MODE $CELL init3 COUNT=$COUNT NIC_TOTAL=$NIC_TOTAL STARTS=[$STARTS] $(date)"; nvidia-smi -L
$PY -c "import seqm,inspect; print('seqm:',inspect.getfile(seqm))"
if [ "$MODE" = joint ]; then grep -n "decohere_on_hop\|tdc_method" $DRV/hippynn_fssh.py | head -1; else grep -n "decohere_on_hop\|tdc_method" $DRV/hippynn_fssh_composed.py | head -1; fi
mapfile -t U < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d " "); NG=${#U[@]}; echo "NGPU=$NG"
run_task(){ local gpu=$1 c=$2; local t0=$SECONDS
  if [ "$MODE" = joint ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY namd_batch.py --model-dir $BG/joint/$CELL/model \
      --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
      --start $c --count $COUNT --nsteps $NSTEPS --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
      --output-dir $O/${CELL}/c${c} --output-prefix keto_${MODE}_${CELL}_decohere --log-interval $INTERVAL --trajectory-interval $INTERVAL \
      > $O/${CELL}_c${c}.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu $PY namd_composed_batch.py --modelA-dir $BG/modelA/$CELL/model --modelB-dir $BG/modelB/$CELL/model \
      --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
      --start $c --count $COUNT --nsteps $NSTEPS --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
      --output-dir $O/${CELL}/c${c} --output-prefix keto_${MODE}_${CELL}_decohere --log-interval $INTERVAL --trajectory-interval $INTERVAL \
      > $O/${CELL}_c${c}.log 2>&1
  fi
  local rc=$?; local dt=$((SECONDS-t0))
  awk -v t=$dt -v n=$NSTEPS -v g=$gpu -v tag="keto $MODE $CELL c$c" -v rc=$rc "BEGIN{printf \"TIMING %s gpu=%s: %ds wall, %d steps, %.4f s/step, rc=%d\n\",tag,g,t,n,t/n,rc}"
}
tasks=(${STARTS//[,:]/ })
POOL=$(mktemp -u); mkfifo "$POOL"; exec 9<>"$POOL"; rm -f "$POOL"
for u in "${U[@]}"; do printf "%s\n" "$u" >&9; done            # seed NG gpu tokens
for c in "${tasks[@]}"; do
  read -u 9 gpu                                                 # wait for a free gpu
  { run_task "$gpu" "$c"; printf "%s\n" "$gpu" >&9; } &         # run, then hand the gpu back
done
wait
exec 9>&-
echo "n h5 in $CELL: $(find $O/$CELL -name "*.h5" 2>/dev/null | wc -l)"
echo "KMLIP10_DONE keto $MODE $CELL $(date)"

#!/bin/bash
#SBATCH --job-name=ks0s5Afssh
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH -o /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/ks0s5Afssh_%x_%j.out
#SBATCH -e /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/ks0s5Afssh_%x_%j.out
# KETO FSSH@MLIP, S0-S3 (states 0,1,2,3; pairs 12,13,23), init3, dt=0.05, 10 ps (200000 steps).
# NEW models: Model A = S0-S5 (bg10p), Model B = 3-pair (keto_s0s5A_3pairB); joint = S0-S5 E/F + 3-pair NACR.
# venv_master = master PySEQM e273f6d. FIXED driver: nac = dENACR/(E_j-E_i) RAW gap NO floor;
# tdc_method=nac_dot_v; decohere_on_hop=True (SAME as the previous keto FSSH@MLIP run).
# Params via --export: MODE={joint|comp}, CELL={BALF|GALF}, COUNT (default 125), NIC_TOTAL (default 1000).
set -o pipefail
: ${MODE:?}; : ${CELL:?}
COUNT=${COUNT:-125}; NIC_TOTAL=${NIC_TOTAL:-1000}
if [ -z "${STARTS:-}" ]; then STARTS=""; s=0; while [ $s -lt $NIC_TOTAL ]; do STARTS="${STARTS}${STARTS:+:}$s"; s=$((s+COUNT)); done; fi
RUN=/lustre/scratch5/akhanna2/keto_mlip_10ps
DRV=$RUN/driver; IC=$RUN/ics
PY=/usr/projects/ml4chem/akhanna2/venv_master/bin/python
EXP=/users/akhanna2/data/ml_namd_project/experiments
MA=$EXP/bg10p/nf145/modelA/$CELL/keto/model                 # S0-S5 Model A (composed E/F+GDV+gap provider)
MB=$EXP/keto_s0s5A_3pairB/nf145/modelB/$CELL/keto/model      # 3-pair Model B (scaled NACR)
JT=$EXP/keto_s0s5A_3pairB/nf145/joint/$CELL/keto/model       # S0-S5 E/F + 3-pair NACR joint
O=$RUN/prod_s0s5A/keto_${MODE}_init3; mkdir -p $O $RUN/logs
NSTEPS=200000; INTERVAL=40
export PYTHONPATH=$DRV OMP_NUM_THREADS=4 PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
cd $DRV
echo "START keto-s0s5A $MODE $CELL init3 COUNT=$COUNT NIC_TOTAL=$NIC_TOTAL $(date)"; nvidia-smi -L
$PY -c "import seqm,inspect; print('seqm:',inspect.getfile(seqm))"
mapfile -t U < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d " "); NG=${#U[@]}; echo "NGPU=$NG"
run_task(){ local gpu=$1 c=$2; local t0=$SECONDS
  if [ "$MODE" = joint ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY namd_batch.py --model-dir $JT \
      --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
      --start $c --count $COUNT --nsteps $NSTEPS --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
      --output-dir $O/${CELL}/c${c} --output-prefix keto_${MODE}_${CELL}_s0s5A --log-interval $INTERVAL --trajectory-interval $INTERVAL \
      > $O/${CELL}_c${c}.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu $PY namd_composed_batch.py --modelA-dir $MA --modelB-dir $MB \
      --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
      --start $c --count $COUNT --nsteps $NSTEPS --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
      --output-dir $O/${CELL}/c${c} --output-prefix keto_${MODE}_${CELL}_s0s5A --log-interval $INTERVAL --trajectory-interval $INTERVAL \
      > $O/${CELL}_c${c}.log 2>&1
  fi
  local rc=$?; local dt=$((SECONDS-t0))
  awk -v t=$dt -v n=$NSTEPS -v g=$gpu -v tag="keto $MODE $CELL c$c" -v rc=$rc "BEGIN{printf \"TIMING %s gpu=%s: %ds wall, %d steps, %.4f s/step, rc=%d\n\",tag,g,t,n,t/n,rc}"
}
tasks=(${STARTS//[,:]/ })
POOL=$(mktemp -u); mkfifo "$POOL"; exec 9<>"$POOL"; rm -f "$POOL"
for u in "${U[@]}"; do printf "%s\n" "$u" >&9; done
for c in "${tasks[@]}"; do read -u 9 gpu; { run_task "$gpu" "$c"; printf "%s\n" "$gpu" >&9; } & done
wait; exec 9>&-
echo "n h5 in $CELL: $(find $O/$CELL -name "*.h5" 2>/dev/null | wc -l)"
echo "KS0S5A_FSSH_DONE keto $MODE $CELL $(date)"

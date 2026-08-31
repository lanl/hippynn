#!/bin/bash
#SBATCH --job-name=kmlipSMK
#SBATCH --account=s17_cint_g
#SBATCH --partition=gpu_debug
#SBATCH --qos=debug
#SBATCH --reservation=gpu_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/kmlipSMK_%j.out
#SBATCH -e /lustre/scratch5/akhanna2/keto_mlip_10ps/logs/kmlipSMK_%j.out
# SMOKE: keto FSSH@MLIP on Chicoma venv_master (e273f6d GPU). Validate BOTH driver paths
# (joint + composed) load the keto BALF models under the FIXED driver, propagate, write h5,
# fire hops, and produce NO amplitude NaN. COUNT=5 IC, 500 steps each. init3, S0-S3, dt=0.05.
set -o pipefail
RUN=/lustre/scratch5/akhanna2/keto_mlip_10ps
DRV=$RUN/driver; PY=/usr/projects/ml4chem/akhanna2/venv_master/bin/python
BG=$RUN/bg_ladder_keto/nf145; IC=$RUN/ics
O=$RUN/smoke; mkdir -p $O $RUN/logs
export PYTHONPATH=$DRV OMP_NUM_THREADS=4 PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
cd $DRV
echo "SMOKE keto init3 $(date)"; nvidia-smi -L
$PY -c "import seqm,inspect,torch; print('seqm:',inspect.getfile(seqm)); print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
grep -n "decohere_on_hop\|tdc_method\|gap_floor" $DRV/hippynn_fssh.py | head
echo "=== JOINT BALF ==="
$PY namd_batch.py --model-dir $BG/joint/BALF/model \
  --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
  --start 0 --count 5 --nsteps 500 --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
  --output-dir $O/joint_BALF --output-prefix keto_joint_BALF_smk --log-interval 20 --trajectory-interval 20 \
  > $O/joint_BALF.log 2>&1; echo "joint rc=$?"
echo "=== COMP BALF ==="
$PY namd_composed_batch.py --modelA-dir $BG/modelA/BALF/model --modelB-dir $BG/modelB/BALF/model \
  --coords-npy $IC/keto10ps_R0.npy --vel-npy $IC/keto10ps_V0.npy --z-npy $IC/keto10ps_Z.npy \
  --start 0 --count 5 --nsteps 500 --timestep 0.05 --states 0,1,2,3 --initial-state 3 --nacr-pairs 12,13,23 \
  --output-dir $O/comp_BALF --output-prefix keto_comp_BALF_smk --log-interval 20 --trajectory-interval 20 \
  > $O/comp_BALF.log 2>&1; echo "comp rc=$?"
echo "===================== SMOKE SUMMARY ====================="
for m in joint comp; do
  L=$O/${m}_BALF.log
  nan=$(grep -ic "nan" $L 2>/dev/null); done=$(grep -c "_DONE" $L 2>/dev/null)
  h5=$(find $O/${m}_BALF -name "*.h5" 2>/dev/null | wc -l)
  hops=$(grep -c "Hop events" $L 2>/dev/null)
  echo "$m: DONE=$done h5=$h5 NaN_lines=$nan hop_log_blocks=$hops"
  echo "  --- tail ---"; tail -6 $L 2>/dev/null | sed 's/^/    /'
done
echo "KMLIP_SMOKE_DONE $(date)"

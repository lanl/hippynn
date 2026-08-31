#!/bin/bash
#SBATCH --job-name=ref6hfd
#SBATCH --account=y2020-bf
#SBATCH --partition=shared-spr
#SBATCH --qos=long
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -o /vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/logs/ref6_%x_%j.out
#SBATCH -e /vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/logs/ref6_%x_%j.out
# enol init3, n_states=4 (-> NS=5 S0-S4), hamiltonian_fd + decohere_on_hop=True (baked in run_fssh_ref),
# 1000 Vishikh IC (same coords.xyz/velocity.out the MLIP used), 6 nodes x 112 = 672 one-FSSH-per-core. Resume-safe.
set -o pipefail
: ${FORM:?}; : ${INIT:?}; : ${NSTATES:?}; NIC=${NIC:-1000}; START=${START:-0}; NSTEPS=${NSTEPS:-20000}
export B=/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
export OUTDIR=${OUTDIR:-$B/out/enol_init3_hfd_decohere} PREFIX_BASE=${PREFIX_BASE:-enol_s3hfd} TDC=${TDC:-hamiltonian_fd}
export FORM INIT NSTATES NIC START NSTEPS OUTDIR PREFIX_BASE TDC
echo "6-node ref FORM=$FORM INIT=$INIT NSTATES=$NSTATES NIC=$NIC TDC=$TDC OUT=$OUTDIR ntasks=$SLURM_NTASKS $(date)"
srun --cpu-bind=cores $B/ref_worker_tdc.sh
echo "REF6_DONE FORM=$FORM INIT=$INIT TDC=$TDC $(date) ok=$(grep -l REF_DONE $B/logs/${PREFIX_BASE}_i*.log 2>/dev/null|wc -l)/$NIC"

#!/bin/bash
#SBATCH --job-name=ns3ref
#SBATCH --account=y2020-bf
#SBATCH --partition=shared-spr
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH -o /vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/logs/ns3ref/ns3ref_%j.out
#SBATCH -e /vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/logs/ns3ref/ns3ref_%j.out
# n_states=3 (S0-S3, init S3) reference matched to the S0-S3 MLIP: hamiltonian_fd + decohere, 100 matched Vishikh IC.
# Apples-to-apple partner for the S0-S3 MLIP (joint GALF 17187396, idx 0-99). Same ICs, seeds (422+idx).
set -o pipefail
FORM=enol; TDC=hamiltonian_fd; INIT=3; NSTATES=3; NIC=100; START=0; NSTEPS=20000
export B=/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
export OUTDIR=$B/out/ns3ref/enol_hamiltonian_fd
export PREFIX_BASE=enol_ns3_hfd
export FORM INIT NSTATES NIC START NSTEPS TDC OUTDIR PREFIX_BASE
mkdir -p $B/logs/ns3ref
echo "ns3ref FORM=$FORM TDC=$TDC INIT=$INIT NSTATES=$NSTATES NIC=$NIC NSTEPS=$NSTEPS ntasks=$SLURM_NTASKS $(date)"
srun --cpu-bind=cores $B/ns3_worker.sh
echo "NS3REF_DONE ok=$(grep -l REF_DONE $B/logs/ns3ref/${PREFIX_BASE}_i*.log 2>/dev/null|wc -l)/$NIC"

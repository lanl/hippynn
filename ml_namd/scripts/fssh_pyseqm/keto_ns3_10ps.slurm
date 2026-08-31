#!/bin/bash
#SBATCH --job-name=keto_ns3_10ps
#SBATCH --account=y2020-bf
#SBATCH --partition=shared-spr
#SBATCH --qos=long
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -o /vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/logs/keto_ns3_10ps/keto_ns3_10ps_%j.out
#SBATCH -e /vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/logs/keto_ns3_10ps/keto_ns3_10ps_%j.out
# KETO 10 ps FSSH@PySEQM reference — IDENTICAL method to the enol ns3-1000 reference, only the
# coordinates (keto) and length (10 ps = 200000 steps) change. init=3, n_states=3, tdc=hamiltonian_fd,
# decohere_on_hop=True (baked into run_fssh_ref.py, L-081), dt=0.05 fs, store every 40 steps (2 fs).
# 8 nodes x 112 = 896 one-FSSH-per-physical-core tasks, ONE job (sidesteps QOSMaxJobsPerUser), qos=long.
# Resume-safe at trajectory granularity: ns3_1000_worker.sh skips any idx whose log already has REF_DONE
# (NO intra-trajectory checkpoint — a timed-out job resumes only whole, not-yet-finished trajectories).
# =============================================================================
# PURPOSE : KETO 10 ps FSSH@PYSEQM reference (200000 steps @ 0.05 fs). init3, n_states=3,
#           hamiltonian_fd + decohere, 1000 Vishikh ICs, 8 nodes x 112 = 896 one-FSSH-per-core, ONE job.
# INPUTS  : none (all vars hard-set below); identical METHOD to the enol ns3-1000 reference, keto coords.
# OUTPUTS : per-trajectory h5 under $OUTDIR; per-IC .log under $LOGDIR; claim files under $CLAIMDIR;
#           KETO_NS3_10PS_DONE tally. Resume-safe at trajectory granularity (REF_DONE trajs skipped).
# CONFIG  : user-tunable = the #SBATCH block above + the fixed vars below. $B = confirmed Darwin root.
# DEPENDS : $B/keto_worker_dynamic.sh + run_fssh_ref.py (worker NOT vendored here; on the cluster $B).
#           Progress monitor: $B/venv_master/bin/python $B/ref_progress.py (printed at runtime).
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -o pipefail
FORM=keto; TDC=hamiltonian_fd; INIT=3; NSTATES=3; NIC=1000; START=0; NSTEPS=200000
export B=/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
export OUTDIR=$B/out/keto_ns3_10ps/keto_hamiltonian_fd
export PREFIX_BASE=keto_ns3_hfd
export LOGDIR=$B/logs/keto_ns3_10ps
export CLAIMDIR=$B/claim_keto_ns3_10ps
export FORM INIT NSTATES NIC START NSTEPS TDC OUTDIR PREFIX_BASE LOGDIR CLAIMDIR
mkdir -p "$OUTDIR" "$LOGDIR"
rm -rf "$CLAIMDIR"; mkdir -p "$CLAIMDIR"    # fresh claims each (re)start; REF_DONE trajs still skipped -> resume-safe
echo "keto-ns3-10ps FORM=$FORM TDC=$TDC INIT=$INIT NSTATES=$NSTATES NIC=$NIC NSTEPS=$NSTEPS OUT=$OUTDIR ntasks=$SLURM_NTASKS $(date) [DYNAMIC queue, 8 nodes]"
echo "MONITOR: $B/venv_master/bin/python $B/ref_progress.py $OUTDIR $LOGDIR --nsteps $NSTEPS --nic $NIC --watch 300"
srun --cpu-bind=cores $B/keto_worker_dynamic.sh
echo "KETO_NS3_10PS_DONE ok=$(grep -l REF_DONE $LOGDIR/${PREFIX_BASE}_i*.log 2>/dev/null | wc -l)/$NIC $(date)"

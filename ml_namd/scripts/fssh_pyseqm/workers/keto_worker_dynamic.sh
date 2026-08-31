#!/bin/bash
# DYNAMIC cross-node work queue for the keto 10 ps FSSH@PySEQM reference.
# Same physics/command as ns3_1000_worker.sh (run_fssh_ref.py, identical args) — ONLY the
# trajectory->core assignment changes: instead of static round-robin (k=P; k+=NT), every core
# scans idx 0..NIC-1 and CLAIMS each via an atomic `mkdir` (atomic on shared fs, unlike flock).
# The first core to create claim/<idx> owns it; others get EEXIST and move on. A core that
# finishes a trajectory keeps scanning and picks up any still-unclaimed overflow index -> fast
# cores absorb the tail, no core is stranded with two slow keto trajectories near the walltime.
#
# Resume-safe: the SLURM script wipes $CLAIMDIR before srun, so on a restart every trajectory
# whose log lacks REF_DONE is re-claimed and re-run (there is no intra-trajectory checkpoint);
# trajectories already marked REF_DONE are skipped by the grep before any claim.
PY=$B/venv_master/bin/python
OUT=${OUTDIR}; PB=${PREFIX_BASE}; TDC=${TDC:-hamiltonian_fd}; LOGS=${LOGDIR:-$B/logs}; CLAIM=${CLAIMDIR}
mkdir -p "$OUT" "$LOGS" "$CLAIM"
for ((idx=START; idx<NIC; idx++)); do
  pfx=${PB}_i$(printf "%04d" $idx)
  grep -q REF_DONE "$LOGS/${pfx}.log" 2>/dev/null && continue     # already finished (resume) -> skip
  mkdir "$CLAIM/$(printf '%04d' $idx)" 2>/dev/null || continue     # atomic claim; EEXIST -> another core owns it
  $PY $B/run_fssh_ref.py --coords $B/ics/sampled_${FORM}/coords.xyz --velocities $B/ics/sampled_${FORM}/velocity.out \
    --idx $idx --n-states $NSTATES --initial-state $INIT --nsteps $NSTEPS --timestep 0.05 --interval 40 \
    --tdc ${TDC} --out-dir "$OUT" --prefix "$pfx" --device cpu > "$LOGS/${pfx}.log" 2>&1
done

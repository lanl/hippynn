#!/bin/bash
# one srun task per physical core; round-robin share. n_states=3 reference (run_fssh_ref.py, detect_crossings default True).
P=${SLURM_PROCID}; NT=${SLURM_NTASKS}
PY=$B/venv_master/bin/python
OUT=${OUTDIR}; PB=${PREFIX_BASE}
mkdir -p "$OUT" "$B/logs/ns3ref"
for ((k=P; k<NIC; k+=NT)); do
  idx=$((START+k)); pfx=${PB}_i$(printf "%04d" $idx)
  grep -q REF_DONE "$B/logs/ns3ref/${pfx}.log" 2>/dev/null && continue
  $PY $B/run_fssh_ref.py --coords $B/ics/sampled_${FORM}/coords.xyz --velocities $B/ics/sampled_${FORM}/velocity.out \
    --idx $idx --n-states $NSTATES --initial-state $INIT --nsteps $NSTEPS --timestep 0.05 --interval 40 \
    --tdc ${TDC} --out-dir "$OUT" --prefix "$pfx" --device cpu > "$B/logs/ns3ref/${pfx}.log" 2>&1
done

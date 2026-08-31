#!/bin/bash
# one srun task per physical core; round-robin share. n_states=3 reference (run_fssh_ref.py, detect_crossings default True).
# =============================================================================
# PURPOSE : Per-core dispatch worker for the enol n_states=3 (S0-S3) FSSH@PYSEQM reference, the
#           100-IC apple-to-apple partner. Each srun task runs one instance on one physical core and
#           calls run_fssh_ref.py once per IC (one FSSH trajectory per core). Static round-robin:
#           core P handles k = P, P+NT, P+2NT, ... (idx = START + k).
# INPUTS  : environment exported by the SLURM launcher (ns3_ref.slurm) -- $B (Darwin project root),
#           FORM, INIT, NSTATES, NIC, START, TDC; SLURM_PROCID/SLURM_NTASKS from srun.
# OUTPUTS : per-trajectory h5 under $OUT; per-IC .log under $B/logs/ns3ref/${pfx}.log.
# CONFIG  : OUT = $OUTDIR; PB = $PREFIX_BASE (both required from the launcher).
# DEPENDS : $B/run_fssh_ref.py; coords/velocity at $B/ics/sampled_${FORM}/{coords.xyz,velocity.out};
#           $B/venv_master interpreter. Resume-safe: an IC whose log has REF_DONE is skipped.
# NOTE    : behavior-preserving refactor -- header/comments only; no executable line changed.
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
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

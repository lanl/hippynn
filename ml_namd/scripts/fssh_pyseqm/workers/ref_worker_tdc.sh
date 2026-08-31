#!/bin/bash
# one srun task per physical core; round-robin share of IC indices. TDC + OUTDIR/PREFIX parameterized.
# =============================================================================
# PURPOSE : Per-core dispatch worker for the enol FSSH@PYSEQM reference. Each srun task runs one
#           instance on one physical core and calls run_fssh_ref.py once per claimed IC index
#           (one FSSH trajectory per core). Static round-robin: core P handles k = P, P+NT, P+2NT, ...
# INPUTS  : environment exported by the SLURM launcher (run_ref_6node_hfd.slurm) --
#           $B (Darwin project root), FORM, INIT, NSTATES, NIC, START, and optional
#           OUTDIR/PREFIX_BASE/TDC (defaulted below); SLURM_PROCID/SLURM_NTASKS from srun.
# OUTPUTS : per-trajectory h5 under $OUT; per-IC .log under $B/logs/${pfx}.log.
# CONFIG  : OUT defaults to $B/out/${FORM}_init${INIT}; PB to ${FORM}_s${INIT}; TDC to hamiltonian_fd.
# DEPENDS : $B/run_fssh_ref.py; coords/velocity at $B/ics/sampled_${FORM}/{coords.xyz,velocity.out};
#           $B/venv_master interpreter. Resume-safe: an IC whose log has REF_DONE is skipped.
# NOTE    : behavior-preserving refactor -- header/comments only; no executable line changed.
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
P=${SLURM_PROCID}; NT=${SLURM_NTASKS}
PY=$B/venv_master/bin/python
OUT=${OUTDIR:-$B/out/${FORM}_init${INIT}}; PB=${PREFIX_BASE:-${FORM}_s${INIT}}; TDC=${TDC:-hamiltonian_fd}
mkdir -p $OUT $B/logs
for ((k=P; k<NIC; k+=NT)); do
  idx=$((START+k)); pfx=${PB}_i$(printf "%04d" $idx)
  grep -q REF_DONE $B/logs/${pfx}.log 2>/dev/null && continue
  $PY $B/run_fssh_ref.py --coords $B/ics/sampled_${FORM}/coords.xyz --velocities $B/ics/sampled_${FORM}/velocity.out \
    --idx $idx --n-states $NSTATES --initial-state $INIT --nsteps $NSTEPS --timestep 0.05 --interval 40 \
    --tdc $TDC --out-dir $OUT --prefix $pfx --device cpu > $B/logs/${pfx}.log 2>&1
done

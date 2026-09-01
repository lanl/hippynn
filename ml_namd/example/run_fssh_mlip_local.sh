#!/bin/bash
# =============================================================================
# PURPOSE : (EXAMPLE, NO SLURM) run FSSH@MLIP -- composed decoupled Model A + Model B -- directly on a
#           workstation or interactive node. Defaults to a SMALL smoke (2 ICs, 500 steps) so it finishes
#           quickly; scale up with COUNT / NSTEPS. Uses the GPU if torch sees one, else CPU.
# USAGE   : PY=/path/to/python ./run_fssh_mlip_local.sh
#           DATASET=GALF COUNT=100 NSTEPS=20000 PY=... ./run_fssh_mlip_local.sh   # full 1 ps, 100 ICs
# ENV     : PY (interpreter with hippynn + PySEQM; default `python`), DATASET={BALF|GALF} (default BALF),
#           COUNT (default 2), NSTEPS (default 500), DEVICE (default cuda -> auto-falls-back to cpu).
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -eo pipefail
PY="${PY:-python}"
DATASET="${DATASET:-BALF}"
COUNT="${COUNT:-2}"
NSTEPS="${NSTEPS:-500}"
DEVICE="${DEVICE:-cuda}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
DRV="$HERE/scripts/fssh_mlip/namd_composed_batch.py"
MA="$HERE/models/modelA_enol_nf145/${DATASET}"
MB="$HERE/models/modelB_enol_nf145/${DATASET}"
IC="$HERE/ics/fssh_mlip"
OUT="$HERE/out/fssh_mlip_${DATASET}_local"; mkdir -p "$OUT"
for p in "$DRV" "$MA/best_model.pt" "$MB/best_model.pt" "$IC/R0_100.npy"; do
  [ -e "$p" ] || { echo "ERROR: missing $p (did you copy models/ics + dereference the scripts symlink?)" >&2; exit 2; }
done
export HIPPYNN_USE_CUSTOM_KERNELS=False PYTHONUTF8=1
echo "PY=$PY DATASET=$DATASET COUNT=$COUNT NSTEPS=$NSTEPS DEVICE=$DEVICE"
"$PY" "$DRV" \
  --modelA-dir "$MA" --modelB-dir "$MB" \
  --coords-npy "$IC/R0_100.npy" --vel-npy "$IC/V0_100.npy" --z-npy "$IC/Z.npy" \
  --states 0,1,2,3 --nacr-pairs 12,13,23 --initial-state 3 \
  --nsteps "$NSTEPS" --timestep 0.05 --count "$COUNT" --start 0 \
  --trajectory-interval 40 --log-interval 40 --device "$DEVICE" \
  --output-dir "$OUT" --output-prefix "enol_${DATASET}"
echo "EX_FSSH_MLIP_LOCAL_DONE rc=$? out=$OUT"

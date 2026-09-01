#!/bin/bash
# =============================================================================
# PURPOSE : build a SELF-CONTAINED, portable FSSH@MLIP bundle (symlinks dereferenced) that Tomas can
#           copy to a workstation and run with run_fssh_mlip_local.sh -- no repo, no SLURM, no cluster.
#           Includes: the fssh_mlip driver + its predictor bundle, the decoupled Model A + Model B for
#           the chosen cell, the 100-IC subset, the local runner, and this README.
# USAGE   : DATASET=BALF ./pack_fssh_mlip_example.sh   ->  fssh_mlip_example_<DATASET>.tar.gz
# Developer: Ajay Khanna | LANL/Sergei Lab | Sep-01-2026 | ml_namd
# =============================================================================
set -eo pipefail
DATASET="${DATASET:-BALF}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
STAGE="$(mktemp -d)"; PKG="$STAGE/fssh_mlip_example_${DATASET}"
mkdir -p "$PKG/scripts" "$PKG/models" "$PKG/ics"
# dereference the fssh_mlip scripts symlink -> real files (driver + predictor bundle)
cp -rL "$HERE/scripts/fssh_mlip"    "$PKG/scripts/fssh_mlip"
cp -rL "$HERE/scripts/analysis"     "$PKG/scripts/analysis"
cp -r  "$HERE/models/modelA_enol_nf145/$DATASET" "$PKG/models/modelA_enol_nf145_$DATASET"
cp -r  "$HERE/models/modelB_enol_nf145/$DATASET" "$PKG/models/modelB_enol_nf145_$DATASET"
cp -r  "$HERE/ics/fssh_mlip"        "$PKG/ics/fssh_mlip"
cp     "$HERE/run_fssh_mlip_local.sh" "$PKG/"
cp     "$HERE/README.md"            "$PKG/EXAMPLE_README.md"
# rewrite the local runner's model paths to the flattened package layout
sed -i "s#models/modelA_enol_nf145/\${DATASET}#models/modelA_enol_nf145_${DATASET}#; s#models/modelB_enol_nf145/\${DATASET}#models/modelB_enol_nf145_${DATASET}#" "$PKG/run_fssh_mlip_local.sh"
tar -C "$STAGE" -czf "$HERE/fssh_mlip_example_${DATASET}.tar.gz" "fssh_mlip_example_${DATASET}"
rm -rf "$STAGE"
echo "wrote $HERE/fssh_mlip_example_${DATASET}.tar.gz  ($(du -h "$HERE/fssh_mlip_example_${DATASET}.tar.gz"|cut -f1))"

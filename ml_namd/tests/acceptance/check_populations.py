#!/usr/bin/env python3
"""Acceptance test: compare harvested ML-TSH populations to the locked Table I targets.

Reads the per-seed guarded-harvest outputs (topo_*.npz from the example's 05_harvest.py), aggregates the
final survivor-conditioned classical populations as mean +/- sigma_ML (sample std, ddof=1) across the split
seeds, and compares the mean to the target `classical_ml` values within
`tolerance_sigma * max(target_sigma, FLOOR)`.

The targets file encodes the frozen manuscript Table I. It must have `"locked": true` before this test will
certify a PASS, so provisional numbers cannot be mistaken for a verified result.

Usage:
  python check_populations.py --harvest-dir <dir with topo_*.npz> --targets targets_keto.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

FLOOR = 0.03   # minimum absolute tolerance band when a target sigma is ~0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-dir", required=True, help="directory holding topo_*.npz from 05_harvest.py")
    ap.add_argument("--targets", required=True, help="JSON of frozen Table I targets (see targets_*.json)")
    a = ap.parse_args()

    with open(a.targets) as fh:
        tgt = json.load(fh)
    npzs = sorted(glob.glob(os.path.join(a.harvest_dir, "topo_*.npz")))
    if not npzs:
        sys.exit(f"no topo_*.npz in {a.harvest_dir} -- run the example's 05_harvest.py first")

    per = {1: [], 2: [], 3: []}
    for p in npzs:
        d = np.load(p)
        cd = d["clas_abcd"]
        for s in (1, 2, 3):
            per[s].append(float(cd[-1, s]))

    cml = tgt["classical_ml"]
    tol = float(tgt.get("tolerance_sigma", 2.0))
    print(f"targets: {a.targets}  (locked={tgt.get('locked', False)})  n_seeds={len(npzs)}  "
          f"tautomer={tgt.get('tautomer')} head={tgt.get('head')}")
    all_ok = True
    for s in (1, 2, 3):
        v = np.array(per[s]); mean = v.mean(); sig = v.std(ddof=1)
        tmean, tsig = cml[f"S{s}"]
        band = tol * max(tsig, FLOOR)
        ok = abs(mean - tmean) <= band
        all_ok = all_ok and ok
        print(f"  S{s}: harvested {mean:.3f} +/- {sig:.3f}   target {tmean:.3f} +/- {tsig:.3f}   "
              f"|delta|={abs(mean - tmean):.3f} <= {band:.3f} ? {'OK' if ok else 'FAIL'}")

    if not tgt.get("locked", False):
        print("\nRESULT: NOT CERTIFIED -- the targets file has locked=false. Confirm the values against the "
              "frozen Table I and set locked=true to enable certification.")
        return 2
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

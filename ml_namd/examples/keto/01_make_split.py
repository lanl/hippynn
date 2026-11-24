#!/usr/bin/env python3
"""Step 1 (optional): preview the gap-stratified split for one seed -- no training.

Reports the per-set near-CI (|GAP12| < 10 meV) fraction so you can see the equitable representation the
stratified split enforces. The actual training split is generated and saved by step 2.

Usage: python 01_make_split.py [--seed 42] [--split stratified]
"""
import argparse
import sys

import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="stratified", choices=["stratified", "random"])
    a = ap.parse_args()
    sys.path.insert(0, str(C.TOOLS))
    import make_split_masks as msm
    masks = msm.make_masks(str(C.pool_dir()), "acn_", a.seed, a.split, 0.1, 0.1)
    msm._report(masks, str(C.pool_dir()), "acn_", a.split)


if __name__ == "__main__":
    main()

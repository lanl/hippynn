#!/usr/bin/env python3
"""Wrapper that runs the UNMODIFIED production trainer (newds for A / gamma for B) with the two
campaign additions, via monkeypatch (train_frozen_variant1 pattern — production files untouched):

  1) manual_seed(cfg.seed) + cuda right before build_network  -> seeds the model INIT
     (init-seed check confirmed init is otherwise UNSEEDED, so this is load-bearing for the
      "split + init full-ensemble" design).
  2) replace DirectoryDatabase's internal random split with OUR saved, reproducible split masks
     (random or gap-stratified, from make_split_masks) -> owned + recorded index tracking.

Split type is read from the output_dir path (/random/ or /stratified/). Masks are generated from
cfg.seed, SAVED into the run leaf (parent of .../model), and injected via make_automatic_splits().

Usage: python run_train_rndsplit.py --config_file <inp> --trainer-dir <newds|gamma dir> --tools-dir <scripts/tools>
"""
import argparse, os, sys


def read_cfg(path):
    c = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); c[k.strip()] = v.strip()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--trainer-dir", required=True, help="dir containing the unmodified unified_training.py (+ its deps)")
    ap.add_argument("--tools-dir", required=True, help="dir containing make_split_masks.py")
    ap.add_argument("--reuse-masks-from", default="",
                    help="Model-A run leaf holding acn_split_mask_*.npy. When set (Model B runs), REUSE "
                         "A's exact partition instead of regenerating, so A and B share an identical "
                         "train/valid/test split (the same geometry's E/F and couplings stay together). "
                         "Exact for BOTH random and stratified; sidesteps A-pred-vs-reference gap for B.")
    a = ap.parse_args()

    sys.path.insert(0, a.tools_dir)
    sys.path.insert(0, a.trainer_dir)
    import numpy as np  # noqa
    import torch
    import make_split_masks as msm

    cfg = read_cfg(a.config_file)
    seed = int(cfg["seed"]); prefix = cfg["dataset_prefix"]
    data_dir = cfg["data_dir"]; output_dir = cfg["output_dir"].rstrip("/")
    test_size = float(cfg["test_size"]); valid_size = float(cfg["valid_size"])
    split_type = "stratified" if "/stratified/" in output_dir else "random"
    leaf = os.path.dirname(output_dir)  # parent of .../model

    # (2a) obtain the split masks. Model A GENERATES + saves; Model B REUSES A's exact masks so the
    # partition is identical (geometry i stays in the same set for E/F and for the couplings/dENACR,
    # which are all properties of that one geometry). Reuse is exact for BOTH random and stratified
    # and sidesteps the A-predicted-vs-reference gap question for B's stratified bins.
    N = int(np.load(os.path.join(data_dir, prefix + "R.npy"), mmap_mode="r").shape[0])
    if a.reuse_masks_from:
        masks = {}
        for nm in ("train", "valid", "test"):
            p = os.path.join(a.reuse_masks_from, f"{prefix}split_mask_{nm}.npy")
            if not os.path.exists(p):
                raise SystemExit(f"[rndsplit] --reuse-masks-from given but missing {p}")
            masks[nm] = np.load(p)
        cover = masks["train"].astype(int) + masks["valid"].astype(int) + masks["test"].astype(int)
        if cover.shape[0] != N or cover.min() != 1 or cover.max() != 1:
            raise SystemExit(f"[rndsplit] reused masks do not partition all {N} rows exactly once "
                             f"(len={cover.shape[0]} min={int(cover.min())} max={int(cover.max())})")
        tag = f"REUSED A masks ({split_type}) from {a.reuse_masks_from}"
    else:
        masks = msm.make_masks(data_dir, prefix, seed, split_type, test_size, valid_size)
        tag = f"GENERATED split={split_type} seed={seed}"
    msm.save_masks(masks, leaf, prefix)  # always record a copy in this run's own leaf
    cnt = {nm: int(masks[nm].sum()) for nm in ("train", "valid", "test")}
    print(f"[rndsplit] {tag} -> saved {leaf} "
          f"(train/valid/test={cnt['train']}/{cnt['valid']}/{cnt['test']})", flush=True)

    # (2b) monkeypatch DirectoryDatabase: drop internal split, inject our masks
    import hippynn
    try:  # explicit per-run kernel evidence (import already fail-closed if env=True and no backend)
        from hippynn import custom_kernels as _ck
        print(f"[rndsplit] HIPPYNN kernels ACTIVE={_ck.CUSTOM_KERNELS_ACTIVE} "
              f"AVAILABLE={_ck.CUSTOM_KERNELS_AVAILABLE} "
              f"(env HIPPYNN_USE_CUSTOM_KERNELS={os.environ.get('HIPPYNN_USE_CUSTOM_KERNELS')})", flush=True)
    except Exception as _e:
        print(f"[rndsplit] kernel-mode probe skipped: {_e}", flush=True)
    _odb = hippynn.databases.DirectoryDatabase
    def _patched_db(*aa, **kk):
        kk.pop("test_size", None); kk.pop("valid_size", None)
        db = _odb(*aa, **kk)
        msm.inject_into_db(db, masks)
        print("[rndsplit] injected saved split masks + make_automatic_splits()", flush=True)
        return db
    hippynn.databases.DirectoryDatabase = _patched_db

    # (1) monkeypatch build_network: seed init right before the network is built
    import unified_training as ut
    _ob = ut.build_network
    def _seeded_build(*aa, **kk):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        print(f"[rndsplit] seeded model init with {seed} before build_network", flush=True)
        return _ob(*aa, **kk)
    ut.build_network = _seeded_build

    # run the unmodified trainer
    sys.argv = ["unified_training.py", "--config_file", a.config_file]
    ut.main()
    print("[rndsplit] TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()

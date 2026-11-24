#!/usr/bin/env python3
"""Owned, reproducible, SAVED train/valid/test split masks for the 5-rnd-split campaign.

Two split types (same 80/10/10 fractions), both seeded + saved so the exact geometry->set
membership is recorded (row i of acn_*.npy = geometry i; the mask indexes that row order):

  random      : uniform permutation by seed.
  stratified  : bin all geometries by the REFERENCE |GAP12| (= |sE2-sE1|, straight from the
                labels in the pool), split 80/10/10 WITHIN each gap bin -> every set gets a
                proportional slice of the rare near-CI apex frames (equitable representation).

Saves acn_split_mask_{train,valid,test}.npy into out_dir (the per-seed run leaf, NOT the shared
read-only pool), asserts a complete partition, and (importable) injects them into a hippynn
DirectoryDatabase via make_automatic_splits() (the landmine-safe path, per hippynn-predefined-split-injection).

CLI (standalone verify / pre-generate):
  python make_split_masks.py --data-dir <pool> --prefix acn_ --seed 42 --split random \
      --test-size 0.1 --valid-size 0.1 --out-dir <leaf> [--nbins 20]
"""
import argparse, os
import numpy as np

GAP_FILE = "GAP12"   # reference S1-S2 gap (= sE2 - sE1), present in every pool


def make_masks(data_dir, prefix, seed, split, test_size, valid_size, nbins=20):
    """Return dict of bool masks {train,valid,test} of length N (complete partition)."""
    N = int(np.load(os.path.join(data_dir, prefix + "R.npy"), mmap_mode="r").shape[0])
    rng = np.random.default_rng(int(seed))
    nt = int(round(test_size * N)); nv = int(round(valid_size * N))
    if split == "random":
        perm = rng.permutation(N)
        test_idx, val_idx, train_idx = perm[:nt], perm[nt:nt + nv], perm[nt + nv:]
    elif split == "stratified":
        gap = np.abs(np.asarray(np.load(os.path.join(data_dir, prefix + GAP_FILE + ".npy"))).reshape(-1))
        if gap.shape[0] != N:
            raise SystemExit(f"{GAP_FILE} len {gap.shape[0]} != N {N}")
        edges = np.quantile(gap, np.linspace(0.0, 1.0, nbins + 1))
        bin_id = np.clip(np.digitize(gap, edges[1:-1], right=False), 0, nbins - 1)
        te, va, tr = [], [], []
        for b in range(nbins):
            ib = np.where(bin_id == b)[0]
            rng.shuffle(ib)                    # per-bin shuffle, seeded
            m = ib.size
            bt = int(round(test_size * m)); bv = int(round(valid_size * m))
            te.append(ib[:bt]); va.append(ib[bt:bt + bv]); tr.append(ib[bt + bv:])
        test_idx = np.concatenate(te); val_idx = np.concatenate(va); train_idx = np.concatenate(tr)
    else:
        raise SystemExit(f"unknown split '{split}' (random|stratified)")
    masks = {}
    for nm, idx in (("train", train_idx), ("valid", val_idx), ("test", test_idx)):
        m = np.zeros(N, dtype=bool); m[idx] = True; masks[nm] = m
    cover = masks["train"].astype(int) + masks["valid"].astype(int) + masks["test"].astype(int)
    assert cover.min() == 1 and cover.max() == 1, "masks must partition every row exactly once"
    masks["_N"] = N
    return masks


def save_masks(masks, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for nm in ("train", "valid", "test"):
        p = os.path.join(out_dir, f"{prefix}split_mask_{nm}.npy")
        np.save(p, masks[nm]); paths[nm] = p
    return paths


def inject_into_db(database, masks):
    """Inject saved bool masks into a hippynn DirectoryDatabase (built WITHOUT test_size/valid_size)
    and convert to global index splits up-front (robust to arr_dict shrink)."""
    import torch
    for nm in ("train", "valid", "test"):
        database.arr_dict["split_mask_" + nm] = torch.as_tensor(np.asarray(masks[nm]), dtype=torch.bool)
    database.make_automatic_splits()
    return database


def _report(masks, data_dir, prefix, split):
    N = masks["_N"]
    counts = {nm: int(masks[nm].sum()) for nm in ("train", "valid", "test")}
    print(f"[{split}] N={N}  train/valid/test = {counts['train']}/{counts['valid']}/{counts['test']} "
          f"({counts['train']/N:.3f}/{counts['valid']/N:.3f}/{counts['test']/N:.3f})")
    # apex representation: fraction of near-CI (|GAP12|<10 meV) in each set
    try:
        gap = np.abs(np.asarray(np.load(os.path.join(data_dir, prefix + GAP_FILE + ".npy"))).reshape(-1))
        apex = gap < 0.010  # eV  (< 10 meV = near-CI)
        tot_apex = int(apex.sum())
        print(f"    near-CI (|GAP12|<10meV) total={tot_apex} ({tot_apex/N:.4f}); "
              + ", ".join(f"{nm}={int((apex & masks[nm]).sum())/max(masks[nm].sum(),1):.4f}"
                          for nm in ("train", "valid", "test")) + "  (fraction-within-set)")
    except Exception as e:
        print(f"    (apex report skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--prefix", default="acn_")
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--split", required=True, choices=["random", "stratified"])
    ap.add_argument("--test-size", type=float, default=0.1); ap.add_argument("--valid-size", type=float, default=0.1)
    ap.add_argument("--nbins", type=int, default=20); ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    masks = make_masks(a.data_dir, a.prefix, a.seed, a.split, a.test_size, a.valid_size, a.nbins)
    _report(masks, a.data_dir, a.prefix, a.split)
    if a.out_dir:
        paths = save_masks(masks, a.out_dir, a.prefix)
        print("  saved:", list(paths.values()))


if __name__ == "__main__":
    main()

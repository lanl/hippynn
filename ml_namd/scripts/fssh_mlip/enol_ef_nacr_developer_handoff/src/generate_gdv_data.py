#!/usr/bin/env python3
"""Standalone generator for oracle/reference GDV input arrays.

The GDV ("gradient-difference vector") used by the GDV-aware scaled-NACR heads
(``direct_vector_v2``) is an oracle/reference quantity built from per-state
forces:

    GDV_ij(R) = F_i(R) - F_j(R)              (physical forces, F = -dE/dR)
              = grad_E_j(R) - grad_E_i(R)    (if files store gradients dE/dR)

Both conventions describe the same physical vector; the only difference is the
sign convention of the input force files.

This tool is intentionally *standalone*: it does not depend on HIPPYNN and is
not imported by ``unified_training.py`` or the hyperparameter-optimization
script.  Run it once, inspect the output, then move the resulting
``{prefix}GDV{pair}.npy`` files into the SAME directory that already holds the
rest of the dataset (``{prefix}Z.npy``, ``{prefix}R.npy``, ``{prefix}sE*.npy``,
``{prefix}F*.npy``, ``{prefix}dENACR*.npy``).  Both the unified trainer and the
hyperopt script then load everything from that single directory via
``DirectoryDatabase``.

The output file naming and on-disk layout MATCH what the V2 head and
``DirectoryDatabase`` expect:

    file  : {prefix}GDV{i}{j}.npy            (e.g. acn_GDV12.npy)
    shape : (N_FRAMES, 3 * N_ATOMS)          (atom-major: a0x,a0y,a0z,a1x,...)
    dtype : float (matches force-file dtype, cast to float32 by default)

Usage examples
--------------
    # Physical-force files (default convention): GDV = F_i - F_j
    python generate_gdv_data.py \
        --data_dir /path/to/dataset \
        --out_dir  ./gdv_inputs \
        --dataset_prefix acn_ --n_atoms 15 \
        --pairs 12 23 34 45

    # Files store gradients dE/dR instead of physical forces:
    python generate_gdv_data.py --data_dir ... --pairs 12 23 --from_gradients

Developer Information
---------------------
  __developer__: "Ajay Khanna"
  __place__: "LANL"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Pair / path helpers
# =============================================================================


def parse_pair(token: str) -> Tuple[int, int]:
    """Parse a NACR pair token such as ``12``, ``1,2`` or ``1-2``.

    Returns ``(i, j)`` with the original order preserved.  Validation that
    ``i < j`` and both ``>= 1`` is performed by the caller.
    """
    raw = token.strip()
    for sep in (",", "-"):
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Invalid pair token: {token!r}")
            return int(parts[0]), int(parts[1])
    if len(raw) != 2 or not raw.isdigit():
        raise ValueError(
            f"Invalid pair token: {token!r}. Expected a two-digit string "
            "(e.g. 12) or i,j / i-j form."
        )
    return int(raw[0]), int(raw[1])


def resolve_template(
    data_dir: Path, template: str, prefix: str, **kwargs
) -> Path:
    """Resolve a filename template relative to ``data_dir`` (unless absolute)."""
    text = template.format(prefix=prefix, **kwargs)
    path = Path(text)
    if not path.is_absolute():
        path = data_dir / path
    return path


# =============================================================================
# Array loading / normalization
# =============================================================================


def normalize_vector_array(
    arr: np.ndarray, *, n_atoms: int, name: str
) -> np.ndarray:
    """Return an array shaped ``(N_FRAMES, 3 * N_ATOMS)`` in atom-major order.

    Accepts either ``(N, N_ATOMS, 3)`` or already-flattened ``(N, 3*N_ATOMS)``.
    """
    expected = 3 * n_atoms
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1:] == (n_atoms, 3):
        return arr.reshape(arr.shape[0], expected)
    if arr.ndim == 2 and arr.shape[1] == expected:
        return arr
    raise ValueError(
        f"{name} has shape {arr.shape}; expected (N,{n_atoms},3) or "
        f"(N,{expected}). Check --n_atoms."
    )


def load_force(
    path: Path, *, n_atoms: int, name: str
) -> np.ndarray:
    """Load and normalize a force/gradient ``.npy`` array."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return normalize_vector_array(np.load(path), n_atoms=n_atoms, name=name)


# =============================================================================
# GDV computation
# =============================================================================


def compute_gdv(
    force_i: np.ndarray,
    force_j: np.ndarray,
    *,
    from_gradients: bool,
) -> np.ndarray:
    """Compute the physical GDV vector for a state pair (i, j).

    Physical-force convention (F = -dE/dR):   GDV_ij = F_i - F_j
    Gradient convention      (G =  dE/dR):    GDV_ij = G_j - G_i

    Both expressions evaluate to the same physical vector.
    """
    if from_gradients:
        return force_j - force_i
    return force_i - force_j


def zero_sum_max(vec: np.ndarray, *, n_atoms: int) -> float:
    """Max per-frame norm of the summed atomic vector (translational residual)."""
    atoms = vec.reshape(vec.shape[0], n_atoms, 3)
    sums = atoms.sum(axis=1)
    return float(np.linalg.norm(sums, axis=1).max())


# =============================================================================
# Driver
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate oracle GDV input arrays {prefix}GDV{pair}.npy from "
            "per-state force files. Standalone; no HIPPYNN dependency."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing the per-state force files.",
    )
    p.add_argument(
        "--out_dir",
        default="gdv_inputs",
        help=(
            "Directory to write {prefix}GDV{pair}.npy into. Keep this SEPARATE "
            "from the dataset dir; move the files in manually afterward."
        ),
    )
    p.add_argument("--dataset_prefix", default="acn_")
    p.add_argument("--n_atoms", type=int, default=15)
    p.add_argument(
        "--pairs",
        nargs="+",
        default=["12", "23"],
        help="State pairs (i<j, both >=1), e.g. 12 23 34 45.",
    )
    p.add_argument(
        "--force_template",
        default="{prefix}F{state}.npy",
        help="Template for force files. Fields: {prefix}, {state}.",
    )
    p.add_argument(
        "--from_gradients",
        action="store_true",
        help=(
            "Interpret force files as gradients dE/dR (not physical forces). "
            "Then GDV_ij = grad_E_j - grad_E_i (== F_i - F_j physically)."
        ),
    )
    p.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Skip a pair if its output file already exists in --out_dir.",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Output dtype for the saved GDV arrays.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"ERROR: data_dir does not exist: {data_dir}", file=sys.stderr)
        return 2

    prefix = args.dataset_prefix
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    # --- Parse + validate pairs ---
    pairs: List[Tuple[int, int]] = []
    for tok in args.pairs:
        i, j = parse_pair(tok)
        if i >= j:
            print(f"ERROR: pair must have i < j, got {tok!r}", file=sys.stderr)
            return 2
        if i < 1:
            print(
                f"ERROR: pair indices must be >= 1 (excited states), got "
                f"{tok!r}",
                file=sys.stderr,
            )
            return 2
        pairs.append((i, j))

    states = sorted({s for pair in pairs for s in pair})

    sep = "=" * 72
    print(sep)
    print("GDV INPUT GENERATION")
    print(sep)
    print(f"  Data directory:   {data_dir}")
    print(f"  Output directory: {out_dir}")
    print(f"  Dataset prefix:   {prefix}")
    print(f"  N_ATOMS:          {args.n_atoms}")
    print(f"  Pairs:            {[f'{i}{j}' for i, j in pairs]}")
    print(f"  Required states:  {states}")
    print(
        "  Force convention: "
        + (
            "gradient files (dE/dR); GDV = grad_E_j - grad_E_i"
            if args.from_gradients
            else "physical forces (F = -dE/dR); GDV = F_i - F_j"
        )
    )
    print(f"  Output dtype:     {args.dtype}")
    print(sep)

    # --- Load forces once per state ---
    forces: Dict[int, np.ndarray] = {}
    n_frames: Optional[int] = None
    for state in states:
        path = resolve_template(
            data_dir, args.force_template, prefix, state=state
        )
        arr = load_force(path, n_atoms=args.n_atoms, name=f"F{state}")
        if n_frames is None:
            n_frames = arr.shape[0]
        elif arr.shape[0] != n_frames:
            print(
                f"ERROR: F{state} has {arr.shape[0]} frames; expected "
                f"{n_frames}.",
                file=sys.stderr,
            )
            return 2
        forces[state] = arr
        print(f"  Loaded F{state}: {path}  shape={arr.shape}")

    print(sep)

    # --- Compute + write GDV per pair ---
    written = 0
    skipped = 0
    for i, j in pairs:
        pair_key = f"{i}{j}"
        out_path = out_dir / f"{prefix}GDV{pair_key}.npy"

        if args.no_overwrite and out_path.exists():
            print(f"  [skip] {out_path.name} already exists.")
            skipped += 1
            continue

        gdv = compute_gdv(
            forces[i], forces[j], from_gradients=args.from_gradients
        ).astype(out_dtype, copy=False)

        # Sanity: expected flattened shape
        expected_cols = 3 * args.n_atoms
        if gdv.ndim != 2 or gdv.shape[1] != expected_cols:
            print(
                f"ERROR: computed GDV{pair_key} has shape {gdv.shape}; "
                f"expected (N,{expected_cols}).",
                file=sys.stderr,
            )
            return 1

        np.save(out_path, gdv)
        written += 1

        norms = np.linalg.norm(gdv.reshape(gdv.shape[0], args.n_atoms, 3),
                               axis=2).sum(axis=1)
        print(
            f"  [write] {out_path.name}  shape={gdv.shape}  "
            f"mean||GDV||(sum-over-atoms)={float(np.mean(norms)):.4f}  "
            f"zero_sum_max={zero_sum_max(gdv, n_atoms=args.n_atoms):.2e}"
        )

    print(sep)
    print(f"  Wrote {written} file(s); skipped {skipped}.")
    print(f"  Next step: move {prefix}GDV*.npy from")
    print(f"    {out_dir}")
    print("  into the single dataset directory used by training, then point")
    print("  DATA_DIR / --db_path at that directory.")
    print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

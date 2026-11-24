#!/usr/bin/env python3
"""genA -- generate Model-A-predicted inputs for the decoupled Model B.

The decoupled composed stack trains Model B on the GDV/GAP that the FROZEN Model A predicts, so that
training matches inference (the composed FSSH driver feeds Model A's predicted GDV/GAP to Model B). This
is the "genA" step that sits between training Model A and training Model B:

    train Model A  ->  genA (this script)  ->  train Model B

It runs the frozen Model A (E+F provider) over a geometry pool and writes, per state pair (i, j):
    acn_GDV{ij}.npy = (F_i - F_j), shape (N, 45), float32   -- A-predicted GDV (the GDV{ij} model input)
    acn_GAP{ij}.npy = (sE_j - sE_i), shape (N,),  float64   -- A-predicted gap (the GAP{ij} model input)
It also symlinks acn_R / acn_Z and the ORACLE acn_dENACR{ij} TARGETS into the output directory, and stamps
`genA_source.json` with the Model A checkpoint path + sha256 -- the train-A == deploy-A record (the FSSH
deploy Model A must be the same checkpoint that produced these inputs).

IMPORTANT (DirectoryDatabase caveat): the training DirectoryDatabase loads each input array by
``prefix + db_name`` (e.g. ``acn_GDV12.npy``). The A-predicted arrays this script writes MUST live in a
directory that does NOT also contain reference/oracle ``acn_GDV*`` / ``acn_GAP*`` copies, or the trainer
will silently read the oracle values instead of the A-predicted ones.

Environment / arguments:
    MODELA_DIR  (required)   frozen Model A directory (contains best_model.pt + experiment_structure.pt).
    DATA_DIR    (required)   geometry pool to run Model A over (holds acn_R.npy, acn_Z.npy, and the oracle
                             acn_dENACR{ij}.npy targets).
    MODELB_OUT  (required)   output directory for the A-predicted GDV/GAP + the symlinked targets (the genA
                             pool that Model B then trains on).
    GENA_TRAIN_DIR (optional) sys.path directory holding the trainer modules needed to unpickle Model A
                             (default: the sibling ../training directory in this repository).

Usage:
    MODELA_DIR=.../model DATA_DIR=.../pool MODELB_OUT=.../genA_pool \
        python gen_modelA_inputs.py --tautomer enol

Developer Information
---------------------
  __developer__: "Ajay Khanna"
  __place__: "LANL/Sergei Lab"
  __project__: "ml_namd"
"""
import os, sys, argparse

os.environ["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"
import numpy as np, torch

torch.set_default_dtype(torch.float32)   # Model A is float32; float64 + dense HIP-NN can OOM

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TRAIN = os.path.join(os.path.dirname(_HERE), "training")   # ml_namd/scripts/training
TRAIN = os.environ.get("GENA_TRAIN_DIR", _DEFAULT_TRAIN)
sys.path.insert(0, TRAIN)

_orig_load = torch.load
def _cpu_load(*a, **k):
    k["map_location"] = torch.device("cpu"); return _orig_load(*a, **k)
torch.load = _cpu_load

import unified_training, direct_nacr_head_v2, direct_nacr_head  # noqa: E402,F401
import __main__ as _M
for _n in dir(unified_training):
    if not _n.startswith("__"):
        setattr(_M, _n, getattr(unified_training, _n))
from hippynn.graphs import GraphModule
from hippynn.experiment.serialization import load_model_from_cwd

PAIRS = [(1, 2), (1, 3), (2, 3)]


def _require(name):
    """Return a required environment variable, or exit with an actionable message."""
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"set the {name} environment variable (see the module docstring)")
    return val


def main():
    ap = argparse.ArgumentParser(description="genA: write Model-A-predicted GDV/GAP for Model B training.")
    ap.add_argument("--tautomer", required=True, choices=["enol", "keto"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-frames", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250)
    a = ap.parse_args()
    t = a.tautomer
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    MODEL = _require("MODELA_DIR")
    DATA = _require("DATA_DIR")
    OUT = _require("MODELB_OUT")
    os.makedirs(OUT, exist_ok=True)
    states = sorted({s for p in PAIRS for s in p})   # {1,2,3} for {12,13,23}
    print(f"[genA] TRAIN={TRAIN}\n[genA] MODELA_DIR={MODEL}\n[genA] DATA_DIR={DATA}\n[genA] MODELB_OUT={OUT}", flush=True)

    os.chdir(MODEL)
    model = load_model_from_cwd()

    def inp(kind):
        for n in model.input_nodes:
            if kind in type(n).__name__:
                return n
        raise SystemExit(f"no {kind} input")

    Zn, Rn = inp("Species"), inp("Positions")

    def by_db(name):
        hits = [n for n in model.nodes_to_compute if getattr(n, "db_name", None) == name]
        if len(hits) != 1:
            raise SystemExit(f"{name}: {len(hits)} nodes")
        return hits[0]

    enodes = [by_db(f"sE{s}") for s in states]
    fnodes = [by_db(f"F{s}") for s in states]
    pruned = GraphModule([Zn, Rn], enodes + fnodes).to(dev)
    fdtype = next(p.dtype for p in pruned.parameters() if p.is_floating_point())

    R = np.load(f"{DATA}/acn_R.npy"); Z = np.load(f"{DATA}/acn_Z.npy")
    N = R.shape[0] if a.n_frames == 0 else min(a.n_frames, R.shape[0])
    R, Z = R[:N], (Z[:N] if Z.ndim == 2 else np.tile(Z, (N, 1)))
    print(f"[{t}] dev={dev} N={N} states={states} chunk={a.chunk} fdtype={fdtype}", flush=True)

    ns = len(states)
    sE = {s: np.empty(N) for s in states}
    F = {s: np.empty((N, 15, 3)) for s in states}
    for c in range(0, N, a.chunk):
        e = min(c + a.chunk, N)
        Zt = torch.as_tensor(Z[c:e], dtype=torch.long, device=dev)
        Rt = torch.as_tensor(R[c:e], dtype=fdtype, device=dev)
        out = pruned(Zt, Rt)
        for k, s in enumerate(states):
            sE[s][c:e] = out[k].detach().cpu().numpy().reshape(-1)
            F[s][c:e] = out[ns + k].detach().cpu().numpy().reshape(-1, 15, 3)
        print(f"  frames {c}-{e}", flush=True)

    for i, j in PAIRS:
        agdv = (F[i].astype(np.float64) - F[j].astype(np.float64)).reshape(N, -1).astype(np.float32)
        gap = (sE[j].astype(np.float64) - sE[i].astype(np.float64))
        np.save(f"{OUT}/acn_GDV{i}{j}.npy", agdv)
        np.save(f"{OUT}/acn_GAP{i}{j}.npy", gap)
        orc = f"{DATA}/acn_GDV{i}{j}.npy"; msg = ""
        if os.path.exists(orc):
            o = np.load(orc)[:N].reshape(N, -1).astype(np.float64)
            mae = np.abs(agdv.astype(np.float64) - o).mean(); rel = mae / (np.abs(o).mean() + 1e-30)
            msg = f"  MAE(A-GDV vs oracle)={mae:.4e} rel={rel:.3e}"
        print(f"  pair {i}{j}: acn_GDV{agdv.shape} acn_GAP{gap.shape} |gap|mean={np.abs(gap).mean():.4f}{msg}", flush=True)
    for f in ["acn_R.npy", "acn_Z.npy"] + [f"acn_dENACR{i}{j}.npy" for i, j in PAIRS]:
        src = f"{DATA}/{f}"
        if os.path.exists(src) and not os.path.exists(f"{OUT}/{f}"):
            os.symlink(src, f"{OUT}/{f}")
    try:
        import hashlib as _hl, json as _js, time as _tm
        _bm = os.path.join(MODEL, "best_model.pt")
        _hh = _hl.sha256()
        with open(_bm, "rb") as _f:
            for _b in iter(lambda: _f.read(1 << 20), b""):
                _hh.update(_b)
        _h = _hh.hexdigest()
        _js.dump({"modelA_dir": os.path.realpath(MODEL), "best_model_sha256": _h,
                  "stamped": _tm.strftime("%Y-%m-%dT%H:%M:%S%z")},
                 open(os.path.join(OUT, "genA_source.json"), "w"), indent=2)
        print(f"[genA] stamped genA_source.json modelA={MODEL} sha256={_h[:16]} (train-A == deploy-A record)", flush=True)
    except Exception as _e:
        print(f"[genA] WARNING: could not stamp genA_source.json: {_e}", flush=True)
    print(f"DONE {t} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

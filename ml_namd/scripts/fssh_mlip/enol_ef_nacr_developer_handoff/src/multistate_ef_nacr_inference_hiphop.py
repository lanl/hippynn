#!/usr/bin/env python3
"""
Multistate ENERGY + FORCE + scaled-NACR (dENACR) inference with a HIPPYNN model.

Predicts sE0..sE5, F0..F5, and dENACR{ij} for the ten S1-S5 pairs, keeping the
graph's oracle GDV inputs:

    GraphModule([Z, R, GDV_ij...], [sE_n..., F_n..., dENACR_ij...])

E / F / NACR are independently switchable. With --no-nacr this degenerates to
`multistate_ef_inference_hiphop.py` (the GDV inputs are pruned away entirely).

###########################################################################
#  READ THIS BEFORE USING --gdv-source forces                             #
#                                                                          #
#  The direct_vector_v2 NACR head was TRAINED ON ORACLE GDV: ground-truth  #
#  GDV_ij = F_i - F_j built from the reference (PYSEQM/CIS) forces, fed in #
#  as an INPUT. The head has `GDV_SOURCE = file` hard-coded and cannot     #
#  compute GDV internally, so scaled NACRs CANNOT be predicted standalone  #
#  from (Z, R) alone -- an external GDV must always be supplied.           #
#                                                                          #
#  `--gdv-source forces` closes the loop with the model's OWN predicted    #
#  F1..F5. That is a TRAIN/INFERENCE MISMATCH: at training time the head   #
#  saw exact GDV; at inference it would see GDV carrying the force model's #
#  error (test F MAE 4.2 -> 23.4 meV/A, rising with state index). The      #
#  resulting dENACR has NOT been validated. It is provided as the hook the #
#  PySEQM developer asked for, not as a certified prediction path.         #
#                                                                          #
#  `--gdv-source file` (the default) reproduces the validated behaviour.   #
###########################################################################

CONVENTIONS THAT ARE EASY TO GET WRONG
--------------------------------------
* `GDV_ij = F_i - F_j` -- physical forces, NO division by 2, NO normalization.
  Subtract in float64, THEN cast to float32. Shape (N, 3*n_atoms) = (N, 45).
  Verified bitwise against `scripts/dataset/generate_gdv_data.py` for all 10 pairs.
* Pair labels are 1-indexed EXCITED states: `GDV12` = S1 - S2. GDV therefore
  needs F1..F5 and NEVER F0.
* Output is the NATIVE `dENACR{ij}` (no division by the energy gap).
* The model was trained with PHASESIGN_CORRECTED_SCALED_NACR = False, so the
  SIGN of dENACR is arbitrary per frame. The trial's headline `dENACR-MAE` is
  hippynn's PHASE-LESS metric (one sign per frame over the whole 45-vector);
  the raw elementwise error is 3-6x larger and is reported as `StdMAE`.
  Any comparison against reference NACRs must be phase-aware.

ENERGY CONVENTION
-----------------
`sE_n = E_n - E0_min` (one shared shift). `E_abs = sE_n + offset`,
`gap = sE_n - sE_0`. offset = -1391.422970121047 eV for enol AcAc.

AUTOGRAD / MEMORY
-----------------
Forces are GradientNodes -- NEVER evaluate under `torch.no_grad()`.
The NACR head allocates `n_sensitivities * n_features^2 * batch` floats PER PAIR
(29 * 193^2 * batch), times 10 pairs, with the graph retained for 6 force
gradients. chunk=200 OOMs a 40 GB A100; chunk=16 peaks at ~4.6 GiB. Default 16.

Provenance: enol AcAc, HPO job 41970606, trial f310c8da. Validated on 5000
held-out frames (Darwin job 17116275): all 22 quantities at 0.98-1.02x the
trial's own test MAE, with oracle GDV.

Examples
--------
  # validated path: oracle GDV from disk, frames selected by index
  python multistate_ef_nacr_inference_hiphop.py \
      --model-dir /vast/home/akhanna2/data/models/enol_ef_nacr_s0s5_f310c8da \
      --xyz-file geom/heldout_frame_50000.xyz \
      --gdv-source file --gdv-dir .../gdv_100k --gdv-prefix acn_ --gdv-frames 50000 \
      --format both

  # E+F only (GDV inputs pruned away)
  python multistate_ef_nacr_inference_hiphop.py ... --no-nacr

Developer: Ajay Khanna
LANL / Triad National Security, LLC.  LA-UR-XX-XXXXX
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

def _wants_float64(argv):
    """True if --dtype float64 was requested, in either `--dtype x` or `--dtype=x` form."""
    for k, tok in enumerate(argv):
        if tok == "--dtype" and k + 1 < len(argv) and argv[k + 1] == "float64":
            return True
        if tok == "--dtype=float64":
            return True
    return False


# hippynn reads HIPPYNN_USE_CUSTOM_KERNELS at IMPORT time. Triton has no float64 path.
if _wants_float64(sys.argv):
    os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # handoff: head modules sit beside this script
import direct_nacr_head_v2  # noqa: F401,E402  MUST precede the unpickle

from hippynn.experiment.serialization import load_model_from_cwd  # noqa: E402
from hippynn.graphs import GraphModule  # noqa: E402
from hippynn.tools import active_directory  # noqa: E402

from seqm.api import read_xyz  # noqa: E402

DEFAULT_OFFSET = -1391.422970121047
EXPECTED_SPECIES = [8, 8, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
ALL_PAIRS = ["12", "13", "14", "15", "23", "24", "25", "34", "35", "45"]
MAX_STATE = 5
N_ATOMS = 15


# =========================================================================
#  THE GDV HOOK -- this is the single seam the PySEQM developer plugs into
# =========================================================================
def get_gdv(pairs, *, forces=None, gdv_dir=None, prefix=None, frames=None):
    """Return {pair: (N, 3*n_atoms) float32} oracle-GDV arrays.

    Exactly one of `forces` or `gdv_dir` must be given.

    forces : dict {state_index -> array (N, n_atoms, 3) or (N, 3*n_atoms)}
        Physical forces for states 1..5. Computes `GDV_ij = F_i - F_j`, the
        subtraction done in float64 and cast to float32 afterwards -- this
        reproduces `scripts/dataset/generate_gdv_data.py` BITWISE.
        NOTE: supplying *predicted* forces here is a train/inference mismatch;
        the head was teacher-forced on ground-truth GDV. See the banner above.

    gdv_dir : path
        Directory of `{prefix}GDV{ij}.npy`, each (N_total, 45) float32.
    prefix : str, e.g. "acn_"
    frames : array-like of int, optional
        Row indices to slice out of the on-disk arrays.

    Pair labels are 1-indexed EXCITED states: "12" -> S1 - S2, so `forces` must
    provide states 1..5 and never state 0.
    """
    if (forces is None) == (gdv_dir is None):
        raise ValueError("get_gdv: pass exactly one of `forces` or `gdv_dir`")

    for p in pairs:
        i, j = int(p[0]), int(p[1])
        if not (1 <= i < j <= MAX_STATE):
            raise ValueError(f"bad pair {p!r}: need 1 <= i < j <= {MAX_STATE} (excited-state labels)")

    out = {}
    if gdv_dir is not None:
        gdv_dir = Path(gdv_dir)
        for p in pairs:
            arr = np.load(gdv_dir / f"{prefix}GDV{p}.npy", mmap_mode="r")
            arr = np.asarray(arr[frames]) if frames is not None else np.asarray(arr)
            arr = arr.reshape(-1, 3 * N_ATOMS).astype(np.float32, copy=False)
            out[p] = arr
        return out

    need = sorted({int(c) for p in pairs for c in p})
    missing = [s for s in need if s not in forces]
    if missing:
        raise ValueError(f"get_gdv(forces=...): missing states {missing}; need {need} (never state 0)")
    F = {s: np.asarray(forces[s], dtype=np.float64).reshape(-1, 3 * N_ATOMS) for s in need}
    for p in pairs:
        i, j = int(p[0]), int(p[1])
        out[p] = (F[i] - F[j]).astype(np.float32, copy=False)   # f64 subtract, then cast
    return out


# ---------------------------------------------------------------- config file
def _coerce(v):
    v = v.strip()
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    try:
        if "." not in v and "e" not in v.lower():
            return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_infile(path):
    cfg = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line[0] in "#!":
            continue
        for ch in "#!":
            i = line.find(ch)
            if i > 0:
                line = line[:i].strip()
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        cfg[k.strip().lower().replace("-", "_")] = _coerce(v)
    return cfg


def parse_states(spec):
    states = [spec] if isinstance(spec, int) else [
        int(s) for s in str(spec).replace(" ", "").split(",") if s
    ]
    for s in states:
        if not 0 <= s <= MAX_STATE:
            raise ValueError(f"state {s} out of range 0..{MAX_STATE}")
    return sorted(set(states))


def parse_pairs(spec):
    if spec is None:
        return list(ALL_PAIRS)
    pairs = [s for s in str(spec).replace(" ", "").split(",") if s]
    for p in pairs:
        if p not in ALL_PAIRS:
            raise ValueError(f"unknown pair {p!r}; valid: {ALL_PAIRS}")
    return sorted(set(pairs), key=ALL_PAIRS.index)


def parse_frames(spec):
    if spec is None:
        return None
    return np.array([int(s) for s in str(spec).replace(",", " ").split()], dtype=np.int64)


# ------------------------------------------------------------------ the model
def node_by_db_name(nodes, name):
    hits = [n for n in nodes if getattr(n, "db_name", None) == name]
    if len(hits) != 1:
        raise ValueError(f"expected 1 node named {name!r}, found {len(hits)}")
    return hits[0]


def build_graph(model, states, pairs, want_e, want_f, want_nacr, device, dtype):
    z = node_by_db_name(model.input_nodes, "Z")
    r = node_by_db_name(model.input_nodes, "R")
    inputs = [z, r]
    in_names = ["Z", "R"]
    if want_nacr:
        for p in pairs:
            inputs.append(node_by_db_name(model.input_nodes, f"GDV{p}"))
            in_names.append(f"GDV{p}")

    outputs, keys = [], []
    for s in states:
        if want_e:
            outputs.append(node_by_db_name(model.nodes_to_compute, f"sE{s}")); keys.append(f"sE{s}")
    for s in states:
        if want_f:
            outputs.append(node_by_db_name(model.nodes_to_compute, f"F{s}")); keys.append(f"F{s}")
    if want_nacr:
        for p in pairs:
            outputs.append(node_by_db_name(model.nodes_to_compute, f"dENACR{p}")); keys.append(f"dENACR{p}")
    if not outputs:
        raise ValueError("nothing to compute (all of --no-energies/--no-forces/--no-nacr)")

    try:
        graph = GraphModule(inputs, outputs)
    except ValueError as e:
        raise SystemExit(
            f"[prune] GraphModule rejected the requested subset: {e}\n"
            "        The dENACR head mixes pairs, so a pair subset may still need other "
            "GDV inputs. Request all 10 pairs, or drop --nacr."
        ) from e
    graph.eval()
    graph.to(device=device, dtype=dtype)
    print(f"[prune] inputs={in_names}")
    print(f"[prune] outputs={keys}")
    return graph, in_names, keys


def load_geometries(xyz_files, skip_check):
    files = [str(Path(f).expanduser().resolve()) for f in xyz_files]
    species, coords = read_xyz(files, sort=False)   # sort=True would reorder by decreasing Z
    if not skip_check:
        for i, row in enumerate(species):
            got = [int(z) for z in row if int(z) > 0]
            if got != EXPECTED_SPECIES:
                raise ValueError(f"{files[i]}: species order {got} != training order {EXPECTED_SPECIES}")
    return species, coords, files


def run(graph, in_names, keys, species, coords, gdv, device, dtype, chunk, need_grad):
    n = species.shape[0]
    store = {k: [] for k in keys}
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        pool = {
            "Z": torch.as_tensor(species[sl], dtype=torch.int64, device=device),
            # fresh tensor: GraphModule.forward calls requires_grad_() IN PLACE
            "R": torch.as_tensor(coords[sl], dtype=dtype, device=device).clone(),
        }
        if need_grad:
            pool["R"].requires_grad_(True)
        for name in in_names:
            if name.startswith("GDV"):
                pool[name] = torch.as_tensor(gdv[name[3:]][sl], dtype=dtype, device=device)
        with torch.enable_grad():          # NEVER torch.no_grad(): F_n is a GradientNode
            outs = graph(*(pool[nm] for nm in in_names))
        if not isinstance(outs, (list, tuple)):
            outs = (outs,)
        for k, v in zip(keys, outs):
            store[k].append(v.detach().to("cpu").numpy())
        del outs, pool
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {k: np.concatenate(v, axis=0) for k, v in store.items()}


# ---------------------------------------------------------------------- main
def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--infile", "-i", default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infile", "-i", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--xyz-file", nargs="+", default=None)
    p.add_argument("--states", default="0,1,2,3,4,5")
    p.add_argument("--energies", dest="energies", action="store_true", default=True)
    p.add_argument("--no-energies", dest="energies", action="store_false")
    p.add_argument("--forces", dest="forces", action="store_true", default=True)
    p.add_argument("--no-forces", dest="forces", action="store_false")
    p.add_argument("--nacr", dest="nacr", action="store_true", default=True)
    p.add_argument("--no-nacr", dest="nacr", action="store_false")
    p.add_argument("--nacr-pairs", default=None, help="default: all 10")
    p.add_argument("--gdv-source", default="file", choices=("file", "forces"))
    p.add_argument("--gdv-dir", default=None)
    p.add_argument("--gdv-prefix", default="acn_")
    p.add_argument("--gdv-frames", default=None,
                   help="row indices into the on-disk GDV arrays, one per xyz frame")
    p.add_argument("--energy-offset", type=float, default=DEFAULT_OFFSET)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    p.add_argument("--chunk", type=int, default=16,
                   help="frames per forward; the NACR head is the memory hog (200 OOMs a 40GB A100)")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--output-prefix", default="multistate_ef_nacr")
    p.add_argument("--format", dest="fmt", default="table", choices=("table", "npz", "json", "both"))
    p.add_argument("--skip-species-check", action="store_true")
    if pre_args.infile:
        p.set_defaults(**parse_infile(pre_args.infile))
    a = p.parse_args()

    if not a.model_dir or not a.xyz_file:
        p.error("--model-dir and --xyz-file are required")
    if isinstance(a.xyz_file, str):
        a.xyz_file = a.xyz_file.split()

    states = parse_states(a.states)
    pairs = parse_pairs(a.nacr_pairs) if a.nacr else []
    frames = parse_frames(a.gdv_frames)
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if a.device == "auto" else a.device))

    species, coords, files = load_geometries(a.xyz_file, a.skip_species_check)
    nframes, natoms = species.shape
    print(f"[geom] {nframes} frame(s), {natoms} atoms")

    with active_directory(str(a.model_dir)):
        model = load_model_from_cwd(map_location="cpu")
    model.eval()
    n_gdv = sum(1 for n in model.input_nodes if str(getattr(n, "db_name", "")).startswith("GDV"))
    print(f"[prune] full graph: {len(model.input_nodes)} inputs ({n_gdv} oracle GDV), "
          f"{len(model.nodes_to_compute)} outputs")

    # ---- GDV ----
    gdv = {}
    if a.nacr:
        if a.gdv_source == "file":
            if not a.gdv_dir:
                p.error("--gdv-source file requires --gdv-dir")
            if frames is not None and len(frames) != nframes:
                p.error(f"--gdv-frames has {len(frames)} entries but {nframes} geometries were given")
            gdv = get_gdv(pairs, gdv_dir=a.gdv_dir, prefix=a.gdv_prefix, frames=frames)
            print(f"[gdv ] oracle GDV from {a.gdv_dir} ({len(pairs)} pairs)  -- VALIDATED PATH")
        else:
            print("=" * 74)
            print("WARNING: --gdv-source forces. The NACR head was TRAINED ON ORACLE GDV.")
            print("Feeding it the model's own predicted forces is a train/inference")
            print("mismatch and the resulting dENACR is NOT validated. See the module")
            print("docstring. Use --gdv-source file for the certified path.")
            print("=" * 74)
            need = sorted({int(c) for q in pairs for c in q})     # 1..5, never 0
            # Pass 1: prune to [Z,R] -> [F_need...] and predict the forces.
            fgraph, fin_names, fkeys = build_graph(model, need, [], False, True, False, device, dtype)
            fpred = run(fgraph, fin_names, fkeys, species, coords, {}, device, dtype,
                        max(a.chunk, 64), need_grad=True)
            gdv = get_gdv(pairs, forces={s: fpred[f"F{s}"].reshape(nframes, natoms, 3) for s in need})
            print(f"[gdv ] self-consistent GDV from PREDICTED F{need[0]}..F{need[-1]} ({len(pairs)} pairs)")
        for q in pairs:
            if gdv[q].shape != (nframes, 3 * natoms):
                raise ValueError(f"GDV{q} shape {gdv[q].shape}, expected {(nframes, 3*natoms)}")

    graph, in_names, keys = build_graph(model, states, pairs, a.energies, a.forces, a.nacr, device, dtype)
    pred = run(graph, in_names, keys, species, coords, gdv, device, dtype, a.chunk, need_grad=a.forces)

    # ---- assemble ----
    offset = float(a.energy_offset)
    have_gs = 0 in states and a.energies
    arrays = {}
    if a.energies:
        for s in states:
            arrays[f"sE{s}"] = pred[f"sE{s}"].reshape(nframes)
            # float64: |offset| ~ 1391 eV, where float32's ulp (~1e-4 eV) is the
            # same order as the model's own sE0 error.
            arrays[f"E_abs{s}"] = arrays[f"sE{s}"].astype(np.float64) + offset
        if have_gs:
            for s in states:
                arrays[f"gap{s}"] = (arrays[f"sE{s}"].astype(np.float64)
                                     - arrays["sE0"].astype(np.float64))
    if a.forces:
        for s in states:
            arrays[f"F{s}"] = pred[f"F{s}"].reshape(nframes, natoms, 3)
    for q in pairs:
        arrays[f"dENACR{q}"] = pred[f"dENACR{q}"].reshape(nframes, 3 * natoms)

    meta = {
        "model_dir": str(a.model_dir), "xyz_files": files, "states": states, "nacr_pairs": pairs,
        "energy_offset_eV": offset, "device": str(device), "dtype": a.dtype,
        "gdv_source": a.gdv_source if a.nacr else None,
        "gdv_dir": str(a.gdv_dir) if (a.nacr and a.gdv_source == "file") else None,
        "units": {"sE_shifted": "eV", "E_abs": "eV", "gap": "eV",
                  "F": "eV/Angstrom", "dENACR": "native (as trained; no gap division)"},
        "convention": "sE_n = E_n - E0_min; E_abs = sE_n + offset; gap = sE_n - sE_0; GDV_ij = F_i - F_j",
        "pair_labels": "1-indexed EXCITED states: dENACR12 = S1-S2 (needs F1..F5, never F0)",
        "phase_warning": ("PHASESIGN_CORRECTED_SCALED_NACR=False: the sign of dENACR is arbitrary "
                          "per frame. Compare phase-lessly (one sign per frame over the 45-vector)."),
        "gdv_warning": (None if a.gdv_source == "file" else
                        "dENACR computed from PREDICTED forces: train/inference mismatch, NOT validated."),
        "provenance": "enol AcAc, HPO 41970606, trial f310c8da; held-out validation Darwin job 17116275",
    }
    results = {"meta": meta, "frames": []}
    for i in range(nframes):
        fr = {"index": i, "xyz_file": files[i], "states": {}, "nacr": {}}
        for s in states:
            d = {}
            if a.energies:
                d["sE_shifted_eV"] = float(arrays[f"sE{s}"][i])
                d["E_abs_eV"] = float(arrays[f"E_abs{s}"][i])
                if have_gs:
                    d["gap_eV"] = float(arrays[f"gap{s}"][i])
            if a.forces:
                d["F_eV_per_Angstrom"] = arrays[f"F{s}"][i].reshape(natoms, 3).tolist()
            fr["states"][f"S{s}"] = d
        for q in pairs:
            fr["nacr"][f"dENACR{q}"] = arrays[f"dENACR{q}"][i].tolist()
        results["frames"].append(fr)

    # ---- emit ----
    outdir = Path(a.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / a.output_prefix

    if a.fmt == "table" or a.fmt == "both":
        lines = []
        if a.energies or a.forces:
            hdr = f"{'frame':>5} {'state':>5}"
            if a.energies:
                hdr += f" {'sE_shifted[eV]':>16} {'E_abs[eV]':>18}" + (f" {'gap[eV]':>12}" if have_gs else "")
            if a.forces:
                hdr += f" {'|F|max[eV/A]':>14}"
            lines += [hdr, "-" * len(hdr)]
            for i in range(nframes):
                for s in states:
                    row = f"{i:>5} {'S'+str(s):>5}"
                    if a.energies:
                        row += f" {arrays[f'sE{s}'][i]:>16.9f} {arrays[f'E_abs{s}'][i]:>18.9f}"
                        if have_gs:
                            row += f" {arrays[f'gap{s}'][i]:>12.6f}"
                    if a.forces:
                        row += f" {np.abs(arrays[f'F{s}'][i]).max():>14.6f}"
                    lines.append(row)
        if pairs:
            lines += ["", f"{'frame':>5} {'pair':>10} {'|dENACR|max':>14} {'|dENACR|rms':>14}",
                      "-" * 47]
            for i in range(nframes):
                for q in pairs:
                    v = arrays[f"dENACR{q}"][i]
                    lines.append(f"{i:>5} {'dENACR'+q:>10} {np.abs(v).max():>14.6f} "
                                 f"{np.sqrt((v**2).mean()):>14.6f}")
            lines += ["", "dENACR sign is arbitrary per frame (phase not corrected)."]
        table = "\n".join(lines)
        print("\n" + table)
        Path(f"{stem}_table.txt").write_text(table + "\n")
        print(f"\n[write] {stem}_table.txt")

    if a.fmt in ("npz", "both"):
        np.savez_compressed(f"{stem}.npz", **arrays, states=np.array(states),
                            pairs=np.array(pairs), energy_offset=np.array(offset))
        print(f"[write] {stem}.npz")
    if a.fmt in ("json", "both"):
        Path(f"{stem}.json").write_text(json.dumps(results, indent=2))
        print(f"[write] {stem}.json")


if __name__ == "__main__":
    main()

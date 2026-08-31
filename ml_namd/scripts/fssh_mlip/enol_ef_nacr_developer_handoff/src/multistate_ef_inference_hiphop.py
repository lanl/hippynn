#!/usr/bin/env python3
"""
Multistate ENERGY + FORCE inference with a HIPPYNN model. No dynamics, no NACR.

Predicts sE0..sE5 and F0..F5 for one or more geometries, using the graph pruned
to `GraphModule([Z, R], [sE_n..., F_n...])`.

WHY PRUNE
---------
The trained graph's inputs are Z, R **and ten oracle GDV12..GDV45 InputNodes**
(the direct_vector_v2 NACR head consumes them). `Predictor.from_graph()` does NOT
prune, so it would demand GDV arrays this script neither has nor needs. Energies
depend only on network(Z, R), and each F_n is a GradientNode of (sE_n, R);
neither touches GDV. `GraphModule.__init__` raises if an output required an
unspecified input, so a clean construction *proves* the prune is sound.

If you need scaled NACRs, use `multistate_ef_nacr_inference_hiphop.py`, which
keeps the GDV inputs.

ENERGY CONVENTION
-----------------
The model predicts `sE_n = E_n - E0_min`, one shared shift for every state.
  * `sE_shifted` : raw model output (eV)
  * `E_abs`      : sE_n + energy_offset (eV)   [offset = -1391.422970121047 for enol AcAc]
  * `gap`        : sE_n - sE_0 (eV)            [requires state 0 in --states]

AUTOGRAD
--------
Forces are GradientNodes. NEVER evaluate under `torch.no_grad()` -- they go to
zero silently. Frames are chunked and detached per chunk so the retained graph
does not accumulate.

Provenance: enol AcAc, HPO job 41970606, trial f310c8da (n_features=193,
n_sensitivities=29, lr=2.9626e-4, seed 114514). Validated on 5000 held-out
frames (Darwin job 17116275): all quantities at 0.98-1.02x the trial's test MAE.

Examples
--------
  python multistate_ef_inference_hiphop.py \
      --model-dir /vast/home/akhanna2/data/models/enol_ef_nacr_s0s5_f310c8da \
      --xyz-file geom/heldout_frame_50000.xyz \
      --states 0,1,2,3,4,5 --format both

  python multistate_ef_inference_hiphop.py -i ef.input --no-forces

Developer: Ajay Khanna <akhanna2@ucmerced.edu>
LANL / Triad National Security, LLC.  LA-UR-XX-XXXXX
"""

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


# hippynn reads HIPPYNN_USE_CUSTOM_KERNELS at IMPORT time, so decide before
# importing it. The triton kernels have no float64 path.
if _wants_float64(sys.argv):
    os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
# `direct_nacr_head_v2` must be a TOP-LEVEL importable module before the unpickle:
# experiment_structure.pt references the custom head class by module path.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # handoff: head modules sit beside this script
import direct_nacr_head_v2  # noqa: F401,E402

from hippynn.experiment.serialization import load_model_from_cwd  # noqa: E402
from hippynn.graphs import GraphModule  # noqa: E402
from hippynn.tools import active_directory  # noqa: E402

from seqm.api import read_xyz  # noqa: E402

DEFAULT_OFFSET = -1391.422970121047        # eV, enol AcAc
EXPECTED_SPECIES = [8, 8, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
MAX_STATE = 5


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
    if isinstance(spec, int):
        states = [spec]
    else:
        states = [int(s) for s in str(spec).replace(" ", "").split(",") if s != ""]
    for s in states:
        if not 0 <= s <= MAX_STATE:
            raise ValueError(f"state {s} out of range 0..{MAX_STATE} for this checkpoint")
    if len(set(states)) != len(states):
        raise ValueError(f"duplicate states in {states}")
    return sorted(states)


# ------------------------------------------------------------------ the model
def node_by_db_name(nodes, name):
    hits = [n for n in nodes if getattr(n, "db_name", None) == name]
    if len(hits) != 1:
        raise ValueError(f"expected 1 node named {name!r}, found {len(hits)}")
    return hits[0]


def load_pruned_graph(model_dir, states, want_energies, want_forces, device, dtype):
    with active_directory(str(model_dir)):
        model = load_model_from_cwd(map_location="cpu")
    model.eval()

    n_gdv = sum(1 for n in model.input_nodes if str(getattr(n, "db_name", "")).startswith("GDV"))
    print(f"[prune] full graph: {len(model.input_nodes)} inputs ({n_gdv} oracle GDV), "
          f"{len(model.nodes_to_compute)} outputs")

    z = node_by_db_name(model.input_nodes, "Z")
    r = node_by_db_name(model.input_nodes, "R")

    outputs, keys = [], []
    # Forces are needed whenever forces are requested. Energies come along free
    # (F_n is a GradientNode of sE_n), but only emit what was asked for.
    for s in states:
        if want_energies:
            outputs.append(node_by_db_name(model.nodes_to_compute, f"sE{s}"))
            keys.append(f"sE{s}")
    for s in states:
        if want_forces:
            outputs.append(node_by_db_name(model.nodes_to_compute, f"F{s}"))
            keys.append(f"F{s}")
    if not outputs:
        raise ValueError("nothing to compute: both --no-energies and --no-forces")

    # Raises if any output needed an input outside [Z, R] -> the prune is self-validating.
    graph = GraphModule([z, r], outputs)
    graph.eval()
    graph.to(device=device, dtype=dtype)
    print(f"[prune] pruned graph: inputs=['Z','R'] outputs={keys}")
    return graph, keys


def load_geometries(xyz_files, skip_species_check):
    files = [str(Path(f).expanduser().resolve()) for f in xyz_files]
    # sort=False: sort=True reorders atoms by decreasing Z. Forces are equivariant.
    species, coords = read_xyz(files, sort=False)
    if not skip_species_check:
        for i, row in enumerate(species):
            got = [int(z) for z in row if int(z) > 0]
            if got != EXPECTED_SPECIES:
                raise ValueError(
                    f"{files[i]}: atom order does not match training convention acn_Z.\n"
                    f"  expected {EXPECTED_SPECIES}\n  got      {got}\n"
                    "Reorder the .xyz; do not sort here (forces are equivariant)."
                )
    return species, coords, files


def run(graph, keys, species, coords, device, dtype, chunk, need_grad):
    n = species.shape[0]
    store = {k: [] for k in keys}
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        Z = torch.as_tensor(species[sl], dtype=torch.int64, device=device)
        # Fresh tensor: GraphModule.forward calls requires_grad_() IN PLACE.
        R = torch.as_tensor(coords[sl], dtype=dtype, device=device).clone()
        if need_grad:
            R.requires_grad_(True)
        # NEVER under torch.no_grad(): F_n is a GradientNode.
        with torch.enable_grad():
            outs = graph(Z, R)
        if not isinstance(outs, (list, tuple)):
            outs = (outs,)
        for k, v in zip(keys, outs):
            store[k].append(v.detach().to("cpu").numpy())
        del outs, Z, R
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
    p.add_argument("--infile", "-i", default=None, help="key = value config file")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--xyz-file", nargs="+", default=None,
                   help="one or more .xyz files (one geometry each)")
    p.add_argument("--states", default="0,1,2,3,4,5", help="comma-separated, 0..5")
    p.add_argument("--energies", dest="energies", action="store_true", default=True)
    p.add_argument("--no-energies", dest="energies", action="store_false")
    p.add_argument("--forces", dest="forces", action="store_true", default=True)
    p.add_argument("--no-forces", dest="forces", action="store_false")
    p.add_argument("--energy-offset", type=float, default=DEFAULT_OFFSET)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    p.add_argument("--chunk", type=int, default=100,
                   help="frames per forward. E+F only; no NACR head, so 100-250 is safe.")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--output-prefix", default="multistate_ef")
    p.add_argument("--format", dest="fmt", default="table",
                   choices=("table", "npz", "json", "both"))
    p.add_argument("--skip-species-check", action="store_true")
    if pre_args.infile:
        p.set_defaults(**parse_infile(pre_args.infile))
    a = p.parse_args()

    if not a.model_dir or not a.xyz_file:
        p.error("--model-dir and --xyz-file are required")
    if isinstance(a.xyz_file, str):
        a.xyz_file = a.xyz_file.split()

    states = parse_states(a.states)
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if a.device == "auto" else a.device))

    species, coords, files = load_geometries(a.xyz_file, a.skip_species_check)
    nframes, natoms = species.shape
    print(f"[geom] {nframes} frame(s), {natoms} atoms, species order verified"
          f"{' (SKIPPED)' if a.skip_species_check else ''}")

    graph, keys = load_pruned_graph(a.model_dir, states, a.energies, a.forces, device, dtype)
    pred = run(graph, keys, species, coords, device, dtype, a.chunk, need_grad=a.forces)

    # ---- assemble ----
    offset = float(a.energy_offset)
    have_gs = 0 in states and a.energies
    results = {"meta": {
        "model_dir": str(a.model_dir), "xyz_files": files, "states": states,
        "energy_offset_eV": offset, "device": str(device), "dtype": a.dtype,
        "energies": bool(a.energies), "forces": bool(a.forces),
        "units": {"sE_shifted": "eV", "E_abs": "eV", "gap": "eV", "F": "eV/Angstrom"},
        "convention": "sE_n = E_n - E0_min (one shared shift); E_abs = sE_n + offset; gap = sE_n - sE_0",
        "provenance": "enol AcAc, HPO 41970606, trial f310c8da",
        "gap_note": None if have_gs else "gap not computed: state 0 absent or --no-energies",
    }, "frames": []}

    arrays = {}
    if a.energies:
        for s in states:
            arrays[f"sE{s}"] = pred[f"sE{s}"].reshape(nframes)
            # Accumulate in float64: |offset| ~ 1391 eV, and float32 there has a
            # ~1e-4 eV (0.1 meV) ulp -- comparable to the model's own sE0 error.
            arrays[f"E_abs{s}"] = arrays[f"sE{s}"].astype(np.float64) + offset
        if have_gs:
            for s in states:
                arrays[f"gap{s}"] = (arrays[f"sE{s}"].astype(np.float64)
                                     - arrays["sE0"].astype(np.float64))
    if a.forces:
        for s in states:
            arrays[f"F{s}"] = pred[f"F{s}"].reshape(nframes, natoms, 3)

    for i in range(nframes):
        fr = {"index": i, "xyz_file": files[i], "states": {}}
        for s in states:
            d = {}
            if a.energies:
                d["sE_shifted_eV"] = float(arrays[f"sE{s}"][i])
                d["E_abs_eV"] = float(arrays[f"E_abs{s}"][i])
                if have_gs:
                    d["gap_eV"] = float(arrays[f"gap{s}"][i])
            if a.forces:
                d["F_eV_per_Angstrom"] = arrays[f"F{s}"][i].tolist()
            fr["states"][f"S{s}"] = d
        results["frames"].append(fr)

    # ---- emit ----
    outdir = Path(a.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / a.output_prefix

    if a.fmt in ("table", "both") or a.fmt == "table":
        lines = []
        hdr = f"{'frame':>5} {'state':>5}"
        if a.energies:
            hdr += f" {'sE_shifted[eV]':>16} {'E_abs[eV]':>18}" + (f" {'gap[eV]':>12}" if have_gs else "")
        if a.forces:
            hdr += f" {'|F|max[eV/A]':>14} {'|F|rms[eV/A]':>14}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for i in range(nframes):
            for s in states:
                row = f"{i:>5} {'S'+str(s):>5}"
                if a.energies:
                    row += f" {arrays[f'sE{s}'][i]:>16.9f} {arrays[f'E_abs{s}'][i]:>18.9f}"
                    if have_gs:
                        row += f" {arrays[f'gap{s}'][i]:>12.6f}"
                if a.forces:
                    F = arrays[f"F{s}"][i]
                    row += f" {np.abs(F).max():>14.6f} {np.sqrt((F**2).mean()):>14.6f}"
                lines.append(row)
        table = "\n".join(lines)
        print("\n" + table)
        (stem.with_name(stem.name + "_table.txt")).write_text(table + "\n")
        print(f"\n[write] {stem}_table.txt")

    if a.fmt in ("npz", "both"):
        np.savez_compressed(f"{stem}.npz", **arrays,
                            states=np.array(states), energy_offset=np.array(offset))
        print(f"[write] {stem}.npz")
    if a.fmt in ("json", "both"):
        Path(f"{stem}.json").write_text(json.dumps(results, indent=2))
        print(f"[write] {stem}.json")


if __name__ == "__main__":
    main()

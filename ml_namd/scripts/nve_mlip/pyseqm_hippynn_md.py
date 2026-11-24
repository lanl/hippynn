#!/usr/bin/env python3
"""
PySEQM NVE/NVT molecular dynamics driven by a HIPPYNN E+F model (single state).

Derived from PYSEQM `examples/hippynn_md/pyseqm_hippynn_md.py`, rewritten for the
enol AcAc E+F+NACR checkpoint (HPO job 41970606, trial f310c8da).

WHY THE STOCK SCRIPT CRASHES ON THIS CHECKPOINT
-----------------------------------------------
The trained graph's inputs are Z, R **and ten oracle GDV12..GDV45 InputNodes**.
`Predictor.from_graph()` does NOT prune (it takes `graph.input_nodes` verbatim),
so the stock driver reaches `raise ValueError("Unsupported HIPNN graph input
'GDV12'")` on the first step.

Fix: prune to `GraphModule([Z, R], [sE_n, F_n])`. This is provably valid --
`sE*` depends only on network(Z, R) and `F*` is a GradientNode of (sE*, R);
neither touches GDV. `GraphModule.__init__` itself raises if an output needed an
unspecified input, so the prune is self-validating.

Pruning is required ONLY for MD. The Phase-1 held-out validation deliberately ran
the FULL unpruned graph and passed (a held-out validation run: all 22 quantities at
0.98-1.02x the trial's test MAE).

OTHER DEPARTURES FROM THE STOCK SCRIPT
--------------------------------------
1. `load_model_from_cwd()` not `load_checkpoint_from_cwd()`. The latter wants
   `best_checkpoint.pt` (1.29 GB); the inference bundle ships only
   `experiment_structure.pt` + `best_model.pt` (736 MB), which is all the former needs.
2. `patch_pickle_for_hippynn()` is DELETED. The pickle references no `__main__`
   classes -- it needs the top-level module `direct_nacr_head_v2` importable
   BEFORE the unpickle, which the sys.path bootstrap below provides.
3. The graph's existing `F{state}` GradientNode (sign=-1) is REUSED. The stock
   script builds a fresh `GradientNode`, which is redundant here.
4. `read_xyz(..., sort=False)`. `sort=True` (the default) reorders atoms by
   DECREASING atomic number. For this molecule `acn_Z = [8,8,6,6,6,6,6,1x8]` is
   already in that order so it happens to be a no-op, but a stable sort only
   preserves within-element order (the five carbons) if the file already agrees.
   We pass sort=False and assert the species vector explicitly.
5. `molecule.coordinates` is NEVER handed to the graph directly. `GraphModule.forward`
   does `value.requires_grad_(...)` **in place**, which would permanently flip the
   flag on the MD integrator's own coordinate tensor. We pass a detached clone.
6. The graph call is wrapped in `torch.enable_grad()`. The forces are a
   GradientNode; evaluating under `torch.no_grad()` silently yields zeros.
7. Custom kernels are disabled (see below) because the graph is cast to float64.

Developer Information
---------------------
  __developer__: "Ajay Khanna"
  __place__: "LANL/Sergei Lab"
  __date__: "Sep-01-2026"
  __project__: "ml_namd"
"""

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# hippynn reads HIPPYNN_USE_CUSTOM_KERNELS at import time. The triton kernels are
# not float64 paths; this graph is cast to float64 for energy-conserving MD, and
# the system is 15 atoms, so the dense fallback costs nothing.
os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")

import torch

# scripts/inference/<this file>  ->  parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
# `direct_nacr_head_v2` must be importable as a TOP-LEVEL module before torch.load
# unpickles experiment_structure.pt (the pickle references it by module path).
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))
# the cluster: repo scripts/train lives on /users even when this driver runs from <SCRATCH>.
_TRAIN_DIR = "<ML_NAMD_ROOT>/scripts/train"
if os.path.isdir(_TRAIN_DIR):
    sys.path.insert(0, _TRAIN_DIR)
import direct_nacr_head_v2  # noqa: F401,E402
# Models trained by the current unified_training.py (run as __main__) pickle a
# reference to __main__.DataDumper; inject unified_training's symbols into __main__
# so torch.load can resolve them. (The older f310c8da handoff model had no such ref.)
try:
    import unified_training as _ut, direct_nacr_head as _dnh  # noqa: F401,E402
    import __main__ as _MAIN
    for _n in dir(_ut):
        if not _n.startswith("__"):
            setattr(_MAIN, _n, getattr(_ut, _n))
except Exception as _e:  # noqa: BLE001
    print(f"[warn] unified_training __main__ inject skipped: {_e}")

from hippynn.experiment.serialization import load_model_from_cwd  # noqa: E402
from hippynn.graphs import GraphModule  # noqa: E402
from hippynn.tools import active_directory  # noqa: E402

from seqm.api import read_xyz  # noqa: E402
from seqm.MolecularDynamics import (  # noqa: E402
    Molecular_Dynamics_Basic,
    Molecular_Dynamics_Langevin,
)
from seqm.Molecule import Molecule  # noqa: E402
from seqm.seqm_functions.constants import Constants  # noqa: E402

# enol AcAc, matching acn_Z. O,O then five C then eight H.
EXPECTED_SPECIES = [8, 8, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]


def normalize_device(device):
    """Return a device with an explicit CUDA index.

    PySEQM's `initialize()` compares `esdriver.device` to
    `molecule.coordinates.device` and raises if they differ. A bare
    `torch.device("cuda")` (index None) never equals `torch.device("cuda:0")`,
    which is what `.to("cuda")` actually produces -- so the un-normalized form
    fails on any GPU run while working fine on CPU.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def write_final_xyz(path, molecule):
    species = molecule.species.detach().cpu()
    coordinates = molecule.coordinates.detach().cpu()
    natoms = int(torch.sum(species[0] > 0))
    lines = [str(natoms), "final geometry from pyseqm_hippynn_md.py"]
    for z, xyz in zip(species[0, :natoms], coordinates[0, :natoms]):
        symbol = molecule.const.label[int(z.item())]
        lines.append(f"{symbol:<2} {float(xyz[0]): .10f} {float(xyz[1]): .10f} {float(xyz[2]): .10f}")
    Path(path).write_text("\n".join(lines) + "\n")


def coerce(value):
    v = value.strip()
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


def parse_mdip(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    config = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line[0] in ("#", "!"):
            continue
        if line.upper().startswith(("%BLOCK", "%END")):
            continue
        for ch in ("#", "!"):
            idx = line.find(ch)
            if idx > 0:
                line = line[:idx].strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        config[key.strip().lower()] = coerce(value)
    return config


def build_parser():
    p = argparse.ArgumentParser(description="PySEQM MD driven by a HIPPYNN energy/force model.")
    p.add_argument("--infile", "-i", default=None, help="Path to .mdip/.input file.")
    p.add_argument("--xyz-file", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--state-index", type=int, default=None, help="0=S0, 1=S1, ... (max 5)")
    p.add_argument("--energy-offset", type=float, default=None, help="Constant shift in eV.")
    p.add_argument("--ensemble", choices=["nve", "nvt"], default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--timestep", type=float, default=None, help="fs")
    p.add_argument("--nsteps", type=int, default=None)
    p.add_argument("--friction", type=float, default=None, help="Langevin friction, 1/fs")
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--trajectory-interval", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--device", default="auto")
    return p


class HIPPYNN_ESDriver(torch.nn.Module):
    """Electronic-structure stand-in: HIPPYNN supplies E and F for ONE state."""

    def __init__(self, model_dir, state_index=0, energy_offset=0.0, device="auto"):
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = normalize_device(device)
        self.energy_offset = float(energy_offset)
        self.energy_key = f"sE{state_index}"
        self.force_key = f"F{state_index}"

        # PySEQM master's Molecular_Dynamics_Basic.initialize() does
        #     self.esdriver.conservative_force.energy.md = True
        # (MolecularDynamics.py:917). The real SEQM esdriver exposes that chain;
        # ours does not, and torch.nn.Module.__getattr__ raises AttributeError.
        # A plain namespace shim satisfies it. The flag is never read back for
        # NVE/NVT -- the other conservative_force accesses are XL_BOMD/XL_ESMD only.
        self.conservative_force = SimpleNamespace(energy=SimpleNamespace(md=False))

        with active_directory(model_dir):
            model = load_model_from_cwd(map_location="cpu")
        model.eval()

        def by_db_name(nodes, name):
            hits = [n for n in nodes if getattr(n, "db_name", None) == name]
            if len(hits) != 1:
                raise ValueError(
                    f"expected exactly 1 node with db_name={name!r}, found {len(hits)}. "
                    f"Available: {sorted(str(getattr(n,'db_name',None)) for n in nodes)}"
                )
            return hits[0]

        z_node = by_db_name(model.input_nodes, "Z")
        r_node = by_db_name(model.input_nodes, "R")
        e_node = by_db_name(model.nodes_to_compute, self.energy_key)
        f_node = by_db_name(model.nodes_to_compute, self.force_key)

        n_gdv = sum(1 for n in model.input_nodes if str(getattr(n, "db_name", "")).startswith("GDV"))
        print(f"[prune] full graph: {len(model.input_nodes)} inputs "
              f"({n_gdv} oracle GDV), {len(model.nodes_to_compute)} outputs")

        # THE PRUNE. GraphModule.__init__ raises if [sE_n, F_n] needed any input
        # outside [Z, R] -- so a clean construction *proves* neither touches GDV.
        self.graph = GraphModule([z_node, r_node], [e_node, f_node])
        self.graph.eval()
        self.graph.to(device=self.device, dtype=torch.float64)

        self.input_db_names = [n.db_name for n in self.graph.input_nodes]
        self.output_db_names = [getattr(n, "db_name", None) for n in self.graph.nodes_to_compute]
        print(f"[prune] pruned graph: inputs={self.input_db_names} outputs={self.output_db_names}")
        print(f"[prune] driving state S{state_index}: energy={self.energy_key} force={self.force_key}")

    def forward(self, molecule, *args, **kwargs):
        # MD passes learned_parameters / P0 / dm_prop / cis_amp; all irrelevant here.
        pool = {}
        for db_name in self.input_db_names:
            if db_name == "Z":
                pool["Z"] = molecule.species.to(self.device)
            elif db_name == "R":
                # Detached CLONE: GraphModule.forward calls requires_grad_() in place,
                # and molecule.coordinates is the integrator's own leaf tensor.
                pool["R"] = (
                    molecule.coordinates.detach()
                    .to(device=self.device, dtype=torch.float64)
                    .clone()
                    .requires_grad_(True)
                )
            else:
                raise ValueError(f"Unsupported HIPNN graph input {db_name!r} after pruning.")

        # NEVER under torch.no_grad(): F* is a GradientNode and would go to zero.
        with torch.enable_grad():
            out_values = self.graph(*(pool[n] for n in self.input_db_names))

        out = {k: v for k, v in zip(self.output_db_names, out_values)}
        energy = out[self.energy_key].reshape(-1) + self.energy_offset
        force = out[self.force_key]

        dtype = molecule.coordinates.dtype
        device = molecule.coordinates.device
        nmol = molecule.coordinates.shape[0]

        molecule.Etot = energy.detach().to(device=device, dtype=dtype)
        molecule.force = force.detach().to(device=device, dtype=dtype).reshape(molecule.coordinates.shape)
        molecule.Hf = molecule.Etot.clone()
        molecule.Eelec = molecule.Etot.clone()
        molecule.Enuc = torch.zeros_like(molecule.Etot)
        molecule.Eiso = torch.zeros_like(molecule.Etot)
        molecule.e_gap = torch.full((nmol,), float("nan"), dtype=dtype, device=device)
        molecule.dipole = torch.zeros((nmol, 3), dtype=dtype, device=device)
        molecule.dm = None
        molecule.cis_amplitudes = None
        return molecule.force, molecule.Etot


class HIPNN_NVE(Molecular_Dynamics_Basic):
    def __init__(self, model_dir, state_index=0, energy_offset=0.0, device="auto", *a, **kw):
        super().__init__(*a, **kw)
        self.esdriver = HIPPYNN_ESDriver(model_dir, state_index, energy_offset, device)


class HIPNN_NVT(Molecular_Dynamics_Langevin):
    def __init__(self, model_dir, state_index=0, energy_offset=0.0, device="auto", *a, **kw):
        super().__init__(*a, **kw)
        self.esdriver = HIPPYNN_ESDriver(model_dir, state_index, energy_offset, device)


def make_output_config(cfg, prefix_path):
    return {
        "molid": [0],
        "prefix": str(prefix_path),
        "print every": int(cfg["log_interval"]),
        "checkpoint every": 0,
        "xyz": int(cfg["trajectory_interval"]),
        "h5": {
            "data": int(cfg["log_interval"]),
            "coordinates": int(cfg["trajectory_interval"]),
            "velocities": int(cfg["trajectory_interval"]),
            "forces": int(cfg["trajectory_interval"]),
        },
    }


def build_molecule(xyz_file, device):
    # sort=False: keep the file's atom order. sort=True reorders by decreasing Z,
    # which is equivariant nonsense for forces if the file disagrees with acn_Z.
    species_np, coordinates_np = read_xyz([str(Path(xyz_file).expanduser().resolve())], sort=False)

    got = [int(z) for z in species_np[0] if int(z) > 0]
    if got != EXPECTED_SPECIES:
        raise ValueError(
            "Atom order does not match the training convention acn_Z.\n"
            f"  expected: {EXPECTED_SPECIES}\n  got     : {got}\n"
            "Forces and NACR are equivariant -- reorder the .xyz, do not sort here."
        )
    print(f"[geom] species order matches acn_Z ({len(got)} atoms)")

    species = torch.tensor(species_np, dtype=torch.int64, device=device)
    coordinates = torch.tensor(coordinates_np, dtype=torch.float64, device=device)

    # Dummy SEQM settings: HIPPYNN supplies E and F. These only satisfy
    # Molecule/MD construction and never enter the physics.
    seqm_parameters = {"method": "AM1", "scf_eps": 123.456, "scf_converger": [-1]}
    const = Constants().to(device)
    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    return molecule, seqm_parameters


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--infile", "-i", default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    if pre_args.infile:
        parser.set_defaults(**parse_mdip(pre_args.infile))
    args = parser.parse_args()
    cfg = vars(args)

    required = ["model_dir", "xyz_file", "state_index", "energy_offset", "ensemble",
                "temperature", "timestep", "nsteps", "random_seed", "output_prefix"]
    missing = [k for k in required if cfg.get(k) is None]
    if cfg.get("ensemble") == "nvt" and cfg.get("friction") is None:
        missing.append("friction")
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    state_index = int(cfg["state_index"])
    if not 0 <= state_index <= 5:
        raise ValueError(f"state_index must be 0..5 for this checkpoint, got {state_index}")

    device = normalize_device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg["device"] == "auto" else cfg["device"]
    )

    output_dir = Path(cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = output_dir / cfg["output_prefix"]

    molecule, seqm_parameters = build_molecule(cfg["xyz_file"], device)
    # Label the surface this run integrates on (0=S0,1=S1,...5=S5) so the output h5's
    # data/excitation/active_state records it correctly (default is 0). The MLIP force
    # is already F{state_index}; this is metadata only, not consumed by the ES driver.
    molecule.active_state = state_index

    print(f"Model dir: {Path(cfg['model_dir']).expanduser()}")
    print(f"XYZ file:  {Path(cfg['xyz_file']).expanduser()}")
    print(f"Device:    {device}")
    print(f"Ensemble:  {cfg['ensemble']}")
    print(f"State:     S{state_index}")
    print(f"Offset:    {cfg['energy_offset']:.12f} eV")
    print(f"Steps:     {cfg['nsteps']} x {cfg['timestep']} fs")

    output = make_output_config(cfg, prefix_path)
    common = dict(model_dir=cfg["model_dir"], state_index=state_index,
                  energy_offset=float(cfg["energy_offset"]), device=str(device),
                  seqm_parameters=seqm_parameters, timestep=float(cfg["timestep"]),
                  Temp=float(cfg["temperature"]), step_offset=0, output=output)

    if cfg["ensemble"] == "nve":
        md = HIPNN_NVE(**common).to(device)
    else:
        md = HIPNN_NVT(damp=1.0 / float(cfg["friction"]), **common).to(device)

    md.run(molecule, steps=int(cfg["nsteps"]), reuse_P=False,
           remove_com=None, seed=int(cfg["random_seed"]))

    final_xyz = output_dir / f"{cfg['output_prefix']}_final.xyz"
    write_final_xyz(final_xyz, molecule)
    print(f"Final geometry written to {final_xyz}")


if __name__ == "__main__":
    main()

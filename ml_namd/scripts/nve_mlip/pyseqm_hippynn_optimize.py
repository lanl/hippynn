#!/usr/bin/env python3
"""geomeTRIC geometry optimization on a chosen adiabatic state, driven by the SAME
HIPPYNN E+F wiring as pyseqm_hippynn_md.py.

WHY THIS EXISTS
---------------
The excited-state NVE workflow was missing an optimization step: excited runs
started from the S0-region frame (geom/{t}_s0_start.xyz) instead of the relevant
excited-state minimum. This relaxes the start geometry onto the S{active_state}
PES with the MLIP, then hands the optimized .xyz to pyseqm_hippynn_md.py for the
NVE on that same surface. S0 keeps its original recipe (frame -> NVE directly).

DESIGN
------
- Optimizer = geomeTRIC (PYSEQM's recommended production optimizer;
  seqm.optimization.geomeTRIC_optimization uses it). That wrapper hardcodes the
  stock SEQM Electronic_Structure engine inside a closure, so we cannot subclass
  it to inject the MLIP. Instead we replicate its geomeTRIC `Engine`
  (`HIPNN_GeomTRIC_Engine`) with ONE change: self.esdriver = HIPPYNN_ESDriver(...)
  -- the pruned GraphModule [Z,R]->[sE_k,F_k] (float64) the MD propagator uses.
  Energy/gradient contract + unit conversions (ev2au, bohr2ang) are copied
  verbatim from the stock wrapper so results match PYSEQM conventions.
- `active_state` (0..5) selects the surface via HIPPYNN_ESDriver's sE{k}/F{k},
  the same convention as the MD's state_index.
- Model load/prune, Molecule build (+ acn_Z species-order assert), .input parsing,
  device normalization and the final-xyz writer are IMPORTED from
  pyseqm_hippynn_md.py so optimizer and propagator stay in lockstep.

PREREQ: geomeTRIC must be importable in the run env (the project environment):
  pip install geometric        # not yet installed as of staging

INPUT: same .input files as the MD runner (reads `active_state` or `state_index`),
plus optional --opt-maxsteps. Writes {output_dir}/{output_prefix}_opt.xyz.

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
from pathlib import Path

os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")
import torch  # noqa: E402

# Importing pyseqm_hippynn_md also runs its sys.path bootstrap (makes
# direct_nacr_head_v2 / unified_training importable for the unpickle) and imports
# seqm -- so this file must sit next to pyseqm_hippynn_md.py (the nve_md dir).
from pyseqm_hippynn_md import (  # noqa: E402
    HIPPYNN_ESDriver,
    build_molecule,
    parse_mdip,
    normalize_device,
    write_final_xyz,
)


def _write_trajectory(path, step, molecule):
    """Append one frame to a multi-frame .xyz (mirrors the stock PYSEQM helper)."""
    species = molecule.species[0]
    coordinates = molecule.coordinates[0]
    with open(path, "a") as out:
        out.write(f"{species.shape[0]}\nStep {step}, E = {molecule.Etot[0]:.8f} eV\n")
        for atom, xyz in zip(species, coordinates):
            out.write(f"{molecule.const.label[atom.item()]} {xyz[0]:15.5f} {xyz[1]:15.5f} {xyz[2]:15.5f}\n")


def build_hipnn_engine(molecule, model_dir, active_state, energy_offset, device, traj_file):
    """geomeTRIC Engine that gets E/grad from the HIPPYNN MLIP on S{active_state}.

    Replicates seqm.optimization.geometry.pyseqm_engine, swapping the stock SEQM
    esdriver for HIPPYNN_ESDriver. Built inside a function because geometric is an
    optional dependency (imported lazily so this module imports without it)."""
    try:
        from geometric.engine import Engine
        from geometric.molecule import Elements
        from geometric.molecule import Molecule as GeomMolecule
        from geometric.nifty import bohr2ang, ev2au
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "geomeTRIC is required for this optimizer but is not installed in the "
            "run env. Install it first:  pip install geometric"
        ) from e

    class HIPNN_GeomTRIC_Engine(Engine):
        def __init__(self):
            geomol = GeomMolecule()
            geomol.elem = [Elements[z] for z in molecule.species.cpu().numpy()[0]]
            geomol.xyzs = [molecule.coordinates.clone().detach().cpu().numpy()[0]]
            super().__init__(geomol)
            self.molecule = molecule
            self.esdriver = HIPPYNN_ESDriver(model_dir, active_state, energy_offset, device)
            self.step = 0
            self.traj_file = traj_file

        def calc_new(self, coords, dirname=None):
            # geomeTRIC hands coords in Bohr (flat); HIPPYNN wants Angstrom (1,N,3).
            xyz_ang = (
                torch.as_tensor(
                    coords.reshape(-1, 3),
                    device=self.molecule.coordinates.device,
                    dtype=self.molecule.coordinates.dtype,
                )
                .mul(bohr2ang)
                .unsqueeze(0)
            )
            with torch.no_grad():
                self.molecule.coordinates = torch.nn.Parameter(xyz_ang)

            # HIPPYNN_ESDriver ignores P0/dm_prop/learned_parameters; it sets
            # molecule.Etot (eV, incl. offset) and molecule.force (eV/Ang).
            self.esdriver(self.molecule, P0=self.molecule.dm, dm_prop="SCF", learned_parameters={})

            E_h = self.molecule.Etot.item() * ev2au                       # Hartree
            grad_eV_A = -self.molecule.force.detach().cpu().numpy()[0]     # (N,3) eV/Ang
            grad_ha = (grad_eV_A * ev2au * bohr2ang).reshape(-1)          # Hartree/Bohr
            self.step += 1
            if self.traj_file:
                _write_trajectory(self.traj_file, self.step, self.molecule)
            return {"energy": E_h, "gradient": grad_ha}

    return HIPNN_GeomTRIC_Engine()


def build_parser():
    p = argparse.ArgumentParser(description="geomeTRIC geometry opt on S{active_state} via a HIPPYNN model.")
    p.add_argument("--infile", "-i", default=None, help="Path to .input/.mdip file.")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--xyz-file", default=None, help="Start geometry (acn_Z atom order).")
    p.add_argument("--active-state", type=int, default=None, help="0=S0,...,5=S5. Alias: state_index.")
    p.add_argument("--energy-offset", type=float, default=None, help="Constant eV shift (cosmetic for opt).")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--opt-maxsteps", type=int, default=500, help="geomeTRIC max optimization steps.")
    p.add_argument("--save-opt-traj", action="store_true", help="Also write {prefix}_opt_traj.xyz.")
    return p


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--infile", "-i", default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    if pre_args.infile:
        file_cfg = parse_mdip(pre_args.infile)
        if "active_state" not in file_cfg and "state_index" in file_cfg:
            file_cfg["active_state"] = file_cfg["state_index"]
        known = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in file_cfg.items() if k in known})
    args = parser.parse_args()
    cfg = vars(args)

    required = ["model_dir", "xyz_file", "active_state", "output_prefix"]
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    active_state = int(cfg["active_state"])
    if not 0 <= active_state <= 5:
        raise ValueError(f"active_state must be 0..5, got {active_state}")
    energy_offset = float(cfg["energy_offset"]) if cfg.get("energy_offset") is not None else 0.0

    device = normalize_device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg["device"] == "auto" else cfg["device"]
    )

    output_dir = Path(cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_file = str(output_dir / f"{cfg['output_prefix']}_opt_traj.xyz") if cfg["save_opt_traj"] else None
    if traj_file and os.path.exists(traj_file):
        os.remove(traj_file)  # _write_trajectory appends; start clean

    molecule, seqm_parameters = build_molecule(cfg["xyz_file"], device)
    molecule.active_state = active_state
    molecule.dm = None  # engine passes P0=molecule.dm; HIPPYNN ignores it

    print(f"Model dir:  {Path(cfg['model_dir']).expanduser()}")
    print(f"Start XYZ:  {Path(cfg['xyz_file']).expanduser()}")
    print(f"Device:     {device}")
    print(f"Optimizing: S{active_state}  (energy=sE{active_state}, force=F{active_state})  geomeTRIC maxsteps={cfg['opt_maxsteps']}")

    import tempfile
    from geometric.optimize import run_optimizer  # raises clearly if geomeTRIC missing

    engine = build_hipnn_engine(molecule, cfg["model_dir"], active_state,
                                energy_offset, str(device), traj_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpf = os.path.join(tmpdir, "geometric.in")
        result = run_optimizer(customengine=engine, input=tmpf, qccnv=True,
                               maxsteps=int(cfg["opt_maxsteps"]))

    # Final geometry = last frame of the optimization (Angstrom), in acn_Z order.
    final_ang = result.xyzs[-1]
    with torch.no_grad():
        molecule.coordinates = torch.nn.Parameter(
            torch.as_tensor(final_ang, device=device, dtype=molecule.coordinates.dtype).unsqueeze(0)
        )
    opt_xyz = output_dir / f"{cfg['output_prefix']}_opt.xyz"
    write_final_xyz(opt_xyz, molecule)
    print(f"Optimized S{active_state} geometry written to {opt_xyz}")
    print(f"Next: point the S{active_state} NVE .input's xyz_file at this file.")


if __name__ == "__main__":
    main()

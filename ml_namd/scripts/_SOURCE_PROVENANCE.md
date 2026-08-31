# Script source provenance (Phase 1 staging)

These files are **verbatim copies** of the canonical working scripts as of 2026-08-30,
pulled from the clusters into this private repo for refactoring in Phase 2. Nothing here is
refactored yet; behavior is exactly as it ran on the clusters. Each subfolder keeps a
`_MD5SUMS.txt` recording the source checksums for verification.

Item numbers refer to the project brief (`ml_namd/PROJECT_BRIEF.md`).

| Folder | Brief item | Source host | Source path |
|--------|-----------|-------------|-------------|
| `training/` | [4] Model A (E+F), [5] Model B v3 (dENACR) | Chicoma | `/users/akhanna2/data/ml_namd_project/scripts/train/` |
| `fssh_mlip/` | [7] FSSH@MLIP (composed A+B, vectorized phase-sign) | Chicoma | `/lustre/scratch5/akhanna2/keto_mlip_10ps/driver/` |
| `fssh_pyseqm/` | [8] FSSH@PYSEQM reference | Darwin | `/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/` |
| `nve_mlip/` | [6] NVE@MLIP (Model A driven) | Chicoma | `/users/akhanna2/data/ml_namd_project/experiments/nve_md/` |
| `data_generation/` | [3b] PySEQM dataset extraction/generation | Chicoma | `/users/akhanna2/data/ml_namd_project/data_gen/` |
| `slurm/` | [9] SLURM (training + FSSH@MLIP) | Chicoma | `/lustre/scratch5/akhanna2/keto_mlip_10ps/slurm*/` |
| `analysis/` | harvest / population utilities | Chicoma | `/lustre/scratch5/akhanna2/keto_mlip_10ps/` |

## Notes on specific files
- **`training/unified_training.py`** — one config-driven script serving BOTH Model A
  (`--train_energies --train_forces`, NACR off) and Model B (`--train_scaled_nacrs`,
  `targeted_scaled_nacr_pairs = 12 13 23`). This is the real training script (verified live copy).
- **`training/direct_nacr_head_v3.py`** — the current GDV+gap-aware dENACR head (Model B v3).
  `direct_nacr_head.py` / `_v2.py` are earlier variants, kept for provenance/comparison (legacy).
- **`training/joint_state_nacr_loss.py`** — joint per-state phase loss node; required at load time to
  unpickle any joint-loss Model B.
- **`fssh_mlip/hippynn_fssh.py`** — the FSSH engine with the vectorized `_phase_match`.
  `hippynn_fssh_composed.py` + `namd_composed_batch.py` are the composed (Model A inference +
  Model B dENACR) driver + batch runner. `namd_batch.py` is the older single-model (joint) driver, legacy.
- **`fssh_pyseqm/run_fssh_ref.py`** — the FSSH@PYSEQM reference driver (master PySEQM).
  `run_fssh_ref_dc.py` is the decoherence-on-hop variant (reference config = `hamiltonian_fd` TDC +
  `decohere_on_hop=True`). `harvest_enol_ref.py` is the enol-specific harvest.
- **`nve_mlip/pyseqm_hippynn_md.py`** — NVE@MLIP runner (PySEQM `Molecular_Dynamics_Basic` driven by a
  HIPPYNN E+F model). `pyseqm_hippynn_optimize.py` optimizes excited-state start geometries onto S_k
  before NVE. Production timestep is 0.1 fs.

## Still to add (pending or later phases)
- Environment reproducibility records (`ml_namd/env/`) — Chicoma + Darwin venv_master + HIPPYNN envs.
- Data manifests (`ml_namd/data_manifests/`) — S0-dyn (Model A), enol BALF (Model B), Vishikh 1000-IC enol.
- Phase 2: full refactor + docstrings + developer header on every file here.

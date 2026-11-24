# Script provenance

The scripts under `ml_namd/scripts/` are the project's canonical MLIP-for-NAMD pipeline, refactored for
clarity while preserving numerical behavior. They map to the pipeline stages as follows:

| Folder | Stage |
|---|---|
| `training/` | Model A (energies + forces) and Model B (scaled NACR / dENACR, v3 head) training |
| `data_generation/` | reference-structure dataset generation + extraction + the genA pool builder |
| `fssh_mlip/` | FSSH@MLIP driver (composed Model A + Model B, vectorized phase-sign) + predictor bundle |
| `fssh_pyseqm/` | FSSH@PYSEQM reference driver + harvest utilities |
| `nve_mlip/` | NVE@MLIP runner (Model A driven) + excited-state optimizer + validation |
| `analysis/` | ensemble harvest / population utilities |

## Notes on specific files
- `training/unified_training.py` — one config-driven script serving BOTH Model A
  (`--train_energies --train_forces`) and Model B (`--train_scaled_nacrs`,
  `targeted_scaled_nacr_pairs = 12 13 23`).
- `training/direct_nacr_head_v3.py` — the GDV+gap-aware dENACR head (Model B v3).
  `direct_nacr_head.py` / `_v2.py` are earlier variants kept for provenance.
- `training/joint_state_nacr_loss.py` — the joint per-state phase loss node; required at load time to
  unpickle any joint-loss Model B.
- `fssh_mlip/hippynn_fssh.py` — the FSSH engine with the GPU-vectorized `_phase_match`.
  `hippynn_fssh_composed.py` + `namd_composed_batch.py` are the composed (Model A + Model B) driver + runner.
- `fssh_pyseqm/run_fssh_ref.py` — the FSSH@PYSEQM reference driver (`run_fssh_ref_dc.py` is the
  decoherence-on-hop variant; the reference uses `hamiltonian_fd` TDC + `decohere_on_hop=True`).
- `nve_mlip/pyseqm_hippynn_md.py` — the NVE@MLIP runner; `pyseqm_hippynn_optimize.py` relaxes an
  excited-state start geometry onto a chosen surface before NVE.

Environment pins and dataset schema are documented in `../env/` and
`../data_manifests/DATA_MANIFEST.md`.

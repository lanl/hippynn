# Shared ensemble-harvest utilities

Canonical home for the harvesters used by **both** FSSH@MLIP and FSSH@PYSEQM, so a run from either
driver is post-processed identically (the requirement for a fair apple-to-apple comparison).

| Script | Purpose |
|---|---|
| `harvest_fssh_full.py` | The canonical 6-panel apple-to-apple harvest. Reads a directory of per-trajectory h5, computes survivor-conditional (live-ensemble) classical populations (active-surface fraction) and quantum populations (excited \|c\|², renormalized) plus gaps and geometry diagnostics, and writes a compact `.npz`. Schema-agnostic in `n_states`. Usage: `python harvest_fssh_full.py <h5_glob> <out.npz> <ps_per_frame>`. |
| `harvest_clean.py` | Strict clean-survivor ensemble: only trajectories that survive the full length AND contain no NaN in the amplitude norm; reports per-source counts and excluded trajectories. |

These were previously duplicated in `../fssh_pyseqm/`; that copy has been removed and both drivers now
reference this canonical location (de-duplication). The reference-**specific** harvesters remain under
`../fssh_pyseqm/` (`survivors_ensemble.py`, `fixedN_harvest.py`, `harvest_enol_ref.py`).

Use the **same** harvester and conventions for the reference and the MLIP so the resulting `.npz` files
are directly comparable; see `../../docs/fssh_pyseqm_guide.md` (apple-to-apple matching rules).

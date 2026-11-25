# Shared ensemble-harvest utilities

**Harvesting is common to both drivers.** These scripts are the canonical home for the harvesters used
by **both FSSH@MLIP and FSSH@PYSEQM**, so a run from either driver is post-processed *identically* --
that identical post-processing is what makes the MLIP-vs-reference comparison fair (like-for-like). Point
the same script at the reference h5 and at the MLIP h5 and compare the two `.npz` outputs directly.

## The three harvesters are a stringency ladder, not duplicates

Each one reads the same per-trajectory h5 and keeps a progressively stricter subset of trajectories.
Pick the one whose trajectory-selection policy you want; they all emit the same `.npz` population schema.

| Script | Trajectories kept | Use it when |
|---|---|---|
| `harvest_fssh_full.py` | **(a)** survivors only (reached the final frame). The canonical 6-panel harvest: survivor-conditional (live-ensemble) classical populations (active-surface fraction) and quantum populations (excited \|c\|², renormalized), plus gaps and geometry diagnostics. Schema-agnostic in `n_states`. | The main like-for-like population comparison. This is the default. |
| `harvest_clean.py` | **(a)** survive the full length **and (b)** no NaN in the amplitude norm at any alive frame. Constant-N clean ensemble; reports per-source counts and every excluded NaN trajectory. | You want to drop numerically broken (NaN) trajectories for a clean constant-N ensemble. |
| `harvest_topo.py` | `harvest_clean.py`'s **(a)+(b)**, **plus (c)** peak pairwise distance <= 15 Angstrom and **(d)** no bond stretched past a covalent-radius factor at any alive frame (fragmentation guard). Reports old-clean (a,b) vs frag-clean (a-d) side by side. | You want to exclude exploded / fragmented trajectories (most relevant to MLIP runs that can run away). Note: the bond/covalent-radius table is parameterised for acetylacetone (H, C, O); adjust it for other molecules. |

Usage (positional args differ per script -- see each file's header docstring), e.g.:

```bash
python harvest_fssh_full.py <h5_glob> <out.npz> <ps_per_frame>
python harvest_clean.py      <ps_per_frame> <label> <h5_glob> <out.npz>
python harvest_topo.py       <h5_glob> <label> <out.npz> [maxdist=15.0] [fragfac=1.8] [limit=0]
```

## Reference-specific harvesters live elsewhere

The harvesters above were previously duplicated under `../fssh_pyseqm/`; that copy was removed and both
drivers now reference this canonical location (de-duplication). What remains under `../fssh_pyseqm/` is
only the harvesting that is **specific to the reference workflow**: `survivors_ensemble.py`,
`fixedN_harvest.py`, `harvest_enol_ref.py`.

Use the **same** harvester and conventions for the reference and the MLIP so the resulting `.npz` files
are directly comparable; see `../../docs/fssh_pyseqm_guide.md` (like-for-like matching rules).

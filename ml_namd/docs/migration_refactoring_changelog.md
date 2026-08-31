# Migration and refactoring changelog

Record of how the MLIP-for-NAMD code was packaged into this repository and refactored. The guiding
constraint throughout: **preserve numerical and scientific behavior exactly** while improving clarity,
documentation, and reproducibility. Nothing in the upstream HIPPYNN library tree was modified.

Commit hashes are on the `ml_namd` branch of `Ajaykhanna/hippynn_NAMD`.

---

## Repository creation

- Created a **private** standalone copy of `lanl/hippynn` (a fork of a public repo cannot itself be
  private), default branch **`ml_namd`** (renamed from upstream `development`), with `upstream` pointing
  at `lanl/hippynn` for future syncing.
- All project code lives under a top-level **`ml_namd/`** folder so the upstream tree stays untouched and
  a future upstream merge is clean.
- Salvaged an earlier fork's HIPPYNN-internals documentation (a 72 KB energy-prediction reference, a
  D3.js computation-flow page, and PyTorch-extraction notes) into `ml_namd/docs/reference/` before the
  old public fork was removed. *(commit `ae476e3`)*

## Phase 1 — staging (verbatim)

- **45 canonical scripts** (30 Python + 15 SLURM) copied verbatim from the clusters into
  `ml_namd/scripts/{training,fssh_mlip,fssh_pyseqm,nve_mlip,data_generation,slurm,analysis}`, with a
  source-provenance manifest and per-folder md5 sums. *(commit `2da9e95`)*
- **Environment reproducibility records** for the three canonical environments (Chicoma `venv_master`,
  Darwin `venv_master`, Darwin `hiphopnn_env`): full captures, per-env `pip freeze`, a cross-env table,
  and a `verify_env.py` checker. *(commit `01e4308`)*
- **Data manifests** for the three enol datasets (Model A S0-dynamics, Model B BALF pool, the FSSH
  reference), schema and provenance only — the data stays on the ground-zero drive / clusters.
  *(commit `cb44e87`)*

## Phase 2 — refactor (behavior-preserving)

Every project-owned script was refactored **additively**: full module + function/class docstrings
(purpose, parameters, units, shapes, exceptions), a developer header, type hints, intent comments, and —
where safe — descriptive local names and structured constants. **Public names, CLI/config keys, `acn_`
array names, and all numerics were preserved.** This was mandatory because the HIPPYNN node/loss class
names and module filenames are pickled into `experiment_structure.pt` at training time and resolved when
a model is loaded for FSSH; renaming any of them would break every existing trained checkpoint.

Per group:

| Group | Commit | Notes |
|---|---|---|
| `training/` (5 py) | `5a8e75f`, `d63091c`, `e922771`, `be87af5` | loss + heads + `unified_training.py`; the latter was already well-documented (light header/docstring pass) |
| `fssh_mlip/` (4 py) | `689a0d5` | engine (vectorized `_phase_match`) + composed driver + runners |
| `fssh_pyseqm/` (7 py + 3 slurm) | `29e095a` | reference driver + harvest utilities |
| `nve_mlip/` (8 py + 3 slurm) | `f14bd8e` | runner + optimizer + validation |
| `data_generation/` (4 py + 3 slurm) | `52110ab` | already heavily engineered; header pass |
| `analysis/` (2 py) + `slurm/` (6 slurm) | `c8064e4` | analysis harvests reused the refactored `fssh_pyseqm/` copies |

SLURM scripts received a `PURPOSE / INPUTS / OUTPUTS / CONFIG / DEPENDS / Developer` header before
`set -o pipefail`, with every `#SBATCH` directive, environment activation, and command left untouched.

### Verification method (how behavior preservation was proven)

- **Pickled HIPPYNN node/loss files** (the heads and the joint loss): seed-matched instantiation of the
  git-original and the refactored class, then `state_dict` and `forward()` output compared on random
  inputs — required **max |Δ| = 0** (run on the cluster environment, whose HIPPYNN matches the pin).
- **All other Python files**: an **AST-equivalence** check — parse the original and refactored, strip
  docstrings, type annotations, and the metadata dunders, and compare the normalized `ast.dump`. Every
  file came back logically identical. This proves the executable statements are byte-for-byte the same.
- **SLURM scripts**: `#SBATCH` directive count and command presence grep-verified after each edit.

## Runtime dependencies pulled in

To make the FSSH groups runnable-complete, two runtime dependencies were **vendored** (verbatim, with
READMEs):

- the FSSH@MLIP **predictor bundle** (`ml_namd/scripts/fssh_mlip/enol_ef_nacr_developer_handoff/src/`,
  13 modules incl. `hiphop_predictor_v3`, `modelb_composed_predictor`) *(commit `e17da4d`)*;
- the FSSH@PYSEQM **per-core workers** (`ml_namd/scripts/fssh_pyseqm/workers/`) *(commit `8f20aa7`)*.

The predictor bundle also carries older copies of the training modules (kept so runtime pickles resolve
exactly as on the cluster); the canonical refactored versions are in `ml_namd/scripts/training/`.

One dependency is **documented but not vendored** (too large): the **PySEQM-ml_namd fork** used only for
data-generation labeling, referenced on `PYTHONPATH` by the data-gen slurms.

## Documentation added

Under `ml_namd/docs/`: the Model B math background and usage guide; Model A, NVE@MLIP, FSSH@MLIP, and
FSSH@PYSEQM guides; installation/environment guide; SLURM job guide; this changelog. Plus the top-level
`README.md` (project front door) and per-group READMEs / provenance files. The data manifest lives in
`ml_namd/data_manifests/DATA_MANIFEST.md`.

## Corrections made during documentation

- **`n_states` semantics.** `n_states` is the number of *excited* states, so S0–S5 is `n_states = 5`
  (an earlier Model A guide draft wrongly used 6). Corrected in the guide, and `n_states = 3` added to
  the Model B usage config, after cross-checking the live param-swap configs. *(commit `8ba8c55`)*
- **Missing training launchers.** The repo had `unified_training.py` but no SLURM script showing how to
  train a model; added self-contained enol Model A and Model B training slurms + configs. *(commit `8ba8c55`)*
- A stale `"(keto)"` comment on the enol data-generation job was flagged (not altered) and clarified in
  the job header.

## What was deliberately NOT changed

- The upstream HIPPYNN library tree (its `README.rst` is preserved and still used by packaging; the
  root `README.md` only changes the rendered landing page).
- Any script's control flow, numerics, CLI/config keys, or `acn_` array conventions.
- The `print()`-based training progress output was kept (intentional user-facing output; converting
  ~2800 lines to logging would be risk for no benefit).

## Cleanups completed

- **Predictor-bundle refactor.** The 9 predictor-specific modules in the vendored bundle received the
  developer header (AST-verified behavior-preserving); the 4 training-module copies were left verbatim on
  purpose (older functional versions the runtime pickle needs).
- **Harvester de-duplication.** `harvest_fssh_full.py` / `harvest_clean.py` were consolidated under
  `scripts/analysis/` (canonical, with a README); the `scripts/fssh_pyseqm/` duplicates were removed and
  both drivers now reference the analysis copies.
- **FSSH@PYSEQM worker `.sh` refactor.** The three per-core workers
  (`scripts/fssh_pyseqm/workers/{ref_worker_tdc,ns3_worker,keto_worker_dynamic}.sh`) received the project
  `PURPOSE/INPUTS/OUTPUTS/CONFIG/DEPENDS` header block plus the dev-header line. The pass is additive
  (header and comments only); every executable line is byte-identical to the vendored original, verified
  by diffing the non-comment/non-blank lines against `HEAD` (all three reported IDENTICAL). The workers'
  README was updated to drop the "candidate follow-up" note.

## Deferred cleanups (candidate follow-ups)

Each now has a runnable close-out staged in the repo; all three are gated on cluster or external-drive
access, not on further authoring.

- Pin the Darwin PySEQM `master_src` commit (its checkout had no `.git`) — content-match recipe staged in
  `../env/README.md` under "Pending capture".
- Capture the GPU driver version on a compute node (env records were login-node only) — one-shot
  `nvidia-smi` command staged in `../env/README.md` under "Pending capture".
- Record the raw 1000-IC `R0/V0/Z` md5s — **resolved 2026-08-31 (`D:` mounted).** A full-drive search
  found the launch ICs are cluster-only (not on `D:` or `C:`); `data_manifests/DATA_MANIFEST.md` §4 now
  records that with the cluster-side `md5sum` command, and §3 md5-pins the reference survivor arrays that
  do live on `D:` (Python `hashlib.md5`; the laptop's FIPS policy blocks .NET MD5). Only the cluster-side
  launch-IC hashes remain, to be taken on the cluster.

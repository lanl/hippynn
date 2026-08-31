# ml_namd — MLIP-driven nonadiabatic molecular dynamics on HIPPYNN

Machine-learned interatomic potentials (MLIPs) that drive nonadiabatic excited-state molecular
dynamics (NA-ESMD), in particular fewest-switches surface hopping powered by MLIPs (FSSH@MLIP),
benchmarked against reference FSSH@PYSEQM. Applied to acetylacetone (enol and keto tautomers).

This repository is a **private working copy of [lanl/hippynn](https://github.com/lanl/hippynn)**. The
surrounding tree is HIPPYNN itself; everything project-specific lives under **`ml_namd/`** so it stays
cleanly separable from the upstream library (the long-term goal is to merge selected features upstream).
For the HIPPYNN library see its own `README.rst`.

- **Collaborators:** Ajay Khanna and Tomas Karella (LANL, Sergei Tretiak group).
- **Objectives:** (a) inference optimization for all MLIP predictions, especially the FSSH driver;
  (b) merge the MLIP-for-NAMD features into HIPPYNN and maintain them upstream.
- **Provenance:** upstream `lanl/hippynn` branch `development` → this repo `Ajaykhanna/hippynn_NAMD`
  (private), working branch `ml_namd`; the `upstream` git remote points at `lanl/hippynn` for syncing.

## The pipeline at a glance

```
                 ┌─────────────┐   E, F           ┌─────────────┐   dENACR
   Z, R ───────► │  Model A    │ ───────────────► │  Model B    │ ──────────►  FSSH@MLIP driver
                 │  (E + F,    │   GAP = E_j−E_i   │  (scaled    │              d = dENACR/GAP
                 │   S0–S5)    │   GDV = F_i−F_j   │   NACR, v3) │              σ = d·v  → hops
                 └─────────────┘                   └─────────────┘
                        │                                                 ▲
                        │ NVE@MLIP (single state)                         │ benchmark
                        ▼                                                 │
                 energy-conserving MD                              FSSH@PYSEQM (AM1/CIS reference)
```

- **Model A** — a multistate energy + force network (S0–S5). Also supplies the energy gaps and
  gradient-difference vectors Model B needs at inference.
- **Model B** — the scaled nonadiabatic coupling (dENACR) network, v3 GDV+gap-aware head. See
  `ml_namd/docs/model_b_math_background.md` for the theory and `model_b_usage_guide.md` for the how-to.
- **NVE@MLIP** — energy-conserving molecular dynamics driven by Model A (validation of E+F quality).
- **FSSH@MLIP** — surface hopping with Model A + Model B as the electronic-structure engine.
- **FSSH@PYSEQM** — the semiempirical AM1/CIS surface-hopping reference the MLIP is measured against.

## Layout

```
<repo root>/                    upstream HIPPYNN library (README.rst, hippynn/, docs/, tests/, ...)
ml_namd/                        all project-owned code + documentation
  README.md                     project overview (folder-level)
  PROJECT_BRIEF.md              original task brief
  scripts/
    training/                   Model A + Model B training (unified_training.py + heads + loss)
    data_generation/            PySEQM dataset generation + extraction + pool builders
    fssh_mlip/                  FSSH@MLIP driver (composed A+B, vectorized phase-match) + predictor bundle
    fssh_pyseqm/                FSSH@PYSEQM reference driver + harvest + per-core workers
    nve_mlip/                   NVE@MLIP runner + excited-state optimizer + validation
    slurm/                      keto FSSH@MLIP orchestration (prod/smoke)
    analysis/                   ensemble harvest utilities
  env/                          environment reproducibility records + verify_env.py
  data_manifests/               dataset manifests + schema (data itself stays on D:/clusters)
  docs/                         guides, math background, reports (this documentation set)
    reference/                  salvaged HIPPYNN-internals reference notes
```

## Quickstart

1. **Environment.** Recreate the recorded environment (see `ml_namd/env/README.md` for the exact pins:
   Python 3.13, torch 2.7.1+cu118, editable HIPPYNN at the pinned commit, PySEQM master), then verify:

   ```bash
   python ml_namd/env/verify_env.py
   ```

2. **Data.** Datasets stay on the ground-zero drive / clusters; see `ml_namd/data_manifests/DATA_MANIFEST.md`
   for locations, array names, shapes, and provenance. Generate/extract with `ml_namd/scripts/data_generation/`.

3. **Train Model A** (E + F, S0–S5) then **Model B** (dENACR) with `ml_namd/scripts/training/unified_training.py`
   (config-file driven; see `model_b_usage_guide.md` for a complete `.inp`).

4. **Run dynamics.** NVE@MLIP via `ml_namd/scripts/nve_mlip/pyseqm_hippynn_md.py`; FSSH@MLIP via
   `ml_namd/scripts/fssh_mlip/namd_composed_batch.py`; the FSSH@PYSEQM reference via
   `ml_namd/scripts/fssh_pyseqm/run_fssh_ref.py`. SLURM orchestration is in `ml_namd/scripts/slurm/`
   and the group READMEs.

Machine-specific paths in example commands are the confirmed cluster locations; replace them with your own.

## Documentation

| Document | Status | Purpose |
|---|---|---|
| `ml_namd/docs/model_b_math_background.md` | ✅ | Theory: why dENACR, the branching plane, the v3 head, the sign-consistent loss |
| `ml_namd/docs/model_b_usage_guide.md` | ✅ | How-to: train + deploy Model B, data contract, gotchas |
| `ml_namd/env/README.md` | ✅ | Environment reproducibility records + recreate/verify |
| `ml_namd/data_manifests/DATA_MANIFEST.md` | ✅ | Dataset schema + provenance |
| `ml_namd/scripts/*/README.md`, `_SOURCE_PROVENANCE.md` | ✅ (partial) | Per-group provenance + vendored-dependency notes |
| `ml_namd/docs/model_a_guide.md` | ✅ | E+F multistate model training + inference |
| `ml_namd/docs/nve_at_mlip_guide.md` | ✅ | energy-conserving MD workflow + 4-check validation |
| FSSH@MLIP guide | planned | the composed A→B surface-hopping driver |
| FSSH@PYSEQM guide | planned | the reference driver + apple-to-apple harvest |
| SLURM job guide | planned | submitting + monitoring the cluster jobs |
| `ml_namd/docs/installation_and_environment.md` | ✅ | end-to-end environment setup (step-by-step recreate + verify + troubleshooting) |
| Validation report, inference benchmark report | planned | accuracy + throughput evidence |
| Migration / refactoring changelog, known limitations | planned | what changed, what is unresolved |

## Reproducibility and provenance notes

- Every project-owned script was refactored for clarity while **preserving numerical behavior**
  (verified per file); the surrounding HIPPYNN library tree is unmodified.
- Some runtime dependencies are **vendored** for completeness: the FSSH@MLIP predictor bundle
  (`ml_namd/scripts/fssh_mlip/enol_ef_nacr_developer_handoff/`) and the FSSH@PYSEQM per-core workers
  (`ml_namd/scripts/fssh_pyseqm/workers/`), each with a README. One dependency is documented but not
  vendored (too large): the PySEQM-ml_namd data-generation fork.
- The salvaged HIPPYNN-internals notes under `ml_namd/docs/reference/` are third-party-style reference
  material, preserved from an earlier fork (see its `_PROVENANCE.md`).

## License

The HIPPYNN library retains its upstream license (`LICENSE.txt`, `COPYRIGHT.txt`). Project-owned code
under `ml_namd/` is developed for LANL research; consult the collaborators before redistribution.

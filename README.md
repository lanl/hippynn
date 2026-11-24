# ml_namd — MLIP-driven nonadiabatic molecular dynamics on HIPPYNN

Machine-learned interatomic potentials (MLIPs) that drive nonadiabatic excited-state molecular
dynamics (NA-ESMD), in particular fewest-switches surface hopping powered by MLIPs (FSSH@MLIP),
benchmarked against a semiempirical AM1/CIS FSSH@PYSEQM reference. Applied to the enol and keto
tautomers of acetylacetone, and designed to generalize to other molecules and datasets.

This repository is a working copy of [lanl/hippynn](https://github.com/lanl/hippynn): the surrounding
tree is HIPPYNN itself (see its own `README.rst`), and everything project-specific lives under
**`ml_namd/`** so it stays cleanly separable from the upstream library. Start in
[`ml_namd/README.md`](ml_namd/README.md) and the runnable examples under `ml_namd/examples/`.

- **Collaborators:** Ajay Khanna and Tomas Karella (LANL, Sergei Tretiak group).
- **Objectives:** (a) inference optimization for all MLIP predictions, especially the FSSH driver;
  (b) merge the MLIP-for-NAMD features into HIPPYNN and maintain them upstream.
- **Provenance:** upstream `lanl/hippynn` branch `development` → this repo; the `upstream` git remote
  points at `lanl/hippynn` for syncing.

## The pipeline at a glance

```
                 ┌─────────────┐   E, F            ┌─────────────┐   dENACR
   Z, R ───────► │  Model A    │ ────────────────► │  Model B    │ ──────────►  FSSH@MLIP driver
                 │  (E + F)    │   GAP = E_j−E_i    │  (scaled    │              d = dENACR / GAP
                 │             │   GDV = F_i−F_j    │   NACR, v3) │              σ = d·v  → hops
                 └─────────────┘                    └─────────────┘
                        │                                                  ▲
                        │ NVE@MLIP (single state)                          │ benchmark
                        ▼                                                  │
                 energy-conserving MD                               FSSH@PYSEQM (AM1/CIS reference)
```

- **Model A** — a multistate energy + force network. It also supplies the energy gaps and
  gradient-difference vectors that Model B consumes at inference.
- **genA** — runs a *frozen* Model A over the training pool to write the **A-predicted** gaps/GDV that
  the deployed Model B trains on, so training matches inference (train-A == deploy-A).
- **Model B** — the scaled nonadiabatic-coupling (dENACR) network, a v3 GDV + gap-aware head with three
  variants: `both` (hybrid, main result), `gdv_only`, and `geom_only`. See
  `ml_namd/docs/model_b_math_background.md` for the theory and `model_b_usage_guide.md` for the how-to.
- **NVE@MLIP** — energy-conserving molecular dynamics driven by Model A (validation of E + F quality).
- **FSSH@MLIP** — surface hopping with the composed Model A + Model B as the electronic-structure engine.
- **FSSH@PYSEQM** — the AM1/CIS surface-hopping reference the MLIP is measured against.

## Layout

```
<repo root>/        upstream HIPPYNN library (README.rst, hippynn/, docs/, tests/, LICENSE.txt, COPYRIGHT.txt, ...)
ml_namd/            all project-owned code + documentation
  README.md         project overview — start here
  examples/         runnable, self-contained worked examples (enol, keto) — plain python, steps 00-05
  scripts/
    training/       Model A + Model B training (unified_training.py + heads + loss + split tools)
    data_generation/ dataset generation/extraction, pool builders, and genA (gen_modelA_inputs.py)
    fssh_mlip/      composed FSSH@MLIP driver (per-IC seeded) + the predictor bundle
    fssh_pyseqm/    FSSH@PYSEQM reference driver + harvest
    nve_mlip/       NVE@MLIP runner + excited-state optimizer + validation
    analysis/       ensemble harvest utilities
  env/              pinned requirements + verify_env.py
  data_manifests/   dataset schema (data is distributed via the Harvard Dataverse DOI)
  optional_hpc/     a single generic batch-scheduler template (not required)
  tools/            repository checks (the cluster-info scrub gate)
  docs/             guides, math background, reports
    reference/      salvaged HIPPYNN-internals reference notes (third-party-style)
```

## Quickstart

1. **Environment.** Recreate the pinned stack (Python 3.13.5, torch 2.7.1+cu118, editable HIPPYNN and
   PySEQM at the pinned commits) by following `ml_namd/docs/installation_and_environment.md`, then verify:

   ```bash
   python ml_namd/env/verify_env.py
   ```

2. **Run a worked example end to end.** Each example runs the full pipeline as plain `python` commands —
   download data → gap-stratified seed-split → Model A + genA → Model B → composed ML-TSH → harvest —
   with **no job scheduler** (a CUDA GPU is required for training and dynamics):

   ```bash
   # full walkthrough: ml_namd/examples/enol/README.md (default) or ml_namd/examples/keto/README.md
   python ml_namd/examples/enol/02_train_model_A.py --seed 42
   python ml_namd/examples/enol/03_train_model_B.py --seed 42 --head both
   python ml_namd/examples/enol/04_run_ml_tsh.py    --seed 42 --head both
   python ml_namd/examples/enol/05_harvest.py       --head both
   ```

3. **Data.** Training pools and initial conditions are **not** stored in the repository; they are
   distributed through the Harvard Dataverse record (DOI `10.7910/DVN/HCL8EK`, base active-learning
   pools). Each example's `00_download_data.md` explains what to download and where to place it, and
   `ml_namd/data_manifests/DATA_MANIFEST.md` documents the array schema.

To fan the examples across multiple GPUs, adapt the generic template in `ml_namd/optional_hpc/` to your
own scheduler. Before committing, run the scrub gate so no machine-specific information enters the
repository:

```bash
python ml_namd/tools/check_no_cluster_info.py
```

## Documentation (`ml_namd/docs/`)

| Document | Purpose |
|---|---|
| `installation_and_environment.md` | End-to-end environment setup: recreate, verify, troubleshoot |
| `model_a_guide.md` | The energy + force multistate model (training + inference) |
| `model_b_math_background.md` | Theory: dENACR, the branching plane, the v3 head, the sign-consistent loss |
| `model_b_usage_guide.md` | Train + deploy Model B; the data contract; genA; gotchas |
| `stratified_split_guide.md` | The gap-stratified data-splitting procedure |
| `fssh_mlip_guide.md` | The composed A→B surface-hopping driver + guarded harvest |
| `fssh_pyseqm_guide.md` | The AM1/CIS reference driver + apples-to-apple harvest |
| `nve_at_mlip_guide.md` | Energy-conserving MD workflow + the four-check validation |
| `bring_your_own_dataset.md` | Adapting the pipeline to a new molecule/dataset |
| `validation_report.md` | Accuracy: Model A parity, NVE conservation, FSSH@MLIP vs reference |
| `inference_benchmark_report.md` | Throughput, batching/OOM, multi-GPU evidence |
| `migration_refactoring_changelog.md` | What changed during packaging + refactor |
| `known_limitations.md` | Open issues and unresolved questions |

## Reproducibility and provenance notes

- Every project-owned script was refactored for clarity while **preserving numerical behavior**; the
  surrounding HIPPYNN library tree is unmodified.
- The environment is reproduced from an explicit pinned stack (`ml_namd/env/`), and a scrub gate
  (`ml_namd/tools/check_no_cluster_info.py`) keeps the repository free of machine-, account-, and
  scheduler-specific information.
- Some runtime dependencies are **vendored** for completeness, each with a note: the FSSH@MLIP predictor
  bundle (`ml_namd/scripts/fssh_mlip/enol_ef_nacr_developer_handoff/`). The larger PySEQM data-generation
  fork used only for reference labeling is documented but not vendored.
- The salvaged HIPPYNN-internals notes under `ml_namd/docs/reference/` are third-party-style reference
  material, preserved from an earlier fork (see its `_PROVENANCE.md`).

## Citation

If you use this code, please cite the accompanying paper:

> A. Khanna, T. Karella, *et al.*, "Machine Learning for Non Adiabatic Excited State Molecular
> Dynamics," 2026 (DOI to be added on publication).

and, if you use the training pools or initial conditions, the replication dataset:

> A. Khanna, *Replication Data for: Machine Learning for Non Adiabatic Excited State Molecular
> Dynamics*, Harvard Dataverse, 2026. https://doi.org/10.7910/DVN/HCL8EK

A machine-readable citation is in [`ml_namd/CITATION.cff`](ml_namd/CITATION.cff); the full author list and
the paper DOI will be finalized there at publication.

## License and copyright

This software is released under the **BSD-3-Clause** license; see `LICENSE.txt`. Triad National Security,
LLC (Triad) owns the copyright to HIPPYNN, which it identifies as project number **LA-CC-19-093**; the
project-owned `ml_namd/` code is distributed under the same terms (`COPYRIGHT.txt`).

Copyright 2019. Triad National Security, LLC. All rights reserved. This program was produced under U.S.
Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad
National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All
rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of
Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on
its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

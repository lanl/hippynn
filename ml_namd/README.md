# ml_namd — MLIP-driven nonadiabatic molecular dynamics on HIPPYNN

Project-owned code and documentation for machine-learned interatomic potentials (MLIPs) that drive
nonadiabatic excited-state molecular dynamics (NA-ESMD), in particular fewest-switches surface hopping
powered by MLIPs (FSSH@MLIP), benchmarked against a semiempirical AM1/CIS FSSH@PYSEQM reference. Applied
to the enol and keto tautomers of acetylacetone, and designed to generalize to other molecules/datasets.

This folder lives inside a working copy of [lanl/hippynn](https://github.com/lanl/hippynn). The
surrounding repository is HIPPYNN itself; everything project-specific is kept here under `ml_namd/` so it
stays cleanly separable from the upstream library (objective: eventually merge selected features upstream).

## Collaborators
Ajay Khanna and Tomas Karella (LANL, Sergei Tretiak group).

## Objectives
- Inference optimization for all MLIP predictions, especially the FSSH driver.
- Merge the MLIP-for-NAMD features into HIPPYNN and keep them maintained upstream.

## Repository provenance
- Upstream: `lanl/hippynn`, branch `development` (base of this copy).
- The `upstream` git remote points at `lanl/hippynn` for future syncing and pull requests.
- Upstream integration is proposed in [lanl/hippynn#214](https://github.com/lanl/hippynn/pull/214)
  (base branch `development`).

## Layout
```
ml_namd/
  README.md              this file
  scripts/               project-owned Python pipeline (training, data_generation, fssh_mlip,
                         fssh_pyseqm, nve_mlip, analysis)
  examples/              runnable, self-contained worked examples (enol, keto) -- start here
  env/                   environment reproducibility (pinned requirements + verify_env.py)
  data_manifests/        dataset schema (data is distributed via the Harvard Dataverse DOI)
  optional_hpc/          a single generic batch-scheduler template (not required)
  tools/                 repository checks (e.g. the cluster-info scrub gate)
  docs/                  guides, math background, reports
    reference/           salvaged HIPPYNN-internals reference notes (third-party-style)
```

## Quick start
Run a complete, self-contained example end to end -- download data -> gap-stratified seed-split -> train
Model A + genA -> train Model B -> composed ML-TSH -> harvest: see `examples/enol/README.md` (default) or
`examples/keto/README.md`. Each step is a plain `python` command; a CUDA GPU is required for training and
dynamics. Recreate the environment first via `docs/installation_and_environment.md`.

## Datasets
Training pools and initial conditions are **not** stored in this repository; they are distributed through
the Harvard Dataverse record (DOI `10.7910/DVN/HCL8EK`). Each example's `00_download_data.md` explains what
to download and where to put it; `data_manifests/DATA_MANIFEST.md` documents the array schema.

## Salvaged reference documentation
`docs/reference/` contains a prior deep-dive into HIPPYNN's energy-prediction internals, preserved from an
earlier fork. These are third-party-style reference notes about HIPPYNN, not project deliverables.

## Status
The canonical pipeline is staged, refactored behavior-preservingly, documented, and made
environment-independent. Nothing under the upstream HIPPYNN tree is modified. Upstream integration
into HIPPYNN is proposed in [lanl/hippynn#214](https://github.com/lanl/hippynn/pull/214).

## Contributing & citation
See `CONTRIBUTING.md` for conventions (run `tools/check_no_cluster_info.py` before committing) and
`CITATION.cff` for how to cite this work.

## License and copyright
This software is released under the BSD-3-Clause license (see the root `LICENSE.txt`). Triad National
Security, LLC (Triad) owns the copyright to HIPPYNN, which it identifies as project number LA-CC-19-093;
the project-owned `ml_namd/` code is distributed under the same terms.

Copyright 2019. Triad National Security, LLC. All rights reserved. This program was produced under U.S.
Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad
National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All
rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of
Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on
its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

# ml_namd — MLIP-driven nonadiabatic molecular dynamics on HIPPYNN

Project-owned code and documentation for machine-learned interatomic potentials (MLIPs)
that drive nonadiabatic excited-state molecular dynamics (NA-ESMD), in particular fewest-switches
surface hopping powered by MLIPs (FSSH@MLIP), benchmarked against FSSH@PYSEQM.

This folder lives inside a private working copy of [lanl/hippynn](https://github.com/lanl/hippynn).
The surrounding repository is HIPPYNN itself; everything project-specific is kept here under
`ml_namd/` so it stays cleanly separable from the upstream library (objective: eventually merge
selected features upstream).

## Collaborators
Ajay Khanna and Tomas Karella (LANL, Sergei Tretiak group).

## Objectives
- Inference optimization for all MLIP predictions, especially the FSSH driver.
- Merge the MLIP-for-NAMD features into HIPPYNN and keep them maintained upstream.

## Repository provenance
- Upstream: `lanl/hippynn`, branch `development` (base of this copy).
- This repo: `Ajaykhanna/hippynn_NAMD` (private), working branch `ml_namd`.
- `upstream` remote points at `lanl/hippynn` for future syncing and pull requests.

## Layout (populated in phases)
```
ml_namd/
  README.md              this file
  PROJECT_BRIEF.md       the original task brief
  scripts/               project-owned Python + SLURM scripts (Phase 2)
  env/                   environment reproducibility records (Chicoma + Darwin)
  data_manifests/        dataset manifests + schema (data itself stays on D:/clusters)
  docs/                  guides, math background, validation + benchmark reports
    reference/           salvaged HIPPYNN-internals documentation (see below)
```

## Salvaged reference documentation
`docs/reference/` contains a prior deep-dive into HIPPYNN's energy-prediction internals,
preserved from an earlier public fork before it was removed. See
`docs/reference/_PROVENANCE.md` for the source commit. These are third-party-style reference
notes about HIPPYNN, not project deliverables.

## Status
Phase 1 (staging) in progress: repository created, project folder scaffolded. Still to come in
Phase 1: copy the canonical project scripts (Model A/B training, NVE@MLIP, FSSH@MLIP, FSSH@PYSEQM,
dataset extraction, SLURM) and record the Chicoma and Darwin environments. Phase 2 refactors and
documents every project-owned script. Nothing under the upstream HIPPYNN tree is modified.

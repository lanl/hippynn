# Contributing

Thanks for your interest in ml_namd. All project code lives under `ml_namd/`; the surrounding HIPPYNN
library tree is upstream and is kept **unmodified** -- changes to HIPPYNN itself should go upstream to
`lanl/hippynn`.

## Development setup
1. Recreate the environment: `docs/installation_and_environment.md` (Python 3.13, torch 2.7.1+cu118,
   editable HIPPYNN + PySEQM at the pinned commits). Verify with `env/verify_env.py`.
2. A CUDA GPU is required for training and dynamics; CPU is fine for inspection and small smokes.

## Before you commit
- **No environment-specific information.** The repository must be generic and independent of any one
  machine. Run the scrub gate and make sure it passes:

  ```bash
  python ml_namd/tools/check_no_cluster_info.py
  ```

  It rejects cluster filesystem paths, host/cluster names, account/group names, named software
  environments, and batch-scheduler directives (the last are allowed only under `optional_hpc/`).
- **Preserve behavior.** HIPPYNN node/loss class names and module filenames are pickled into trained
  checkpoints, so do not rename public classes, modules, CLI/config keys, or the `acn_` array names -- that
  breaks existing models. Prefer additive changes, and `python -m py_compile` your edits.
- **Respect the invariants** (see the guides): train-A == deploy-A (genA + `fssh_provenance`),
  raw-gap-no-floor, the fixed reference method (`hamiltonian_fd` + decoherence), per-IC seeding, guarded
  harvest, and float64 inference.

## Pull requests
Work on a branch; keep all project changes under `ml_namd/`. Describe what changed and how you verified it
(scrub-gate pass, `py_compile`, and a smoke run if the change touches the pipeline).

## License
This repository — HIPPYNN and the project-owned `ml_namd/` code — is released under the **BSD-3-Clause**
license; see the root `LICENSE.txt` and `COPYRIGHT.txt`. Triad National Security, LLC owns the copyright to
HIPPYNN (project number **LA-CC-19-093**), and the `ml_namd/` additions are distributed under the same
terms. Contributions are accepted under that license.

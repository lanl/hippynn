# Environment reproducibility records

Full captured records for the environments used by this project, on both clusters. Each record in
`records/` is the raw output of `capture_env.sh` (OS, CPU, Python, env manager, full CUDA/torch stack,
GPU/driver, HIPPYNN + PYSEQM commits, editable packages, environment variables, HPC modules, complete
`pip freeze`, and `conda list`). Per-environment `pip freeze` is also split out under `requirements/`.

Captured 2026-08-31. **These records document the working environments; they have not been re-built
and re-tested from scratch yet.** Do not overwrite any existing cluster environment file with a
generated one until the regenerated environment has been created and verified (see "Rebuild + verify").

## Environments at a glance

| Field | Chicoma `venv_master` | Darwin `venv_master` | Darwin `hiphopnn_env` |
|---|---|---|---|
| Role | Model A/B **training** + **FSSH@MLIP** (composed) | **FSSH@PYSEQM** reference | HIPPYNN **training** |
| Python bin | `/usr/projects/ml4chem/akhanna2/venv_master/bin/python` | `/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/venv_master/bin/python` | `/usr/projects/cint/anaconda3/hiphopnn_env/bin/python` |
| OS | SLES 15 SP5 | RHEL 8.10 | RHEL 8.10 |
| CPU | AMD EPYC 7662 (64-core), x86_64 | x86_64 | x86_64 |
| Env manager | `venv` on conda 23.1.0 env `mlip_md_dyn` (`--system-site-packages`) | `venv` on conda 23.11.0 / micromamba 2.8.1 | conda 23.11.0 / micromamba 2.8.1 |
| Python | 3.13.5 | 3.13.5 | 3.13.5 |
| PyTorch | 2.7.1+cu118 | 2.7.1+cu118 | 2.7.1+cu118 |
| CUDA (torch) | 11.8 | 11.8 | 11.8 |
| cuDNN | 9.1.0 | (see record) | (see record) |
| NumPy | 2.5.0 | 2.4.1 | 2.4.1 |
| HIPPYNN | 0.1.3.post0.dev11, git `d260cd5` @ `/users/akhanna2/data/software/hippynn` | 0.1.3.post0.dev11, git `d260cd5` @ `/vast/home/akhanna2/data/software/hiphopnn/hippynn` | same as Darwin `venv_master` |
| PYSEQM | git `e273f6d` (master) @ `/usr/projects/ml4chem/akhanna2/master_src` | master checkout at `…/fssh_ref_updated/master_src` (no `.git`; project memory records `107f5b0` — **verify before relying**) | n/a |

Notes:
- GPU model / driver version are **not** in these records because they were captured on login nodes
  where `nvidia-smi` is unavailable. Chicoma GPUs are A100 (40/80 GB partitions). To record the driver
  version, run `nvidia-smi --query-gpu=name,driver_version --format=csv` inside a GPU job on each cluster.
- Chicoma and Darwin PYSEQM master checkouts differ (`e273f6d` vs the memory-recorded `107f5b0`). Pin the
  exact commit per experiment; the FSSH@PYSEQM reference uses `hamiltonian_fd` TDC + `decohere_on_hop=True`.
- Both HIPPYNN installs are editable (`pip install -e`) from a local source tree; the git commit above is
  the reproducible pin, not the PyPI version string.

## Files
- `records/chicoma_venv_master.txt`, `records/darwin_venv_master.txt`, `records/darwin_hiphopnn_env.txt`
  — complete captured records (source of truth for every field).
- `requirements/<env>.requirements.txt` — `pip freeze` for each env (pip-managed packages only).
- `verify_env.py` — checks the running interpreter against the pinned key versions (below).

## Recreate an environment
These are venv-on-conda / conda environments with CUDA-enabled PyTorch, so `pip install -r` alone is
**not** sufficient (the CUDA toolkit, driver, and compiled wheels must match). Recommended order:

1. Create the base with the recorded Python and env manager, e.g. conda:
   `conda create -n <name> python=3.13.5` (or replicate the venv-on-conda layout the record shows).
2. Install the matching CUDA PyTorch wheel first, from the cu118 index:
   `pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
3. Install the remaining pip packages: `pip install -r requirements/<env>.requirements.txt`
   (drop the `torch` line already installed in step 2).
4. Install HIPPYNN editable at the pinned commit:
   `git clone <hippynn_src> && cd hippynn && git checkout d260cd5 && pip install -e .`
5. Install PYSEQM editable at the pinned commit (`e273f6d` for the Chicoma master).
6. Verify: `python verify_env.py`.

## Rebuild + verify
`verify_env.py` compares the active interpreter to the pinned versions and confirms the key imports
(torch + CUDA build, numpy, hippynn, seqm) succeed. Treat a regenerated environment as reproduced only
after `verify_env.py` passes AND a smoke run of the relevant script (training smoke / FSSH smoke)
completes. Only then consider replacing any existing cluster environment file.

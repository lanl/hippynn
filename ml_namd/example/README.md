# ml_namd enol working example (decoupled Model A + Model B)

A self-contained, runnable bundle that reproduces the completed **enol-form** FSSH@MLIP work: a
**decoupled** composed driver = a frozen energy+force network (**Model A**) feeding a scaled-NACR
coupling network (**Model B**), benchmarked against the AM1/CIS **FSSH@PYSEQM** reference on the same
initial conditions. Everything is at nf145 (145 features), enol, coupling pairs 12/13/23 (S1-S2, S1-S3,
S2-S3), initial surface S3.

The pipeline, physics, and per-script details are documented in `../docs/` (start with
`fssh_mlip_guide.md`, `model_a_guide.md`, `model_b_math_background.md`, `fssh_pyseqm_guide.md`). This
README is the how-to for *this* example.

## Layout

```
example/
  README.md                     this file
  configs/                      training configs (BALF and GALF, Model A and Model B)   [in git]
  slurm/                        4 runnable SLURM scripts                                 [in git]
  scripts/                      symlinks into ../scripts (single source of truth)        [Darwin]
  data/                         BALF/ and GALF/ training-ready arrays                    [Darwin]
  models/                       shipped trained checkpoints (A and B, BALF and GALF)     [Darwin]
  ics/                          100-IC subset for both FSSH streams                      [Darwin]
  out/ , runs/                  created by the runs (FSSH outputs / retrained models)    [Darwin]
```

`configs/` and `slurm/` are versioned in the repo. The heavy or site-specific assets (`data/`,
`models/`, `ics/`, the `scripts/` symlinks, `out/`, `runs/`) live as files on the Darwin path and are
git-ignored; use them directly from the shared filesystem (see "Assets live on Darwin").

## Prerequisites (Darwin)

- **HIPPYNN training env** (Model A / Model B training): `hiphopnn_env`
  (`/usr/projects/cint/anaconda3/hiphopnn_env/bin/python`).
- **FSSH env** (both FSSH drivers): the master-PySEQM `venv_master`
  (`/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/venv_master/bin/python`).
- Each SLURM script has a `USER-EDIT (site)` block at the top: set `--account` and `--partition` for
  your allocation, and override the interpreter with `PY=...` if your env path differs.

Verify the environment first: `python ../env/verify_env.py`.

## 1. Datasets (BALF and GALF)

`data/BALF/` and `data/GALF/` each hold the training-ready enol S0-dynamics arrays (50,000 frames,
`acn_` prefix, 15-atom AcAc, species order `[8,8,6,6,6,6,6,1x8]`):

| arrays | role | shape / dtype |
|---|---|---|
| `acn_R`, `acn_Z` | geometry, species | (50000,15,3) f64 ; (50000,15) i64 |
| `acn_sE0 .. acn_sE5` | Model A **targets** -- shifted state energies [eV] | (50000,) f64 |
| `acn_F0 .. acn_F5` | Model A **targets** -- state forces [eV/A] | (50000,15,3) f64 |
| `acn_dENACR12/13/23` | Model B **targets** -- scaled coupling (flattened 15x3) | (50000,45) f64 |
| `acn_GDV12/13/23` | Model B **inputs** -- gradient-difference vectors | (50000,45) f32 |
| `acn_GAP12/13/23` | Model B **inputs** -- state-pair energy gaps [eV] | (50000,) f64 |

Energy shift: `sE = E_raw - E0_opt`, `E0_opt(enol) = -1391.422970121047 eV`. BALF (base active-learning,
topology sampling) and GALF (gap-seeking active-learning) are two 50K sampling strategies over the same
enol manifold. Provenance and the full schema are in `../data_manifests/DATA_MANIFEST.md`.

## 2. Model A (energy + force, S0-S5)

Shipped checkpoint: `models/modelA_enol_nf145/{BALF,GALF}/` (`best_model.pt` + `experiment_structure.pt`).
This is the frozen E+F provider for the composed FSSH driver.

Retrain it on either pool (S0-S5):

```bash
DATASET=BALF sbatch slurm/train_model_a.slurm          # or DATASET=GALF
SMOKE=1 DATASET=BALF sbatch slurm/train_model_a.slurm  # 2-epoch smoke
```

The config is `configs/model_a_enol_<DATASET>.inp`. It trains S0-S5 (`n_states=5`). The **shipped**
decoupled Model A used the S0-S3 manifold that the enol FSSH needs; an S0-S5 model is a superset and is
FSSH-compatible (FSSH uses S0-S3). To reproduce the shipped checkpoint exactly, set `n_states=3`.

## 3. Model B (scaled NACR / dENACR, pairs 12/13/23)

Shipped checkpoint: `models/modelB_enol_nf145/{BALF,GALF}/`. v3 gap-aware head,
`gdv_pair_mixing_residual` assembly, phaseless loss. At training the ORACLE GDV/GAP are read from the
pool; at inference Model A supplies them.

Retrain it on either pool:

```bash
DATASET=BALF sbatch slurm/train_model_b.slurm          # or DATASET=GALF
```

Config: `configs/model_b_enol_<DATASET>.inp` (`n_states=3`, pairs 12/13/23). The math is in
`../docs/model_b_math_background.md`; the training/deploy how-to in `../docs/model_b_usage_guide.md`.

## 4. FSSH@MLIP (composed decoupled A + B)

100-IC composed sweep, states S0-S3, pairs 12/13/23, init S3, 1 ps @ 0.05 fs:

```bash
DATASET=BALF sbatch slurm/run_fssh_mlip.slurm          # or DATASET=GALF
```

Uses `models/model{A,B}_enol_nf145/<DATASET>/` and `ics/fssh_mlip/{R0_100.npy,V0_100.npy,Z.npy}`.
Per-trajectory h5 land in `out/fssh_mlip_<DATASET>/`. Driver: `scripts/fssh_mlip/namd_composed_batch.py`
(see `../docs/fssh_mlip_guide.md`).

## 5. FSSH@PYSEQM (AM1/CIS reference)

The same 100 ICs, run as the reference (n_states=4, hamiltonian_fd TDC + decohere, one trajectory per
physical core):

```bash
sbatch slurm/run_fssh_pyseqm.slurm
```

Uses `ics/fssh_pyseqm/{coords_100.xyz,velocity_100.out}` (the *same* 100 indices as the MLIP set, so the
two are apple-to-apple). Outputs in `out/fssh_pyseqm/`. Driver: `scripts/fssh_pyseqm/run_fssh_ref.py`
(see `../docs/fssh_pyseqm_guide.md`).

## 6. Analysis and plotting

Harvest **both** runs with the same harvester, then compare:

```bash
PY=/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/venv_master/bin/python
$PY scripts/analysis/harvest_fssh_full.py  out/fssh_mlip_BALF  -o mlip_BALF.npz
$PY scripts/analysis/harvest_fssh_full.py  out/fssh_pyseqm     -o ref.npz
```

`scripts/analysis/harvest_fssh_full.py` produces the survivor-conditional 6-panel apple-to-apple
ensemble (classical = active-surface fraction, quantum = excited-state |c|^2, S0 excluded);
`harvest_clean.py` is the strict clean-survivor variant. See `scripts/analysis/README.md` for the exact
conventions and CLI (confirm the flag names before scripting a batch).

## Assets live on Darwin (how they got here)

Per the packaging decision, `data/`, `models/`, `ics/`, and the `scripts/` symlinks are files on this
Darwin path, not committed to git. To rebuild them on a fresh checkout:

- **scripts/** -- relative symlinks into the repo's single source of truth:
  ```bash
  cd example && mkdir -p scripts
  ln -s ../../scripts/training    scripts/training
  ln -s ../../scripts/fssh_mlip   scripts/fssh_mlip
  ln -s ../../scripts/fssh_pyseqm scripts/fssh_pyseqm
  ln -s ../../scripts/analysis    scripts/analysis
  ```
- **models/** -- copied from the enol nf145 decoupled ladder
  `/usr/projects/cint/akhanna2/nve_md_darwin/bg_ladder_enol/nf145/{modelA,modelB}/{BALF,GALF}/model`
  (inference essentials only: `best_model.pt` + `experiment_structure.pt`).
- **data/** -- the training-ready columns built from `D:/ml_namd_project/data_gen/enol_balf_galf/{BALF,GALF}`
  (S0-S3 arrays copied from `pool_s0s3`; sE4/sE5 and F4/F5 derived from `energies_all_states.npy` /
  `forces_all_states.npy` with the enol energy shift).
- **ics/** -- the first 100 of the 1000 enol ICs:
  `fssh_pyseqm/` sliced from `.../fssh_ref_updated/ics/sampled_enol/{coords.xyz,velocity.out}`;
  `fssh_mlip/` sliced from `.../fssh_ref_updated/ics_matched/enol1ps_{R0,V0,Z}.npy`.

## Notes

- **Decoupled A + B** is the deployed composition: Model A gives E, F, gaps, and gradient-difference
  vectors; Model B maps those to the scaled coupling; the FSSH engine forms `d = dENACR / gap` (raw gap,
  no floor) and drives the hops. Do not clamp the gap -- it inverts small MLIP gaps and NaNs the amplitudes.
- **Species order** `[8,8,6,6,6,6,6,1x8]` and **pair order** `12,13,23` must stay consistent across the
  data, the configs, the driver, and the harvest.
- The BALF vs GALF split lets you compare sampling strategies; both are shipped so Model A and Model B
  can be retrained on either.

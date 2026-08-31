# Model B usage guide: training and deploying the scaled-NACR (dENACR) network

Practical how-to for the scaled nonadiabatic coupling model (Model B, v3 head). For the theory behind
the quantities used here (why the target is $\mathbf{D}_{ij}=\Delta_{ij}\mathbf{d}_{ij}$, what GDV and
GAP are, how the loss handles the electronic phase), read `model_b_math_background.md` first.

Model B is trained by the same script as Model A — `../scripts/training/unified_training.py` — with the
scaled-NACR target turned on. At inference it is paired with a frozen Model A (energies + forces) in a
composed A→B stack driven by the FSSH engine in `../scripts/fssh_mlip/`.

Placeholders: `<TRAIN>` = `../scripts/training`, `<DATA>` = a Model-B dataset directory, `<OUT>` = the
output model directory, `$PY` = the project Python (see `../env/README.md`; on Chicoma this is
`venv_master`). Replace machine paths with your own.

---

## 1. Prerequisites

- The project environment (see `../env/`): Python 3.13, torch 2.7.1+cu118, an editable HIPPYNN at the
  pinned commit. Verify with `python ../env/verify_env.py`.
- A **Model B dataset** with the arrays in Section 2.
- For inference: a trained **Model A** (E+F, S0–S5) whose predicted gaps and gradient-difference
  vectors feed Model B. Model A is trained by the same `unified_training.py` with
  `--train_energies --train_forces` (NACR off).

## 2. Data contract

Model B reads per-frame `.npy` arrays with the prefix `acn_` from `data_dir`. For the three-pair S0–S3
model (pairs `12, 13, 23`) the required arrays are:

| Array | Role | Shape | Dtype |
|---|---|---|---|
| `acn_R` | positions [Å] | `(N, n_at, 3)` | float64 |
| `acn_Z` | species | `(N, n_at)` | int64 |
| `acn_dENACR12`, `acn_dENACR13`, `acn_dENACR23` | **targets** (scaled coupling, flattened) | `(N, 45)` | float64 |
| `acn_GDV12`, `acn_GDV13`, `acn_GDV23` | **inputs** (`F_i − F_j`, flattened) | `(N, 45)` | float32 |
| `acn_GAP12`, `acn_GAP13`, `acn_GAP23` | **inputs** (`E_j − E_i`) [eV] | `(N,)` | float64 |

Building the pool from extracted E/F/dENACR is done by `../scripts/data_generation/build_bg_pool_enol.py`
(enol) — it copies R/Z/sE/F/dENACR into a `pool_s0s3/` dir and derives `GDV{ij}=F_i-F_j` and
`GAP{ij}=sE_j-sE_i`. The frame ordering of every array must match (same rows), or GDV/GAP will be
misaligned with the target.

## 3. Training a Model B

Training is config-file driven (one self-contained `.inp` per run; no CLI overrides). A minimal
three-pair v3 config:

```ini
# --- network (nf145 / ns20 example) ---
network_type = hiphopnn
n_features = 145
n_sensitivities = 20
n_interaction_layers = 2
n_atom_layers = 3
l_max = 3
n_max = 4
dist_soft_min = 0.7
dist_soft_max = 5.67638290906325
dist_hard_max = 11.833504244685173
n_atoms = 15
possible_species = 0 1 6 8

# --- targets: scaled NACR only, v3 GDV+gap-aware head ---
train_energies = false
train_forces = false
train_tdipoles = false
train_scaled_nacrs = true
n_states = 3                              # three excited states (S1,S2,S3); coupling pairs 12/13/23
targeted_scaled_nacr_pairs = 12 13 23
scaled_nacr_head = direct_vector_v3
direct_nacr_v2_mode = gdv_pair_mixing_residual
direct_nacr_v2_edge_scale = 0.5
direct_nacr_v2_symmetric_coeffs = true
direct_nacr_v2_enforce_zero_sum = true
gdv_source = file
gdv_file_template = {prefix}GDV{pair}.npy
gap_file_template = {prefix}GAP{pair}.npy
phasesign_corrected_scaled_nacr = false   # false => phaseless loss path (sign resolved in the loss)

# --- loss sign strategy (Section 4) ---
joint_state_nacr_loss = false             # true => joint per-state phase loss

# --- training ---
learning_rate = 0.000431
batch_size = 64
max_epochs = 200
dataset_prefix = acn_
data_dir  = <DATA>
output_dir = <OUT>
seed = 114514
```

Launch:

```bash
$PY <TRAIN>/unified_training.py --config_file model_b.inp
```

Key points:
- `scaled_nacr_head = direct_vector_v3` selects the gap-aware head (`direct_nacr_head_v3.py`);
  `direct_nacr_v2_mode = gdv_pair_mixing_residual` is the production assembly mode.
- `gdv_source = file` reads GDV/GAP from disk via the two templates (`{prefix}` = `acn_`,
  `{pair}` = `12`/`13`/`23`). At training these are the reference GDV/GAP; at inference Model A supplies
  them (Section 5).
- `targeted_scaled_nacr_pairs = 12 13 23` restricts the head to the S0–S3 excited pairs (S0 does not
  couple). The order defines the pair order everywhere downstream.
- The head's gap gate is zero-initialized, so the model starts equal to the gapless v2 and learns the
  gap correction.

The script validates the GDV and GAP input shapes before building the graph and prints, per pair,
whether the loss term is phaseless or joint.

## 4. Choosing the loss (phaseless vs joint)

- **Phaseless (default, `joint_state_nacr_loss = false`).** Each pair's sign is resolved independently
  per sample: term $= \mathrm{RMSE}/\sqrt{3 n_\text{at}} + \mathrm{MAE}$ over the min-sign residual. Robust
  and simple, but can settle into a pair-inconsistent sign set the FSSH driver cannot re-align.
- **Joint per-state (`joint_state_nacr_loss = true`).** Minimizes over a single global per-state gauge
  shared by all pairs (`joint_state_nacr_loss.py`), enforcing sign consistency. Note this term is a
  pure sum-of-squares; it emphasizes large near-CI magnitudes more than the phaseless blend, which can
  inflate the coupling norm — match the loss form when comparing the two (see the math doc, Section 5).
- `state_nacr_weights = 12:8.0` up-weights a specific pair (phaseless path only), e.g. to emphasize the
  S1–S2 near-CI coupling.

## 5. Deploying: the composed A→B FSSH driver

At inference Model B does **not** read GDV/GAP from disk — Model A produces them on the fly. The
predictor contract (`modelb_composed_predictor.ModelBComposedPredictor` in
`../scripts/fssh_mlip/enol_ef_nacr_developer_handoff/src/`) is:

```python
pred = predictor.predict(Z, R)          # Z:(n,natoms) int, R:(n,natoms,3) float
# returns a dict with, per state/pair:
#   pred["E_abs"][k]   absolute state energies        (used for gaps + total energy)
#   pred["sE"][k]      shifted state energies
#   pred["F"][k]       state forces                    (Model A; GDV = F_i - F_j internally)
#   pred["dENACR"][ij] scaled coupling, flattened 45   (Model B)
```

The FSSH engine (`hippynn_fssh.py`) then reconstructs the derivative coupling
$\mathbf{d}_{ij}=\mathbf{D}_{ij}/(E_j-E_i)$ using the **raw** gap (no floor), phase-matches the signs
against the previous step, and forms the time-derivative coupling $\sigma_{ij}=\mathbf{d}_{ij}\cdot\mathbf{v}$
that drives the hops. `ComposedSurfaceHopping` (`hippynn_fssh_composed.py`) swaps this composed predictor
in for the single joint model.

Run a batched composed FSSH sweep:

```bash
$PY <fssh_mlip>/namd_composed_batch.py \
    --modelA-dir <MODEL_A>/model --modelB-dir <MODEL_B>/model \
    --coords-npy R0.npy --vel-npy V0.npy --z-npy Z.npy \
    --states 0,1,2,3 --nacr-pairs 12,13,23 --initial-state 3 \
    --nsteps 200000 --timestep 0.05 --count 125 --start 0 \
    --output-dir out --output-prefix run
```

`--count` chunks the initial-condition set (125 is the OOM-safe maximum on a 40 GB A100); a SLURM array
or the dynamic GPU queue in `../scripts/slurm/keto_mlip_prod.slurm` fans chunks across GPUs.

## 6. Loading a trained Model B (unpickle contract)

A trained model ships `experiment_structure.pt` + `best_model.pt`. Loading rebuilds the graph, so the
modules the graph references must be importable **before** `torch.load`:

- `direct_nacr_head_v3` (the head class) must be on `sys.path`.
- If the model was trained with the joint loss, `joint_state_nacr_loss` must also be importable (its
  node class is pickled into the structure) or the load fails with `ModuleNotFound`.
- The composed driver's path bootstrap (see `hippynn_fssh.py`) adds the training modules and the
  predictor bundle to `sys.path` and injects `unified_training`'s symbols into `__main__` (some pickles
  reference training symbols via `__main__`).

## 7. Gotchas

- **Raw gap, no floor.** Never clamp the gap in the $\mathbf{d}=\mathbf{D}/\Delta$ division; flooring it
  inverts small MLIP gaps and produces amplitude NaNs that freeze hopping.
- **Pair order is load-bearing.** `12 13 23` must be consistent across the config, the data arrays, the
  driver `--nacr-pairs`, and any analysis. S0 is never a coupling pair here.
- **dtype.** dENACR/GAP are float64, GDV is float32; the training graph runs in the model dtype. The
  FSSH driver casts to float64.
- **Frame alignment.** GDV/GAP must be built from the same rows (same frame order) as the dENACR target
  and R/Z, or the branching-plane inputs will not correspond to the target.
- **Custom kernels.** For float64 MD/inference, custom kernels are disabled
  (`HIPPYNN_USE_CUSTOM_KERNELS=False`); this is expected and costs nothing at 15 atoms.

## 8. Verify before a big run

1. Environment: `python ../env/verify_env.py`.
2. Training smoke: a 2-epoch run of the `.inp` above with `max_epochs = 2` on a small `data_dir` — it
   should build the graph, print the per-pair loss tag, and write a `training_summary.json`.
3. Inference smoke: `../scripts/slurm/keto_s0s3_jointB_smoke.slurm` (5 ICs, 500 steps) — a pass is a
   clean model load, a written h5, hops firing, and zero amplitude NaN.

See also: `model_a_guide.md` (the E+F model that feeds GDV/GAP), `fssh_mlip_guide.md` (the driver), and
`../scripts/data_generation/` (building the dataset).

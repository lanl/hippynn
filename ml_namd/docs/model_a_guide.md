# Model A guide: the multistate energy + force network (E + F, S0–S5)

Model A is the workhorse MLIP: a single HIPPYNN network that predicts per-state adiabatic energies and
forces across S0–S5. It plays two roles:

1. It is the potential-energy engine for **NVE@MLIP** (energy-conserving dynamics on one state).
2. At FSSH inference it supplies the **gaps and gradient-difference vectors** that Model B consumes:
   $\mathrm{GAP}_{ij}=E_j-E_i$ and $\mathbf{GDV}_{ij}=\mathbf{F}_i-\mathbf{F}_j$ (see
   `model_b_math_background.md`). The deployed FSSH system is a frozen Model A + Model B stack.

Trained by the same script as Model B — `../scripts/training/unified_training.py` — with energies and
forces on and the coupling target off.

Placeholders: `<TRAIN>` = `../scripts/training`, `<DATA>` = a Model-A dataset directory, `<OUT>` = the
output model directory, `$PY` = the project Python (`../env/README.md`).

---

## 1. What Model A predicts

For each frame it emits, per state $k\in\{0,\dots,5\}$:

- `E_k` — the adiabatic state energy. Trained on the **shifted** energy $sE_k = E_k^\text{raw} - E_0^\text{opt}$,
  with $E_0^\text{opt}(\text{enol}) = -1391.422970121047$ and $E_0^\text{opt}(\text{keto}) = -1391.451136550848$ eV.
- $\mathbf{F}_k = -\nabla_{\mathbf{R}} E_k$ — the state force, obtained by autodifferentiation of `E_k`
  (a HIPPYNN `GradientNode`, sign $-1$).

Derived at inference without extra network cost: gaps `E_j − E_i` (subtractions) and gradient-difference
vectors `F_i − F_j` (subtractions) — the inputs Model B needs.

## 2. Data contract

Per-frame `acn_` arrays in `data_dir`:

| Array | Role | Shape | Dtype |
|---|---|---|---|
| `acn_R` | positions [Å] | `(N, n_at, 3)` | float64 |
| `acn_Z` | species | `(N, n_at)` | int64 |
| `acn_sE0 … acn_sE5` | **target** state energies (shifted) [eV] | `(N,)` | float64 |
| `acn_F0 … acn_F5` | **target** state forces [eV/Å] | `(N, n_at, 3)` | float64 |

Atom order in `acn_Z` is fixed: `[8, 8, 6, 6, 6, 6, 6, 1×8]` (2 O, 5 C, 8 H). The S0-dynamics dataset
carries the full S0–S10 payload; Model A uses the S0–S5 subset. Dataset locations and provenance are in
`../data_manifests/DATA_MANIFEST.md`.

## 3. Training

Config-file driven (one self-contained `.inp`; no CLI overrides). A multistate S0–S5 E+F config:

```ini
# --- network (nf145 / ns20) ---
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

# --- targets: energies + forces, coupling OFF ---
train_energies = true
train_forces = true
train_tdipoles = false
train_scaled_nacrs = false

# --- state selection: S0-S5 ---
n_states = 5                # n_states = number of EXCITED states (S1..S5); with the ground state => S0-S5
include_ground_state = true

# --- training ---
learning_rate = 0.0005
batch_size = 64
max_epochs = 200
dataset_prefix = acn_
data_dir  = <DATA>
output_dir = <OUT>
seed = 114514
```

Launch:

```bash
$PY <TRAIN>/unified_training.py --config_file model_a.inp
```

Notes:
- `train_scaled_nacrs = false` is what makes this **Model A** (energies + forces only). The same network
  architecture with the coupling target on would be a joint E+F+NACR model, which is a different object.
- Forces are trained via autodiff of the energies, so energy nodes are always built when forces are on.
- Per-state loss weights (`state_energy_weights`, `state_force_weights`) can emphasize particular states;
  higher excited states are harder and can overfit, so watch the validation/train gap on S4–S5.

**Multistate vs per-state.** A per-dynamics multistate model (one network across S0–S5, trained on one
dynamics distribution) is the default here — it is the demonstrated multistate MLIP and keeps one
consistent model family across states. A dedicated single-state E+F model can give marginally lower
per-surface force error at the cost of many models; the difference is small. Match the **training
geometry distribution** to the target surface (e.g. an S3-dynamics model for S3 NVE).

## 4. Outputs and loading

A trained model directory ships `experiment_structure.pt` + `best_model.pt` (and a
`training_summary.json` with the final metrics). Load with `load_model_from_cwd()` from the model dir —
not `load_checkpoint_from_cwd()`, which wants the larger `best_checkpoint.pt` the inference bundle omits.

**The MD pruning gotcha.** If the trained graph also carries oracle GDV input nodes (from a joint or
GDV-aware training), the stock PySEQM MD driver rejects them (`Unsupported HIPNN graph input 'GDV12'`).
The NVE runner (`../scripts/nve_mlip/pyseqm_hippynn_md.py`) prunes the graph to
`GraphModule([Z, R], [sE_k, F_k])` for a chosen state $k$; the prune is self-validating (it raises if an
output needed an unspecified input). Energies depend only on `network(Z, R)` and forces are a
`GradientNode` of the energies, so neither touches GDV.

## 5. Inference

Two ways Model A is evaluated:

**Energies + forces (for NVE, or to feed Model B).**

```python
# under torch.enable_grad() -- forces are a GradientNode; no_grad yields zeros
graph = GraphModule([Z_node, R_node], [sE_k_node, F_k_node])   # pruned to one state for MD
E_k, F_k = graph(Z, R.detach().clone())                        # pass a detached clone of R (see below)
```

For Model B's inputs, evaluate all needed states and take differences: $\mathrm{GAP}_{ij}=E_j-E_i$ and
$\mathbf{GDV}_{ij}=\mathbf{F}_i-\mathbf{F}_j$. The composed predictor
(`modelb_composed_predictor.ModelBComposedPredictor`) does exactly this internally, so the FSSH driver
just calls `predict(Z, R)`.

**Throughput (A100, reference).** Energy-only inference runs at roughly 3.3–3.6k geometries/s
(~0.28 ms/geom); adding forces drops it to ~405 geometries/s (~2.46 ms/geom) because the force is an
autograd pass (~9× the energy cost). Gaps and GDVs are then free subtractions. Plan force-bearing
inference (FSSH, NVE) around the ~2.5 ms/geom figure.

## 6. NVE@MLIP

Model A drives energy-conserving dynamics through the PySEQM integrator. See `nve_at_mlip_guide.md` for
the full workflow (start-geometry generation, excited-state optimization, the runner, and the
drift/integrity/reference-QM validation). Production timestep is 0.1 fs.

## 7. Gotchas

- **Energy offset.** Energies are trained shifted by $E_0^\text{opt}$; the NVE runner adds the offset
  back (`energy_offset`). Use the tautomer-correct value (enol vs keto).
- **Species order.** The driver asserts `acn_Z = [8,8,6,6,6,6,6,1×8]` and reads XYZ with `sort=False`;
  a reordered file breaks the correspondence.
- **`torch.enable_grad()` for forces.** Evaluating under `no_grad` silently returns zero forces.
- **Detached coordinate clone.** `GraphModule.forward` flips `requires_grad` in place; pass a detached
  clone so the integrator's own coordinate tensor is not permanently modified.
- **Custom kernels off** for float64 MD/inference (`HIPPYNN_USE_CUSTOM_KERNELS=False`).

## 8. Verify

1. Environment: `python ../env/verify_env.py`.
2. Training smoke: the config above with `max_epochs = 2` on a small `data_dir` → builds the graph,
   trains, writes `training_summary.json` with per-state E/F metrics.
3. Inference/NVE smoke: `../scripts/nve_mlip/nve_smoke_dt01.slurm` (short NVE) → the energy-conservation
   drift check should pass (drift ≈ 0 at dt = 0.1 fs).

See also: `model_b_usage_guide.md` (the coupling model Model A feeds), `nve_at_mlip_guide.md`, and
`fssh_mlip_guide.md`.

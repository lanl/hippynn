# Molecular Energy Prediction with HIPPYNN: Architecture and Implementation Deep-Dive

## Abstract

This document provides a comprehensive reverse-engineering analysis of a molecular energy prediction system built on the HIPPYNN framework. The system predicts ground and excited electronic state energies (S0-S3) for molecular configurations using the HIP-HOP-NN architecture, a message-passing neural network with spherical tensor decomposition. This analysis traces the complete computation flow from raw numpy arrays through HIPPYNN's graph-based abstraction layer to PyTorch tensor operations, with detailed documentation of tensor shapes, mathematical operations, and source code locations. This work serves as a foundation for developing custom prediction nodes and HIPPYNN-independent PyTorch implementations.

**Key Results**: Multi-state energy prediction for 25,000 molecular configurations of a 15-atom system with 75-dimensional atomic features, 2 interaction layers, and 63 radial basis functions.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [HIPPYNN Framework Overview](#2-hippynn-framework-overview)
3. [Data Pipeline](#3-data-pipeline)
4. [Neural Network Architecture](#4-neural-network-architecture)
5. [Energy Prediction Mechanism](#5-energy-prediction-mechanism)
6. [Loss Function Architecture](#6-loss-function-architecture)
7. [Training Infrastructure](#7-training-infrastructure)
8. [Under the Hood: PyTorch Integration](#8-under-the-hood-pytorch-integration)
9. [Configuration Reference](#9-configuration-reference)
10. [Output Artifacts](#10-output-artifacts)
11. [Extension Guide](#11-extension-guide)
12. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Physical Problem

**What are molecular electronic states?**

Molecules exist in discrete electronic energy levels characterized by different electron configurations:

- **S0 (Ground State)**: The lowest energy configuration where the molecule naturally resides at low temperatures
- **S1, S2, S3 (Excited States)**: Higher energy configurations accessed through photon absorption or other excitation mechanisms

These states are solutions to the time-independent Schrödinger equation under the **Born-Oppenheimer approximation**, which treats nuclear positions as parameters rather than dynamical variables:

```
Ĥₑₗ(R) Ψₙ(r; R) = Eₙ(R) Ψₙ(r; R)
```

Where:
- `R` = nuclear coordinates (15 atoms × 3 dimensions in our case)
- `r` = electronic coordinates
- `Eₙ(R)` = potential energy surface for state n
- `Ψₙ` = electronic wavefunction for state n

**Why predict energies?**

1. **Geometry Optimization**: Finding minimum energy structures (e.g., reaction intermediates, product structures)
2. **Reaction Dynamics**: Computing forces as `F = -∇ᴿE(R)` for molecular dynamics simulations
3. **Photochemistry**: Understanding non-adiabatic transitions between excited states (conical intersections, photoisomerization)
4. **Spectroscopy**: Predicting absorption/emission spectra from state energy differences

Traditional quantum chemistry methods (DFT, TDDFT, CASSCF) are computationally expensive, scaling as O(N³) to O(N⁷) with system size. Machine learning offers O(N) to O(N²) scaling with comparable accuracy after training.

### 1.2 Machine Learning Approach

**Why neural networks for potential energy surfaces?**

Potential energy surfaces (PES) are smooth, continuous functions of atomic positions with specific symmetries:
- **Translational invariance**: Energy unchanged by rigid translation
- **Rotational invariance**: Energy unchanged by rigid rotation
- **Permutational invariance**: Energy unchanged by swapping identical atoms

Neural networks can learn these properties implicitly from data or enforce them through architecture design.

**What makes atomistic ML different?**

Unlike image/text data with fixed-size grids or sequences, molecular systems have:
1. **Variable atom counts**: Networks must handle 10-1000+ atoms
2. **Permutation symmetry**: Atom ordering is arbitrary
3. **Locality**: Energy contributions are primarily from local neighborhoods
4. **Long-range interactions**: Electrostatics require global information

**Message-Passing Neural Networks (MPNNs)**

The solution is graph neural networks where:
- **Nodes** = atoms with feature vectors `hᵢ`
- **Edges** = interatomic pairs (typically within a cutoff radius)
- **Message passing** = iterative feature updates using neighbor information:

```
hᵢ⁽ᵗ⁺¹⁾ = φ(hᵢ⁽ᵗ⁾, Σⱼ ψ(hᵢ⁽ᵗ⁾, hⱼ⁽ᵗ⁾, rᵢⱼ, r̂ᵢⱼ))
```

Where `φ` and `ψ` are learned functions, `rᵢⱼ` is distance, `r̂ᵢⱼ` is direction.

HIPPYNN implements the **HIPNN** (Hierarchically Interacting Particle Neural Network) and **HIP-HOP-NN** (HIPNN with Higher-Order Potentials) architectures, which extend this paradigm with hierarchical feature aggregation and spherical tensor message passing.

---

## 2. HIPPYNN Framework Overview

### 2.1 Design Philosophy

HIPPYNN separates concerns across three abstraction layers:

```
Layer 1: GraphNodes      (User-facing API, automatic graph construction)
         ↓
Layer 2: Modules         (PyTorch nn.Modules, tensor operations)
         ↓
Layer 3: Custom Kernels  (Optimized CUDA/CPU operations)
```

**Key principles**:
1. **Graph-based computation**: Users build computation graphs by connecting nodes; HIPPYNN handles execution order
2. **Automatic differentiation integration**: Gradients flow naturally through PyTorch autograd
3. **Database abstraction**: Nodes reference data by `db_name`, decoupling model from storage
4. **Lazy evaluation**: Graphs are constructed but not executed until training/prediction

### 2.2 Core Abstractions

#### GraphNode (`hippynn/graphs/nodes/base/base.py:55`)

Base class for all computation nodes. Key responsibilities:

**Attributes**:
- `name`: Unique identifier for this node
- `parents`: List of parent nodes (inputs to this operation)
- `children`: List of child nodes (consumers of this output)
- `_output_names`: Names of output tensors (e.g., `["mol_energy", "atom_energies"]`)
- `main_output`: Default output when node is used as parent
- `db_name`: Database key for loading/saving (only for I/O nodes)
- `index_state`: Type annotation for tensor indexing (e.g., `IdxType.Atoms`, `IdxType.Systems`)

**Key methods**:
- `__init__(name, parents, **kwargs)`: Constructs node and registers with parents
- `_auto_module_()`: Automatically instantiates associated PyTorch module
- `get_ancestors()`: Returns all upstream nodes (transitive closure of parents)
- `torch_module`: Property accessing the underlying `nn.Module`

**Parent expansion**:
Nodes use `parent_expander` decorators to auto-complete parent lists. Example from `HEnergyNode`:

```python
@parent_expander.match(Network)
def expansion0(self, net, **kwargs):
    if "feature_sizes" not in self.module_kwargs:
        self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes
    pdindexer = find_unique_relative(net, AtomIndexer)
    return net, pdindexer.system_index, pdindexer.n_systems
```

This says: "If initialized with only a `Network` node, automatically find the `AtomIndexer` and extract `system_index` and `n_systems` outputs."

#### InputNode Family

**SpeciesNode** (`hippynn/graphs/nodes/inputs.py:10`)
```python
class SpeciesNode(AtomInTopologyNode, InputNode):
    input_names = ()  # No parents, data from database
    output_names = ("species",)
    output_index_states = (IdxType.SysAtom,)  # Per-atom in batch
```

Represents atomic numbers. During training, `db_name="Z"` loads data from `database["Z"]`.

**PositionsNode** (`hippynn/graphs/nodes/inputs.py:15`)
```python
class PositionsNode(PositionNode, InputNode):
    input_names = ()
    output_names = ("positions",)
    output_index_states = (IdxType.SysAtom, IdxType.ThreeDim)  # (atoms, 3)
```

Represents atomic coordinates. Setting `requires_grad=True` enables force computation:
```python
positions.requires_grad = True  # ∇ᴿ E = -F
```

#### NetworkNode (`hippynn/graphs/nodes/networks.py`)

Wraps a PyTorch `nn.Module` implementing the message-passing network.

```python
class Hipnn(Network, ExpandParents, AutoKw, MultiNode):
    _auto_module = HipnnModule  # The actual PyTorch model
    input_names = "species", "positions"
    output_names = ("features",)  # Actually returns list of feature tensors
```

The `AutoKw` mixin automatically passes `module_kwargs` to the module constructor. The `ExpandParents` mixin enables parent auto-completion.

#### TargetNode (`hippynn/graphs/nodes/targets.py`)

Connects network outputs to trainable prediction heads.

**HEnergyNode** (`hippynn/graphs/nodes/targets.py:15`)
```python
class HEnergyNode(Energies, HAtomRegressor, ExpandParents, AutoKw, MultiNode):
    input_names = "hier_features", "system_index", "n_systems"
    output_names = "mol_energy", "atom_energies", "energy_terms", ...
    main_output_name = "mol_energy"
    auto_module_class = target_modules.HEnergy
```

Creates a hierarchical energy predictor. The module applies learned linear layers to each feature block and sums contributions.

**HChargeNode** (`hippynn/graphs/nodes/targets.py:48`)
```python
class HChargeNode(Charges, HAtomRegressor, ExpandParents, AutoKw, MultiNode):
    input_names = ("hier_features",)
    output_names = "atom_charges", "partial_sums", "charge_hierarchality"
    auto_module_class = target_modules.HCharge
```

Similar to `HEnergyNode` but predicts per-atom charges instead of molecular energy.

#### LossNode (`hippynn/graphs/nodes/loss.py`)

Computes scalar loss values for training.

**MSELoss.of_node()** (`hippynn/graphs/nodes/loss.py:151`)
```python
@classmethod
def of_node(cls, node, db_name=None):
    """
    Automatically creates LossPredNode (from network output) and
    LossTrueNode (from database) and computes MSE between them.
    """
    if db_name is None:
        db_name = node.db_name
    return cls(node.name + "_mse", (node.main_output, db_name))
```

This class method is syntactic sugar for:
```python
pred = LossPredNode(network_output)  # Prediction
true = LossTrueNode(db_name)         # Ground truth
loss = MSELoss((pred, true))         # Compute loss
```

**l2reg()** (`hippynn/graphs/nodes/loss.py:175`)
```python
def l2reg(network_node):
    """
    L2 regularization: sum of squared weights.
    Calls network.torch_module.regularization_params() to find weights.
    """
    return Loss.of_module(L2Regularization(), (network_node,))
```

---

## 3. Data Pipeline

### 3.1 Input Data Format

The training script expects numpy arrays in a directory:

| File | Shape | Description | Units | Data Type |
|------|-------|-------------|-------|-----------|
| `acn_Z.npy` | `(25000, 15)` | Atomic numbers | Dimensionless | int64 |
| `acn_R.npy` | `(25000, 15, 3)` | Cartesian coordinates | Ångström | float32/64 |
| `acn_sE0.npy` | `(25000,)` | Ground state energy | eV | float32/64 |
| `acn_sE1.npy` | `(25000,)` | 1st excited state energy | eV | float32/64 |
| `acn_sE2.npy` | `(25000,)` | 2nd excited state energy | eV | float32/64 |
| `acn_sE3.npy` | `(25000,)` | 3rd excited state energy | eV | float32/64 |

**Atomic numbers**: Values are 0 (padding/missing), 1 (H), 6 (C), 8 (O), etc. The species set `[0, 1, 6, 8]` is derived from unique values in the data.

**Coordinates**: Right-handed Cartesian system. No specific origin or orientation required (network is translation/rotation invariant).

**Energies**: Typically relative to separated atoms or some reference geometry. The network learns these offsets during training.

### 3.2 DirectoryDatabase Internals

**Construction** (`hippynn/databases/ondisk.py`):
```python
database = hippynn.databases.DirectoryDatabase(
    name="acn_",           # Prefix for files
    directory="...",       # Path to data
    seed=114514,           # Random seed for splits
    test_size=0.1,         # 10% test set
    valid_size=0.1,        # 10% validation set
    **db_info              # Keys: ["Z", "R", "sE0", "sE1", ...]
)
```

**Loading process**:
1. **File discovery**: Scans `directory` for files matching `{name}{db_key}.npy`
2. **Data loading**: Uses `np.load()` with memory mapping for large files
3. **Type conversion**: Converts to `torch.get_default_dtype()` (float32 or float64)
4. **Index generation**: Creates random splits using `seed`:
   ```python
   indices = np.random.RandomState(seed).permutation(n_examples)
   train_idx = indices[:(1-test_size-valid_size)*n_examples]
   valid_idx = indices[...]
   test_idx = indices[...]
   ```
5. **Arrdict construction**: Builds dictionary `{db_key: torch.Tensor}`

**Batching**:
Database provides `get_batch()` method:
```python
def get_batch(self, indices, requires_grad=None):
    """
    Returns dict of tensors for given indices.
    Applies requires_grad settings for inputs like positions.
    """
    batch = {key: self.arr_dict[key][indices] for key in self.db_info["inputs"] + self.db_info["targets"]}
    if requires_grad:
        for key in requires_grad:
            batch[key].requires_grad = True
    return batch
```

### 3.3 Tensor Flow Through Training

**Batch shapes at each stage**:

```
Database arrays (stored):
  Z: (25000, 15)           → Batch: (64, 15)
  R: (25000, 15, 3)        → Batch: (64, 15, 3)
  sE0: (25000,)            → Batch: (64,)

After batching (batch_size=64):
  species: (64, 15)        [int64]
  positions: (64, 15, 3)   [float32]

Flattened for network:
  species: (960,)          [64 * 15 atoms]
  positions: (960, 3)

After embedding:
  features: (960, 3)       [one-hot for H, C, O]

After interaction layer 1:
  features: (960, 75)

After interaction layer 2:
  features: (960, 75)

HEnergy module:
  atom_energies: (960, 1)
  mol_energies: (64, 1)    [scatter_add by molecule]
```

The key transformation is **batched atoms → molecules**:
- Input: 64 molecules × 15 atoms = 960 atoms (some may be padding)
- `system_index`: `[0,0,...,0, 1,1,...,1, ..., 63,63,...,63]` (960 values)
- `scatter_add(atom_energies, system_index)` → (64,) molecular energies

---

## 4. Neural Network Architecture

### 4.1 HIPHopNN vs HIPNN

| Feature | HIPNN | HIPHopNN |
|---------|-------|----------|
| **Source** | `hippynn/networks/hipnn.py:61` | `hippynn/networks/hiphop.py:7` |
| **Interaction layer** | `InteractLayer` (scalar-only) | `HOPInteractionLayer` (tensor) |
| **Angular dependence** | None (radial only) | Spherical harmonics up to `l_max` |
| **Radial basis** | Same (`n_sensitivities`) | Same + Bessel functions (`n_max`) |
| **Best for** | Fast training, simple systems | High accuracy, directional bonds |
| **Parameters** | `~O(n_features² × n_sensitivities)` | `~O(n_features² × n_sensitivities × (l_max+1)² × n_max)` |

**When to use HIPHopNN**: Systems with strong directional bonding (π-bonds, hydrogen bonds), excited states where electronic density anisotropy matters.

### 4.2 Layer-by-Layer Breakdown

#### 4.2.1 Species Embedding

**Location**: Implicit in network construction via `OneHotEncoder`

**Mathematical operation**:
```
Z ∈ {0, 1, 6, 8}  (atomic numbers)
↓
One-hot encoding:
  Z=0 → [0, 0, 0]  (padding, ignored)
  Z=1 → [1, 0, 0]  (Hydrogen)
  Z=6 → [0, 1, 0]  (Carbon)
  Z=8 → [0, 0, 1]  (Oxygen)
↓
Convert to float32: features₀ ∈ ℝⁿᵃᵗᵒᵐˢ ˣ ³
```

**Implementation** (`hippynn/layers/indexers.py`):
```python
class OneHotEncoder(torch.nn.Module):
    def forward(self, Z):
        # Z: (n_atoms,) integer tensor
        one_hot = torch.zeros((Z.shape[0], len(self.species_set) - 1))
        for i, species in enumerate(self.species_set[1:]):  # Skip 0
            one_hot[:, i] = (Z == species).float()
        return one_hot
```

**Output shape**: `(n_atoms_in_batch, n_species-1)` = `(960, 3)` for our example

#### 4.2.2 Sensitivity Functions

**Purpose**: Transform pairwise distances into smooth radial basis functions.

**Mathematical form** (Inverse sensitivity, `hippynn/layers/hiplayers/sensitivity.py:20`):
```
Given distance r_ij between atoms i and j:

Sensitivity kernel:
s_k(r) = 1 / (1 + exp((r - μ_k) / σ)) × f_cutoff(r)

Where:
  μ_k = dist_soft_min + k × (dist_soft_max - dist_soft_min) / (n_sensitivities - 1)
  σ = (dist_soft_max - dist_soft_min) / (4 × n_sensitivities)
  k ∈ {0, 1, ..., n_sensitivities-1}

Cutoff function:
f_cutoff(r) = 0.5 × [cos(π × r / dist_hard_max) + 1]   if r < dist_hard_max
            = 0                                           if r ≥ dist_hard_max
```

**Parameters used**:
- `dist_soft_min = 0.7` Å: Minimum "comfortable" distance (near bond lengths)
- `dist_soft_max = 5.67638` Å: Maximum informative distance
- `dist_hard_max = 11.833504` Å: Absolute cutoff (saves computation, pairs beyond this are ignored)
- `n_sensitivities = 63`: Number of radial basis functions

**PyTorch implementation**:
```python
class InverseSensitivityModule(torch.nn.Module):
    def __init__(self, n_dist, mind_soft, maxd_soft, hard_cutoff):
        super().__init__()
        centers = torch.linspace(mind_soft, maxd_soft, n_dist)
        self.register_buffer("centers", centers)
        self.sharpness = 4 * n_dist / (maxd_soft - mind_soft)
        self.cutoff_distance = hard_cutoff

    def forward(self, dists):
        # dists: (n_pairs,)
        dists = dists.unsqueeze(1)  # (n_pairs, 1)
        sense = 1.0 / (1.0 + torch.exp(self.sharpness * (dists - self.centers)))
        # Cutoff
        cutoff = 0.5 * (torch.cos(np.pi * dists / self.cutoff_distance) + 1.0)
        cutoff = torch.where(dists < self.cutoff_distance, cutoff, torch.zeros_like(cutoff))
        return sense * cutoff  # (n_pairs, n_sensitivities)
```

**Tensor shapes**:
- Input: `(n_pairs,)` distances
- Output: `(n_pairs, 63)` sensitivity values

#### 4.2.3 Interaction Layers

**Core message-passing equation** (`hippynn/layers/hiplayers/interactions.py:44`):

```
Z'_i = (VSZ)_i + (WZ)_i + b

Where:
  Z_i: Input features for atom i  (n_features,)
  Z'_i: Output features for atom i (n_features_out,)

  (SZ)_i = Σ_{j∈neighbors(i)} s(r_ij) ⊙ Z_j    [environment aggregation]
  V: Interaction weights (n_sensitivities, n_features_out, n_features_in)
  W: Self-interaction weights (n_features_in, n_features_out)
  b: Bias (n_features_out,)

  s(r_ij): Sensitivity values (n_sensitivities,)
  ⊙: Element-wise multiplication
```

**Step-by-step forward pass**:

1. **Compute environment features** (`custom_kernels.envsum`):
   ```python
   # For each atom i, aggregate neighbor features weighted by sensitivities
   env_features = custom_kernels.envsum(
       sense_vals,    # (n_pairs, n_sensitivities)
       in_features,   # (n_atoms, n_features_in)
       pair_first,    # (n_pairs,) indices of atom i
       pair_second    # (n_pairs,) indices of atom j
   )
   # Output: (n_atoms, n_sensitivities, n_features_in)
   ```

   This performs: `env[i, k, :] = Σ_j sense[ij, k] * features[j, :]`

2. **Reshape for matrix multiplication**:
   ```python
   env_features = env_features.reshape(n_atoms, n_sensitivities * n_features_in)
   weights_rs = self.int_weights.permute(0, 2, 1).reshape(n_sensitivities * n_features_in, n_features_out)
   ```

3. **Apply interaction weights**:
   ```python
   features_out = torch.mm(env_features, weights_rs)  # (n_atoms, n_features_out)
   ```

4. **Add self-interaction**:
   ```python
   features_out += self.selfint(in_features)  # Linear layer with bias
   ```

**HIPHopNN extension** (`hippynn/networks/hiphop.py:21`):

Replaces `s(r_ij)` with tensor-valued sensitivities:
```
s(r_ij) → T_lm(r̂_ij) × R_n(r_ij)

Where:
  T_lm: Spherical harmonics (l ≤ l_max, m ∈ [-l, l])
  R_n: Radial basis functions (Bessel-like, n ≤ n_max)
  r̂_ij = (r_ij) / |r_ij|  (unit vector)
```

**Implementation**:
```python
# Compute spherical tensor basis
rhats = pair_coord / pair_dist.unsqueeze(1)  # (n_pairs, 3)
tensor_terms = self.tensor_extractor(rhats)  # List of tensors for each l
tensor_rhats = torch.cat(tensor_terms[:l_max+1], dim=-1)  # (n_pairs, n_tensor_features)

# Message passing with tensor sensitivities
features = interaction_layer(features, pair_first, pair_second, pair_dist, tensor_rhats)
```

**Tensor shapes per layer**:
```
Input features:   (n_atoms, 75)
Pair sensitivities: (n_pairs, 63) or (n_pairs, 63×n_tensor) for HIPHopNN
Environment features: (n_atoms, 63×75) or (n_atoms, 63×n_tensor×75)
Output features:  (n_atoms, 75)
```

#### 4.2.4 Atom Layers

**Purpose**: Local (non-message-passing) feature refinement after each interaction block.

**Implementation** (`hippynn/networks/hipnn.py:200`):
```python
for j in range(n_atom_layers):  # n_atom_layers = 3
    layer = torch.nn.Linear(out_size, middle_size)
    torch.nn.init.xavier_normal_(layer.weight.data)

    if self.resnet:  # ResNet connections for gradient flow
        layer = ResNetWrapper(layer, out_size, middle_size, out_size, activation)
    this_block.append(layer)
```

**ResNet wrapper** (`hippynn/layers/transform.py`):
```python
class ResNetWrapper(torch.nn.Module):
    def forward(self, x):
        # Standard residual connection:
        # out = activation(base_layer(x)) + x
        hidden = self.base_layer(x)        # (n, middle_size)
        hidden = self.activation(hidden)
        out = self.output_projection(hidden)  # (n, out_size)
        return out + x  # Skip connection
```

**Why ResNet?** Prevents gradient vanishing in deep networks. With 2 interaction blocks × 3 atom layers = 6 total layers, gradients can attenuate. Skip connections provide direct paths.

**Activation function**: `Softplus(x) = log(1 + exp(x))` (smooth ReLU alternative)

### 4.3 Network Configuration Reference

| Parameter | Value | Valid Range | Effect on Architecture |
|-----------|-------|-------------|------------------------|
| **n_features** | 75 | 16-256 | Width of feature vectors at each layer. Higher = more expressive but slower/overfits. |
| **n_sensitivities** | 63 | 10-100 | Number of radial basis functions. Controls resolution of distance encoding. |
| **dist_soft_min** | 0.7 Å | 0.5-1.5 Å | Minimum distance for sensitivity functions. Set near typical bond lengths. |
| **dist_soft_max** | 5.676 Å | 3-8 Å | Maximum distance for informative interactions. System-dependent. |
| **dist_hard_max** | 11.833 Å | 5-15 Å | Absolute cutoff. Set to ~2× typical non-bonded interaction range. |
| **n_interaction_layers** | 2 | 1-5 | Number of message-passing rounds. More layers = larger receptive field but slower. |
| **n_atom_layers** | 3 | 1-5 | Local transformations per interaction block. Balances expressiveness vs. depth. |
| **l_max** | 3 | 0-4 | Max angular momentum for HIPHopNN. 0=scalar, 1=vector, 2=quadrupole, etc. Higher = more accurate but expensive. |
| **n_max** | 4 | 1-6 | Radial basis order for HIPHopNN. Higher = better radial resolution. |
| **possible_species** | [0,1,6,8] | - | Atomic numbers present in dataset. 0 is padding/missing. |

**Rule of thumb**:
- Small molecules (<20 atoms): `n_features=50-100`, `n_interaction_layers=2`
- Medium systems (20-100 atoms): `n_features=75-150`, `n_interaction_layers=2-3`
- Large systems (>100 atoms): Consider more layers over more features for efficiency

---

## 5. Energy Prediction Mechanism

### 5.1 From Atomic Features to Molecular Energy

**Hierarchical energy scheme**:

HIPPYNN implements **Behler-Parrinello style** atomic decomposition:
```
E_mol = Σ_i E_atom(Z_i, {features_i})

Where E_atom is learned per atomic environment, NOT per species.
```

The "hierarchical" aspect means predictions from multiple feature blocks are summed:
```
E_atom(i) = Σ_t w_t^T · features_i^(t)

Where:
  t ∈ {0, 1, ..., n_interaction_layers}: Feature blocks from each layer
  w_t: Learned linear weights for block t
  features_i^(t): Features of atom i after t interaction layers
```

### 5.2 HEnergyNode Internals

**Module definition** (`hippynn/layers/targets.py:10`):

```python
class HEnergy(torch.nn.Module):
    def __init__(self, feature_sizes, first_is_interacting=False, n_target=1):
        """
        feature_sizes: Tuple of feature dimensions at each layer
                       e.g., (3, 75, 75) for [embedding, layer1, layer2]
        first_is_interacting: If True, skip first block (non-interacting one-hot features)
        n_target: Number of energy outputs (1 for single state, 4 for S0-S3)
        """
        self.summer = indexers.MolSummer()  # Atom → Molecule aggregator

        # Create one Linear layer per feature block
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(nf, n_target, bias=bias)
            for nf, bias in zip(feature_sizes, biases)
        ])

    def forward(self, all_features, system_index, n_systems):
        """
        all_features: List of [features₀, features₁, ..., features_T]
        system_index: (n_atoms,) molecule ID for each atom
        n_systems: Total number of molecules in batch

        Returns: (mol_energy, atom_energies, partial_sums, hierarchicality, ...)
        """
        # Apply each layer to corresponding feature block
        partial_energies = [
            layer(features)
            for features, layer in zip(all_features, self.layers)
        ]
        # partial_energies: List of (n_atoms, 1) tensors

        # Sum over atoms to get molecular contributions per block
        partial_mol_energies = [
            self.summer(pe, system_index, n_systems)
            for pe in partial_energies
        ]
        # partial_mol_energies: List of (n_molecules, 1) tensors

        # Total atomic energy
        total_atom_energy = sum(partial_energies)  # (n_atoms, 1)

        # Total molecular energy
        total_mol_energy = self.summer(total_atom_energy, system_index, n_systems)
        # (n_molecules, 1)

        return total_mol_energy, total_atom_energy, partial_mol_energies, ...
```

**MolSummer implementation** (`hippynn/layers/indexers.py`):
```python
class MolSummer(torch.nn.Module):
    def forward(self, atom_data, system_index, n_systems):
        """
        Aggregate per-atom quantities to per-molecule.

        atom_data: (n_atoms, n_features)
        system_index: (n_atoms,) integer, molecule ID for each atom
        n_systems: int, total number of molecules

        Returns: (n_systems, n_features)
        """
        output_shape = (n_systems, atom_data.shape[1])
        mol_data = torch.zeros(output_shape, dtype=atom_data.dtype, device=atom_data.device)

        # Scatter-add: mol_data[system_index[i]] += atom_data[i]
        mol_data.scatter_add_(0, system_index.unsqueeze(1).expand_as(atom_data), atom_data)

        return mol_data
```

**Example with concrete shapes**:
```
Batch: 64 molecules × 15 atoms = 960 atoms

all_features = [
    features₀: (960, 3),   # One-hot embedding
    features₁: (960, 75),  # After interaction block 1
    features₂: (960, 75)   # After interaction block 2
]

self.layers = [
    Linear(3 → 1),
    Linear(75 → 1),
    Linear(75 → 1)
]

Forward pass:
  partial_energies[0] = Linear(features₀) → (960, 1)
  partial_energies[1] = Linear(features₁) → (960, 1)
  partial_energies[2] = Linear(features₂) → (960, 1)

  total_atom_energy = sum(partial_energies) → (960, 1)

  total_mol_energy = MolSummer(total_atom_energy, system_index, 64) → (64, 1)
```

### 5.3 Multi-State Prediction

**Shared backbone, separate heads**:

```python
# Single network for all states
network = networks.HipHopnn("hiphopnn_model", (species, positions), ...)

# Separate energy prediction head per state
energy_nodes = []
for state in ["sE0", "sE1", "sE2", "sE3"]:
    energy = targets.HEnergyNode(state, network, module_kwargs=None)
    energy.mol_energy.db_name = state  # Link to database targets
    energy_nodes.append(energy)
```

**Parameter sharing**:
- **Network backbone** (embedding, interaction layers, atom layers): **Shared** across all states
- **HEnergy linear layers**: **Independent** for each state (4 separate `HEnergy` modules)

**Why this works**:
Electronic states share the same molecular geometry and atomic features. Differences emerge in the final energy prediction. This is analogous to multi-task learning: the backbone learns general atomic environments, heads specialize to states.

**Total parameters**:
```
Shared network: ~500k parameters (depends on n_features, n_sensitivities)
Per-state heads: ~10k parameters each × 4 states = 40k
Total: ~540k parameters
```

**Benefits**:
1. **Efficiency**: Training 4 states costs ~1.2× single-state (not 4×)
2. **Regularization**: States regularize each other through shared features
3. **Consistency**: Energy ordering (S0 < S1 < S2...) emerges naturally if data is consistent

**Potential issues**:
- If states have very different physics (e.g., charge-transfer vs. Rydberg), may need state-specific features
- Solution: Use separate networks or add state-conditional layers

---

## 6. Loss Function Architecture

### 6.1 Per-State Loss Components

**Combined loss for robustness**:
```
L_state = √(MSE(pred, true)) + MAE(pred, true)
        = RMSE(pred, true) + MAE(pred, true)
```

**Why RMSE + MAE?**

| Metric | Mathematical Form | Characteristics | Use Case |
|--------|-------------------|-----------------|----------|
| **MSE** | `(1/N) Σ(yᵢ - ŷᵢ)²` | Sensitive to outliers, differentiable | Penalizes large errors heavily |
| **RMSE** | `√MSE` | Same units as target, less outlier-sensitive than MSE | Interpretable error magnitude |
| **MAE** | `(1/N) Σ|yᵢ - ŷᵢ|` | Robust to outliers, non-smooth at 0 | Median-like behavior |

**Combination benefits**:
1. RMSE encourages fitting "most" points well (mean-like)
2. MAE prevents overfitting to outliers (median-like)
3. Together: balanced optimization

**Implementation** (`hippynn/graphs/nodes/loss.py:151`):
```python
for i, energy_output in enumerate(energy_outputs):
    state_name = f"sE{i}"

    # MSELoss.of_node() creates pred/true nodes automatically
    state_rmse = loss.MSELoss.of_node(energy_output) ** 0.5
    validation_losses[f"{state_name}-RMSE"] = state_rmse

    state_mae = loss.MAELoss.of_node(energy_output)
    validation_losses[f"{state_name}-MAE"] = state_mae

    state_loss = state_rmse + state_mae
    validation_losses[f"{state_name}-Loss"] = state_loss
```

**Under the hood**:
```python
class MSELoss(Loss):
    @classmethod
    def of_node(cls, node, db_name=None):
        # Auto-create pred and true nodes
        pred_node = LossPredNode(f"{node.name}_pred", (node.main_output,))
        true_node = LossTrueNode(f"{node.name}_true", db_name or node.db_name)
        return cls(f"{node.name}_mse", (pred_node, true_node))

    def forward(self, pred, true):
        return torch.mean((pred - true) ** 2)
```

### 6.2 Total Loss Aggregation

**Full training objective**:
```
L_total = Σ_states (ENERGY_WEIGHT × L_state) + L2_WEIGHT × ||θ||²

Where:
  ENERGY_WEIGHT = 1.0  (equal weighting for all states)
  L2_WEIGHT = 2e-5
  ||θ||² = Σ_p ∈ regularization_params ||p||²
```

**Code**:
```python
total_loss = None
for i, energy_output in enumerate(energy_outputs):
    state_loss = state_rmse + state_mae

    if total_loss is None:
        total_loss = ENERGY_WEIGHT * state_loss
    else:
        total_loss += ENERGY_WEIGHT * state_loss

# Add L2 regularization
l2_reg = loss.l2reg(network)
total_loss += L2_WEIGHT * l2_reg
```

**Adjusting relative weights**:

If certain states are more important (e.g., ground state for dynamics):
```python
STATE_WEIGHTS = {"sE0": 2.0, "sE1": 1.0, "sE2": 1.0, "sE3": 1.0}
total_loss = sum(STATE_WEIGHTS[state] * state_losses[state] for state in states)
```

### 6.3 Loss Node Implementation

**Automatic graph traversal** (`hippynn/experiment/assembly.py:72`):

When `assemble_for_training()` is called with a loss node, HIPPYNN:

1. **Finds all upstream nodes**:
   ```python
   loss_required_nodes = get_subgraph([train_loss, *validation_losses])
   ```

2. **Identifies predictions vs. targets**:
   ```python
   predictions = [x.origin_node for x in loss_required_nodes if isinstance(x, LossPredNode)]
   targets = [x.origin_node for x in loss_required_nodes if isinstance(x, LossTrueNode)]
   ```

3. **Traces back to inputs**:
   ```python
   inputs = [p for node in predictions
             for p in node.get_ancestors()
             if isinstance(p, InputNode)]
   ```

4. **Creates GraphModules**:
   ```python
   model = GraphModule(inputs, predictions)       # Z, R → predicted energies
   loss_module = GraphModule([*predictions, *targets], [train_loss])  # pred, true → loss
   ```

**L2 regularization** (`hippynn/graphs/nodes/loss.py:175`):

```python
def l2reg(network_node):
    """
    Finds all parameters marked for regularization in the network.
    """
    class L2Regularization(torch.nn.Module):
        def forward(self, network_features):
            # Get parameters from network's regularization_params()
            params = self.network.regularization_params()
            return sum(torch.sum(p ** 2) for p in params)

    return Loss.of_module(L2Regularization(), (network_node,))
```

**Which parameters are regularized?**

In `Hipnn` (`hippynn/networks/hipnn.py:220`):
```python
def regularization_params(self):
    params = []
    for block in self.blocks:
        for layer in block:
            if hasattr(layer, "regularization_params"):
                params.extend(layer.regularization_params())
            else:
                params.append(layer.weight)  # Default: all weights
    return list(set(params))  # Remove duplicates
```

For `InteractLayer`, explicitly defined (`hippynn/layers/hiplayers/interactions.py:41`):
```python
def regularization_params(self):
    return [self.int_weights, self.selfint.weight]
    # Excludes biases (common practice)
```

---

## 7. Training Infrastructure

### 7.1 Assembly Phase

**`assemble_for_training()`** (`hippynn/experiment/assembly.py:98`):

**Purpose**: Convert node graph into executable PyTorch modules.

**Inputs**:
- `train_loss`: Loss node for backpropagation
- `validation_losses`: Dict of {name: loss_node} for monitoring
- `plot_maker`: Optional plotting callback

**Process**:

1. **Determine required nodes**:
   ```python
   loss_required_nodes = (train_loss, *validation_losses.values())
   if plot_maker:
       loss_required_nodes += plot_maker.required_nodes

   inputs, outputs, targets = determine_out_in_targ(*loss_required_nodes)
   ```

2. **Build database info**:
   ```python
   db_info = {
       "inputs": [node.db_name for node in inputs],    # ["Z", "R"]
       "targets": [node.db_name for node in targets]   # ["sE0", "sE1", "sE2", "sE3"]
   }
   ```

3. **Create GraphModules**:
   ```python
   # Model: inputs → predictions
   model = GraphModule(inputs, outputs)

   # Loss: predictions + targets → scalar loss
   pred_nodes = [LossPredNode(out) for out in outputs]
   true_nodes = [LossTrueNode(target.db_name) for target in targets]
   train_loss_module = GraphModule([*pred_nodes, *true_nodes], [train_loss])

   # Validation: separate copy to avoid training/eval conflicts
   valid_loss_module = copy.deepcopy(train_loss_module)
   ```

4. **Create Evaluator**:
   ```python
   evaluator = Evaluator(
       model, valid_loss_module, validation_names,
       plot_maker, stopping_key="Loss"
   )
   ```

5. **Return TrainingModules**:
   ```python
   return TrainingModules(model, train_loss_module, evaluator), db_info
   ```

**GraphModule** (`hippynn/graphs/__init__.py`):

A `GraphModule` is a PyTorch `nn.Module` that executes a subgraph:
```python
class GraphModule(torch.nn.Module):
    def __init__(self, input_nodes, output_nodes):
        # Topologically sort nodes from inputs to outputs
        self.execution_order = topological_sort(input_nodes, output_nodes)
        self.input_names = [n.name for n in input_nodes]
        self.output_names = [n.name for n in output_nodes]

    def forward(self, **inputs):
        # Execute nodes in order
        cache = {}
        for node in self.execution_order:
            if node in input_nodes:
                cache[node] = inputs[node.name]
            else:
                parent_outputs = [cache[p] for p in node.parents]
                cache[node] = node.torch_module(*parent_outputs)

        return tuple(cache[n] for n in output_nodes)
```

### 7.2 Controller System

**PatienceController** (`hippynn/experiment/controllers.py`):

Manages training loop decisions: when to stop, when to adjust learning rate, when to increase batch size.

**Key attributes**:
```python
class PatienceController:
    optimizer: AdamW
    scheduler: RaiseBatchSizeOnPlateau
    batch_size: int = 64
    eval_batch_size: int = 256
    max_epochs: int = 200
    stopping_key: str = "Loss"  # Which validation metric to monitor
    termination_patience: int = 50  # Epochs without improvement before stopping
    fraction_train_eval: float = 0.1  # Fraction of training set for evaluation
```

**Early stopping logic**:
```python
def check_termination(self, metrics):
    """
    Called after each epoch with validation metrics.
    """
    current_loss = metrics["valid"][self.stopping_key]

    if current_loss < self.best_loss:
        self.best_loss = current_loss
        self.patience_counter = 0
        self.save_best_model()
    else:
        self.patience_counter += 1

    if self.patience_counter >= self.termination_patience:
        print(f"No improvement for {self.termination_patience} epochs. Stopping.")
        return True  # Signal to stop training

    return False
```

**RaiseBatchSizeOnPlateau** (`hippynn/experiment/controllers.py`):

Dynamic batch size scaling for better GPU utilization:
```python
class RaiseBatchSizeOnPlateau:
    patience: int = 16  # Epochs of plateau before increasing batch size
    factor: float = 0.2  # Learning rate reduction factor
    max_batch_size: int = 256

    def step(self, metrics):
        current_loss = metrics["valid"]["Loss"]

        if current_loss < self.best_loss - self.threshold:
            self.best_loss = current_loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

        if self.plateau_counter >= self.patience:
            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.factor

            # Increase batch size (if possible)
            if self.batch_size < self.max_batch_size:
                self.batch_size = min(self.batch_size * 2, self.max_batch_size)
                print(f"Increasing batch size to {self.batch_size}")

            self.plateau_counter = 0
```

**Why increase batch size?**
- Larger batches → more stable gradients → can use lower learning rate effectively
- As training progresses, loss landscape becomes smoother → benefit from larger batches
- GPU underutilization early in training (small batches) → increase to improve throughput

### 7.3 Training Loop

**`train_model()`** (`hippynn/experiment/routines.py:218`):

```python
def train_model(training_modules, database, controller, metric_tracker, callbacks=None):
    """
    Main training loop.

    training_modules: TrainingModules(model, loss, evaluator)
    database: DirectoryDatabase with train/valid/test splits
    controller: PatienceController
    metric_tracker: MetricTracker for logging
    """
    model = training_modules.model
    loss_module = training_modules.loss
    evaluator = training_modules.evaluator
    optimizer = controller.optimizer

    for epoch in range(controller.max_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_indices in database.make_batches(controller.batch_size, "train"):
            # Get batch data
            batch = database.get_batch(batch_indices)

            # Forward pass
            predictions = model(**{k: batch[k] for k in model.input_names})
            loss_inputs = {**dict(zip(model.output_names, predictions)),
                          **{k: batch[k] for k in database.db_info["targets"]}}
            loss = loss_module(**loss_inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation phase
        model.eval()
        with torch.no_grad():
            valid_metrics = evaluator.evaluate(database, "valid", controller.eval_batch_size)
            train_eval_metrics = evaluator.evaluate(database, "train", controller.eval_batch_size,
                                                     fraction=controller.fraction_train_eval)

        # Log metrics
        metrics = {
            "train": {"Loss": epoch_loss / n_batches, **train_eval_metrics},
            "valid": valid_metrics
        }
        metric_tracker.update(epoch, metrics)

        # Check stopping criteria
        controller.scheduler.step(metrics["valid"]["Loss"])
        if controller.check_termination(metrics):
            break

        # Callbacks (e.g., plotting)
        if callbacks and epoch % callbacks.frequency == 0:
            callbacks.run(model, database, epoch)

    return metric_tracker
```

**Batch construction**:
```python
class DirectoryDatabase:
    def make_batches(self, batch_size, split="train"):
        """
        Yields batches of indices for the given split.
        """
        indices = self.splits[split]  # e.g., train_idx
        n_batches = len(indices) // batch_size

        # Shuffle each epoch
        np.random.shuffle(indices)

        for i in range(n_batches):
            yield indices[i * batch_size : (i+1) * batch_size]
```

**Metric tracking**:
```python
class MetricTracker:
    def __init__(self):
        self.history = {"train": {}, "valid": {}}
        self.epoch_times = []
        self.best_metric_values = {}

    def update(self, epoch, metrics):
        for split in ["train", "valid"]:
            for key, value in metrics[split].items():
                if key not in self.history[split]:
                    self.history[split][key] = []
                self.history[split][key].append(value)

        # Track best validation metrics
        if epoch == 0 or metrics["valid"]["Loss"] < self.best_metric_values.get("Loss", float("inf")):
            self.best_metric_values = metrics["valid"]
```

---

## 8. Under the Hood: PyTorch Integration

### 8.1 Graph to Module Conversion

**How HIPPYNN converts node graphs to PyTorch**:

1. **Node graph construction** (user-facing):
   ```python
   species = inputs.SpeciesNode(db_name="Z")
   positions = inputs.PositionsNode(db_name="R")
   network = networks.HipHopnn("net", (species, positions), module_kwargs={...})
   energy = targets.HEnergyNode("sE0", network)
   loss = loss.MSELoss.of_node(energy.mol_energy)
   ```

2. **Auto-module creation** (during `__init__`):
   Each node with `auto_module_class` creates its PyTorch module:
   ```python
   class HEnergyNode(AutoKw, MultiNode):
       auto_module_class = target_modules.HEnergy

       def _auto_module(self):
           # Called automatically during __init__
           return self.auto_module_class(**self.module_kwargs)
   ```

3. **Graph traversal** (during `assemble_for_training`):
   ```python
   def get_subgraph(nodes):
       """Returns all nodes reachable from given nodes."""
       visited = set()
       stack = list(nodes)
       while stack:
           node = stack.pop()
           if node not in visited:
               visited.add(node)
               stack.extend(node.parents)
       return visited
   ```

4. **Topological sorting**:
   ```python
   def topological_sort(inputs, outputs):
       """Orders nodes so parents are executed before children."""
       subgraph = get_subgraph(outputs)
       sorted_nodes = []
       visited = set()

       def visit(node):
           if node in visited:
               return
           visited.add(node)
           for parent in node.parents:
               if parent in subgraph:
                   visit(parent)
           sorted_nodes.append(node)

       for output in outputs:
           visit(output)

       return sorted_nodes
   ```

5. **GraphModule execution**:
   ```python
   class GraphModule(torch.nn.Module):
       def forward(self, **inputs):
           cache = {}
           for node in self.sorted_nodes:
               if isinstance(node, InputNode):
                   cache[node] = inputs[node.name]
               else:
                   args = [cache[p][node.input_names[i]]
                          for i, p in enumerate(node.parents)]
                   outputs = node.torch_module(*args)
                   cache[node] = dict(zip(node.output_names, outputs))

           return tuple(cache[out][out.main_output_name] for out in self.output_nodes)
   ```

**Result**: A standard PyTorch `nn.Module` that can be trained/saved/deployed independently of HIPPYNN.

### 8.2 Automatic Differentiation

**Force computation via autograd**:

Setting `positions.requires_grad = True` enables force calculation:

```python
positions = inputs.PositionsNode(db_name="R")
positions.requires_grad = True  # Critical for forces

# During forward pass:
energy = model(species=Z, positions=R)  # R requires grad
forces = -torch.autograd.grad(energy.sum(), R, create_graph=True)[0]
# forces: (n_atoms, 3)
```

**Why `create_graph=True`?**

Needed if you want second derivatives (e.g., Hessian for vibrational frequencies) or if computing forces during training (though this script doesn't).

**Memory considerations**:

Computing forces requires storing the entire computation graph, roughly doubling memory usage. For training, we typically:
1. Train on energies only (faster, less memory)
2. Fine-tune on energies + forces (more accurate, slower)

### 8.3 Key PyTorch Operations Used

| Operation | Location | Purpose | Tensor Shape Example |
|-----------|----------|---------|----------------------|
| **torch.scatter_add** | `hippynn/layers/indexers.py:MolSummer` | Aggregate atoms to molecules | `(960,1)` → `(64,1)` via index `(960,)` |
| **custom_kernels.envsum** | `hippynn/layers/hiplayers/interactions.py:68` | Gather neighbor features | `features(960,75)` + `pairs(10000,2)` → `env(960,63,75)` |
| **torch.mm** | `hippynn/layers/hiplayers/interactions.py:75` | Matrix multiply for message passing | `(960, 63×75) @ (63×75, 75)` → `(960,75)` |
| **torch.nn.Linear** | Throughout | Learnable affine transformations | `(n, in_features)` → `(n, out_features)` |
| **torch.exp, torch.cos** | `hippynn/layers/hiplayers/sensitivity.py` | Sensitivity basis functions | `(n_pairs,)` → `(n_pairs, 63)` |
| **torch.cat** | `hippynn/networks/hiphop.py:36` | Concatenate tensor features | List `[(n,a), (n,b)]` → `(n,a+b)` |

**Custom kernel: envsum**:

HIPPYNN provides optimized implementations for neighbor aggregation:

```python
# Custom CUDA kernel (simplified):
def envsum(sensitivities, features, pair_first, pair_second):
    """
    For each atom i, sum neighbor features weighted by sensitivities.

    sensitivities: (n_pairs, n_sens)
    features: (n_atoms, n_feat)
    pair_first: (n_pairs,) atom i indices
    pair_second: (n_pairs,) atom j indices

    Returns: (n_atoms, n_sens, n_feat)
    """
    n_atoms, n_feat = features.shape
    n_pairs, n_sens = sensitivities.shape

    env = torch.zeros(n_atoms, n_sens, n_feat, device=features.device)

    for pair_idx in range(n_pairs):
        i = pair_first[pair_idx]
        j = pair_second[pair_idx]

        for k in range(n_sens):
            for f in range(n_feat):
                env[i, k, f] += sensitivities[pair_idx, k] * features[j, f]

    return env
```

Actual implementation uses parallelized CUDA for 10-100× speedup.

---

## 9. Configuration Reference

### 9.1 Complete Parameter List

#### Network Hyperparameters

```python
NETWORK_PARAMS = {
    # Species configuration
    "possible_species": [0, 1, 6, 8],  # 0=padding, 1=H, 6=C, 8=O

    # Feature dimensions
    "n_features": 75,                  # Width of hidden layers
    "n_sensitivities": 63,             # Number of radial basis functions

    # Distance parameters (Ångström)
    "dist_soft_min": 0.7,              # Start of sensitivity functions
    "dist_soft_max": 5.67638290906325, # End of sensitivity functions
    "dist_hard_max": 11.833504244685173, # Hard cutoff for pair search

    # Architecture depth
    "n_interaction_layers": 2,         # Message-passing rounds
    "n_atom_layers": 3,                # Local layers per interaction block

    # HIPHopNN-specific (tensor interactions)
    "l_max": 3,                        # Max angular momentum (0=s, 1=p, 2=d, 3=f)
    "n_max": 4,                        # Radial basis order for Bessel functions
}
```

#### Training Hyperparameters

```python
# Batch sizes
BATCH_SIZE = 64                    # Training batch size
EVAL_BATCH_SIZE = 256              # Validation batch size (larger OK, no backprop)

# Optimization
LEARNING_RATE = 5e-4               # AdamW learning rate
L2_WEIGHT = 2e-5                   # L2 regularization strength

# Loss weights
ENERGY_WEIGHT = 1.0                # Weight for energy losses (all states equal)

# Training duration
MAX_EPOCHS = 200                   # Maximum training epochs
TERMINATION_PATIENCE = 50          # Early stopping patience (epochs)

# Monitoring
PLOT_FREQUENCY = 100               # How often to generate plots

# Data
N_ATOMS = 15                       # Atoms per molecule (fixed)
N_STATES = 3                       # Number of excited states (S1-S3)
INCLUDE_GROUND_STATE = True        # Train on S0 as well
```

#### Database Splits

```python
database = DirectoryDatabase(
    name="acn_",
    directory="/path/to/data",
    seed=114514,         # Reproducibility
    test_size=0.1,       # 10% test set
    valid_size=0.1,      # 10% validation set
    # train_size = 0.8 (implicit)
)
```

### 9.2 Recommended Starting Points

#### For small organic molecules (<20 atoms):

```python
NETWORK_PARAMS = {
    "n_features": 50,
    "n_sensitivities": 40,
    "dist_soft_min": 0.7,
    "dist_soft_max": 5.0,
    "dist_hard_max": 10.0,
    "n_interaction_layers": 2,
    "n_atom_layers": 2,
    "l_max": 2,  # Reduce for faster training
    "n_max": 3,
}
BATCH_SIZE = 128  # Can use larger batches
```

#### For medium systems (20-100 atoms):

```python
NETWORK_PARAMS = {
    "n_features": 100,
    "n_sensitivities": 60,
    "dist_soft_min": 0.7,
    "dist_soft_max": 6.0,
    "dist_hard_max": 12.0,
    "n_interaction_layers": 3,
    "n_atom_layers": 3,
    "l_max": 3,
    "n_max": 4,
}
BATCH_SIZE = 32
```

#### For large biomolecules (>100 atoms):

```python
NETWORK_PARAMS = {
    "n_features": 128,
    "n_sensitivities": 80,
    "dist_soft_min": 0.7,
    "dist_soft_max": 8.0,
    "dist_hard_max": 15.0,
    "n_interaction_layers": 4,  # More layers for larger receptive field
    "n_atom_layers": 2,          # Fewer atom layers to control depth
    "l_max": 2,                  # Lower l_max to manage memory
    "n_max": 3,
}
BATCH_SIZE = 8  # Memory-limited
```

**Tuning guidelines**:
1. **`n_features`**: Increase if underfitting (train loss plateau), decrease if overfitting
2. **`n_sensitivities`**: Match to typical bond length variation in your system
3. **`dist_hard_max`**: Set to ~2× longest relevant interaction (e.g., hydrogen bonds ~3-4Å → use 8Å)
4. **`n_interaction_layers`**: Each layer adds one "hop" of information propagation (~5Å per layer)
5. **`l_max`**: Higher for systems with π-bonds, conjugation, or strong directionality

---

## 10. Output Artifacts

### 10.1 Generated Files

After training completes, the following files are created in `ENERGY_TRAINING_MODEL/`:

| File | Contents | Format |
|------|----------|--------|
| **training_log.txt** | Epoch-by-epoch metrics (train/valid loss, time) | Plain text |
| **training_summary.json** | Final hyperparameters + best metrics | JSON |
| **best_model.pt** | Saved model checkpoint (lowest validation loss) | PyTorch state_dict |
| **Sensitivity_0.pdf** | Radial basis functions for interaction layer 0 | PDF plot |
| **Sensitivity_1.pdf** | Radial basis functions for interaction layer 1 | PDF plot |
| **Energy_sE0.pdf** | Predicted vs. true energies for ground state | PDF scatter plot |
| **Energy_sE1.pdf** | Predicted vs. true energies for S1 | PDF scatter plot |
| **Energy_sE2.pdf** | Predicted vs. true energies for S2 | PDF scatter plot |
| **Energy_sE3.pdf** | Predicted vs. true energies for S3 | PDF scatter plot |

### 10.2 Interpreting Results

#### training_log.txt

```
Epoch   Train_Loss   Valid_Loss   sE0-RMSE   sE1-RMSE   sE2-RMSE   sE3-RMSE   Time(s)
0       0.523        0.487        0.145      0.162      0.180      0.205      12.3
1       0.412        0.398        0.118      0.139      0.158      0.183      11.8
...
```

**What to look for**:
- **Train vs. Valid loss divergence**: If valid >> train after convergence → overfitting (increase L2_WEIGHT or reduce n_features)
- **Individual state RMSE**: Which states are hardest to predict? May need state-specific architectures
- **Training time**: Should stabilize after a few epochs; sudden increases may indicate numerical issues

#### training_summary.json

```json
{
  "n_features": 75,
  "n_sensitivities": 63,
  ...
  "metric": {
    "valid": {
      "Loss": 0.342,
      "sE0-RMSE": 0.089,
      "sE0-MAE": 0.067,
      ...
    }
  },
  "avg_epoch_time": 11.5
}
```

**Chemical accuracy benchmark**:
- **RMSE < 1 kcal/mol (0.043 eV)**: "Chemical accuracy" for reaction energies
- **RMSE < 0.1 eV**: Good for excited state photochemistry
- **RMSE < 0.01 eV**: Excellent, approaching DFT accuracy

For this system with RMSE ~0.09 eV for S0, the model achieves near-chemical-accuracy for ground state.

#### Sensitivity plots (Sensitivity_*.pdf)

Shows the learned radial basis functions:

```
        ↑ Sensitivity value
     1.0│     ╱╲
        │    ╱  ╲     ╱╲
        │   ╱    ╲   ╱  ╲
     0.5│  ╱      ╲ ╱    ╲   ╱╲
        │ ╱        ╲      ╲ ╱  ╲
     0.0│╱──────────╲──────╲────╲──────→
         0    2    4    6    8   10  12 (Å)
```

**Interpretation**:
- **Peak locations**: Where the network finds important distance scales (e.g., bond lengths, van der Waals radii)
- **Width**: Narrow peaks → precise distance encoding, wide peaks → coarse encoding
- **Cutoff behavior**: Should smoothly go to zero at `dist_hard_max`

**Diagnostic**:
- All sensitivities nearly identical → network not using distance information (check data quality)
- Extreme spikes → numerical instability (reduce learning rate)

#### Energy plots (Energy_sE*.pdf)

Scatter plot: predicted vs. true energies

```
Predicted (eV)
     ↑
  -5 │                   ╱
     │                 ╱
     │               ╱  ●
     │             ╱ ●● ●
 -10 │           ╱ ●●●●●
     │         ╱ ●●●●●●●
     │       ╱  ●●●●●●
 -15 │     ╱   ●●●●●
     │   ╱
 -20 │ ╱────────────────→
    -20  -15  -10  -5   True (eV)
```

**Ideal**: Points hug the diagonal (perfect predictions)

**Common issues**:
- **Systematic offset**: Diagonal but shifted → check energy reference/units
- **Scatter around diagonal**: Random errors → normal, check RMSE magnitude
- **Curvature**: Non-linear relationship → may need more training or different architecture
- **Outliers**: Few points far from diagonal → examine these configurations (geometry errors? convergence issues in DFT?)

**Per-state comparison**:
- S0 (ground state) typically has lowest error
- S1-S3 errors increase with excitation energy (less training data focus, harder physics)
- If S2 error < S1 error → investigate data quality

---

## 11. Extension Guide

### 11.1 Adding More Electronic States

**Scenario**: You have S4, S5 data and want to predict them.

**Modifications**:

1. **Add .npy files**:
   ```bash
   # In data directory:
   ls acn_sE*.npy
   # Output: acn_sE0.npy ... acn_sE5.npy
   ```

2. **Update script**:
   ```python
   N_STATES = 5  # Was 3, now includes S4, S5

   # The for loop automatically creates nodes:
   for i in range(state_start, state_end):  # Now goes up to S5
       state_name = f"sE{i}"
       energy = targets.HEnergyNode(state_name, network)
       ...
   ```

3. **No other changes needed!** The loss aggregation, plotting, and training loop adapt automatically.

### 11.2 Modifying Network Architecture

#### Increasing capacity:

```python
NETWORK_PARAMS = {
    "n_features": 150,  # Was 75 → doubles capacity
    "n_interaction_layers": 3,  # Was 2 → longer-range interactions
    ...
}
```

**Expected effects**:
- **Accuracy**: May improve on complex systems
- **Training time**: ~2× slower (scales as O(n_features²))
- **Memory**: ~2× increase
- **Overfitting risk**: Higher (increase L2_WEIGHT to 5e-5)

#### Switching to HIPNN (no tensor interactions):

```python
network = build_network(
    species, positions,
    NETWORK_PARAMS,
    choice=0,  # Was 1 (HIPHopNN), now 0 (HIPNN)
)
```

**Trade-offs**:
- **Speed**: ~3× faster training
- **Accuracy**: Slightly lower (~10% higher RMSE) for systems with directional interactions
- **Memory**: Lower

#### Adding attention mechanisms:

This requires modifying HIPPYNN source or building custom layers:

```python
class AttentionInteraction(InteractLayer):
    def __init__(self, *args, n_heads=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.nf_in,
            num_heads=n_heads
        )

    def forward(self, features, ...):
        # Standard message passing
        env_features = super().forward(features, ...)

        # Add attention over neighbors
        attn_out, _ = self.attention(features, features, features)
        return env_features + attn_out
```

### 11.3 Custom Property Prediction (Atomic Charges)

**Goal**: Predict partial atomic charges alongside energies.

**Approach 1: Use built-in HChargeNode**

```python
# After creating network:
network = build_network(...)

# Add charge prediction head
charges = targets.HChargeNode("charges", network)
charges.atom_charges.db_name = "charges"  # Links to database

# Add to loss
charge_mae = loss.MAELoss.of_node(charges.atom_charges)
total_loss += CHARGE_WEIGHT * charge_mae
```

**Approach 2: Custom charge node with total charge constraint**

Ensure predicted charges sum to molecular charge:

```python
from hippynn.graphs.nodes.base import MultiNode, AutoKw

class ConstrainedChargeNode(MultiNode, AutoKw):
    input_names = "features", "total_charge", "system_index", "n_systems"
    output_names = "atom_charges",

    def __init__(self, name, network, total_charge_node):
        """
        total_charge_node: InputNode with db_name="total_charge" (molecular charge)
        """
        parents = (network, total_charge_node, ...)
        super().__init__(name, parents)

    class Module(torch.nn.Module):
        def forward(self, features, total_charge, system_index, n_systems):
            # Predict unconstrained charges
            raw_charges = self.predictor(features[-1])  # (n_atoms, 1)

            # Sum per molecule
            mol_charges = scatter_add(raw_charges, system_index, n_systems)

            # Compute correction to match total_charge
            correction = (total_charge - mol_charges) / n_atoms_per_molecule

            # Apply correction
            constrained_charges = raw_charges + correction[system_index]
            return constrained_charges
```

**PyTorch-only implementation** (for hippynn-independent deployment):

```python
class ChargePredictor(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.charge_layer = torch.nn.Linear(n_features, 1)

    def forward(self, features, system_index, n_systems, total_charge):
        """
        features: (n_atoms, n_features) from network
        system_index: (n_atoms,) molecule ID
        n_systems: int, number of molecules
        total_charge: (n_systems, 1) target total charge
        """
        # Predict raw charges
        atom_charges = self.charge_layer(features)  # (n_atoms, 1)

        # Sum to molecular total
        mol_charges = torch.zeros(n_systems, 1, device=features.device)
        mol_charges.scatter_add_(0, system_index.unsqueeze(1), atom_charges)

        # Count atoms per molecule
        n_atoms = torch.bincount(system_index, minlength=n_systems).unsqueeze(1).float()

        # Correction per atom
        correction = (total_charge - mol_charges) / n_atoms
        atom_charges_corrected = atom_charges + correction[system_index]

        return atom_charges_corrected, mol_charges
```

**Roadmap for charge prediction**:
1. Generate charge data (e.g., from Mulliken analysis, RESP fitting)
2. Save as `acn_charges.npy` with shape `(25000, 15)`
3. Implement `ConstrainedChargeNode` or use `HChargeNode`
4. Add to loss function
5. Train multi-task model (energies + charges)

**Benefits of multi-task learning**:
- Charges provide additional supervision for atomic features
- Improved energy predictions (charges and energies are coupled)
- Enables electrostatics-based force fields

---

## Appendices

### Appendix A: Complete Function Call Trace

```
main script
│
├── build_network(species, positions, NETWORK_PARAMS, choice=1, l_max=3, n_max=4)
│   └── networks.HipHopnn.__init__(...)
│       ├── networks.Hipnn.__init__(...)  [Parent class]
│       │   ├── Create sensitivity_layers: InverseSensitivityModule × n_interaction_layers
│       │   ├── Create interaction_layers: HOPInteractionLayer × n_interaction_layers
│       │   │   └── layers.hiplayers.interactions.HOPInteractionLayer.__init__(...)
│       │   │       ├── Create self.int_weights: (n_sensitivities, n_features_out, n_features_in)
│       │   │       └── Create self.selfint: Linear(n_features_in, n_features_out)
│       │   └── Create atom_layers: Linear × (n_interaction_layers × n_atom_layers)
│       │       └── ResNetWrapper(Linear, ...) if resnet=True
│       └── Create self.tensor_extractor: TensorExtractor(l_max=3)
│
├── targets.HEnergyNode.__init__("sE0", network, module_kwargs=None)
│   ├── parent_expander.match(Network) → expansion0(network)
│   │   ├── find_unique_relative(network, AtomIndexer) → pdindexer
│   │   └── Return (network, pdindexer.system_index, pdindexer.n_systems)
│   └── _auto_module() → layers.targets.HEnergy(feature_sizes=network.torch_module.feature_sizes)
│       ├── Create self.layers: Linear(feature_size, 1) × n_feature_blocks
│       └── Create self.summer: indexers.MolSummer()
│
├── loss.MSELoss.of_node(energy.mol_energy)
│   ├── Create LossPredNode("sE0_pred", energy.mol_energy)
│   ├── Create LossTrueNode("sE0_true", db_name="sE0")
│   └── Create MSELoss("sE0_mse", (pred, true))
│       └── forward(pred, true) → torch.mean((pred - true) ** 2)
│
├── loss.MAELoss.of_node(energy.mol_energy)
│   └── [Similar to MSELoss]
│
├── loss.l2reg(network)
│   └── Create L2Regularization module
│       └── forward() → sum(p**2 for p in network.regularization_params())
│
├── experiment.assemble_for_training(total_loss, validation_losses, plot_maker=plotter)
│   ├── determine_out_in_targ(total_loss, *validation_losses.values())
│   │   ├── get_subgraph([total_loss, ...]) → all_nodes_needed
│   │   ├── Find InputNode instances → inputs = [species, positions]
│   │   ├── Find LossPredNode instances → outputs = [mol_energy_sE0, ...]
│   │   └── Find LossTrueNode instances → targets = [sE0_true, ...]
│   ├── generate_database_info(inputs, targets)
│   │   └── Return {"inputs": ["Z", "R"], "targets": ["sE0", "sE1", ...]}
│   ├── Create model = GraphModule(inputs, outputs)
│   │   ├── topological_sort(inputs, outputs) → sorted_nodes
│   │   └── Store execution order
│   ├── Create loss_module = GraphModule([outputs, targets], [total_loss])
│   ├── Create evaluator = Evaluator(model, valid_loss_module, validation_names, ...)
│   └── Return TrainingModules(model, loss_module, evaluator), db_info
│
├── databases.DirectoryDatabase(name="acn_", directory="...", **db_info)
│   ├── Discover files: acn_Z.npy, acn_R.npy, acn_sE0.npy, ...
│   ├── Load with np.load() and convert to torch tensors
│   ├── Generate train/valid/test splits using seed
│   └── Build arr_dict: {"Z": tensor, "R": tensor, "sE0": tensor, ...}
│
├── experiment.setup_training(training_modules, setup_params=experiment_params)
│   ├── Create optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
│   ├── Create scheduler = RaiseBatchSizeOnPlateau(optimizer, ...)
│   ├── Create controller = PatienceController(optimizer, scheduler, ...)
│   ├── Create metric_tracker = MetricTracker()
│   └── Return training_modules, controller, metric_tracker
│
└── experiment.train_model(training_modules, database, controller, metric_tracker)
    ├── For epoch in range(MAX_EPOCHS):
    │   ├── For batch_indices in database.make_batches(BATCH_SIZE, "train"):
    │   │   ├── batch = database.get_batch(batch_indices)
    │   │   │   └── Return {name: arr_dict[name][batch_indices] for name in inputs+targets}
    │   │   ├── predictions = model(species=batch["Z"], positions=batch["R"])
    │   │   │   └── GraphModule.forward()
    │   │   │       ├── Execute nodes in topological order:
    │   │   │       │   ├── SpeciesNode: batch["Z"]
    │   │   │       │   ├── PositionsNode: batch["R"]
    │   │   │       │   ├── Network.forward(species, positions)
    │   │   │       │   │   └── HipHopnn.forward()
    │   │   │       │   │       ├── Compute pair distances and directions
    │   │   │       │   │       ├── For block in self.blocks:
    │   │   │       │   │       │   ├── sensitivity = sensitivity_layer(pair_dist)
    │   │   │       │   │       │   ├── tensor_terms = tensor_extractor(pair_coord/pair_dist)
    │   │   │       │   │       │   ├── features = interaction_layer(features, ..., tensor_terms)
    │   │   │       │   │       │   └── For atom_layer in block[1:]:
    │   │   │       │   │       │       └── features = activation(atom_layer(features))
    │   │   │       │   │       └── Return [features₀, features₁, ...]
    │   │   │       │   └── HEnergyNode.forward(features, system_index, n_systems)
    │   │   │       │       └── HEnergy.forward()
    │   │   │       │           ├── partial_energies = [layer(feat) for layer, feat in zip(self.layers, features)]
    │   │   │       │           ├── total_atom_energy = sum(partial_energies)
    │   │   │       │           └── total_mol_energy = self.summer(total_atom_energy, system_index, n_systems)
    │   │   │       └── Return [mol_energy_sE0, mol_energy_sE1, ...]
    │   │   ├── loss = loss_module(predictions=predictions, targets=batch["sE0", "sE1", ...])
    │   │   │   └── Compute total_loss = Σ(RMSE + MAE) per state + L2_reg
    │   │   ├── optimizer.zero_grad()
    │   │   ├── loss.backward()
    │   │   └── optimizer.step()
    │   ├── valid_metrics = evaluator.evaluate(database, "valid", EVAL_BATCH_SIZE)
    │   ├── metric_tracker.update(epoch, {"train": ..., "valid": valid_metrics})
    │   ├── controller.scheduler.step(valid_metrics["Loss"])
    │   └── If controller.check_termination(valid_metrics): break
    └── Return metric_tracker
```

### Appendix B: Glossary

**For ML Engineers**:
- **Atom indexer**: Node that tracks which atoms belong to which molecules in a batch
- **GraphNode**: User-facing abstraction representing a computation step with automatic parent tracking
- **Message passing**: Iterative information aggregation along graph edges (atom-atom pairs)
- **Scatter-add**: PyTorch operation that sums values into bins specified by indices
- **System**: HIPPYNN term for "molecule" or "configuration"

**For Computational Chemists**:
- **Born-Oppenheimer approximation**: Separation of electronic and nuclear motion
- **Excited state**: Electronic configuration with energy above ground state (S₁, S₂, ...)
- **Force**: Negative gradient of energy with respect to nuclear positions: F = -∇ᴿE
- **Potential energy surface (PES)**: Function E(R) mapping molecular geometry to energy
- **Radial basis functions**: Set of functions encoding distance information (like Gaussian orbitals)

**Shared**:
- **Hierarchical energy**: Summing contributions from multiple network layers
- **Interaction layer**: Message-passing layer that aggregates neighbor information
- **L₂ regularization**: Penalty term ||θ||² added to loss to prevent overfitting
- **Sensitivity functions**: Smooth radial basis functions for distance encoding
- **Spherical harmonics**: Angular basis functions (like atomic orbitals: s, p, d, f)

### Appendix C: References

**HIPPYNN Framework**:
- HIPPYNN Documentation: https://hippynn.readthedocs.io/
- GitHub Repository: https://github.com/lanl/hippynn
- Preprint: "HIPPYNN: A High-Performance Python Library for Machine Learning Potentials" (2021)

**Original HIPNN Paper**:
- Lubbers, N., Smith, J. S., & Barros, K. (2018). "Hierarchical modeling of molecular energies using a deep neural network". *The Journal of Chemical Physics*, 148(24), 241715.
- DOI: 10.1063/1.5011181

**HIP-HOP-NN Extension**:
- Smith, J. S., Nebgen, B., Mathew, N., et al. (2021). "Automated discovery of a robust interatomic potential for aluminum". *Nature Communications*, 12, 1257.
- DOI: 10.1038/s41467-021-21376-0

**Related Methods**:
- **SchNet**: Schütt et al. (2017). "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions". *NeurIPS*.
- **PaiNN**: Schütt et al. (2021). "Equivariant message passing for the prediction of tensorial properties and molecular spectra". *ICML*.
- **NequIP**: Batzner et al. (2022). "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials". *Nature Communications*, 13, 2453.
- **MACE**: Batatia et al. (2022). "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields". *NeurIPS*.

**Excited State Potentials**:
- Westermayr, J., & Marquetand, P. (2021). "Machine learning for electronically excited states of molecules". *Chemical Reviews*, 121(16), 9873-9926.
- Dral, P. O. (2020). "Quantum chemistry in the age of machine learning". *The Journal of Physical Chemistry Letters*, 11(6), 2336-2347.

**General ML for Chemistry**:
- Unke, O. T., et al. (2021). "Machine learning force fields". *Chemical Reviews*, 121(16), 10142-10186.
- Keith, J. A., et al. (2021). "Combining machine learning and computational chemistry for predictive insights into chemical systems". *Chemical Reviews*, 121(16), 9816-9872.

---

## Conclusion

This document provides a complete architecture and implementation analysis of a HIPPYNN-based molecular energy prediction system. Key takeaways:

1. **HIPPYNN's strength**: Separates high-level graph construction from low-level tensor operations, enabling rapid prototyping
2. **HIP-HOP-NN architecture**: Combines radial and angular information via spherical tensor decomposition for high accuracy
3. **Multi-state prediction**: Efficient parameter sharing across electronic states through shared network backbone
4. **Production readiness**: The final model is a standard PyTorch `nn.Module` deployable independently of HIPPYNN

**Next steps for development**:
- Implement pure PyTorch versions of key components for deployment
- Extend to forces (enable molecular dynamics)
- Add custom nodes for properties like dipole moments, polarizabilities
- Explore transfer learning: pretrain on large dataset, fine-tune on target system

For questions or contributions, refer to the HIPPYNN documentation and GitHub issues.

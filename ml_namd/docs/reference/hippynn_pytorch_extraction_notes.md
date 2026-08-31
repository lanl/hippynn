# HIPPYNN-to-PyTorch Extraction Guide

## Purpose

This document provides a roadmap for implementing HIPPYNN's energy prediction capabilities in pure PyTorch, without dependencies on the HIPPYNN framework. This is essential for:

1. **Deployment**: Production environments where HIPPYNN may not be available
2. **Custom modifications**: Building novel architectures inspired by HIPPYNN
3. **Performance optimization**: Hand-tuning operations for specific hardware
4. **Understanding**: Demystifying what HIPPYNN does "under the hood"

---

## Core Components to Extract

### 1. Minimal Energy Prediction Module

The absolute minimum needed to run inference on a trained model:

```python
import torch
import torch.nn as nn

class MinimalHipnnEnergy(nn.Module):
    """
    Standalone energy predictor extracted from HIPPYNN HEnergyNode.
    Requires pre-computed atomic features from network.
    """

    def __init__(self, feature_sizes, n_target=1):
        """
        Args:
            feature_sizes: List of feature dimensions [n_feat_0, n_feat_1, ...]
                           e.g., [3, 75, 75] for one-hot + 2 interaction layers
            n_target: Number of energy outputs (1 for single state)
        """
        super().__init__()
        self.feature_sizes = feature_sizes
        self.n_target = n_target

        # Create linear layers for each feature block
        # First layer has no bias (embedding layer is just species identity)
        biases = [False] + [True] * (len(feature_sizes) - 1)
        self.layers = nn.ModuleList([
            nn.Linear(nf, n_target, bias=bias)
            for nf, bias in zip(feature_sizes, biases)
        ])

    def forward(self, all_features, system_index, n_systems):
        """
        Args:
            all_features: List of feature tensors from each network layer
                          [(n_atoms, feat_0), (n_atoms, feat_1), ...]
            system_index: (n_atoms,) tensor, molecule ID for each atom
            n_systems: Number of molecules in batch

        Returns:
            mol_energy: (n_systems, n_target) total molecular energies
            atom_energy: (n_atoms, n_target) per-atom energy contributions
        """
        # Apply each linear layer to corresponding features
        partial_energies = [
            layer(features)
            for layer, features in zip(self.layers, all_features)
        ]

        # Sum all contributions to get total per-atom energy
        total_atom_energy = sum(partial_energies)  # (n_atoms, n_target)

        # Aggregate atoms to molecules using scatter_add
        mol_energy = torch.zeros(
            (n_systems, self.n_target),
            dtype=total_atom_energy.dtype,
            device=total_atom_energy.device
        )
        mol_energy.scatter_add_(
            0,
            system_index.unsqueeze(1).expand_as(total_atom_energy),
            total_atom_energy
        )

        return mol_energy, total_atom_energy
```

**Usage example**:
```python
# Load HIPPYNN-trained model
henergy_hippynn = torch.load("best_model.pt")

# Extract feature_sizes from HIPPYNN model
feature_sizes = henergy_hippynn.feature_sizes  # e.g., [3, 75, 75]

# Create standalone module
henergy_pytorch = MinimalHipnnEnergy(feature_sizes, n_target=1)

# Copy weights
for i, (layer_hippynn, layer_pytorch) in enumerate(
    zip(henergy_hippynn.layers, henergy_pytorch.layers)
):
    layer_pytorch.weight.data = layer_hippynn.weight.data.clone()
    if layer_pytorch.bias is not None:
        layer_pytorch.bias.data = layer_hippynn.bias.data.clone()

# Now use for inference
with torch.no_grad():
    mol_energy, atom_energy = henergy_pytorch(features, system_index, n_systems)
```

---

### 2. Message-Passing Network Core

The interaction layer is the heart of HIPNN. Here's a pure PyTorch implementation:

```python
class InteractionLayer(nn.Module):
    """
    HIPNN interaction layer: message passing with radial basis functions.

    Equation: Z'_i = Σ_j V(r_ij) ⊙ Z_j + W Z_i + b
    """

    def __init__(self, n_features_in, n_features_out, n_sensitivities):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.n_sens = n_sensitivities

        # Interaction weights: (n_sens, n_out, n_in)
        self.int_weights = nn.Parameter(
            torch.randn(n_sensitivities, n_features_out, n_features_in)
        )
        nn.init.xavier_normal_(self.int_weights)

        # Self-interaction: W Z_i + b
        self.self_layer = nn.Linear(n_features_in, n_features_out)
        nn.init.xavier_normal_(self.self_layer.weight)

    def forward(self, features, sensitivity_values, pair_first, pair_second):
        """
        Args:
            features: (n_atoms, n_features_in) atomic features
            sensitivity_values: (n_pairs, n_sensitivities) radial basis values
            pair_first: (n_pairs,) atom i indices
            pair_second: (n_pairs,) atom j indices

        Returns:
            features_out: (n_atoms, n_features_out) updated features
        """
        n_atoms = features.shape[0]

        # Compute environment: E[i,k,f] = Σ_j S[ij,k] * Z[j,f]
        # This is the expensive operation; in HIPPYNN uses custom CUDA kernel
        env_features = torch.zeros(
            (n_atoms, self.n_sens, self.n_features_in),
            dtype=features.dtype,
            device=features.device
        )

        # Aggregate neighbor features weighted by sensitivities
        for pair_idx in range(len(pair_first)):
            i = pair_first[pair_idx]
            j = pair_second[pair_idx]
            # env[i, :, :] += sensitivity[pair_idx, :].unsqueeze(1) * features[j, :].unsqueeze(0)
            env_features[i] += sensitivity_values[pair_idx].unsqueeze(1) * features[j].unsqueeze(0)

        # Reshape for matrix multiplication: (n_atoms, n_sens * n_in)
        env_flat = env_features.reshape(n_atoms, self.n_sens * self.n_features_in)

        # Reshape weights: (n_sens * n_in, n_out)
        weights_flat = self.int_weights.permute(0, 2, 1).reshape(
            self.n_sens * self.n_features_in, self.n_features_out
        )

        # Apply interaction weights
        interaction_out = torch.mm(env_flat, weights_flat)  # (n_atoms, n_out)

        # Add self-interaction
        self_out = self.self_layer(features)  # (n_atoms, n_out)

        return interaction_out + self_out


# Optimized version using PyTorch scatter operations:
class InteractionLayerOptimized(InteractionLayer):
    def forward(self, features, sensitivity_values, pair_first, pair_second):
        n_atoms = features.shape[0]
        n_pairs = len(pair_first)

        # Expand neighbor features: (n_pairs, n_sens, n_in)
        neighbor_features = features[pair_second]  # (n_pairs, n_in)
        weighted = (
            sensitivity_values.unsqueeze(2) *  # (n_pairs, n_sens, 1)
            neighbor_features.unsqueeze(1)      # (n_pairs, 1, n_in)
        )  # → (n_pairs, n_sens, n_in)

        # Sum over neighbors using scatter_add
        env_features = torch.zeros(
            (n_atoms, self.n_sens, self.n_features_in),
            dtype=features.dtype,
            device=features.device
        )

        # Expand pair_first to match weighted shape
        indices = pair_first.view(-1, 1, 1).expand_as(weighted)
        env_features.scatter_add_(0, indices, weighted)

        # Rest is same as base class
        env_flat = env_features.reshape(n_atoms, -1)
        weights_flat = self.int_weights.permute(0, 2, 1).reshape(-1, self.n_features_out)
        interaction_out = torch.mm(env_flat, weights_flat)
        self_out = self.self_layer(features)

        return interaction_out + self_out
```

---

### 3. Sensitivity (Radial Basis) Functions

```python
class InverseSensitivity(nn.Module):
    """
    Inverse-type radial basis functions used in HIPNN.

    s_k(r) = 1 / (1 + exp((r - μ_k) / σ)) × f_cutoff(r)
    """

    def __init__(self, n_sensitivities, dist_soft_min, dist_soft_max, dist_hard_max):
        super().__init__()
        self.n_sens = n_sensitivities
        self.dist_hard_max = dist_hard_max

        # Centers evenly spaced from soft_min to soft_max
        centers = torch.linspace(dist_soft_min, dist_soft_max, n_sensitivities)
        self.register_buffer("centers", centers)

        # Sharpness parameter (inverse of width)
        self.sharpness = 4.0 * n_sensitivities / (dist_soft_max - dist_soft_min)

    def forward(self, distances):
        """
        Args:
            distances: (n_pairs,) pairwise distances

        Returns:
            sensitivities: (n_pairs, n_sensitivities) basis function values
        """
        import math

        # Expand for broadcasting: (n_pairs, 1) vs (n_sens,)
        dists = distances.unsqueeze(1)  # (n_pairs, 1)

        # Inverse sensitivity: 1 / (1 + exp(sharpness * (r - center)))
        sense = 1.0 / (1.0 + torch.exp(self.sharpness * (dists - self.centers)))

        # Cosine cutoff: 0.5 * (cos(π r / r_cut) + 1)
        cutoff = 0.5 * (torch.cos(math.pi * dists / self.dist_hard_max) + 1.0)
        cutoff = torch.where(
            dists < self.dist_hard_max,
            cutoff,
            torch.zeros_like(cutoff)
        )

        return sense * cutoff  # (n_pairs, n_sens)
```

---

### 4. Complete Minimal Network

Putting it all together:

```python
class MinimalHIPNN(nn.Module):
    """
    Minimal HIPNN network for energy prediction.
    No species encoding (assumes one-hot input), no ResNet, single interaction layer.
    """

    def __init__(
        self,
        n_species,
        n_features,
        n_sensitivities,
        dist_soft_min,
        dist_soft_max,
        dist_hard_max
    ):
        super().__init__()

        self.n_species = n_species
        self.n_features = n_features

        # Sensitivity module
        self.sensitivity = InverseSensitivity(
            n_sensitivities, dist_soft_min, dist_soft_max, dist_hard_max
        )

        # Single interaction layer
        self.interaction = InteractionLayerOptimized(
            n_species, n_features, n_sensitivities
        )

        # Activation
        self.activation = nn.Softplus()

        # Energy prediction
        self.energy_predictor = MinimalHipnnEnergy(
            feature_sizes=[n_species, n_features],
            n_target=1
        )

    def forward(self, species_onehot, positions, pair_first, pair_second):
        """
        Args:
            species_onehot: (n_atoms, n_species) one-hot encoded species
            positions: (n_atoms, 3) atomic coordinates
            pair_first: (n_pairs,) atom i indices
            pair_second: (n_pairs,) atom j indices

        Returns:
            mol_energy: (n_molecules,) predicted energies
        """
        # Compute pairwise distances
        pair_vectors = positions[pair_second] - positions[pair_first]  # (n_pairs, 3)
        pair_distances = torch.norm(pair_vectors, dim=1)  # (n_pairs,)

        # Compute sensitivity values
        sensitivities = self.sensitivity(pair_distances)  # (n_pairs, n_sens)

        # Initial features from species
        features_0 = species_onehot.float()  # (n_atoms, n_species)

        # Message passing
        features_1 = self.interaction(
            features_0, sensitivities, pair_first, pair_second
        )
        features_1 = self.activation(features_1)  # (n_atoms, n_features)

        # Predict energy
        # Need system_index and n_systems (not shown here, typically computed from batch)
        system_index = torch.zeros(features_0.shape[0], dtype=torch.long)  # Placeholder
        n_systems = 1  # Placeholder

        mol_energy, atom_energy = self.energy_predictor(
            [features_0, features_1],
            system_index,
            n_systems
        )

        return mol_energy.squeeze()
```

---

### 5. Custom Charge Node Template

Building on the minimal network to add charge prediction:

```python
class ChargePredictor(nn.Module):
    """
    Predict atomic partial charges with total charge constraint.
    """

    def __init__(self, n_features, charge_activation="tanh"):
        super().__init__()

        # Charge prediction layer
        self.charge_layer = nn.Linear(n_features, 1)

        # Optional: constrain charges to reasonable range
        if charge_activation == "tanh":
            self.activation = nn.Tanh()  # Output in [-1, 1]
            self.scale = 2.0  # Allow up to ±2 e
        elif charge_activation is None:
            self.activation = nn.Identity()
            self.scale = 1.0
        else:
            raise ValueError(f"Unknown activation: {charge_activation}")

    def forward(self, features, system_index, n_systems, total_charge=None):
        """
        Args:
            features: (n_atoms, n_features) final atomic features
            system_index: (n_atoms,) molecule ID per atom
            n_systems: Number of molecules
            total_charge: (n_systems,) target total charge per molecule (optional)

        Returns:
            atom_charges: (n_atoms, 1) partial charges
            mol_charges: (n_systems, 1) total charges per molecule
        """
        # Predict raw charges
        raw_charges = self.charge_layer(features)  # (n_atoms, 1)
        raw_charges = self.activation(raw_charges) * self.scale

        # Sum to molecular total
        mol_charges = torch.zeros(
            (n_systems, 1),
            dtype=raw_charges.dtype,
            device=raw_charges.device
        )
        mol_charges.scatter_add_(
            0,
            system_index.unsqueeze(1).expand_as(raw_charges),
            raw_charges
        )

        # If total charge constraint is provided, enforce it
        if total_charge is not None:
            # Count atoms per molecule
            atom_counts = torch.bincount(system_index, minlength=n_systems).float().unsqueeze(1)

            # Compute per-atom correction
            correction = (total_charge.unsqueeze(1) - mol_charges) / atom_counts

            # Apply correction
            atom_charges = raw_charges + correction[system_index]

            # Recompute molecular charges (should now match total_charge)
            mol_charges.zero_()
            mol_charges.scatter_add_(0, system_index.unsqueeze(1).expand_as(atom_charges), atom_charges)
        else:
            atom_charges = raw_charges

        return atom_charges, mol_charges


class HIPNNWithCharges(MinimalHIPNN):
    """
    Extend minimal HIPNN to predict both energies and charges.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add charge predictor
        self.charge_predictor = ChargePredictor(
            self.n_features,
            charge_activation="tanh"
        )

    def forward(self, species_onehot, positions, pair_first, pair_second, total_charge=None):
        # ... (same feature extraction as MinimalHIPNN)

        features_0 = species_onehot.float()
        sensitivities = self.sensitivity(
            torch.norm(positions[pair_second] - positions[pair_first], dim=1)
        )
        features_1 = self.activation(
            self.interaction(features_0, sensitivities, pair_first, pair_second)
        )

        # Predict energy
        system_index = torch.zeros(features_0.shape[0], dtype=torch.long)
        n_systems = 1
        mol_energy, _ = self.energy_predictor([features_0, features_1], system_index, n_systems)

        # Predict charges
        atom_charges, mol_charges = self.charge_predictor(
            features_1,  # Use final features
            system_index,
            n_systems,
            total_charge
        )

        return {
            "energy": mol_energy.squeeze(),
            "atom_charges": atom_charges.squeeze(),
            "mol_charge": mol_charges.squeeze()
        }
```

---

## Weight Transfer from HIPPYNN

### Extracting Trained Weights

```python
def extract_hippynn_weights(hippynn_model_path, output_dir="extracted_weights"):
    """
    Extract weights from trained HIPPYNN model for PyTorch reimplementation.
    """
    import os
    import torch

    # Load HIPPYNN model
    checkpoint = torch.load(hippynn_model_path)

    os.makedirs(output_dir, exist_ok=True)

    # Network weights
    network_state = checkpoint["training_modules"]["model"]["network"]

    # Extract interaction layer weights
    for i, block in enumerate(network_state["blocks"]):
        int_layer = block[0]  # First in block is interaction layer

        torch.save({
            "int_weights": int_layer["int_weights"],
            "self_weight": int_layer["selfint.weight"],
            "self_bias": int_layer["selfint.bias"]
        }, os.path.join(output_dir, f"interaction_{i}.pt"))

        # Extract atom layers
        for j, atom_layer in enumerate(block[1:]):
            torch.save({
                "weight": atom_layer["weight"],
                "bias": atom_layer["bias"]
            }, os.path.join(output_dir, f"atom_{i}_{j}.pt"))

    # Extract sensitivity parameters
    for i, sens_layer in enumerate(network_state["sensitivity_layers"]):
        torch.save({
            "centers": sens_layer["centers"],
            # sharpness is computed, not stored
        }, os.path.join(output_dir, f"sensitivity_{i}.pt"))

    # Extract energy prediction weights
    energy_state = checkpoint["training_modules"]["model"]["henergy"]
    for i, layer in enumerate(energy_state["layers"]):
        torch.save({
            "weight": layer["weight"],
            "bias": layer.get("bias", None)
        }, os.path.join(output_dir, f"energy_layer_{i}.pt"))

    print(f"Weights extracted to {output_dir}/")


def load_into_pytorch(pytorch_model, weight_dir="extracted_weights"):
    """
    Load extracted weights into pure PyTorch model.
    """
    import os

    # Load interaction layers
    for i, interaction in enumerate(pytorch_model.interactions):
        weights = torch.load(os.path.join(weight_dir, f"interaction_{i}.pt"))
        interaction.int_weights.data = weights["int_weights"]
        interaction.self_layer.weight.data = weights["self_weight"]
        interaction.self_layer.bias.data = weights["self_bias"]

    # Load energy predictor
    for i, layer in enumerate(pytorch_model.energy_predictor.layers):
        weights = torch.load(os.path.join(weight_dir, f"energy_layer_{i}.pt"))
        layer.weight.data = weights["weight"]
        if layer.bias is not None and weights["bias"] is not None:
            layer.bias.data = weights["bias"]

    print("Weights loaded successfully!")
```

---

## Performance Optimization Notes

### 1. Neighbor List Construction

HIPPYNN computes pair lists on-the-fly. For production, pre-compute and cache:

```python
def build_neighbor_list(positions, cutoff, cell=None):
    """
    Efficient neighbor list using spatial hashing or cell lists.
    For periodic systems, use ASE or matscipy.
    """
    from scipy.spatial import cKDTree

    n_atoms = positions.shape[0]
    tree = cKDTree(positions)

    pairs = tree.query_pairs(cutoff, output_type="ndarray")

    pair_first = pairs[:, 0]
    pair_second = pairs[:, 1]

    # Make symmetric: add (j, i) for each (i, j)
    pair_first = torch.cat([
        torch.from_numpy(pair_first),
        torch.from_numpy(pair_second)
    ])
    pair_second = torch.cat([
        torch.from_numpy(pair_second),
        torch.from_numpy(pair_first)
    ])

    return pair_first, pair_second
```

### 2. Batch Processing

For inference on many molecules, batch them by padding to max size:

```python
def batch_molecules(molecule_list, max_atoms=None):
    """
    Pad molecules to same size for batch processing.
    """
    if max_atoms is None:
        max_atoms = max(mol["positions"].shape[0] for mol in molecule_list)

    batch_size = len(molecule_list)

    # Padded arrays
    species_batch = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    positions_batch = torch.zeros((batch_size, max_atoms, 3))
    mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)

    for i, mol in enumerate(molecule_list):
        n_atoms = mol["positions"].shape[0]
        species_batch[i, :n_atoms] = mol["species"]
        positions_batch[i, :n_atoms] = mol["positions"]
        mask[i, :n_atoms] = True

    return species_batch, positions_batch, mask
```

### 3. JIT Compilation

Use TorchScript for 2-10× speedup:

```python
# After training, trace the model
example_input = (species_onehot, positions, pair_first, pair_second)
traced_model = torch.jit.trace(pytorch_model, example_input)

# Save traced model
torch.jit.save(traced_model, "model_traced.pt")

# Load and use
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(*example_input)
```

---

## Testing Correctness

Always verify that extracted model matches HIPPYNN:

```python
def test_parity(hippynn_model, pytorch_model, test_data):
    """
    Verify that PyTorch extraction produces identical results.
    """
    hippynn_model.eval()
    pytorch_model.eval()

    with torch.no_grad():
        for batch in test_data:
            # Run through HIPPYNN
            hippynn_output = hippynn_model(**batch)

            # Run through PyTorch
            pytorch_output = pytorch_model(**batch)

            # Compare
            energy_diff = torch.abs(hippynn_output["energy"] - pytorch_output["energy"])
            max_diff = energy_diff.max().item()
            mean_diff = energy_diff.mean().item()

            print(f"Max energy difference: {max_diff:.2e} eV")
            print(f"Mean energy difference: {mean_diff:.2e} eV")

            assert max_diff < 1e-5, "Outputs differ! Check weight transfer."

    print("✓ Parity test passed!")
```

---

## Deployment Example

```python
# Complete inference pipeline
class ProductionEnergyPredictor:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_energy(self, species, positions):
        """
        Args:
            species: (n_atoms,) numpy array of atomic numbers
            positions: (n_atoms, 3) numpy array of coordinates in Angstrom

        Returns:
            energy: float, predicted energy in eV
        """
        # Convert to tensors
        species_tensor = torch.from_numpy(species).long().to(self.device)
        positions_tensor = torch.from_numpy(positions).float().to(self.device)

        # Build neighbor list
        pair_first, pair_second = build_neighbor_list(
            positions, cutoff=self.model.dist_hard_max
        )

        # One-hot encode species
        species_onehot = F.one_hot(
            species_tensor, num_classes=self.model.n_species
        ).float()

        # Predict
        energy = self.model(species_onehot, positions_tensor, pair_first, pair_second)

        return energy.cpu().item()

    def predict_forces(self, species, positions):
        """
        Compute forces as negative energy gradients.
        """
        positions_tensor = torch.from_numpy(positions).float().to(self.device)
        positions_tensor.requires_grad = True

        # ... (same as predict_energy but with grad enabled)
        energy = self.model(...)

        # Compute gradient
        forces = -torch.autograd.grad(energy, positions_tensor)[0]

        return forces.cpu().numpy()
```

---

## Summary

**Essential extractions for production**:
1. `MinimalHipnnEnergy`: Energy prediction from features (5 lines core logic)
2. `InteractionLayerOptimized`: Message passing (20 lines core logic)
3. `InverseSensitivity`: Radial basis functions (10 lines core logic)

**Total**: ~50 lines of core PyTorch code replaces HIPPYNN framework for inference.

**Next steps**:
1. Extract weights from trained HIPPYNN model using `torch.save(model.state_dict())`
2. Implement minimal modules above
3. Load weights and verify parity with HIPPYNN
4. Optimize with JIT compilation
5. Deploy in production environment

This extraction maintains full compatibility with HIPPYNN-trained models while eliminating the framework dependency.

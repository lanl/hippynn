"""
Layers for physical operations
"""

import torch
from torch import Tensor



class Gradient(torch.nn.Module):
    def __init__(self, sign):
        super().__init__()
        assert sign in (-1, 1), "Sign of gradient must be +1 (gradient) or -1 (force)"
        self.sign = sign

    def forward(self, molecular_energies, positions):
        return self.sign * torch.autograd.grad(molecular_energies.sum(), positions, create_graph=True)[0]
        
class MultiGradient(torch.nn.Module):
    def __init__(self, signs):
        super().__init__()
        if isinstance(signs, int):
            signs = (signs,)
        for sign in signs:
            assert sign in (-1,1), "Sign of gradient must be -1 or +1"
        self.signs = signs

    def forward(self, molecular_energies: Tensor, *generalized_coordinates: Tensor):
        if isinstance(generalized_coordinates, Tensor):
            generalized_coordinates = (generalized_coordinates,)
        assert len(generalized_coordinates) == len(self.signs), f"Number of items to take derivative w.r.t ({len(generalized_coordinates)}) must match number of provided signs ({len(self.signs)})."
        grads = torch.autograd.grad(molecular_energies.sum(), generalized_coordinates, create_graph=True)
        return tuple((sign * grad for sign, grad in zip(self.signs, grads)))


class Hessian(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source, positions, padding_mask):
        """
        Computes the Hessian using second derivatives of energy or first derivatives of forces.
        Assumes:
            - source: either energy (B, 1) or forces (B, N_max, 3)
            - positions: (B, N_max, 3)
        Returns:
            - hessians: (B, 3N_max, 3N_max)
        """
        B, N, D = positions.shape

        hessian_mask = self.expand_padding_mask_to_hessian_mask(padding_mask)

        if source.ndim == 2 and source.shape[1] == 1:
            # Case: source is energy (B, 1)
            forces = self._forces_from_energy(source, positions)
            return self._hessian_from_forces(forces, positions), hessian_mask
        elif source.ndim == 3 and source.shape[2] == 3:
            # Case: source is forces (B, N, 3)
            return self._hessian_from_forces(source, positions), hessian_mask
        else:
            raise ValueError(f"Unsupported source shape: {source.shape}")

    def _forces_from_energy(self, energy, positions):
        return -torch.autograd.grad(energy.sum(), positions, create_graph=True)[0]

    def _hessian_from_forces(self, force, positions):
        force_flat = force.flatten(start_dim=1)
        force_components = force_flat.unbind(dim=1)
        return -torch.stack([
            torch.autograd.grad(f.sum(), positions, create_graph=True)[0].flatten(start_dim=1)
            for f in force_components
        ], dim=1)

    @staticmethod
    def expand_padding_mask_to_hessian_mask(padding_mask):
        """
        Expand a (B, N) atom mask to a (B, 3N, 3N) Hessian mask.

        Parameters:
            padding_mask: Boolean tensor of shape (B, N_max)

        Returns:
            Boolean tensor of shape (B, 3N_max, 3N_max)
        """
        B, N = padding_mask.shape

        expanded_mask = padding_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(B, 3 * N)
        mask_matrix = expanded_mask.unsqueeze(2) & expanded_mask.unsqueeze(1)  # (B, 3N, 3N)

        return mask_matrix


class HVPVector(torch.nn.Module):
    def __init__(self, vector_type="random"):
        super().__init__()
        self.vector_type = vector_type

    def forward(self, positions, nonblank):
        """
        positions: (B, N_max, 3)
        nonblank: (B, N_max, 3), boolean mask
        Returns: (B, N_max, 3) vector (zeroed on padded atoms)
        """
        num_atoms = nonblank.sum(dim=1, dtype=torch.int64)
        N_max = nonblank.shape[1] # This is the maximum number of atoms across batches
        vectors = torch.zeros(len(num_atoms), 3*N_max, dtype=positions.dtype, device=positions.device)

        if self.vector_type == "random":
            for i in range(len(num_atoms)): # For each system,
                N = num_atoms[i]               # Get the number of atoms
                # Create a vector with i.i.d. values from a Gaussian distribution with zero mean and unit deviation
                values = torch.randn(3*N, dtype=positions.dtype, device=positions.device)
                # Divide by its norm to get a unit vector (adds a 1/(3N) factor to the expected squared values)
                values = values / torch.norm(values)
                vectors[i][:3*N] = values

        elif self.vector_type == "one-hot":
            for i in range(len(num_atoms)): # For each system,
                N = num_atoms[i]               # Get the number of atoms
                column_idx = torch.randint(0,3*N, (1,)) # Create a random integer from 0 to 3N inclusive
                vectors[i][column_idx] = 1.0            # Replace the 0.0 at the random index for 1.0

        else:
            raise ValueError(f"Unknown vector type {self.vector_type}")

        vectors = vectors.view(len(num_atoms), N_max, 3)
        return vectors


class HVP(torch.nn.Module):
    def forward(self, force, coordinates, vector, padding_mask):
        """
        source:       (B, N_max, 3)  force tensor
        coordinates:  (B, N_max, 3), requires_grad=True
        vector:       (B, N_max, 3), perturbation direction
        padding_mask: (B, N_max, 3), HVP padding mask with 3N non-zero elements
        Returns:      (B, N_max, 3), Hessian-vector product
        """

        # hessian_mask = self.expand_padding_mask_to_hessian_mask(padding_mask)
        hvp = -torch.autograd.grad(force, coordinates, grad_outputs=vector, create_graph=True, retain_graph=True)[0]

        return hvp, padding_mask.unsqueeze(-1).expand(-1, -1, 3)


class TrueHVP(torch.nn.Module):
    def forward(self, hessian, vector):
        """
        hessian: (B, 3N_max, 3N_max)
        vector:  (B, N_max, 3)
        Returns: (B, N_max, 3)
        """
        B, N, _ = vector.shape
        vector_flat = vector.flatten(start_dim=1).unsqueeze(-1)  # (B, 3N, 1)
        hvp_flat = torch.bmm(hessian, vector_flat).squeeze(-1)  # (B, 3N)
        hvp = hvp_flat.view(B, N, 3)

        return hvp  # (B, N, 3)
    

class StressForce(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbc = True

    def forward(self, energy, strain, coordinates, cell):
        total_energy = energy.sum()
        straingrad, grad = torch.autograd.grad(total_energy, [strain, coordinates], create_graph=True)
        if self.pbc:
            volume = torch.det(cell)
            stress = straingrad / volume.unsqueeze(1).unsqueeze(1)
        else:
            stress = straingrad

        return -grad, stress


class CellScaleInducer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbc = False

    def forward(self, coordinates, cell):
        strain = torch.eye(
            coordinates.shape[2], dtype=coordinates.dtype, device=coordinates.device, requires_grad=True
        ).tile(coordinates.shape[0],1,1)
        strained_coordinates = torch.bmm(coordinates, strain)
        strained_cell = torch.bmm(cell, strain)
        return strained_coordinates, strained_cell, strain
    
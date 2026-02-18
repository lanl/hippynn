import torch
from torch import Tensor

from .. import indexers


class Dipole(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.summer = indexers.MolSummer()

    def forward(self, charges: Tensor, positions: Tensor, system_index: Tensor, n_systems: int):
        if charges.shape[1] > 1:
            # charges contain multiple targets, so set up broadcasting
            charges = charges.unsqueeze(2)
            positions = positions.unsqueeze(1)

        # shape is (n_atoms, 3, n_targets) in multi-target mode
        # shape is (n_atoms, 3) in single target mode
        dipole_elements = charges * positions
        dipoles = self.summer(dipole_elements, system_index, n_systems)
        return dipoles


class Quadrupole(torch.nn.Module):
    """Computes quadrupoles as a flattened (n_systems,9) array.
    NOTE: Uses normalization sum_a q_a (r_a,i*r_a,j - 1/3 delta_ij r_a^2)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.summer = indexers.MolSummer()

    def forward(self, charges, positions, system_index, n_systems):
        # positions shape: (atoms, xyz)
        # charge shape: (atoms,1)
        ri_rj = positions.unsqueeze(1) * positions.unsqueeze(2)
        ri_rj_flat = ri_rj.reshape(-1, 9)  # Flatten to component
        rsq = (positions**2).sum(dim=1).unsqueeze(1)  # unsqueeze over component index
        delta_ij = torch.eye(3, device=rsq.device).flatten().unsqueeze(0)  # unsqueeze over atom index
        quad_elements = charges * (ri_rj_flat - (1 / 3) * (rsq * delta_ij))
        quadrupoles = self.summer(quad_elements, system_index, n_systems)
        return quadrupoles

import warnings

import torch

from .. import indexers


class PerAtom(torch.nn.Module):
    def forward(self, features, species):
        n_atoms = (species != 0).type(features.dtype).sum(dim=1)
        return features / n_atoms.unsqueeze(1)


class VecMag(torch.nn.Module):
    def forward(self, vector_feature):
        return torch.norm(vector_feature, dim=1)


class CombineEnergy(torch.nn.Module):
    """
    Combines the energies (molecular and atom energies) from two different 
    nodes, e.g. HEnergy, Coulomb, or ScreenedCoulomb Energy Nodes. 
    """
    def __init__(self):
        super().__init__()
        self.summer = indexers.MolSummer()

    def forward(self, atom_energy_1, atom_energy_2, system_index, n_systems):
        """
        :param: atom_energy_1 per-atom energy from first node. 
        :param: atom_energy_2 per atom energy from second node. 
        :param: system_index the molecular index for atoms in the batch
        :param: total number of molecules in the batch
        :return: Total Energy
        """
        total_atom_energy = atom_energy_1 + atom_energy_2
        mol_energy = self.summer(total_atom_energy, system_index, n_systems)
        
        return mol_energy, total_atom_energy

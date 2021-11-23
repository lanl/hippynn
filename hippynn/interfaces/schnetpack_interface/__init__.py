"""
Interface to use schnetpack networks with hippynn.

Note: Only open boundary conditions are supported for now.

"""
try:
    import schnetpack
except ImportError as ie:
    raise ImportError("Schnetpack installation is required to use the schnetpack interface.") from ie

import torch

from schnetpack import Properties

from ...graphs.nodes.base import SingleNode, AutoKw
from ...graphs.nodes.networks import Network
from ...graphs.indextypes import IdxType


class SchNetWrapper(torch.nn.Module):
    def __init__(self, schnet):
        super().__init__()
        self.schnet = schnet
        feature_sizes = [x.dense.out_features for x in schnet.interactions]
        if not schnet.return_intermediates:
            feature_sizes = [feature_sizes[-1]]
        self.feature_sizes = feature_sizes

    def forward(self, z_arr, r_arr, nonblank):
        """
        Wrap a call into the underlying schnet, which uses
        dictionaries.
        :param z_arr:
        :param r_arr:
        :param nonblank:
        :return:
        """
        packed = create_schnetpack_inputs(z_arr, r_arr, nonblank)
        outputs = self.schnet(packed)
        if self.schnet.return_intermediate:
            outputs = outputs[1]
        else:
            outputs = [outputs]
        outputs = [x[packed["nonblank"]] for x in outputs]
        return outputs


class SchNetNode(AutoKw, Network, SingleNode):
    _input_names = "species", "positions", "nonblank"
    _index_state = IdxType.Atoms
    _auto_module_class = SchNetWrapper

    def __init__(self, name, parents, module="auto", module_kwargs=None):
        if module == "auto":
            self.module_kwargs = module_kwargs
            module = self.auto_module()
        super().__init__(name, parents, module=module)


def create_schnetpack_inputs(z_arr, r_arr, nonblank):

    dtype = r_arr.dtype
    device = r_arr.device
    n_atoms_per_mol = (z_arr > 0).sum(axis=1)

    n_mols = n_atoms_per_mol.shape[0]
    n_atoms = n_atoms_per_mol.max()
    z_arr = z_arr[:, :n_atoms]
    r_arr = r_arr[:, :n_atoms]
    nonblank = nonblank[:, :n_atoms]

    atom_range = torch.arange(n_atoms).unsqueeze(0).expand(n_mols, -1)

    atom_mask = torch.zeros(n_mols, n_atoms, dtype=z_arr.dtype, device=device)
    atom_mask[atom_range < n_atoms_per_mol.unsqueeze(1)] = 1

    neighbor_base = torch.arange(n_atoms - 1, device=device, dtype=dtype).unsqueeze(0).expand(n_mols, n_atoms, -1)
    neighbor_base = neighbor_base + torch.triu(torch.ones_like(neighbor_base), diagonal=0)

    neighbor_mask = atom_mask.unsqueeze(2) * atom_mask.unsqueeze(1)[:, :, 1:]
    neighbors = neighbor_base * neighbor_mask

    cell = torch.zeros((n_mols, 3, 3), device=device, dtype=dtype)
    cell_offset = torch.zeros((n_mols, n_atoms, n_atoms - 1, 3), device=device, dtype=dtype)

    return {
        Properties.atom_mask: atom_mask.to(dtype),
        Properties.neighbors: neighbors,
        Properties.neighbor_mask: neighbor_mask.to(dtype),
        Properties.Z: z_arr,
        Properties.R: r_arr,
        Properties.cell: cell,
        Properties.cell_offset: cell_offset,
        "nonblank": nonblank,
    }

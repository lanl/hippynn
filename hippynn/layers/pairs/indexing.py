"""
Layers for pair finding
"""
import torch

from .open import _PairIndexer


class ExternalNeighbors(_PairIndexer):
    """
    module for ASE-like neighbors list fed into the graph
    """

    def forward(self, coordinates, real_atoms, shifts, cell, pair_first, pair_second):
        n_molecules, n_atoms, _ = coordinates.shape
        atom_coordinates = coordinates.reshape(n_molecules * n_atoms, 3)[real_atoms]
        paircoord = atom_coordinates[pair_second] - atom_coordinates[pair_first] + shifts.to(cell.dtype) @ cell
        distflat = paircoord.norm(dim=1)

        # Trim the list to only include relevant atoms, improving performance.
        within_cutoff_pairs = distflat < self.hard_dist_cutoff
        distflat = distflat[within_cutoff_pairs]
        pair_first = pair_first[within_cutoff_pairs]
        pair_second = pair_second[within_cutoff_pairs]
        paircoord = paircoord[within_cutoff_pairs]

        return distflat, pair_first, pair_second, paircoord


class PairReIndexer(torch.nn.Module):
    def forward(self, molatomatom_thing, molecule_index, atom_index, pair_first, pair_second):

        molecule_position = molecule_index[pair_first]
        absolute_first = atom_index[pair_first]
        absolute_second = atom_index[pair_second]
        out = molatomatom_thing[molecule_position, absolute_first, absolute_second]
        if out.ndimension() == 1:
            out = out.unsqueeze(1)
        return out


class PairDeIndexer(torch.nn.Module):
    def forward(self, features, molecule_index, atom_index, n_molecules, n_atoms_max, pair_first, pair_second):
        molecule_position = molecule_index[pair_first]
        absolute_first = atom_index[pair_first]
        absolute_second = atom_index[pair_second]

        if features.ndimension() == 1:
            features = features.unsqueeze(-1)
        featshape = features.shape[1:]
        out_shape = (n_molecules, n_atoms_max, n_atoms_max, *featshape)

        result = torch.zeros(*out_shape, device=features.device, dtype=features.dtype)
        result[molecule_position, absolute_first, absolute_second] = features
        return result


class MolPairSummer(torch.nn.Module):
    def forward(self, pairfeatures, mol_index, n_molecules, pair_first):
        pair_mol = mol_index[pair_first]
        feat_shape = (1,) if pairfeatures.ndimension() == 1 else pairfeatures.shape[1:]
        out_shape = (n_molecules, *feat_shape)
        result = torch.zeros(out_shape, device=pairfeatures.device, dtype=pairfeatures.dtype)
        result.index_add_(0, pair_mol, pairfeatures)
        return result


class PairCacher(torch.nn.Module):
    def __init__(self, n_images=1):
        super().__init__()
        self.set_images(n_images=n_images)

    def set_images(self, n_images):
        self.n_images = n_images

    def forward(
        self, pair_first, pair_second, cell_offsets, offset_index, real_atoms, mol_index, n_molecules, n_atoms_max
    ):
        # Set up absolute indices
        abs_atoms = real_atoms % n_atoms_max
        pfabs = abs_atoms[pair_first]
        psabs = abs_atoms[pair_second]
        mol = mol_index[pair_first]

        n_offsets = (2 * self.n_images + 1) ** 3

        # Reorder the indices
        order = offset_index + n_offsets * (psabs + n_atoms_max * (pfabs + n_atoms_max * mol))
        order = torch.argsort(order)
        mol = mol[order]
        pfabs = pfabs[order]
        psabs = psabs[order]
        offset_index = offset_index[order]
        cell_offsets = cell_offsets[order]

        # Create sparse tensor
        indices = torch.stack([mol, pfabs, psabs, offset_index], axis=0)
        values = cell_offsets
        size = (n_molecules, n_atoms_max, n_atoms_max, n_offsets, 3)
        s = torch.sparse_coo_tensor(
            indices=indices, values=values, size=size, dtype=torch.int, device=pair_first.device
        )
        s = s.coalesce()
        return s


class PairUncacher(torch.nn.Module):
    def __init__(self, n_images=1):
        super().__init__()
        self.set_images(n_images=n_images)

    def set_images(self, n_images):
        self.n_images = n_images

    def forward(self, sparse, coordinates, cell, real_atoms, inv_real_atoms, n_atoms_max, n_molecules):

        if not sparse.is_sparse:
            sparse = sparse.to_sparse(sparse_dim=4)
        sparse = sparse.coalesce()
        index = sparse.indices()
        values = sparse.values()

        mol, pfabs, psabs, offset_index = index.unbind(0)
        pfb = pfabs + mol * n_atoms_max
        psb = psabs + mol * n_atoms_max
        cell_offsets = values

        pair_first = inv_real_atoms[pfb]
        pair_second = inv_real_atoms[psb]

        atom_coordinates = coordinates.reshape(n_molecules * n_atoms_max, 3)[real_atoms]
        offsets = torch.bmm(cell_offsets.to(cell.dtype).unsqueeze(1), cell[mol]).squeeze(1)

        paircoord = atom_coordinates[pair_first] - atom_coordinates[pair_second] + offsets

        distflat = paircoord.norm(dim=1)

        return distflat, pair_first, pair_second, paircoord, cell_offsets, offset_index

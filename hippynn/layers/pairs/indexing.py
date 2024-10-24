"""
Layers for pair finding
"""
import torch

from ...custom_kernels.utils import get_id_and_starts
from .open import _PairIndexer
from .periodic import filter_pairs


class ExternalNeighbors(_PairIndexer):
    """
    module for ASE-like neighbors list fed into the graph
    """

    def forward(self, coordinates, real_atoms, shifts, cell, pair_first, pair_second):
        if (coordinates.ndim > 3) or (coordinates.ndim == 3 and coordinates.shape[0] != 1):
            raise ValueError(f"coordinates must have (n,3) or (1,n,3) but has shape {coordinates.shape}")
        if coordinates.ndim == 3:
            coordinates = coordinates.squeeze(0)
        if (cell.ndim > 3) or (cell.ndim == 3 and cell.shape[0] != 1):
            raise ValueError(f"cell must have (3,3) or (1,3,3) but has shape {cell.shape}")
        if cell.ndim == 3:
            cell = cell.squeeze(0)
            
        atom_coordinates = coordinates[real_atoms]
        paircoord = atom_coordinates[pair_second] - atom_coordinates[pair_first] + shifts.to(cell.dtype) @ cell
        distflat = paircoord.norm(dim=1)
        # We filter the lists to only send forward relevant pairs (those with distance under cutoff), improving performance.   
        return filter_pairs(self.hard_dist_cutoff, distflat, pair_first, pair_second, paircoord)


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
        if pairfeatures.shape[0] == 1:
            feat_shape = (1,)
            pairfeatures.unsqueeze(-1)
        else:
            feat_shape = pairfeatures.shape[1:]
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
        indices = torch.stack([mol, pfabs, psabs, offset_index], dim=0)
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


def padded_neighlist(pair_first, pair_second, pair_coord, atom_array):
    """
    Convert from index list pair_first, pair_second
    to index list of `jlist`
    where jlist has shape (n_atoms, n_neigh_max).
    jlists consists of index of atom neighbors for each atom i,
    and is padded with values of -1.
    """

    # Scalar sizes required
    n_atoms = atom_array.shape[0]

    if pair_first.shape[0] == 0:  # empty neighbors list
        dev = pair_coord.device
        rijlist_pad = torch.empty((n_atoms, 0, 3), device=dev, dtype=pair_coord.dtype)
        jlist_pad = torch.empty((n_atoms, 0), device=dev, dtype=torch.int64)
        return jlist_pad, rijlist_pad

    with torch.no_grad():
        nneigh_max = torch.unique(pair_first, return_counts=True)[1].max()

        # Sorting pair list
        # Note, this isn't quite what resort_pairs does,
        # because resort_pairs only sorts on the first index
        # Note: The index is given so that the index functions
        # as a base-n_atoms representation of the pair of indices as a scalar number.
        # todo: possibly, just ensure a good sorting order for all pair finders.
        # todo: possibly, include cell offset information.
        # then this could be used to cache pairs without sparse tensors.
        ind = n_atoms * pair_first + pair_second
        sort = torch.argsort(ind)
        ilist = pair_first[sort]
        jlist = pair_second[sort]
        pair_coord = pair_coord[sort]

        #  List of where new i's start
        i_vals, istart = get_id_and_starts(ilist)
        istart = istart[:-1]

        # Relative index for j values
        # When boundaries of i-atoms are not crossed, we increment by 1
        diffrelj = torch.ones(len(jlist), dtype=jlist.dtype, device=jlist.device)
        # When boundaries of i-atoms are crossed, we decrement by the difference
        # in the i-atom position.
        diffrelj[istart[1:]] -= torch.diff(istart)
        # By summing the increments we get the index of j in the padded neighbors list
        jrellist = torch.cumsum(diffrelj, dim=0) - 1

        # Build Padded neighbor list and padded rij
        jlist_pad = -torch.ones(n_atoms, nneigh_max, dtype=ilist.dtype, device=ilist.device)
        jlist_pad[ilist, jrellist] = jlist

    rijlist_pad = torch.zeros(n_atoms, nneigh_max, 3, dtype=pair_coord.dtype, device=pair_coord.device)
    rijlist_pad[ilist, jrellist] = pair_coord

    return jlist_pad, rijlist_pad


class PaddedNeighModule(torch.nn.Module):
    def forward(self, pair_first, pair_second, pair_coord, atom_array):
        return padded_neighlist(pair_first, pair_second, pair_coord, atom_array)

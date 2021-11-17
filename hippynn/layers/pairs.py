"""
Layers for pair finding
"""
import itertools
import torch
import numpy as np


# Possible TODO do periodic coordinates ever need to be differentiable with respect to the cell
# For volume changes?

class _PairIndexer(torch.nn.Module):
    def __init__(self, hard_dist_cutoff):
        super().__init__()
        self.hard_dist_cutoff = hard_dist_cutoff


class OpenPairIndexer(_PairIndexer):

    def forward(self, coordinates, nonblank, real_atoms, inv_real_atoms):

        with torch.no_grad():

            n_molecules, n_atoms, _ = coordinates.shape
            cur_device = coordinates.device
            # Figure out how many molecules and atoms we have

            atom_index = torch.reshape(torch.arange(n_molecules * n_atoms, dtype=torch.long, device=cur_device),
                                       (n_molecules, n_atoms))
            n_atoms_max_batch = int(nonblank.sum(axis=1).max())
            namax_box = n_atoms_max_batch

            coordinates_trimmed = coordinates[:, :namax_box]
            atom_index = atom_index[:, :namax_box]
            nonblank_trimmed = nonblank[:, :namax_box]

            # Construct all the pair distances
            pair_dists = torch.norm(coordinates_trimmed.unsqueeze(2) - coordinates_trimmed.unsqueeze(1), dim=3)

            # Construct a unique index for each atom (including blanks here)

            # Pairs which are both not blank
            nonblank_pair = nonblank_trimmed.unsqueeze(1) * nonblank_trimmed.unsqueeze(2)
            # Don't connect an atom to itself
            nonself_atoms = (~torch.eye(namax_box, dtype=nonblank.dtype, device=cur_device)).unsqueeze(
                0)
            # Only take pairs that are close enough
            if self.hard_dist_cutoff is not None:
                close_atoms = (pair_dists < self.hard_dist_cutoff)
                pair_presence = nonblank_pair & nonself_atoms & close_atoms
            else:
                pair_presence = nonblank_pair & nonself_atoms

            # Now that we have a n_mol,n_atom,n_atom boolean mask of real pairs, we extract
            # the atom indices and pair distances associated with each.
            pair_indices = torch.nonzero(pair_presence, as_tuple=True)

            pair_first = atom_index.unsqueeze(2).expand(-1, -1, namax_box)[pair_indices]
            pair_second = atom_index.unsqueeze(1).expand(-1, namax_box, -1)[pair_indices]

            # This converts the pair index from an index that counts blank atoms to one that doesn't.
            pair_first = inv_real_atoms[pair_first]
            pair_second = inv_real_atoms[pair_second]

        coordflat = coordinates.reshape(n_molecules * n_atoms, 3)[real_atoms]
        paircoord = coordflat[pair_second] - coordflat[pair_first]
        distflat2 = paircoord.norm(dim=1)

        return distflat2, pair_first, pair_second, paircoord


class PeriodicPairIndexer(_PairIndexer):
    def __init__(self,*args,n_images=1,**kwargs):
        super().__init__(*args,**kwargs)
        import itertools
        # shape is (n_images+2)**3 by 3
        combinator = torch.tensor(list(itertools.product(range(-n_images,n_images+1),repeat=3)))
        self.n_images=n_images
        self.register_buffer("combinator",combinator.to(torch.float))

    def forward(self, coordinates, nonblank, real_atoms, inv_real_atoms, cells):

        original_coordinates = coordinates

        with torch.no_grad():
            cur_device = coordinates.device

            # Figure out how many molecules and atoms we have
            n_molecules, n_atoms, _ = coordinates.shape
            n_atoms_max_batch = int(nonblank.sum(axis=1).max())
            # Construct a unique index for each atom (including blanks here)
            atom_index = torch.reshape(torch.arange(n_molecules * n_atoms,
                                                    dtype=torch.long, device=cur_device), (n_molecules, n_atoms))

            nonblank = nonblank[:, :n_atoms_max_batch]
            coordinates = coordinates[:, :n_atoms_max_batch]
            atom_index = atom_index[:, :n_atoms_max_batch]

            # Eliminate blank atom pairs, construct indices for them
            nonblank_pair = nonblank.unsqueeze(1) * nonblank.unsqueeze(2)
            nb_b, nb_a1, nb_a2 = torch.nonzero(nonblank_pair, as_tuple=True)

            # Get the indices of non-blank atoms
            pair_first = atom_index[nb_b, nb_a1]
            pair_second = atom_index[nb_b, nb_a2]
            # shape pair

            pair_diffcoords = coordinates[nb_b, nb_a1] - coordinates[nb_b, nb_a2]

            # Begin periodic part: find offsets for all images of cells.
            # Shape: n_shifts, n_batch, 3
            cell_shifts = torch.tensordot(self.combinator, cells, dims=((1,), (1,)))
            # Shape n_shifts,n_atoms,3
            offsets = cell_shifts[:, nb_b]

            # Find pairs/images close enough to be within the cutoff
            # shape n_shifts, pair
            pair_dist = (pair_diffcoords.unsqueeze(0) + offsets).norm(dim=2)
            close_pairs = (pair_dist < self.hard_dist_cutoff) & (pair_dist > 1e-5)  # get rid of self-pairs
            cp_offset, cp_pair = torch.nonzero(close_pairs, as_tuple=True)
            # cp_pair holds the indices of pairs
            # cp_offset holds the indices of the offsets

            # Extract the pairs within the cutoff
            pair_first = pair_first[cp_pair]
            pair_second = pair_second[cp_pair]
            pair_offsets = offsets[cp_offset, cp_pair]
            cell_offsets = self.combinator[cp_offset]

            # This converts the atom index from an index that counts blank atoms to one that doesn't;
            # Afterwards it indexes the flat array of atoms in the batch.
            pair_first = inv_real_atoms[pair_first]
            pair_second = inv_real_atoms[pair_second]

            del nb_b, nb_a1, nb_a2, cp_pair, pair_dist, offsets, pair_diffcoords, nonblank_pair, atom_index

        coordflat = original_coordinates.reshape(n_molecules * n_atoms, 3)[real_atoms]
        paircoord = coordflat[pair_first] - coordflat[pair_second] + pair_offsets
        distflat2 = paircoord.norm(dim=1)

        return distflat2, pair_first, pair_second, paircoord, cell_offsets, cp_offset


class ExternalNeighbors(_PairIndexer):
    """
    module for ASE-like neighbors list fed into the graph
    """
    def forward(self, coordinates, real_atoms, shifts, cell, pair_first, pair_second):
        n_molecules, n_atoms, _ = coordinates.shape
        atom_coordinates  = coordinates.reshape(n_molecules * n_atoms, 3)[real_atoms]
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
    def forward(self, features, molecule_index, atom_index, n_molecules,n_atoms_max, pair_first, pair_second):
        molecule_position = molecule_index[pair_first]
        absolute_first = atom_index[pair_first]
        absolute_second = atom_index[pair_second]

        if features.ndimension()==1:
            features = features.unsqueeze(-1)
        featshape = features.shape[1:]
        out_shape = (n_molecules, n_atoms_max,n_atoms_max, *featshape)

        result = torch.zeros(*out_shape, device=features.device, dtype=features.dtype)
        result[molecule_position,absolute_first,absolute_second] = features
        return result


class MolPairSummer(torch.nn.Module):
    def forward(self,pairfeatures, mol_index, n_molecules, pair_first):
        pair_mol = mol_index[pair_first]
        feat_shape = (1,) if pairfeatures.ndimension() == 1 else pairfeatures.shape[1:]
        out_shape = (n_molecules,*feat_shape)
        result = torch.zeros(out_shape,device=pairfeatures.device,dtype=pairfeatures.dtype)
        result.index_add_(0,pair_mol,pairfeatures)
        return result


class PairCacher(torch.nn.Module):
    def __init__(self, n_images=1):
        super().__init__()
        self.set_images(n_images=n_images)

    def set_images(self,n_images):
        self.n_images = n_images

    def forward(self, pair_first, pair_second, cell_offsets, offset_index,
                real_atoms, mol_index, n_molecules, n_atoms_max):
        # Set up absolute indices
        abs_atoms = real_atoms % n_atoms_max
        pfabs = abs_atoms[pair_first]
        psabs = abs_atoms[pair_second]
        mol = mol_index[pair_first]

        n_offsets = (2*self.n_images+1)**3

        # Reorder the indices
        order = offset_index + n_offsets*(psabs + n_atoms_max * (pfabs + n_atoms_max * mol))
        order = torch.argsort(order)
        mol = mol[order]
        pfabs = pfabs[order]
        psabs = psabs[order]
        offset_index = offset_index[order]
        cell_offsets = cell_offsets[order]

        # Create sparse tensor
        indices = torch.stack([mol, pfabs, psabs, offset_index], axis=0)
        values = cell_offsets
        size = (n_molecules, n_atoms_max, n_atoms_max,n_offsets, 3)
        s = torch.sparse_coo_tensor(indices=indices,
                                    values=values,
                                    size=size,
                                    dtype=torch.int,
                                    device=pair_first.device)
        s = s.coalesce()
        return s


class PairUncacher(torch.nn.Module):
    def __init__(self, n_images=1):
        super().__init__()
        self.set_images(n_images=n_images)

    def set_images(self,n_images):
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

        paircoord = atom_coordinates[pair_first] \
                    - atom_coordinates[pair_second] \
                    + offsets

        distflat = paircoord.norm(dim=1)

        return distflat, pair_first, pair_second, paircoord, cell_offsets, offset_index


class RDFBins(torch.nn.Module):
    def __init__(self, bins, species_set):
        super().__init__()
        if bins is None:
            raise TypeError("Bins must not be None!")
        bins = torch.as_tensor(bins)
        species_set = torch.as_tensor(species_set)
        self.register_buffer("bins", bins.to(torch.float))
        self.register_buffer("species_set", species_set.to(torch.int))

    def bin_info(self):
        # Note: widths don't make perfect sense for non-evenly-spaced bins.
        centers = (self.bins[1:] + self.bins[:-1])/2
        widths = (self.bins[1:] - self.bins[:-1])
        return centers, widths

    def forward(self, pair_dist, pair_first, pair_second, one_hot, n_molecules):
        n_species = one_hot.shape[-1]
        n_bins = self.bins.shape[0] - 1

        rdf = torch.zeros((n_species, n_species, n_bins), dtype=pair_dist.dtype, device=pair_dist.device)
        for i in range(n_species):
            for j in range(n_species):
                mask = one_hot[:, i][pair_first] & one_hot[:, j][pair_second]
                maskpairs = pair_dist[mask]
                less = maskpairs.unsqueeze(-1) < self.bins.unsqueeze(0)
                less_counts = less.sum(dim=0)
                rdf[i, j] = less_counts[..., 1:] - less_counts[..., :-1]
        return (rdf/n_molecules).unsqueeze(0)


#####
# Numpy implementation of pair finding

def wrap_points_np(coords, cell, inv_cell):
    # cell is (basis,cartesian)
    # inv is (cartesian,basis)
    projections = coords @ inv_cell
    wrapped_offset, fractional_coords = np.divmod(projections, 1)
    # wrapped_coords is (atoms,cartesian)
    wrapped_coords = fractional_coords @ cell
    return wrapped_coords,wrapped_offset.astype(np.int64)


def neighbor_list_np(cutoff, coords, cell):
    # cell is (basis,cartesian)
    # inv is (cartesian,basis)
    inv_cell = np.linalg.inv(cell)
    coords, wrapped_offset = wrap_points_np(coords, cell, inv_cell)

    eps = 1e-5
    inv_lengths = np.linalg.norm(inv_cell, axis=0)
    n_bounds = (np.floor(cutoff * inv_lengths + eps) + 1).astype(int)
    n1, n2, n3 = n_bounds

    drij = coords[:, np.newaxis] - coords[np.newaxis, :]
    wrap_offset_ij = wrapped_offset[:, np.newaxis] - wrapped_offset[np.newaxis, :]

    pair_first = []
    pair_second = []
    pair_image = []
    for i1 in range(-n1, n1 + 1):
        for i2 in range(-n2, n2 + 1):
            for i3 in range(-n3, n3 + 1):
                oi = (i1, i2, i3)
                oi_float = np.asarray(oi, dtype=coords.dtype)
                offset = np.dot(oi_float, cell)
                diff = drij + offset
                dist = np.linalg.norm(diff, axis=2)
                pf, ps = np.nonzero(dist < cutoff)

                if i1 == i2 == i3 == 0:
                    nonself = pf != ps
                    pf = pf[nonself]
                    ps = ps[nonself]

                if len(pf) > 0:
                    pi = np.repeat(np.asarray(oi)[np.newaxis, :], repeats=len(pf), axis=0)
                    # pi is wrapped image locations, need to convert back to absolute.
                    pi = pi - wrap_offset_ij[pf,ps]
                    pair_first.append(pf)
                    pair_second.append(ps)
                    pair_image.append(pi)

    pair_first = np.concatenate(pair_first)
    pair_second = np.concatenate(pair_second)
    pair_image = np.concatenate(pair_image)

    return pair_first, pair_second, pair_image

# End Numpy implementation of pair finding
####

####
# Torch implementation of pair finding

def wrap_points_torch(coords, cell, inv_cell):
    # cell is (basis,cartesian)
    # inv is (cartesian,basis)
    projections = coords @ inv_cell
    fractional_coords = torch.fmod(projections,1)
    wrapped_offset =  torch.div(projections,1,rounding_mode='floor')
    # wrapped_coords is (atoms,cartesian)
    wrapped_coords = fractional_coords @ cell
    return wrapped_coords,wrapped_offset.to(torch.int64)

def neighbor_list_torch(cutoff, coords, cell):
    # cell is (basis,cartesian)
    # inv is (cartesian,basis)
    inv_cell = torch.linalg.inv(cell)
    coords, wrapped_offset = wrap_points_torch(coords, cell, inv_cell)

    eps = 1e-5
    inv_lengths = torch.linalg.norm(inv_cell, axis=0)
    n_bounds = (torch.floor(cutoff * inv_lengths + eps) + 1).to(torch.int64)
    n1, n2, n3 = n_bounds.unbind()

    drij = coords.unsqueeze(1) - coords.unsqueeze(0)

    wrap_offset_ij = wrapped_offset.unsqueeze(1) - wrapped_offset.unsqueeze(0)

    pair_first = []
    pair_second = []
    pair_image = []
    for i1 in range(-n1, n1 + 1):
        for i2 in range(-n2, n2 + 1):
            for i3 in range(-n3, n3 + 1):
                oi = (i1, i2, i3)
                oi = torch.as_tensor(oi,device=coords.device)
                oi_float = oi.to(coords.dtype)
                offset = oi_float @ cell
                diff = drij + offset
                dist = torch.linalg.norm(diff, axis=2)
                pf, ps = torch.nonzero(dist < cutoff, as_tuple=True)

                if i1 == i2 == i3 == 0:
                    nonself = pf != ps
                    pf = pf[nonself]
                    ps = ps[nonself]

                if len(pf) > 0:
                    pi = oi.unsqueeze(0).expand(pf.shape[0],-1)
                    # pi is wrapped image locations, need to convert back to absolute.
                    pi = pi - wrap_offset_ij[pf,ps]
                    pair_first.append(pf)
                    pair_second.append(ps)
                    pair_image.append(pi)

    pair_first = torch.cat(pair_first)
    pair_second = torch.cat(pair_second)
    pair_image = torch.cat(pair_image)

    return pair_first, pair_second, pair_image

# End Torch implementation of pair finding
####


class _DispatchNeighbors(torch.nn.Module):
    def __init__(self ,dist_hard_max):
        super().__init__()
        self.dist_hard_max = dist_hard_max

        self.set_combinator(1)
        self.n_images =1

    def set_combinator(self ,n_images):
        self.n_images = n_images

    def compute_one(self,r,c):
        return NotImplemented

    def forward(self ,coordinates, nonblank, real_atoms, inv_real_atoms, cell, mol_index, n_molecules, n_atoms_max):

        with torch.no_grad():
            dev = coordinates.device  # where to put the results.

            cell_list = cell.unbind(0)
            coord_list = [c[nb] for c ,nb in zip(coordinates.unbind(0) ,nonblank.unbind(0))]
            nlist_data = []
            for mol_num ,(r ,c) in enumerate(zip(coord_list ,cell_list)):
                outs = self.compute_one(r ,c)
                pf, ps, of = outs
                if len(pf) == 0:
                    continue
                pf += mol_num * n_atoms_max
                ps += mol_num * n_atoms_max
                max_images = of.abs().max()
                if max_images > self.n_images:
                    self.set_combinator(max_images)
                nlist_data.append((pf ,ps ,of))

            # transpose
            pair_first ,pair_second ,offsets = zip(*nlist_data)

            # concatenate
            pair_first = inv_real_atoms[torch.cat(pair_first).to(dev)]
            pair_second = inv_real_atoms[torch.cat(pair_second).to(dev)]
            # offset_index = torch.cat(offset_index)
            offsets = torch.cat(offsets).to(dev)

            # Number the offsets
            n_off = self.n_images * 2 + 1
            o1, o2, o3 = (offsets + self.n_images).unbind(dim=1)
            offset_index = o3 + n_off *(o2 + n_off *o1)

            pair_mol = mol_index[pair_first]
            pair_cell = cell[pair_mol]
            pair_offsets = torch.bmm(offsets.unsqueeze(1).to(pair_cell.dtype) ,pair_cell).squeeze(1)

            # now calculate pair_dist, paircoord differentiably

        coordflat = coordinates.reshape(n_molecules * n_atoms_max, 3)[real_atoms]
        paircoord = coordflat[pair_first] - coordflat[pair_second] + pair_offsets
        distflat2 = paircoord.norm(dim=1)

        return distflat2, pair_first, pair_second, paircoord, offsets ,offset_index


class NPNeighbors(_DispatchNeighbors):
    def compute_one(self,positions,cell):
        positions = positions.detach().cpu().numpy()
        cell = cell.detach().cpu().numpy()
        outputs = neighbor_list_np(self.dist_hard_max, positions, cell)
        return [torch.from_numpy(o) for o in outputs]


class TorchNeighbors(_DispatchNeighbors):
    def compute_one(self,positions,cell):
        with torch.no_grad():
            outputs = neighbor_list_torch(self.dist_hard_max,positions,cell)
        return outputs

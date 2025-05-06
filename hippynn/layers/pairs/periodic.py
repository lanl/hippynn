import torch

from .open import _PairIndexer, PairMemory

# Deprecated?
class StaticImagePeriodicPairIndexer(_PairIndexer):
    """Finds Pairs within a given number of images"""
    def __init__(self, *args, n_images=1, **kwargs):
        super().__init__(*args, **kwargs)
        import itertools

        # shape is (n_images+2)**3 by 3
        combinator = torch.tensor(list(itertools.product(range(-n_images, n_images + 1), repeat=3)))
        self.n_images = n_images
        self.register_buffer("combinator", combinator.to(torch.float))

    def forward(self, coordinates, nonblank, real_atoms, inv_real_atoms, cells):

        original_coordinates = coordinates

        with torch.no_grad():
            cur_device = coordinates.device

            # Figure out how many molecules and atoms we have
            n_molecules, n_atoms, _ = coordinates.shape
            n_atoms_max_batch = int(nonblank.sum(axis=1).max())
            # Construct a unique index for each atom (including blanks here)
            atom_index = torch.reshape(
                torch.arange(n_molecules * n_atoms, dtype=torch.long, device=cur_device), (n_molecules, n_atoms)
            )

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


@torch.jit.script
def tracebatch(mat):
    """
    Trace of  of batch of matrices.
    Defaults to last two dimensions.
    """
    return torch.diagonal(mat, dim1=-2, dim2=-1).sum(dim=-1)

# For some reason torch.linalg on GPU tends to
# spend a lot of time allocating memory, especially
# when it is given a large batch of matrices.
# So we use the Cayley-Hamilton version of a matrix inverse
# Without calling any linalg functions.
@torch.jit.script
def cayley_inv(A):
    """
    Inverse of a batch of 3x3 matrices.
    """
    # Uses A^-1 = 1/det(A) ([tr(A)**2-tr(A**2)]I/2 - Atr(A) + A**2)
    assert A.shape[-1] == 3
    assert A.shape[-2] == 3

    trA = tracebatch(A)
    Asq = torch.bmm(A, A)
    trAsq = tracebatch(Asq)
    eye = torch.eye(3, dtype=A.dtype, device=A.device).broadcast_to(A.shape)
    eyeterm = (torch.pow(trA, 2) - trAsq).unsqueeze(-1).unsqueeze(-1) * eye / 2
    Ainvunscale = eyeterm - A * (trA.unsqueeze(-1).unsqueeze(-1)) + Asq
    det = tracebatch(torch.bmm(A, Ainvunscale)) / 3.

    return Ainvunscale / det.unsqueeze(-1).unsqueeze(-1)

@torch.jit.script
def wrap_systems_torch(coords, cell, cutoff: float):
    """
    Wraps coordinates to within the unit cell.

    Returns Inverse cells of shape (n_systems, 3 [cartesian], 3 [basis])
    Wrapped coords in same shape of coords
    Wrapped offset in shape (n_systems,n_atoms,3 [basis])
    and n_bounds in shape (n_systems,3) in terms of the number of image cells
    to search for each basis vector (radius).

    :param coords: coordinates of shape (n_systems,n_atoms,3)
    :param cell: cells of shape (n_systems, 3 [basis], 3 [cartesian])
    :param cutoff: cutoff radius to search.
    :return: inv_cells, wrapped_coords, wrapped_offset, n_bounds
    """
    # cell is (sys,basis,cartesian)

    # This is faster than calling the torch.linalg.inv directly as of torch 1.10
    inv_cell = cayley_inv(cell)
    # inv is (sys,cartesian,basis)
    projections = torch.bmm(coords, inv_cell)
    fractional_coords = torch.remainder(projections, 1)
    wrapped_offset = torch.div(projections, 1, rounding_mode="floor")
    wrapped_coords = torch.bmm(fractional_coords, cell)
    # wrapped_coords is (system,atoms,cartesian)

    # Compute number images to search in each basis direction
    eps = 1e-5
    inv_lengths = torch.linalg.norm(inv_cell, dim=1)
    n_bounds = (torch.floor(cutoff * inv_lengths + eps) + 1).to(torch.int64)

    return inv_cell, wrapped_coords, wrapped_offset.to(torch.int64), n_bounds


def filter_pairs(cutoff, distflat, *addn_features):
    filter = distflat < cutoff
    return tuple((array[filter] for array in [distflat, *addn_features]))


def union_indices_2d_masks(*arrays):
    """
    Compute the indices of the union of a list of 2D mask arrays.

    Assume a set of arrays a_1...a_n, each with shape (b,f_1),(b,f_2)...(b,f_n)
    The output is equivalent to broadcasting the combined multiplication:
        m = a1*a2*a3*a4
    into shape (b,f_1,f_2,..f_n)  and return `torch.nonzero(m)`.
    However, this function does not materialize the large intermediate array
    of all possible products. It does call nonzero several times which causes
    blocking ops on the GPU, but because they are almost back-to-back
    this is usually not worse than calling nonzero in the first place.

    Performance heuristic note: Provide the arrays in ascending sparsity order.

    """
    # Standalone function, so could be moved to elsewhere if needed.

    n_masks = len(arrays)

    if n_masks == 0:
        raise ValueError("Union function requires masks.")
        # Also possible: return zero-length batch indices if passed no masks?

    full_shapes = [arr.shape for arr in arrays]
    batch_sizes, mask_lengths = zip(*full_shapes)

    # Validate shapes.
    b = batch_sizes[0]
    assert all(b == bb for bb in batch_sizes), "batch broadcasting must be possible."

    start_mask, *arrays = arrays

    # Setup batch and feature indices list for first mask.
    batch_indices, feature_indices = torch.nonzero(start_mask, as_tuple=True)
    feature_indices = [feature_indices]

    # Eat each additional mask.
    for arr in arrays:
        # Compute the needed rows to consider, and where these are nonzero for this mask.
        arr_rows = arr[batch_indices]
        new_batch_indices, new_feature_indices = torch.nonzero(arr_rows, as_tuple=True)

        # Update indices by acessing the old value relative to the batch indices just found.
        feature_indices = [f[new_batch_indices] for f in feature_indices]
        batch_indices = batch_indices[new_batch_indices]

        # Update feature list with next features
        feature_indices = *feature_indices, new_feature_indices

    return batch_indices, *feature_indices


class PeriodicPairIndexer(_PairIndexer):
    """
    Finds pairs in general periodic conditions.
    """
    def forward(self, coordinates, nonblank, real_atoms, inv_real_atoms, cells):
        
        original_coordinates = coordinates

        with torch.no_grad():
            cur_device = coordinates.device

            # Figure out how many molecules and atoms we have
            n_molecules, n_atoms, _ = coordinates.shape
            n_atoms_max_batch = nonblank.sum(dim=1).max()
            # Construct a unique index for each atom (including blanks here)
            atom_index = torch.arange(n_molecules * n_atoms, dtype=torch.long, device=cur_device)
            atom_index = atom_index.reshape(n_molecules, n_atoms)

            # Trim padding in this batch to minimal level
            nonblank = nonblank[:, :n_atoms_max_batch]
            coordinates = coordinates[:, :n_atoms_max_batch]
            atom_index = atom_index[:, :n_atoms_max_batch]

            inv_cell, coordinates, wrapped_offsets, nbounds = (
                wrap_systems_torch(coordinates, cells, self.hard_dist_cutoff)
            )
            # inv cell: n_sys, n_cartesian, n_basis
            # wraped_coords: nsys, n_atoms, 3 (coords)
            # wrapped_offsets: nsys, n_atoms, 3 (basis)
            # nbounds: (nsys, 3) : number of cells-lengths to search in each direction.

            # ## make combinator of images ## #
            a1, a2, a3 = nbounds.max(dim=0)[0].unbind()
            a1arr = torch.arange(-a1, a1+1, device=cur_device)
            a2arr = torch.arange(-a2, a2+1, device=cur_device)
            a3arr = torch.arange(-a3, a3+1, device=cur_device)
            combinator = torch.cartesian_prod(a1arr, a2arr, a3arr)
            # shape n_combos, 3
            # ## end make combinator ## #


            # ## Begin expensive part. ## #
            # In this zone we aggressively `del` things we don't need anymore
            # to allow pytorch to free memory. Delete anything that looks proportional
            # to the number of pairs. Many variables are proportional to the number of
            # 'possible' pairs when counting which atoms to compare and which image offsets
            # to consider.

            # First. we compute the indices of boolean factors that exclude
            # combinations of system,atom,atom,image based on padding
            # and image size for the corresponding cell.
            # Then a helper function makes the product of these boolean factors.
            # We call the product of these factors "nb" for "nonblank"
            # in that this extends the nonblank concept of
            # filtering pairs to the PBC neighbor finder.

            # Determine which images are required.
            # shape n_sys, aa1 = aa1 * n_sys,1 ## aa1 = 2*a1+1
            nbounds = nbounds.unsqueeze(2)
            in_x = (a1arr >= -nbounds[:, 0, :]) & (a1arr <= nbounds[:, 0, :])
            in_y = (a2arr >= -nbounds[:, 1, :]) & (a2arr <= nbounds[:, 1, :])
            in_z = (a3arr >= -nbounds[:, 2, :]) & (a3arr <= nbounds[:, 2, :])

            # Compute the product mask of all factors.
            # Note for future: We might be able to insert another factor to reduce pair-finding
            # costs in large boxes if we can additionally mask out the atoms
            # that are not near the edge of the box - requires careful testing.
            fancy_outs = union_indices_2d_masks(nonblank, nonblank, in_x, in_y, in_z)
            nb_sys, nb_p1, nb_p2, nb_x, nb_y, nb_z = fancy_outs
            # Note, maybe small optimization if we changed order to in_z, in_y, in_x,
            # due to common conventions about cells in datasets, however, this does change
            # the order of the neighbors.

            # Compute the index of the image in the combinator.
            nb_image = (nb_x*(2*a2+1)*(2*a3+1)) + (nb_y*(2*a3+1)) + nb_z

            # Now, we have the factors of system, atom1, atom2, and image number
            # required to find coordinate differences for pairs.
            # Next we simply compute the coordinate differences.

            # pair displacements without shifts
            pair_diffcoords = coordinates[nb_sys, nb_p1] - coordinates[nb_sys, nb_p2]

            # Find displacements offsets due to images.
            cell_shifts = torch.matmul(combinator.to(cells.dtype).unsqueeze(0), cells)
            # cell_shifts is  n_sys, n_perm, 3 (cartesian)
            dist_offsets = cell_shifts[nb_sys, nb_image]
            # Find pairs/images close enough to be within the cutoff
            pair_dist = (pair_diffcoords + dist_offsets).norm(dim=1)
            del dist_offsets, pair_diffcoords

            # Calculate mask of pairs that map to same atom.
            int_offsets = combinator[nb_image]
            nonself_pairs = torch.logical_or(int_offsets.to(torch.bool).any(dim=1), torch.ne(nb_p1, nb_p2))

            # Compute indices of considered pairs relative to MolAtom format.
            pair_first = atom_index[nb_sys, nb_p1]
            del nb_p1
            pair_second = atom_index[nb_sys, nb_p2]
            del nb_p2

            # Compute close pairs.
            close_pairs = (pair_dist < self.hard_dist_cutoff) & nonself_pairs  # get rid of self-pairs
            del nonself_pairs, pair_dist
            close_pairs = torch.nonzero(close_pairs)[:, 0]
            # close_pairs holds the indices of pairs

            # Extract the needed data within the cutoff.
            pair_mol = nb_sys[close_pairs]
            del nb_sys
            pair_first = pair_first[close_pairs]
            pair_second = pair_second[close_pairs]
            cell_only_offsets = int_offsets[close_pairs]  # Offset due to cell images only.
            offset_num = nb_image[close_pairs]  # index of where to store cell_offset values if caching pairs.
            del int_offsets, close_pairs
            # ## End expensive part ## #

            # This converts the atom index from an index that counts blank atoms (MolAtom) to one that doesn't;
            # Afterwards it indexes the flat array of atoms in the batch. (Atoms format)
            pair_first = inv_real_atoms[pair_first]
            pair_second = inv_real_atoms[pair_second]

            # Add back in offsets due to atom wrapping.
            atom_wrap_offsets = wrapped_offsets[nonblank]
            pf_shift = atom_wrap_offsets[pair_first]  # Offset of wrapping of first atom
            ps_shift = atom_wrap_offsets[pair_second]  # Offset of wrapping of second atom

            cell_offsets = cell_only_offsets - (pf_shift - ps_shift)  # Total image offset of pairs.
            del cell_only_offsets, pf_shift, ps_shift

        # Compute distance differentiably from total image offsets and pair indices
        pair_shifts = torch.matmul(cell_offsets.unsqueeze(1).to(cells.dtype), cells[pair_mol]).squeeze(1)
        coordflat = original_coordinates.reshape(n_molecules * n_atoms, 3)[real_atoms]
        paircoord = coordflat[pair_first] - coordflat[pair_second] + pair_shifts
        distflat2 = paircoord.norm(dim=1)

        return distflat2, pair_first, pair_second, paircoord, cell_offsets, offset_num, pair_mol
    
class PeriodicPairIndexerMemory(PairMemory):
    '''
    Implementation of PeriodicPairIndexer with additional memory component.

    Stores current pair indices in memory and reuses them to compute the pair distances if no 
    particle has moved more than skin/2 since last pair calculation. Otherwise uses the
    _pair_indexer_class to recompute the pairs.

    Increasing the value of 'skin' will increase the number of pair distances computed at
    each step, but decrease the number of times new pairs must be computed. Skin should be 
    set to zero while training for fastest results.
    '''
    _pair_indexer_class = PeriodicPairIndexer

    def forward(self, coordinates, nonblank, real_atoms, inv_real_atoms, cells):
        if self.recalculation_needed(coordinates, cells):
            self.n_molecules, self.n_atoms, _ = coordinates.shape
            self.recalculations += 1

            inputs = (coordinates, nonblank, real_atoms, inv_real_atoms, cells)
            outputs = self._pair_indexer(*inputs)
            distflat, pair_first, pair_second, paircoord, cell_offsets, offset_num, pair_mol = outputs

            for name, var in [
                ("cell_offsets", cell_offsets),
                ("pair_first", pair_first),
                ("pair_second", pair_second),
                ("offset_num", offset_num),
                ("positions", coordinates),
                ("cells", cells),
                ("pair_mol", pair_mol)
                ]:
                self.__setattr__(name, var)

        else:
            self.reuses += 1
            pair_shifts = torch.matmul(self.cell_offsets.unsqueeze(1).to(cells.dtype), cells[self.pair_mol]).squeeze(1)
            coordflat = coordinates.reshape(self.n_molecules * self.n_atoms, 3)[real_atoms]
            paircoord = coordflat[self.pair_first] - coordflat[self.pair_second] + pair_shifts
            distflat = paircoord.norm(dim=1)

        # We filter the lists to only send forward relevant pairs (those with distance under cutoff), improving performance.   
        return filter_pairs(self.hard_dist_cutoff, distflat, self.pair_first, self.pair_second, paircoord, self.cell_offsets, self.offset_num)
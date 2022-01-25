import torch

from .open import _PairIndexer


class PeriodicPairIndexer(_PairIndexer):
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

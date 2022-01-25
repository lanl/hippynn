import torch


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

            atom_index = torch.reshape(
                torch.arange(n_molecules * n_atoms, dtype=torch.long, device=cur_device), (n_molecules, n_atoms)
            )
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
            nonself_atoms = (~torch.eye(namax_box, dtype=nonblank.dtype, device=cur_device)).unsqueeze(0)
            # Only take pairs that are close enough
            if self.hard_dist_cutoff is not None:
                close_atoms = pair_dists < self.hard_dist_cutoff
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

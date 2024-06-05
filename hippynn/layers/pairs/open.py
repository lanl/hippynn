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

class PairMemory(torch.nn.Module):
    '''
    Stores current pair indices and reuses them to compute the pair distances if no 
    particle has moved more than skin/2 since last pair calculation. Otherwise uses the
    _pair_indexer_class to recompute the pairs.

    Increasing the value of 'skin' will increase the number of pair distances computed at
    each step, but decrease the number of times new pairs must be computed. Skin should be 
    set to zero while training for fastest results.
    '''

    # TODO: Adapt to work system-by-system

    # ## Subclasses should update the following ## #
    _pair_indexer_class = NotImplemented

    def forward(*args, **kwargs):
        return NotImplementedError
    # ## End ## #

    def __init__(self, skin, dist_hard_max=None, hard_dist_cutoff=None):
        super().__init__()

        if dist_hard_max is None and hard_dist_cutoff is None:
            raise ValueError("One of 'dist_hard_max' and 'hard_dist_cutoff' must be specified.")
        if dist_hard_max is not None and hard_dist_cutoff is not None and dist_hard_max != hard_dist_cutoff:
            raise ValueError("Must only specify one of 'dist_hard_max' and 'hard_dist_cutoff.'")
        
        self.hard_dist_cutoff = (dist_hard_max or hard_dist_cutoff)
        self.dist_hard_max = (dist_hard_max or hard_dist_cutoff)
        self.set_skin(skin)

    @property
    def skin(self):
        return self._skin
    
    def set_skin(self, skin):
        self._skin = skin

        try:
            self._pair_indexer = self._pair_indexer_class(hard_dist_cutoff = self._skin + self.hard_dist_cutoff)
        except TypeError:
            self._pair_indexer = self._pair_indexer_class(dist_hard_max = self._skin + self.hard_dist_cutoff)

        self.reset_reuse_percentage()
        self.initialize_buffers()

    @skin.setter
    def skin(self, skin):
        self.set_skin(skin)
        
    @property
    def reuse_percentage(self):
        '''
        Returns None if there are no model calls on record.
        '''
        try:
            return self.reuses / (self.reuses + self.recalculations) * 100
        except ZeroDivisionError:
            print("No model calls on record.")
            return

    def reset_reuse_percentage(self):
        self.reuses = 0
        self.recalculations = 0
        
    def initialize_buffers(self):
        for name in ["pair_mol", "cell_offsets", "pair_first", "pair_second", "offset_num", "positions", "cells"]:
            self.register_buffer(name=name, tensor=None, persistent=False)

    def recalculation_needed(self, coordinates, cells):
        if coordinates.shape[0] != 1:  # does not support batch size larger than 1
            return True
        if self.positions is None:  # ie. forward function has not been called
            return True
        if self.skin == 0:
            return True
        if (self.cells != cells).any() or (((self.positions - coordinates)**2).sum(1).max() > (self._skin/2)**2):
            return True
        return False
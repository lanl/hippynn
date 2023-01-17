
import torch

from .open import _PairIndexer

class FilterDistance(_PairIndexer):
    """ Filters a list of tensors in *pair_lists by distance. 
    pair_dist is first positional argument.

    :param _PairIndexer: FilterDistance subclasses _PairIndexer so that the
    FilterPairIndexers behave as regular PairIndexers. 
    """
    
    def forward(self, *pair_lists):
        r_cut = self.hard_dist_cutoff 
        idx = torch.argwhere(pair_lists[0] <= r_cut)[:,0]

        return tuple(pl[idx] for pl in pair_lists)



import torch

from .open import _PairIndexer

class FilterDistance(_PairIndexer):
    """
    Filters a list of tensors in pair_tensors by distance.
    pair_dist is first positional argument.

    FilterDistance subclasses _PairIndexer so that the
    FilterPairIndexers behave as regular PairIndexers. 
    """
    
    def forward(self, pair_dist, *pair_tensors):
        r_cut = self.hard_dist_cutoff 
        idx = torch.argwhere(pair_dist <= r_cut).squeeze(1)
        pair_tensors = pair_dist, *pair_tensors
        return tuple(pl[idx] for pl in pair_tensors)


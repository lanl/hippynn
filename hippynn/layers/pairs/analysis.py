"""
Modules for analyzing pair-valued data
"""
import torch


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
        centers = (self.bins[1:] + self.bins[:-1]) / 2
        widths = self.bins[1:] - self.bins[:-1]
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
        return (rdf / n_molecules).unsqueeze(0)

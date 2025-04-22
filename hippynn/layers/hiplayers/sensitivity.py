"""
Layers for HIP-NN
"""
import numpy as np
import torch
import warnings

from ... import settings


def warn_if_under(distance, threshold):
    if len(distance) == 0:  # no pairs
        return
    dmin = distance.min()
    if dmin < threshold:
        d_count = distance < threshold
        d_frac = d_count.to(distance.dtype).mean()
        d_sum = (d_count.sum() / 2).to(torch.int)
        warnings.warn(
            "Provided distances are underneath sensitivity range!\n"
            f"Minimum distance in current batch: {dmin}\n"
            f"Threshold distance for warning: {threshold}.\n"
            f"Fraction of pairs under the threshold: {d_frac}\n"
            f"Number of pairs under the threshold: {d_sum}"
        )


class CosCutoff(torch.nn.Module):
    def __init__(self, hard_max_dist):
        super().__init__()
        self.hard_max_dist = hard_max_dist

    def forward(self, dist_tensor):
        cutoff_sense = torch.cos(np.pi / 2 * dist_tensor / self.hard_max_dist) ** 2
        cutoff_sense = cutoff_sense * (dist_tensor <= self.hard_max_dist).to(cutoff_sense.dtype)
        return cutoff_sense


class SensitivityModule(torch.nn.Module):
    def __init__(self, hard_max_dist, cutoff_type):
        super().__init__()
        self.cutoff = cutoff_type(hard_max_dist)
        self.hard_max_dist = hard_max_dist


class GaussianSensitivityModule(SensitivityModule):
    def __init__(self, n_dist, min_dist_soft, max_dist_soft, hard_max_dist, cutoff_type=CosCutoff):

        super().__init__(hard_max_dist, cutoff_type)
        init_mu = 1.0 / torch.linspace(1.0 / max_dist_soft, 1.0 / min_dist_soft, n_dist)
        self.mu = torch.nn.Parameter(init_mu.unsqueeze(0))

        self.sigma = torch.nn.Parameter(torch.Tensor(n_dist).unsqueeze(0))
        init_sigma = min_dist_soft * 2 * n_dist  # pulled from theano code
        self.sigma.data.fill_(init_sigma)

    def forward(self, distflat, warn_low_distances=None):
        if warn_low_distances is None:
            warn_low_distances = settings.WARN_LOW_DISTANCES
        if warn_low_distances:
            with torch.no_grad():
                mu, argmin = self.mu.min(dim=1)
                sig = self.sigma[:, argmin]
                # Warn if distance is less than the -inside- edge of the shortest sensitivity function
                thresh = mu + sig
                warn_if_under(distflat, thresh)
        distflat_ds = distflat.unsqueeze(1)
        mu_ds = self.mu
        sig_ds = self.sigma

        nondim = (distflat_ds**-1 - mu_ds**-1) ** 2 / (sig_ds**-2)
        base_sense = torch.exp(-0.5 * nondim)

        total_sense = base_sense * self.cutoff(distflat).unsqueeze(1)
        return total_sense


class InverseSensitivityModule(SensitivityModule):
    def __init__(self, n_dist, min_dist_soft, max_dist_soft, hard_max_dist, cutoff_type=CosCutoff):

        super().__init__(hard_max_dist, cutoff_type)
        init_mu = torch.Tensor(1.0 / torch.linspace(1.0 / max_dist_soft, 1.0 / min_dist_soft, n_dist))
        self.mu = torch.nn.Parameter(init_mu.unsqueeze(0))
        self.sigma = torch.nn.Parameter(torch.Tensor(n_dist).unsqueeze(0))
        init_sigma = min_dist_soft * 2 * n_dist
        self.sigma.data.fill_(init_sigma)

    def forward(self, distflat, warn_low_distances=None):
        if warn_low_distances is None:
            warn_low_distances = settings.WARN_LOW_DISTANCES
        if warn_low_distances:
            with torch.no_grad():
                # Warn if distance is less than the -inside- edge of the shortest sensitivity function
                mu, argmin = self.mu.min(dim=1)
                sig = self.sigma[:, argmin]
                thresh = (mu**-1 - sig**-1) ** -1

                warn_if_under(distflat, thresh)
        distflat_ds = distflat.unsqueeze(1)

        nondim = (distflat_ds**-1 - self.mu**-1) ** 2 / (self.sigma**-2)
        base_sense = torch.exp(-0.5 * nondim)

        total_sense = base_sense * self.cutoff(distflat).unsqueeze(1)

        return total_sense


class SensitivityBottleneck(torch.nn.Module):
    def __init__(
        self,
        n_dist,
        min_soft_dist,
        max_dist_soft,
        hard_max_dist,
        n_dist_bare,
        cutoff_type=CosCutoff,
        base_sense=InverseSensitivityModule,
    ):
        super().__init__()
        self.hard_max_dist = hard_max_dist

        self.base_sense = base_sense(n_dist_bare, min_soft_dist, max_dist_soft, hard_max_dist, cutoff_type)
        self.matching = torch.nn.Parameter(torch.Tensor(n_dist_bare, n_dist))

        self.cutoff = self.base_sense.cutoff

        torch.nn.init.orthogonal_(self.matching.data)

    def forward(self, distflat):
        base_sense = self.base_sense(distflat)
        reduced_sense = torch.mm(base_sense, self.matching)
        return reduced_sense

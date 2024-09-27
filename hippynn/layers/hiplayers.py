"""
Layers for HIP-NN
"""
import numpy as np
import torch
import warnings
from .. import custom_kernels

from .. import settings


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


class InteractLayer(torch.nn.Module):
    """
    Hipnn's interaction layer
    """

    def __init__(self, nf_in, nf_out, n_dist, mind_soft, maxd_soft, hard_cutoff, sensitivity_module, cusp_reg=None):
        """
        Constructor

        :param nf_in: number of input features
        :param nf_out: number of output features
        :param n_dist: number of distance sensitivities
        :param mind_soft: minimum distance for initial sensitivities
        :param maxd_soft: maximum distance for initial sensitivities
        :param hard_cutoff: maximum distance for cutoff function
        :param sensitivity_module: class or callable that builds sensitivity functions, should return nn.Module
        :param cusp_reg: ignored, only provided with compatibility for tensor sensitivity API
        """
        super().__init__()

        if type(self) is InteractLayer and cusp_reg is not None:
            # Parameter is not used in this class.
            warnings.warn(f"Parameter `cusp_reg`={cusp_reg} is ignored in this class, and is only provided for API compatibility.")
        self.n_dist = n_dist
        self.nf_in = nf_in
        self.nf_out = nf_out

        # Sensitivity module
        self.sensitivity = sensitivity_module(n_dist, mind_soft, maxd_soft, hard_cutoff)

        # Interaction weights
        self.int_weights = torch.nn.Parameter(torch.Tensor(n_dist, nf_out, nf_in))
        torch.nn.init.xavier_normal_(self.int_weights.data)

        # Self-term and bias
        self.selfint = torch.nn.Linear(nf_in, nf_out)  # includes bias term and self-interactions
        torch.nn.init.xavier_normal_(self.selfint.weight.data)

    def regularization_params(self):
        return [self.int_weights, self.selfint.weight]

    def forward(self, in_features, pair_first, pair_second, dist_pairs):
        """
        Pytorch Enforced Forward function

        :param in_features:
        :param pair_first:
        :param pair_second:
        :param dist_pairs:
        :return: Interaction output features
        """

        # Z' = (VSZ) + (WZ) + b
        n_atoms_real = in_features.shape[0]
        sense_vals = self.sensitivity(dist_pairs)
        # For HIPNN equation, the interaction term is VSZ, which we evaluate as V(E) where E=(SZ)
        # V: interaction weights
        # S: sensitivities
        # Z: input features
        # E: environment features (S*Z)

        # Q = (VZ) #  torch.mm
        # E = (QS) #  custom_kernels.featsum

        # E = (SZ)
        env_features = custom_kernels.envsum(sense_vals, in_features, pair_first, pair_second)

        # (VSZ)
        env_features = torch.reshape(env_features, (n_atoms_real, self.n_dist * self.nf_in))
        # The weight permutation can be completely eliminated by reshaping the initialization
        weights_rs = torch.reshape(self.int_weights.permute(0, 2, 1), (self.n_dist * self.nf_in, self.nf_out))
        # Multiply the environment of each atom by weights
        features_out = torch.mm(env_features, weights_rs)

        # WZ + B
        features_out_selfpart = self.selfint(in_features)

        # VSZ + WZ + B
        features_out_total = features_out + features_out_selfpart

        return features_out_total


class InteractLayerVec(InteractLayer):
    def __init__(self, nf_in, nf_out, n_dist, mind_soft, maxd_soft, hard_cutoff, sensitivity_module, cusp_reg):
        super().__init__(nf_in, nf_out, n_dist, mind_soft, maxd_soft, hard_cutoff, sensitivity_module, cusp_reg)
        self.vecscales = torch.nn.Parameter(torch.Tensor(nf_out))
        torch.nn.init.normal_(self.vecscales.data)
        self.cusp_reg = cusp_reg

    def __setstate__(self, state):
        output = super().__setstate__(state)
        if not hasattr(self, "cusp_reg"):
            # The layer was created before the cusp regularization was a parameter.
            # Add a patch that if a state dict is loaded in with no cusp parameter,
            # use the pre-introduction static value.
            warnings.warn(
                "Loading a module which does not contain the 'cusp_reg' parameter. "
                "In the future, this behavior will cause an error. "
                "To avoid this warning, re-save this model to disk. "
            )
            self.handle = self.register_load_state_dict_post_hook(self.compatibility_hook)
        return output

    @staticmethod
    def compatibility_hook(self, incompatible_keys):
        missing = incompatible_keys.missing_keys
        if not missing:
            # No need for compatibility!
            return

        if len(missing) != 1:
            warnings.warn("Backwards compatibility hook may have failed due to the presence of multiple missing keys!")
            return

        for m in missing:
            if m.endswith("_extra_state"):
                break
        else:
            # Python reminder: The mysterious "else" clause of the for loop
            # activates when python does not break out of the for loop.
            return  # No _extra_state type variable was missing: just return.

        DEPRECATED_CUSP_REG = 1e-30
        warnings.warn(
            f"Loaded state does not contain 'cusp_reg' parameter. "
            f"Using deprecated value of 1e-30. "
            f"This compatibility behavior will be removed in the future. "
            f"To avoid this warning, re-save this model."
        )
        self.set_extra_state({"cusp_reg": DEPRECATED_CUSP_REG})
        missing.remove(m)

    def get_extra_state(self):
        return {"cusp_reg": self.cusp_reg}

    def set_extra_state(self, state):
        self.cusp_reg = state["cusp_reg"]

    def forward(self, in_features, pair_first, pair_second, dist_pairs, coord_pairs):

        n_atoms_real = in_features.shape[0]
        sense_vals = self.sensitivity(dist_pairs)

        # Sensitivity stacking
        sense_vec = sense_vals.unsqueeze(1) * (coord_pairs / dist_pairs.unsqueeze(1)).unsqueeze(2)
        sense_vec = sense_vec.reshape(-1, self.n_dist * 3)
        sense_stacked = torch.concatenate([sense_vals, sense_vec], dim=1)

        # Message passing, stack sensitivities to coalesce custom kernel call.
        # shape (n_atoms, n_nu + 3*n_nu, n_feat)
        env_features_stacked = custom_kernels.envsum(sense_stacked, in_features, pair_first, pair_second)
        # shape (n_atoms, 4, n_nu, n_feat)
        env_features_stacked = env_features_stacked.reshape(-1, 4, self.n_dist, self.nf_in)

        # separate to tensor components
        env_features, env_features_vec = torch.split(env_features_stacked, [1, 3], dim=1)

        # Scalar part
        env_features = torch.reshape(env_features, (n_atoms_real, self.n_dist * self.nf_in))
        weights_rs = torch.reshape(self.int_weights.permute(0, 2, 1), (self.n_dist * self.nf_in, self.nf_out))
        features_out = torch.mm(env_features, weights_rs)

        # Vector part
        env_features_vec = env_features_vec.reshape(n_atoms_real * 3, self.n_dist * self.nf_in)
        features_out_vec = torch.mm(env_features_vec, weights_rs)
        features_out_vec = features_out_vec.reshape(n_atoms_real, 3, self.nf_out)
        features_out_vec = torch.square(features_out_vec).sum(dim=1) + self.cusp_reg
        features_out_vec = torch.sqrt(features_out_vec)
        features_out_vec = features_out_vec * self.vecscales.unsqueeze(0)

        # Self interaction
        features_out_selfpart = self.selfint(in_features)

        features_out_total = features_out + features_out_vec + features_out_selfpart

        return features_out_total


class InteractLayerQuad(InteractLayerVec):
    def __init__(self, nf_in, nf_out, n_dist, mind_soft, maxd_soft, hard_cutoff, sensitivity_module, cusp_reg):
        super().__init__(nf_in, nf_out, n_dist, mind_soft, maxd_soft, hard_cutoff, sensitivity_module, cusp_reg)
        self.quadscales = torch.nn.Parameter(torch.Tensor(nf_out))
        torch.nn.init.normal_(self.quadscales.data)
        # upper indices of flattened 3x3 array minus the (3,3) component
        # which is not needed for a traceless tensor
        upper_ind = torch.as_tensor([0, 1, 2, 4, 5], dtype=torch.int64)
        self.register_buffer("upper_ind", upper_ind, persistent=False)  # Static, not part of module state

    def forward(self, in_features, pair_first, pair_second, dist_pairs, coord_pairs):

        n_atoms_real = in_features.shape[0]
        sense_vals = self.sensitivity(dist_pairs)

        ####
        # Sensitivity calculations
        # scalar: sense_vals
        # vector: sense_vec
        # quadrupole: sense_quad
        rhats = coord_pairs / dist_pairs.unsqueeze(1)
        sense_vec = sense_vals.unsqueeze(1) * rhats.unsqueeze(2)
        sense_vec = sense_vec.reshape(-1, self.n_dist * 3)
        rhatsquad = rhats.unsqueeze(1) * rhats.unsqueeze(2)
        rhatsquad = (rhatsquad + rhatsquad.transpose(1, 2)) / 2
        tr = torch.diagonal(rhatsquad, dim1=1, dim2=2).sum(dim=1) / 3.0  # Add divide by 3 early to save flops
        tr = tr.unsqueeze(1).unsqueeze(2) * torch.eye(3, dtype=tr.dtype, device=tr.device).unsqueeze(0)
        rhatsquad = rhatsquad - tr
        rhatsqflat = rhatsquad.reshape(-1, 9)[:, self.upper_ind]  # Upper-diagonal part
        sense_quad = sense_vals.unsqueeze(1) * rhatsqflat.unsqueeze(2)
        sense_quad = sense_quad.reshape(-1, self.n_dist * 5)
        sense_stacked = torch.concatenate([sense_vals, sense_vec, sense_quad], dim=1)

        # Message passing, stack sensitivities to coalesce custom kernel call.
        # shape (n_atoms, n_nu + 3*n_nu + 5*n_nu, n_feat)
        env_features_stacked = custom_kernels.envsum(sense_stacked, in_features, pair_first, pair_second)
        # shape (n_atoms, 9, n_nu, n_feat)
        env_features_stacked = env_features_stacked.reshape(-1, 9, self.n_dist, self.nf_in)

        # separate to tensor components
        env_features, env_features_vec, env_features_quad = torch.split(env_features_stacked, [1, 3, 5], dim=1)

        # Scalar stuff.
        env_features = torch.reshape(env_features, (n_atoms_real, self.n_dist * self.nf_in))
        weights_rs = torch.reshape(self.int_weights.permute(0, 2, 1), (self.n_dist * self.nf_in, self.nf_out))
        features_out = torch.mm(env_features, weights_rs)

        # Vector part
        # Sensitivity
        # Weights
        env_features_vec = env_features_vec.reshape(n_atoms_real * 3, self.n_dist * self.nf_in)
        features_out_vec = torch.mm(env_features_vec, weights_rs)
        # Norm and scale
        features_out_vec = features_out_vec.reshape(n_atoms_real, 3, self.nf_out)
        features_out_vec = torch.square(features_out_vec).sum(dim=1) + self.cusp_reg
        features_out_vec = torch.sqrt(features_out_vec)
        features_out_vec = features_out_vec * self.vecscales.unsqueeze(0)

        # Quadrupole part
        # Sensitivity
        # Weights
        env_features_quad = env_features_quad.reshape(n_atoms_real * 5, self.n_dist * self.nf_in)
        features_out_quad = torch.mm(env_features_quad, weights_rs)  ##sum v b
        features_out_quad = features_out_quad.reshape(n_atoms_real, 5, self.nf_out)
        # Norm. (of traceless two-tensor from 5 component representation)
        quadfirst = torch.square(features_out_quad).sum(dim=1)
        quadsecond = features_out_quad[:, 0, :] * features_out_quad[:, 3, :]
        features_out_quad = 2 * (quadfirst + quadsecond)
        features_out_quad = torch.sqrt(features_out_quad + self.cusp_reg)
        # Scales
        features_out_quad = features_out_quad * self.quadscales.unsqueeze(0)

        # Combine
        features_out_selfpart = self.selfint(in_features)

        features_out_total = features_out + features_out_vec + features_out_quad + features_out_selfpart

        return features_out_total

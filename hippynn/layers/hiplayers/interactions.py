import torch
from ... import custom_kernels
from .tensors import HopInvariantLayer
import warnings


class InteractLayer(torch.nn.Module):
    """
    Hipnn's interaction layer
    """

    def __init__(self, nf_in, nf_out, n_dist, mind_soft, maxd_soft, hard_cutoff, sensitivity_module):
        """
        Constructor

        :param nf_in: number of input features
        :param nf_out: number of output features
        :param n_dist: number of distance sensitivities
        :param mind_soft: minimum distance for initial sensitivities
        :param maxd_soft: maximum distance for initial sensitivities
        :param hard_cutoff: maximum distance for cutoff function
        :param sensitivity_module: class or callable that builds sensitivity functions, should return nn.Module
        """
        super().__init__()

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
        super().__init__(nf_in, nf_out, n_dist, mind_soft, maxd_soft, hard_cutoff, sensitivity_module)
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


# n_max, l_max: warning counts for invariants.
_invariant_counts = {
    (4, 3): 13,
    (4, 2): 6,
    (4, 1): 2,  # Similar to HIP-NN-TS
    (4, 0): 1,  # Quasi-redundant with HIP-NN
    (3, 3): 7,
    (3, 2): 5,
    (3, 1): 2,  # Similar to HIP-NN-TS
    (3, 0): 1,  # Quasi-redundant with HIP-NN
    (2, 3): 4,  # Similar to HIP-NN-TS
    (2, 2): 3,  # Similar to HIP-NN-TS
    (2, 1): 2,  # Similar to HIP-NN-TS
    (2, 0): 1,  # Quasi-redundant with HIP-NN
    (1, 3): 1,  # Quasi-redundant with HIP-NN
    (1, 2): 1,  # Quasi-redundant with HIP-NN
    (1, 1): 1,  # Quasi-redundant with HIP-NN
    (1, 0): 1,  # Quasi-redundant with HIP-NN
}


class HOPInteractionLayer(InteractLayer):
    def __init__(self, *args, n_max, l_max, group_norm, group_norm_eps, **kwargs):
        super().__init__(*args, **kwargs)

        if l_max < 0:
            raise ValueError(f"{l_max=} must be a non-negative integer.")

        if n_max <= 0:
            raise ValueError(f"{n_max=} must be a positive integer.")
        elif n_max == 1:
            if l_max > 0:
                warnings.warn(f"If variable n_max==1, l_max>0 is unneeded. ({n_max=},{l_max=})")
        elif n_max > 1:
            if l_max == 0:
                warnings.warn(f"If variable n_max>1, l_max>1 is required for" f" non-trivial many-body interactions. ({n_max=},{l_max=})")
            if n_max > 2 and l_max == 1:
                warnings.warn(f"If variable l_max==1, n_max>2 is redundant. ({n_max=},{l_max=})")

        try:
            n_invariants = _invariant_counts[n_max, l_max]
        except KeyError:
            raise ValueError(f"HIP-HOP parameters {l_max=},{n_max=} implementation not presently available.")

        if n_invariants == 1:
            warnings.warn(
                f"Number of invariants is only 1 for HIP-HOP with ({n_max=},{l_max=}); for these settings"
                f" it may be preferable to use vanilla HIP-NN."
            )

        self.n_invariants = n_invariants
        mixing_weights = torch.zeros(self.nf_out, self.n_invariants, self.nf_out)
        self.invars = HopInvariantLayer(n_max=n_max, l_max=l_max)
        self.mixing_weights = torch.nn.Parameter(mixing_weights)
        torch.nn.init.xavier_normal_(self.mixing_weights)
        if group_norm:
            self.group_norm = torch.nn.GroupNorm(self.n_invariants, self.n_invariants * self.nf_out, eps=group_norm_eps, affine=True)
        else:
            self.group_norm = None

    def forward(self, in_features, pair_first, pair_second, dist_pairs, tensor_rhats):

        features_out_selfpart = self.selfint(in_features)

        n_atoms_real = in_features.shape[0]
        n_pair, n_tensor_comp = tensor_rhats.shape

        # set up sensitivity for message passing
        sense_scalar = self.sensitivity(dist_pairs)
        sensitivity = sense_scalar.unsqueeze(1) * tensor_rhats.unsqueeze(2)
        sense_flat = sensitivity.reshape(n_pair, n_tensor_comp * self.n_dist)

        env_features = custom_kernels.envsum(sense_flat, in_features, pair_first, pair_second)

        # apply weights to tensor features
        weights_rs = torch.reshape(self.int_weights.permute(0, 2, 1), (self.n_dist * self.nf_in, self.nf_out))
        env_rs = env_features.reshape(n_atoms_real * n_tensor_comp, self.n_dist * self.nf_in)
        tensor_features = torch.mm(env_rs, weights_rs)
        tensor_features = tensor_features.reshape(n_atoms_real, n_tensor_comp, self.nf_out)

        # move tensor features to last dimension and compute invariants
        # shape n_atom, n_feat, n_tensor
        tensor_features = tensor_features.permute(0, 2, 1).reshape(n_atoms_real * self.nf_out, n_tensor_comp)

        invariants = self.invars(tensor_features)
        invariants = invariants.reshape(n_atoms_real, self.nf_out, self.n_invariants)

        if self.group_norm:
            # Group norm operates on n_batch, n_groups*n_features_per_group,
            # so the group index (invariant index) should come first.
            invariants = invariants.permute(0, 2, 1).reshape(n_atoms_real, self.n_invariants * self.nf_out)
            normalized_invariants = self.group_norm(invariants)
            # Restore shape/order; Put invariants last again.
            normalized_invariants = normalized_invariants.reshape(n_atoms_real, self.n_invariants, self.nf_out)
            normalized_invariants = normalized_invariants.permute(0, 2, 1)
        else:
            normalized_invariants = invariants

        normalized_invariants = normalized_invariants.reshape(n_atoms_real, self.nf_out * self.n_invariants)

        # (n_a,n_f*n_i) @ (n_f*n_i,n_f) -> (n_a, n_f)
        mixing_features = normalized_invariants @ self.mixing_weights.reshape(-1, self.nf_out)

        total_out = mixing_features + features_out_selfpart

        return total_out

import torch

"""
This module implements pure PyTorch implementations of the message passing step for both HIP-NN-TS and HIP-NN-TS.

We do not implement the message passing step for vanilla HIP-NN because there is currently only one implementation of its
message passing step, so it is hard-coded into the model's definition.
"""

def torchMessagePassingHop(T,s,z,pair_first,pair_second):
    """
    PyTorch implementation of the message passing step for HIP-HOP-NN.

    :param T: Matrix. Each row represents a pair of neighbors. Each row stores the entries of all of
              irreducible tensors (of each order being used) of the delta function defined on S^2 which is
              nonzero in the direction from the first atom in the pair to the second. The tensors are stored as
              lists of basis coefficients and concatenated into a single row of T.

    :param s: Matrix. Each row corresponds to a pair of neighbors. Each row stores the outputs of the sensitivity
              function applied to the corresponding pair.

    :param z: Matrix. Each row corresponds to an atom and stores the features associated with that atom.

    :param pair_first: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the first atom in each pair.

    :param pair_second: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the second atom in each pair.

    :return: A 3D tensor storing the result of message passing. The first dimension indexes each individual atom. The second dimension
             indexes tensor components and sensitivities (sensitivities varies vaster). The third dimension indexes features.
    """
    # delayed import of envsum to prevent a circular dependency between __init__.py and this file.
    from . import envsum

    ij,t = T.shape
    _,nu = s.shape
    sensitivity = s.unsqueeze(1) * T.unsqueeze(2)
    sense_flat = sensitivity.reshape(ij, t * nu)
    return envsum(sense_flat, z, pair_first, pair_second)

def torchMessagePassingVec(in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs):
    """
    PyTorch implementation of the message passing step for HIP-NN-TS if l_max is equal to 1.

    :param in_features: Matrix. Each row corresponds to an atom and stores the features associated with that atom.

    :param sense_vals: Matrix. Each row corresponds to a pair of neighbors. Each row stores the outputs of the sensitivity
                       function applied to the corresponding pair.

    :param pair_first: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the first atom in each pair.

    :param pair_second: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the second atom in each pair.

    :param dist_pairs: Vector. Each entry stores the distance between a pair of neighbors.

    :param coord_pairs: Matrix. Each row corresponds to a pair of neighbors. Each row stores the position offset (as a vector) from the first atom
                        to the second.

    :return: The output of message passing for HIP-NN-TS.
    """

    # delayed import of envsum to prevent a circular dependency between __init__.py and this file.
    from . import envsum

    _, nu = sense_vals.shape
    sense_vec = sense_vals.unsqueeze(1) * (coord_pairs / dist_pairs.unsqueeze(1)).unsqueeze(2)
    sense_vec = sense_vec.reshape(-1, nu * 3)
    sense_stacked = torch.concatenate([sense_vals, sense_vec], dim=1)
    return envsum(sense_stacked, in_features, pair_first, pair_second)

def torchMessagePassingQuad(in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs, upper_ind):
    """
    PyTorch implementation of the message passing step for HIP-NN-TS if l_max is equal to 2.

    :param in_features: Matrix. Each row corresponds to an atom and stores the features associated with that atom.

    :param sense_vals: Matrix. Each row corresponds to a pair of neighbors. Each row stores the outputs of the sensitivity
                       function applied to the corresponding pair.

    :param pair_first: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the first atom in each pair.

    :param pair_second: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the second atom in each pair.

    :param dist_pairs: Vector. Each entry stores the distance between a pair of neighbors.

    :param coord_pairs: Matrix. Each row corresponds to a pair of neighbors. Each row stores the position offset (as a vector) from the first atom
                        to the second.

    :return: The output of message passing for HIP-NN-TS.
    """

    # delayed import of envsum to prevent a circular dependency between __init__.py and this file.
    from . import envsum

    _, nu = sense_vals.shape

    rhats = coord_pairs / dist_pairs.unsqueeze(1)
    sense_vec = sense_vals.unsqueeze(1) * rhats.unsqueeze(2)
    sense_vec = sense_vec.reshape(-1, nu * 3)
    rhatsquad = rhats.unsqueeze(1) * rhats.unsqueeze(2)
    rhatsquad = (rhatsquad + rhatsquad.transpose(1, 2)) / 2
    tr = torch.diagonal(rhatsquad, dim1=1, dim2=2).sum(dim=1) / 3.0  # Add divide by 3 early to save flops
    tr = tr.unsqueeze(1).unsqueeze(2) * torch.eye(3, dtype=tr.dtype, device=tr.device).unsqueeze(0)
    rhatsquad = rhatsquad - tr
    rhatsqflat = rhatsquad.reshape(-1, 9)[:, upper_ind]  # Upper-diagonal part
    sense_quad = sense_vals.unsqueeze(1) * rhatsqflat.unsqueeze(2)
    sense_quad = sense_quad.reshape(-1, nu * 5)
    sense_stacked = torch.concatenate([sense_vals, sense_vec, sense_quad], dim=1)

    # Message passing, stack sensitivities to coalesce custom kernel call.
    # shape (n_atoms, n_nu + 3*n_nu + 5*n_nu, n_feat)
    return envsum(sense_stacked, in_features, pair_first, pair_second)
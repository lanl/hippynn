"""
Pure pytorch implementation of envsum operations.

Note that these implementations are usually replaced by custom cuda kernels,
as explained in the :doc:`/user_guide/ckernels` section of the documentation.

"""
import torch
from torch import Tensor


def envsum(sensitivities: Tensor, features: Tensor, pair_first: Tensor, pair_second: Tensor) -> Tensor:
    """
    Computes outer product of sensitivities of pairs and atom features from pair_second,
    whilst accumulating them onto indices pair_first.

    See the :doc:`/user_guide/ckernels` section of the documentation for more information.

    :param sensitivities: (n_pairs, n_sensitivities) floating point tensor
    :param features: (n_atoms, n_features) floating point tensor
    :param pair_first: (n_pairs,) index tensor indicating first atom of pair
    :param pair_second: (n_pairs,) index tensor indicating second atom of pair
    :return: env (n_atoms, n_sensitivities, n_features) floating tensor
    """
    n_pairs, n_nu = sensitivities.shape
    n_atom, n_feat = features.shape
    pair_features = features[pair_second].unsqueeze(1)
    sensitivities = sensitivities.unsqueeze(2)
    pair_env_features = sensitivities * pair_features
    env_features = torch.zeros((n_atom, n_nu, n_feat), device=features.device, dtype=features.dtype)
    env_features.index_add_(0, pair_first, pair_env_features)
    return env_features


def sensesum(env, features, pair_first, pair_second):
    """
    Computes product of environment at pair_first with features from pair_second,
    whilst summing over feature indices.

    See the :doc:`/user_guide/ckernels` section of the documentation for more information.

    :param env: (n_atoms, n_sensitivities, n_features) floating tensor
    :param features: (n_atoms, n_features) floating point tensor
    :param pair_first: (n_pairs,) index tensor indicating first atom of pair
    :param pair_second: (n_pairs,) index tensor indicating second atom of pair
    :return: sense (n_pairs, n_sensitivities) floating point tensor
    """
    n_atoms, n_nu, n_feat = env.shape
    pair_env = env[pair_first]
    pair_feat = features[pair_second]
    sense = (pair_env * pair_feat.unsqueeze(1)).sum(dim=2)
    return sense


def featsum(env, sense, pair_first, pair_second):
    """

    Compute inner product of sensitivities with environment tensor over atoms
    from pair_first, while accumulating them on to pair_second.

    The summation order is different from envsum because this
    signature naturally supports the use of featsum as a backwards pass
    for envsum, and vise-versa.

    See the :doc:`/user_guide/ckernels` section of the documentation for more information.

    :param env: (n_atoms, n_sensitivities, n_features) floating tensor
    :param sense: (n_pairs, n_sensitivities) floating point tensor
    :param pair_first: (n_pairs,) index tensor indicating first atom of pair
    :param pair_second: (n_pairs,) index tensor indicating second atom of pair
    :return: feat (n_atoms, n_features) floating point tensor
    """
    n_atoms, n_nu, n_feat = env.shape
    pair_env = env[pair_first]
    pair_feat = (pair_env * sense.unsqueeze(2)).sum(dim=(1,))
    feat = torch.zeros(n_atoms, n_feat, device=env.device, dtype=env.dtype)
    feat.index_add_(0, pair_second, pair_feat)
    return feat

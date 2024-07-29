"""
Pure pytorch implementation of envsum operations
"""
import torch


def envsum(sensitivities, features, pair_first, pair_second):
    n_pairs, n_nu = sensitivities.shape
    n_atom, n_feat = features.shape
    pair_features = features[pair_second].unsqueeze(1)
    sensitivities = sensitivities.unsqueeze(2)
    pair_env_features = sensitivities * pair_features
    env_features = torch.zeros((n_atom, n_nu, n_feat), device=features.device, dtype=features.dtype)
    env_features.index_add_(0, pair_first, pair_env_features)
    return env_features


def sensesum(env, features, pair_first, pair_second):
    n_atoms, n_nu, n_feat = env.shape
    pair_env = env[pair_first]
    pair_feat = features[pair_second]
    sense = (pair_env * pair_feat.unsqueeze(1)).sum(dim=2)
    return sense

def featsum(env, sense, pair_first, pair_second):
    n_atoms, n_nu, n_feat = env.shape
    pair_env = env[pair_first]
    pair_feat = (pair_env * sense.unsqueeze(2)).sum(dim=(1,))
    feat = torch.zeros(n_atoms, n_feat, device=env.device, dtype=env.dtype)
    feat.index_add_(0, pair_second, pair_feat)
    return feat

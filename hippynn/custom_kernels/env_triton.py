import torch
import triton
import triton.language as tl
from .utils import resort_pairs_cached

from .env_pytorch import envsum as envsum_pt, sensesum as sensesum_pt, featsum as featsum_pt


@triton.jit
def envsum_kernel(
    out_env_ptr,
    sens_ptr,
    feat_ptr,
    psecond_ptr,
    atom_ids_ptr,
    atom_starts_ptr,
    atom_size,
    sens_size: tl.constexpr,
    feat_size: tl.constexpr,
    p2_sens_size: tl.constexpr,
    p2_feat_size: tl.constexpr,
    dtype: tl.constexpr = tl.float32,
):
    atom_id = tl.program_id(axis=0)
    start = tl.load(atom_starts_ptr + atom_id, mask=atom_id < atom_size, other=0)
    end = tl.load(atom_starts_ptr + atom_id + 1, mask=atom_id < atom_size, other=0)
    target_id = tl.load(atom_ids_ptr + atom_id, mask=atom_id < atom_size, other=0)
    sens_block_ids = tl.arange(0, p2_sens_size)
    feat_block_ids = tl.arange(0, p2_feat_size)
    tmp = tl.zeros((p2_sens_size, p2_feat_size), dtype=dtype)
    for ind in range(start, end):
        # [p2_sens_size,], coming from the pair sensitivity
        s = tl.load(sens_ptr + (ind * sens_size) + sens_block_ids, mask=sens_block_ids < sens_size, other=0.0)
        pair_ind = tl.load(psecond_ptr + ind)  # TODO do we need mask here
        # [p2_feat_size,], coming from the neighbor feature
        feat = tl.load(feat_ptr + (pair_ind * feat_size) + feat_block_ids, mask=feat_block_ids < feat_size, other=0.0)
        # temp_mat and tmp is [p2_sens_size, p2_feat_size]
        temp_mat = s[:, None] * feat[None, :]
        tmp = tmp + temp_mat
    mask = (sens_block_ids[:, None] < sens_size) & (feat_block_ids[None, :] < feat_size)
    block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]
    tl.store(out_env_ptr + (target_id * sens_size * feat_size) + block_ids, tmp, mask=mask)


def envsum_triton(sensitivities, features, pair_first, pair_second, atom_ids, atom_starts, out_env_fetures=None):
    n_pairs, n_nu = sensitivities.shape
    n_atom, n_feat = features.shape
    (n_atom_with_pairs,) = atom_ids.shape
    if out_env_fetures == None:
        out_env_fetures = torch.zeros((n_atom, n_nu, n_feat), dtype=features.dtype, device=features.device)
    dtype = tl.float32
    if features.dtype == torch.float64:
        dtype = tl.float64
    p2_sens_size = triton.next_power_of_2(n_nu)
    p2_feat_size = triton.next_power_of_2(n_feat)
    envsum_kernel[(n_atom_with_pairs,)](
        out_env_fetures,
        sensitivities,
        features,
        pair_second,
        atom_ids,
        atom_starts,
        n_atom_with_pairs,
        n_nu,
        n_feat,
        p2_sens_size,
        p2_feat_size,
        dtype=dtype,
    )
    return out_env_fetures


def envsum(sense, features, pfirst, psecond):
    if sense.device == torch.device("cpu"):
        return envsum_pt(sense, features, pfirst, psecond)
    psecond_hold = psecond
    argsort, atom1_ids, atom1_starts, pfirst, (sense, psecond) = resort_pairs_cached(pfirst, [sense, psecond])
    resort_pairs_cached(psecond_hold, [])
    return envsum_triton(sense, features, pfirst, psecond, atom1_ids, atom1_starts, out_env_fetures=None)


@triton.jit
def sensesum_kernel(
    out_sense_ptr,
    env_ptr,
    feat_ptr,
    pfirst_ptr,
    psecond_ptr,
    pair_size,
    sens_size: tl.constexpr,
    feat_size: tl.constexpr,
    p2_sens_size: tl.constexpr,
    p2_feat_size: tl.constexpr,
    dtype: tl.constexpr = tl.float32,
):
    pair_id = tl.program_id(axis=0)
    first = tl.load(pfirst_ptr + pair_id, mask=pair_id < pair_size, other=0)
    second = tl.load(psecond_ptr + pair_id, mask=pair_id < pair_size, other=0)
    sens_block_ids = tl.arange(0, p2_sens_size)
    feat_block_ids = tl.arange(0, p2_feat_size)
    mask = (sens_block_ids[:, None] < sens_size) & (feat_block_ids[None, :] < feat_size)
    block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]
    # [p2_sens_size, p2_feat_size]
    env = tl.load(env_ptr + (first * sens_size * feat_size) + block_ids, mask=mask, other=0.0)
    # [p2_feat_size, ]
    feat = tl.load(feat_ptr + (second * feat_size) + feat_block_ids, mask=feat_block_ids < feat_size, other=0.0)
    """
    type_f32: tl.constexpr = tl.float32
    type_check: tl.constexpr = (dtype == type_f32)
    if type_check:
        res = tl.dot(env, feat[:, None])
    else:
        res = tl.sum(env * feat[None, :], axis=1)
    """
    res = tl.sum(env * feat[None, :], axis=1)
    tl.store(out_sense_ptr + (pair_id * sens_size) + sens_block_ids, res, mask=sens_block_ids < sens_size)


def sensesum(env, features, pair_first, pair_second, out_sense=None):
    if env.device == torch.device("cpu"):
        return sensesum_pt(env, features, pair_first, pair_second)
    _, n_nu, _ = env.shape
    n_atom, n_feat = features.shape
    n_pairs = len(pair_first)
    if out_sense == None:
        out_sense = torch.zeros((n_pairs, n_nu), dtype=features.dtype, device=features.device)
    dtype = tl.float32
    if features.dtype == torch.float64:
        dtype = tl.float64
    p2_sens_size = triton.next_power_of_2(n_nu)
    p2_feat_size = triton.next_power_of_2(n_feat)
    sensesum_kernel[(n_pairs,)](
        out_sense, env, features, pair_first, pair_second, n_pairs, n_nu, n_feat, p2_sens_size, p2_feat_size, dtype=dtype
    )
    return out_sense


@triton.jit
def featsum_kernel(
    out_feat,
    env_ptr,
    sens_ptr,
    pfirst_ptr,
    psecond_ptr,
    atom2_ids_ptr,
    atom2_starts_ptr,
    atom_size,
    sens_size: tl.constexpr,
    feat_size: tl.constexpr,
    p2_sens_size: tl.constexpr,
    p2_feat_size: tl.constexpr,
    dtype: tl.constexpr = tl.float32,
):
    atom_id = tl.program_id(axis=0)
    start = tl.load(atom2_starts_ptr + atom_id, mask=atom_id < atom_size, other=0)
    end = tl.load(atom2_starts_ptr + atom_id + 1, mask=atom_id < atom_size, other=0)
    target_id = tl.load(atom2_ids_ptr + atom_id, mask=atom_id < atom_size, other=0)
    sens_block_ids = tl.arange(0, p2_sens_size)
    feat_block_ids = tl.arange(0, p2_feat_size)
    tmp = tl.zeros((p2_feat_size,), dtype=dtype)
    for ind in range(start, end):
        # [p2_sens_size,], coming from the pair sensitivity
        sense = tl.load(sens_ptr + (ind * sens_size) + sens_block_ids, mask=sens_block_ids < sens_size, other=0.0)
        pair_ind = tl.load(pfirst_ptr + ind)  # TODO do we need mask here
        mask = (sens_block_ids[:, None] < sens_size) & (feat_block_ids[None, :] < feat_size)
        block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]
        # [p2_sens_size, p2_feat_size]
        env = tl.load(env_ptr + (pair_ind * sens_size * feat_size) + block_ids, mask=mask, other=0.0)
        # temp_mat and tmp is [p2_feat_size,]
        temp_mat = tl.sum(env * sense[:, None], axis=0)
        tmp = tmp + temp_mat
    tl.store(out_feat + (target_id * feat_size) + feat_block_ids, tmp, mask=feat_block_ids < feat_size)


def featsum_triton(env, sense, pair_first, pair_second, atom2_ids, atom2_starts, out_feat=None):
    n_atom, n_nu, n_feat = env.shape
    (n_pairs,) = pair_first.shape
    (n_atoms_with_pairs,) = atom2_ids.shape
    if out_feat == None:
        out_feat = torch.zeros((n_atom, n_feat), dtype=env.dtype, device=env.device)
    dtype = tl.float32
    if env.dtype == torch.float64:
        dtype = tl.float64
    p2_sens_size = triton.next_power_of_2(n_nu)
    p2_feat_size = triton.next_power_of_2(n_feat)
    featsum_kernel[(n_atoms_with_pairs,)](
        out_feat,
        env,
        sense,
        pair_first,
        pair_second,
        atom2_ids,
        atom2_starts,
        n_atoms_with_pairs,
        n_nu,
        n_feat,
        p2_sens_size,
        p2_feat_size,
        dtype=dtype,
    )
    return out_feat


def featsum(env, sense, pfirst, psecond):
    if env.device == torch.device("cpu"):
        return featsum_pt(env, sense, pfirst, psecond)
    pfirst_hold = pfirst
    argsort, atom2_ids, atom2_starts, psecond, (sense, pfirst) = resort_pairs_cached(psecond, [sense, pfirst])
    resort_pairs_cached(pfirst_hold, [])
    return featsum_triton(env, sense, pfirst, psecond, atom2_ids, atom2_starts, out_feat=None)

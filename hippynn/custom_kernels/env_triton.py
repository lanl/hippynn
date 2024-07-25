import torch
import triton
import triton.language as tl
from .utils import resort_pairs_cached

# Load backup implementation for CPU tensors.
from .env_pytorch import envsum as envsum_alternative, sensesum as sensesum_alternative, featsum as featsum_alternative

# If numba is available, this implementation will default to numba on CPU. If not, use vanilla pytorch.
try:
    from .env_numba import new_envsum as envsum_alternative, new_sensesum as sensesum_alternative, new_featsum as featsum_alternative
except ImportError:
    pass


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

    valid_atom_id = atom_id < atom_size

    start = tl.load(atom_starts_ptr + atom_id, mask=valid_atom_id, other=0)
    end = tl.load(atom_starts_ptr + atom_id + 1, mask=valid_atom_id, other=0)
    target_id = tl.load(atom_ids_ptr + atom_id, mask=valid_atom_id, other=0)

    sens_block_ids = tl.arange(0, p2_sens_size)
    feat_block_ids = tl.arange(0, p2_feat_size)
    env_block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]

    valid_sens = sens_block_ids < sens_size
    valid_feat = feat_block_ids < feat_size
    valid_env = valid_sens[:, None] & valid_feat[None, :]

    tmp = tl.zeros((p2_sens_size, p2_feat_size), dtype=dtype)

    for ind in range(start, end):
        # [p2_sens_size,], coming from the pair sensitivity
        s = tl.load(sens_ptr + (ind * sens_size) + sens_block_ids, mask=valid_sens, other=0.0)
        atom2_id = tl.load(psecond_ptr + ind)  # TODO C: do we need mask here # N: I don't think so
        # [p2_feat_size,], coming from the neighbor feature
        feat = tl.load(feat_ptr + (atom2_id * feat_size) + feat_block_ids, mask=valid_feat, other=0.0)
        # temp_mat and tmp is [p2_sens_size, p2_feat_size]
        temp_mat = s[:, None] * feat[None, :]
        tmp = tmp + temp_mat

    atom_offset = target_id * sens_size * feat_size

    # TODO: use sparsity of sensitivities to reduce workload? (see numba envsum implementation)
    tl.store(out_env_ptr + atom_offset + env_block_ids, tmp, mask=valid_env)


def envsum_triton(sensitivities, features, pair_first, pair_second, atom_ids, atom_starts, out_env=None):
    n_pairs, n_nu = sensitivities.shape
    n_atom, n_feat = features.shape
    (n_atom_with_pairs,) = atom_ids.shape
    if out_env is None:
        out_env = torch.zeros((n_atom, n_nu, n_feat), dtype=features.dtype, device=features.device)
    dtype = tl.float32
    if features.dtype == torch.float64:
        dtype = tl.float64
    p2_sens_size = triton.next_power_of_2(n_nu)
    p2_feat_size = triton.next_power_of_2(n_feat)
    envsum_kernel[(n_atom_with_pairs,)](
        out_env,
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
    return out_env


def envsum(sense, features, pfirst, psecond):
    if sense.device == torch.device("cpu"):
        return envsum_alternative(sense, features, pfirst, psecond)
    psecond_hold = psecond
    argsort, atom1_ids, atom1_starts, pfirst, (sense, psecond) = resort_pairs_cached(pfirst, [sense, psecond])
    resort_pairs_cached(psecond_hold, [])  # Preemptively sort for backwards pass.
    return envsum_triton(sense, features, pfirst, psecond, atom1_ids, atom1_starts, out_env=None)


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
    valid_pair = pair_id < pair_size

    first = tl.load(pfirst_ptr + pair_id, mask=valid_pair, other=0)
    second = tl.load(psecond_ptr + pair_id, mask=valid_pair, other=0)

    sens_block_ids = tl.arange(0, p2_sens_size)
    feat_block_ids = tl.arange(0, p2_feat_size)
    env_block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]

    valid_sens = sens_block_ids < sens_size
    valid_feat = feat_block_ids < feat_size
    valid_env = valid_sens[:, None] & valid_feat[None, :]

    # [p2_sens_size, p2_feat_size]
    env = tl.load(env_ptr + (first * sens_size * feat_size) + env_block_ids, mask=valid_env, other=0.0)
    # [p2_feat_size, ]
    feat = tl.load(feat_ptr + (second * feat_size) + feat_block_ids, mask=valid_feat, other=0.0)
    # TODO N: What is going on in this string?
    """
    type_f32: tl.constexpr = tl.float32
    type_check: tl.constexpr = (dtype == type_f32)
    if type_check:
        res = tl.dot(env, feat[:, None])
    else:
        res = tl.sum(env * feat[None, :], axis=1)
    """
    res = tl.sum(env * feat[None, :], axis=1)
    # TODO: use sparsity of sensitivities to reduce workload? (see numba envsum implementation)
    tl.store(out_sense_ptr + (pair_id * sens_size) + sens_block_ids, res, mask=valid_sens)


def sensesum(env, features, pair_first, pair_second, out_sense=None):
    if env.device == torch.device("cpu"):
        return sensesum_alternative(env, features, pair_first, pair_second)
    _, n_nu, _ = env.shape
    n_atom, n_feat = features.shape
    n_pairs = len(pair_first)
    if out_sense is None:
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
    valid_atom = atom_id < atom_size

    start = tl.load(atom2_starts_ptr + atom_id, mask=valid_atom, other=0)
    end = tl.load(atom2_starts_ptr + atom_id + 1, mask=valid_atom, other=0)
    target_id = tl.load(atom2_ids_ptr + atom_id, mask=valid_atom, other=0)

    sens_block_ids = tl.arange(0, p2_sens_size)
    feat_block_ids = tl.arange(0, p2_feat_size)
    env_block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]

    valid_sens = sens_block_ids < sens_size
    valid_feat = feat_block_ids < feat_size
    valid_env = valid_sens[:, None] & valid_feat[None, :]

    tmp = tl.zeros((p2_feat_size,), dtype=dtype)

    for ind in range(start, end):
        # [p2_sens_size,], coming from the pair sensitivity
        sense = tl.load(sens_ptr + (ind * sens_size) + sens_block_ids, mask=valid_sens, other=0.0)
        atom1_ind = tl.load(pfirst_ptr + ind)  # C: TODO do we need mask here #N: Don't think so
        # [p2_sens_size, p2_feat_size]
        env = tl.load(env_ptr + (atom1_ind * sens_size * feat_size) + env_block_ids, mask=valid_env, other=0.0)
        # temp_mat and tmp is [p2_feat_size,]
        temp_mat = tl.sum(env * sense[:, None], axis=0)
        tmp = tmp + temp_mat
    tl.store(out_feat + (target_id * feat_size) + feat_block_ids, tmp, mask=valid_feat)


def featsum_triton(env, sense, pair_first, pair_second, atom2_ids, atom2_starts, out_feat=None):
    n_atom, n_nu, n_feat = env.shape
    (n_pairs,) = pair_first.shape
    (n_atoms_with_pairs,) = atom2_ids.shape
    if out_feat is None:
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
        return featsum_alternative(env, sense, pfirst, psecond)
    pfirst_hold = pfirst
    argsort, atom2_ids, atom2_starts, psecond, (sense, pfirst) = resort_pairs_cached(psecond, [sense, pfirst])
    resort_pairs_cached(pfirst_hold, [])  # preemptively sort (probably no-op)
    return featsum_triton(env, sense, pfirst, psecond, atom2_ids, atom2_starts, out_feat=None)

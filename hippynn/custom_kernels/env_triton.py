"""
triton implementation of envsum custom kernels for GPU.
"""
import warnings
import torch
import triton
import triton.language as tl
from .utils import resort_pairs_cached
from .registry import MessagePassingKernels

# If numba is available, this implementation will default to numba on CPU. If not, use vanilla pytorch.
try:
    from .env_numba import new_envsum as envsum_alternative, new_sensesum as sensesum_alternative, new_featsum as featsum_alternative
except ImportError:
    # Load backup implementation for CPU tensors.
    from .env_pytorch import envsum as envsum_alternative, sensesum as sensesum_alternative, featsum as featsum_alternative


if torch.cuda.is_available():
    device_capability = torch.cuda.get_device_capability()
    if not device_capability[0] > 6:
        msg = f"`triton` package found, but does not support GPU's compute capability: {device_capability}"
        # First warn, then error, because:
        # - the warning should be seen by the user.
        # - The error is caught by the __init__ module and uses this as a signal not to include
        #   'triton' as an available implementation
        warnings.warn(msg, stacklevel=2)
        raise ImportError(msg)


def config_pruner(configs, nargs, **kwargs):
    """
    Trims the unnecessary config options based on the sens. and feat. sizes
    """
    p2_sens_size = triton.next_power_of_2(nargs["sens_size"])
    p2_feat_size = triton.next_power_of_2(nargs["feat_size"])

    used = set()
    for config in configs:

        # Don't use block sizes bigger than p2_sens_size or p2_feat_size; they will give the same result
        # because there will only be one block.
        sense_block_size = min(p2_sens_size, config.kwargs["SENS_BLOCK_SIZE"])
        feat_block_size = min(p2_feat_size, config.kwargs["FEAT_BLOCK_SIZE"])

        if (sense_block_size, feat_block_size, config.num_stages, config.num_warps) in used:
            continue

        used.add((sense_block_size, feat_block_size, config.num_stages, config.num_warps))

        yield triton.Config(
            {
                "SENS_BLOCK_SIZE": sense_block_size,
                "FEAT_BLOCK_SIZE": feat_block_size,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps,
        )


def get_autotune_config():
    """
    Create a list of config options for the kernels
    TODO: Need to spend time actually figuring out more reasonable options
    targeted for modern GPUs
    """
    return [
        triton.Config({"SENS_BLOCK_SIZE": 16, "FEAT_BLOCK_SIZE": 16}),
        triton.Config({"SENS_BLOCK_SIZE": 16, "FEAT_BLOCK_SIZE": 32}),
        triton.Config({"SENS_BLOCK_SIZE": 16, "FEAT_BLOCK_SIZE": 64}),
        triton.Config({"SENS_BLOCK_SIZE": 16, "FEAT_BLOCK_SIZE": 128}),
        triton.Config({"SENS_BLOCK_SIZE": 16, "FEAT_BLOCK_SIZE": 256}),
        triton.Config({"SENS_BLOCK_SIZE": 32, "FEAT_BLOCK_SIZE": 32}),
        triton.Config({"SENS_BLOCK_SIZE": 32, "FEAT_BLOCK_SIZE": 64}),
        triton.Config({"SENS_BLOCK_SIZE": 32, "FEAT_BLOCK_SIZE": 128}),
        triton.Config({"SENS_BLOCK_SIZE": 32, "FEAT_BLOCK_SIZE": 128}, num_warps=8),
        triton.Config({"SENS_BLOCK_SIZE": 32, "FEAT_BLOCK_SIZE": 256}),
        triton.Config({"SENS_BLOCK_SIZE": 32, "FEAT_BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"SENS_BLOCK_SIZE": 64, "FEAT_BLOCK_SIZE": 32}),
        triton.Config({"SENS_BLOCK_SIZE": 64, "FEAT_BLOCK_SIZE": 64}),
        triton.Config({"SENS_BLOCK_SIZE": 64, "FEAT_BLOCK_SIZE": 128}),
        triton.Config({"SENS_BLOCK_SIZE": 64, "FEAT_BLOCK_SIZE": 256}),
        triton.Config({"SENS_BLOCK_SIZE": 128, "FEAT_BLOCK_SIZE": 32}),
        triton.Config({"SENS_BLOCK_SIZE": 128, "FEAT_BLOCK_SIZE": 64}),
        triton.Config({"SENS_BLOCK_SIZE": 128, "FEAT_BLOCK_SIZE": 64}, num_warps=8),
        triton.Config({"SENS_BLOCK_SIZE": 256, "FEAT_BLOCK_SIZE": 32}),
        triton.Config({"SENS_BLOCK_SIZE": 256, "FEAT_BLOCK_SIZE": 64}),
        triton.Config({"SENS_BLOCK_SIZE": 256, "FEAT_BLOCK_SIZE": 64}, num_warps=8),
    ]


@triton.autotune(configs=get_autotune_config(), key=["sens_size", "feat_size"], prune_configs_by={"early_config_prune": config_pruner})
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
    SENS_BLOCK_SIZE: tl.constexpr,
    FEAT_BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr = tl.float32,
):
    atom_id = tl.program_id(axis=0)
    sens_id = tl.program_id(axis=1)
    feat_id = tl.program_id(axis=2)

    valid_atom_id = atom_id < atom_size

    start = tl.load(atom_starts_ptr + atom_id, mask=valid_atom_id, other=0)
    end = tl.load(atom_starts_ptr + atom_id + 1, mask=valid_atom_id, other=0)
    target_id = tl.load(atom_ids_ptr + atom_id, mask=valid_atom_id, other=0)

    sens_block_ids = tl.arange(0, SENS_BLOCK_SIZE) + (sens_id * SENS_BLOCK_SIZE)
    feat_block_ids = tl.arange(0, FEAT_BLOCK_SIZE) + (feat_id * FEAT_BLOCK_SIZE)
    env_block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]

    valid_sens = sens_block_ids < sens_size
    valid_feat = feat_block_ids < feat_size
    valid_env = valid_sens[:, None] & valid_feat[None, :]

    tmp = tl.zeros((SENS_BLOCK_SIZE, FEAT_BLOCK_SIZE), dtype=dtype)

    for ind in range(start, end):
        # [SENS_BLOCK_SIZE,], coming from the pair sensitivity
        s = tl.load(sens_ptr + (ind * sens_size) + sens_block_ids, mask=valid_sens, other=0.0)
        atom2_id = tl.load(psecond_ptr + ind)
        # [FEAT_BLOCK_SIZE,], coming from the neighbor feature
        feat = tl.load(feat_ptr + (atom2_id * feat_size) + feat_block_ids, mask=valid_feat, other=0.0)
        # temp_mat and tmp is [SENS_BLOCK_SIZE, FEAT_BLOCK_SIZE]
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

    grid = lambda META: (n_atom_with_pairs, triton.cdiv(n_nu, META["SENS_BLOCK_SIZE"]), triton.cdiv(n_feat, META["FEAT_BLOCK_SIZE"]))

    envsum_kernel[grid](
        out_env,
        sensitivities,
        features,
        pair_second,
        atom_ids,
        atom_starts,
        n_atom_with_pairs,
        n_nu,
        n_feat,
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


@triton.autotune(configs=get_autotune_config(), key=["sens_size", "feat_size"], prune_configs_by={"early_config_prune": config_pruner})
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
    SENS_BLOCK_SIZE: tl.constexpr,
    FEAT_BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr = tl.float32,
):
    pair_id = tl.program_id(axis=0)
    sense_id = tl.program_id(axis=1)
    num_feat_blocks: tl.constexpr = tl.cdiv(feat_size, FEAT_BLOCK_SIZE)
    valid_pair = pair_id < pair_size

    first = tl.load(pfirst_ptr + pair_id, mask=valid_pair, other=0)
    second = tl.load(psecond_ptr + pair_id, mask=valid_pair, other=0)

    sens_block_ids = tl.arange(0, SENS_BLOCK_SIZE) + (sense_id * SENS_BLOCK_SIZE)
    feat_block_ids = tl.arange(0, FEAT_BLOCK_SIZE)

    valid_sens = sens_block_ids < sens_size

    tmp = tl.zeros((SENS_BLOCK_SIZE,), dtype=dtype)
    for feat_id in range(num_feat_blocks):
        valid_feat = feat_block_ids < feat_size
        env_block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]
        valid_env = valid_sens[:, None] & valid_feat[None, :]
        # [SENS_BLOCK_SIZE, FEAT_BLOCK_SIZE]
        env = tl.load(env_ptr + (first * sens_size * feat_size) + env_block_ids, mask=valid_env, other=0.0)
        # [FEAT_BLOCK_SIZE, ]
        feat = tl.load(feat_ptr + (second * feat_size) + feat_block_ids, mask=valid_feat, other=0.0)
        # TODO: Here we use outer product followed by sum b/c built-in triton dot needs batches and FP<64.
        # Can we make this better then?
        # For future reference:
        """
        type_f32: tl.constexpr = tl.float32
        type_check: tl.constexpr = (dtype == type_f32)
        if type_check:
            res = tl.dot(env, feat[:, None])
        else:
            res = tl.sum(env * feat[None, :], axis=1)
        """
        tmp += tl.sum(env * feat[None, :], axis=1)
        # increment the feat block id
        feat_block_ids += FEAT_BLOCK_SIZE
    # TODO: use sparsity of sensitivities to reduce workload? (see numba envsum implementation)
    tl.store(out_sense_ptr + (pair_id * sens_size) + sens_block_ids, tmp, mask=valid_sens)


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

    grid = lambda META: (n_pairs, triton.cdiv(n_nu, META["SENS_BLOCK_SIZE"]))
    sensesum_kernel[grid](out_sense, env, features, pair_first, pair_second, n_pairs, n_nu, n_feat, dtype=dtype)
    return out_sense


@triton.autotune(configs=get_autotune_config(), key=["sens_size", "feat_size"], prune_configs_by={"early_config_prune": config_pruner})
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
    SENS_BLOCK_SIZE: tl.constexpr,
    FEAT_BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr = tl.float32,
):
    atom_id = tl.program_id(axis=0)
    feat_id = tl.program_id(axis=1)
    num_sense_blocks: tl.constexpr = tl.cdiv(sens_size, SENS_BLOCK_SIZE)
    valid_atom = atom_id < atom_size

    start = tl.load(atom2_starts_ptr + atom_id, mask=valid_atom, other=0)
    end = tl.load(atom2_starts_ptr + atom_id + 1, mask=valid_atom, other=0)
    target_id = tl.load(atom2_ids_ptr + atom_id, mask=valid_atom, other=0)

    feat_block_ids = tl.arange(0, FEAT_BLOCK_SIZE) + (feat_id * FEAT_BLOCK_SIZE)

    valid_feat = feat_block_ids < feat_size

    tmp = tl.zeros((FEAT_BLOCK_SIZE,), dtype=dtype)

    for ind in range(start, end):
        sens_block_ids = tl.arange(0, SENS_BLOCK_SIZE)
        for sense_id in range(num_sense_blocks):
            valid_sens = sens_block_ids < sens_size
            # [SENS_BLOCK_SIZE,], coming from the pair sensitivity
            sense = tl.load(sens_ptr + (ind * sens_size) + sens_block_ids, mask=valid_sens, other=0.0)
            atom1_ind = tl.load(pfirst_ptr + ind)
            # [SENS_BLOCK_SIZE, FEAT_BLOCK_SIZE]
            env_block_ids = sens_block_ids[:, None] * feat_size + feat_block_ids[None, :]
            valid_env = valid_sens[:, None] & valid_feat[None, :]
            env = tl.load(env_ptr + (atom1_ind * sens_size * feat_size) + env_block_ids, mask=valid_env, other=0.0)
            # temp_mat and tmp is [FEAT_BLOCK_SIZE,]
            temp_mat = tl.sum(env * sense[:, None], axis=0)
            tmp = tmp + temp_mat
            # increment the sense block id
            sens_block_ids += SENS_BLOCK_SIZE
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

    grid = lambda META: (n_atoms_with_pairs, triton.cdiv(n_feat, META["FEAT_BLOCK_SIZE"]))

    featsum_kernel[grid](
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


triton_kernels = MessagePassingKernels(
    "triton",
    envsum,
    sensesum,
    featsum,
)

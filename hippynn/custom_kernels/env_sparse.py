"""
Pure pytorch implementation of envsum operations
"""
import torch

from .registry import MessagePassingKernels


# TODO: Does resort_pairs_cached give enough to allow direct construction of CSR?


def make_sparse_sense(sensitivities, pair_first, pair_second, n_atom: int):
    """
    Construct sensitivities as a sparse matrix with shape
    (n_atoms * n_nu, n_atoms).

    The n_atoms * n_nu  is needed because of limitations in the hybrid-sparse
    matrix multiply routines available in pytorch.  This function
    is implemented seprately because both envsum and sensesum use it.
    The to_csr call is done afterwards because envsum needs to be
    transposed.

    :param sensitivities:
    :param pair_first:
    :param pair_second:
    :param n_atom:
    :return:
    """
    n_pairs, n_nu = sensitivities.shape

    pf_unsqueeze = pair_first.unsqueeze(1).expand(n_pairs, n_nu)
    nu_range = torch.arange(n_nu, device=pair_first.device).unsqueeze(0)
    first_index = (pf_unsqueeze * n_nu + nu_range).flatten()
    second_index = pair_second.unsqueeze(1).expand(n_pairs, n_nu).flatten()

    indices = torch.stack([first_index, second_index])
    sparse_sense = torch.sparse_coo_tensor(
        values=sensitivities.flatten(),
        indices=indices,
        size=(n_atom * n_nu, n_atom),
        dtype=sensitivities.dtype,
        device=sensitivities.device)

    return sparse_sense


def envsum(sensitivities, features, pair_first, pair_second):
    n_pairs, n_nu = sensitivities.shape
    n_atom, n_feat = features.shape

    sparse_sense = make_sparse_sense(sensitivities, pair_first, pair_second, n_atom)
    sparse_sense = sparse_sense.to_sparse_csr()

    env = torch.mm(sparse_sense, features).reshape(n_atom, n_nu, n_feat)

    return env


def sensesum(env, features, pair_first, pair_second):
    """

    Sparse sensesum implementation uses a sparsity matrix
    of zeros with shape (n_atoms x n_atoms), combined with
    the crucial function torch.sparse sampled_addmm.

    :param env:
    :param features:
    :param pair_first:
    :param pair_second:
    :return:
    """

    n_atoms, n_nu, n_feat = env.shape

    indices = torch.stack([pair_first, pair_second])
    sparse_pairs = torch.sparse_coo_tensor(
        values=torch.zeros_like(pair_first, dtype=env.dtype),
        indices=indices,
        size=(n_atoms, n_atoms),
        dtype=env.dtype,
        device=pair_first.device)

    sparse_pairs = sparse_pairs.to_sparse_csr()

    env_rs = env.permute(1, 0, 2)  # Putting sensitivity index first
    feat_rs = features.transpose(0, 1)  # 2D transpose
    feat_rs = feat_rs.unsqueeze(0).expand(n_nu, -1, -1)  # feat needs same batch size
    sense_sparse = torch.sparse.sampled_addmm(sparse_pairs, env_rs, feat_rs)
    sense_sparse = sense_sparse.to_sparse_coo()

    sense_values = sense_sparse.values()

    # This will error if the same pair appears twice.
    try:
        n_pair, = pair_first.shape
        sense_values = sense_values.reshape(n_nu, n_pair)
    except RuntimeError as ee:
        raise ValueError(
            f"Sensitivity values shape changed. Likely more than one pair entry "
            f"connecting the same atoms. The 'sparse' implementation custom kernels do not support "
            f"pair lists containing duplicate items. "
            f"Input shape: {n_pair} Output shape: {sense_values.shape[0] // n_nu}") from ee

    sense_values = sense_values.transpose(0, 1)

    # Note: indices emerge sorted. If we have not sorted by pfirst and psecond,
    # we must then invert how they appear.
    pair_rank_array = n_atoms * pair_first + pair_second
    inverse_order = torch.argsort(torch.argsort(pair_rank_array))
    sense_values = sense_values[inverse_order]

    return sense_values


def featsum(env, sense, pair_first, pair_second):
    n_atoms, n_nu, n_feat = env.shape

    sparse_sense = make_sparse_sense(sense, pair_first, pair_second, n_atoms)
    sparse_sense = sparse_sense.transpose(0, 1)
    sparse_sense = sparse_sense.to_sparse_csr()

    feat = torch.mm(sparse_sense, env.reshape(n_atoms * n_nu, n_feat))
    return feat


sparse_kernels = MessagePassingKernels(
    "sparse",
    envsum, sensesum, featsum,
)

# Note: no sparse_jit because try/except not supported.
# to sparse_compile because it won't transpose a matrix??
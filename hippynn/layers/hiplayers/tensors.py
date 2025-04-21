"""
Interaction functions for hip-hop-nn
"""
import collections
import itertools
import torch


from typing import List
from torch import Tensor
from typing import Dict

# use opt_einsum when it is available
try:
    import opt_einsum

    shared_intermediates = opt_einsum.shared_intermediates
except ImportError:
    from contextlib import nullcontext

    shared_intermediates = nullcontext

ndim = 3  # spatial dimensions


def trace(tensor, axis_1, axis_2):
    return torch.diagonal(tensor, 0, axis_1, axis_2).sum(dim=-1)


def tpl(int_tensor):
    return tuple(x.item() for x in int_tensor.unbind())


def cartesian_irreducible_mapping(ind_order):
    # First, construct the map from the full cartesian space to a symmetric basis,
    # mapping each element from the cartesian space to the sorted version of its indices.
    input_ind = []
    output_ind = []
    for ind in itertools.product(range(ndim), repeat=ind_order):
        input_ind.append(ind)
        output_ind.append(tuple(sorted(ind)))
    out_ind = torch.as_tensor(output_ind)
    in_ind = torch.as_tensor(input_ind)

    # List of symmetric indices only.
    needed_ind = torch.unique(out_ind, dim=0)

    # Figure out numbering for nonsymmetric indices within the symmetric ones.
    # # This step is of somewhat large size, could maybe be reduced with better algorithm.
    equal = (needed_ind == out_ind.unsqueeze(1)).all(dim=2)
    out_order = torch.where(equal)[1]

    # build sparse matrix -> all values are 1
    ind_comb = torch.stack(
        (torch.arange(len(out_order)), out_order),
    )
    xform = torch.sparse_coo_tensor(ind_comb, torch.ones(len(out_order)), dtype=torch.int64).to_dense()
    # This maps symmetric space to d^order space
    # shape (d^o, t(o)) where t(o) is the triangular number.

    # Now we need to build the map from traceless symmetric tensors to symmetric ones.

    # Which indices are constrained by tracelessness
    # Constrained elements are ones that end with 2,2 in the full cartesian space.
    # # Again, may use more memory than needed here.
    constrained_elements = (needed_ind[:, -2:] == torch.as_tensor([2, 2])).all(dim=-1)
    constrained_pos = torch.where(constrained_elements)[0]
    # Set of indices which are constrained by trace.
    constrained_ind = needed_ind[constrained_elements]

    # Which indices are not constrained
    free_elements = ~constrained_elements
    free_pos = torch.where(free_elements)[0]

    if ind_order == 1:
        # Exception to above which is wrong when only one dimension
        constrained_ind = []
        free_elements = [True, True, True]

    # Now we use regular python b/c it is easier to construct these loops.

    # Map from the index space to the symmetric basis number.
    bare_sym_map = {tpl(x): i for i, x in enumerate(needed_ind.unbind(0))}
    # Map from the index space to traceless symmetric baseless number, but only for values
    # that are in both (i.e. not affected by tracelessness)
    bare_traceless_map = {tpl(x): i for i, x in enumerate(needed_ind[free_elements].unbind(0))}

    # Map from symmetric indices to traceless ones, but only for values that are in both.
    sym_traceless_map = {bare_sym_map[x]: j for x, j in bare_traceless_map.items()}

    # Initialize full map in "csr" form based on unconstrained elements.
    sym_traceless_csr = collections.defaultdict(list)
    for k, v in sym_traceless_map.items():
        sym_traceless_csr[k].append((v, 1))

    # Extend map for trace-constrained elements
    for x in map(tpl, constrained_ind):
        x_sym = bare_sym_map[x]
        new_sym_vals = []
        # Get the two other components x...00 and x_...11 (constraint is on x...22)
        for k in (0, 1):
            a = tuple(sorted(x[:-2] + (k, k)))
            aa = bare_sym_map[a]
            new_sym_vals.append(aa)

        # Write this is in terms of
        new_traceless_vals = collections.Counter()
        for aa in new_sym_vals:
            for aaa, v in sym_traceless_csr[aa]:
                new_traceless_vals[aaa] += -v
        new_traceless_vals = [(k, v) for k, v in new_traceless_vals.items()]
        sym_traceless_csr[x_sym].extend(new_traceless_vals)

    # Rewrite in coo form
    sym_traceless_coo = collections.Counter()
    for i, row in sym_traceless_csr.items():
        for j, val in row:
            sym_traceless_coo[i, j] += val

    # Convert into pytorch tensor
    ind, vals = zip(*list(sym_traceless_coo.items()))
    ind = torch.as_tensor(ind).T
    vals = torch.as_tensor(vals)
    xform2 = torch.sparse_coo_tensor(ind, vals, size=(xform.shape[1], 2 * ind_order + 1))

    # The full map is simply the map (cartesian index, symmetric) @ (symmetric, sym. traceless)
    full_xform = xform @ xform2.to_dense()

    # unflatten cartesian set.
    full_xform = full_xform.reshape(*((ndim,) * ind_order), -1)

    return full_xform


def norm_project(projector):
    rank = len(projector.shape) - 1
    flattened = projector.reshape(3**rank, 2 * rank + 1)
    svd = torch.linalg.svd(flattened, full_matrices=False)
    normed = svd.U @ svd.Vh
    reshaped = normed.reshape(((3,) * rank + (2 * rank + 1,)))
    return reshaped


def pinv_project(projector):
    rank = len(projector.shape) - 1
    flattened = projector.reshape(3**rank, 2 * rank + 1)
    pinv = torch.linalg.pinv(flattened).T
    pinv = pinv.reshape(
        *(3,) * rank,
        2 * rank + 1,
    )
    return pinv


c = {}
p = {}
c[0] = torch.ones((1, 1))
p[0] = torch.ones((1, 1))
for i in range(1, 4):
    c[i] = cartesian_irreducible_mapping(i).to(torch.get_default_dtype())
    p[i] = pinv_project(c[i])
cmaps = c
pmaps = p


class TensorExtractor(torch.nn.Module):
    def __init__(self, l_max, pmaps=pmaps):
        super().__init__()
        self.l_max = l_max
        all_pmaps = list(pmaps.values())
        self.pmaps = torch.nn.ParameterList(all_pmaps[: l_max + 1])
        for p in self.pmaps:
            p.requires_grad_(False)

    def forward(self, rhats):
        s = v = q = t = None
        s = torch.ones(rhats.shape[0], device=rhats.device, dtype=rhats.dtype).unsqueeze(1)
        with shared_intermediates():  # opt_einsum or null, see import logic
            if self.l_max > 0:
                v = rhats
            if self.l_max > 1:
                q = torch.einsum("ijk,bi,bj->bk", self.pmaps[2], rhats, rhats)
            if self.l_max > 2:
                t = torch.einsum("ijkl,bi,bj,bk->bl", self.pmaps[3], rhats, rhats, rhats)
        return s, v, q, t


# Note:!! Jitting this function with torch.jit.script does something bad:
# the training code slows down over time. (Some kind of leak)
# even though it is about 20% faster at first, it is very bad overall.
# Compiling does not help.
def calc_invariants(l_max: int, n_max: int, tensor_features, C: List[Tensor]):
    intermediates: Dict[str, Tensor] = {}
    invariants: Dict[int, Tensor] = {}
    im = intermediates
    I = invariants

    s = v = q = t = None

    if l_max == 0:
        s = tensor_features
    elif l_max == 1:
        s, v = tensor_features.split([1, 3], dim=-1)
    elif l_max == 2:
        s, v, q = tensor_features.split([1, 3, 5], dim=-1)
    elif l_max == 3:
        s, v, q, t = tensor_features.split([1, 3, 5, 7], dim=-1)
    else:
        raise ValueError(f"Invalid l_max:{l_max}")

    I[0] = s.squeeze()
    if l_max >= 1:
        if n_max >= 2:
            I[1] = (v * v).sum(dim=-1)  # no matmul required

    c2_flat = C[2].reshape(9, 5)
    if l_max >= 2:
        if n_max >= 2:
            # a=im['a'] = torch.einsum('bij,bik->bjk',q,q) # n = 2
            # a=im['a'] = torch.einsum('bc,bd,ijc,ikd->bjk',q,q,C[2],C[2]) # n = 2
            # This is probably not the best way b/c it invovles 3^3 = 27 element arrays
            # if we just do q,q outer then we have 25?? maybe not a big deal?
            c2_perm = C[2].permute(2, 0, 1).reshape(5, 9)
            a_half = torch.matmul(q, c2_perm).reshape(-1, 3, 3)
            a = im["a"] = (a_half.unsqueeze(3) * a_half.unsqueeze(2)).sum(dim=1)

            # I[2] = torch.einsum('bii->b',a) # n=2
            # I[2] = torch.einsum('bii->b',a) # No change!
            I[2] = torch.diagonal(a, 0, 1, 2).sum(dim=-1)

        if n_max >= 3:
            # I[3] = torch.einsum('bij,bij->b',a,q)
            # I[3] = torch.einsum('bij,bc,ijc->b',a,q,C[2])
            c2_flat = C[2].reshape(9, 5)
            a2 = torch.matmul(a.reshape(-1, 9), c2_flat)
            I[3] = (a2 * q).sum(dim=-1)

            # c=im['c'] = torch.einsum('bij,bi->bj',q,v)
            # c=im['c'] = torch.einsum('bc,bi,ijc->bj',q,v,C[2])
            c = im["c"] = (a_half * v.unsqueeze(2)).sum(dim=1)  # 2 = torch.matmul(q,c2_perm)

            # I[6] = torch.einsum('bi,bi->b',c,v)
            # I[6] = torch.einsum('bi,bi->b',c,v) # No change!
            I[6] = (c * v).sum(dim=-1)  # # no matmul required

        if n_max >= 4:
            # I[7] = torch.einsum('bi,bi->b',c,c) # no change!
            I[7] = (c * c).sum(dim=-1)  # no matmul required

    if l_max >= 3:
        if n_max >= 2:
            # b=im['b'] = torch.einsum('bijk,bijm->bkm',t,t) #n = 2
            # b=im['b'] = torch.einsum('bc,bd,ijkc,ijmd->bkm',t,t,C[3],C[3])

            # seems very inefficient!! now involves 81 elements
            c3_perm = C[3].permute(3, 0, 1, 2).reshape(7, 27)
            b_half = torch.matmul(t, c3_perm).reshape(-1, 3, 3, 3)
            b = im["b"] = (b_half.unsqueeze(4) * b_half.unsqueeze(3)).sum(dim=(1, 2))

            # I[4] = torch.einsum('bii->b',b)
            # I[4] = torch.einsum('bii->b',b) # no change
            I[4] = torch.diagonal(b, 0, 1, 2).sum(dim=-1)

        if n_max >= 3:
            # d=im['d'] = torch.einsum('bijk,bi->bjk',t,v)
            # d=im['d'] = torch.einsum('bc,bi,ijkc->bjk',t,v,C[3])
            d = im["d"] = (b_half * v.unsqueeze(1).unsqueeze(1)).sum(-1)

            # I[10] = torch.einsum('bij,bij->b',b,q)
            # I[10] = torch.einsum('bij,bc,ijc->b',b,q,C[2])
            b2 = torch.matmul(b.reshape(-1, 9), c2_flat)  # flattened b graph
            I[10] = (b2 * q).sum(dim=-1)

        if n_max >= 4:
            # I[5] = torch.einsum('bij,bij->b',b,b)
            # I[5] = torch.einsum('bij,bij->b',b,b) # no change
            I[5] = (b * b).sum(dim=(-1, -2))

            # I[8] = torch.einsum('bi,bj,bij->b',v,v,d) # no change
            I[8] = (v.unsqueeze(2) * v.unsqueeze(1) * d).sum(dim=(-1, -2))

            # I[9] = torch.einsum('bi,bj,bij->b',v,v,b) # no change
            I[9] = (v.unsqueeze(2) * v.unsqueeze(1) * b).sum(dim=(-1, -2))

            # I[11] = torch.einsum('bij,bij->b',b,a) # no change
            I[11] = (b * a).sum(dim=(-1, -2))

            # e=im['e'] = torch.einsum('bijk,bij->bk',t,q)
            # e=im['e'] = torch.einsum('bc,bd,ijkc,ijd->bk',t,q,C[3],C[2])
            c3rs = C[3].reshape(9, 21).T
            c3c2 = (c3rs @ c2_flat).reshape(3, 7, 5)  # shape 21,(9x9),5-> 3,7,5
            c3c2_rs = c3c2.permute(1, 0, 2).reshape(7, 15)
            e_half = torch.matmul(t, c3c2_rs).reshape(-1, 3, 5)
            e = im["e"] = (e_half * q.unsqueeze(1)).sum(dim=2)

            # I[12] = torch.einsum('bi,bi->b',e,e)
            # I[12] = torch.einsum('bi,bi->b',e,e) # no change
            I[12] = (e * e).sum(dim=-1)

    invariants = [invariants[i] for i in sorted(invariants.keys())]

    invars = torch.stack(invariants, dim=-1)
    return invars


class HopInvariantLayer(torch.nn.Module):
    def __init__(self, n_max, l_max, _cmaps=cmaps):
        super().__init__()
        self.l_max = l_max
        self.n_max = n_max
        self.cmaps = torch.nn.ParameterList(list(_cmaps.values()))
        for c in self.cmaps:
            c.requires_grad_(False)
        if self.n_max < 1 or self.n_max > 4:
            raise ValueError(f"Bad n: {n_max}")

        if self.l_max < 0 or self.l_max > 3:
            raise ValueError(f"Bad l: {l_max}")

    def extra_repr(self):
        return f"n_max={self.n_max}, l_max={self.l_max}"

    def forward(self, tensor_features):
        s = v = q = t = True
        if self.l_max > 0:
            assert v is not None
            if self.l_max > 1:
                assert q is not None
                if self.l_max > 2:
                    assert t is not None
        C = self.cmaps
        return calc_invariants(self.l_max, self.n_max, tensor_features, C)

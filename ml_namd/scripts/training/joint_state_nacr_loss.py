"""Tier-1 sign-inconsistency fix: JOINT PER-STATE phase loss for scaled NACR (dENACR).

Replaces the per-pair phaseless min (which grants each pair an INDEPENDENT sign -> the
pair-scramble gauge, r_ij != r_i r_j) with a JOINT minimization over per-STATE signs:

    L = mean_samples  min_{s_i=+-1}  sum_{i<j} || d_ij^ML - s_i s_j d_ij^QC ||^2 / n_pairs

The bilinear s_i s_j ties all pairs to a single consistent state gauge (2^{n-1} choices,
s_1 fixed +1), re-imposing the n(n-1)/2 - (n-1) cocycle constraints the per-pair loss drops.
Same math as the FSSH driver's `_phase_match`, applied at TRAINING against the QC label.

Standalone module + hippynn Loss node.  Run this file directly for the unit test.
"""
import itertools
import torch


def build_sign_matrix(pairs):
    """pairs: list of (i,j) 1-based excited-state index tuples, e.g. [(1,2),(1,3),(2,3)].
    Returns (signmat (G,n_pairs) float, states, gauges) with the first state fixed to +1."""
    states = sorted({s for p in pairs for s in p})
    free = states[1:]                                   # first state's sign fixed to +1
    gauges = []
    rows = []
    for bits in itertools.product((1, -1), repeat=len(free)):
        smap = {states[0]: 1}
        smap.update(dict(zip(free, bits)))
        rows.append([smap[i] * smap[j] for (i, j) in pairs])
        gauges.append(smap)
    return torch.tensor(rows, dtype=torch.float64), states, gauges


class JointStatePhaseModule(torch.nn.Module):
    """forward(*tensors) with tensors = interleaved (pred_0, true_0, pred_1, true_1, ...).
    Each pred/true is (B, D) or (B, 1, D) etc.; flattened to (B, D)."""
    def __init__(self, pairs):
        super().__init__()
        self.pairs = list(pairs)
        self.n_pairs = len(pairs)
        signmat, self.states, self.gauges = build_sign_matrix(pairs)
        self.register_buffer("signmat", signmat)        # (G, n_pairs)

    def forward(self, *tensors):
        assert len(tensors) == 2 * self.n_pairs, (len(tensors), self.n_pairs)
        preds = tensors[0::2]; trues = tensors[1::2]
        B = preds[0].shape[0]
        P = torch.stack([p.reshape(B, -1) for p in preds], dim=1)   # (B, np, D)
        T = torch.stack([t.reshape(B, -1) for t in trues], dim=1)   # (B, np, D)
        S = self.signmat.to(P.dtype)                                # (G, np)
        # diff[b,g,p,:] = P[b,p,:] - S[g,p]*T[b,p,:]
        diff = P.unsqueeze(1) - S[None, :, :, None] * T.unsqueeze(1)  # (B, G, np, D)
        per_gauge = (diff ** 2).sum(dim=(2, 3))                       # (B, G)
        best = per_gauge.min(dim=1).values                           # (B,)
        return best.mean() / self.n_pairs


# ------------------- hippynn Loss node (MODULE-LEVEL so it is picklable) -------------------
# NOTE: the node class MUST be defined at module scope (not inside a function) or hippynn's
# graph serialization fails with "Can't get local object ...". Imports guarded so the pure-torch
# unit test still runs where hippynn is absent.
try:
    from hippynn.graphs.nodes.base import SingleNode as _SingleNode
    from hippynn.graphs import IdxType as _IdxType
    from hippynn.graphs.indextypes import elementwise_compare_reduce as _ecr

    class JointStatePhaseLoss(_SingleNode):
        """Scalar hippynn loss node = JointStatePhaseModule over per-pair dENACR nodes.
        Mirrors `_BaseCompareLoss.of_node` (uses node.pred / node.true)."""
        index_state = _IdxType.Scalar

        def __init__(self, nacr_nodes, pairs, name="JointStatePhaseNACR"):
            parents = []
            for nnode in nacr_nodes:
                n = nnode.main_output
                p, t = _ecr(n.pred, n.true)
                parents += [p, t]
            super().__init__(name, tuple(parents), module=JointStatePhaseModule(pairs))
except Exception:  # hippynn not importable (e.g. the standalone unit test) -> node unavailable
    JointStatePhaseLoss = None


def make_joint_loss_node(nacr_nodes, pairs, name="JointStatePhaseNACR"):
    """Build the joint per-state phase loss node over the given per-pair dENACR prediction nodes."""
    if JointStatePhaseLoss is None:
        raise RuntimeError("hippynn not available; cannot build JointStatePhaseLoss node")
    return JointStatePhaseLoss(nacr_nodes, pairs, name)


# ------------------- unit test -------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    pairs = [(1, 2), (1, 3), (2, 3)]                     # S0-S3: 3 pairs, states {1,2,3}
    mod = JointStatePhaseModule(pairs)
    print("states:", mod.states, "| n_gauges:", len(mod.gauges), "| signmat:\n", mod.signmat.int().tolist())
    B, D = 200, 45

    # planted CONSISTENT state gauge: per-sample per-state signs s_i; base vectors b_ij
    s = torch.sign(torch.randn(B, 4)); s[:, 1] = 1.0     # s[:,k]=sign of state k (k=1..3 used); state1 fixed +1
    base = {p: torch.randn(B, D) for p in pairs}
    def dvec(sign_of):  # sign_of: (B, n_states+1) with sign per state
        return {p: sign_of[:, p[0]].unsqueeze(1) * sign_of[:, p[1]].unsqueeze(1) * base[p] for p in pairs}
    QC = dvec(s)                                          # QC in a consistent (scrambled-per-state) gauge
    ML = {p: base[p].clone() for p in pairs}             # ML predicts the canonical (+) gauge

    def loss_of(MLd, QCd):
        t = []
        for p in pairs: t += [MLd[p], QCd[p]]
        return mod(*t).item()

    # (1) consistent state gauge -> joint loss should be ~0 (min finds the right per-state signs)
    l_consistent = loss_of(ML, QC)
    # (2) PAIR-scramble: flip each pair's QC sign INDEPENDENTLY (not a state gauge) -> joint loss HIGH
    rpair = {p: torch.sign(torch.randn(B, 1)) for p in pairs}
    QC_pair = {p: rpair[p] * QC[p] for p in pairs}
    l_pairscramble = loss_of(ML, QC_pair)
    # per-pair phaseless loss (the OLD loss) on the pair-scramble -> should be ~0 (each pair fixes itself)
    def phaseless_perpair(MLd, QCd):
        tot = 0.0
        for p in pairs:
            a = ((MLd[p]-QCd[p])**2).sum(1); b = ((MLd[p]+QCd[p])**2).sum(1)
            tot += torch.minimum(a, b).mean()
        return (tot/len(pairs)).item()
    l_old_pairscramble = phaseless_perpair(ML, QC_pair)

    print(f"(1) consistent state gauge  : JOINT loss = {l_consistent:.3e}   (expect ~0)")
    print(f"(2) PAIR-scramble QC        : JOINT loss = {l_pairscramble:.3e}   (expect LARGE)")
    print(f"    same, OLD per-pair loss : phaseless  = {l_old_pairscramble:.3e}   (expect ~0 -> the bug)")
    ok = (l_consistent < 1e-9) and (l_pairscramble > 1.0) and (l_old_pairscramble < 1e-9)
    print("UNIT TEST", "PASS" if ok else "FAIL")

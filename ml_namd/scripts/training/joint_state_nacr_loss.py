"""Joint per-state phase loss for scaled nonadiabatic coupling vectors (dENACR).

Scientific purpose
------------------
A Model-B network predicts, for each ordered excited-state pair ``(i, j)``, the scaled
nonadiabatic coupling vector ``dENACR_ij`` (the derivative coupling multiplied by the
state-pair energy gap, flattened over atoms). The electronic wavefunction phase is
arbitrary at every geometry, so each coupling carries an unobservable overall sign. The
older training loss resolved that sign PER PAIR and INDEPENDENTLY, which lets the network
settle into a "pair-scramble" gauge where the per-pair signs ``r_ij`` do not factor through
a single set of per-state signs (``r_ij != s_i s_j``). The trajectory-surface-hopping driver
can only re-align a CONSISTENT per-state gauge, so a pair-scramble survives into the dynamics
and drives spurious relaxation (documented for keto acetylacetone).

This module supplies the fix at TRAINING time: a loss that is minimized over a single,
globally consistent set of per-state signs shared by every pair,

    L  =  mean_frames  min_{s_i in {+1,-1}}  ( 1 / n_pairs )  sum_{i<j} || dENACR_ij^ML  -  s_i s_j  dENACR_ij^QC ||^2 ,

with the first state's sign fixed to ``+1`` (leaving ``2^{n_states-1}`` candidate gauges).
The bilinear factor ``s_i s_j`` re-imposes the cocycle constraints that the per-pair loss
drops. This is the same construction the FSSH driver applies at inference (``_phase_match``),
here applied against the quantum-chemistry (QC) label during training.

Contents
--------
- :func:`build_sign_matrix`      enumerate the candidate per-state gauges as pair-sign rows.
- :class:`JointStatePhaseModule` the pure-torch loss (no hippynn dependency).
- :class:`JointStatePhaseLoss`   a hippynn scalar Loss node wrapping the module.
- :func:`make_joint_loss_node`   factory for the node from per-pair dENACR nodes.

Run this file directly (``python joint_state_nacr_loss.py``) to execute the unit test.

Behavior stability note
-----------------------
The public names in this module (module filename, ``JointStatePhaseModule``,
``JointStatePhaseLoss``, ``make_joint_loss_node``, and the default node name
``"JointStatePhaseNACR"``) are serialized into ``experiment_structure.pt`` when a model is
trained with this loss, and must resolve at load time. Do NOT rename them, and keep both the
module and the node class at MODULE scope (a function-local class breaks hippynn's graph
pickling with "Can't get local object ..."). The numerical reduction in
:meth:`JointStatePhaseModule.forward` is likewise load-bearing and must not change.
"""

from __future__ import annotations

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import itertools
from typing import Dict, List, Optional, Tuple

import torch

# A "gauge" is one assignment of a +/-1 sign to every excited state, with the first state
# pinned to +1. Represented as {state_index: sign}. The 1-based state indices match the
# excited-state labels used throughout the pipeline (S1, S2, ...).
Gauge = Dict[int, int]
Pair = Tuple[int, int]


def build_sign_matrix(pairs: List[Pair]) -> Tuple[torch.Tensor, List[int], List[Gauge]]:
    """Enumerate every candidate per-state gauge and its induced per-pair signs.

    Parameters
    ----------
    pairs
        Ordered 1-based excited-state index pairs, e.g. ``[(1, 2), (1, 3), (2, 3)]`` for the
        S1-S2 / S1-S3 / S2-S3 couplings.

    Returns
    -------
    sign_matrix
        Float64 tensor of shape ``(n_gauges, n_pairs)``. Row ``g`` holds the per-pair sign
        ``s_i * s_j`` induced by gauge ``g``. ``n_gauges = 2 ** (n_states - 1)`` because the
        first state's sign is fixed to ``+1``.
    states
        Sorted list of the distinct 1-based state indices appearing in ``pairs``.
    gauges
        The list of gauges (one ``{state: sign}`` dict per row of ``sign_matrix``).

    Notes
    -----
    Units/shapes: dimensionless signs; no physical units. The matrix is later cast to the
    prediction dtype inside the module.
    """
    states = sorted({state for pair in pairs for state in pair})
    free_states = states[1:]  # first state's sign is fixed to +1; the rest are free

    gauges: List[Gauge] = []
    pair_sign_rows: List[List[int]] = []
    for sign_bits in itertools.product((1, -1), repeat=len(free_states)):
        gauge: Gauge = {states[0]: 1}
        gauge.update(dict(zip(free_states, sign_bits)))
        pair_sign_rows.append([gauge[i] * gauge[j] for (i, j) in pairs])
        gauges.append(gauge)

    sign_matrix = torch.tensor(pair_sign_rows, dtype=torch.float64)
    return sign_matrix, states, gauges


class JointStatePhaseModule(torch.nn.Module):
    """Scalar joint per-state phase loss over a fixed set of state pairs.

    The module holds no learnable parameters; it only stores the candidate-gauge sign table
    as a buffer. ``forward`` receives the per-pair predicted and reference coupling tensors
    interleaved and returns the scalar loss ``L`` defined in the module docstring.
    """

    def __init__(self, pairs: List[Pair]) -> None:
        super().__init__()
        self.pairs: List[Pair] = list(pairs)
        self.n_pairs: int = len(pairs)
        sign_matrix, self.states, self.gauges = build_sign_matrix(self.pairs)
        # (n_gauges, n_pairs); registered so it moves with .to(device)/.cuda().
        self.register_buffer("signmat", sign_matrix)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Compute the joint per-state phase loss.

        Parameters
        ----------
        *tensors
            ``2 * n_pairs`` tensors, interleaved as
            ``(pred_0, true_0, pred_1, true_1, ...)``. Each entry has leading batch
            dimension ``n_frames`` and is flattened over the remaining axes to the coupling
            dimension ``D`` (``= 3 * n_atoms``); shapes ``(n_frames, D)`` or
            ``(n_frames, 1, D)`` are both accepted.

        Returns
        -------
        torch.Tensor
            Zero-dimensional loss tensor (differentiable), the batch mean of the best-gauge
            squared residual, divided by ``n_pairs``.

        Raises
        ------
        AssertionError
            If the number of tensors is not exactly ``2 * n_pairs``.
        """
        assert len(tensors) == 2 * self.n_pairs, (len(tensors), self.n_pairs)
        predicted_per_pair = tensors[0::2]
        reference_per_pair = tensors[1::2]

        n_frames = predicted_per_pair[0].shape[0]
        # Stack pairs into (n_frames, n_pairs, D); flatten trailing axes to the coupling dim D.
        predicted = torch.stack([p.reshape(n_frames, -1) for p in predicted_per_pair], dim=1)
        reference = torch.stack([t.reshape(n_frames, -1) for t in reference_per_pair], dim=1)

        sign_table = self.signmat.to(predicted.dtype)  # (n_gauges, n_pairs)

        # residual[frame, gauge, pair, :] = predicted[frame, pair, :]
        #                                   - sign_table[gauge, pair] * reference[frame, pair, :]
        residual = predicted.unsqueeze(1) - sign_table[None, :, :, None] * reference.unsqueeze(1)
        squared_residual_per_gauge = (residual ** 2).sum(dim=(2, 3))       # (n_frames, n_gauges)
        best_gauge_residual = squared_residual_per_gauge.min(dim=1).values  # (n_frames,)
        return best_gauge_residual.mean() / self.n_pairs


# ------------------------------------------------------------------------------------------
# hippynn Loss node. Defined at MODULE scope (a function-local class breaks graph pickling).
# The hippynn imports are guarded so the pure-torch unit test still runs where hippynn is
# absent; in that case the node symbol is set to None and the factory raises a clear error.
# ------------------------------------------------------------------------------------------
try:
    from hippynn.graphs.nodes.base import SingleNode as _SingleNode
    from hippynn.graphs import IdxType as _IdxType
    from hippynn.graphs.indextypes import elementwise_compare_reduce as _elementwise_compare_reduce

    class JointStatePhaseLoss(_SingleNode):
        """hippynn scalar Loss node wrapping :class:`JointStatePhaseModule`.

        Mirrors ``_BaseCompareLoss.of_node``: for each per-pair dENACR node it pulls the
        matched ``(pred, true)`` reduced tensors and feeds them, interleaved, to the module.
        """

        index_state = _IdxType.Scalar

        def __init__(self, nacr_nodes, pairs: List[Pair], name: str = "JointStatePhaseNACR") -> None:
            parents = []
            for nacr_node in nacr_nodes:
                output_node = nacr_node.main_output
                predicted_node, reference_node = _elementwise_compare_reduce(
                    output_node.pred, output_node.true
                )
                parents += [predicted_node, reference_node]
            super().__init__(name, tuple(parents), module=JointStatePhaseModule(pairs))

except Exception:  # hippynn not importable (e.g. the standalone unit test) -> node unavailable
    JointStatePhaseLoss = None  # type: ignore[assignment,misc]


def make_joint_loss_node(nacr_nodes, pairs: List[Pair], name: str = "JointStatePhaseNACR"):
    """Build the joint per-state phase loss node over the given per-pair dENACR nodes.

    Parameters
    ----------
    nacr_nodes
        The per-pair dENACR prediction nodes, in the same order as ``pairs``.
    pairs
        The 1-based state-index pairs (see :func:`build_sign_matrix`).
    name
        Node name (kept stable because it is serialized into the trained graph).

    Returns
    -------
    JointStatePhaseLoss
        The scalar loss node ready to be accumulated into the training loss.

    Raises
    ------
    RuntimeError
        If hippynn is not importable in this environment.
    """
    if JointStatePhaseLoss is None:
        raise RuntimeError("hippynn not available; cannot build JointStatePhaseLoss node")
    return JointStatePhaseLoss(nacr_nodes, pairs, name)


def _run_unit_test() -> bool:
    """Self-test the loss on planted gauges; returns True on PASS.

    Checks the three defining behaviors:
    (1) a globally consistent per-state gauge -> loss ~ 0 (the min finds the right signs);
    (2) an independent per-pair sign scramble -> joint loss LARGE (no single gauge fixes it);
    (3) the same scramble under the OLD per-pair phaseless loss -> ~ 0 (illustrating the bug
        the joint loss is designed to remove).
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    pairs: List[Pair] = [(1, 2), (1, 3), (2, 3)]  # S1-S2 / S1-S3 / S2-S3
    module = JointStatePhaseModule(pairs)
    print("states:", module.states, "| n_gauges:", len(module.gauges),
          "| signmat:\n", module.signmat.int().tolist())

    n_frames, coupling_dim = 200, 45  # coupling_dim = 3 * n_atoms for a 15-atom system
    per_state_sign = torch.sign(torch.randn(n_frames, 4))
    per_state_sign[:, 1] = 1.0  # fix state 1's sign to +1 (matches the gauge convention)
    base_vectors = {pair: torch.randn(n_frames, coupling_dim) for pair in pairs}

    def apply_state_gauge(sign_per_state: torch.Tensor) -> Dict[Pair, torch.Tensor]:
        return {
            pair: sign_per_state[:, pair[0]].unsqueeze(1)
            * sign_per_state[:, pair[1]].unsqueeze(1)
            * base_vectors[pair]
            for pair in pairs
        }

    reference = apply_state_gauge(per_state_sign)          # QC in a consistent per-state gauge
    predicted = {pair: base_vectors[pair].clone() for pair in pairs}  # ML in the canonical (+) gauge

    def joint_loss(pred: Dict[Pair, torch.Tensor], ref: Dict[Pair, torch.Tensor]) -> float:
        interleaved: List[torch.Tensor] = []
        for pair in pairs:
            interleaved += [pred[pair], ref[pair]]
        return module(*interleaved).item()

    loss_consistent = joint_loss(predicted, reference)

    pair_flip = {pair: torch.sign(torch.randn(n_frames, 1)) for pair in pairs}
    reference_pair_scrambled = {pair: pair_flip[pair] * reference[pair] for pair in pairs}
    loss_pair_scramble = joint_loss(predicted, reference_pair_scrambled)

    def phaseless_per_pair(pred: Dict[Pair, torch.Tensor], ref: Dict[Pair, torch.Tensor]) -> float:
        total = 0.0
        for pair in pairs:
            plus = ((pred[pair] - ref[pair]) ** 2).sum(1)
            minus = ((pred[pair] + ref[pair]) ** 2).sum(1)
            total += torch.minimum(plus, minus).mean()
        return (total / len(pairs)).item()

    loss_old_pair_scramble = phaseless_per_pair(predicted, reference_pair_scrambled)

    print(f"(1) consistent state gauge  : JOINT loss = {loss_consistent:.3e}   (expect ~0)")
    print(f"(2) PAIR-scramble QC        : JOINT loss = {loss_pair_scramble:.3e}   (expect LARGE)")
    print(f"    same, OLD per-pair loss : phaseless  = {loss_old_pair_scramble:.3e}   (expect ~0 -> the bug)")

    passed = (loss_consistent < 1e-9) and (loss_pair_scramble > 1.0) and (loss_old_pair_scramble < 1e-9)
    print("UNIT TEST", "PASS" if passed else "FAIL")
    return passed


def main() -> int:
    """Entry point for the standalone unit test. Returns a process exit code."""
    return 0 if _run_unit_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())

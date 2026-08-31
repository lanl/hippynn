"""Local direct equivariant scaled-NACR vector head for HIPPYNN (Model B, v1 — LEGACY).

Status
------
LEGACY / earliest of the three direct scaled-NACR heads, superseded by
:mod:`direct_nacr_head_v2` (adds gradient-difference-vector inputs) and
:mod:`direct_nacr_head_v3` (adds the energy-gap gate; the production head). Retained for
provenance and comparison.

Scientific purpose
------------------
Predict the scaled nonadiabatic coupling vector ``dENACR_ij`` directly from invariant pair
coefficients and geometry alone. A symmetric edge-coefficient layer produces one scalar per pair;
each is multiplied by the relative pair vector and added antisymmetrically over the pair to build an
equivariant per-atom vector, which is scattered to a dense per-system flattened coupling (with the
per-system mean optionally removed). Unlike v2/v3, this head uses no external GDV input.

This module intentionally lives outside the installed HIPPYNN package and wraps a scalar pair
coefficient model from HIPPYNN's public API.

Behavior stability note
-----------------------
Class names :class:`DirectScaledNACRLayer` / :class:`DirectScaledNACRNode`, the module filename,
the node ``input_names`` tuple, and the layer ``__init__`` keyword names are serialized into any
model trained with this head and must resolve at load time. Documentation + typing only; the tensor
operations are unchanged.
"""

from __future__ import annotations

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

from typing import Any, Dict, List, Optional, Sequence

import torch

from hippynn.graphs import IdxType  # type: ignore[import]
from hippynn.graphs.nodes.base import (  # type: ignore[import]
    AutoKw,
    ExpandParents,
    SingleNode,
    find_unique_relative,
)
from hippynn.graphs.nodes.inputs import PositionsNode  # type: ignore[import]
from hippynn.graphs.nodes.tags import (  # type: ignore[import]
    AtomIndexer,
    Network,
    PairIndexer,
)
from hippynn.layers import targets as target_modules  # type: ignore[import]


class DirectScaledNACRLayer(torch.nn.Module):
    """Predict scaled NACR vectors directly from invariant pair coefficients and geometry."""

    def __init__(
        self,
        feature_sizes: Sequence[int],
        n_dist: int,
        dist_soft_min: float,
        dist_soft_max: float,
        dist_hard_max: float,
        n_target: int = 1,
        symmetric_coeffs: bool = True,
        edge_scale: float = 0.5,
        enforce_zero_sum: bool = True,
        first_is_interacting: bool = False,
    ) -> None:
        """Build the v1 layer. Distances are in Angstrom; ``n_target`` = coupling channels."""
        super().__init__()
        self.n_target = int(n_target)
        self.edge_scale = float(edge_scale)
        self.enforce_zero_sum = bool(enforce_zero_sum)
        self.first_is_interacting = bool(first_is_interacting)

        coeff_feature_sizes = feature_sizes
        if self.first_is_interacting:
            coeff_feature_sizes = coeff_feature_sizes[1:]

        # Symmetric per-pair scalar coefficient layer (one scalar per pair per target channel).
        self.coeff_layer = target_modules.HBondSymmetric(
            feature_sizes=coeff_feature_sizes,
            n_dist=n_dist,
            dist_soft_min=dist_soft_min,
            dist_soft_max=dist_soft_max,
            dist_hard_max=dist_hard_max,
            positive=False,
            symmetric=symmetric_coeffs,
            antisymmetric=False,
            n_target=self.n_target,
        )

    def forward(
        self,
        all_features: List[torch.Tensor],
        pair_first: torch.Tensor,
        pair_second: torch.Tensor,
        pair_dist: torch.Tensor,
        pair_coord: torch.Tensor,
        coordinates: torch.Tensor,
        system_index: torch.Tensor,
        atom_index: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the geometry-only scaled-NACR vector.

        Returns ``(n_systems, n_target, 3*n_atoms)``, or ``(n_systems, 3*n_atoms)`` when
        ``n_target == 1``; removes the per-system atom-mean first if ``enforce_zero_sum``.
        """
        features = all_features
        if self.first_is_interacting:
            features = features[1:]

        # One scalar coefficient per pair, times the relative pair vector = equivariant edge term.
        coeff, _ = self.coeff_layer(features, pair_first, pair_second, pair_dist)
        edge_vec = self.edge_scale * coeff.unsqueeze(-1) * pair_coord.unsqueeze(1)

        n_real_atoms = system_index.shape[0]
        atom_vec = coordinates.new_zeros((n_real_atoms, self.n_target, 3))
        atom_vec.index_add_(0, pair_first, edge_vec)
        atom_vec.index_add_(0, pair_second, -edge_vec)

        n_systems = coordinates.shape[0]
        n_atoms = coordinates.shape[1]
        out = coordinates.new_zeros((n_systems, n_atoms, self.n_target, 3))
        out[system_index, atom_index] = atom_vec

        if self.enforce_zero_sum:
            out = out - out.mean(dim=1, keepdim=True)

        out = out.permute(0, 2, 1, 3)
        out = out.reshape(n_systems, self.n_target, 3 * n_atoms)

        if self.n_target == 1:
            out = out[:, 0, :]

        return out


class DirectScaledNACRNode(ExpandParents, AutoKw, SingleNode):
    """HIPPYNN graph node for local direct (geometry-only) scaled-NACR prediction (legacy; see v3)."""

    input_names = (
        "hier_features",
        "pair_first",
        "pair_second",
        "pair_dist",
        "pair_coord",
        "coordinates",
        "system_index",
        "atom_index",
    )
    index_state = IdxType.Systems
    auto_module_class = DirectScaledNACRLayer

    @parent_expander.match(Network, PositionsNode)
    def expansion0(self, net, positions, *, purpose=None, **kwargs):
        """Resolve the network's pair/atom indexers and assemble the parent tuple."""
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes

        pairfinder = find_unique_relative(net, PairIndexer, why_desc=purpose)
        atom_indexer = find_unique_relative(net, AtomIndexer, why_desc=purpose)

        return (
            net.main_output,
            pairfinder.pair_first,
            pairfinder.pair_second,
            pairfinder.pair_dist,
            pairfinder.pair_coord,
            positions,
            atom_indexer.system_index,
            atom_indexer.atom_index,
        )

    parent_expander.assertlen(8)
    parent_expander.get_main_outputs()

    def __init__(
        self,
        name: str,
        parents,
        module: Any = "auto",
        module_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.module_kwargs = module_kwargs or {}
        super().__init__(name, parents, module=module, **kwargs)

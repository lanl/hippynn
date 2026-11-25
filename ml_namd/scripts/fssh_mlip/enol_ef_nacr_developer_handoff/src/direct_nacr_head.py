"""Local direct equivariant scaled-NACR vector head.

This module intentionally lives outside the installed HIPPYNN package.  It
wraps a scalar pair coefficient model from HIPPYNN's public API and scatters the
coefficients against relative pair vectors to produce system-level NACR vectors.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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
    """Predict scaled NACR vectors directly from invariant pair coefficients."""

    def __init__(
        self,
        feature_sizes,
        n_dist,
        dist_soft_min,
        dist_soft_max,
        dist_hard_max,
        n_target=1,
        symmetric_coeffs=True,
        edge_scale=0.5,
        enforce_zero_sum=True,
        first_is_interacting=False,
    ):
        super().__init__()
        self.n_target = int(n_target)
        self.edge_scale = float(edge_scale)
        self.enforce_zero_sum = bool(enforce_zero_sum)
        self.first_is_interacting = bool(first_is_interacting)

        coeff_feature_sizes = feature_sizes
        if self.first_is_interacting:
            coeff_feature_sizes = coeff_feature_sizes[1:]

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
        all_features,
        pair_first,
        pair_second,
        pair_dist,
        pair_coord,
        coordinates,
        system_index,
        atom_index,
    ):
        features = all_features
        if self.first_is_interacting:
            features = features[1:]

        coeff, _ = self.coeff_layer(
            features,
            pair_first,
            pair_second,
            pair_dist,
        )

        edge_vec = (
            self.edge_scale
            * coeff.unsqueeze(-1)
            * pair_coord.unsqueeze(1)
        )

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
    """HIPPYNN graph node for local direct scaled-NACR vector prediction."""

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
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = (
                net.torch_module.feature_sizes
            )

        pairfinder = find_unique_relative(
            net,
            PairIndexer,
            why_desc=purpose,
        )
        atom_indexer = find_unique_relative(
            net,
            AtomIndexer,
            why_desc=purpose,
        )

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
        name,
        parents,
        module="auto",
        module_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.module_kwargs = module_kwargs or {}
        super().__init__(name, parents, module=module, **kwargs)

"""Local GDV-aware direct scaled-NACR vector heads.

This V2 module is intentionally separate from ``direct_nacr_head.py``.  It
implements oracle/reference-GDV experiments where frame-wise excited-state
force differences are provided as ordinary database inputs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from hippynn.graphs import IdxType  # type: ignore[import]
from hippynn.graphs.nodes.base import (  # type: ignore[import]
    AutoKw,
    ExpandParents,
    InputNode,
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


V2_MODES = (
    "gdv_local_gate",
    "gdv_pair_mixing",
    "gdv_pair_mixing_residual",
)


class DirectScaledNACRV2Layer(torch.nn.Module):
    """Predict scaled NACRs using reference GDV vectors as oracle inputs."""

    def __init__(
        self,
        feature_sizes,
        n_dist,
        dist_soft_min,
        dist_soft_max,
        dist_hard_max,
        mode="gdv_pair_mixing_residual",
        n_target=1,
        symmetric_coeffs=True,
        edge_scale=0.5,
        enforce_zero_sum=True,
        first_is_interacting=False,
        atom_feature_index=-1,
        atom_gate_hidden=None,
    ):
        super().__init__()
        if mode not in V2_MODES:
            raise ValueError(
                f"Unsupported DirectScaledNACRV2 mode {mode!r}. "
                f"Expected one of {V2_MODES}."
            )

        self.mode = mode
        self.n_target = int(n_target)
        self.edge_scale = float(edge_scale)
        self.enforce_zero_sum = bool(enforce_zero_sum)
        self.first_is_interacting = bool(first_is_interacting)
        self.atom_feature_index = int(atom_feature_index)

        coeff_feature_sizes = feature_sizes
        if self.first_is_interacting:
            coeff_feature_sizes = coeff_feature_sizes[1:]

        self.uses_local_gate = self.mode == "gdv_local_gate"
        self.uses_gdv_mixing = self.mode in (
            "gdv_pair_mixing",
            "gdv_pair_mixing_residual",
        )
        self.uses_geom_residual = self.mode == "gdv_pair_mixing_residual"

        if self.uses_local_gate:
            atom_feature_size = int(coeff_feature_sizes[self.atom_feature_index])
            hidden = (
                int(atom_gate_hidden)
                if atom_gate_hidden is not None
                else atom_feature_size
            )
            self.atom_gate_net = torch.nn.Sequential(
                torch.nn.Linear(atom_feature_size, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, self.n_target),
            )
        else:
            self.atom_gate_net = None

        if self.uses_gdv_mixing:
            self.gdv_coeff_layer = target_modules.HBondSymmetric(
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
        else:
            self.gdv_coeff_layer = None

        if self.uses_geom_residual:
            self.geom_coeff_layer = target_modules.HBondSymmetric(
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
        else:
            self.geom_coeff_layer = None

    def _real_atom_gdv(self, gdv_flat, coordinates, system_index, atom_index):
        """Map system-level flattened GDV input onto real atoms."""
        n_systems = coordinates.shape[0]
        n_atoms = coordinates.shape[1]
        gdv_system = gdv_flat.reshape(n_systems, n_atoms, 3)
        return gdv_system[system_index, atom_index]

    def _scatter_real_atoms(
        self,
        atom_vec,
        coordinates,
        system_index,
        atom_index,
    ):
        """Scatter real-atom vectors to padded system-major output."""
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
        gdv_flat,
    ):
        features = all_features
        if self.first_is_interacting:
            features = features[1:]

        gdv_atoms = self._real_atom_gdv(
            gdv_flat,
            coordinates,
            system_index,
            atom_index,
        )
        n_real_atoms = system_index.shape[0]
        atom_vec = coordinates.new_zeros((n_real_atoms, self.n_target, 3))

        if self.uses_local_gate:
            gate_features = features[self.atom_feature_index]
            gate = self.atom_gate_net(gate_features)
            atom_vec = atom_vec + gate.unsqueeze(-1) * gdv_atoms.unsqueeze(1)

        if self.uses_gdv_mixing:
            coeff, _ = self.gdv_coeff_layer(
                features,
                pair_first,
                pair_second,
                pair_dist,
            )
            gdv_delta = gdv_atoms[pair_second] - gdv_atoms[pair_first]
            edge_vec = (
                self.edge_scale
                * coeff.unsqueeze(-1)
                * gdv_delta.unsqueeze(1)
            )
            atom_vec.index_add_(0, pair_first, edge_vec)
            atom_vec.index_add_(0, pair_second, -edge_vec)

        if self.uses_geom_residual:
            coeff, _ = self.geom_coeff_layer(
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
            atom_vec.index_add_(0, pair_first, edge_vec)
            atom_vec.index_add_(0, pair_second, -edge_vec)

        return self._scatter_real_atoms(
            atom_vec,
            coordinates,
            system_index,
            atom_index,
        )


class DirectScaledNACRV2Node(ExpandParents, AutoKw, SingleNode):
    """HIPPYNN graph node for V2 oracle-GDV scaled-NACR prediction."""

    input_names = (
        "hier_features",
        "pair_first",
        "pair_second",
        "pair_dist",
        "pair_coord",
        "coordinates",
        "system_index",
        "atom_index",
        "gdv_flat",
    )
    index_state = IdxType.Systems
    auto_module_class = DirectScaledNACRV2Layer

    @parent_expander.match(Network, PositionsNode, InputNode)
    def expansion0(self, net, positions, gdv_flat, *, purpose=None, **kwargs):
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
            gdv_flat,
        )

    parent_expander.assertlen(9)
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

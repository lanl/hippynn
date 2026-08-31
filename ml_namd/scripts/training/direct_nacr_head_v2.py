"""GDV-aware direct scaled-NACR vector head for HIPPYNN (Model B, v2 — LEGACY).

Status
------
LEGACY / superseded by :mod:`direct_nacr_head_v3`. v3 is the production head; it equals this v2
head plus a zero-initialized per-frame energy-gap gate. This module is retained for provenance and
comparison and is intentionally separate from both v1 (:mod:`direct_nacr_head`) and v3.

Scientific purpose
------------------
Predict the scaled nonadiabatic coupling vector ``dENACR_ij`` from network features together with
reference gradient-difference vectors (GDV) supplied as ordinary database inputs ("oracle" GDV
experiments). Three assembly modes are available (see :data:`V2_MODES`):
- ``gdv_local_gate``           : a per-atom learned gate applied to each atom's own GDV;
- ``gdv_pair_mixing``          : symmetric edge coefficients mix GDV differences between paired atoms;
- ``gdv_pair_mixing_residual`` : the above plus a geometry (pair-coordinate) residual term.

The construction is equivariant: edge contributions are added antisymmetrically over each pair and
the per-system mean is optionally removed (no net translation).

Behavior stability note
-----------------------
Class names :class:`DirectScaledNACRV2Layer` / :class:`DirectScaledNACRV2Node`, the module filename,
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

from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# Assembly modes (see the module docstring). Identical set to v3's, since v3 = v2 + gap gate.
V2_MODES: Tuple[str, ...] = (
    "gdv_local_gate",
    "gdv_pair_mixing",
    "gdv_pair_mixing_residual",
)


class DirectScaledNACRV2Layer(torch.nn.Module):
    """Predict scaled NACRs from network features and reference (oracle) GDV inputs."""

    def __init__(
        self,
        feature_sizes: Sequence[int],
        n_dist: int,
        dist_soft_min: float,
        dist_soft_max: float,
        dist_hard_max: float,
        mode: str = "gdv_pair_mixing_residual",
        n_target: int = 1,
        symmetric_coeffs: bool = True,
        edge_scale: float = 0.5,
        enforce_zero_sum: bool = True,
        first_is_interacting: bool = False,
        atom_feature_index: int = -1,
        atom_gate_hidden: Optional[int] = None,
    ) -> None:
        """Build the v2 layer. See :data:`V2_MODES` for ``mode``; distances are in Angstrom.

        Raises
        ------
        ValueError
            If ``mode`` is not one of :data:`V2_MODES`.
        """
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

        # Which sub-terms this mode activates.
        self.uses_local_gate = self.mode == "gdv_local_gate"
        self.uses_gdv_mixing = self.mode in ("gdv_pair_mixing", "gdv_pair_mixing_residual")
        self.uses_geom_residual = self.mode == "gdv_pair_mixing_residual"

        # Per-atom gate MLP (gdv_local_gate mode only).
        if self.uses_local_gate:
            atom_feature_size = int(coeff_feature_sizes[self.atom_feature_index])
            hidden = int(atom_gate_hidden) if atom_gate_hidden is not None else atom_feature_size
            self.atom_gate_net = torch.nn.Sequential(
                torch.nn.Linear(atom_feature_size, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, self.n_target),
            )
        else:
            self.atom_gate_net = None

        # Symmetric edge-coefficient layer that mixes GDV differences along pairs.
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

        # Second symmetric edge-coefficient layer for the geometry residual term.
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

    def _real_atom_gdv(
        self,
        gdv_flat: torch.Tensor,
        coordinates: torch.Tensor,
        system_index: torch.Tensor,
        atom_index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather each real atom's GDV vector from the flattened per-system input.

        ``gdv_flat`` ``(n_systems, 3*n_atoms)`` -> ``(n_real_atoms, 3)``.
        """
        n_systems = coordinates.shape[0]
        n_atoms = coordinates.shape[1]
        gdv_system = gdv_flat.reshape(n_systems, n_atoms, 3)
        return gdv_system[system_index, atom_index]

    def _scatter_real_atoms(
        self,
        atom_vec: torch.Tensor,
        coordinates: torch.Tensor,
        system_index: torch.Tensor,
        atom_index: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter per-real-atom vectors to the dense per-system flattened coupling.

        Returns ``(n_systems, n_target, 3*n_atoms)``, or ``(n_systems, 3*n_atoms)`` when
        ``n_target == 1``; removes the per-system atom-mean first if ``enforce_zero_sum``.
        """
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
        all_features: List[torch.Tensor],
        pair_first: torch.Tensor,
        pair_second: torch.Tensor,
        pair_dist: torch.Tensor,
        pair_coord: torch.Tensor,
        coordinates: torch.Tensor,
        system_index: torch.Tensor,
        atom_index: torch.Tensor,
        gdv_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the GDV-aware scaled-NACR vector (no gap gate; see v3 for the gap-conditioned head)."""
        features = all_features
        if self.first_is_interacting:
            features = features[1:]

        gdv_atoms = self._real_atom_gdv(gdv_flat, coordinates, system_index, atom_index)
        n_real_atoms = system_index.shape[0]
        atom_vec = coordinates.new_zeros((n_real_atoms, self.n_target, 3))

        # (a) per-atom local gate on each atom's own GDV
        if self.uses_local_gate:
            gate_features = features[self.atom_feature_index]
            gate = self.atom_gate_net(gate_features)
            atom_vec = atom_vec + gate.unsqueeze(-1) * gdv_atoms.unsqueeze(1)

        # (b) equivariant edge mixing of GDV differences (antisymmetric add over each pair)
        if self.uses_gdv_mixing:
            coeff, _ = self.gdv_coeff_layer(features, pair_first, pair_second, pair_dist)
            gdv_delta = gdv_atoms[pair_second] - gdv_atoms[pair_first]
            edge_vec = self.edge_scale * coeff.unsqueeze(-1) * gdv_delta.unsqueeze(1)
            atom_vec.index_add_(0, pair_first, edge_vec)
            atom_vec.index_add_(0, pair_second, -edge_vec)

        # (c) geometry residual: same edge construction using pair coordinates
        if self.uses_geom_residual:
            coeff, _ = self.geom_coeff_layer(features, pair_first, pair_second, pair_dist)
            edge_vec = self.edge_scale * coeff.unsqueeze(-1) * pair_coord.unsqueeze(1)
            atom_vec.index_add_(0, pair_first, edge_vec)
            atom_vec.index_add_(0, pair_second, -edge_vec)

        return self._scatter_real_atoms(atom_vec, coordinates, system_index, atom_index)


class DirectScaledNACRV2Node(ExpandParents, AutoKw, SingleNode):
    """HIPPYNN graph node for v2 oracle-GDV scaled-NACR prediction (legacy; see v3)."""

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
            gdv_flat,
        )

    parent_expander.assertlen(9)
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

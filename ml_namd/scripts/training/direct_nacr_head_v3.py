"""Gap-aware (GDV + energy-gap) scaled-NACR vector head for HIPPYNN (Model B, v3).

Scientific purpose
------------------
Model B predicts the scaled nonadiabatic coupling vector ``dENACR_ij`` (the derivative
coupling between excited states ``i`` and ``j``, scaled by their energy gap and flattened over
atoms) for each requested state pair. This "v3" head builds on v2
(:mod:`direct_nacr_head_v2`) by adding a per-frame energy-gap conditioning gate.

Construction (v3 = v2 + gap gate):
- v2 assembles an equivariant per-atom vector from the gradient-difference vectors (GDV) and,
  in the residual mode, from geometry (pair coordinates), using symmetric edge coefficients.
- v3 multiplies that vector by a learned scalar gate ``gate = 1 + MLP(gap)``, where ``gap`` is
  the invariant state-pair energy gap ``sE_j - sE_i``. The MLP's final layer is zero-initialized,
  so at initialization ``gate == 1`` and v3 is bit-identical to v2; training then learns a
  gap-dependent multiplicative correction to the coupling magnitude.

Equivariance: the gap is a per-system invariant scalar, so multiplying the already-equivariant
per-atom output by it preserves rotational equivariance. (For the raw-NACR repurpose, the same
gap becomes the explicit multiplier, ``dENACR = gap * raw_NACR``.)

This module is intentionally separate from v2; v2 is left untouched.

Behavior stability note
-----------------------
The class names :class:`DirectScaledNACRV3Layer` and :class:`DirectScaledNACRV3Node`, the module
filename, the node ``input_names`` tuple, and the layer ``__init__`` keyword names are all
serialized into ``experiment_structure.pt`` at training time and resolved when a trained Model B
is loaded for FSSH. Do NOT rename them or change ``forward`` numerics. This file is documentation
+ typing only; the tensor operations are unchanged.
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


# Supported assembly modes. Each names how the equivariant per-atom coupling vector is built:
#   gdv_local_gate           : per-atom learned gate applied directly to the atom's GDV
#   gdv_pair_mixing          : edge (pair) coefficients mix GDV differences between atoms
#   gdv_pair_mixing_residual : gdv_pair_mixing PLUS a geometry (pair-coordinate) residual term
#   geom_residual_only       : the geometry (pair-coordinate) residual term ALONE, no GDV mixing
V3_MODES: Tuple[str, ...] = (
    "gdv_local_gate",
    "gdv_pair_mixing",
    "gdv_pair_mixing_residual",
    "geom_residual_only",
)


class DirectScaledNACRV3Layer(torch.nn.Module):
    """v2 assembly layer plus a zero-initialized per-frame energy-gap gate.

    The forward pass builds an equivariant per-real-atom vector of shape
    ``(n_real_atoms, n_target, 3)`` from network features, GDVs, and (in the residual mode)
    pair geometry, then multiplies it by ``gate = 1 + MLP(gap)`` and scatters it back to a
    dense per-system flattened output of shape ``(n_systems, n_target, 3 * n_atoms)`` (or
    ``(n_systems, 3 * n_atoms)`` when ``n_target == 1``).
    """

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
        gap_hidden: int = 16,
    ) -> None:
        """Build the v3 layer.

        Parameters
        ----------
        feature_sizes
            Per-interaction-layer feature widths from the HIPPYNN network.
        n_dist, dist_soft_min, dist_soft_max, dist_hard_max
            Radial-basis settings for the symmetric edge-coefficient layers (Angstrom).
        mode
            One of :data:`V3_MODES` (see the module comment).
        n_target
            Number of coupling channels predicted (one per state pair handled by this head).
        symmetric_coeffs
            Whether the edge-coefficient layer is symmetric under pair-atom exchange.
        edge_scale
            Scalar multiplier on each edge contribution.
        enforce_zero_sum
            If True, remove the per-system mean so the coupling carries no net translation.
        first_is_interacting
            If True, drop the first feature block (a HIPPYNN convention toggle).
        atom_feature_index
            Which feature block feeds the per-atom gate in ``gdv_local_gate`` mode.
        atom_gate_hidden
            Hidden width of the per-atom gate MLP (defaults to the atom feature size).
        gap_hidden
            Hidden width of the energy-gap gate MLP.

        Raises
        ------
        ValueError
            If ``mode`` is not one of :data:`V3_MODES`.
        """
        super().__init__()
        if mode not in V3_MODES:
            raise ValueError(
                f"Unsupported DirectScaledNACRV3 mode {mode!r}. "
                f"Expected one of {V3_MODES}."
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
        self.uses_geom_residual = self.mode in ("gdv_pair_mixing_residual", "geom_residual_only")

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

        # Energy-gap gate: MLP(gap) -> per-target scalar. The final layer is zero-initialized so
        # that gate == 1 at init (v3 == v2), after which it learns a gap-dependent correction.
        gap_hidden_width = int(gap_hidden)
        self.gap_gate_net = torch.nn.Sequential(
            torch.nn.Linear(1, gap_hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(gap_hidden_width, self.n_target),
        )
        torch.nn.init.zeros_(self.gap_gate_net[-1].weight)
        torch.nn.init.zeros_(self.gap_gate_net[-1].bias)

    def _real_atom_gdv(
        self,
        gdv_flat: torch.Tensor,
        coordinates: torch.Tensor,
        system_index: torch.Tensor,
        atom_index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather each real atom's GDV vector.

        ``gdv_flat`` is ``(n_systems, 3 * n_atoms)``; it is reshaped to ``(n_systems, n_atoms, 3)``
        and indexed by ``(system_index, atom_index)`` to give ``(n_real_atoms, 3)``.
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
        """Scatter per-real-atom vectors into a dense per-system flattened coupling.

        Parameters
        ----------
        atom_vec
            ``(n_real_atoms, n_target, 3)`` per-atom coupling contributions.

        Returns
        -------
        torch.Tensor
            ``(n_systems, n_target, 3 * n_atoms)``, or ``(n_systems, 3 * n_atoms)`` when
            ``n_target == 1``. If ``enforce_zero_sum`` is set, the per-system atom-mean is
            removed first (no net translational component).
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
        gap_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the gap-conditioned scaled-NACR vector.

        Shapes: ``coordinates`` ``(n_systems, n_atoms, 3)``; ``pair_*`` are per-edge tensors;
        ``system_index``/``atom_index`` ``(n_real_atoms,)``; ``gdv_flat`` ``(n_systems, 3*n_atoms)``;
        ``gap_flat`` ``(n_systems,)`` or ``(n_systems, 1)``. Returns the flattened coupling from
        :meth:`_scatter_real_atoms`.
        """
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

        # (d) energy-gap conditioning: invariant scalar -> multiplicative per-target gate
        gap = gap_flat.reshape(coordinates.shape[0], 1).to(atom_vec.dtype)  # (n_systems, 1)
        gate = 1.0 + self.gap_gate_net(gap)                                 # (n_systems, n_target)
        gate_atoms = gate[system_index]                                     # (n_real_atoms, n_target)
        atom_vec = atom_vec * gate_atoms.unsqueeze(-1)                      # (n_real_atoms, n_target, 3)

        return self._scatter_real_atoms(atom_vec, coordinates, system_index, atom_index)


class DirectScaledNACRV3Node(ExpandParents, AutoKw, SingleNode):
    """HIPPYNN graph node for GDV + gap-aware scaled-NACR prediction.

    Wires the network output, pair indexer, atom indexer, positions, and the two input nodes
    (``gdv_flat``, ``gap_flat``) to a :class:`DirectScaledNACRV3Layer`. The ``input_names`` tuple
    and the parent-expander wiring are load-bearing (serialized into the trained graph).
    """

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
        "gap_flat",
    )
    index_state = IdxType.Systems
    auto_module_class = DirectScaledNACRV3Layer

    @parent_expander.match(Network, PositionsNode, InputNode, InputNode)
    def expansion0(self, net, positions, gdv_flat, gap_flat, *, purpose=None, **kwargs):
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
            gap_flat,
        )

    parent_expander.assertlen(10)
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

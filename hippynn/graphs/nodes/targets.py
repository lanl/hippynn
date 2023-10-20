"""
Nodes for prediction of variables from network features.
"""

from .base import MultiNode, AutoKw, ExpandParents, find_unique_relative, _BaseNode
from .tags import AtomIndexer, Network, PairIndexer, HAtomRegressor, Charges, Energies
from .indexers import PaddingIndexer
from ..indextypes import IdxType, index_type_coercion
from ...layers import targets as target_modules


class HEnergyNode(Energies, HAtomRegressor, AutoKw, ExpandParents, MultiNode):
    """
    Predict a system-level scalar such as energy from a sum over local components.
    """

    _input_names = "hier_features", "mol_index", "n_molecules"
    _output_names = "mol_energy", "atom_energies", "energy_terms", "hierarchicality", "atom_hier", "mol_hier", "batch_hier"
    _main_output = "mol_energy"
    _output_index_states = IdxType.Molecules, IdxType.Atoms, None, IdxType.Molecules, IdxType.Atoms, IdxType.Molecules, IdxType.Scalar
    _auto_module_class = target_modules.HEnergy

    @_parent_expander.match(Network)
    def expansion0(self, net, **kwargs):
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes
        pdindexer = find_unique_relative(net, AtomIndexer)
        return net, pdindexer.mol_index, pdindexer.n_molecules

    def __init__(self, name, parents, first_is_interacting=False, module="auto", module_kwargs=None, **kwargs):
        self.module_kwargs = {"first_is_interacting": first_is_interacting}
        if module_kwargs is not None:
            self.module_kwargs = {**self.module_kwargs, **module_kwargs}
        parents = self.expand_parents(parents, **kwargs)
        super().__init__(name, parents, module=module, **kwargs)


class HChargeNode(Charges, HAtomRegressor, AutoKw, ExpandParents, MultiNode):
    """
    Predict an atom-level scalar such as charge from local features.
    """

    _input_names = ("hier_features",)
    _output_names = "atom_charges", "partial_sums", "charge_hierarchality"
    _main_output = "atom_charges"
    _output_index_states = IdxType.Atoms, None, IdxType.Atoms
    _auto_module_class = target_modules.HCharge

    @_parent_expander.match(Network)
    def expansion0(self, net, **kwargs):
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes
        return (net.main_output,)

    def __init__(self, name, parents, module="auto", module_kwargs=None, **kwargs):
        self.module_kwargs = {} if module_kwargs is None else module_kwargs
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class LocalChargeEnergy(Energies, ExpandParents, HAtomRegressor, MultiNode):
    _input_names = "charges", "features", "mol_index", "n_molecules"
    _output_names = "molenergies", "atomenergies"
    _main_output = "molenergies"
    _output_index_states = IdxType.Molecules, IdxType.Atoms
    _auto_module_class = target_modules.LocalChargeEnergy

    @_parent_expander.match(_BaseNode, Network)
    def expansion0(self, charge, network, **kwargs):
        charge = index_type_coercion(charge.main_output, IdxType.Atoms)
        pdxer = find_unique_relative(network, PaddingIndexer)
        return charge, network.main_output, pdxer.mol_index, pdxer.n_molecules

    def __init__(self, name, parents, **kwargs):
        parents = self.parents_expand(parents)
        super().__init__(name, parents, **kwargs)


class HBondNode(ExpandParents, AutoKw, MultiNode):
    """
    Predict an pair-level scalar such as bond order from local features on both atoms
    """

    _auto_module_class = target_modules.HBondSymmetric
    _output_names = "bonds", "bond_hierarchality"
    _output_index_states = IdxType.Pair, IdxType.Pair
    _input_names = "features", "pair_first", "pair_second", "pair_dist"
    _main_output = "bonds"
 
    @_parent_expander.matchlen(1)
    def expand0(self, features, *, purpose, **kwargs):
        pairfinder = find_unique_relative(features, PairIndexer, why_desc=purpose)
        return features, pairfinder

    @_parent_expander.matchlen(2)
    def expand1(self, features, pairfinder, **kwargs):
        return features.main_output, pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist

    def __init__(self, name, parents, module="auto", module_kwargs=None, **kwargs):
        self.module_kwargs = {} if module_kwargs is None else module_kwargs
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)



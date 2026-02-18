"""
Nodes for physics transformations
"""

from ....layers import indexers as index_layers
from ....layers import pairs as pair_layers
from ....layers import physics as physics_layers
from ...indextypes import IdxType, elementwise_compare_reduce, index_type_coercion
from ..base import (
    AutoKw,
    AutoNoKw,
    ExpandParents,
    MultiNode,
    SingleNode,
    Node,
    find_unique_relative,
)
from ..base.node_functions import NodeNotFound
from ..indexers import AtomIndexer, PaddingIndexer, acquire_encoding_padding
from ..inputs import PositionsNode, SpeciesNode
from ..pairs import OpenPairIndexer
from ..tags import Charges, Encoder, Energies, PairIndexer
from ...._deprecations import _DeprecatedNamesMixin


class VecMag(ExpandParents, AutoNoKw, SingleNode):
    input_names = ("vector",)
    auto_module_class = physics_layers.VecMag
    index_state = IdxType.Unlabeled

    @parent_expander.match(Node, Node)
    def expansion2(self, vector, helper, *, purpose, **kwargs):
        # This somewhat strange construction allows us to
        # find a padding indexer if the vector is detached from the padding indexer.
        vector, helper = elementwise_compare_reduce(vector, helper)
        return (vector,)

    parent_expander.assertlen(1)
    parent_expander.get_main_outputs()

    def __init__(self, name, parents, module="auto", _helper=None, **kwargs):
        parents = self.expand_parents(parents)
        self.index_state = parents[0].index_state
        assert len(parents) == 1, "Improper number of parents for {}".format(self.__class__.__name__)
        super().__init__(name, parents, module=module, **kwargs)


class PerAtom(ExpandParents, AutoNoKw, SingleNode):
    input_names = "features", "species"
    index_state = IdxType.Systems
    auto_module_class = physics_layers.PerAtom

    @parent_expander.match(Node)
    def expansion0(self, features, *, purpose, **kwargs):
        return features, find_unique_relative(features, SpeciesNode, purpose)

    @parent_expander.match(Node, Node)
    def expansion1(self, features, species, **kwargs):
        features = features.main_output
        assert (
            features.index_state == IdxType.Systems
        ), "Can only calculate Per Atom averages on Molecular quantities"
        return features, species

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)



class AtomToMolSummer(ExpandParents, AutoNoKw, SingleNode):
    input_names = "features", "system_index", "n_systems"
    auto_module_class = index_layers.MolSummer
    index_state = IdxType.Systems

    @parent_expander.match(Node)
    def expansion0(self, features, **kwargs):
        pdxer = find_unique_relative(features, AtomIndexer, why_desc="Generating Molecular summer")
        return features, pdxer

    @parent_expander.match(Node, AtomIndexer)
    def expansion1(self, features, pdxer, **kwargs):
        return features, pdxer.system_index, pdxer.n_systems

    parent_expander.assertlen(3)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


# TODO: This seems broken for parent expanders, check the signature of the layer.
class BondToMolSummmer(ExpandParents, AutoNoKw, SingleNode):
    input_names = "pairfeatures", "system_index", "n_systems", "pair_first"
    auto_module_class = pair_layers.MolPairSummer
    index_state = IdxType.Systems

    @parent_expander.match(Node)
    def expansion0(self, features, *, purpose, **kwargs):
        pdxer = find_unique_relative(features, AtomIndexer, why_desc=purpose)
        pair_idxer = find_unique_relative(features, PairIndexer, why_desc=purpose)
        return features, pdxer, pair_idxer

    @parent_expander.match(Node, AtomIndexer, PairIndexer)
    def expansion1(self, features, pdxer, pair_idxer, **kwargs):
        return features, pdxer.system_index, pdxer.n_systems, pair_idxer.pair_first

    @parent_expander.match(Node, Node, Node, Node, Node)
    def expansion2(self, features, system_index, n_systems, **kwargs):
        return index_type_coercion(features.main_output, IdxType.Pairs), system_index, n_systems

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class CombineEnergyNode(AutoNoKw, Energies,  ExpandParents, MultiNode, _DeprecatedNamesMixin):
    """
    Combines Local atom energies from different Energy Nodes.
    """
    _DEPRECATED_NAMES = {"mol_energy": "system_energy"}

    # Used for combining coulomb energies with local energies in LAMMPS interface

    input_names = "input_atom_energy_1", "input_atom_energy_2", "system_index", "n_systems"
    output_names = "system_energy", "atom_energies"
    main_output_name = "system_energy"
    output_index_states = (
        IdxType.Systems,
        IdxType.Atoms,
    )
    auto_module_class = physics_layers.CombineEnergy

    @parent_expander.match(Node, Energies)
    def expansion0(self, energy_1, energy_2, **kwargs):
        return energy_1, energy_2.atom_energies

    @parent_expander.match(Energies, Node)
    def expansion0(self, energy_1, energy_2, **kwargs):
        return energy_1.atom_energies, energy_2

    @parent_expander.match(Node, Node)
    def expansion1(self, energy_1, energy_2, **kwargs):
        pdindexer = find_unique_relative([energy_1, energy_2], AtomIndexer, why_desc="Generating CombineEnergies")
        return energy_1, energy_2, pdindexer

    @parent_expander.match(Node, Node, PaddingIndexer)
    def expansion2(self, energy_1, energy_2, pdindexer, **kwargs):
        return energy_1, energy_2, pdindexer.system_index, pdindexer.n_systems

    parent_expander.assertlen(4)
    parent_expander.require_idx_states(IdxType.Atoms, IdxType.Atoms, None, None)



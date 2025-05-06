"""
Nodes for networks.
"""
from .tags import Encoder, PairIndexer, Network, AtomIndexer
from .base import _BaseNode, AutoKw, ExpandParents, SingleNode
from .base.multi import IndexNode
from .indexers import OneHotEncoder, PaddingIndexer, acquire_encoding_padding
from .pairs import OpenPairIndexer, PeriodicPairIndexer
from .inputs import SpeciesNode, PositionsNode, CellNode
from ..indextypes import IdxType
from ... import networks as network_modules


class DefaultNetworkExpansion(ExpandParents):
    @_parent_expander.match(SpeciesNode, PositionsNode)
    @_parent_expander.match(SpeciesNode, PositionsNode, CellNode)
    def expansion0(self, species, *other_parents, species_set, purpose, **kwargs):
        """
        Finds or sets up a default one-hot encoder if species are passed as first argument.

        :return: (encoder,padding_indexer,*other_parents)

        """
        encoder, pidxer = acquire_encoding_padding(species, species_set, purpose=purpose)
        return (encoder, pidxer, *other_parents)

    @_parent_expander.match(Encoder, AtomIndexer, PositionsNode)
    @_parent_expander.match(Encoder, AtomIndexer, PositionsNode, CellNode)
    def expansion1(self, encoder, pidxer, positions, cell=None, *, dist_hard_max, periodic, **kwargs):
        """
        Setup pair finder if positions and cell are passed with encoding.

        :return: (padding_indexer, pairfinder)
        """
        if periodic:
            assert isinstance(cell, CellNode), "Periodic networks require a cell input"
            pair_parents = (positions, encoder, pidxer, cell)
            pair_cls = PeriodicPairIndexer
        else:
            assert cell is None, "When providing a cell node, periodic must be set to true"
            pair_parents = (positions, encoder, pidxer)
            pair_cls = OpenPairIndexer
        pairfinder = pair_cls("PairIndexer", pair_parents, dist_hard_max=dist_hard_max)
        return pidxer, pairfinder

    @_parent_expander.match(AtomIndexer, PairIndexer)
    def expansion1(self, pidxer, pairfinder, **kwargs):
        """
        Get indexed features from the atom indexer.

        :return: (indexed_features, pair_finder)
        """
        return pidxer.indexed_features, pairfinder


class _FeatureNodesMixin:
    @property
    def feature_nodes(self):
        if not hasattr(self, "_feature_nodes"):
            self._make_feature_nodes()
        return self._feature_nodes

    def _make_feature_nodes(self):
        """
        This function can be used on a network to make nodes that refer to the individual feature blocks.
        We use this function/class to provide backwards compatibility with models that did not have this
        attribute when created.
        :param self: the input network, which is modified in-place
        :return: None
        """

        net_module = self.torch_module
        n_interactions = net_module.ni

        feature_nodes = []

        index_state = IdxType.Atoms
        parents = (self,)
        for i in range(n_interactions + 1):
            name = f"{self.name}_features_{i}"
            fnode = IndexNode(name=name, parents=parents, index=i, index_state=index_state)
            feature_nodes.append(fnode)
        self._feature_nodes = feature_nodes


class Hipnn(DefaultNetworkExpansion, AutoKw, Network, SingleNode, _FeatureNodesMixin):
    """
    Node for HIP-NN neural networks
    """

    _input_names = "input_features", "pair_first", "pair_second", "pair_dist"
    _index_state = IdxType.Atoms
    _auto_module_class = network_modules.hipnn.Hipnn

    @_parent_expander.match(_BaseNode, PairIndexer)
    def expansion2(self, features, pairfinder, **kwargs):
        return features, pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist

    _parent_expander.assertlen(4)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, None, None, None)

    def __init__(self, name, parents, periodic=False, module="auto", module_kwargs=None):
        if module == "auto":
            self.module_kwargs = module_kwargs
            net_module = self.auto_module()
        else:
            net_module = module
        parents = self.expand_parents(
            parents, species_set=net_module.species_set, dist_hard_max=net_module.dist_hard_max, periodic=periodic
        )
        super().__init__(name, parents, module=net_module)


class HipnnVec(DefaultNetworkExpansion, AutoKw, Network, SingleNode, _FeatureNodesMixin):
    """
    Node for HIP-NN-TS neural network, l=2
    """

    _input_names = "input_features", "pair_first", "pair_second", "pair_dist", "pair_coord"
    _index_state = IdxType.Atoms
    _auto_module_class = network_modules.hipnn.HipnnVec

    @_parent_expander.match(_BaseNode, PairIndexer)
    def expansion2(self, features, pairfinder, **kwargs):
        return features, pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist, pairfinder.pair_coord

    _parent_expander.assertlen(5)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, None, None, None, None)

    def __init__(self, name, parents, periodic=False, module="auto", module_kwargs=None):
        if module == "auto":
            self.module_kwargs = module_kwargs
            net_module = self.auto_module()
        else:
            net_module = module
        parents = self.expand_parents(
            parents, species_set=net_module.species_set, dist_hard_max=net_module.dist_hard_max, periodic=periodic
        )

        super().__init__(name, parents, module=net_module)


class HipnnQuad(HipnnVec):
    """
    Node for HIP-NN-TS neural network, l=2
    """
    _auto_module_class = network_modules.hipnn.HipnnQuad

class HipHopnn(HipnnVec):
    _auto_module_class = network_modules.hiphop.HipHopNNModule

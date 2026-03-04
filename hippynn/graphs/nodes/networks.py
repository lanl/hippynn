"""
Nodes for networks.
"""
from .tags import Encoder, PairIndexer, Network, AtomIndexer
from .base import Node, AutoKw, ExpandParents, SingleNode
from .base.multi import IndexNode
from .indexers import OneHotEncoder, PaddingIndexer, acquire_encoding_padding
from .pairs import OpenPairIndexer, PeriodicPairIndexer, SparsePairIndexer
from .tags import PairIndexer
from .inputs import SpeciesNode, PositionsNode, CellNode
from ..indextypes import IdxType
from ... import networks as network_modules


class DefaultNetworkExpansion(ExpandParents):
    parent_expansion_kwargs = {
        "dist_hard_max": "dist_hard_max",
        "species_set": "possible_species",
        "periodic": "periodic",
        }

    @parent_expander.match(SpeciesNode, PositionsNode)
    @parent_expander.match(SpeciesNode, PositionsNode, CellNode)
    def expansion0(self, species, *other_parents, species_set, purpose, **kwargs):
        """
        Finds or sets up a default one-hot encoder if species are passed as first argument.

        :return: (encoder,padding_indexer,*other_parents)

        """
        encoder, pidxer = acquire_encoding_padding(species, species_set, purpose=purpose)
        return (encoder, pidxer, *other_parents)

    @parent_expander.match(Encoder, AtomIndexer, PositionsNode)
    @parent_expander.match(Encoder, AtomIndexer, PositionsNode, CellNode)
    def expansion1(self, encoder, pidxer, positions, cell=None, *, dist_hard_max, periodic, **kwargs):
        """
        Setup pair finder if positions and cell are passed with encoding.

        :param periodic: [bool | PairIndexer]: if True, use default pair indexer. if False, use open boundary conditions. If node, use that node for pair indexing.

        :return: (padding_indexer, pairfinder)
        """
        if periodic:
            assert isinstance(cell, CellNode), f"Processsing periodic data requires a cell input (got: {cell})"

            if isinstance(periodic, PairIndexer):
                pair_cls = periodic
            elif isinstance(periodic,bool):
                pair_cls = SparsePairIndexer                
            pair_parents = (positions, encoder, pidxer, cell)
            pair_cls = SparsePairIndexer
        else:
            assert cell is None, "When providing a cell node, periodic must be set to true"
            pair_parents = (positions, encoder, pidxer)
            pair_cls = OpenPairIndexer
        pairfinder = pair_cls("PairIndexer", pair_parents, dist_hard_max=dist_hard_max)
        return pidxer, pairfinder

    @parent_expander.match(AtomIndexer, PairIndexer)
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


class Hipnn(AutoKw, DefaultNetworkExpansion,  Network, SingleNode, _FeatureNodesMixin):
    """
    Node for HIP-NN neural networks
    """

    input_names = "input_features", "pair_first", "pair_second", "pair_dist"
    index_state = IdxType.Unlabeled
    auto_module_class = network_modules.hipnn.Hipnn

    @parent_expander.match(Node, PairIndexer)
    def expansion2(self, features, pairfinder, **kwargs):
        return features, pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist

    parent_expander.assertlen(4)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, None, None, None)

    def __init__(self, name, parents, periodic=False, **kwargs):
        super().__init__(name, parents, periodic=periodic, **kwargs)



class HipnnVec(AutoKw, DefaultNetworkExpansion, Network, SingleNode, _FeatureNodesMixin):
    """
    Node for HIP-NN-TS neural network, l=2
    """

    input_names = "input_features", "pair_first", "pair_second", "pair_dist", "pair_coord"
    index_state = IdxType.Unlabeled
    auto_module_class = network_modules.hipnn.HipnnVec

    @parent_expander.match(Node, PairIndexer)
    def expansion2(self, features, pairfinder, **kwargs):
        return features, pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist, pairfinder.pair_coord

    parent_expander.assertlen(5)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, None, None, None, None)

    def __init__(self, name, parents, periodic=False, **kwargs):
        super().__init__(name, parents, periodic=periodic, **kwargs)


class HipnnQuad(HipnnVec):
    """
    Node for HIP-NN-TS neural network, l=2
    """
    auto_module_class = network_modules.hipnn.HipnnQuad

class HipHopnn(HipnnVec):
    """
    Node for HIP-HOP_NN neural network.
    """
    auto_module_class = network_modules.hiphop.HipHopNNModule

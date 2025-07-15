"""
Nodes for pair index state conversion
"""
from .. import find_unique_relative
from ..indextypes import register_index_transformer, IdxType, _debprint as idx_debprint
from ..nodes.base.node_functions import NodeOperationError
from ..nodes.indexers import PaddingIndexer, OneHotEncoder
from ..nodes.inputs import SpeciesNode, PositionsNode
from ..nodes.pairs import PairReIndexer, PairDeIndexer
from ..nodes.tags import PairIndexer

from .atoms import make_search_nodes

# TODO: Rewrite so it can use non-one-hot-encodings?

@register_index_transformer(IdxType.SysAtomAtom, IdxType.Pairs)
def idx_sysatomatom_pairs(node, hints=None):
    purpose = "auto-generating indexing for {}".format(node)

    search_nodes = make_search_nodes(node, hints)

    funrel = find_unique_relative  # Abbreviation because we so many calls in this function.
    if node.origin_node is None:
        # If we are auto-indexing in the model graph, it should be easy.
        pair_idx = funrel(search_nodes, PairIndexer, why_desc=purpose)
        pidxer = funrel(search_nodes, PaddingIndexer, why_desc=purpose)
        idx_debprint("Using reindexer for ", node)

    else:
        # If we are auto-indexing in the loss graph, it gets a bit complicated.

        search_nodes = set(n.origin_node for n in search_nodes)
        # The species we link to will be the true version of species, that is, the node where
        species = funrel(search_nodes, SpeciesNode, why_desc=purpose).true
        coordinates = funrel(search_nodes, PositionsNode, why_desc=purpose).true
        try:
            encoder = funrel(species, OneHotEncoder, why_desc=purpose)
        except NodeOperationError as ne:
            idx_debprint("Creating new Padding indexer for {}".format(node))
            # If this fails, something bad has happened -- the loss graph is trying to do
            # something not defined by the model graph
            origin_encoder = funrel(species.origin_node, OneHotEncoder, why_desc=purpose)
            encoder = OneHotEncoder("one-hot", (species,), species_set=origin_encoder.species_set)
            # If we can't find an encoder, let's assume we won't find an indexer
            pidxer = PaddingIndexer("Auto(PaddingIndexer)", encoder)
        else:
            # If we did find an encoder, look for an indexer and assume it exists
            pidxer = funrel(encoder, PaddingIndexer, why_desc=purpose)

        try:
            pair_idx = funrel(species, PairIndexer, why_desc=purpose)
        except NodeOperationError as ne:
            idx_debprint("Creating new Pair indexer for {}".format(node))
            # If this fails, something bad has happened -- the loss graph is trying to do
            # something not defined by the model graph
            origin_pairs = funrel(species.origin_node, PairIndexer, why_desc=purpose)

            new_PairType = type(origin_pairs)

            pair_parents = (
                coordinates,
                pidxer.nonblank,
                pidxer.real_atoms,
                pidxer.inv_real_atoms,
            )
            try:
                cell = origin_pairs.cells
            except AttributeError:
                pass  # The pair indexer does not use cells
            else:
                pair_parents = pair_parents, *cell

            pair_idx = new_PairType("Pairs", pair_parents, dist_hard_max=origin_pairs.dist_hard_max)

    cls = PairReIndexer
    parents = node, pidxer.system_index, pidxer.atom_index, pair_idx.pair_first, pair_idx.pair_second
    return parents, cls


@register_index_transformer(IdxType.Pairs, IdxType.SysAtomAtom)
def idx_pairs_sysatomatom(node, hints=None):
    purpose = "auto-generating indexing for {}".format(node)

    search_nodes = make_search_nodes(node,hints)

    if node.origin_node is None:
        pidx = find_unique_relative(search_nodes, PairIndexer, why_desc=purpose)
        padidx = find_unique_relative(search_nodes, PaddingIndexer, why_desc=purpose)

        parents = (
            node,
            padidx.system_index,
            padidx.atom_index,
            padidx.n_systems,
            padidx.n_atoms_max,
            pidx.pair_first,
            pidx.pair_second,
        )

        return parents, PairDeIndexer
    else:
        raise NotImplementedError("De-indexing not yet implemented in loss graph")
        # TODO: refactor out padding indexer creation for loss from the sysatom-atom indexer, then re-use it here.

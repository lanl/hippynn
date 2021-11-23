"""
Nodes for atom index state conversion
"""
from .. import find_unique_relative
from ..indextypes import register_index_transformer, IdxType, _debprint as idx_debprint
from ..nodes.base.node_functions import NodeOperationError
from ..nodes.indexers import PaddingIndexer, OneHotEncoder, AtomReIndexer, AtomDeIndexer
from ..nodes.inputs import SpeciesNode


# TODO: Rewrite so it can use non-one-hot-encodings?


@register_index_transformer(IdxType.MolAtom, IdxType.Atoms)
def idx_molatom_atom(node):
    purpose = "auto-generating indexing for {}".format(node)

    funrel = find_unique_relative  # Abbreviation because we so many calls in this function.
    if node.origin_node is None:
        # If we are auto-indexing in the model graph, it should be easy.
        # species = funrel(node, SpeciesNode,
        #                 why_desc=purpose)
        pidxer = funrel(node, PaddingIndexer, why_desc=purpose)
        idx_debprint("Using reindexer for ", node)

    else:
        # If we are auto-indexing in the loss graph, it gets a bit complicated.

        # The species we link to will be the true version of species, that is, the node where
        species = funrel(node.origin_node, SpeciesNode, why_desc=purpose).true
        try:
            encoder = funrel(species, OneHotEncoder, why_desc=purpose)
        except NodeOperationError as ne:
            idx_debprint("Creating new ENCODER")
            # If this fails, something bad has happened -- the loss graph is trying to do something not defined by the
            # model graph
            origin_encoder = funrel(species.origin_node, OneHotEncoder, why_desc=purpose)
            encoder = OneHotEncoder("Auto(One-hot)", (species,), species_set=origin_encoder.species_set)
            # If we can't find an encoder, let's assume we won't find an indexer
            pidxer = PaddingIndexer("Auto(PaddingIndexer)", encoder)
        else:
            # If we did find an encoder, look for an indexer and assume it exists
            pidxer = funrel(encoder, PaddingIndexer, why_desc=purpose)

    cls = AtomReIndexer
    parents = node, pidxer.real_atoms
    return parents, cls


@register_index_transformer(IdxType.Atoms, IdxType.MolAtom)
def idx_atom_molatom(node):

    if node.origin_node is None:
        parents = (node,)
        return parents, AtomDeIndexer
    else:
        raise NotImplementedError("De-indexing not yet implemented in loss graph")
        # TODO: refactor out padding indexer creation for loss from the molatom-atom indexer, then re-use it here.

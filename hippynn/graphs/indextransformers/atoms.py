"""
Nodes for atom index state conversion
"""
from .. import find_unique_relative, find_relatives
from ..indextypes import register_index_transformer, IdxType, _debprint as idx_debprint
from ..nodes.base import InputNode
from ..nodes.base.node_functions import NodeOperationError, is_in_loss_graph
from ..nodes.indexers import PaddingIndexer, OneHotEncoder, AtomReIndexer, AtomDeIndexer
from ..nodes.inputs import SpeciesNode


# TODO: Rewrite so it can use non-one-hot-encodings?
def make_search_nodes(node, hints):
    if hints is None:
        return set([node])
    else:
        if None in hints:
            raise ValueError("None cannot be a hint!")
        return set([node, *hints])
    

@register_index_transformer(IdxType.SysAtom, IdxType.Atoms)
def idx_sysatom_atom(node, hints=None):
    purpose = "auto-generating indexing for {}".format(node)

    search_nodes = make_search_nodes(node, hints)
    
    # This 'if' call can raise NodeAmbiguity Error, but we want it to error in that case.
    if not is_in_loss_graph(search_nodes, why_desc=purpose):
        # If we are auto-indexing in the model graph, it should be easy.
        pidxer = find_unique_relative(search_nodes, PaddingIndexer, why_desc=purpose)
        idx_debprint("Using reindexer for ", node)
    else:
        # If we are auto-indexing in the loss graph, it gets a bit complicated.
        # We want to find the species by examining the corresponding model graph.

        # Make new hints which are the origin nodes for the input nodes of this set.
        search_nodes = find_relatives(search_nodes, InputNode)  # get all inputs
        search_nodes = set(n.origin_node for n in search_nodes) # find model version of these inputs
        
        # The species we link to will be the true version of species (the loss-graph version of the tensor)
        species = find_unique_relative(search_nodes, SpeciesNode, why_desc=purpose).true
        try:
            encoder = find_unique_relative(species, OneHotEncoder, why_desc=purpose)
        except NodeOperationError as ne:
            idx_debprint("Creating new encoder in loss graph.")
            # If this fails, something bad has happened -- the loss graph is trying to do something not defined by the
            # model graph
            origin_encoder = find_unique_relative(species.origin_node, OneHotEncoder, why_desc=purpose)
            encoder = OneHotEncoder("Auto(One-hot)", (species,), species_set=origin_encoder.species_set)
            # If we can't find an encoder, let's assume we won't find an indexer
            pidxer = PaddingIndexer("Auto(PaddingIndexer)", encoder)
        else:
            # If we did find an encoder, look for an indexer and assume it exists
            pidxer = find_unique_relative(encoder, PaddingIndexer, why_desc=purpose)

    cls = AtomReIndexer
    parents = node, pidxer.real_atoms
    return parents, cls


@register_index_transformer(IdxType.Atoms, IdxType.SysAtom)
def idx_atom_sysatom(node, hints=None):

    if node.origin_node is None:
        parents = (node,)
        return parents, AtomDeIndexer
    else:
        raise NotImplementedError("De-indexing not yet implemented in loss graph")
        # TODO: refactor out padding indexer creation for loss from the sysatom-atom indexer, then re-use it here.

"""
Nodes for indexing information.
"""
from .tags import Encoder, AtomIndexer
from .base import SingleNode, AutoNoKw, AutoKw, find_unique_relative, MultiNode, ExpandParents, _BaseNode
from .base.node_functions import NodeNotFound
from .inputs import SpeciesNode

# Index generating functions need access to appropriately raise this
from ..indextypes import IdxType
from ...layers import indexers as index_modules


class OneHotEncoder(AutoKw, Encoder, MultiNode):
    """
    Node for encoding species as one-hot vectors
    """

    _output_names = "encoding", "nonblank"
    _output_index_states = IdxType.MolAtom, IdxType.MolAtom
    _auto_module_class = index_modules.OneHotSpecies

    def __init__(self, name, parents, species_set, module="auto", **kwargs):
        try:
            species_set = species_set.clone()
        except AttributeError:
            pass  # If was not passed a tensor.
        self.species_set = species_set

        # Can be passed just a single species node, not a tuple
        if isinstance(parents, _BaseNode):
            parents = (parents,)
        super().__init__(name, parents, module=module, **kwargs)

    def auto_module(self):
        return self._auto_module_class(species_set=self.species_set)


class PaddingIndexer(AtomIndexer, AutoNoKw, ExpandParents, MultiNode):
    """
    Node for building information to convert from
    MolAtom to Atom index state.
    """

    _output_names = (
        "indexed_features",
        "real_atoms",
        "inv_real_atoms",
        "mol_index",
        "atom_index",
        "n_molecules",
        "n_atoms_max",
    )
    _output_index_states = IdxType.Atoms, None, None, None, None, None, None  # optional?
    _input_names = "encoding", "nonblank"
    _auto_module_class = index_modules.PaddingIndexer

    @_parent_expander.match(Encoder)
    def expand0(self, encoder, **kwargs):
        return encoder.encoding, encoder.nonblank

    _parent_expander.assertlen(2)

    def __init__(self, name, parents, *args, **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, *args, **kwargs)


class AtomReIndexer(ExpandParents, AutoNoKw, SingleNode):
    """
    Node for re-using index information to convert MolAtom->Atom.
    """

    _auto_module_class = index_modules.AtomReIndexer
    _index_state = IdxType.Atoms

    @_parent_expander.match(SingleNode)
    def expand0(self, features, *, purpose, **kwargs):
        pad_idx = find_unique_relative(features, PaddingIndexer, why_desc=purpose)
        return features, pad_idx

    @_parent_expander.match(SingleNode, PaddingIndexer)
    def expand1(self, features, pad_idx, **kwargs):
        return features, pad_idx.real_atoms

    _parent_expander.assertlen(2)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class AtomDeIndexer(ExpandParents, AutoNoKw, SingleNode):
    """
    Node for converting Atom->MolAtom
    """

    _auto_module_class = index_modules.AtomDeIndexer
    _index_state = IdxType.MolAtom

    @_parent_expander.matchlen(1)
    def expand0(self, features, *, purpose, **kwargs):
        pad_idx = find_unique_relative(features, PaddingIndexer, why_desc=purpose)
        return features, pad_idx.mol_index, pad_idx.atom_index, pad_idx.n_molecules, pad_idx.n_atoms_max

    @_parent_expander.matchlen(2)
    def expand0(self, features, mol_index, atom_index, n_mol, n_atom, **kwargs):
        return features.main_output, mol_index, atom_index, n_mol, n_atom

    _parent_expander.assertlen(5)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class QuadUnpackNode(AutoNoKw, SingleNode):
    _auto_module_class = index_modules.QuadUnpack
    _index_state = IdxType.Molecules

    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)


class FilterBondsOneway(AutoNoKw, SingleNode):
    """
    Node which filters the set of pairs to a one-way list.
    """

    _input_names = "input_bonds", "pair_first", "pair_second"
    _index_state = IdxType.NotFound
    _auto_module_class = index_modules.FilterBondsOneway

    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)


def acquire_encoding_padding(search_nodes, species_set, purpose=None):
    """
    Automatically finds/builds a one-hot encoder and padding indexer starting from ``search_nodes``.

    If the encoder and padder exist as relatives of the search nodes.

    :param search_nodes: Node or nodes to start from.
    :param species_set: Species set to use if an encoder needs to be created.
    :param purpose: String for error information if the process fails. (optional)
    :return: encoder, padding indexer

    """
    try:
        encoder = find_unique_relative(search_nodes, Encoder, why_desc=purpose)
    except NodeNotFound:
        if species_set is None:
            raise ValueError(
                "Building encode and padder requires a species_set, but a species set is not specified,"
                " Make an encoder for the needed species node e.g. using the command: \n"
                "`encoder = OneHotEncoder('OneHot', species_node, species_set=species_set)`.\n"
            )
        species_node = find_unique_relative(search_nodes, SpeciesNode, why_desc=purpose)
        encoder = OneHotEncoder("OneHot", (species_node,), species_set=species_set)

    try:
        pidxer = find_unique_relative(search_nodes, PaddingIndexer, why_desc=purpose)
    except NodeNotFound:
        pidxer = PaddingIndexer("PaddingIndexer", (encoder.encoding, encoder.nonblank))

    return encoder, pidxer

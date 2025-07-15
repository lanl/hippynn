"""
Nodes for indexing information.
"""
from .tags import Encoder, AtomIndexer
from .base import SingleNode, AutoNoKw, AutoKw, find_unique_relative, MultiNode, ExpandParents, Node, IndexNode
from .base.node_functions import NodeNotFound
from .inputs import SpeciesNode

# Index generating functions need access to appropriately raise this
from ..indextypes import IdxType
from ..indextypes.reduce_funcs import index_type_coercion
from ...layers import indexers as index_modules

from ..._deprecations import _DeprecatedNamesMixin


class OneHotEncoder(AutoKw, Encoder, MultiNode):
    """
    Node for encoding species as one-hot vectors
    """

    output_names = "encoding", "nonblank"
    output_index_states = IdxType.SysAtom, IdxType.SysAtom
    auto_module_class = index_modules.OneHotSpecies
    auto_module_kwargs = "species_set",

    def __init__(self, name, parents, species_set, module="auto", **kwargs):
        try:
            species_set = species_set.clone()
        except AttributeError:
            pass  # If was not passed a tensor.
        self.species_set = species_set

        super().__init__(name, parents, species_set=species_set, module=module, **kwargs)


class PaddingIndexer(AtomIndexer, AutoNoKw, ExpandParents, MultiNode, _DeprecatedNamesMixin):
    """
    Node for building information to convert from
    SysAtom to Atom index state.
    """
    _DEPRECATED_NAMES = {
        "mol_index": "system_index",
        "n_molecules": "n_systems",
        }
    output_names = (
        "indexed_features",
        "real_atoms",
        "inv_real_atoms",
        "system_index",
        "atom_index",
        "n_systems",
        "n_atoms_max",
    )
    output_index_states = IdxType.Atoms, None, None, None, None, None, None  # optional?
    input_names = "encoding", "nonblank"
    auto_module_class = index_modules.PaddingIndexer

    @parent_expander.match(Encoder)
    def expand0(self, encoder, **kwargs):
        return encoder.encoding, encoder.nonblank

    parent_expander.assertlen(2)

    def __init__(self, name, parents, *args, **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, *args, **kwargs)


class AtomReIndexer(ExpandParents, AutoNoKw, SingleNode):
    """
    Node for re-using index information to convert SysAtom->Atom.
    """

    auto_module_class = index_modules.AtomReIndexer
    index_state = IdxType.Atoms

    @parent_expander.match(SingleNode)
    def expand0(self, features, *, purpose, **kwargs):
        pad_idx = find_unique_relative(features, PaddingIndexer, why_desc=purpose)
        return features, pad_idx

    @parent_expander.match(SingleNode, PaddingIndexer)
    def expand1(self, features, pad_idx, **kwargs):
        return features, pad_idx.real_atoms

    parent_expander.assertlen(2)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class AtomDeIndexer(ExpandParents, AutoNoKw, SingleNode):
    """
    Node for converting Atom->SysAtom
    """

    auto_module_class = index_modules.AtomDeIndexer
    index_state = IdxType.SysAtom

    @parent_expander.matchlen(1)
    def expand0(self, features, *, purpose, **kwargs):
        pad_idx = find_unique_relative(features, PaddingIndexer, why_desc=purpose)
        return features, pad_idx.system_index, pad_idx.atom_index, pad_idx.n_systems, pad_idx.n_atoms_max

    @parent_expander.matchlen(2)
    def expand0(self, features, system_index, atom_index, n_mol, n_atom, **kwargs):
        return features.main_output, system_index, atom_index, n_mol, n_atom

    parent_expander.assertlen(5)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class QuadUnpackNode(AutoNoKw, SingleNode):
    auto_module_class = index_modules.QuadUnpack
    index_state = IdxType.Systems

    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)


class FilterBondsOneway(AutoNoKw, SingleNode):
    """
    Node which filters the set of pairs to a one-way list.
    """
    input_names = "input_bonds", "pair_first", "pair_second"
    index_state = IdxType.Unlabeled
    auto_module_class = index_modules.FilterBondsOneway

    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)


class SysMaxOfAtomsNode(ExpandParents, AutoNoKw, SingleNode):
    input_names = "var", "system_index", "n_systems"
    index_state = IdxType.Systems
    auto_module_class = index_modules.SysMaxOfAtoms

    @parent_expander.match(Node)
    def expansion0(self, node, *, purpose, **kwargs):
        pidxer = find_unique_relative(node, AtomIndexer, why_desc=purpose)
        return node, pidxer

    @parent_expander.match(Node, AtomIndexer)
    def expansion1(self, node, pidxer, *, purpose, **kwargs):
        return node, pidxer.system_index, pidxer.n_systems

    parent_expander.assertlen(3)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
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

class FuzzyHistogrammer(AutoKw, SingleNode):
    """ 
    Node for transforming a scalar feature into a vectorized feature via 
    the fuzzy/soft histogram method.

    :param length: length of vectorized feature
    """

    input_names = "values"
    auto_module_class = index_modules.FuzzyHistogram
    auto_module_kwargs = "length", "vmin", "vmax"

    def __init__(self, name, parents, length, vmin, vmax, module="auto", **kwargs):
        self.output_index_state = parents[0].index_state
        super().__init__(name, parents, length=length, vmin=vmin, vmax=vmax, module=module, **kwargs)

class SpeciesIndexer(AutoNoKw, SingleNode, ExpandParents):
    """
    Separate an atom-wise tensor into sub-tensors for each species.
    """
    input_names = "input_values", "onehot_encoding"
    auto_module_class = index_modules.SpeciesIndexer
    index_state = IdxType.Atoms


    @parent_expander.match(Node, Node)
    def expansion0(self, node_to_index, hint_node, **kwargs):
        """
        For loss-graph quantities, we may need the information about /which/ species to use.
        """
        atom_node_to_index = index_type_coercion(node_to_index, IdxType.Atoms, hints=[hint_node])

        return atom_node_to_index,

    @parent_expander.match(Node)
    def expansion1(self, node_to_index, species_set, **kwargs):
        """
        find onehot encoding.
        """
        atom_node_to_index = index_type_coercion(node_to_index, IdxType.Atoms)
        onehot = find_unique_relative(atom_node_to_index, OneHotEncoder)
        self.species_set = species_set or onehot.species_set
        return atom_node_to_index, onehot.encoding

    # add asserts for parent expansion
    parent_expander.assertlen(2)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, IdxType.Atoms)

    def __init__(self, name, parents, *args, module="auto", species_set=None, **kwargs):
        parents = self.expand_parents(parents, species_set=species_set)
        super().__init__(name, parents, *args, module=module, **kwargs)

        nonzero_species = [species for species in self.species_set if species != 0]
        self.species_to_idx = {species: idx for idx, species in enumerate(nonzero_species)}

        self.children = tuple(
            IndexNode(name=f"{name}_{species}", parents=(self,), index=idx, index_state=IdxType.Atoms)
            for species, idx in self.species_to_idx.items()
        )

    def with_species_equal(self, z_value):
        return self.children[self.species_to_idx(z_value)]
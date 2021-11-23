"""
Storage for the logical relationships between index states,
and a cache reduce redundant computations. Also includes
the indextransformer decorator which registers computations
that change index states.
The index transformers do not live here because they are themselves
nodes, and this package should be minimally dependent on the nodes themselves.
"""
import functools

from . import _debprint
from .type_def import IdxType

# fmt: off
_db_index_states = {
### TYPE PREDICTED BY MODEL | DEFAULT TYPE OF STORAGE IN DATABASE
    IdxType.Molecules       : IdxType.Molecules,
    IdxType.Atoms           : IdxType.MolAtom,
    IdxType.MolAtom         : IdxType.MolAtom,
    IdxType.QuadMol         : IdxType.QuadPack,
    IdxType.QuadPack        : IdxType.QuadPack,
    IdxType.Pair            : IdxType.MolAtomAtom,
    IdxType.Scalar          : IdxType.Scalar,
    IdxType.NotFound        : IdxType.NotFound,
}


elementwise_compare_rules = {
#### TYPES OF ARGUMENTS TO REDUCE             |   TYPE OF OUTPUTS
    ((IdxType.Scalar,)                        ,   IdxType.Scalar),
    ((IdxType.Molecules,)                     ,   IdxType.Molecules),
    ((IdxType.Atoms,)                         ,   IdxType.Atoms),
    ((IdxType.MolAtom,)                       ,   IdxType.Atoms),
    ((IdxType.Atoms, IdxType.MolAtom,)        ,   IdxType.Atoms),
    ((IdxType.QuadMol,)                       ,   IdxType.QuadMol),
    ((IdxType.QuadPack,)                      ,   IdxType.QuadMol),
    ((IdxType.QuadMol, IdxType.QuadPack)      ,   IdxType.QuadMol),
    ((IdxType.Pair,)                          ,   IdxType.Pair),
    ((IdxType.Pair, IdxType.MolAtomAtom)      ,   IdxType.Pair),
    ((IdxType.MolAtomAtom,)                   ,   IdxType.Pair),
    ((IdxType.NotFound,)                      ,   IdxType.NotFound),
}
# fmt: on

# Add default rule: (_some_type,scalar) -> _some_type if scalar is not in rule already
for idxset, idxtarget in elementwise_compare_rules.copy():
    if IdxType.Scalar not in idxset:
        idxset = *idxset, IdxType.Scalar
        elementwise_compare_rules.add((idxset, idxtarget))
del idxset, idxtarget

# Assemble rules
elementwise_compare_rules = {frozenset(in_types): out_type for in_types, out_type in elementwise_compare_rules}

# The index cache holds information about previously indexed nodes.
# This information comes in two types
# 1) (parent, node_type, name) -> An generated index from an index transformer
# 2) (index_type, node) -> new_node a cache of `node` in index state `index_state`
_index_cache = {}

# The index dispatch table holds the transformer functions which are capable
# of finding the transformation between index states
_index_dispatch_table = {}


def clear_index_cache():
    global _index_cache
    _index_cache = {}


def register_index_transformer(input_idxstate, output_idxstate):
    """
    Decorator for registering a transformer from one IdxType to another.
    """

    def decorator(f):
        if (input_idxstate, output_idxstate) in _index_dispatch_table:
            raise ValueError("Index dispatch for ({})->({}) already defined!".format(input_idxstate, output_idxstate))

        @functools.wraps(f)
        def wrapped(node, *args, **kwargs):
            idxcache_info = (output_idxstate, node)
            if idxcache_info in _index_cache:
                output_node = _index_cache[idxcache_info]
                _debprint("\tFound in node cache: {} : {}".format(output_node.name, output_node))
            else:
                parents, indexer_nodetype = f(node, *args, **kwargs)
                name = node.name + f"[{output_idxstate}]"
                output_node = indexer_nodetype(name, parents)

                # Forward-link this new node so we can avoid creating
                # redundant nodes.
                _index_cache[idxcache_info] = output_node
                # Back-link this new node to the old node's index state,
                # so if we end up with the reverse index transformation we
                # can just substitute the original node.
                _index_cache[(input_idxstate, output_node)] = node

                _debprint("\tGenerated new indexer: {}".format(output_node.name))
            return output_node

        _index_dispatch_table[input_idxstate, output_idxstate] = wrapped
        _debprint("Index dispatch for ({})->({}) defined!".format(input_idxstate, output_idxstate))
        return wrapped

    return decorator


#### Scalars can be viewed as any index type: Allow this ####
for idxt in IdxType:
    if idxt is IdxType.Scalar:
        continue  # No pass-through necessary for scalar-scalar

    @register_index_transformer(IdxType.Scalar, idxt)  # register no-op transformer with debug print
    def scalar_promote(node):
        _debprint("\t Allowing view of Scalar {} as type {}".format(node, idxt))
        return node


del scalar_promote

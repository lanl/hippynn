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
    IdxType.MolAtomAtom     : IdxType.MolAtomAtom,
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
    """
    Remove any cached index types for nodes.

    The index cache holds global references to nodes that have been
    transformed automatically, so these nodes won't be garbage collected
    automatically. Clearing the cache will free any reference to them.

    If you modify the index dispatch table or type comparison table on the fly,
    you may want to clear the cache to remove references to nodes created
    using the old rules.
    """
    global _index_cache
    _index_cache = {}


def assign_index_aliases(*nodes):
    """
    Store the input set of nodes in the index cache as index aliases of each other.

    Errors if the nodes contain two different nodes with the same index state.

    :param nodes:
    :return: None
    """
    # Developer node:
    # In the interest of safety this function currently errors rather than over-write information.
    # Operationally, it is not clear if it really needs to be safe or not.
    # If there becomes a convenient reason to make this function overwrite current info,
    # it could be changed.
    
    nodes = set(nodes)
    state_map = {n._index_state: n for n in nodes}
    if len(state_map) != len(nodes):
        raise ValueError(f"Input nodes did not each have a unique index state!\n"
                         f"Nodes and corresponding states: \n"
                         f"\t{[(n, n._index_state) for n in nodes]}")

    for target_state, target_node in state_map.items():
        for n in nodes:
            if n is target_node:
                continue
            idxcache_info = (target_state, n)
            if idxcache_info in _index_cache:
                raise ValueError(f"Index state for node is already cached: {idxcache_info,_index_cache[idxcache_info]}")
            else:
                _index_cache[idxcache_info] = target_node



def register_index_transformer(input_idxstate, output_idxstate):
    """
    Decorator for registering a transformer from one IdxType to another.

    An index transformer should have a signature::

         f(node) -> (parents, child_node_type)

    with types::

        node: Node
        parents : Tuple[Node]
        child_node_type : Union[Type[Node],Callable[[str,Tuple[node]],[node]]]

    That is, the child_node_type can be a class of node, or a factory function
    which acts with the same signature as a node constructor.
    If f supports additional arguments, they must have default values,
    as it will be invoked by the automatic index transformation system
    as f(node).

    The decorator results in a new function of type::

        f(node: Node) -> Node

    which is cached so that the same index transformation may be repeatedly
    applied to yield the same output node, and registered with the dispatch
    table for index transformations. No two functions may be registered for the
    same index type conversion simultaneously.

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

    # register no-op transformer
    @register_index_transformer(IdxType.Scalar, idxt)
    def scalar_promote(node, *, __idxt=idxt):
        _debprint("\t Allowing view of Scalar {} as type {}".format(node, __idxt))

        # Node index transformers (see #register_index_transformer) return
        # a set of parents and the node type to create.
        # So here we'll define a trivial factory function which mimics this
        # behavior and passes the node through unchanged.
        parents = node,  # Parents are defined as a tuple

        def passthrough_node_factory(name, parents):
            (node,) = parents  # Unwrap the tuple
            return node

        return parents, passthrough_node_factory

del scalar_promote, idxt

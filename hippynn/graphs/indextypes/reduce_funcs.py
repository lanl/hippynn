"""
Functions for changing index states.
"""
import warnings

from . import _debprint
from .registry import _index_dispatch_table, elementwise_compare_rules, _db_index_states, _index_cache
from .type_def import IdxType


def dispatch_indexing(input_is, output_is):
    """
    Acquire the function for converting between two index states.

    :param input_is: input index state
    :param output_is: output index state
    :return: the function

    """
    _debprint("dispatching:", input_is, output_is)
    try:
        func = _index_dispatch_table[input_is, output_is]
    except KeyError as ke:
        raise ValueError("No index conversion ({})->({}) found.".format(input_is, output_is))
    return func


def index_type_coercion(node, output_index_state):
    """
    Attempt to convert a node to a given index state.

    .. Note::
       1. If given a MultiNode, this function operates on the node's main_output.
       2. If there is no IdxType on the node, this function is a no-op, so that if one has not been implemented,
          no error will be raised.
       3. If the index type already matches, then this function just returns the input node.
       4. If the index type doesn't match, an index transformer of the appropriate type will be looked for.
          If no conversion is found, a ValueError is raised.

    :param node:  the node to convert
    :param output_index_state: the index state to convert to.
    :return: a node in the requested index state
    """
    _debprint("Requesting transformation : {} -> {}".format(node.name, output_index_state))
    node = node.main_output
    try:
        needs_index = node._index_state != output_index_state
    except AttributeError as e:
        _debprint("\t Indexing information not found! Something is likely to fail.")
        needs_index = False

    if needs_index:
        prior_index_ref = (output_index_state, node)
        if (output_index_state, node) in _index_cache:
            # If this node is linked to a previous index state
            _debprint(f"\t Previous indexed version found for node {node} type {output_index_state}")
            outnode = _index_cache[prior_index_ref]
        else:
            _debprint(f"\t No index version found for node {node} type {output_index_state}")
            outnode = dispatch_indexing(node._index_state, output_index_state)(node)

    else:
        outnode = node
        _debprint("\t -> Autoindex unnecessary; index form correct.")

    return outnode


def soft_index_type_coercion(node, output_index_state):
    """
    Coerce if information is available.

    :param node:
    :param output_index_state:
    :return:
    """
    if output_index_state == IdxType.NotFound or node._index_state == IdxType.NotFound:
        _debprint(
            f"Soft request: Skipping requested conversion of {node} ({node._index_state})to {output_index_state}"
            " as information was not available."
        )
        return node
    else:
        return index_type_coercion(node, output_index_state)


def get_reduced_index_state(*nodes_to_reduce):
    """
    Find the index state for comparison between values in a loss function or plot.

    .. Note::
        This function is unlikely to be directly needed as a user.
        it's more likely you want to use :func:`elementwise_compare_reduce`.

    :param nodes_to_reduce:
    :return:
    """
    typeset = frozenset(getattr(n,"_index_state",IdxType.NotFound) for n in nodes_to_reduce)
    _debprint("Finding index comparison for for :", [n.name for n in nodes_to_reduce])
    try:
        coerced_type = elementwise_compare_rules[typeset]
        _debprint("Coerced type found: {}->{}".format(set(typeset), coerced_type))
    except KeyError as oke:
        _debprint("index rules:", elementwise_compare_rules)
        raise KeyError("index rule not found for {}".format(typeset)) from oke
    return coerced_type


def elementwise_compare_reduce(*nodes_to_reduce):
    """
    Return nodes converted to a mutually compatible index state with no padding.

    :param nodes_to_reduce: nodes to put in a comparable, reduced index state
    :return: node (if single argument) or tuple(nodes) (if multiple arguments)
    """
    if len(nodes_to_reduce) == 0:
        _debprint("Compare reduce on nothing, returning nothing")
        return tuple()
    coerced_type = get_reduced_index_state(*nodes_to_reduce)
    coerced_nodes = [index_type_coercion(node, coerced_type) for node in nodes_to_reduce]
    _debprint("Coerced:", [x.name for x in coerced_nodes])
    if len(coerced_nodes) == 1:
        return coerced_nodes[0]
    return coerced_nodes


def db_state_of(idxt):
    """
    Return the IdxType expected in the database for a given index type.

    .. Note::
        This function is unlikely to be directed needed as a user.
        it's more likely you want to use :func:`db_form`.
    """
    try:
        return _db_index_states[idxt]
    except KeyError:
        raise KeyError(f"No index state found for index type '{idxt}'")


def db_form(node):
    """
    Return a node converted to the index state of the database.
    """
    return index_type_coercion(node, db_state_of(node._index_state))

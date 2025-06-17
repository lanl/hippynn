"""
Graph Operations ("gops") that process or transform a set of nodes.
"""
import collections
import copy

from .nodes.base import InputNode, MultiNode
from .nodes.base.algebra import ValueNode
from .nodes.base.node_functions import NodeNotFound, NodeOperationError
from .indextypes import soft_index_type_coercion

from . import get_connected_nodes, find_unique_relative
from ..tools import is_equal_state_dict


def get_subgraph(required_nodes):
    """
    Get the subgraph associated with some target (required) nodes.

    :param required_nodes: the nodes to compute

    :return: a list of nodes involved in the computation of the required nodes.
    """

    required_nodes = list(set(required_nodes))
    subgraph_nodes = required_nodes + [p for node in required_nodes for p in node.get_all_parents()]

    subgraph_nodes = subgraph_nodes + [c for mn in subgraph_nodes if isinstance(mn, MultiNode) for c in mn.children]
    # ^-- a note on this:
    # Children are included because bad things happen if a multinode is left without any of its children;
    # The API using dot syntax will break.
    # Adding these to a GraphModule is not expensive - it adds a bit of unwrapping time.

    return list(set(subgraph_nodes))


def compute_evaluation_order(all_nodes):
    """
    Computes the evaluation order for forward computation.

    :param all_nodes: The nodes to compute

    :return:  outputs_list, inputs_list
        Outputs list is the node to evaluate next, which is fed the inputs in the
        corresponding entry of inputs_list

    """

    # The strategy here is to emulate trying everything, and record an order that works.
    # It's computationally inefficient, but it's very easy to implement.
    # The strategy is greedy and brute-force --  it won't scale to large numbers of nodes."""

    evaluation_inputs_list = []
    evaluation_outputs_list = []
    # need to sort to get stable results between runs/processes.
    unsatisfied_nodes = list(sorted(all_nodes, key=lambda node: node.name))
    satisfied_nodes = set()
    n = -1
    while len(unsatisfied_nodes) > 0:
        n += 1

        # Consider each node we haven't found yet computable
        for considered_node in unsatisfied_nodes.copy():
            parents = considered_node.parents
            uncomputed_parents = [x for x in parents if x not in satisfied_nodes]

            # If that node has zero uncomputed parents, we can add it to the evaluation order.
            if len(uncomputed_parents) == 0:
                unsatisfied_nodes.remove(considered_node)
                satisfied_nodes.add(considered_node)
                # Input nodes need not be considered; their values are given to `forward()`
                if not isinstance(considered_node, InputNode):
                    evaluation_inputs_list.append(parents)
                    evaluation_outputs_list.append(considered_node)
                break  # This break is necessary to make the `else` clause work;
                # if we find a node, we break and the `else` statement won't be hit.
        else:
            # For-else structure will land here if no satisfiable nodes are found during the iteration.
            # If we made it through all the unsatisfied nodes, something must be wrong.
            raise ValueError("Graph inputs do not seem to specify outputs! Iteration: {}".format(n))

    check_evaluation_order(evaluation_outputs_list, evaluation_inputs_list)
    return evaluation_outputs_list, evaluation_inputs_list


def check_evaluation_order(evaluation_outputs_list, evaluation_inputs_list):
    """
    Validate an evaluation order.

    :param evaluation_outputs_list:
    :param evaluation_inputs_list:
    :return:
    """
    pseudo_evaluated = list()
    for out_node, inputs_for_node in zip(evaluation_outputs_list, evaluation_inputs_list):
        for in_node in inputs_for_node:
            if in_node not in pseudo_evaluated:
                if not isinstance(in_node, InputNode):
                    raise AssertionError("Nodes are not ordered correctly.")
                else:
                    pseudo_evaluated.append(in_node)
        pseudo_evaluated.append(out_node)


def copy_subgraph(required_nodes, assume_inputed, tag=None):
    """
    Copy a subgraph for the required nodes.
    Doesn't copy the modules.
    Returns copies of the specified nodes, linked implicitly to copies of their parents.

    If assume_inputted is not empty, this will result in partially disconnected nodes.
    As such, we recommend running `check_link_consistency` after
    you have finished your graph-level operations.

    :param required_nodes: output nodes of the computation
    :param assume_inputed: nodes whose parents we will not include in the copy (these will be left dangling)
    :param tag: a string to preprend to the resulting node's names.
    :return: new_required, new_subgraph
        ``new_required``: a list containing the new copies of ``required_nodes`` in the new subgraph.

        ``new_subgraph``: a list containing new copies of all nodes in the subgraph.
    """
    subgraph_nodes = get_subgraph(required_nodes)

    # Maps old nodes to new nodes.
    node_mapping = {n: copy.copy(n) for n in subgraph_nodes}  # Shallow copy is still linked to the same torch modules
    if tag:
        for n in node_mapping.values():
            n.name = "({}){}".format(tag, n.name)

    # Set the parents and children of the new nodes to point to the new copies.
    # Children not used in the subgraph are dropped
    for n_old, n_new in node_mapping.items():
        n_new.parents = tuple(node_mapping[p] for p in n_old.parents)
        n_new.children = tuple(node_mapping[c] for c in n_old.children if c in node_mapping)
        # print()
        # print("Old node: ",n_old)
        # print("\t parents:",n_old.parents)
        # print("\t children:",n_old.children)
        # print("New node:",n_new)
        # print("\t parents:",n_new.parents)
        # print("\t children:",n_new.children)

    new_subgraph = [node_mapping[n] for n in subgraph_nodes]
    new_required = [node_mapping[r] for r in required_nodes]

    ### Checks that the operation correctly worked.

    # Graph is complete
    assert all(x in new_subgraph for x in get_connected_nodes(new_subgraph))

    # Nothing from the old graph is in the new graph
    assert all(x not in new_subgraph for x in get_connected_nodes(required_nodes))
    # None of the parents of cut-off nodes

    # For each parent p of the assumed inputted:
    # If p not a a parent of an unassumed node, then p should not be in the graph.
    assert all(
        p not in new_subgraph
        for ai in assume_inputed
        for p in ai.parents
        if all(p not in n.parents for n in new_subgraph if n not in assume_inputed)
    )

    # print(get_connected_nodes(new_required)) # Prior debug... change to logging?
    return new_required, new_subgraph


class GraphInconsistency(NodeOperationError):
    pass


def check_link_consistency(node_set):
    """
    Make sure that back-links in the graph are correct.
    Raises GraphInconsistency if the graph is not consistent.
    All connected nodes to the node set are analyzed.
    The links are consistent if each pair of child->parent has
    a corresponding link parent<-child.

    :param node_set: iterable of nodes to check

    :raises GraphInconsistency: if the graph is not consistent.

    :return: True
    """
    node_set = set(node_set)
    node_set = get_connected_nodes(node_set)
    forward_links = set((parent, child) for parent in node_set for child in parent.children)
    back_links = set((parent, child) for child in node_set for parent in child.parents)

    if forward_links == back_links:
        return True

    broken_forward = forward_links - back_links
    broken_back = back_links - forward_links
    raise GraphInconsistency(
        "Graph is inconsistent!\n" f"Broken forward links:{broken_forward}" f"Broke back links:{broken_back}"
    )


def replace_node(old_node, new_node, disconnect_old=False):
    """
    :param old_node: Node to replace
    :param new_node: Node to insert
    :param disconnect_old: If True, remove the old node connections from the graph.

    Replaces the children of old node with new node. Effectively,
    this just means going to the children and swapping out their parents to point to
    the new node. Ignores the children of old node that are parents of the new node
    in order to prevent cycles in the graph.

    If disconnect_old, remove references to the old node -- it will become unusable.

    .. Warning::
       This function changes the graph structure in-place.
       As such, if this function raises an error, it may result in a corrupted graph state.

    .. Warning::
       This function will try to coerce index states where possible. If the index types
       of the nodes are not listed, the index state will not be modified. If they are incompatible,
       this means that the graph will not function correctly.

    :return: None
    """

    new_node_requires = set(new_node.get_all_parents())

    if disconnect_old:
        if old_node in new_node_requires:
            raise NodeOperationError(
                "Cannot replace this node and remove the old one, because the"
                f"new node {new_node} depends on the old node {old_node}."
            )

    if isinstance(old_node, MultiNode):
        if not isinstance(new_node, MultiNode):
            raise TypeError("MultiNodes cannot be replaced with single nodes.")
        # Multi-nodes should always keep their children. To replace a multinode with
        # another one, we attempt to match their children with each other and perform
        # the replacement operation on the children.

        for c, cprime in _determine_multinode_child_match(old_node, new_node).items():
            replace_node(c, cprime, disconnect_old=False)

        if disconnect_old:
            old_node.disconnect()

    else:
        # If new node is a multinode, this will ensure we swap the main output in
        new_node = new_node.main_output
        # Convert index state if possible.
        new_node = soft_index_type_coercion(new_node, old_node._index_state)

        # Find children that need replacing
        swap_children = set(old_node.children) - set(new_node_requires)
        # Actually replace them
        for c in swap_children:
            c.swap_parent(old_node, new_node)

        if disconnect_old:
            old_node.disconnect()


def _determine_multinode_child_match(old_node: MultiNode, new_node: MultiNode):
    """
    Try to determine match between multinode children for `replace_node` on multinodes.
    :param old_node:
    :param new_node:
    :return:
    """

    try:
        # Try name-based matching first.
        matches = {getattr(old_node, name): getattr(new_node, name) for name in new_node._output_names}
    except AttributeError:
        # If name-based matching does not work, just match by order.
        matches = {co: cn for co, cn in zip(old_node.children, new_node.children)}
        # Raise an error if this match would leave a residual child used on the old node.

    # Raise error if there are children of the old node which do not match,
    # but are still needed for a computation.
    if any(node not in matches and len(node.children) > 0 for node in old_node.children):
        raise NodeOperationError(f"A match cannot be made between {old_node} and {new_node} children.")

    return matches


def replace_node_with_constant(node, value, name=None):
    vnode = ValueNode(value)
    if name is not None:
        vnode.name = name
    replace_node(node, vnode)


def search_by_name(nodes, name_or_dbname):
    """
    Look for a unique related node with a given db_name or name.
    The db_name will be given higher precedence.

    :param nodes: starting point for search
    :param name_or_dbname: name or dbname for node search
    :return: node that matches criterion

    Raises NodeAmbiguityError if more than one node found
    Raises NodeNotFoundError if no nodes found

    """
    try:
        return find_unique_relative(nodes, lambda n: n.db_name == name_or_dbname)
    except NodeNotFound:
        return find_unique_relative(nodes, lambda n: n.name == name_or_dbname)


def merge_children_recursive(start_nodes):
    """
    Merge children of some seed nodes if they are identical computations,
    and apply to future children until no more merges can be performed.

    This function changes a graph in-place.

    :param start_nodes:
    :return: Merged nodes.
    """

    all_merged_nodes = []
    while start_nodes:
        merged_nodes = merge_children(start_nodes)
        all_merged_nodes += merged_nodes
        next_nodes = []
        for node in merged_nodes:
            if isinstance(node, MultiNode):
                next_nodes += node.children
            else:
                next_nodes.append(node)

        start_nodes = next_nodes

    return all_merged_nodes


def merge_children(start_nodes):
    """
    Merge the children of some seed nodes if those children are identical.

    This function changes a graph in-place.

    :param start_nodes:
    :return: child_nodes: the merged children, post merge

    """

    from .nodes.tags import PairIndexer
    next_generation = list(set([c for s in start_nodes for c in s.children]))

    # Check same parents
    # Check same node type
    # Check same module type
    node_class_map = collections.defaultdict(list)
    for node in next_generation:
        node_class = (type(node), type(node.torch_module), node.parents)
        node_class_map[node_class].append(node)

    # Only merge things when there is more than one node of the same class
    considered_node_classes = {k: v for k, v in node_class_map.items() if len(v) > 1}

    # Check all nodes from the class have the same module state dict
    # Add exceptional code for PairIndexer by finding the one with the maximum distance threshold.
    mergeable_node_classes = []
    for nodes_to_merge in considered_node_classes.values():  # CODA

        first, *rest = nodes_to_merge
        # check if the state dict of nodes is all equal.
        d1 = first.torch_module.state_dict()
        for node in rest:
            d2 = node.torch_module.state_dict()
            if not is_equal_state_dict(d1, d2):
                nodes_can_merge = False
                break
        else:
            nodes_can_merge = True

        if not nodes_can_merge:
            continue  # DS AL CODA (back to considered_node_classes.values() iteration.)

        # Extra code to handle merging of pair indexers.
        # Even if we clean up the pair indexer with an extra state we would still need logic to merge them.
        # TODO: Clean up hard_dist_cutoff vs dist_hard max and make it impossible for the node and
        # TODO module to disagree.
        if isinstance(first, PairIndexer):
            max_r = first.torch_module.hard_dist_cutoff
            max_r_node = first
            swap_first_in = False
            for other_node in rest:
                this_r = other_node.torch_module.hard_dist_cutoff
                if this_r > max_r:
                    max_r_node = other_node
                    max_r = this_r
                    swap_first_in = True

            if swap_first_in:
                # Make the max_radius node the first one.
                first = max_r_node
                rest = [n for n in nodes_to_merge if n is not first]
                nodes_to_merge = first, *rest

        mergeable_node_classes.append(nodes_to_merge)

    # actually perform the merging after all the analysis is done.
    new_children = []  # These are the nodes that have been swapped into the graphs.
    for (first, *rest) in mergeable_node_classes:
        for equivalent_node in rest:
            replace_node(equivalent_node, first)
            equivalent_node.disconnect_recursive()
        new_children.append(first)

    return new_children


def vacuum_outputs(node_list, species_set):
    """

    :param node_list:
    :param species_set:
    :return:
    """
    # Dev note: assumes there is exactly one species and positions, and pair finder.
    # A more robust approach can be made by doing this for each set of these objects.
    # Dev note: this function is a bit domain specific and may not ultimately belong
    # in gops.py. As such we collect the domain-oriented imports in this function.

    import torch
    from .nodes.inputs import SpeciesNode, PositionsNode
    from .nodes.indexers import Encoder, PaddingIndexer, OneHotEncoder
    from .nodes.tags import PairIndexer

    species = find_unique_relative(node_list, SpeciesNode)
    pairs = find_unique_relative(node_list, PairIndexer)
    positions = find_unique_relative(node_list, PositionsNode)
    encoder = find_unique_relative(node_list, Encoder)
    pad_idxer = find_unique_relative(node_list, PaddingIndexer)

    copy_nodes = [*node_list, species, pairs, positions, encoder, pad_idxer]
    new_nodes, new_subgraph = copy_subgraph(copy_nodes, assume_inputed=[], tag='vacuum')

    *new_nodes, species, pairs, positions, encoder, pad_idxer = new_nodes

    with torch.no_grad():
        if not isinstance(species_set, torch.Tensor):
            species_set = torch.as_tensor(species_set, dtype=torch.int64)
        species_input = species_set.unsqueeze(1)
        n_species = species_input.shape[0]
        replace_node_with_constant(species, species_input, name="vacuum_species")

        pos_input = torch.zeros((n_species, 1, 3))
        replace_node_with_constant(positions, pos_input, name="vacuum_positions")

        empty_int = torch.empty((0,), dtype=torch.int64)

        # further specialize if encoder is OneHotEncoder
        if isinstance(encoder, OneHotEncoder):
            species_encoding = torch.eye(n_species, dtype=torch.int64)
            # Unsqueeze adds atom-wise axis for MolAtom index type.
            replace_node_with_constant(encoder.encoding, species_encoding.unsqueeze(1), name="vacuum_encoding")
            replace_node_with_constant(pad_idxer.indexed_features, species_encoding, name="vacuum_indexed_features")

        empty_pairs = ValueNode(empty_int)
        empty_pairs.name = "empty_pair_indices"
        for pn in pairs.pair_first, pairs.pair_second:
            replace_node(pn, empty_pairs)

        empty_float = torch.empty((0,), dtype=torch.get_default_dtype())
        replace_node_with_constant(pairs.pair_dist, empty_float, "empty_distances")

        empty_displacement = empty_float.unsqueeze(1).expand(-1, 3)
        replace_node_with_constant(pairs.pair_coord, empty_displacement, "empty_displacements")

    return new_nodes

def swap_pairfinders(node_or_nodes, new_pairfinder, new_node_name=None, cell_node=None, module_kwargs={}):
    """
    Finds and replaces existing PairIndexer node with a new one, potentially adjusting its parent nodes
    if needed.

    :param node_or_nodes: the PairIndexer node to be replaced, or a node or list of nodes connected
    to the unique PairIndexer to be replaced
    :param new_pairfinder: class of new PairIndexer 
    :param new_node_name: name for new PairIndexer node, if None the name of the replaced PairIndexer
    will be used, defaults to None
    :param cell_node: should be specified if new PairIndexer requires a CellNode and one does not currently
    exist in computational graph, if None a search of existing nodes will be conducted if a CellNode is 
    needed, defaults to None
    :param module_kwargs: arguments to feed into the new PairIndexer constructor, defaults to {}

    .. Note::
    
        If this function is used to add a CellNode to the computational graph, any existing GraphModule or 
        Predictor will need to be reinitialized.
    """

    from .nodes.tags import PairIndexer, Positions, Species
    from .nodes.inputs import CellNode

    if isinstance(node_or_nodes, PairIndexer):
        old_pf = node_or_nodes
    else:
        old_pf = find_unique_relative(node_or_nodes, PairIndexer)

    positions = find_unique_relative(old_pf, Positions)
    species = find_unique_relative(old_pf, Species)

    new_node_name = (new_node_name or old_pf.name)

    try:
        new_pf = new_pairfinder(new_node_name, parents=(positions, species), **module_kwargs)
    except RuntimeError:
        cell = (cell_node or find_unique_relative(old_pf, CellNode))
        new_pf = new_pairfinder(new_node_name, parents=(positions, species, cell), **module_kwargs)
    
    replace_node(old_pf, new_pf, disconnect_old=True)
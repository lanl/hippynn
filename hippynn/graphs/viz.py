"""
Functions for visualizing nodes and their links with graphviz.
"""
from graphviz import Digraph

from . import get_connected_nodes
from .nodes.base import MultiNode


def visualize_graph_module(graph, compactify=True):
    """
    Make a dot graph for a GraphModule.

    :param graph: A GraphModule
    :param compactify: see :func:`visualize_node_set`

    :returns: graphviz.Digraph
    """
    node_set = {*graph.forward_output_list, *graph.input_nodes}
    preserve = set(graph.nodes_to_compute)
    return visualize_node_set(node_set, compactify=compactify, preserve=preserve)


def visualize_connected_nodes(node_iterable, compactify=True):
    """

    Make a dot graph for a all nodes connected to a group of nodes.

    :param node_iterable: An iterable (tuple, list, etc)  of nodes.
    :param compactify: see :func:`visualize_node_set`

    :returns: graphviz.DiGraph
    """
    preserve = set(node_iterable)
    node_set = get_connected_nodes(preserve)
    return visualize_node_set(node_set, preserve=preserve, compactify=compactify)


def visualize_node_set(node_set, compactify, preserve=None):
    """

    Make a dot graph for a precise set of nodes.

    :param node_set: The set of nodes to visualize.
    :param compactify: If True, do not show the individual outputs of a `MultiNode`. If True, show all nodes.
    :param preserve:

    :returns: graphviz.Digraphs

    """

    dot = Digraph()
    starting_dot_names = get_viz_node_names(node_set)
    if compactify:
        dot_names = compactify_node_names(starting_dot_names, preserve=preserve)
    else:
        dot_names = starting_dot_names

    # Create the visible nodes
    for node in dot_names:
        if dot_names[node] == starting_dot_names[node]:
            node_attr = {"style": "rounded", "shape": "rect"}

            if isinstance(node, MultiNode):
                node_attr["style"] = "rounded,bold"

            if node.parents == ():
                node_attr["color"] = "green"
            elif preserve and node in preserve:
                node_attr["color"] = "blue"
            elif all(c not in dot_names for c in node.children):
                node_attr["color"] = "red"

            dn = dot.node(dot_names[node], **node_attr)

    # Create the links between nodes:
    for node in node_set:
        for child in node.children:
            if child in node_set:
                n1 = dot_names[node]
                n2 = dot_names[child]
                if n1 != n2:  # supports compactify option
                    dot.edge(n1, n2)

    return dot


def get_viz_node_names(node_set):
    """
    :param node_set:
    :return:
    :meta private:
    """

    base_names = set()
    nonunique_names = set()
    for node in node_set:
        name = node.name
        if name in base_names:
            nonunique_names.add(node.name)
        base_names.add(node.name)

    unique_names = {}
    for node in node_set:
        if node.name in nonunique_names:
            unique_names[node] = "{}<id={}>".format(node.name, hex(id(node)))
        else:
            unique_names[node] = node.name
    return unique_names


def compactify_node_names(names, preserve):
    """
    :meta private:
    Generate a list of node names where multinode outputs
    are mapped to the base multinode name, unless that
    output is in the ``preserve`` set.

    :param names: dict mapping nodes to names.
    :param preserve: nodes to keep explicit.
    :return: new dict of nodes to names.
    """
    copy_names = names.copy()
    if preserve is None:
        preserve = set()  # Empty; nothing
    for node in names:
        if isinstance(node, MultiNode):
            for child in node.children:
                if child not in preserve:
                    copy_names[child] = names[node]

    return copy_names

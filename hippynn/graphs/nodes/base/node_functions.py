"""
Here we define some of the core operations for Nodes in the directed
acyclic graph associated with a computation.
"""
import warnings
from typing import Optional, Tuple

import torch
from .. import _debprint
from ...._deprecations import _DeprecatedNamesMixin

DEFAULT_WHY_DESC = "<purpose not specified>"

import typing
if typing.TYPE_CHECKING:
    from .base import Node



class _NodeFunctions(_DeprecatedNamesMixin):
    """
    Base Node methods without dynamic algebra behavior.
    """
    input_names = NotImplemented
    
    _DEPRECATED_NAMES = {
        "_input_names" : "input_names",
    }

    def __init__(self, name, parents, db_name=None, module=None):
        """

        :param name: name of node
        :param parents: parent nodes of the target
        :param db_name: name of target in database (if any)
        :param module: input pytorch module (if 'auto', attempt to create)


        """

        assert self not in parents, "Nodes cannot be their own parents."

        if not isinstance(name, str):
            raise TypeError("Node names must be strings. Instead got: {}".format(name))
        
        if isinstance(parents, _NodeFunctions):
            parents = parents, # wrap a single node as a tuple of parents.

        self.name: str = name
        self.origin_node: Optional[Node] = None  # Loss input nodes set this attribute to find references to the model graph
        self.parents: Tuple[Node] = tuple(parents)
        self.children: Tuple[Node] = tuple()
        for p in self.parents:
            p.children = p.children + (self,)

        self._pred: Optional[Node] = None
        self._true: Optional[Node] = None
        
        # The db_name must be set after parents, _pred, and _true,
        # because assigning a db_name will assign to the true and pred versions of a node, as well;
        # this in turn requires having these things as well as the parents to define if they exist or not.
        self.db_name: Optional[str] = db_name 

        
            
        # Otherwise, glue the module on
        if module is not None:
            if module == "auto":
                import warnings
                warnings.warn("Auto module was specified but not created during node construction." \
                " If a torch module is not supplied then execution will fail.")
            self.torch_module: torch.nn.Module = module
        
            # In this case, the node must represent input tensors.

    def get_ancestors(self):
        """
        Gets all parents of this node and recursively to input nodes. This node is not included in the output.
        """
        return get_connected_nodes(self.parents, ancestors=True, descendants=False)
        #return self.parents + tuple(pnode for parent in self.parents for pnode in parent.get_all_parents())

    def get_descendants(self):
        """
        Gets all children of this node and recursively forward. This node is not included in the output.
        """
        return get_connected_nodes(self.children, ancestors=False, descendants=True)
        #return self.children + tuple(ccnode for child in self.children for ccnode in child.get_all_children())

    # Functions that take either a node or a node set can be accessed as attributes.

    def get_all_connected(self, ancestors=True, descendants=True):
        return get_connected_nodes({self}, ancestors=ancestors, descendants=descendants)

    def find_unique_relative(self, constraint, why_desc=DEFAULT_WHY_DESC):
        return find_unique_relative(self, constraint, why_desc=why_desc)

    def find_relatives(self, constraint, ancestors=True, descendants=True, why_desc=DEFAULT_WHY_DESC):
        return find_relatives(self, constraint, ancestors=ancestors, descendants=descendants, why_desc=why_desc)
    
    def is_in_loss_graph(self, why_desc=DEFAULT_WHY_DESC):
        """Return whether this node is in the loss graph or the model graph."""
        return is_in_loss_graph(self, why_desc=why_desc)
        

    def swap_parent(self, old, new):
        if old not in self.parents:
            raise NodeOperationError(f"Node {old} is not a parent of node {self}")
        old.children = tuple(c for c in old.children if c is not self)
        new.children = *new.children, self
        self.parents = tuple(new if p is old else p for p in self.parents)

    def disconnect(self):
        """
        Remove this node from the graph, leaving it uncomputable.

        :return:
        """
        for p in self.parents:
            p.children = tuple(c for c in p.children if c is not self)
        self.parents = ()

    def disconnect_recursive(self):
        """
        Remove this node from the graph, then disconnect all children, recursively.

        Afterwards this node and all children thereof will no longer be computable.

        :return:
        """
        self.disconnect()
        for c in self.children:
            c.disconnect_recursive()


    ## Properties to implement in concrete nodes.

    def true(self):
        return NotImplemented
    
    def pred(self):
        return NotImplemented
    
    def main_output(self):
        return NotImplemented    
    
    def db_name(self):
        return NotImplemented

    def __dir__(self):
        dir_ = super().__dir__()
        # need to protect against a case where input names are not specified.
        # otherwise dir() will raise an error. Debuggers hate that!
        if self.input_names is not NotImplemented:
            dir_ = dir_ + list(self.input_names)
        return dir_

    def __getattr__(self, item):
        
        if item in ("parents", "input_names"):  # Guard against recursion
            raise AttributeError("Attribute {} not yet present".format(item))
        try:
            return self.parents[self.input_names.index(item)]
        except (AttributeError, ValueError) as ee:
            pass

        return super().__getattr__(item)
        


    def __repr__(self):
        try:
            name = self.name
        except AttributeError:
            name = "UNINITIALIZED"
        return "{}('{}')<{}>".format(self.__class__.__name__, name, hex(id(self)))


class NodeOperationError(Exception):
    pass

class NodeNotFound(NodeOperationError):
    pass

class NodeAmbiguityError(NodeOperationError):
    pass



def get_connected_nodes(node_set, ancestors=True, descendants=True):
    """
    Recursively return nodes connected to the specified node_set.

    Nodes in the supplied set are included in the output.

    :param node_set: iterable collection of nodes (list, tuple, set,...)
    :param ancestors: whether to search ancestors of the node set
    :param descendants: whether to search descendants of the node set

    :return: set of nodes with some relationship to the input set.
    """
    search_from = set(node_set)
    search_found = set()
    # Very naive algorithm, but we don't anticipate large graphs.
    while len(search_from) != 0:
        for node in search_from.copy():
            search_found.add(node)
            search_from.remove(node)
            if ancestors:
                for node_relative in node.parents:
                    if node_relative not in search_found:
                        search_from.add(node_relative)
            if descendants:
                for node_relative in node.children:
                    if node_relative not in search_found:
                        search_from.add(node_relative)
    return search_found

def get_ancestors(node_set):
    return get_connected_nodes(node_set, ancestors=True, descendants=False)

def get_descendants(node_set):
    return get_connected_nodes(node_set, ancestors=False, descendants=True)

def find_relatives(node_or_nodes, constraint_key, ancestors=True, descendants=True, why_desc=DEFAULT_WHY_DESC):
    """

    :param node_or_nodes: a node or iterable of nodes to start the search.
    :param constraint_key: 1) callable to filter nodes by or
                        2) type spec to be used with `isinstance`.
    :param ancestors: whether to search ancestors of the node set
    :param descendants: whether to search descendants of the node set
    :param why_desc: If a node cannot be found satisfying the constraint, raise an error with this message.

    :return: set of nodes related to this node that obey a constraint
    """

    if isinstance(constraint_key, type):
        # We must bind the constraint key to a name within the lambda -- search_type
        constraint_key = lambda node, *, search_type=constraint_key: isinstance(node, search_type)
    elif callable(constraint_key):
        pass
    else:
        raise ValueError("constraint must be a type or callable filter function")

    from . import Node
    if isinstance(node_or_nodes, Node):  # if we search from a node, wrap it as a collection
        node_or_nodes = [node_or_nodes]
        _debprint("Starting search from single node")

    relatives = get_connected_nodes(node_or_nodes, ancestors=ancestors, descendants=descendants)
    candidates = {n for n in relatives if constraint_key(n)}

    if len(candidates) == 0:
        _debprint("Node not found, all relatives:")
        for n in relatives:
            _debprint(n)
        raise NodeNotFound("({}) Missing: Could not automatically find satisfying node in graph.".format(why_desc))

    return candidates


def find_unique_relative(node_or_nodes, constraint, ancestor_fallback=True, why_desc=DEFAULT_WHY_DESC):
    """
    Look for a unique parent or child node type in the graph connected to the starting node.

    :param node_or_nodes: a node or iterable of nodes to start the search.
    :param constraint:
        1. callable to filter nodes by or
        2. type to be used with `isinstance`.
    :param ancestor_fallback: This sets whether or not a unique relative in only the ancestors is acceptable.
         If set to true, and multiple nodes are found, the search will be run
         again with only the ancestors of the initial node set.

    :param why_desc: specification of error message

    :return: Node compatible with constraint.

    .. Note::
        If no node is found, a NodeNotFoundError is rasied.
        If more than one node is found, a NodeambiguityError is raised.
    """

    candidates = find_relatives(node_or_nodes, constraint_key=constraint, why_desc=why_desc)

    if ancestor_fallback and len(candidates) > 1:
        # We wanted a unique node but could not find one. But this fallback
        # allows us to look only at ancestors and see if that result is unique.
        candidates = find_relatives(node_or_nodes,
                                    constraint_key=constraint,
                                    descendants=False,
                                    ancestors=True,
                                    why_desc=why_desc)

    if len(candidates) > 1:
        raise NodeAmbiguityError("({}) Ambiguity: Multiple {} nodes found:{}".format(why_desc, constraint, candidates))

    result = candidates.pop()
    _debprint("Found node {} of type {}: {}".format(result, constraint.__name__, why_desc))
    return result


def is_in_loss_graph(node_or_nodes, why_desc=DEFAULT_WHY_DESC):
    """
    Decide if a node or collection of nodes is in the loss graph.

    (If not, they are in the model graph)
    (If neither, raise NodeAmbiguityError)

    .. Warning::
        If you call this function, it ought to be on a set of nodes assumed in the same graph.
        If not, be prepared for the case that the question was malformed (mixture) and so ``NodeAmbiguityError`` is raised.

    :param node_or_nodes: a node or iterable of nodes to examine.
    :param why_desc: optional specification of error message clarifying reason why this was requested.

    :return: boolean
    """

    from .base import InputNode, LossInputNode
    try:
        inputs_for_nodes = find_relatives(node_or_nodes, InputNode, descendants=False)
    except NodeNotFound:
        # In this case, we probably have a tree of pure ValueNodes. We will arbitrarily call this "in the model graph."
        return False 
    
    if any(isinstance(in_node, LossInputNode) for in_node in inputs_for_nodes): 
        # If any inputs are in the loss graph, we must ensure that they all are, or else
        # the graph state has been corrupted.
        if not all(isinstance(in_node, LossInputNode) for in_node in inputs_for_nodes):
            raise NodeAmbiguityError("This node_or_nodes is both in and out of the loss graph. " \
                    f"Requested for purpose: {why_desc}")
        # Graph is not corrupted!
        return True
    else:
        # No inputs were in the loss graph
        return False
    






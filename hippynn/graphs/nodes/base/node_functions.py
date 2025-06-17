"""
Base node definition.
"""
from .. import _debprint

DEFAULT_WHY_DESC = "<purpose not specified>"


class _BaseNode:
    _input_names = NotImplemented
    _LossPredNode = None  # Will be set by this child class when it exists
    _LossTrueNode = None  # Same

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

        self.db_name = db_name
        self.origin_node = None  # Loss input nodes set this attribute to find references to the model graph
        self.parents = tuple(parents)
        self.name = name
        self._pred = None
        self._true = None
        self.children = tuple()
        for p in self.parents:
            p.children = p.children + (self,)

        # If specified, trigger automatic module generation
        if module == "auto":
            _debprint("Making auto module for", self.name)
            module = self.auto_module()
        # Otherwise, glue the module on
        if module is not None:
            self.torch_module = module

    def set_dbname(self, db_name):
        self.db_name = db_name
        if self._pred is not None:
            self._pred.db_name = db_name
        if self._true is not None:
            self._true.db_name = db_name

    @property
    def pred(self):
        if self._pred is None:
            self._pred = self._LossPredNode(self.name + "-pred", origin_node=self, db_name=self.db_name)
        return self._pred

    @property
    def true(self):
        if self._true is None:
            self._true = self._LossTrueNode(self.name + "-true", origin_node=self, db_name=self.db_name)
        return self._true

    def get_all_parents(self):
        return self.parents + tuple(pnode for parent in self.parents for pnode in parent.get_all_parents())

    def get_all_children(self):
        return self.children + tuple(ccnode for child in self.children for ccnode in child.get_all_children())

    # Functions that take either a node or a node set can be accessed as attributes.

    def get_all_connected(self):
        return get_connected_nodes({self})

    def find_unique_relative(self, constraint, why_desc=DEFAULT_WHY_DESC):
        return find_unique_relative(self, constraint, why_desc=why_desc)

    def find_relatives(self, constraint, why_desc=DEFAULT_WHY_DESC):
        return find_relatives(self, constraint, why_desc=why_desc)

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

    def auto_module(self):
        raise NotImplementedError("Auto module not defined for node {} of type {}".format(self, type(self)))

    def __dir__(self):
        dir_ = super().__dir__()
        # need to protect against a case where input names are not specified.
        # otherwise dir() will raise an error. Debuggers hate that!
        if self._input_names is not NotImplemented:
            dir_ = dir_ + list(self._input_names)
        return dir_

    def __getattr__(self, item):
        if item in ("parents", "_input_names"):  # Guard against recursion
            raise AttributeError("Attribute {} not yet present".format(item))
        try:
            return self.parents[self._input_names.index(item)]
        except (AttributeError, ValueError) as ee:
            raise AttributeError("{} object has no attribute '{}'".format(self.__class__, item))

    def __repr__(self):
        try:
            name = self.name
        except AttributeError:
            name = "UNINITIALIZED"
        return "{}('{}')<{}>".format(self.__class__.__name__, name, hex(id(self)))

    # Overridden by MultiNode, LossInputNode
    @property
    def main_output(self):
        return self


class NodeOperationError(Exception):
    pass


class NodeNotFound(NodeOperationError):
    pass


def get_connected_nodes(node_set, ancestors=True, descendants=True):
    """
    Recursively return nodes connected to the specified node_set.

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
                for node_relative in node.get_all_parents():
                    if node_relative not in search_found:
                        search_from.add(node_relative)
            if descendants:
                for node_relative in node.get_all_children():
                    if node_relative not in search_found:
                        search_from.add(node_relative)
    return search_found


class NodeAmbiguityError(NodeOperationError):
    pass


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

    if isinstance(node_or_nodes, _BaseNode):  # if we search from a node, wrap it as a collection
        node_or_nodes = [node_or_nodes]
        _debprint("Starting search from single node")

    candidates = {n for n in get_connected_nodes(node_or_nodes) if constraint_key(n)}

    for node in node_or_nodes:
        if constraint_key(node):
            candidates.add(node)

    if len(candidates) == 0:
        _debprint("Node not found, all relatives:")
        for n in get_connected_nodes(node_or_nodes):
            _debprint(n)
        raise NodeNotFound("({}) Missing: Could not automatically satisfying node in graph.".format(why_desc))

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

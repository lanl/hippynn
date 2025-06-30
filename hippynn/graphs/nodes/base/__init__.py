"""
Core classes and functions for nodes.

This subpackage is separated into separate files to reduce the complexity of understanding the definitions.

For a user, all relevant functions and names should be exposed here.

The basic functionality of a node in a directed graph is defined in node_functions.py as _NodeMethods.

To augment this, the registration of dunder methods is implemented in algebra.py.

Combining these, base.py defines many of the core node classes.

the multi.py module goes on to define a particular core class of import, the MultiNode, which itself represents
a single pytorch computation which returns multiple tensors.

Last but not least is definition_helpers, which contains optional features which help to simplify the creation
nodes using flexible parent types.
"""
from .node_functions import get_connected_nodes, get_ancestors, get_descendants, get_connected_nodes, find_relatives, find_unique_relative, is_in_loss_graph
from .node_functions import NodeAmbiguityError, NodeOperationError, NodeNotFound

# Basic node classes
from .base import Node, SingleNode, InputNode, LossInputNode, LossPredNode, LossTrueNode, ValueNode


# Node that provides multiple outputs
from .multi import MultiNode, IndexNode

# Optional mixins for simplifying the process of defining BaseNode subclasses
from .definition_helpers import AutoKw, AutoNoKw, ExpandParents

def __getattr__(name: str):
    import warnings
    if name == "_BaseNode":
        # Backwards compatibility for unpickling prior models
        warnings.warn(
            "'BaseNode' is a deprecated class name, please use 'Node'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Node
    raise AttributeError(f"module {__name__!r} has no attribute {name}")

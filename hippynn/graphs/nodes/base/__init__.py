"""
Core classes and functions for nodes
"""
from .node_functions import get_connected_nodes, get_ancestors, get_descendants, get_connected_nodes, find_relatives, find_unique_relative, is_in_loss_graph
from .node_functions import NodeAmbiguityError, NodeOperationError, NodeNotFound

# Basic node classes
from .base import Node, SingleNode, InputNode, LossInputNode, LossPredNode, LossTrueNode, BaseNode, ValueNode

#from .algebra import ValueNode

# Node that provides multiple outputs
from .multi import MultiNode, IndexNode

# Optional mixins for simplifying the process of defining BaseNode subclasses
from .definition_helpers import AutoKw, AutoNoKw, ExpandParents

def __getattr__(name: str):
    import warnings
    if name == "_BaseNode":
        # Backwards compatibility for unpickling prior models
        warnings.warn(
            "'BaseNode' is a deprecated class name, please use 'BaseNode'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return BaseNode
    raise AttributeError(f"module {__name__!r} has no attribute {name}")

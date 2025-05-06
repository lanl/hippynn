"""
Core classes and functions for nodes
"""
from .node_functions import find_unique_relative, find_relatives, get_connected_nodes

# Basic node classes
from .base import Node, SingleNode, InputNode, LossInputNode, LossPredNode, LossTrueNode, _BaseNode

# Node that provides multiple outputs
from .multi import MultiNode, IndexNode

# Optional mixins for simplifying the process of defining BaseNode subclasses
from .definition_helpers import AutoKw, AutoNoKw, ExpandParents

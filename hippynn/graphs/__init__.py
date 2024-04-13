"""
Graph objects in hipppynn. This package allows definitions of interfaces for torch modules.
The interfaces allow dynamic reprogramming of control flow between pytorch modules.

Index Types (i.e. Molecule, Atom, Pair) track the physical basis for a tensor.
Automatic index type conversion can be performed between compatible types.

Furthermore, "parent expansions" allow for flexible signatures for node creation
which attempt to hide book-keeping from the end-user.
"""
from . import indextypes
from .indextypes import clear_index_cache, IdxType

from .nodes import base, inputs
from .nodes.base import find_unique_relative, find_relatives, get_connected_nodes

from .gops import get_subgraph, copy_subgraph, replace_node, compute_evaluation_order

from .nodes import networks, targets, physics, loss, excited

# Needed to populate the registry of index transformers.
# This has to happen before the indextypes package can work,
# however, we don't want the indextypes package to depend on actual
# implementations of nodes.
from . import indextransformers

from .graph import GraphModule

from .predictor import Predictor
from .ensemble import make_ensemble

__all__ = [
    "get_subgraph",
    "copy_subgraph",
    "replace_node",
    "compute_evaluation_order",
    "find_unique_relative",
    "find_relatives",
    "get_connected_nodes",
    "GraphModule",
    "Predictor",
    "IdxType",
    "make_ensemble",
]

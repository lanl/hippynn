"""
Node class registry and validation. Per-class reconstruction dispatch on the structure (read)
side lives as a `from_record` classmethod on each `SerializableNode` subclass (e.g.
`AutoModule.from_record`); `ContextConverter.structure_node` looks up the class via `_ID_REGISTRY` and
calls `cls.from_record(...)` directly.
"""

import warnings

from ..experiment import _debprint
from .types import NodeTypeIdentifier
from ..graphs.nodes.base.base import SerializableNode


_NODE_REGISTRY: dict[type, NodeTypeIdentifier] = {}
_ID_REGISTRY: dict[NodeTypeIdentifier, type] = {}


def _populate_registry():
    for cls in SerializableNode._unregistered_classes.copy():
        register_node_cls(cls)
        SerializableNode._unregistered_classes.remove(cls)


def register_node_cls(cls):
    nodetype = get_nodetype(cls)
    _debprint(f"Registering {cls=}, {type(nodetype)=}, {nodetype=}")

    _NODE_REGISTRY[cls] = nodetype
    _ID_REGISTRY[nodetype] = cls


def get_nodetype(node_cls):
    qualname = node_cls.__module__.split(".")

    # first two locations are common locations for hippynn nodes
    # they are treated separately as packages. Nodes defined within these subpcakges
    # must have references in the corresponding namespace.
    if qualname[:3] == "hippynn.graphs.nodes".split("."):
        if len(qualname) < 4:
            raise ValueError(
                f"{node_cls!r} is defined directly in 'hippynn.graphs.nodes' ({node_cls.__module__!r}); "
                "node classes must live in a namespaced subpackage, e.g. 'hippynn.graphs.nodes.base'."
            )
        namespace = qualname[3]
        package = ".".join(qualname[:3])
    elif qualname[:2] == "hippynn.interfaces".split("."):
        if len(qualname) < 3:
            raise ValueError(
                f"{node_cls!r} is defined directly in 'hippynn.interfaces' ({node_cls.__module__!r}); "
                "node classes must live in a namespaced subpackage, e.g. 'hippynn.interfaces.ase_interface'."
            )
        namespace = qualname[2]
        package = ".".join(qualname[:2])
    else:
        package = qualname[0]
        namespace = ".".join(qualname[1:-1])

    name = node_cls.__qualname__

    return NodeTypeIdentifier(name=name, namespace=namespace, package=package)


def ensure_registry_consistency(raise_=False):
    """
    Verify that `_NODE_REGISTRY` (class -> identifier) and `_ID_REGISTRY` (identifier -> class)
    are inverses of one another.
    """
    _populate_registry()

    problems = []
    for cls, nodetype in _NODE_REGISTRY.items():
        found = _ID_REGISTRY.get(nodetype)
        if found is not cls:
            problems.append(f"{nodetype}: registered for {cls!r}, but _ID_REGISTRY resolves it to {found!r}")

    for nodetype, cls in _ID_REGISTRY.items():
        found = _NODE_REGISTRY.get(cls)
        if found != nodetype:
            problems.append(f"{cls!r}: registered as {nodetype}, but _NODE_REGISTRY resolves it to {found!r}")

    if problems:
        message = "Registry round-trip mismatch(es):\n" + "\n".join(problems)
        if raise_:
            raise ValueError(message)
        else:
            warnings.warn(message, stacklevel=2)

    return problems

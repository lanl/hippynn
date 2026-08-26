"""
Node class registry and validation. Per-class reconstruction dispatch on the structure
(read) side now lives in `hooks.py` as `ContextConverter.register_structure_family` hooks,
matched on the same node class hierarchy this module registers.
"""

import importlib
import warnings

from ..experiment import _debprint
from .types import NodeTypeIdentifier
from ..graphs.nodes.base import LossInputNode
from ..graphs.nodes.base.base import _SerializableNode


_NODE_REGISTRY: dict[type, NodeTypeIdentifier] = {}
_ID_REGISTRY: dict[NodeTypeIdentifier, type] = {}

# Attribute names on a Node subclass that hold embedded live-Node values outside `.parents`,
# consumed by the Node <-> NodeReference hook pair in hooks.py.
NODE_REFERENCE_ATTRS: dict[type, tuple[str, ...]] = {
    LossInputNode: ("origin_node",),
}


def node_reference_attrs(cls: type) -> tuple[str, ...]:
    """Return the reference-attribute names for `cls`, walking its MRO."""
    for klass in cls.__mro__:
        attrs = NODE_REFERENCE_ATTRS.get(klass)
        if attrs is not None:
            return attrs
    return ()


def _populate_registry():
    for cls in _SerializableNode._unregistered_classes.copy():
        register_node_cls(cls)
        _SerializableNode._unregistered_classes.remove(cls)


def register_node_cls(cls):
    _debprint(f"Registering {cls=}")
    nodetype = get_nodetype(cls)
    _debprint(f"{type(nodetype)=}, {nodetype=}")
    if nodetype in _ID_REGISTRY:
        raise ValueError(f"Registry of duplicate {nodetype=}")
    _NODE_REGISTRY[cls] = nodetype
    _ID_REGISTRY[nodetype] = cls


def get_nodetype(node_cls):
    qualname = node_cls.__module__.split(".")

    # first two locations are common locations for hippynn nodes
    # they are treated separately as packages. Nodes defined within these subpcakges
    # must have references in the corresponding namespace.
    # todo: add validation to this.
    if qualname[:3] == "hippynn.graphs.nodes".split("."):
        namespace = qualname[3]
        package = ".".join(qualname[:3])
    elif qualname[:2] == "hippynn.interfaces".split("."):
        namespace = qualname[2]
        package = ".".join(qualname[:2])
    else:
        package = qualname[0]
        namespace = ".".join(qualname[1:-1])

    name = node_cls.__qualname__

    return NodeTypeIdentifier(name=name, namespace=namespace, package=package)


def ensure_registry_namespaces(raise_=False):
    """
    Verify that every registered node class is actually reachable via
    ``getattr(importlib.import_module(f"{package}.{namespace}"), name)``, and that
    the object found there is the same class that was registered.
    """
    _populate_registry()

    problems = []
    for cls, nodetype in _NODE_REGISTRY.items():
        module_path = f"{nodetype.package}.{nodetype.namespace}"
        try:
            module = importlib.import_module(module_path)
        except ImportError as ee:
            problems.append(f"{nodetype}: could not import {module_path!r}: {ee}")
            continue

        found = getattr(module, nodetype.name, None)
        if found is not cls:
            problems.append(f"{nodetype}: {module_path}.{nodetype.name} resolves to {found!r}, expected {cls!r}")

    if problems:
        message = "Registry namespace mismatch(es):\n" + "\n".join(problems)
        if raise_:
            raise LookupError(message)
        else:
            warnings.warn(message, stacklevel=2)

    return problems


def _automodule_kwarg_mapping(cls) -> dict:
    auto_module_kwargs = cls.auto_module_kwargs
    if not auto_module_kwargs:
        return {}
    if isinstance(auto_module_kwargs, dict):
        return auto_module_kwargs
    return dict(zip(auto_module_kwargs, auto_module_kwargs))



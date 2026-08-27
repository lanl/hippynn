"""
Node class registry and validation. Per-class reconstruction dispatch on the structure (read)
side lives as a `from_record` staticmethod on each direct `_SerializableNode` subclass (e.g.
`AutoModule.from_record`); `register_structure_hooks` below finds and registers them as
`ContextConverter.register_structure_family` hooks.
"""

import warnings

from ..experiment import _debprint
from .processor import ContextConverter
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
    nodetype = get_nodetype(cls)
    _debprint(f"Registering {cls=}, {type(nodetype)=}, {nodetype=}")

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


def _dispatch_from_record(data, typ, *, ctx):
    """Resolves `from_record` on the actual runtime type, so a stricter subclass (e.g. IndexNode
    under AutoModule) can override the class its family hook was registered against."""
    return typ.from_record(data, typ, ctx=ctx)


def register_structure_hooks():
    """
    Find each direct `_SerializableNode` subclass's own `from_record` staticmethod and register
    a `ContextConverter.register_structure_family` hook keyed on that class, dispatching to
    `typ.from_record` at structure time rather than the registering class's, so subclasses further
    down the MRO may override it.
    """
    for cls in _SerializableNode.__subclasses__():
        if "from_record" not in cls.__dict__:
            raise TypeError(f"{cls!r} is a direct subclass of _SerializableNode but does not define its own `from_record`.")
        ContextConverter.register_structure_family(lambda t, cls=cls: isinstance(t, type) and issubclass(t, cls))(_dispatch_from_record)



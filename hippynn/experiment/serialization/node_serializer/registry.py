"""
Node class registry, validation, and per-class reconstruction dispatch.
"""

import importlib
import warnings
from typing import Callable

from .. import _debprint
from .types import NodeInfo, NodeTypeIdentifier
from ....graphs.nodes.base import AutoModule, InputNode, LossInputNode
from ....graphs.nodes.base.base import _SerializableNode


_NODE_REGISTRY: dict[type, NodeTypeIdentifier] = {}
_ID_REGISTRY: dict[NodeTypeIdentifier, type] = {}

# name, parents, kwargs, info -> Node; built once per class at registration time.
_RECONSTRUCTORS: dict[type, Callable[[str | None, tuple, dict, NodeInfo], object]] = {}

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
    _RECONSTRUCTORS[cls] = _build_reconstructor(cls)


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


def _build_reconstructor(cls) -> Callable:
    """
    Build a (name, parents, kwargs, info) -> Node callable for `cls`, computed once at
    registration time instead of re-branched on every deserialize call.

    `db_name`/`index_state`/`requires_grad` post-construction assignment is *not* done here
    for any branch; it is applied uniformly by the resolver in hooks.py after calling this
    reconstructor, regardless of node type. `InputNode` is the one branch that also needs
    `db_name`/`index_state` at construction time (its `__init__` derives a default `name`
    from `db_name`, and assigns `index_state` itself when the class has no fixed one) — for
    that branch `info` (the full `NodeInfo` record) supplies them directly, since input nodes
    have no `module_kwargs` to draw them from.
    """
    if issubclass(cls, LossInputNode):
        # LossInputNode/LossPredNode/LossTrueNode take no `index_state` argument at all;
        # they derive it themselves from `origin_node.index_state`. `origin_node` arrives
        # here already resolved to a real Node, folded into `kwargs` via NODE_REFERENCE_ATTRS
        # on the write side and resolved through the same Serializable machinery as any other
        # embedded reference.
        return lambda name, parents, kwargs, info: cls(name=name, origin_node=kwargs["origin_node"], db_name=info.db_name)

    if issubclass(cls, InputNode):
        return lambda name, parents, kwargs, info: cls(name=name, db_name=info.db_name, index_state=info.index_state)

    if issubclass(cls, AutoModule):
        mapping = _automodule_kwarg_mapping(cls)

        def _build_automodule(name, parents, kwargs, info, mapping=mapping):
            node_kwargs = {}
            leftover_module_kwargs = {}
            for k, v in kwargs.items():
                if k in mapping:
                    node_kwargs[mapping[k]] = v
                else:
                    leftover_module_kwargs[k] = v
            return cls(name, parents, module_kwargs=leftover_module_kwargs, **node_kwargs)

        return _build_automodule

    spec = getattr(cls, "_deserialize_spec", None)
    if spec is not None:

        def _build_from_spec(name, parents, kwargs, info, spec=spec):
            args = []
            call_kwargs = {}
            for token in spec:
                if token == "name":
                    args.append(name)
                elif token == "parents":
                    args.append(parents)
                elif token == "parents*":
                    args.extend(parents)
                elif token == "kwargs":
                    call_kwargs = kwargs
                else:
                    raise ValueError(f"Unknown _deserialize_spec token {token!r} on {cls!r}")
            return cls(*args, **call_kwargs)

        return _build_from_spec

    return lambda name, parents, kwargs, info: cls(name, parents, **kwargs)

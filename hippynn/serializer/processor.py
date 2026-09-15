"""
The ContextConverter class: a global cattrs Converter plus tensor/node identity caches.

"""

import itertools
from typing import Any, get_origin

import torch
from cattrs import Converter

from .registry import _ID_REGISTRY, _NODE_REGISTRY
from .types import (
    NodeIdentifier,
    NodeRecord,
    NodeReference,
    ScalarPrimitive,
    Serializable,
    Tagged,
    TaggedIdxType,
    TaggedSet,
    TaggedTuple,
    TensorReference,
)
from ..graphs.indextypes import IdxType
from ..graphs.nodes.base import Node


CONVERTER = Converter()


class ContextConverter:
    """
    Holds one cattrs `Converter` plus the tensor/node identity caches used by its hooks (defined
    as methods below and registered onto `CONVERTER` at the bottom of this file).
    `context_converter` below is the single instance used for the whole process - a Converter's
    dispatch machinery caches resolved hooks per type, so reusing one is the intended usage, and
    every hook is bound to `self` directly, at registration time, since there is only ever this one
    instance to resolve. `reset()` clears the per-session state back to empty;
    `unstructure_with_context()`/`structure_with_context()` each call it themselves (before
    installing blobs), and again after resolving, so callers never need to manage reset timing
    directly. A session is single-threaded and non-reentrant - two independent sessions cannot be
    interleaved.

    `unstructure`/`structure` are the recursive, in-session calls used by hooks while a session is
    active; `unstructure_with_context`/`structure_with_context` are the one-shot, self-contained
    sessions built on top of them.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all per-session state, ready for a fresh serialize()/deserialize() call."""

        self._unstructuring = False
        self._structuring = False

        # For holding tensors that come as a side-payload
        self.tensor_blob: dict[int, Any] = {}
        self.tensor_ids: dict[int, int] = {}  # id(tensor) -> key, unstructure-side dedup
        self.tensor_objects: dict[int, Any] = {}  # key -> tensor, structure-side cache
        self._tensor_counter = itertools.count()  # dedicated key source, independent of len(tensor_blob)

        # For holding nodes that come as a side-payload
        self.node_blob: dict[int, Any] = {}
        self.node_ids: dict[int, int] = {}  # id(node) -> key, unstructure-side dedup
        self.node_objects: dict[int, Any] = {}  # key -> reconstructed node, structure-side cache
        self._node_resolving: set[int] = set()  # loop guard for the recursive resolver
        self._node_counter = itertools.count()  # dedicated key source, independent of len(node_blob) -

    def ensure_node(self, node):
        """
        Recursively convert node/parents into NodeReferences for unstructuring.
        """
        key = self.node_ids.get(id(node))
        if key is not None:
            return NodeReference(identifier=NodeIdentifier(id=key, name=node.name))

        # Reserve the key before recursing, so a cycle back to this node returns a reference
        # instead of rebuilding it; recurse parents first, so that when reconstructing we'll
        # have topo order.
        key = self.node_ids[id(node)] = next(self._node_counter)
        identifier = NodeIdentifier(id=key, name=node.name)
        reference = NodeReference(identifier=identifier)

        self.node_objects[key] = node

        parents = tuple(self.ensure_node(p) for p in node.parents) 
        state_dict = node.torch_module.state_dict(keep_vars=True) if getattr(node, "torch_module", None) else None
        record = NodeRecord(
            identifier=identifier,
            cls=_NODE_REGISTRY[type(node)],
            parents=parents,
            state=node.to_record(),
            state_dict=state_dict,
        )
        # Unstructure the whole record now, so cattrs' normal recursive hooks discover anything
        # reachable from `state` (e.g. `LossInputNode.origin_node`) as a side effect.
        self.node_blob[identifier.id] = self.unstructure(record)
        return reference

    def consume_node(self, node):
        """cattrs unstructure hook for live `Node` instances anywhere in the structure being unstructured."""
        return self.unstructure(self.ensure_node(node))

    def emit_node(self, data, type_):
        key = data["identifier"]["id"]

        if key in self.node_objects:
            return self.node_objects[key]

        if key in self._node_resolving:
            raise ValueError(f"Cycle detected while resolving node reference {key} - graphs must be acyclic.")

        self._node_resolving.add(key)
        try:
            record = self.node_blob[key]
            # Structuring `record.parents` (tuple[NodeReference, ...]) recurses into this same
            # hook for each parent, so `record.parents` already holds live Node objects here.
            record = self.structure(record, NodeRecord)
            cls = _ID_REGISTRY[record.cls]
            node = cls.from_record(name=record.identifier.name, parents=record.parents, **(record.state or {}))
            if record.state_dict:
                node.torch_module.load_state_dict(record.state_dict)
            self.node_objects[key] = node
            return node
        finally:
            self._node_resolving.discard(key)

    def consume_tensor(self, obj: torch.Tensor):
        key = self.tensor_ids.get(id(obj))
        if key is None:
            key = next(self._tensor_counter)
            self.tensor_ids[id(obj)] = key
            self.tensor_blob[key] = obj
        return self.unstructure(TensorReference(key=key))

    def emit_tensor(self, data: TensorReference, typ) -> torch.Tensor:
        key = data["key"]
        if key not in self.tensor_objects:
            self.tensor_objects[key] = self.tensor_blob[key]
        return self.tensor_objects[key]

    def unstructure_idxtype(self, obj: IdxType):
        return self.unstructure(TaggedIdxType(values=obj.name))

    def emit_idxtype(self, data, type_) -> IdxType:
        return IdxType[data["values"]]

    def unstructure_tuple(self, obj: tuple):
        return self.unstructure(TaggedTuple(values=list(obj)))
    
    def unstructure_set(self, obj: set):
        return self.unstructure(TaggedSet(values=list(obj)))

    def structure_tuple(self, obj, type_):
        return tuple(self.structure(v, Serializable) for v in obj["values"])

    def structure_set(self, obj, type_):
        return set(self.structure(v, Serializable) for v in obj["values"])


    def structure_serializable(self, data, type_):
        if isinstance(data, list):
            return self.structure(data, list[Serializable])
        if isinstance(data, dict):
            if "_type" in data:
                return self.structure(data, Tagged)
            return self.structure(data, dict[str, Serializable])
        return data


    def unstructure_with_context(self, obj):
        self._unstructuring = True
        try:
            structure = self.unstructure(obj)
            return_tensors = self.tensor_blob
            return_nodes = self.node_blob
        finally:
            self.reset()
        # todo validate number of constructed/reconstructed tensors/nodes matches!
        return structure, return_tensors, return_nodes

    def unstructure(self, obj, **kwargs):
        if not self._unstructuring:
            raise ValueError("ContextConverter.unstructure() may only be called during an active unstructure_with_context() session.")
        return CONVERTER.unstructure(obj, **kwargs)

    def structure_with_context(self, data, typ, *, tensor_blob, node_blob):
        self.reset()
        self.tensor_blob = tensor_blob
        self.node_blob = node_blob
        self._structuring = True
        try:
            return self.structure(data, typ)
        finally:
            self.reset()

        # todo validate number of constructed/reconstructed tensors/nodes matches!

    def structure(self, data, typ):
        if not self._structuring:
            raise ValueError("ContextConverter.structure() may only be called during an active structure_with_context() session.")
        return CONVERTER.structure(data, typ)


CONTEXT_CONVERTER = ContextConverter()

from cattrs.strategies import configure_union_passthrough
configure_union_passthrough(ScalarPrimitive.__value__, CONVERTER)

# Direct-type-key hooks: registered via cattrs' decorator-style dispatch (the key is inferred from
# the bound method's annotations - see "Use as Decorators" in the cattrs docs). Called rather than
# applied with literal `@` syntax, since these are shared-instance bound methods, not fresh defs.
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.consume_tensor)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.unstructure_tuple)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.unstructure_set)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.unstructure_idxtype)

# `emit_tensor`/`emit_node`/`emit_idxtype`: each one's return value is a live `torch.Tensor`/`Node`/
# `IdxType`, not the `TensorReference`/`NodeReference`/`TaggedIdxType` it's dispatched on - the
# decorator's return-type introspection would infer the wrong key, so all three use the explicit key
# form. This is also what lets the `Tagged` union (`structure_serializable` below) resolve a
# disambiguated member straight through to the live object, rather than cattrs' default structuring.
CONVERTER.register_structure_hook(TensorReference, CONTEXT_CONVERTER.emit_tensor)
CONVERTER.register_structure_hook(NodeReference, CONTEXT_CONVERTER.emit_node)
CONVERTER.register_structure_hook(TaggedIdxType, CONTEXT_CONVERTER.emit_idxtype)

# Predicate-based hooks, plus `structure_tuple`/`structure_set` below via hook factories: `TaggedTuple[Serializable]`/
# `TaggedSet[Serializable]` are parametrized generics, not valid `register_structure_hook` class keys, and their
# hooks intentionally return a different runtime type (`tuple`/`set`) than the type they're dispatched on - a hook
# *factory* (predicate -> hook, decorator-eligible) is the appropriate cattrs API for that, not a plain hook.
CONVERTER.register_unstructure_hook_func(
    lambda t: isinstance(t, type) and issubclass(t, Node), CONTEXT_CONVERTER.consume_node
    )

CONVERTER.register_structure_hook_func(
    lambda t: get_origin(t) is TaggedTuple,
    CONTEXT_CONVERTER.structure_tuple,
)

CONVERTER.register_structure_hook_func(
    lambda t: get_origin(t) is TaggedSet,
    CONTEXT_CONVERTER.structure_set,
)
CONVERTER.register_structure_hook_func(
    lambda t: t is Serializable,
    CONTEXT_CONVERTER.structure_serializable,
    )

## experimental
from functools import reduce
from operator import or_
from typing import get_args

from cattrs._compat import is_generic, is_union_type
from cattrs._generics import deep_copy_with
from cattrs.gen._generics import generate_mapping
from cattrs.typealiases import get_type_alias_base, is_type_alias


def install_union_alias_unwrapping(converter):
    def unwrap(t):
        base = get_type_alias_base(t)
        return deep_copy_with(base, generate_mapping(t)) if is_generic(t) else base

    def contains_alias(t):
        return is_union_type(t) and any(is_type_alias(a) for a in get_args(t))

    def factory(t, converter):
        expanded = reduce(
            or_,
            (unwrap(a) if is_type_alias(a) else a for a in get_args(t)),
        )
        return converter.get_structure_hook(expanded)

    converter.register_structure_hook_factory(contains_alias, factory)


install_union_alias_unwrapping(CONVERTER)

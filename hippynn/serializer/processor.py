"""
The ContextConverter class: a global cattrs Converter plus tensor/node identity caches.

"""

import itertools
from typing import Any, get_origin
from .types import NodeReference, NodeIdentifier, NodeRecord, Serializable, NodeBlob, ScalarPrimitive
from .registry import _NODE_REGISTRY, _ID_REGISTRY
from cattrs import Converter


import torch

from .registry import _ID_REGISTRY, _NODE_REGISTRY
from .types import NodeIdentifier, NodeRecord, NodeReference, Serializable, TaggedDict, TaggedList, TaggedSet, TaggedTuple, TensorReference, Tagged
from ..graphs.nodes.base import InputNode, Node


CONVERTER = Converter()


class ContextConverter:
    """
    Holds one cattrs `Converter` plus the tensor/node identity caches used by its hooks (see
    `hooks.py`/`registry.py`). `context_converter` below is the single instance used for the
    whole process - a Converter's dispatch machinery caches resolved hooks per type, so reusing
    one is the intended usage, and every hook is bound to `ctx=context_converter` directly, at
    registration time, since there is only ever this one instance to resolve. `reset()` clears the
    per-session state back to empty; `unstructure_with_context()`/`structure_with_context()` each
    call it themselves (before installing blobs), and again after resolving, so callers never need
    to manage reset timing directly. A session is single-threaded and non-reentrant - two
    independent sessions cannot be interleaved.

    `unstructure`/`structure` are the recursive, in-session calls used by hooks (`hooks.py`) while
    a session is active; `unstructure_with_context`/`structure_with_context` are the one-shot,
    self-contained sessions built on top of them.
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

        # For holding tensors that come as a side-payload
        self.node_blob: dict[int, Any] = {}
        self.node_ids: dict[int, int] = {}  # id(node) -> key, unstructure-side dedup
        self.node_objects: dict[int, Any] = {}  # key -> reconstructed node, structure-side cache
        self._node_resolving: set[int] = set()  # loop guard for the recursive resolver
        self._node_counter = itertools.count()  # dedicated key source, independent of len(node_blob) -

    def consume_node(self, node):
        """
        For already-visted nodes, return the reference for structuring.
        For new nodes, return a key for storing the node.
        """
        key = self.node_ids.get(id(node))
        if key is not None:
            return self.unstructure(NodeReference(identifier=NodeIdentifier(id=key, name=node.name)))

        # Recurse parents first, so that when reconstructing we'll have topo order.
        parent_references = self.unstructure(node.parents)

        # new key
        key = self.node_ids[id(node)] = next(self._node_counter)
        identifier =  NodeIdentifier(id=key, name=node.name)
        reference = NodeReference(identifier=identifier)
                
        self.node_objects[key]= node

        state_dict = node.torch_module.state_dict() if getattr(node, "torch_module", None) else None
        # trigger unstructuring on parents! But don't store the result in the record!
        # (this will be replaced with node references when the unstructuring is run on the blob)
        record = NodeRecord(
            identifier=identifier,
            cls = _NODE_REGISTRY[type(node)],
            parents = node.parents,
            state = node.to_record(),
            state_dict = state_dict,
        )
        # don't forget to store in blob!
        self.node_blob[identifier.id] = record
        return self.unstructure(reference)

    def emit_node(self, key):

        if key in self.node_objects:
            return self.node_objects[key]
        
        if key in self._node_resolving:
            raise ValueError(f"Cycle detected while resolving node reference {key} - graphs must be acyclic.")
        
        self._node_resolving.add(key)
        print("Deserializing", key, "resolving", self._node_resolving)
        try:
            record = self.node_blob[key]
            # restore record to hold live info

            record = self.structure(record, NodeRecord) # records hold references!
            cls = _ID_REGISTRY[record.cls]
            parents = [self.emit_node(p) for p in record.parents]
            node = cls.from_record(name=record.identifier.name, parents=parents, **(record.state or {}))
            if record.state_dict:
                node.torch_module.load_state_dict(record.state_dict)
            self.node_objects[key] = node
            print("Deserializing YAY finished safely",key)
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
        key = data.key
        if key not in self.tensor_objects:
            self.tensor_objects[key] = self.tensor_blob[key]
        return self.tensor_objects[key]

    def unstructure_idxtype(self, obj):
        return self.unstructure(TaggedTuple(values=list(obj)))

    def unstructure_tuple(self, obj: tuple):
        return self.unstructure(TaggedTuple(values=list(obj)))
    
    def unstructure_set(self, obj: set):
        return self.unstructure(TaggedSet(values=list(obj)))

    def structure_tuple(self, obj, type_):
        return tuple(self.structure(v, Serializable) for v in obj["values"])

    def structure_set(self, obj, type_):
        return set(self.structure(v, Serializable) for v in obj["values"])


    def structure_serializable(self, data, type_):
        print("SERIALIZABLE", repr(data), type(data))
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
            # Worklist: unstructuring a record's `state` may call `consume_node` for a node not yet
            # in `self.node_blob` (e.g. `LossInputNode.origin_node`), growing it mid-loop - so only
            # ever unstructure the keys not yet handled, and repeat until no new ones appear.
            return_nodes = {}
            while len(return_nodes) < len(self.node_blob):
                pending = {k: v for k, v in self.node_blob.items() if k not in return_nodes}
                return_nodes.update(self.unstructure(pending, unstructure_as=NodeBlob))
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
            self.node_blob = self.structure(node_blob, NodeBlob)
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
CONVERTER.register_structure_hook(CONTEXT_CONVERTER.emit_tensor)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.unstructure_tuple)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.unstructure_set)

# `emit_node`: its return value is a live `Node`, not the `NodeReference` it's dispatched on, so
# the decorator's return-type introspection doesn't apply - explicit key form.
CONVERTER.register_structure_hook(NodeReference, CONTEXT_CONVERTER.emit_node)

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
# def _resolve_serializable(data, ctx):
#     if data is None or isinstance(data, (bool, int, float, str)):
#         return data
    
#     if isinstance(data, TensorReference):
#         return ctx.structure(data, torch.Tensor)
#     if isinstance(data, NodeReference):
#         return ctx.structure(data, NodeReference)
#     if isinstance(data, TaggedList):
#         return list(ctx.structure(data.values, list[Serializable]))
#     if isinstance(data, TaggedSet):
#         return set(ctx.structure(data.values, list[Serializable]))
#     if isinstance(data, TaggedTuple):
#         return tuple(ctx.structure(data.values, list[Serializable]))
#     if isinstance(data, TaggedDict):
#         return dict(ctx.structure(data.values, dict[str, Serializable]))

#     # print("Weird type", data)
#     # return data
#     # # Todo maybe that was a horrible mistake?
#     print("Something went wrong! Unexpected data type:",data)
#     raise TypeError(f"Cannot deserialize value of type {type(data)!r}: {data!r}")

# # todo merge with above function, don't seem to have need for two functions??
# def _structure_serializable(data, typ, *, ctx):
#     # todo address note... what does this even mean
#     # NOTE: matching against the PEP-695 `Serializable` type-alias object itself (rather than
#     # something it resolves/unwraps to, e.g. via `.__value__`) is not yet verified against the
#     # pinned cattrs version - flagged in the plan's "verify against actual cattrs version" step.
#     # `_resolve_serializable` also relies on this same identity-match for `list[Serializable]`,
#     # since cattrs re-enters this hook per element when recursing into it.
#     print("structuring call",data,typ)
#     return _resolve_serializable(data, ctx)

# from functools import partial
# CONVERTER.register_structure_hook_func(
#     lambda t: t is Serializable, partial(_structure_serializable, ctx=CONTEXT_CONVERTER)
#)

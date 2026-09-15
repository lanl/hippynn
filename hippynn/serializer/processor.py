"""
The ContextConverter class: a global cattrs Converter plus tensor/node identity caches.

Implements structuring and unstructuring based on types.py

"""

import dataclasses
import itertools
import warnings
from typing import Any, get_origin

import torch
from cattrs import Converter

from .registry import _ID_REGISTRY, _NODE_REGISTRY
from .types import (
    GraphModuleRecord,
    NodeIdentifier,
    NodeListRecord,
    NodeData,
    NodeReference,
    ScalarPrimitive,
    Serializable,
    NodeRecord,
    SerializationRecordVariant,
    Tagged,
    TaggedIdxType,
    TaggedSet,
    TaggedTuple,
    TensorReference,
)
from ..graphs import GraphModule
from ..graphs.indextypes import IdxType
from ..graphs.nodes.base import Node


CONVERTER = Converter()

# `_type` tag values for the three `SerializationRecordVariant` members, used by
# `structure_with_context` to recognize a record dict without needing its live Python type.
_RECORD_TAGS = {cls.__dataclass_fields__["_type"].default for cls in (NodeRecord, NodeListRecord, GraphModuleRecord)}


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
        record = NodeData(
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

    def unstructure_node(self, node):
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
            record = self.structure(record, NodeData)
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

    def structure_single_node_record(self, data, type_) -> Node:
        (resolved,) = self.structure([data["root"]], list[NodeReference])
        return resolved

    def structure_node_list_record(self, data, type_) -> list:
        return self.structure(data["root_list"], list[NodeReference])

    def structure_graph_module_record(self, data, type_) -> GraphModule:
        input_nodes, output_nodes, forward_output_list = self.structure(
            [data["input_nodes"], data["output_nodes"], data["forward_output_list"]],
            list[tuple[NodeReference, ...]],
        )
        module = GraphModule(input_nodes, output_nodes)
        if tuple(module.forward_output_list) != forward_output_list:
            warnings.warn(
                "Deserialized GraphModule's forward_output_list does not match the recorded one. State dict may be invalid.",
                stacklevel=2,
            )
        module.forward_output_list = forward_output_list
        return module


    def unstructure_with_context(self, obj):
        """
        The single root entry point for unstructuring - never reached recursively (hooks recurse
        via `self.unstructure()` instead), so it's free to recognize `obj`'s top-level shape with
        plain `isinstance` checks, unlike a cattrs hook (which would fire identically on the same
        shape found nested anywhere else, e.g. a `Node` reference or an unrelated plain list).
        """
        self.reset()
        self._unstructuring = True
        try:
            if isinstance(obj, GraphModule):
                input_structure, output_structure, forward_structure = self.unstructure(
                    [list(obj.input_nodes), list(obj.nodes_to_compute), list(obj.forward_output_list)]
                )
                structure = GraphModuleRecord(
                    graph=list(self.node_blob.values()),
                    input_nodes=input_structure,
                    output_nodes=output_structure,
                    forward_output_list=forward_structure,
                )
            elif isinstance(obj, (list, tuple)) and all(isinstance(x, Node) for x in obj):
                root_list = self.unstructure(list(obj))
                structure = NodeListRecord(graph=list(self.node_blob.values()), root_list=root_list)
            else:
                structure = self.unstructure(obj)

            if dataclasses.is_dataclass(structure) and not isinstance(structure, type):
                # `structure` never embeds a live tensor - tensors are only ever referenced
                # indirectly via `TensorReference.key`, resolved through `tensor_blob` below - so
                # flattening it here is cheap and safe. From here on `unstructure_with_context`
                # always returns plain data, uniformly, regardless of `obj`'s shape.
                structure = dataclasses.asdict(structure)

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

    def structure_with_context(self, data, *, tensor_blob, node_blob):
        """
        The single root entry point for structuring - mirrors `unstructure_with_context`'s own
        root-only `isinstance` checks by recognizing `data`'s shape instead of requiring the
        caller to pass a target type explicitly.
        """
        self.reset()
        self.tensor_blob = tensor_blob
        self.node_blob = node_blob
        self._structuring = True
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            # `data` is a live NodeRecord/NodeListRecord/GraphModuleRecord - e.g. unpickled or
            # torch.load()ed (both preserve Python objects); `unstructure_with_context` itself
            # never returns one of these live (it flattens them), so this only happens when
            # reading straight from a pkl/pt file.
            typ = type(data)
            data = dict(vars(data))
        elif isinstance(data, dict) and data.get("_type") in _RECORD_TAGS:
            # json/yml lose Python type info, but the `_type` tag still identifies which record
            # variant this is - cattrs disambiguates the union from it, same as any other tagged
            # union member.
            typ = SerializationRecordVariant
        else:
            typ = Serializable
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

# Register direct hooks (implicit type from return type)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.consume_tensor)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.unstructure_tuple)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.unstructure_set)
CONVERTER.register_unstructure_hook(CONTEXT_CONVERTER.unstructure_idxtype)

# Register indirect hooks (which return different types than they take in)
CONVERTER.register_structure_hook(TensorReference, CONTEXT_CONVERTER.emit_tensor)
CONVERTER.register_structure_hook(NodeReference, CONTEXT_CONVERTER.emit_node)
CONVERTER.register_structure_hook(TaggedIdxType, CONTEXT_CONVERTER.emit_idxtype)
CONVERTER.register_structure_hook(NodeRecord, CONTEXT_CONVERTER.structure_single_node_record)
CONVERTER.register_structure_hook(NodeListRecord, CONTEXT_CONVERTER.structure_node_list_record)
CONVERTER.register_structure_hook(GraphModuleRecord, CONTEXT_CONVERTER.structure_graph_module_record)

# Register generic hooks (many classes)
CONVERTER.register_unstructure_hook_func(
    lambda t: isinstance(t, type) and issubclass(t, Node), CONTEXT_CONVERTER.unstructure_node
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

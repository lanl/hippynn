"""
Auxiliary structure/unstructure hooks: tensor <-> TensorReference, Node <-> NodeReference,
and the generic container-walk hook used on the structure (read) side.
"""

from functools import partial

import torch

from .processor import context_converter
from .registry import _ID_REGISTRY, _NODE_REGISTRY
from .types import NodeIdentifier, NodeInfo, NodeReference, Serializable, TaggedDict, TaggedList, TaggedSet, TaggedTuple, TensorReference
from ..graphs.nodes.base import AutoModule, InputNode, Node
from ..graphs.nodes.base.algebra import BinNode, UnaryNode


# ----------------------------------------------------------------------
# torch.Tensor <-> TensorReference: a side-channel blob (`ctx.tensor_blob`), never part
# of the plaintext structure itself.
# ----------------------------------------------------------------------


def _unstructure_tensor(obj: torch.Tensor, *, ctx):
    key = ctx.tensor_ids.get(id(obj))
    if key is None:
        key = next(ctx._tensor_counter)
        ctx.tensor_ids[id(obj)] = key
        ctx.tensor_blob[key] = obj
    return TensorReference(key=key)


context_converter.converter.register_unstructure_hook(torch.Tensor, partial(_unstructure_tensor, ctx=context_converter))


def _structure_tensor(data: TensorReference, typ, *, ctx):
    key = data.key
    if key not in ctx.tensor_objects:
        ctx.tensor_objects[key] = ctx.tensor_blob[key]
    return ctx.tensor_objects[key]


context_converter.converter.register_structure_hook(torch.Tensor, partial(_structure_tensor, ctx=context_converter))


# ----------------------------------------------------------------------
# Node <-> NodeReference: `ctx.node_blob` *is* SerializedGraph.node_list, assembled directly
# in discovery order as nodes are encountered while unstructuring.
# ----------------------------------------------------------------------


def unstructure_node(obj, *, ctx):
    """
    Generic `unstructure` implementation shared by every direct `_SerializableNode` subclass
    that doesn't need its own (attached onto `InputNode`/`AutoModule` below rather than defined
    on those classes directly, since building a `NodeInfo` needs types from this package, which
    `graphs.nodes.base` cannot import without a circular import). Builds the full `NodeInfo` for
    `obj`, recursing into `.parents` and `.origin_node` (set on `LossInputNode` instances; `None`
    on every other node).
    """
    cls = type(obj)
    key = ctx.get_node_id(obj)
    parents = tuple(ctx.converter.unstructure(p) for p in obj.parents)

    node_kwargs = dict(getattr(obj, "module_kwargs", None) or {})
    if getattr(obj, "origin_node", None) is not None:
        node_kwargs["origin_node"] = obj.origin_node

    # `ModuleKwargs = TaggedDict[Serializable]`, so `node_kwargs` goes through the same generic
    # `dict` hook as any other plain dict nested inside a `Serializable` value.
    module_kwargs = ctx.converter.unstructure(node_kwargs) if node_kwargs else None

    return NodeInfo(
        identifier=NodeIdentifier(id=key, name=obj.name or None),
        cls=_NODE_REGISTRY[cls],
        parents=parents,
        module_kwargs=module_kwargs,
        index_state=getattr(obj, "index_state", None),
        db_name=getattr(obj, "db_name", None),
        requires_grad=getattr(obj, "requires_grad", None),
    )

InputNode.to_record = unstructure_node
AutoModule.to_record = unstructure_node
UnaryNode.to_record = unstructure_node
BinNode.to_record = unstructure_node


def _unstructure_node(obj, *, ctx):
    """
    Node <-> NodeReference cattrs hook: memoized by identity in `ctx.node_ids`/`ctx.node_blob`,
    delegating the actual `NodeInfo` construction to `obj.to_record` where present, falling back
    to the generic `unstructure_node` for `Node` subclasses that aren't `_SerializableNode`s (and
    so don't have `to_record` attached).
    """
    name = obj.name or None
    key = ctx.get_node_id(obj)
    if key is not None:
        return NodeReference(identifier=NodeIdentifier(id=key, name=name))

    key = next(ctx._node_counter)
    ctx.node_ids[id(obj)] = key
    ctx.node_objects[key] = obj  # write-side: lets callers map discovered nodes back to their key/reference

    to_record = getattr(obj, "to_record", None)
    ctx.node_blob[key] = to_record(ctx=ctx) if to_record is not None else unstructure_node(obj, ctx=ctx)
    return NodeReference(identifier=NodeIdentifier(id=key, name=name))


context_converter.converter.register_unstructure_hook_func(
    lambda t: isinstance(t, type) and issubclass(t, Node), partial(_unstructure_node, ctx=context_converter)
)


def _resolve_node(key, ctx):
    """Recursive, memoized reconstruction of the node recorded at `ctx.node_blob[key]`."""
    if key in ctx.node_objects:
        return ctx.node_objects[key]
    if key in ctx._node_resolving:
        raise ValueError(f"Cycle detected while resolving node reference {key} - graphs must be acyclic.")

    ctx._node_resolving.add(key)
    try:
        record = ctx.node_blob[key]
        cls = _ID_REGISTRY[record.cls]
        name = record.identifier.name
        parents = tuple(_resolve_node(p.identifier.id, ctx) for p in record.parents)
        kwargs = _resolve_serializable(record.module_kwargs, ctx) if record.module_kwargs is not None else {}

        node = ctx.converter.structure((name, parents, kwargs, record), cls)

        if isinstance(node, InputNode):
            if record.requires_grad is not None:
                node.requires_grad = record.requires_grad
        else:
            if record.db_name is not None:
                node.db_name = record.db_name
            if record.index_state is not None:
                node.index_state = record.index_state
            if record.requires_grad is not None:
                node.requires_grad = record.requires_grad

        ctx.node_objects[key] = node
        return node
    finally:
        ctx._node_resolving.discard(key)


def _resolve_ids(nested, ctx):
    """
    Recursively resolve a (possibly nested) list/tuple of ids, `NodeReference`s, or `NodeInfo`s
    into the nodes they refer to, preserving the input's list/tuple shape - lets a caller batch
    several groups of ids into one combined structure and resolve them all in a single call,
    mirroring how `serialize()` batches groups into one `unstructure_with_context()` call on the
    write side.
    """
    if isinstance(nested, (list, tuple)):
        resolved = [_resolve_ids(item, ctx) for item in nested]
        return tuple(resolved) if isinstance(nested, tuple) else resolved
    key = nested if isinstance(nested, int) else nested.identifier.id
    return _resolve_node(key, ctx)


def _structure_node_reference(data: NodeReference, typ, *, ctx):
    return _resolve_node(data.identifier.id, ctx)


context_converter.converter.register_structure_hook(NodeReference, partial(_structure_node_reference, ctx=context_converter))


# ----------------------------------------------------------------------
# Per-class node reconstruction (structure/read side) is registered generically by
# `registry.register_structure_hooks`, which finds each direct `_SerializableNode` subclass's
# own `from_record` staticmethod (e.g. `AutoModule.from_record`) and registers it as a
# `ContextConverter.register_structure_family` hook keyed on that class. `LossInputNode`'s own
# `from_record` (base.py) overrides the generic `InputNode.from_record` for its subclasses
# (`LossPredNode`/`LossTrueNode`), since dispatch resolves against the runtime type.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Plain container unstructuring (write side): explicit, rather than relying on whatever
# cattrs' own defaults happen to do for these builtin types. Every one of `list`/`set`/`tuple`/
# `dict` is tagged with a `_type` discriminator uniformly, so the read-side walk below dispatches
# on a single `_type` field rather than on shape - `list`/`tuple` would otherwise collapse to the
# same JSON array shape, and an untagged `dict` could collide with the `_type`-discriminator
# scheme used for `NodeReference`/`TensorReference`/etc.
#
# This hook fires on *any* Python `list`/`dict` passed through `ctx.converter.unstructure()`, not
# just ones nested inside `Serializable`-typed values - callers that unstructure a plain
# list/dict of something else (e.g. a list of `Node`s) must avoid passing the container itself
# through `unstructure()` (see `unstructure_node`'s `module_kwargs` handling above, and
# `serialize()`'s node-reference list-building in `api.py`).
# ----------------------------------------------------------------------


def _unstructure_list(obj, *, ctx):
    return TaggedList(values=[ctx.converter.unstructure(v) for v in obj])


context_converter.converter.register_unstructure_hook(list, partial(_unstructure_list, ctx=context_converter))


def _unstructure_tuple(obj, *, ctx):
    return TaggedTuple(values=tuple(ctx.converter.unstructure(v) for v in obj))


context_converter.converter.register_unstructure_hook(tuple, partial(_unstructure_tuple, ctx=context_converter))


def _unstructure_dict(obj, *, ctx):
    assert all(isinstance(k, str) for k in obj), "All dictionary keys must be strings!"
    return TaggedDict(values={k: ctx.converter.unstructure(v) for k, v in obj.items()})


context_converter.converter.register_unstructure_hook(dict, partial(_unstructure_dict, ctx=context_converter))


def _unstructure_set(obj, *, ctx):
    return TaggedSet(values=[ctx.converter.unstructure(v) for v in obj])


context_converter.converter.register_unstructure_hook(set, partial(_unstructure_set, ctx=context_converter))


# ----------------------------------------------------------------------
# Serializable union discrimination (structure/read side only). structure() needs a concrete
# target type up front, so unlike the write side above (dispatched by cattrs on runtime
# type), this decides which member of the `Serializable` union a piece of wire data represents
# by its shape - each of `TaggedList`/`TaggedSet`/`TaggedTuple`/`TaggedDict` is unwrapped via its
# `.values` field - then delegates to cattrs with a concrete type (`list[Serializable]` or
# `dict[str, Serializable]`), or to the already-registered `TensorReference`/`NodeReference`
# hooks, so cattrs performs the actual recursion rather than this module walking containers by
# hand. Every dict-shaped value - including `module_kwargs` (`ModuleKwargs = TaggedDict[Serializable]`)
# - arrives pre-wrapped as a `TaggedDict`, since `_unstructure_dict` above tags any plain Python
# `dict` on the write side regardless of its static type.
# ----------------------------------------------------------------------


def _resolve_serializable(data, ctx):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    
    if isinstance(data, TensorReference):
        return ctx.converter.structure(data, torch.Tensor)
    if isinstance(data, NodeReference):
        return ctx.converter.structure(data, NodeReference)
    if isinstance(data, TaggedList):
        return list(ctx.converter.structure(data.values, list[Serializable]))
    if isinstance(data, TaggedSet):
        return set(ctx.converter.structure(data.values, list[Serializable]))
    if isinstance(data, TaggedTuple):
        return tuple(ctx.converter.structure(data.values, list[Serializable]))
    if isinstance(data, TaggedDict):
        return dict(ctx.converter.structure(data.values, dict[str, Serializable]))
    raise TypeError(f"Cannot deserialize value of type {type(data)!r}: {data!r}")

def _structure_serializable(data, typ, *, ctx):
    # NOTE: matching against the PEP-695 `Serializable` type-alias object itself (rather than
    # something it resolves/unwraps to, e.g. via `.__value__`) is not yet verified against the
    # pinned cattrs version - flagged in the plan's "verify against actual cattrs version" step.
    # `_resolve_serializable` also relies on this same identity-match for `list[Serializable]`,
    # since cattrs re-enters this hook per element when recursing into it.
    return _resolve_serializable(data, ctx)


context_converter.converter.register_structure_hook_func(lambda t: t is Serializable, partial(_structure_serializable, ctx=context_converter))

"""
Auxiliary structure/unstructure hooks: tensor <-> TensorReference, Node <-> NodeReference,
and the generic container-walk hook used on the structure (read) side.
"""

import torch

from .processor import ContextConverter
from .registry import _ID_REGISTRY, _NODE_REGISTRY, node_reference_attrs
from .types import NodeIdentifier, NodeInfo, NodeReference, Serializable, TaggedSet, TensorReference
from ..graphs.nodes.base import AutoModule, InputNode, Node
from ..graphs.nodes.base.algebra import BinNode, UnaryNode


# ----------------------------------------------------------------------
# torch.Tensor <-> TensorReference: a side-channel blob (`ctx.tensor_blob`), never part
# of the plaintext structure itself.
# ----------------------------------------------------------------------


@ContextConverter.register_unstructure(torch.Tensor)
def _unstructure_tensor(obj: torch.Tensor, *, ctx):
    key = ctx.tensor_ids.get(id(obj))
    if key is None:
        key = next(ctx._tensor_counter)
        ctx.tensor_ids[id(obj)] = key
        ctx.tensor_blob[key] = obj
    return TensorReference(key=key)


@ContextConverter.register_structure(torch.Tensor)
def _structure_tensor(data: TensorReference, typ, *, ctx):
    key = data.key
    if key not in ctx.tensor_objects:
        ctx.tensor_objects[key] = ctx.tensor_blob[key]
    return ctx.tensor_objects[key]


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
    `obj`, recursing into `.parents` and any NODE_REFERENCE_ATTRS-named attributes.
    """
    cls = type(obj)
    key = ctx.node_ids[id(obj)]
    parents = tuple(ctx.converter.unstructure(p) for p in obj.parents)

    serialize_kwargs = getattr(obj, "_serialize_kwargs", None)
    node_kwargs = dict(serialize_kwargs()) if serialize_kwargs is not None else dict(getattr(obj, "module_kwargs", None) or {})

    for attr_name in node_reference_attrs(cls):
        node_kwargs[attr_name] = getattr(obj, attr_name)

    module_kwargs = ctx.converter.unstructure(node_kwargs) if node_kwargs else None

    return NodeInfo(
        identifier=NodeIdentifier(id=key, name=obj.name or None),
        cls=_NODE_REGISTRY[cls],
        input_=isinstance(obj, InputNode),
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


@ContextConverter.register_unstructure_family(lambda t: isinstance(t, type) and issubclass(t, Node))
def _unstructure_node(obj, *, ctx):
    """
    Node <-> NodeReference cattrs hook: memoized by identity in `ctx.node_ids`/`ctx.node_blob`,
    delegating the actual `NodeInfo` construction to `obj.to_record` where present, falling back
    to the generic `unstructure_node` for `Node` subclasses that aren't `_SerializableNode`s (and
    so don't have `to_record` attached).
    """
    key = ctx.node_ids.get(id(obj))
    if key is not None:
        return NodeReference(key=key)

    key = next(ctx._node_counter)
    ctx.node_ids[id(obj)] = key
    ctx.node_objects[key] = obj  # write-side: lets callers map discovered nodes back to their key/reference

    to_record = getattr(obj, "to_record", None)
    ctx.node_blob[key] = to_record(ctx=ctx) if to_record is not None else unstructure_node(obj, ctx=ctx)
    return NodeReference(key=key)


def _resolve_node(key, ctx):
    """Recursive, memoized reconstruction of the node recorded at `ctx.node_blob[key]`."""
    if key in ctx.node_objects:
        return ctx.node_objects[key]
    if key in ctx._node_resolving:
        raise ValueError(f"Cycle detected while resolving node reference {key} - node graphs must be acyclic.")

    ctx._node_resolving.add(key)
    try:
        record = ctx.node_blob[key]
        cls = _ID_REGISTRY[record.cls]
        name = record.identifier.name
        parents = tuple(_resolve_node(p.key, ctx) for p in record.parents)
        kwargs = _resolve_serializable(record.module_kwargs, ctx) if record.module_kwargs is not None else {}

        node = ctx.converter.structure((name, parents, kwargs, record), cls)

        # Mirrors the two branches of the pre-Processor `deserialize_node_list`: input nodes
        # already receive db_name/index_state at construction time, and re-applying db_name
        # post-construction would trigger SingleNode.db_name's setter side effect of lazily
        # creating `.pred`/`.true` (LossPredNode/LossTrueNode).
        if record.input_:
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


@ContextConverter.register_structure(NodeReference)
def _structure_node_reference(data: NodeReference, typ, *, ctx):
    return _resolve_node(data.key, ctx)


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
# cattrs' own defaults happen to do for these builtin types, since `set`/`tuple` need to
# round-trip through the read-side walk below without losing their identity as a Python
# type distinct from `list`.
# ----------------------------------------------------------------------


@ContextConverter.register_unstructure(list)
def _unstructure_list(obj, *, ctx):
    return [ctx.converter.unstructure(v) for v in obj]


@ContextConverter.register_unstructure(tuple)
def _unstructure_tuple(obj, *, ctx):
    return tuple(ctx.converter.unstructure(v) for v in obj)


@ContextConverter.register_unstructure(dict)
def _unstructure_dict(obj, *, ctx):
    assert all(isinstance(k, str) for k in obj), "All dictionary keys must be strings!"
    return {k: ctx.converter.unstructure(v) for k, v in obj.items()}


@ContextConverter.register_unstructure(set)
def _unstructure_set(obj, *, ctx):
    return TaggedSet(values=[ctx.converter.unstructure(v) for v in obj])


# ----------------------------------------------------------------------
# Serializable union discrimination (structure/read side only). structure() needs a concrete
# target type up front, so unlike the write side above (dispatched by cattrs on runtime
# type), this decides which member of the `Serializable` union a piece of wire data
# represents by its shape, then delegates to cattrs with a concrete type - `dict[str,
# Serializable]`, `list[Serializable]`, `tuple[Serializable, ...]`, or the already-registered
# `TensorReference`/`NodeReference` hooks - so cattrs performs the actual recursion rather
# than this module walking containers by hand. `set` is the one case with no JSON-native
# container to delegate to, since it's the only member that genuinely needs tagging.
# ----------------------------------------------------------------------


def _resolve_serializable(data, ctx):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, TensorReference):
        return ctx.converter.structure(data, torch.Tensor)
    if isinstance(data, NodeReference):
        return ctx.converter.structure(data, NodeReference)
    if isinstance(data, TaggedSet):
        return set(ctx.converter.structure(data.values, list[Serializable]))
    if isinstance(data, dict):
        return ctx.converter.structure(data, dict[str, Serializable])
    if isinstance(data, list):
        return ctx.converter.structure(data, list[Serializable])
    if isinstance(data, tuple):
        return ctx.converter.structure(data, tuple[Serializable, ...])
    raise TypeError(f"Cannot deserialize value of type {type(data)!r}: {data!r}")


@ContextConverter.register_structure_family(lambda t: t is object or t is Serializable)
def _structure_serializable(data, typ, *, ctx):
    # NOTE: matching against the PEP-695 `Serializable` type-alias object itself (rather than
    # something it resolves/unwraps to, e.g. via `.__value__`) is not yet verified against the
    # pinned cattrs version - flagged in the plan's "verify against actual cattrs version" step.
    # `_resolve_serializable` also relies on this same identity-match for the generic
    # collection types it delegates to below (`dict[str, Serializable]`, `list[Serializable]`,
    # `tuple[Serializable, ...]`), since cattrs re-enters this hook per element when recursing
    # into those.
    return _resolve_serializable(data, ctx)

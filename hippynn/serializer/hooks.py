"""
Auxiliary structure/unstructure hooks: tensor <-> TensorReference, Node <-> NodeReference,
and the generic container-walk hook used on the structure (read) side.
"""

import torch

from .processor import ContextConverter
from .registry import _ID_REGISTRY, _NODE_REGISTRY, _automodule_kwarg_mapping, node_reference_attrs
from .types import NodeIdentifier, NodeInfo, NodeReference, Serializable
from ..graphs.nodes.base import AutoModule, InputNode, Node


# ----------------------------------------------------------------------
# torch.Tensor <-> TensorReference: a side-channel blob (`ctx.tensor_blob`), never part
# of the plaintext structure itself.
# ----------------------------------------------------------------------


@ContextConverter.register_unstructure(torch.Tensor)
def _unstructure_tensor(obj: torch.Tensor, *, ctx):
    key = ctx.tensor_ids.get(id(obj))
    if key is None:
        key = len(ctx.tensor_blob)
        ctx.tensor_ids[id(obj)] = key
        ctx.tensor_blob[key] = obj
    return {"key": key, "kind": "tensor_ref"}


@ContextConverter.register_structure(torch.Tensor)
def _structure_tensor(data, typ, *, ctx):
    key = data["key"]
    if key not in ctx.tensor_objects:
        ctx.tensor_objects[key] = ctx.tensor_blob[key]
    return ctx.tensor_objects[key]


# ----------------------------------------------------------------------
# Node <-> NodeReference: `ctx.node_blob` *is* SerializedGraph.node_list, assembled directly
# in discovery order as nodes are encountered while unstructuring.
# ----------------------------------------------------------------------


def _node_reference_dict(obj, ctx):
    """
    Ensure `obj` is present in `ctx.node_blob` (recursing into its parents and any
    NODE_REFERENCE_ATTRS-named attributes), returning a `{"kind": "node_ref", ...}` dict
    that stands in for `obj` wherever it is referenced in the unstructured tree.
    """
    key = ctx.node_ids.get(id(obj))
    if key is not None:
        return {"key": key, "kind": "node_ref"}

    key = len(ctx.node_blob)
    ctx.node_ids[id(obj)] = key
    ctx.node_objects[key] = obj  # write-side: lets callers map discovered nodes back to their key/reference

    cls = type(obj)
    parents = tuple(_node_reference_dict(p, ctx) for p in obj.parents)

    if isinstance(obj, InputNode):
        node_kwargs = {}
    else:
        serialize_kwargs = getattr(obj, "_serialize_kwargs", None)
        node_kwargs = dict(serialize_kwargs()) if serialize_kwargs is not None else dict(obj.module_kwargs)

    for attr_name in node_reference_attrs(cls):
        node_kwargs[attr_name] = getattr(obj, attr_name)

    module_kwargs = ctx.converter.unstructure(node_kwargs) if node_kwargs else None

    node_info = NodeInfo(
        identifier=NodeIdentifier(id=key, name=obj.name or None),
        cls=_NODE_REGISTRY[cls],
        input_=isinstance(obj, InputNode),
        parents=parents,
        module_kwargs=module_kwargs,
        index_state=getattr(obj, "index_state", None),
        db_name=getattr(obj, "db_name", None),
        requires_grad=getattr(obj, "requires_grad", None),
    )
    ctx.node_blob[key] = node_info
    return {"key": key, "kind": "node_ref"}


@ContextConverter.register_unstructure_family(lambda t: isinstance(t, type) and issubclass(t, Node))
def _unstructure_node(obj, *, ctx):
    return _node_reference_dict(obj, ctx)


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
        parents = tuple(_resolve_node(p["key"], ctx) for p in record.parents)
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
def _structure_node_reference(data, typ, *, ctx):
    return _resolve_node(data["key"], ctx)


# ----------------------------------------------------------------------
# Per-class node reconstruction (structure/read side), keyed on the concrete Node subclass
# rather than the wire-tagged NodeReference above: `_resolve_node` calls
# `ctx.converter.structure((name, parents, kwargs, record), cls)`, and cattrs dispatches to
# whichever family hook's predicate matches `cls`. Only the InputNode and AutoModule families
# are implemented here; LossInputNode (which also derives from InputNode, so must not be
# matched by the generic InputNode hook - not yet resolved) and the `_deserialize_spec`/plain-
# call branches are left unimplemented for now.
# ----------------------------------------------------------------------


@ContextConverter.register_structure_family(lambda t: isinstance(t, type) and issubclass(t, InputNode))
def _structure_input_node(data, typ, *, ctx):
    name, parents, kwargs, record = data
    return typ(name=name, db_name=record.db_name, index_state=record.index_state)


@ContextConverter.register_structure_family(lambda t: isinstance(t, type) and issubclass(t, AutoModule))
def _structure_automodule_node(data, typ, *, ctx):
    name, parents, kwargs, record = data
    mapping = _automodule_kwarg_mapping(typ)
    node_kwargs = {}
    leftover_module_kwargs = {}
    for k, v in kwargs.items():
        if k in mapping:
            node_kwargs[mapping[k]] = v
        else:
            leftover_module_kwargs[k] = v
    return typ(name, parents, module_kwargs=leftover_module_kwargs, **node_kwargs)


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
    return {"kind": "set", "values": [ctx.converter.unstructure(v) for v in obj]}


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
    if isinstance(data, dict):
        kind = data.get("kind")
        if kind == "tensor_ref":
            return ctx.converter.structure(data, torch.Tensor)
        if kind == "node_ref":
            return ctx.converter.structure(data, NodeReference)
        if kind == "set":
            return set(ctx.converter.structure(data["values"], list[Serializable]))
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

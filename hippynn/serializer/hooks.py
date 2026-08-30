"""
Auxiliary structure/unstructure hooks: tensor <-> TensorReference, Node <-> NodeReference,
and the generic container-walk hook used on the structure (read) side.
"""

from functools import partial

import torch

from .processor import CONTEXT_CONVERTER
from .registry import _ID_REGISTRY, _NODE_REGISTRY
from .types import NodeIdentifier, NodeInfo, NodeRecord, NodeReference, Serializable, TaggedDict, TaggedList, TaggedSet, TaggedTuple, TensorReference
from ..graphs.nodes.base import InputNode, Node


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


CONTEXT_CONVERTER.converter.register_unstructure_hook(torch.Tensor, partial(_unstructure_tensor, ctx=CONTEXT_CONVERTER))


def _structure_tensor(data: TensorReference, typ, *, ctx):
    key = data.key
    if key not in ctx.tensor_objects:
        ctx.tensor_objects[key] = ctx.tensor_blob[key]
    return ctx.tensor_objects[key]


CONTEXT_CONVERTER.converter.register_structure_hook(torch.Tensor, partial(_structure_tensor, ctx=CONTEXT_CONVERTER))


# ----------------------------------------------------------------------
# Node <-> NodeReference: `ctx.node_blob` *is* SerializedGraph.node_list, assembled directly
# in discovery order as nodes are encountered while unstructuring.
# ----------------------------------------------------------------------


def _unstructure_node(obj, *, ctx):
    """
    Node <-> NodeReference cattrs hook: memoized by identity in `ctx.node_ids`/`ctx.node_blob`.

    `obj.to_record()` is a plain, `ctx`-free instance method (defined on `_SerializableNode` and
    overridden/chained via `super()` by subclasses) that builds a domain-level `NodeInfo`: live
    parent `Node`s, and any extra per-class data nested under a named key in `extra_state`. This
    hook's only job is to resolve that domain-level `NodeInfo` into the wire-level `NodeRecord`,
    recursing into parents/state_dict/extra_state via `ctx.unstructure`.
    """
    name = obj.name or None
    key = ctx.get_node_id(obj)
    if key is not None:
        return NodeReference(identifier=NodeIdentifier(id=key, name=name))

    key = next(ctx._node_counter)
    ctx.node_ids[id(obj)] = key
    ctx.node_objects[key] = obj  # write-side: lets callers map discovered nodes back to their key/reference

    info = obj.to_record()

    ctx.node_blob[key] = NodeRecord(
        identifier=NodeIdentifier(id=key, name=name),
        cls=_NODE_REGISTRY[type(obj)],
        info=NodeInfo(
            parents=tuple(ctx.unstructure(p) for p in info.parents),
            index_state=info.index_state,
            db_name=info.db_name,
            requires_grad=info.requires_grad,
            state_dict=ctx.unstructure(info.state_dict) if info.state_dict else None,
            extra_state={k: ctx.unstructure(v) for k, v in info.extra_state.items()} if info.extra_state else None,
        ),
    )
    return NodeReference(identifier=NodeIdentifier(id=key, name=name))


CONTEXT_CONVERTER.converter.register_unstructure_hook_func(
    lambda t: isinstance(t, type) and issubclass(t, Node), partial(_unstructure_node, ctx=CONTEXT_CONVERTER)
)

# todo: rename to _structure_node to match _unstructure_node and nomeclature of cattrs?
def _resolve_node(key, ctx):
    """
    Recursive, memoized reconstruction of the node recorded at `ctx.node_blob[key]`.

    Resolves the stored wire-level `NodeRecord.info` into a domain-level `NodeInfo` (parents
    resolved to live `Node`s, `state_dict`/`extra_state` resolved to live values), then calls
    `cls.from_record(name, info, cls, ctx=ctx)` directly - a plain staticmethod lookup that
    follows `cls`'s own MRO, rather than cattrs' `ctx.structure(...)` dispatch.
    """
    if key in ctx.node_objects:
        return ctx.node_objects[key]
    if key in ctx._node_resolving:
        raise ValueError(f"Cycle detected while resolving node reference {key} - graphs must be acyclic.")

    ctx._node_resolving.add(key)
    try:
        record = ctx.node_blob[key]
        cls = _ID_REGISTRY[record.cls]
        name = record.identifier.name
        info = record.info

        resolved_info = NodeInfo(
            parents=tuple(ctx.structure(p, Serializable) for p in info.parents),
            index_state=info.index_state,
            db_name=info.db_name,
            requires_grad=info.requires_grad,
            state_dict=ctx.structure(info.state_dict, Serializable) if info.state_dict is not None else None,
            extra_state={k: ctx.structure(v, Serializable) for k, v in info.extra_state.items()} if info.extra_state is not None else None,
        )

        node = cls.from_record(name, resolved_info, cls, ctx=ctx)

        if isinstance(node, InputNode):
            if resolved_info.requires_grad is not None:
                node.requires_grad = resolved_info.requires_grad
        else:
            if resolved_info.db_name is not None:
                node.db_name = resolved_info.db_name
            if resolved_info.index_state is not None:
                node.index_state = resolved_info.index_state
            if resolved_info.requires_grad is not None:
                node.requires_grad = resolved_info.requires_grad

        if resolved_info.state_dict is not None:
            node.torch_module.load_state_dict(resolved_info.state_dict)

        ctx.node_objects[key] = node
        return node
    finally:
        ctx._node_resolving.discard(key)


def _structure_node_reference(data: NodeReference, typ, *, ctx):
    return _resolve_node(data.identifier.id, ctx)


CONTEXT_CONVERTER.converter.register_structure_hook(NodeReference, partial(_structure_node_reference, ctx=CONTEXT_CONVERTER))


# Plain wrapper types

def _unstructure_list(obj, *, ctx):
    return TaggedList(values=[ctx.unstructure(v) for v in obj])


CONTEXT_CONVERTER.converter.register_unstructure_hook(list, partial(_unstructure_list, ctx=CONTEXT_CONVERTER))


def _unstructure_tuple(obj, *, ctx):
    return TaggedTuple(values=tuple(ctx.unstructure(v) for v in obj))


CONTEXT_CONVERTER.converter.register_unstructure_hook(tuple, partial(_unstructure_tuple, ctx=CONTEXT_CONVERTER))


def _unstructure_dict(obj, *, ctx):
    assert all(isinstance(k, str) for k in obj), "All dictionary keys must be strings!"
    return TaggedDict(values={k: ctx.unstructure(v) for k, v in obj.items()})


CONTEXT_CONVERTER.converter.register_unstructure_hook(dict, partial(_unstructure_dict, ctx=CONTEXT_CONVERTER))


def _unstructure_set(obj, *, ctx):
    return TaggedSet(values=[ctx.unstructure(v) for v in obj])


CONTEXT_CONVERTER.converter.register_unstructure_hook(set, partial(_unstructure_set, ctx=CONTEXT_CONVERTER))



def _resolve_serializable(data, ctx):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    
    if isinstance(data, TensorReference):
        return ctx.structure(data, torch.Tensor)
    if isinstance(data, NodeReference):
        return ctx.structure(data, NodeReference)
    if isinstance(data, TaggedList):
        return list(ctx.structure(data.values, list[Serializable]))
    if isinstance(data, TaggedSet):
        return set(ctx.structure(data.values, list[Serializable]))
    if isinstance(data, TaggedTuple):
        return tuple(ctx.structure(data.values, list[Serializable]))
    if isinstance(data, TaggedDict):
        return dict(ctx.structure(data.values, dict[str, Serializable]))
    raise TypeError(f"Cannot deserialize value of type {type(data)!r}: {data!r}")

# todo merge with above function, don't seem to have need for two functions??
def _structure_serializable(data, typ, *, ctx):
    # todo address note... what does this even mean
    # NOTE: matching against the PEP-695 `Serializable` type-alias object itself (rather than
    # something it resolves/unwraps to, e.g. via `.__value__`) is not yet verified against the
    # pinned cattrs version - flagged in the plan's "verify against actual cattrs version" step.
    # `_resolve_serializable` also relies on this same identity-match for `list[Serializable]`,
    # since cattrs re-enters this hook per element when recursing into it.
    return _resolve_serializable(data, ctx)


CONTEXT_CONVERTER.converter.register_structure_hook_func(
    lambda t: t is Serializable, partial(_structure_serializable, ctx=CONTEXT_CONVERTER)
)

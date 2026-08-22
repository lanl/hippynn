from __future__ import annotations
from typing import NamedTuple, Optional, TypeAlias, TypeVar, Union, Literal, Generic
import importlib
import warnings
import torch

from .. import _debprint
from ...graphs import GraphModule, compute_evaluation_order
from ...graphs.indextypes import IdxType
from ...graphs.nodes.base import Node, InputNode
from ...graphs.nodes.base.base import _SerializableNodeMixin
from ...graphs.nodes.base import get_connected_nodes, find_relatives


_NODE_REGISTRY = {}
_ID_REGISTRY = {}


def _populate_registry():
    for cls in _SerializableNodeMixin._unregistered_classes.copy():
        register_node_cls(cls)
        _SerializableNodeMixin._unregistered_classes.remove(cls)
    pass


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


# import attrs, cattrs

T = TypeVar("T")


class NodeTypeIdentifier(NamedTuple):
    name: str
    namespace: str
    package: Optional[str]


class NodeIdentifier(NamedTuple):
    id: int  # Comes from a valid toposort of the graph.
    # not persistent
    name: Optional[str]


class TensorReference(NamedTuple):
    key: int
    kind: Literal["tensor_ref"] = "tensor_ref"


class TaggedSet(NamedTuple, Generic[T]):
    values: list[T]
    kind: Literal["set"] = "set"


type ScalarPrimitive = None | bool | int | float | str
type Scalar = ScalarPrimitive | TensorReference
type ContainerPrimitive[T] = list[T] | tuple[T, ...] | dict[str, T] | TaggedSet[T]
type Serializable = Scalar | ContainerPrimitive[Serializable]
type StrictSerializable = ScalarPrimitive | ContainerPrimitive[StrictSerializable]
type ModuleKwargs = dict[str, Serializable]


class NodeInfo(NamedTuple):
    identifier: NodeIdentifier
    cls: NodeTypeIdentifier
    input_: bool
    parents: tuple[NodeIdentifier, ...] = ()
    module_kwargs: Optional[ModuleKwargs] = None
    index_state: Optional[IdxType] = None
    db_name: Optional[str] = None
    requires_grad: Optional[bool] = None  # for input nodes


class SerializedGraph(NamedTuple):
    # graph itself
    node_list: list[NodeInfo]
    # list of external tensor names required in the node info
    n_tensor_arguments: int
    metadata: Optional[dict[str, StrictSerializable]] = None
    # additional info


class SerializedGraphModule(NamedTuple):
    input_nodes: tuple[NodeIdentifier, ...]  # `required_inputs`
    output_nodes: tuple[NodeIdentifier, ...]  # `nodes_to_compute`
    forward_output_list: tuple[NodeIdentifier, ...]


class SerializationRecord(NamedTuple):
    graph: SerializedGraph
    module: Optional[SerializedGraphModule] = None
    tensor_data: Optional[dict[int, torch.Tensor]] = None
    state_dict: Optional[dict[str, torch.Tensor]] = None


from typing import Iterable


def serialize(nodes_or_graph: Union[Node, Iterable[Node], GraphModule], descendants=True):
    """
    Interface wrapper taking node(s), graph.
    """
    module = state_dict = None
    if isinstance(nodes_or_graph, GraphModule):
        nodes = nodes_or_graph.nodes_to_compute
        module = nodes_or_graph
    elif isinstance(nodes_or_graph, Node):
        nodes = [nodes_or_graph]
    else:
        # list or other container.
        nodes = nodes_or_graph

    nodes = get_connected_nodes(nodes, descendants=descendants, ancestors=True)
    nodes, _ = compute_evaluation_order(nodes)
    input_nodes = find_relatives(nodes, InputNode, ancestors=True, descendants=False)
    nodes = list(input_nodes) + list(nodes)

    serial_data, tensor_data, identifier_map = serialize_node_list(nodes)

    if module is not None:

        def map_tuple(attr):
            return tuple(identifier_map[node] for node in attr)

        input_nodes = map_tuple(module.input_nodes)
        output_nodes = map_tuple(module.nodes_to_compute)
        forward_output_list = map_tuple(module.forward_output_list)
        state_dict = module.state_dict()
        module = SerializedGraphModule(input_nodes=input_nodes, output_nodes=output_nodes, forward_output_list=forward_output_list)

    record = SerializationRecord(graph=serial_data, module=module)

    return record, tensor_data  # , state_dict


import itertools


def deserialize(
    record: SerializationRecord, tensor_data: Optional[dict[int, torch.Tensor]] = None, state_dict: Optional[dict[str, torch.Tensor]] = None
):
    """
    Interface wrapper taking a SerializationRecord (and its associated tensor data)
    and reconstructing either a bare list of nodes or a GraphModule, mirroring `serialize`.
    """
    if tensor_data is None:
        tensor_data = record.tensor_data or {}

    nodes, identifier_map = deserialize_node_list(record.graph.node_list, tensor_data=tensor_data)

    if record.module is None:
        return nodes

    def map_tuple(attr):
        return tuple(identifier_map[ident] for ident in attr)

    input_nodes = map_tuple(record.module.input_nodes)
    output_nodes = map_tuple(record.module.output_nodes)

    module = GraphModule(input_nodes, output_nodes)

    if state_dict is None:
        state_dict = record.state_dict
    if state_dict is not None:
        module.load_state_dict(state_dict)

    return module


def serialize_node_list(node_list):
    _debprint("populating registry")
    _populate_registry()

    node_identity_map: dict[Node, NodeIdentifier] = {}
    all_node_info: list = []
    tensor_counter = itertools.count()
    tensor_data = {}

    for i, node in enumerate(node_list):
        _debprint(f"Serializing {node=}")
        name = node.name or None
        ident = NodeIdentifier(id=i, name=name)
        node_identity_map[node] = ident

        cls = _NODE_REGISTRY[type(node)]

        db_name = node.db_name if hasattr(node, "db_name") else None
        index_state = node.index_state if hasattr(node, "index_state") else None
        requires_grad = node.requires_grad if hasattr(node, "requires_grad") else None

        if isinstance(node, InputNode):
            node_info = NodeInfo(
                identifier=ident, cls=cls, input_=True, parents=(), index_state=index_state, db_name=db_name, requires_grad=requires_grad
            )
            all_node_info.append(node_info)
            continue

        try:
            # If below line fails, the order was not a valid toposort, by definition
            parent_idents = tuple(node_identity_map[p] for p in node.parents)
        except KeyError as ee:
            raise ValueError(f"Invalid toposort for serializing nodes.") from ee


        node_kwargs = node.module_kwargs
        # If name translation was applied, translate back here so the node reconstructs from the kwargs.
        if (auto_module_kwargs := getattr(node, "auto_module_kwargs", None)) and isinstance(auto_module_kwargs, dict):
            node_kwargs = {auto_module_kwargs.get(k, k): v for k, v in node_kwargs.items()}

        module_kwargs, tensor_data = lower_serializable(node_kwargs, tensor_data=tensor_data, counter=tensor_counter)

        node_info = NodeInfo(
            identifier=ident,
            cls=cls,
            input_=False,
            parents=parent_idents,
            module_kwargs=module_kwargs,
            index_state=index_state,
            db_name=db_name,
            requires_grad=requires_grad,
        )
        all_node_info.append(node_info)

    n_tensors = next(tensor_counter)

    serial_data = SerializedGraph(
        node_list=all_node_info,
        n_tensor_arguments=n_tensors,
    )
    return serial_data, tensor_data, node_identity_map


def deserialize_node_list(node_list: list[NodeInfo], tensor_data: Optional[dict[int, torch.Tensor]] = None):
    _debprint("populating registry")
    _populate_registry()

    if tensor_data is None:
        tensor_data = {}

    node_identity_map: dict[NodeIdentifier, Node] = {}

    for node_info in node_list:
        _debprint(f"Deserializing {node_info=}")
        cls = _ID_REGISTRY[node_info.cls]
        name = node_info.identifier.name

        if node_info.input_:
            node = cls(name=name, db_name=node_info.db_name, index_state=node_info.index_state)
            node_identity_map[node_info.identifier] = node
            if node_info.requires_grad is not None:
                node.requires_grad = node_info.requires_grad
            continue

        try:
            # If below line fails, the order was not a valid toposort, by definition
            parents = tuple(node_identity_map[p] for p in node_info.parents)
        except KeyError as ee:
            raise ValueError(f"Invalid toposort for deserializing nodes.") from ee

        module_kwargs = raise_serializable(node_info.module_kwargs, tensor_data=tensor_data)

        node = cls(name, parents, **module_kwargs)

        if node_info.db_name is not None:
            node.db_name = node_info.db_name
        if node_info.index_state is not None:
            node.index_state = node_info.index_state
        if node_info.requires_grad is not None:
            node.requires_grad = node_info.requires_grad

        node_identity_map[node_info.identifier] = node

    return list(node_identity_map.values()), node_identity_map


def lower_serializable(obj: object, tensor_data: dict[int, torch.Tensor] = None, counter=None) -> Serializable:
    if tensor_data is None:
        tensor_data = {}
    if counter is None:
        counter = itertools.count()

    if (obj is None) or isinstance(obj, (bool, str, int, float)):
        return obj, tensor_data

    elif isinstance(obj, torch.Tensor):
        ref = TensorReference(key=next(counter))
        tensor_data[ref.key] = obj
        return ref, tensor_data

    elif isinstance(obj, list):
        out_obj = list()
        for o in obj:
            o, tensor_data = lower_serializable(o, tensor_data=tensor_data, counter=counter)
            out_obj.append(o)
        return out_obj, tensor_data

    elif isinstance(obj, set):
        obj, tensor_data = lower_serializable(list(obj), tensor_data=tensor_data, counter=counter)
        return TaggedSet(values=obj), tensor_data

    elif isinstance(obj, tuple):
        obj, tensor_data = lower_serializable(list(obj), tensor_data=tensor_data, counter=counter)
        return tuple(obj), tensor_data

    elif isinstance(obj, dict):
        keys = list(obj.keys())
        values = list(obj.values())

        assert all(isinstance(k, str) for k in keys), "All dictionary keys must be strings!"

        values, tensor_data = lower_serializable(values, tensor_data=tensor_data, counter=counter)
        return dict(zip(keys, values)), tensor_data

    raise ValueError(f"Unrecognized type for serializing! {type(obj)=} {obj=}")



def raise_serializable(obj: Serializable, tensor_data: dict[int, torch.Tensor]) -> object:
    if (obj is None) or isinstance(obj, (bool, str, int, float)):
        return obj

    elif isinstance(obj, TensorReference):
        return tensor_data[obj.key]

    elif isinstance(obj, TaggedSet):
        return set(raise_serializable(o, tensor_data) for o in obj.values)

    elif isinstance(obj, list):
        return [raise_serializable(o, tensor_data) for o in obj]

    elif isinstance(obj, tuple):
        return tuple(raise_serializable(o, tensor_data) for o in obj)

    elif isinstance(obj, dict):
        return {k: raise_serializable(v, tensor_data) for k, v in obj.items()}

    raise ValueError(f"Unrecognized type for deserializing! {type(obj)=} {obj=}")


ensure_registry_namespaces(raise_=False)


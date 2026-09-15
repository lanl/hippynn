"""
Plain-data type definitions for graph serialization: node/tensor references and record shapes.
"""

from dataclasses import dataclass
from typing import Literal

PROTOCOL_VERSION = 0


@dataclass(frozen=True)
class NodeTypeIdentifier:
    name: str
    namespace: str
    package: str | None


@dataclass(frozen=True)
class NodeIdentifier:
    id: int  # assigned in discovery order; not persistent
    name: str | None


@dataclass(frozen=True)
class NodeReference:
    identifier: NodeIdentifier
    _type: Literal["node_ref"] = "node_ref"


@dataclass(frozen=True)
class TensorReference:
    key: int
    _type: Literal["tensor_ref"] = "tensor_ref"


@dataclass(frozen=True)
class TaggedSet[T]:
    values: list[T]
    _type: Literal["set"] = "set"


@dataclass(frozen=True)
class TaggedTuple[T]:
    values: list[T]
    _type: Literal["tuple"] = "tuple"


@dataclass(frozen=True)
class TaggedIdxType:
    values: str
    _type: Literal["IdxType"] = "IdxType"


type ScalarPrimitive = None | bool | int | float | str
type Scalar = ScalarPrimitive | TensorReference | NodeReference | TaggedIdxType
type Container[T] = list[T] | TaggedTuple[T] | dict[str, T] | TaggedSet[T]
type ContainerPrimitive = Container[Serializable]
type Serializable = Scalar | ContainerPrimitive

# all tagged types for serialization
type Tagged = NodeReference | TensorReference | TaggedSet[Serializable] | TaggedTuple[Serializable] | TaggedIdxType


@dataclass
class NodeRecord:
    identifier: NodeIdentifier  # holds name
    cls: NodeTypeIdentifier
    parents: tuple[NodeReference, ...] = ()
    state: dict[str, Serializable] | None = None  # class-specific
    state_dict: dict[str, Serializable] | None = None

@dataclass
class SerializedGraph:
    node_list: list[NodeRecord]
    n_tensor_values: int
    protocol_version: int = PROTOCOL_VERSION
    tensor_data: dict[int, object] | None = None


#TODO replace
@dataclass
class SerializedGraphModule:
    input_nodes: tuple[NodeReference, ...]  # `required_inputs`
    output_nodes: tuple[NodeReference, ...]  # `nodes_to_compute`
    forward_output_list: tuple[NodeReference, ...]

#TODO: replace
@dataclass
class SerializationRecord:
    graph: SerializedGraph
    protocol_version: int = PROTOCOL_VERSION
    module: SerializedGraphModule | None = None
    tensor_data: dict[int, object] | None = None
    root_ids: list[NodeReference] | None = None
    single_root: bool = False


# Top-level record structures
@dataclass
class NodeRecord:  # bare list of `Node`s, not wrapped in a `GraphModule`
    nodes: SerializedGraph
    root: NodeReference
    _type: Literal["node_list_record"] = "node_record"

@dataclass
class NodeListRecord:  # bare list of `Node`s, not wrapped in a `GraphModule`
    graph: SerializedGraph
    root_list: list[NodeReference]
    single_root: bool = False
    _type: Literal["node_list_record"] = "node_list_record"


@dataclass
class GraphModuleRecord:  # a `GraphModule`
    graph: SerializedGraph
    input_nodes: tuple[NodeReference, ...]  # `required_inputs`
    output_nodes: tuple[NodeReference, ...]  # `nodes_to_compute`
    forward_output_list: tuple[NodeReference, ...]
    _type: Literal["graph_module_record"] = "graph_module_record"


type SerializationRecordVariant = NodeRecord | NodeListRecord | GraphModuleRecord

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
class TaggedList[T]:
    values: list[T]
    _type: Literal["list"] = "list"


@dataclass(frozen=True)
class TaggedSet[T]:
    values: list[T]
    _type: Literal["set"] = "set"


@dataclass(frozen=True)
class TaggedTuple[T]:
    values: list[T, ...]
    _type: Literal["tuple"] = "tuple"


@dataclass(frozen=True)
class TaggedDict[T]:
    values: dict[str, T]
    _type: Literal["dict"] = "dict"

@dataclass(frozen=True)
class TaggedIdxType:
    values: str
    _type: Literal["IdxType"] = "IdxType"


type ScalarPrimitive = None | bool | int | float | str
type Scalar = ScalarPrimitive | TensorReference | NodeReference | TaggedIdxType
#type Container[T] = TaggedList[T] | TaggedTuple[T] | TaggedDict[T] | TaggedSet[T]
type Container[T] = list[T] | TaggedTuple[T] | dict[str,T] | TaggedSet[T]
type ContainerPrimitive = Container[Serializable]
type Serializable = Scalar | ContainerPrimitive
type ModuleKwargs = TaggedDict[Serializable]

# all tagged types for seralization
# type Tagged = NodeReference | TensorReference | TaggedList[Serializable] | TaggedSet[Serializable] | TaggedTuple[Serializable] | TaggedDict[
#     Serializable]

type Tagged = NodeReference | TensorReference | TaggedSet[Serializable] | TaggedTuple[Serializable] | TaggedIdxType
# @dataclass
# class NodeInfo:
#     pass
    
@dataclass
class NodeRecord:
    identifier: NodeIdentifier # holds name
    cls: NodeTypeIdentifier
    parents: tuple[NodeReference, ...] = ()
    index_state: str | None = None
    state: dict[str, Serializable] | None = None # class-specific
    state_dict: dict[str,Serializable] | None = None

@dataclass
class SerializedGraph:
    node_list: list[NodeRecord]
    n_tensor_values: int

@dataclass
class SerializedGraphModule:
    input_nodes: tuple[NodeReference, ...]  # `required_inputs`
    output_nodes: tuple[NodeReference, ...]  # `nodes_to_compute`
    forward_output_list: tuple[NodeReference, ...]

type NodeBlob = dict[int, NodeRecord]

@dataclass
class SerializationRecord:
    graph: SerializedGraph
    protocol_version: int = PROTOCOL_VERSION
    module: SerializedGraphModule | None = None
    tensor_data: dict[int, object] | None = None
    root_ids: list[NodeReference] | None = None
    single_root: bool = False

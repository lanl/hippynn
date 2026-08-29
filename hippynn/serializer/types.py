"""
Plain-data type definitions for graph serialization: node/tensor references and record shapes.
"""

import dataclasses
from typing import Literal

from ..graphs.indextypes import IdxType

PROTOCOL_VERSION = 0

@dataclasses.dataclass(frozen=True)
class NodeTypeIdentifier:
    name: str
    namespace: str
    package: str | None


@dataclasses.dataclass(frozen=True)
class NodeIdentifier:
    id: int  # assigned in discovery order; not persistent
    name: str | None


@dataclasses.dataclass(frozen=True)
class NodeReference:
    identifier: NodeIdentifier
    _type: Literal["node_ref"] = "node_ref"


@dataclasses.dataclass(frozen=True)
class TensorReference:
    key: int
    _type: Literal["tensor_ref"] = "tensor_ref"


@dataclasses.dataclass(frozen=True)
class TaggedList[T]:
    values: list[T]
    _type: Literal["list"] = "list"


@dataclasses.dataclass(frozen=True)
class TaggedSet[T]:
    values: list[T]
    _type: Literal["set"] = "set"


@dataclasses.dataclass(frozen=True)
class TaggedTuple[T]:
    values: tuple[T, ...]
    _type: Literal["tuple"] = "tuple"


@dataclasses.dataclass(frozen=True)
class TaggedDict[T]:
    values: dict[str, T]
    _type: Literal["dict"] = "dict"


type ScalarPrimitive = None | bool | int | float | str
type Scalar = ScalarPrimitive | TensorReference | NodeReference
type Container[T] = TaggedList[T] | TaggedTuple[T] | TaggedDict[T] | TaggedSet[T]
type ContainerPrimitive = Container[Serializable]
type Serializable = Scalar | ContainerPrimitive
type ModuleKwargs = TaggedDict[Serializable]

# all tagged types for seralization
type Tagged = NodeReference | TensorReference | TaggedList[Serializable] | TaggedSet[Serializable] | TaggedTuple[Serializable] | TaggedDict[Serializable]


@dataclasses.dataclass
class NodeInfo:
    identifier: NodeIdentifier
    cls: NodeTypeIdentifier
    parents: tuple[NodeReference, ...] = ()
    module_kwargs: ModuleKwargs | None = None
    index_state: IdxType | None = None
    db_name: str | None = None
    requires_grad: bool | None = None  # for input nodes


@dataclasses.dataclass
class SerializedGraph:
    node_list: list[NodeInfo]
    n_tensor_values: int


@dataclasses.dataclass
class SerializedGraphModule:
    input_nodes: tuple[NodeReference, ...]  # `required_inputs`
    output_nodes: tuple[NodeReference, ...]  # `nodes_to_compute`
    forward_output_list: tuple[NodeReference, ...]


@dataclasses.dataclass
class SerializationRecord:
    graph: SerializedGraph
    protocol_version: int = PROTOCOL_VERSION
    module: SerializedGraphModule | None = None
    tensor_data: dict[int, object] | None = None
    state_dict: dict[str, object] | None = None
    root_ids: list[int] | None = None
    single_root: bool = False

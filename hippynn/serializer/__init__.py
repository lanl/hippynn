"""
Serialization of hippynn Node graphs to/from plain data.

`types.py` defines the plain-data record shapes, `registry.py` the node-class registry and
per-class reconstruction dispatch, `processor.py` the cattrs-driven `Processor`, and
`hooks.py` the tensor/node hooks `Processor` uses. This module re-exports the public surface
and provides the `serialize`/`deserialize` entry points, thin wrappers around
`Processor.unstructure`/`structure`.
"""

import warnings
from typing import Iterable

from ..experiment import _debprint
from ..graphs import GraphModule
from ..graphs.nodes.base import Node, get_connected_nodes
from .hooks import _resolve_node
from .processor import ContextConverter
from .registry import ensure_registry_consistency, register_node_cls, register_structure_hooks
from .types import (
    NodeIdentifier,
    NodeInfo,
    NodeReference,
    NodeTypeIdentifier,
    SerializationRecord,
    SerializedGraph,
    SerializedGraphModule,
    TensorReference,
)

#todo if possible: rewrite to simply handle all of these cases using cattrs?
def serialize(nodes_or_graph: Node | Iterable[Node] | GraphModule, descendants=True):
    """
    Interface wrapper taking node(s) or a graph and serializing them to a `SerializationRecord`.
    """
    module = original_roots = None
    if isinstance(nodes_or_graph, GraphModule):
        module = nodes_or_graph
        # Set of nodes for serializing in a graph
        expanded = list(module.input_nodes) + list(module.forward_output_list)
    else:
        
        if isinstance(nodes_or_graph, Node):
            original_roots = [nodes_or_graph]
        else:
            original_roots = list(nodes_or_graph)

        # Parents will be found recursively as dependencies.
        expanded = get_connected_nodes(original_roots, descendants=descendants, ancestors=False)

    serial_data, tensor_data, identifier_map = serialize_node_list(expanded)

    graph_module_record = state_dict = root_ids = None
    if module is not None:

        def map_tuple(attr):
            return tuple(identifier_map[node] for node in attr)

        graph_module_record = SerializedGraphModule(
            input_nodes=map_tuple(module.input_nodes),
            output_nodes=map_tuple(module.nodes_to_compute),
            forward_output_list=map_tuple(module.forward_output_list),
        )
        state_dict = module.state_dict()
    else:
        ids = [identifier_map[n].key for n in original_roots]
        root_ids = ids[0] if isinstance(nodes_or_graph, Node) else ids

    record = SerializationRecord(graph=serial_data, module=graph_module_record, state_dict=state_dict, root_ids=root_ids)

    return record, tensor_data


def deserialize(record: SerializationRecord, tensor_data: dict[int, object] | None = None, state_dict: dict[str, object] | None = None):
    """
    Interface wrapper taking a `SerializationRecord` (and its associated tensor data) and
    reconstructing either a bare list of nodes or a `GraphModule`, mirroring `serialize`.
    """
    ### see comments in serialize about how to change this
    if tensor_data is None:
        tensor_data = record.tensor_data or {}

    nodes, identifier_map = deserialize_node_list(record.graph.node_list, tensor_data=tensor_data)
    by_id = {ident.id: node for ident, node in identifier_map.items()}

    if record.module is None:
        if record.root_ids is None:
            return nodes
        if isinstance(record.root_ids, list):
            return [by_id[key] for key in record.root_ids]
        return by_id[record.root_ids]

    def map_tuple(attr):
        return tuple(by_id[ref.key] for ref in attr)

    input_nodes = map_tuple(record.module.input_nodes)
    output_nodes = map_tuple(record.module.output_nodes)

    module = GraphModule(input_nodes, output_nodes)

    if len(module.forward_output_list) != len(record.module.forward_output_list):
        warnings.warn(
            f"Deserialized GraphModule has {len(module.forward_output_list)} forward_output_list "
            f"nodes, but the record was serialized with {len(record.module.forward_output_list)} - "
            "the reconstructed graph structure does not match the original, and loading its "
            "state_dict is likely to fail.",
            stacklevel=2,
        )

    if state_dict is None:
        state_dict = record.state_dict
    if state_dict is not None:
        module.load_state_dict(state_dict)

    return module


#todo: likely can remove the need for this function
def serialize_node_list(node_list: Iterable[Node]):
    """
    Serialize `node_list` - plus everything reachable from it via `.parents` - into a
    `SerializedGraph`, a tensor blob, and a `{Node: NodeReference}` map covering every node
    discovered, not only those originally passed in.
    """
    _debprint(f"Serializing node list: {node_list=}")
    processor = ContextConverter()
    processor.unstructure(list(node_list))

    identifier_map = {processor.node_objects[key]: NodeReference(key=key) for key in processor.node_blob}
    serial_data = SerializedGraph(node_list=list(processor.node_blob.values()), n_tensor_arguments=len(processor.tensor_blob))
    return serial_data, processor.tensor_blob, identifier_map

#todo: likely can remove the need for this function
def deserialize_node_list(node_list: list[NodeInfo], tensor_data: dict[int, object] | None = None):
    """
    Reconstruct every node in `node_list`, returning `(nodes, {NodeIdentifier: Node})`.
    """
    _debprint(f"Deserializing node list: {node_list=}")
    processor = ContextConverter()
    if tensor_data is not None:
        processor.tensor_blob = tensor_data
    for info in node_list:
        processor.node_blob[info.identifier.id] = info

    identifier_map = {info.identifier: _resolve_node(info.identifier.id, processor) for info in node_list}
    return list(identifier_map.values()), identifier_map


ensure_registry_consistency(raise_=False)
register_structure_hooks()


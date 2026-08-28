"""
Public serialize/deserialize entry points, plus text-based codec interfaces.
"""

import warnings
from typing import Iterable

from ..experiment import _debprint
from ..graphs import GraphModule
from ..graphs.nodes.base import Node, get_connected_nodes
from .hooks import _resolve_node
from .processor import ContextConverter
from .types import PROTOCOL_VERSION, SerializationRecord, SerializedGraph, SerializedGraphModule


def serialize(nodes_or_graph: Node | Iterable[Node] | GraphModule, descendants=True):
    """
    Interface wrapper taking node(s) or a graph and serializing them to a `SerializationRecord`.
    """
    processor = ContextConverter()

    graph_module_record = state_dict = root_ids = None
    if isinstance(nodes_or_graph, GraphModule):
        module = nodes_or_graph
        input_refs, _ = processor.unstructure(list(module.input_nodes))
        output_refs, _ = processor.unstructure(list(module.nodes_to_compute))
        forward_refs, _ = processor.unstructure(list(module.forward_output_list))
        graph_module_record = SerializedGraphModule(
            input_nodes=tuple(input_refs),
            output_nodes=tuple(output_refs),
            forward_output_list=tuple(forward_refs),
        )
        state_dict = module.state_dict()
    else:
        if isinstance(nodes_or_graph, Node):
            original_roots = [nodes_or_graph]
        else:
            original_roots = list(nodes_or_graph)

        root_refs, _ = processor.unstructure(original_roots)

        # Descendants aren't found by the Node parent recursion.
        expanded = get_connected_nodes(original_roots, descendants=descendants, ancestors=False)
        _debprint(f"Serializing node list: {expanded=}")
        processor.unstructure(expanded)

        ids = [ref.identifier.id for ref in root_refs]
        root_ids = ids[0] if isinstance(nodes_or_graph, Node) else ids

    serial_data = SerializedGraph(node_list=list(processor.node_blob.values()), n_tensor_arguments=len(processor.tensor_blob))
    record = SerializationRecord(graph=serial_data, module=graph_module_record, state_dict=state_dict, root_ids=root_ids)

    return record, processor.tensor_blob


def deserialize(record: SerializationRecord, tensor_data: dict[int, object] | None = None, state_dict: dict[str, object] | None = None):
    """
    Interface wrapper taking a `SerializationRecord` (and its associated tensor data) and
    reconstructing either a bare list of nodes or a `GraphModule`, mirroring `serialize`.
    """
    if record.protocol_version != PROTOCOL_VERSION:
        warnings.warn(
            f"Deserializing a SerializationRecord with protocol_version={record.protocol_version}, "
            f"but the running codebase's protocol_version is {PROTOCOL_VERSION} - the record shapes "
            "or their cattrs hooks may have changed incompatibly since this record was written.",
            stacklevel=2,
        )

    if tensor_data is None:
        tensor_data = record.tensor_data or {}

    _debprint(f"Deserializing node list: {record.graph.node_list=}")
    processor = ContextConverter()
    processor.tensor_blob = tensor_data
    for info in record.graph.node_list:
        processor.node_blob[info.identifier.id] = info

    if record.module is None:
        if record.root_ids is None:
            return [_resolve_node(info.identifier.id, processor) for info in record.graph.node_list]
        if isinstance(record.root_ids, list):
            return [_resolve_node(key, processor) for key in record.root_ids]
        return _resolve_node(record.root_ids, processor)

    def resolve_tuple(attr):
        return tuple(_resolve_node(ref.identifier.id, processor) for ref in attr)

    input_nodes = resolve_tuple(record.module.input_nodes)
    output_nodes = resolve_tuple(record.module.output_nodes)

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

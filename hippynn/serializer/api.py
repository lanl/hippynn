"""
Public serialize/deserialize entry points, plus text-based codec interfaces.
"""

import json
import pathlib
import pickle
import warnings
from functools import partial
from typing import Iterable

import torch
from cattrs import Converter

from ..experiment import _debprint
from ..graphs import GraphModule
from ..graphs.nodes.base import Node, get_connected_nodes
from .hooks import _resolve_node
from .processor import ContextConverter
from .types import (
    PROTOCOL_VERSION,
    NodeReference,
    SerializationRecord,
    Serializable,
    SerializedGraph,
    SerializedGraphModule,
    TaggedDict,
    TaggedList,
    TaggedSet,
    TaggedTuple,
    TensorReference,
)

# Filetypes supported by `savegraph`/`loadgraph`, plus common extension spellings that map onto
# them.
_FILETYPES = ("pkl", "pt", "json", "yml")
_FILETYPE_ALIASES = {"pickle": "pkl", "pth": "pt", "yaml": "yml"}

# todo: instead of registering some new hooks and making a new type of plain converter,
# borrow the already-dressed converter of a ContextConverter type?
def _structure_serializable(data, _typ, *, converter):
    """
    `Serializable`-boundary structure hook for the plain (JSON/YAML) converter: `Serializable`
    is a recursive, open union with no attrs-class discriminator cattrs can auto-disambiguate,
    so this only selects a concrete target type from the wire value's shape, then delegates the
    actual recursive structuring back to `converter`.
    """
    if isinstance(data, dict):
        target = {
            "list": TaggedList[Serializable],
            "node_ref": NodeReference,
            "tensor_ref": TensorReference,
            "set": TaggedSet[Serializable],
            "tuple": TaggedTuple[Serializable],
            "dict": TaggedDict[Serializable],
        }.get(data.get("kind"))
        if target is None:
            raise ValueError(f"Cannot structure a Serializable dict without a recognized 'kind': {data!r}")
        return converter.structure(data, target)
    return data


def _make_plain_converter() -> Converter:
    """A fresh `cattrs.Converter` with the `Serializable`-boundary structure hook registered."""
    converter = Converter()
    converter.register_structure_hook_func(lambda t: t is Serializable, partial(_structure_serializable, converter=converter))
    return converter




def serialize(nodes_or_graph: Node | Iterable[Node] | GraphModule, include_descendants=True):
    """
    Interface wrapper taking node(s) or a graph and serializing them to a `SerializationRecord`.
    """
    processor = ContextConverter()

    graph_module_record = state_dict = root_ids = None
    single_root = False
    if isinstance(nodes_or_graph, GraphModule):
        module = nodes_or_graph
        # Unstructured per-element (rather than `processor.unstructure(list(...))`) since these
        # are plain lists of `Node`s, not `Serializable`-typed data - passing the list itself
        # through `unstructure()` would wrap it as a `TaggedList` via the generic `list` hook.
        input_refs = [processor.unstructure(n)[0] for n in module.input_nodes]
        output_refs = [processor.unstructure(n)[0] for n in module.nodes_to_compute]
        forward_refs = [processor.unstructure(n)[0] for n in module.forward_output_list]
        graph_module_record = SerializedGraphModule(
            input_nodes=tuple(input_refs),
            output_nodes=tuple(output_refs),
            forward_output_list=tuple(forward_refs),
        )
        state_dict = module.state_dict()
    else:
        single_root = isinstance(nodes_or_graph, Node)
        original_roots = [nodes_or_graph] if single_root else list(nodes_or_graph)

        root_refs = [processor.unstructure(n)[0] for n in original_roots]

        # Descendants aren't found by the Node parent recursion.
        expanded = get_connected_nodes(original_roots, descendants=include_descendants, ancestors=False)
        _debprint(f"Serializing node list: {expanded=}")
        processor.unstructure(expanded)

        root_ids = [ref.identifier.id for ref in root_refs]

    serial_data = SerializedGraph(node_list=list(processor.node_blob.values()), n_tensor_arguments=len(processor.tensor_blob))
    record = SerializationRecord(
        graph=serial_data, module=graph_module_record, state_dict=state_dict, root_ids=root_ids, single_root=single_root
    )

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
        resolved = [_resolve_node(key, processor) for key in record.root_ids]
        return resolved[0] if record.single_root else resolved

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



def _infer_filetype(file, filetype):
    """Resolve the filetype to use: `filetype` if given, else inferred from `file`'s extension."""
    if filetype is not None:
        return filetype
    if isinstance(file, str):
        suffix = pathlib.Path(file).suffix.lstrip(".").lower()
        suffix = _FILETYPE_ALIASES.get(suffix, suffix)
        if suffix in _FILETYPES:
            return suffix
    raise ValueError(f"Could not infer a filetype from {file!r}; pass `filetype` explicitly, one of {_FILETYPES}.")

# todo: make context manager, place in hippynn/utils.py, and import here. func call goes inside context. change name
def _open_and(file, mode, func):
    """Call `func` on `file` as an open file object, opening it first if `file` is a filename."""
    if isinstance(file, str):
        with open(file, mode) as fh:
            return func(fh)
    return func(file)

# sketch of new version
def open_if_needed(filename_or_file, mode='r'):
    # doc that mode will not be used if file is given
    if isinstance(filename_or_file,str):
        with open(filename_or_file,mode=mode) as file:
            yield file
    else:
        yield filename_or_file


def savegraph(
    nodes_or_graph: Node | Iterable[Node] | GraphModule,
    file,
    weights_file=None,
    filetype: str | None = None,
    include_descendants=True,
    overwrite: bool = True,
):
    """
    Serialize `nodes_or_graph` (see `serialize`) and write the resulting record to `file`.

    :param nodes_or_graph: forwarded to `serialize`.
    :param file: destination for the main record - a filename (`str`) or an already-open file
        object. Format is inferred from `file`'s extension when it is a string, one of `pkl`
        (`pickle`), `pt` (`torch.save`), `json`, or `yml`; otherwise `filetype` is required.
    :param weights_file: optional secondary `.pt` file to hold the tensor payload
        (`tensor_data`/`state_dict`) separately from the main record via `torch.save`, e.g. to
        keep a `json`/`yml` record plaintext-only.
    :param filetype: explicit format, used when it cannot be inferred from `file`.
    :param include_descendants: forwarded to `serialize`.
    :param overwrite: if False, raise `FileExistsError` if `file` or `weights_file` already
        exist, checked before anything is written.
    """
    filetype = _infer_filetype(file, filetype)
    if filetype not in _FILETYPES:
        raise ValueError(f"Unknown filetype {filetype!r}; expected one of {_FILETYPES}.")

    if not overwrite:
        for target in (file, weights_file):
            if isinstance(target, str) and pathlib.Path(target).exists():
                raise FileExistsError(f"{target!r} already exists and overwrite=False.")

    # todo use file context manager for weights file as well.

    record, tensor_data = serialize(nodes_or_graph, include_descendants=include_descendants)

    if weights_file is not None:
        torch.save({"tensor_data": tensor_data, "state_dict": record.state_dict}, weights_file)
        record.state_dict = None
    elif filetype in ("json", "yml", "pkl") and (tensor_data or record.state_dict is not None):
        raise ValueError(f"filetype={filetype!r} cannot hold tensor data inline - pass `weights_file` to store it separately.")
    else:
        record.tensor_data = tensor_data


    _plain_converter = _make_plain_converter()

    # todo replace with context manager calls described above
    if filetype == "pkl":
        _open_and(file, "wb", lambda fh: pickle.dump(record, fh))
    elif filetype == "pt":
        _open_and(file, "wb", lambda fh: torch.save(record, fh))
    elif filetype == "json":
        _open_and(file, "w", lambda fh: json.dump(_plain_converter.unstructure(record), fh))
    elif filetype == "yml":
        import yaml

        _open_and(file, "w", lambda fh: yaml.safe_dump(_plain_converter.unstructure(record), fh))
    else:
        raise ValueError(f"Unknown filetype {filetype!r}; expected one of {_FILETYPES}.")


def loadgraph(file, weights_file=None, filetype: str | None = None, weights_only: bool = False):
    """
    Read a record written by `savegraph` from `file` and reconstruct it via `deserialize`.

    :param file: source for the main record - a filename (`str`) or an already-open file object,
        in the same formats accepted by `savegraph`.
    :param weights_file: optional secondary `.pt` file holding `tensor_data`/`state_dict`, loaded
        via `torch.load(..., weights_only=True)`.
    :param filetype: explicit format, used when it cannot be inferred from `file`.
    :param weights_only: forwarded to `torch.load` when `filetype` is `pt`; does not affect
        `weights_file`, which is always loaded with `weights_only=True`.
    """
    filetype = _infer_filetype(file, filetype)

    _plain_converter = _make_plain_converter()

    # todo see notes above on context management
    if filetype == "pkl":
        record = _open_and(file, "rb", pickle.load)
    elif filetype == "pt":
        record = _open_and(file, "rb", lambda fh: torch.load(fh, weights_only=weights_only))
    elif filetype == "json":
        record = _plain_converter.structure(_open_and(file, "r", json.load), SerializationRecord)
    elif filetype == "yml":
        import yaml

        record = _plain_converter.structure(_open_and(file, "r", yaml.safe_load), SerializationRecord)
    else:
        raise ValueError(f"Unknown filetype {filetype!r}; expected one of {_FILETYPES}.")

    tensor_data = state_dict = None
    if weights_file is not None:
        weights = torch.load(weights_file, weights_only=True)
        tensor_data = weights["tensor_data"]
        state_dict = weights["state_dict"]

    return deserialize(record, tensor_data=tensor_data, state_dict=state_dict)

"""
Public serialize/deserialize entry points, plus text-based codec interfaces.
"""

import dataclasses
import json
import pathlib
import pickle
import typing
import warnings
from functools import partial
from typing import Iterable

import torch
from cattrs import Converter
from cattrs.strategies import configure_tagged_union

from ..experiment import _debprint
from ..graphs import GraphModule
from ..graphs.nodes.base import Node, get_connected_nodes
from ..tools import open_if_needed
from .processor import CONTEXT_CONVERTER
from .types import (
    PROTOCOL_VERSION,
    NodeReference,
    SerializationRecord,
    Serializable,
    SerializedGraph,
    SerializedGraphModule,
    Tagged,
)

# Filetypes supported by `savegraph`/`loadgraph`, plus common extension spellings that map onto
# them.
_FILETYPES = ("pkl", "pt", "json", "yml")
_FILETYPE_ALIASES = {"pickle": "pkl", "pth": "pt", "yaml": "yml"}


def _structure_serializable(data, _typ, *, converter):
    """
    `Serializable`-boundary structure hook for the plain (JSON/YAML) converter: `Serializable`
    is a recursive, open union that also contains bare scalars, so this only rules those out,
    then delegates dict-shaped data to the `Tagged` sub-union, which `configure_tagged_union`
    below dispatches on its `_type` field.
    """
    if isinstance(data, dict):
        return converter.structure(data, Tagged)
    return data


def _tagged_type(cls: type) -> str:
    """The `_type` field's default on `cls` - the same discriminator string it writes into its own unstructured payload."""
    cls = typing.get_origin(cls) or cls
    return next(f.default for f in dataclasses.fields(cls) if f.name == "_type")


plain_converter = Converter()
plain_converter.register_structure_hook_func(lambda t: t is Serializable, partial(_structure_serializable, converter=plain_converter))
configure_tagged_union(Tagged, plain_converter, tag_generator=_tagged_type, tag_name="_type")


def serialize(nodes_or_graph: Node | Iterable[Node] | GraphModule, include_descendants=True):
    """
    Interface wrapper taking node(s) or a graph and serializing them to a `SerializationRecord`.
    """
    processor = CONTEXT_CONVERTER
    processor.reset()

    graph_module_record = state_dict = root_ids = None
    single_root = False
    if isinstance(nodes_or_graph, GraphModule):
        module = nodes_or_graph
        structure, tensor_blob, node_blob = processor.unstructure_with_context(
            [module.input_nodes, module.nodes_to_compute, module.forward_output_list]
        )
        input_structure, output_structure, forward_structure = structure.values
        graph_module_record = SerializedGraphModule(
            input_nodes=input_structure.values,
            output_nodes=output_structure.values,
            forward_output_list=forward_structure.values,
        )
        state_dict = module.state_dict()
    else:
        single_root = isinstance(nodes_or_graph, Node)
        original_roots = [nodes_or_graph] if single_root else list(nodes_or_graph)

        # Descendants aren't found by the Node parent recursion.
        expanded = get_connected_nodes(original_roots, descendants=include_descendants, ancestors=False)
        _debprint(f"Serializing node list: {expanded=}")

        structure, tensor_blob, node_blob = processor.unstructure_with_context([original_roots, expanded])
        roots_structure, extended_structure = structure.values
        root_ids = roots_structure.values

    serial_data = SerializedGraph(node_list=list(node_blob.values()), n_tensor_values=len(tensor_blob))
    record = SerializationRecord(
        graph=serial_data, module=graph_module_record, state_dict=state_dict, root_ids=root_ids, single_root=single_root
    )

    return record, tensor_blob


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

    if len(tensor_data) != record.graph.n_tensor_values:
        warnings.warn(
            f"SerializationRecord expects {record.graph.n_tensor_values} tensor value(s), but "
            f"tensor_data has {len(tensor_data)} - the tensor payload may be incomplete or come "
            "from a different record.",
            stacklevel=2,
        )

    _debprint(f"Deserializing node list: {record.graph.node_list=}")

    CONTEXT_CONVERTER.structure_with_context(tensor_data, record.graph.node_list)

    # Bare node list, early return
    if record.module is None:
        refs = record.root_ids
        if refs is None:
            refs = [NodeReference(identifier=info.identifier) for info in record.graph.node_list]
        resolved = CONTEXT_CONVERTER.converter.structure(refs, list[NodeReference])
        if record.single_root:
            return resolved[0]
        else:
            return resolved

    # GraphModule, restructure it
    input_nodes, output_nodes, forward_output_list = CONTEXT_CONVERTER.converter.structure(
        [record.module.input_nodes, record.module.output_nodes, record.module.forward_output_list],
        list[tuple[NodeReference, ...]],
    )

    module = GraphModule(input_nodes, output_nodes)

    if tuple(module.forward_output_list) != forward_output_list:
        warnings.warn(
            "Deserialized GraphModule's forward_output_list does not match the recorded one. " "State dict may be invalid.",
            stacklevel=2,
        )
    module.forward_output_list = forward_output_list

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

    record, tensor_data = serialize(nodes_or_graph, include_descendants=include_descendants)

    if weights_file is not None:
        torch.save({"tensor_data": tensor_data, "state_dict": record.state_dict}, weights_file)
        record.state_dict = None
    elif filetype in ("json", "yml", "pkl") and (tensor_data or record.state_dict is not None):
        raise ValueError(f"filetype={filetype!r} cannot hold tensor data inline - pass `weights_file` to store it separately.")
    else:
        record.tensor_data = tensor_data

    write_mode = "wb" if filetype in ("pkl", "pt") else "w"
    with open_if_needed(file, write_mode) as fh:
        if filetype == "pkl":
            pickle.dump(record, fh)
        elif filetype == "pt":
            torch.save(record, fh)
        elif filetype == "json":
            json.dump(plain_converter.unstructure(record), fh)
        elif filetype == "yml":
            import yaml

            yaml.safe_dump(plain_converter.unstructure(record), fh)
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

    read_mode = "rb" if filetype in ("pkl", "pt") else "r"
    with open_if_needed(file, read_mode) as fh:
        if filetype == "pkl":
            record = pickle.load(fh)
        elif filetype == "pt":
            record = torch.load(fh, weights_only=weights_only)
        elif filetype == "json":
            record = plain_converter.structure(json.load(fh), SerializationRecord)
        elif filetype == "yml":
            import yaml

            record = plain_converter.structure(yaml.safe_load(fh), SerializationRecord)
        else:
            raise ValueError(f"Unknown filetype {filetype!r}; expected one of {_FILETYPES}.")

    tensor_data = state_dict = None
    if weights_file is not None:
        weights = torch.load(weights_file, weights_only=True)
        tensor_data = weights["tensor_data"]
        state_dict = weights["state_dict"]

    return deserialize(record, tensor_data=tensor_data, state_dict=state_dict)

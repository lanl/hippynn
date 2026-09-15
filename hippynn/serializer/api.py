"""
File-facing `savegraph`/`loadgraph` entry points.
"""

import dataclasses
import json
import pathlib
import pickle
import warnings
from typing import Iterable

import torch

from ..experiment import _debprint
from ..graphs import GraphModule
from ..graphs.nodes.base import Node
from ..tools import infer_filetype, open_if_needed
from .processor import CONTEXT_CONVERTER
from .types import (
    PROTOCOL_VERSION,
    SerializationBox,
)

# Filetypes supported by `savegraph`/`loadgraph`, plus common extension spellings that map onto
# them.
_FILETYPES = ("pkl", "pt", "json", "yml")
_FILETYPE_ALIASES = {"pickle": "pkl", "pth": "pt", "yaml": "yml"}


def savegraph(
    nodes_or_graph: Iterable[Node] | GraphModule,
    file,
    weights_file=None,
    filetype: str | None = None,
    overwrite: bool = True,
):
    """
    Serialize `nodes_or_graph` to a `SerializationBox` (wrapping a `SerializationRecord` -
    `NodeListRecord`/`GraphModuleRecord`, chosen by the shape of `nodes_or_graph`) and write it to
    `file`.

    :param nodes_or_graph: an iterable of `Node`s, or a `GraphModule`, to serialize.
    :param file: destination for the main record - a filename (`str`) or an already-open file
        object. Format is inferred from `file`'s extension when it is a string, one of `pkl`
        (`pickle`), `pt` (`torch.save`), `json`, or `yml`; otherwise `filetype` is required.
    :param weights_file: optional secondary `.pt` file to hold the tensor payload
        (`tensor_data`, which includes every node's own weights) separately from the main record
        via `torch.save`, e.g. to keep a `json`/`yml` record plaintext-only.
    :param filetype: explicit format, used when it cannot be inferred from `file`.
    :param overwrite: if False, raise `FileExistsError` if `file` or `weights_file` already
        exist, checked before anything is written.
    """
    filetype = infer_filetype(file, filetype, _FILETYPES, _FILETYPE_ALIASES)
    if filetype not in _FILETYPES:
        raise ValueError(f"Unknown filetype {filetype!r}; expected one of {_FILETYPES}.")

    if not overwrite:
        for target in (file, weights_file):
            if isinstance(target, str) and pathlib.Path(target).exists():
                raise FileExistsError(f"{target!r} already exists and overwrite=False.")

    # convert tuples, etc to list if provided
    payload = nodes_or_graph if isinstance(nodes_or_graph, GraphModule) else list(nodes_or_graph)
    record, tensor_data, _node_blob = CONTEXT_CONVERTER.unstructure_with_context(payload)
    n_tensor_values = len(tensor_data)

    if weights_file is not None:
        torch.save(tensor_data, weights_file)
        tensor_data = None
    elif filetype in ("json", "yml") and tensor_data:
        raise ValueError(f"filetype={filetype!r} cannot hold tensor data inline - pass `weights_file` to store it separately.")

    box = SerializationBox(record=record, n_tensor_values=n_tensor_values, tensor_data=tensor_data)

    write_mode = "wb" if filetype in ("pkl", "pt") else "w"
    with open_if_needed(file, write_mode) as fh:
        if filetype == "pkl":
            pickle.dump(box, fh)
        elif filetype == "pt":
            torch.save(box, fh)
        elif filetype == "json":
            json.dump(dataclasses.asdict(box), fh)
        elif filetype == "yml":
            import yaml

            yaml.safe_dump(dataclasses.asdict(box), fh)
        else:
            raise ValueError(f"Unknown filetype {filetype!r}; expected one of {_FILETYPES}.")


def loadgraph(file, weights_file=None, filetype: str | None = None, weights_only: bool = False):
    """
    Read a `SerializationBox` written by `savegraph` from `file` and reconstruct a list of nodes
    or a `GraphModule` from it, dispatched by the boxed record's own `_type` tag.

    :param file: source for the main record - a filename (`str`) or an already-open file object,
        in the same formats accepted by `savegraph`.
    :param weights_file: optional secondary `.pt` file holding `tensor_data`, loaded via
        `torch.load(..., weights_only=True)`.
    :param filetype: explicit format, used when it cannot be inferred from `file`.
    :param weights_only: forwarded to `torch.load` when `filetype` is `pt`; does not affect
        `weights_file`, which is always loaded with `weights_only=True`.
    """
    filetype = infer_filetype(file, filetype, _FILETYPES, _FILETYPE_ALIASES)

    read_mode = "rb" if filetype in ("pkl", "pt") else "r"
    with open_if_needed(file, read_mode) as fh:
        if filetype == "pkl":
            box = pickle.load(fh)
        elif filetype == "pt":
            box = torch.load(fh, weights_only=weights_only)
        elif filetype == "json":
            box = json.load(fh)
        elif filetype == "yml":
            import yaml

            box = yaml.safe_load(fh)
        else:
            raise ValueError(f"Unknown filetype {filetype!r}; expected one of {_FILETYPES}.")

    # `box` is a live SerializationBox for pkl/pt (pickle/torch preserve Python objects); for
    # json/yml it comes back as a plain dict (no type info survives) - box-ify it here so the
    # rest of this function can use uniform attribute access.
    if isinstance(box, dict):
        box = SerializationBox(**box)

    tensor_data = None
    if weights_file is not None:
        tensor_data = torch.load(weights_file, weights_only=True)

    protocol_version = box.protocol_version
    if protocol_version != PROTOCOL_VERSION:
        warnings.warn(
            f"Deserializing a record with protocol_version={protocol_version}, but the running "
            f"codebase's protocol_version is {PROTOCOL_VERSION} - the record shapes or their cattrs "
            "hooks may have changed incompatibly since this record was written.",
            stacklevel=2,
        )

    if tensor_data is None:
        tensor_data = box.tensor_data or {}

    n_tensor_values = box.n_tensor_values
    if len(tensor_data) != n_tensor_values:
        warnings.warn(
            f"Record expects {n_tensor_values} tensor value(s), but tensor_data has "
            f"{len(tensor_data)} - the tensor payload may be incomplete or come from a different record.",
            stacklevel=2,
        )

    record = box.record
    node_list = record["graph"]
    _debprint(f"Deserializing node list: {node_list=}")
    node_blob = {info["identifier"]["id"]: info for info in node_list}

    return CONTEXT_CONVERTER.structure_with_context(record, tensor_blob=tensor_data, node_blob=node_blob)

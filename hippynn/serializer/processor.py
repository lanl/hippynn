"""
The ContextConverter class: a global cattrs Converter plus tensor/node identity caches.

"""

import itertools
from typing import Any

from cattrs import Converter


class ContextConverter:
    """
    Holds one cattrs `Converter` plus the tensor/node identity caches used by its hooks (see
    `hooks.py`/`registry.py`). `context_converter` below is the single instance used for the
    whole process - a Converter's dispatch machinery caches resolved hooks per type, so reusing
    one is the intended usage, and every hook is bound to `ctx=context_converter` directly, at
    registration time, since there is only ever this one instance to resolve. `reset()` clears the
    per-session state back to empty; `unstructure_with_context()`/`structure()`/
    `structure_with_context()` each call it themselves (before reading for `structure()`/
    `structure_with_context()`, after collecting the blobs for `unstructure_with_context()`), so
    callers never need to manage reset timing directly. A session is single-threaded and
    non-reentrant - two independent sessions cannot be interleaved.
    """

    def __init__(self):
        self.converter = Converter()
        self.reset()

    def reset(self):
        """Clear all per-session state, ready for a fresh serialize()/deserialize() call."""

        self._unstructuring = False

        # For holding tensors that come as a side-payload
        self.tensor_blob: dict[int, Any] = {}
        self.tensor_ids: dict[int, int] = {}  # id(tensor) -> key, unstructure-side dedup
        self.tensor_objects: dict[int, Any] = {}  # key -> tensor, structure-side cache
        self._tensor_counter = itertools.count()  # dedicated key source, independent of len(tensor_blob)

        # For holding tensors that come as a side-payload
        self.node_blob: dict[int, Any] = {}
        self.node_ids: dict[int, int] = {}  # id(node) -> key, unstructure-side dedup
        self.node_objects: dict[int, Any] = {}  # key -> reconstructed node, structure-side cache
        self._node_resolving: set[int] = set()  # loop guard for the recursive resolver
        self._node_counter = itertools.count()  # dedicated key source, independent of len(node_blob) -

    def get_node_id(self, obj):
        """Return the discovery-order id already assigned to `obj` by a prior `unstructure()` call, or None."""
        return self.node_ids.get(id(obj))

    # todo plain unstructure should not return a tensor_blob and should be used in the recursion.
    # add new method for unstructuring WITH blob. and make sure regular unstructure is only called from
    # within a ccall to that method (therefore gauranteeing that the blob gets returned somewhere at a higher level)
    def unstructure_with_context(self, obj):
        """
        Self-contained unstructure session: `obj` may recurse through several nested
        `unstructure()` calls (see the guard below) that all accumulate into the same
        `tensor_blob`/`node_blob`; this method returns those blobs directly and resets
        afterward, so callers never read `self.tensor_blob`/`self.node_blob` themselves.
        """
        self._unstructuring = True
        try:
            structure = self.converter.unstructure(obj)
            return_tensors = self.tensor_blob
            return_nodes = self.node_blob
        finally:
            self.reset()
        return structure, return_tensors, return_nodes

    def unstructure(self, obj):
        """
        Return `(structure, tensor_blob)`: `structure` is ordinary cattrs-produced plain data,
        suitable for JSON/YAML encoding; `tensor_blob` holds tensor payloads out-of-band, to be
        saved separately.
        """
        if not self._unstructuring:
            raise ValueError("ContextConverter root calls require being in")
        structure = self.converter.unstructure(obj)
        return structure, self.tensor_blob

    def structure_with_context(self, tensor_data, node_list):
        """
        Read-side session setup: resets state, then installs `tensor_data`/`node_list` as
        `self.tensor_blob`/`self.node_blob` so recursive resolution helpers (e.g.
        `hooks._resolve_node`) can find them there - mirrors `unstructure_with_context`'s session
        handling for the write side. Does not perform resolution itself, since that logic lives in
        `hooks.py` to avoid a circular import back into this module.
        """
        self.reset()
        self.tensor_blob = tensor_data
        self.node_blob = {info.identifier.id: info for info in node_list}

    def structure(self, structure, typ, *, tensor_blob, node_blob=None):
        """
        Reconstruct an object from `structure` plus the separately-loaded `tensor_blob` (and, for
        a node-graph `structure`, its `node_blob`). Always starts by calling `reset()`, so a read
        session never resolves out of a previous write's identity caches - callers only need to
        hand over the blob(s), not manage reset timing themselves.
        """
        self.reset()
        self.tensor_blob = tensor_blob
        if node_blob is not None:
            self.node_blob = node_blob
        return self.converter.structure(structure, typ)


# The single ContextConverter used for the whole process - see the class docstring.
CONTEXT_CONVERTER = ContextConverter()

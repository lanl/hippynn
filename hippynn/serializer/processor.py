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
    per-session state back to empty; `unstructure_with_context()`/`structure_with_context()` each
    call it themselves (before installing blobs), and again after resolving, so callers never need
    to manage reset timing directly. A session is single-threaded and non-reentrant - two
    independent sessions cannot be interleaved.

    `unstructure`/`structure` are the recursive, in-session calls used by hooks (`hooks.py`) while
    a session is active; `unstructure_with_context`/`structure_with_context` are the one-shot,
    self-contained sessions built on top of them.
    """

    def __init__(self):
        self.converter = Converter()
        self.reset()

    def reset(self):
        """Clear all per-session state, ready for a fresh serialize()/deserialize() call."""

        self._unstructuring = False
        self._structuring = False

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
        """Return the discovery-order id already assigned to `obj` by a prior `unstructure_with_context()` call, or None."""
        return self.node_ids.get(id(obj))

    def unstructure_with_context(self, obj):
        """
        Self-contained unstructure session: `obj` may recurse through several nested `unstructure()`
        calls (see the guard below) that all accumulate into the same `tensor_blob`/`node_blob`;
        this method returns those blobs directly and resets afterward, so callers never read
        `self.tensor_blob`/`self.node_blob` themselves.
        """
        self._unstructuring = True
        try:
            structure = self.unstructure(obj)
            return_tensors = self.tensor_blob
            return_nodes = self.node_blob
        finally:
            self.reset()
        return structure, return_tensors, return_nodes

    def unstructure(self, obj):
        """
        Recursive-call entry point used by hooks (`hooks.py`) while a session is active: returns
        just the unstructured `structure` for `obj` - `tensor_blob`/`node_blob` accumulate as
        side effects on `self` rather than being returned per call, and are collected once by
        `unstructure_with_context` at the end of the whole session.
        """
        if not self._unstructuring:
            raise ValueError("ContextConverter.unstructure() may only be called during an active unstructure_with_context() session.")
        return self.converter.unstructure(obj)

    def structure_with_context(self, data, typ, *, tensor_blob, node_blob=None):
        """
        Self-contained structure session: mirrors `unstructure_with_context` for the read side.
        Resets state, installs `tensor_blob`/`node_blob` so recursive resolution helpers (e.g.
        `hooks._resolve_node`) can find them, resolves `data` against `typ` via `structure()`,
        then resets again - callers only need to hand over the blob(s), not manage session state
        themselves.
        """
        self.reset()
        self.tensor_blob = tensor_blob
        if node_blob is not None:
            self.node_blob = node_blob
        self._structuring = True
        try:
            return self.structure(data, typ)
        finally:
            self.reset()

    def structure(self, data, typ):
        """
        Recursive-call entry point used by hooks (`hooks.py`) while a session is active: mirrors
        `unstructure()` for the read side.
        """
        if not self._structuring:
            raise ValueError("ContextConverter.structure() may only be called during an active structure_with_context() session.")
        return self.converter.structure(data, typ)


# The single ContextConverter used for the whole process - see the class docstring.
CONTEXT_CONVERTER = ContextConverter()

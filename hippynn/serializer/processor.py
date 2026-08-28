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
    per-session state back to empty; it must be called at the start of each independent
    serialize()/deserialize() session (see `api.py`), since state is not cleared automatically - a
    session like `serialize()` makes several `unstructure()` calls that must accumulate into the
    same blobs. A session is single-threaded and non-reentrant - two independent sessions cannot
    be interleaved.
    """

    def __init__(self):
        self.converter = Converter()
        self.reset()

    def reset(self):
        """Clear all per-session state, ready for a fresh serialize()/deserialize() call."""
        # torch.Tensor <-> TensorReference: a side-channel blob, never part of the plaintext
        # structure itself.
        self.tensor_blob: dict[int, Any] = {}
        self.tensor_ids: dict[int, int] = {}  # id(tensor) -> key, unstructure-side dedup
        self.tensor_objects: dict[int, Any] = {}  # key -> tensor, structure-side cache
        self._tensor_counter = itertools.count()  # dedicated key source, independent of len(tensor_blob)

        # Node <-> NodeReference: this blob *is* SerializedGraph.node_list, assembled directly
        # from discovery order by the top-level serialize()/deserialize() wrappers.
        self.node_blob: dict[int, Any] = {}
        self.node_ids: dict[int, int] = {}  # id(node) -> key, unstructure-side dedup
        self.node_objects: dict[int, Any] = {}  # key -> reconstructed node, structure-side cache
        self._node_resolving: set[int] = set()  # loop guard for the recursive resolver
        self._node_counter = itertools.count()  # dedicated key source, independent of len(node_blob) -
        # unstructuring a node recurses into its parents before its own blob slot is written, so
        # len(node_blob) is not yet incremented while nested keys are assigned - a plain counter
        # guarantees each call gets a distinct key regardless of that reentrancy.

    def get_node_id(self, obj):
        """Return the discovery-order id already assigned to `obj` by a prior `unstructure()` call, or None."""
        return self.node_ids.get(id(obj))

    # todo plain unstructure should not return a tensor_blob and should be used in the recursion.
    # add new method for unstructuring WITH blob. and make sure regular unstructure is only called from
    # within a ccall to that method (therefore gauranteeing that the blob gets returned somewhere at a higher level)
    def unstructure(self, obj):
        """
        Return `(structure, tensor_blob)`: `structure` is ordinary cattrs-produced plain data,
        suitable for JSON/YAML encoding; `tensor_blob` holds tensor payloads out-of-band, to be
        saved separately.
        """
        structure = self.converter.unstructure(obj)
        return structure, self.tensor_blob

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
context_converter = ContextConverter()

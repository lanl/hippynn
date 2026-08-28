"""
The ContextConverter class: a per-session cattrs Converter plus tensor/node identity caches.
"""

import itertools
from functools import partial
from typing import Any, ClassVar

from cattrs import Converter


class ContextConverter:
    """
    One ContextConverter instance == one serialization/deserialization context.

    Hook registration is class-level (shared across all instances); the cattrs Converter and
    all blob/identity-cache state are instance-level, created fresh in `__init__`. A Processor
    instance is single-session, single-threaded, and non-reentrant: `structure()` mutates
    `self.tensor_blob` as a side effect before delegating to the converter.
    """

    _structure_hooks: ClassVar[list[tuple[type, Any]]] = []
    _unstructure_hooks: ClassVar[list[tuple[type, Any]]] = []
    _structure_families: ClassVar[list[tuple[Any, Any]]] = []
    _unstructure_families: ClassVar[list[tuple[Any, Any]]] = []

    # ------------------------------------------------------------------
    # Class-level registration decorators
    # ------------------------------------------------------------------

    @classmethod
    def register_structure(cls, typ):
        """Register a context-aware cattrs structure hook for one type: `func(data, typ, *, ctx)`."""

        def decorator(func):
            cls._structure_hooks.append((typ, func))
            return func

        return decorator

    @classmethod
    def register_unstructure(cls, typ):
        """Register a context-aware cattrs unstructure hook for one type: `func(obj, *, ctx)`."""

        def decorator(func):
            cls._unstructure_hooks.append((typ, func))
            return func

        return decorator

    @classmethod
    def register_structure_family(cls, predicate):
        """Register a context-aware structure hook for a family of types matched by `predicate(typ)`."""

        def decorator(func):
            cls._structure_families.append((predicate, func))
            return func

        return decorator

    @classmethod
    def register_unstructure_family(cls, predicate):
        """Register a context-aware unstructure hook for a family of types matched by `predicate(typ)`."""

        def decorator(func):
            cls._unstructure_families.append((predicate, func))
            return func

        return decorator

    # ------------------------------------------------------------------
    # Per-session context
    # ------------------------------------------------------------------

    def __init__(self):
        self.converter = Converter()

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

        self._synced_structure = 0
        self._synced_unstructure = 0
        self._synced_structure_families = 0
        self._synced_unstructure_families = 0
        self._sync_hooks()

    def _sync_hooks(self):
        """
        Register onto `self.converter` any class-level hooks added since this instance last
        synced, so hooks registered after construction (e.g. by a caller that registers a
        type's hooks just before using it) still take effect.
        """
        for typ, func in self._structure_hooks[self._synced_structure :]:
            self.converter.register_structure_hook(typ, partial(func, ctx=self))
        self._synced_structure = len(self._structure_hooks)

        for typ, func in self._unstructure_hooks[self._synced_unstructure :]:
            self.converter.register_unstructure_hook(typ, partial(func, ctx=self))
        self._synced_unstructure = len(self._unstructure_hooks)

        for predicate, func in self._structure_families[self._synced_structure_families :]:
            self.converter.register_structure_hook_func(predicate, partial(func, ctx=self))
        self._synced_structure_families = len(self._structure_families)

        for predicate, func in self._unstructure_families[self._synced_unstructure_families :]:
            self.converter.register_unstructure_hook_func(predicate, partial(func, ctx=self))
        self._synced_unstructure_families = len(self._unstructure_families)

    def get_node_id(self, obj):
        """Return the discovery-order id already assigned to `obj` by a prior `unstructure()` call, or None."""
        return self.node_ids.get(id(obj))

    def unstructure(self, obj):
        """
        Return `(structure, tensor_blob)`: `structure` is ordinary cattrs-produced plain data,
        suitable for JSON/YAML encoding; `tensor_blob` holds tensor payloads out-of-band, to be
        saved separately.
        """
        self._sync_hooks()
        structure = self.converter.unstructure(obj)
        return structure, self.tensor_blob

    def structure(self, structure, typ, *, tensor_blob):
        """Reconstruct an object from `structure` plus the separately-loaded `tensor_blob`."""
        self._sync_hooks()
        self.tensor_blob = tensor_blob
        return self.converter.structure(structure, typ)

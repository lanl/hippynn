"""
Serialization of hippynn Node graphs to/from plain data.

`types.py` defines the plain-data record shapes, `registry.py` the node-class registry, and
`processor.py` the cattrs-driven `ContextConverter` (hooks and hook registration included), used
directly by `api.py`'s file-facing `savegraph`/`loadgraph`. This module re-exports the public
surface.
"""

from .types import PROTOCOL_VERSION
from .registry import ensure_registry_consistency
from .api import savegraph, loadgraph

ensure_registry_consistency(raise_=False)
